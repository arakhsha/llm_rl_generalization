"""Batched multi-turn tool-calling episodes with vLLM: the GRPO rollout function and the evaluation loop.

An episode is one growing token sequence (prompt + model turns + tool results) that is never
re-tokenized, built the way TRL's own tool-calling loop builds it: tool results are rendered with the
chat template and appended as tokens that are masked out of the loss (env_mask 0), and model turns and
tool results share one max_completion_length budget. Each turn, all running episodes are generated in
one vLLM call and their environments step in parallel in an EnvPool.

ALFWorld is the default environment; WebShop evaluation (llm_rl.webshop) passes its own tool and prompt.
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from llm_rl.alfworld import TASK_TYPES, AlfworldEnv, build_prompt
from llm_rl.env_pool import EnvPool


def load_tokenizer(name_or_path: str) -> tuple[PreTrainedTokenizerBase, str | None]:
    """Tokenizer with tool-call parsing, plus the chat template GRPOTrainer would use (None: the model's own)."""
    tokenizer = add_response_schema(AutoTokenizer.from_pretrained(name_or_path))
    chat_template = None if is_chat_template_prefix_preserving(tokenizer) else get_training_chat_template(tokenizer)
    return tokenizer, chat_template


@dataclass
class Episode:
    row: dict  # passed to the environment's reset(): an ALFWorld manifest row or a WebShop goal
    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)  # 1 for model tokens, 0 for tool-result tokens
    stats: dict = field(default_factory=dict)  # the environment's stats(): won, done, steps, invalid, ...
    turns: int = 0
    # Why the model stopped: "stopped" (no tool call), "truncated" (token budget hit mid-turn) or
    # "out_of_tokens" (the next tool result did not fit).
    end: str = "running"

    @property
    def status(self) -> str:
        if self.stats["won"]:
            return "won"
        if self.stats["done"]:
            # A WebShop episode also ends when the model buys a product that does not fully match.
            return "bought" if self.stats.get("bought") else "out_of_steps"
        return self.end


class EpisodeRunner:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None,
        pool: EnvPool,
        max_steps: int,
        max_completion_length: int,
        enable_thinking: bool = False,
        tool: Callable | None = None,
        prompt_fn: Callable[[], list[dict]] = build_prompt,
    ):
        """`tool` is the environment's act method, whose name, signature and docstring define the tool
        schema in the prompt (default: AlfworldEnv.act); `prompt_fn` returns the conversation prefix that
        the first observation is appended to.
        """
        self.tokenizer = tokenizer
        self.pool = pool
        self.max_steps = max_steps
        self.max_completion_length = max_completion_length
        self.template_kwargs = {"chat_template": chat_template, "enable_thinking": enable_thinking}
        self.tool = tool or AlfworldEnv("", max_steps).act
        self.tool_name = self.tool.__name__
        self.prompt_fn = prompt_fn

    def _render(self, messages: list[dict], **kwargs) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, **self.template_kwargs, **kwargs
        )

    def _tool_suffix_ids(self, tool_messages: list[dict]) -> list[int]:
        """Tokens from the end of an assistant turn to the next assistant header (as _get_tool_suffix_ids in TRL)."""
        tool_call = {"type": "function", "function": {"name": self.tool_name, "arguments": {}}}
        dummy = [
            {"role": "user", "content": "dummy"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        ]
        prefix = self._render(dummy, add_generation_prompt=False)
        full = self._render(dummy + tool_messages, add_generation_prompt=True)
        eos = [i for i, t in enumerate(prefix) if t == self.tokenizer.eos_token_id]
        if eos:
            prefix = prefix[: eos[-1] + 1]
        assert full[: len(prefix)] == prefix, "tool suffix is not a continuation of the assistant turn"
        return full[len(prefix) :]

    def run(self, llm: LLM, rows: list[dict], sampling: SamplingParams, lora_request=None) -> list[Episode]:
        """Play one episode per manifest row; env slot i plays rows[i]."""
        episodes = []
        start = time.perf_counter()
        resets = self.pool.run([(i, "reset", row) for i, row in enumerate(rows)])
        env_seconds = time.perf_counter() - start
        for row, (obs, stats) in zip(rows, resets, strict=True):
            if isinstance(obs, Exception):
                raise obs
            messages = self.prompt_fn()
            messages[-1]["content"] += obs
            prompt_ids = self._render(messages, tools=[self.tool], add_generation_prompt=True)
            episodes.append(Episode(row, prompt_ids, stats=stats))

        active = list(range(len(episodes)))
        generate_seconds, turns = 0.0, 0
        while active:
            turns += 1
            params = []
            for i in active:
                p = sampling.clone()
                p.max_tokens = self.max_completion_length - len(episodes[i].completion_ids)
                params.append(p)
            prompts = [TokensPrompt(prompt_token_ids=episodes[i].prompt_ids + episodes[i].completion_ids)
                       for i in active]
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)
            generate_seconds += time.perf_counter() - start

            calls = {}  # episode index -> tool calls of its latest turn
            for i, out in zip(active, outputs, strict=True):
                ep, gen = episodes[i], out.outputs[0]
                token_ids = list(gen.token_ids)
                parsed = parse_response(self.tokenizer, token_ids, prefix=ep.prompt_ids + ep.completion_ids)
                ep.completion_ids += token_ids
                ep.env_mask += [1] * len(token_ids)
                if gen.logprobs is not None:
                    ep.logprobs += [lp[t].logprob for lp, t in zip(gen.logprobs, token_ids, strict=True)]
                ep.turns += 1
                if gen.finish_reason == "length":
                    ep.end = "truncated"
                # GRPOTrainer stops after max_tool_calling_iterations (= max_steps) rounds of tool calls.
                elif not parsed.get("tool_calls") or ep.turns > self.max_steps:
                    ep.end = "stopped"
                else:
                    calls[i] = parsed["tool_calls"]

            # Unknown tools and bad arguments come back as {"error": ...}, as in GRPOTrainer's tool loop.
            requests = [
                (i, "act", c["function"]["arguments"])
                for i, cs in calls.items()
                for c in cs
                if c["function"]["name"] == self.tool_name
            ]
            start = time.perf_counter()
            results = iter(self.pool.run(requests))
            env_seconds += time.perf_counter() - start
            active = []
            for i, cs in calls.items():
                ep, tool_messages = episodes[i], []
                for c in cs:
                    if c["function"]["name"] == self.tool_name:
                        content, ep.stats = next(results)
                    else:
                        content = ValueError(f"Tool {c['function']['name']} not found.")
                    if isinstance(content, Exception):
                        content = {"error": str(content)}
                    tool_messages.append({"role": "tool", "name": c["function"]["name"], "content": str(content)})
                suffix = self._tool_suffix_ids(tool_messages)
                # >= leaves at least one token for the next turn.
                if len(ep.completion_ids) + len(suffix) >= self.max_completion_length:
                    ep.end = "out_of_tokens"
                    continue
                ep.completion_ids += suffix
                ep.env_mask += [0] * len(suffix)
                ep.logprobs += [0.0] * len(suffix)
                active.append(i)
        self.timing = {"generate_seconds": generate_seconds, "env_seconds": env_seconds, "turns": turns}
        return episodes

    def record(self, ep: Episode) -> dict:
        """JSON-serializable summary of an episode, with the decoded transcript."""
        return {
            **ep.row,
            "status": ep.status,
            **ep.stats,
            "turns": ep.turns,
            "completion_tokens": len(ep.completion_ids),
            "model_tokens": sum(ep.env_mask),
            "transcript": self.tokenizer.decode(ep.completion_ids),
        }


def summarize(records: list[dict]) -> dict[str, float]:
    metrics = {
        "success": mean(r["won"] for r in records),
        "steps": mean(r["steps"] for r in records),
        "invalid_rate": sum(r["invalid"] for r in records) / max(1, sum(r["steps"] for r in records)),
        "completion_tokens": mean(r["completion_tokens"] for r in records),
        "num_episodes": len(records),
    }
    for status, count in Counter(r["status"] for r in records).items():
        metrics[f"status/{status}"] = count / len(records)
    for task in TASK_TYPES:  # ALFWorld only
        won = [r["won"] for r in records if r.get("task_type") == task]
        if won:
            metrics[f"success/{task}"] = mean(won)
    return metrics


def make_rollout_func(runner: EpisodeRunner):
    """GRPOTrainer rollout_func: plays the batch's games with the trainer's colocated vLLM engine.

    Expects the trainer to expose the batch's dataset rows as `batch_inputs` (see train_alfworld.py), since
    rollout_func itself only receives the prompts.
    """

    def rollout_func(prompts: list, trainer) -> dict[str, list]:
        rows = trainer.batch_inputs
        assert len(rows) == len(prompts)
        sampling = SamplingParams(
            n=1,
            temperature=trainer.temperature,
            top_p=trainer.top_p,
            top_k=trainer.top_k,
            min_p=trainer.min_p or 0.0,
            repetition_penalty=trainer.repetition_penalty,
            logprobs=0,  # the sampled token's logprob, for TRL's importance-sampling correction
        )
        episodes = runner.run(trainer.vllm_generation.llm, rows, sampling)
        for key, value in runner.timing.items():
            trainer._metrics["train"][f"rollout/{key}"].append(value)
        return {
            "prompt_ids": [ep.prompt_ids for ep in episodes],
            "completion_ids": [ep.completion_ids for ep in episodes],
            "logprobs": [ep.logprobs for ep in episodes],
            "env_mask": [ep.env_mask for ep in episodes],
            # Passed on to the reward function.
            **{key: [ep.stats[key] for ep in episodes] for key in ("won", "done", "steps", "invalid")},
        }

    return rollout_func
