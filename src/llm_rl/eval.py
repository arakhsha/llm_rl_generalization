"""Greedy GSM8K evaluation with vLLM, for the base model or base model + LoRA adapter.

Usage: python -m llm_rl.eval [adapter_path=<run_dir>/final_adapter] [split=test] [limit=N]
"""

import json
import os
from pathlib import Path
from statistics import mean

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llm_rl.data import load_gsm8k
from llm_rl.rewards import extract_boxed, extract_last_number, is_correct, strict_answer


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    ds = load_gsm8k(cfg.data, cfg.split)
    if cfg.limit:
        ds = ds.select(range(min(cfg.limit, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    prompts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=cfg.model.enable_thinking
        )
        for messages in ds["prompt"]
    ]

    llm_kwargs, lora_request, tag = {}, None, "base"
    if cfg.adapter_path:
        adapter = Path(cfg.adapter_path)
        rank = json.loads((adapter / "adapter_config.json").read_text())["r"]
        llm_kwargs = {"enable_lora": True, "max_lora_rank": rank}
        lora_request = LoRARequest("adapter", 1, str(adapter))
        tag = f"{adapter.parent.name}/{adapter.name}"

    llm = LLM(
        model=cfg.model.name_or_path,
        dtype=cfg.model.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        seed=cfg.seed,
        **llm_kwargs,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens)
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    samples = []
    for messages, gold, out in zip(ds["prompt"], ds["gold"], outputs, strict=True):
        completion = out.outputs[0]
        samples.append(
            {
                "question": messages[-1]["content"],
                "gold": gold,
                "completion": completion.text,
                "strict_correct": is_correct(strict_answer(completion.text), gold),
                "lenient_correct": is_correct(extract_last_number(completion.text), gold),
                "boxed": extract_boxed(completion.text) is not None,
                "truncated": completion.finish_reason == "length",
                "num_tokens": len(completion.token_ids),
            }
        )

    metrics = {
        "strict_acc": mean(s["strict_correct"] for s in samples),
        "lenient_acc": mean(s["lenient_correct"] for s in samples),
        "boxed_rate": mean(s["boxed"] for s in samples),
        "truncated_rate": mean(s["truncated"] for s in samples),
        "mean_completion_tokens": mean(s["num_tokens"] for s in samples),
        "num_examples": len(samples),
    }
    print(f"[{tag}] {cfg.split}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    config = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"tag": tag, "split": cfg.split, "metrics": metrics, "config": config}, f, indent=2)
    with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in samples)
    print(f"Wrote results to {out_dir}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"eval/{tag}/{cfg.split}",
            job_type="eval",
            config=config,
        )
        wandb.summary.update({f"eval/{cfg.split}/{k}": v for k, v in metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
