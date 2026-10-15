"""WebShop evaluation with vLLM: score and success rate on the test goals, greedy by default.

Usage: python -m llm_rl.eval_webshop [model_path=<merged model dir>] [adapter_path=<adapter dir>] [limit=N]
"""

import json
import time
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llm_rl.alfworld_rollout import EpisodeRunner, load_tokenizer, summarize
from llm_rl.env_pool import EnvPool
from llm_rl.webshop import WebshopEnv, build_prompt, goal_rows, webshop_metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_webshop")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    env_kwargs = {
        "webshop_dir": cfg.env.webshop_dir,
        "max_steps": cfg.env.max_steps,
        "seed": cfg.env.seed,
        "search_any_page": cfg.env.search_any_page,
    }
    # Before vLLM starts: the pool forks its workers, which load WebShop while vLLM starts up.
    pool = EnvPool(WebshopEnv, cfg.env.num_workers, **env_kwargs)

    llm_kwargs, lora_request = {}, None
    tag = cfg.tag or Path(cfg.model_path).name
    if cfg.adapter_path:
        adapter = Path(cfg.adapter_path)
        rank = json.loads((adapter / "adapter_config.json").read_text())["r"]
        llm_kwargs = {"enable_lora": True, "max_lora_rank": rank}
        lora_request = LoRARequest("adapter", 1, str(adapter))
        tag = cfg.tag or f"{adapter.parent.name}/{adapter.name}"

    llm = LLM(
        model=cfg.model_path,
        dtype=cfg.model.dtype,
        max_model_len=cfg.env.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        seed=cfg.seed,
        language_model_only=True,  # text-only episodes; skips the vision encoder
        **llm_kwargs,
    )
    tokenizer, chat_template = load_tokenizer(cfg.model_path)
    runner = EpisodeRunner(
        tokenizer,
        chat_template,
        pool,
        max_steps=cfg.env.max_steps,
        max_completion_length=cfg.env.max_completion_length,
        enable_thinking=cfg.model.enable_thinking,
        tool=WebshopEnv(**env_kwargs).act,
        prompt_fn=build_prompt,
    )
    sampling = SamplingParams(temperature=cfg.temperature, seed=cfg.seed)

    start, stop = cfg.goals
    rows = goal_rows(start, stop)[: cfg.limit or None]
    start_time = time.time()
    records = [runner.record(ep) for ep in runner.run(llm, rows, sampling, lora_request=lora_request)]
    metrics = webshop_metrics(records) | summarize(records) | runner.timing | {"seconds": time.time() - start_time}
    print(f"[{tag}] test: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
    with open(out_dir / "episodes_test.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    pool.close()
    # Without an explicit shutdown the interpreter can hang at exit waiting for the engine process.
    llm.llm_engine.engine_core.shutdown()

    config = OmegaConf.to_container(cfg, resolve=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"tag": tag, "metrics": {"test": metrics}, "config": config}, f, indent=2)
    print(f"Wrote results to {out_dir}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"eval_webshop/{tag}",
            job_type="eval",
            tags=["webshop"],
            config=config,
        )
        wandb.summary.update({f"webshop/test/{k}": v for k, v in metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
