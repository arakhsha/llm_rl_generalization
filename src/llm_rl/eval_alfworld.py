"""ALFWorld evaluation with vLLM: success rate per split, greedy by default.

Usage: python -m llm_rl.eval_alfworld [model_path=<merged model dir>] [adapter_path=<adapter dir>]
       [splits=[valid_unseen]] [limit=N] [env.admissible_commands=all]
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

from llm_rl.alfworld import AlfworldEnv, extract_games, load_manifest
from llm_rl.alfworld_rollout import EpisodeRunner, load_tokenizer, summarize
from llm_rl.env_pool import EnvPool


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_alfworld")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    games_dir = extract_games(cfg.env.data_dir, cfg.env.games_dir)
    # Before vLLM starts: the pool forks its workers.
    pool = EnvPool(
        AlfworldEnv,
        cfg.env.num_workers,
        games_dir=games_dir,
        max_steps=cfg.env.max_steps,
        admissible_commands=cfg.env.admissible_commands,
    )

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
    )
    sampling = SamplingParams(temperature=cfg.temperature, seed=cfg.seed)

    results = {}
    for split in cfg.splits:
        rows = load_manifest(cfg.env.data_dir, split)[: cfg.limit or None]
        start = time.time()
        records = [runner.record(ep) for ep in runner.run(llm, rows, sampling, lora_request=lora_request)]
        results[split] = summarize(records) | {"seconds": time.time() - start}
        print(f"[{tag}] {split}: " + ", ".join(f"{k}={v:.4g}" for k, v in results[split].items()))
        with open(out_dir / f"episodes_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    pool.close()
    # Without an explicit shutdown the interpreter can hang at exit waiting for the engine process.
    llm.llm_engine.engine_core.shutdown()

    config = OmegaConf.to_container(cfg, resolve=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"tag": tag, "metrics": results, "config": config}, f, indent=2)
    print(f"Wrote results to {out_dir}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"eval_alfworld/{tag}",
            job_type="eval",
            config=config,
        )
        wandb.summary.update({f"eval/{split}/{k}": v for split, m in results.items() for k, v in m.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
