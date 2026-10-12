"""GRPO + LoRA training on GSM8K.

Usage: python -m llm_rl.train [hydra overrides], e.g. `experiment=smoke` or `lora.r=32 lora.lora_alpha=64`.
"""

import os

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from llm_rl.data import load_train_val
from llm_rl.rewards import gsm8k_reward


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    run_id = cfg.run_name if hydra_cfg.mode == RunMode.RUN else f"{cfg.run_name}-{hydra_cfg.job.num}"
    set_seed(cfg.seed)

    if cfg.wandb.enabled:
        # Started here so the run carries the full Hydra config; TRL's callback reuses the active run.
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_id,
            id=run_id,
            resume="allow",
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_ds, val_ds = load_train_val(cfg.data, cfg.seed)
    args = GRPOConfig(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        output_dir=run_dir,
        run_name=run_id,
        seed=cfg.seed,
        report_to="wandb" if cfg.wandb.enabled else "none",
    )
    trainer = GRPOTrainer(
        model=cfg.model.name_or_path,
        reward_funcs=[gsm8k_reward],
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True)),
    )
    trainer.model.print_trainable_parameters()

    checkpoint = get_last_checkpoint(run_dir) if cfg.resume else None
    if checkpoint:
        print(f"Resuming from {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(os.path.join(run_dir, "final_adapter"))

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
