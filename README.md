# llm-rl-generalization

GRPO + LoRA fine-tuning of Qwen3-1.7B on GSM8K with TRL + PEFT, configured with Hydra, managed with uv, logged to wandb.
Runs on a single L40S on Vulcan; vLLM generates rollouts on the same GPU (`vllm_mode=colocate`).

## Layout

```
configs/train.yaml, eval.yaml   # entry configs; groups: paths, model, data, lora, trainer, experiment
src/llm_rl/data.py              # GSM8K prompts + gold answers
src/llm_rl/rewards.py           # \boxed{} answer extraction and the reward
src/llm_rl/train.py             # GRPOTrainer + LoRA
src/llm_rl/eval.py              # greedy GSM8K eval with vLLM (base or base + adapter)
scripts/*.sbatch                # Slurm jobs; submit from the repo root
```

The venv lives in `$SCRATCH/envs/llm_rl_generalization`, runs in `$SCRATCH/llm_rl_generalization/runs/<run_name>`,
Slurm logs in `$SCRATCH/llm_rl_generalization/logs`. `$SCRATCH` is purged after 60 days: copy adapters and eval
results you want to keep to `/project`.

## Usage

```bash
mkdir -p $SCRATCH/llm_rl_generalization/logs     # once, before the first job
uv run --no-sync wandb login                      # once; or export WANDB_API_KEY

sbatch scripts/setup.sbatch                                         # uv sync, tests, download model + data
sbatch --time=00:30:00 scripts/train.sbatch experiment=smoke        # 4-step end-to-end check
sbatch scripts/eval.sbatch                                          # base-model GSM8K test accuracy
sbatch scripts/train.sbatch                                         # main run (r=16)
sbatch scripts/train.sbatch lora.r=32 lora.lora_alpha=64            # r=32
sbatch scripts/eval.sbatch adapter_path=$SCRATCH/llm_rl_generalization/runs/<run_name>/final_adapter
```

Resubmitting `train.sbatch run_name=<run_name>` resumes from that run's latest checkpoint (and the same wandb run).
