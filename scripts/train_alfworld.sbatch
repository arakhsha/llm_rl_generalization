#!/bin/bash
#SBATCH --job-name=grpo_alfworld
#SBATCH --account=aip-sologen
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# One training process per GPU, each with its own colocated vLLM engine and 14 environment workers,
# so request 16 CPUs and 64G per GPU. Episodes per step stay fixed (episodes_per_step).
# Usage (from the repo root): sbatch scripts/train_alfworld.sbatch [hydra overrides]
#   smoke test: sbatch --time=01:00:00 scripts/train_alfworld.sbatch experiment=alfworld_smoke
#   1 GPU:      sbatch --gres=gpu:l40s:1 --cpus-per-task=16 --mem=64G scripts/train_alfworld.sbatch
#   resume:     sbatch scripts/train_alfworld.sbatch run_name=<existing run name>
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh
export HF_HUB_OFFLINE=1
# Same run name in every process (see configs/train_alfworld.yaml).
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
num_gpus=$(nvidia-smi -L | wc -l)
uv run --no-sync accelerate launch --num_processes "$num_gpus" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --main_process_port $((20000 + SLURM_JOB_ID % 20000)) \
    -m llm_rl.train_alfworld "$@"
