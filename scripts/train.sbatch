#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --account=aip-sologen
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Usage (from the repo root): sbatch scripts/train.sbatch [hydra overrides]
#   smoke test: sbatch --time=00:30:00 scripts/train.sbatch experiment=smoke
#   resume:     sbatch scripts/train.sbatch run_name=<existing run name>
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh
export HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-sync python -m llm_rl.train "$@"
