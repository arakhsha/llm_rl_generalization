#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=aip-sologen
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Usage (from the repo root): sbatch scripts/eval.sbatch [adapter_path=<run_dir>/final_adapter] [split=test]
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh
export HF_HUB_OFFLINE=1

uv run --no-sync python -m llm_rl.eval "$@"
