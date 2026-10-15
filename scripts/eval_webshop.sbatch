#!/bin/bash
#SBATCH --job-name=eval_webshop
#SBATCH --account=aip-sologen
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Usage (from the repo root): sbatch scripts/eval_webshop.sbatch [model_path=<merged model dir>] [adapter_path=...]
# Each environment worker holds WebShop's full product set in memory, hence the memory request.
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh
module load java/21.0.1  # pyserini runs Lucene through pyjnius
export HF_HUB_OFFLINE=1

uv run --no-sync python -m llm_rl.eval_webshop "$@"
