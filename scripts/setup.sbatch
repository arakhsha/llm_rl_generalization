#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --account=aip-sologen
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Builds the uv environment, runs the unit tests and pre-downloads the model and dataset.
# Usage (from the repo root): sbatch scripts/setup.sbatch
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh

uv sync
uv run --no-sync python -c "import importlib.metadata as md; print({p: md.version(p) for p in ['torch', 'transformers', 'trl', 'peft', 'vllm', 'accelerate', 'datasets', 'hydra-core', 'wandb']})"
uv run --no-sync pytest -q
uv run --no-sync hf download Qwen/Qwen3-1.7B
uv run --no-sync python -c "from datasets import load_dataset; print(load_dataset('openai/gsm8k', 'main'))"
