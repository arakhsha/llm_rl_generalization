#!/bin/bash
#SBATCH --job-name=setup_alfworld
#SBATCH --account=aip-sologen
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Installs the ALFWorld dependencies, downloads Qwen3.5-2B and packs the ALFWorld text games.
# Usage (from the repo root): sbatch scripts/setup_alfworld.sbatch
#   The CPU queue can be long; adding --gres=gpu:l40s:1 usually starts it right away.
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
source scripts/env.sh

# fast-downward-textworld and jericho are source-only and compile here (gcc/cmake from CVMFS).
uv lock
uv sync
uv run --no-sync python -c "import importlib.metadata as md; print({p: md.version(p) for p in ['torch', 'transformers', 'trl', 'vllm', 'alfworld', 'textworld', 'flash-linear-attention']})"
uv run --no-sync hf download Qwen/Qwen3.5-2B

# Only the trajectory metadata (task types) and the TextWorld game files are needed for the text env.
raw="$ALFWORLD_DATA/raw"
mkdir -p "$raw"
for url in https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip \
           https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip; do
    [[ -f "$raw/$(basename "$url")" ]] || curl -fsSL -o "$raw/$(basename "$url")" "$url"
done
uv run --no-sync python -m llm_rl.alfworld "$raw" "$ALFWORLD_DATA"
# Includes playing a few games with the PDDL planner's solution through AlfworldEnv.
uv run --no-sync pytest -q
