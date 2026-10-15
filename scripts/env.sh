# Shared environment for the Slurm jobs; sourced from the repo root.
# Cache locations follow ~/.bashrc when set there, with $SCRATCH fallbacks.
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/envs/llm_rl_generalization"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/caches/uv}"
export HF_HOME="${HF_HOME:-$SCRATCH/caches/huggingface}"
export WANDB_DIR="$SCRATCH/llm_rl_generalization"  # wandb writes to $WANDB_DIR/wandb
export TRITON_CACHE_DIR="$SCRATCH/caches/triton"   # default is ~/.triton
# Node-local temp files: fast-downward (ALFWorld) dlopens a copy of its library from $TMPDIR, and
# deleting an open file leaves .nfs* placeholders that make the cleanup fail on network filesystems.
export TMPDIR="${SLURM_TMPDIR:-$TMPDIR}"
# alfworld creates this directory on import (default ~/.cache/alfworld); it holds the packed games.
export ALFWORLD_DATA="$SCRATCH/llm_rl_generalization/data/alfworld"
export TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which the nodes don't have; use vLLM's PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$WANDB_DIR"
