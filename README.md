# llm-rl-generalization

GRPO + LoRA fine-tuning with TRL + PEFT, configured with Hydra, managed with uv, logged to wandb, on Vulcan L40S GPUs:

- **GSM8K**: Qwen3-1.7B, single-turn math answers; vLLM generates rollouts on the training GPU (`vllm_mode=colocate`).
- **ALFWorld**: Qwen3.5-2B as a multi-turn agent in the text-only household environment. The model calls an `act`
  tool with one command per turn; the reward is 1 if the game is won. A custom `rollout_func` plays all episodes of
  a step together on the colocated vLLM engine, with the environments in parallel worker processes. One process per GPU.
- **WebShop** (evaluation only): the same `act` tool with WebShop's `search[...]` / `click[...]` actions, on the full
  product set (1.18M products) and the 500 human-written test instructions. Reports WebShop's score (mean reward
  x 100) and success rate (reward 1), e.g. to test whether ALFWorld training transfers.

## Layout

```
configs/train.yaml, eval.yaml       # GSM8K entry configs; groups: paths, model, data, lora, trainer, experiment
configs/train_alfworld.yaml, eval_alfworld.yaml, eval_webshop.yaml   # ALFWorld / WebShop entry configs; add the env group
src/llm_rl/data.py, rewards.py      # GSM8K prompts, answer extraction and reward
src/llm_rl/train.py, eval.py        # GSM8K GRPO training and greedy vLLM eval
src/llm_rl/alfworld.py              # ALFWorld env, reward, game packing
src/llm_rl/alfworld_rollout.py      # batched multi-turn episodes (GRPO rollout_func and eval loop)
src/llm_rl/env_pool.py              # environments in parallel worker processes
src/llm_rl/webshop.py, eval_webshop.py   # WebShop env wrapper (WebShop's code from $WEBSHOP_DIR) and greedy eval
src/llm_rl/train_alfworld.py        # GRPO + LoRA on ALFWorld, in-training validation, final merge
src/llm_rl/eval_alfworld.py         # greedy ALFWorld eval (base, base + adapter, or merged model)
src/llm_rl/merge.py                 # merge a LoRA adapter into a standalone model
scripts/*.sbatch                    # Slurm jobs; submit from the repo root
```

The venv lives in `$SCRATCH/envs/llm_rl_generalization`, runs in `$SCRATCH/llm_rl_generalization/runs/<run_name>`,
Slurm logs in `$SCRATCH/llm_rl_generalization/logs`. `$SCRATCH` is purged after 60 days: ALFWorld training copies
the adapter, a merged model and the validation history to `/project/aip-sologen/aminr/llm_rl_generalization/models/<run_name>`.

## Usage

```bash
mkdir -p $SCRATCH/llm_rl_generalization/logs     # once, before the first job
uv run --no-sync wandb login                      # once; or export WANDB_API_KEY

# GSM8K
sbatch scripts/setup.sbatch                                         # uv sync, tests, download model + data
sbatch --time=00:30:00 scripts/train.sbatch experiment=smoke        # 4-step end-to-end check
sbatch scripts/eval.sbatch                                          # base-model GSM8K test accuracy
sbatch scripts/train.sbatch                                         # main run (r=16)
sbatch scripts/eval.sbatch adapter_path=$SCRATCH/llm_rl_generalization/runs/<run_name>/final_adapter

# ALFWorld
sbatch --gres=gpu:l40s:1 scripts/setup_alfworld.sbatch              # deps, Qwen3.5-2B, pack games, tests
sbatch --time=01:00:00 scripts/train_alfworld.sbatch experiment=alfworld_smoke   # 3-step end-to-end check
sbatch scripts/eval_alfworld.sbatch tag=base                        # base model on valid_seen / valid_unseen
sbatch scripts/train_alfworld.sbatch                                # main run (2 GPUs)
sbatch scripts/eval_alfworld.sbatch model_path=/project/aip-sologen/aminr/llm_rl_generalization/models/<run_name>/merged

# WebShop
sbatch --gres=gpu:l40s:1 scripts/setup_webshop.sbatch               # deps (Java 21 for pyserini), data, search index, tests
sbatch scripts/eval_webshop.sbatch tag=base                         # base model on the 500 test instructions
sbatch scripts/eval_webshop.sbatch model_path=/project/aip-sologen/aminr/llm_rl_generalization/models/<run_name>/merged
```

Resubmitting a training script with `run_name=<run_name>` resumes from that run's latest checkpoint (and the same
wandb run).
