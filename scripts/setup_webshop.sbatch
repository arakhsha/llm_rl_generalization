#!/bin/bash
#SBATCH --job-name=setup_webshop
#SBATCH --account=aip-sologen
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/llm_rl_generalization/logs/%x_%j.out
# Sets up WebShop for evaluation in $WEBSHOP_DIR: its code at a pinned commit, the full product set
# (1.18M products) with the human-written instructions, and the Lucene search index over the products.
# Usage (from the repo root): sbatch scripts/setup_webshop.sbatch
#   The CPU queue can be long; with --gres=gpu:l40s:1 it starts right away and ends with a short eval.
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Submit from the repo root"; exit 1; }
repo=$PWD
source scripts/env.sh
module load java/21.0.1  # pyserini runs Lucene through pyjnius
export JAVA_TOOL_OPTIONS="-Xmx32g"  # the module sets -Xmx2g, too little to index all products

uv lock
uv sync
uv run --no-sync python -c "import importlib.metadata as md; print({p: md.version(p) for p in ['torch', 'transformers', 'vllm', 'pyserini', 'spacy', 'flask', 'beautifulsoup4']})"

if [[ ! -d "$WEBSHOP_DIR/.git" ]]; then
    git clone https://github.com/princeton-nlp/WebShop.git "$WEBSHOP_DIR"
fi
cd "$WEBSHOP_DIR"
git checkout -q 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
# Load all products instead of the 1,000-product preview (WebShop README, setup step 6).
sed -i 's#items_ins_v2_1000.json#items_ins_v2.json#; s#items_shuffle_1000.json#items_shuffle.json#' web_agent_site/utils.py
git diff --stat

# Product data and instructions. The Google Drive files in WebShop's setup.sh are gone (404), so they come
# from a third-party Hugging Face mirror at a pinned revision. Its small files are compared with a second,
# independent mirror, and the full product file with the 1,000-product preview (its first 1,000 products).
hf() { uv run --project "$repo" --no-sync hf "$@"; }
hf download YWZBrandon/webshop-data --repo-type dataset --revision ce990fff5aee388db2706f07820c578ab68e0453 \
    --local-dir data items_shuffle.json items_ins_v2.json items_human_ins.json items_shuffle_1000.json items_ins_v2_1000.json
hf download Ricardo-H/webshop-data --repo-type dataset --revision 206ebf01f1563013f1208e9dd33bf34f0a2f4c54 \
    --local-dir data/check items_human_ins.json items_shuffle_1000.json items_ins_v2_1000.json
for f in items_human_ins.json items_shuffle_1000.json items_ins_v2_1000.json; do
    cmp "data/$f" "data/check/$f" && echo "$f matches the second mirror"
done
ls -la data
uv run --project "$repo" --no-sync python - <<'EOF'
import json
full = json.load(open("data/items_shuffle.json"))
asins = {p["asin"] for p in full}
human = json.load(open("data/items_human_ins.json"))
attrs = json.load(open("data/items_ins_v2.json"))
print(f"{len(full)} products ({len(asins)} ASINs); first 1,000 equal items_shuffle_1000.json: "
      f"{full[:1000] == json.load(open('data/items_shuffle_1000.json'))}; "
      f"products with human instructions: {len(set(human) & asins)}/{len(human)}; "
      f"{sum(len(v) for v in human.values())} instructions; attribute entries: {len(set(attrs) & asins)}/{len(attrs)}")
EOF

# Search index, as in WebShop's setup.sh but only the full one (the env uses search_engine/indexes).
cd search_engine
if [[ ! -f indexes/.done ]]; then
    mkdir -p resources resources_100 resources_1k resources_100k
    uv run --project "$repo" --no-sync python convert_product_file_format.py
    rm -rf indexes
    uv run --project "$repo" --no-sync python -m pyserini.index.lucene \
        --collection JsonCollection --input resources --index indexes \
        --generator DefaultLuceneDocumentGenerator --threads "$SLURM_CPUS_PER_TASK" \
        --storePositions --storeDocvectors --storeRaw
    touch indexes/.done
    rm -rf resources_100 resources_1k resources_100k  # only used for the smaller indexes
fi
du -sh "$WEBSHOP_DIR"/data "$WEBSHOP_DIR"/search_engine/*
cd "$repo"

uv run --no-sync pytest -q -s  # -s shows the WebShop pages played in test_webshop.py
if nvidia-smi -L &>/dev/null; then
    export HF_HUB_OFFLINE=1
    uv run --no-sync python -m llm_rl.eval_webshop limit=8 tag=smoke wandb.enabled=false
fi
