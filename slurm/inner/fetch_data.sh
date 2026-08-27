#!/bin/bash
# Inner half of 02_fetch_data.sbatch. Runs inside the container.
#
# `hf` ships with huggingface_hub, which lerobot pulls in, so the official
# image already has it. On the source route it may not be there yet, hence the
# uvx fallback -- uv is baked into lehome.sif for exactly this kind of gap.
set -euo pipefail

HF=""
if [ -x "$(dirname "$PY")/hf" ]; then
    HF="$(dirname "$PY")/hf"
elif "$PY" -c "import huggingface_hub" 2>/dev/null; then
    HF="$PY -m huggingface_hub.commands.huggingface_cli"
else
    echo "--- huggingface_hub not in $PY, falling back to uvx ---"
    HF="uvx --from huggingface_hub hf"
fi
echo "--- hf entrypoint: $HF"

cd "$LEHOME_DATA"

echo "=== 1. garment assets (1.0 GB) -> $LEHOME_DATA/Assets ==="
$HF download lehome/asset_challenge --repo-type dataset --local-dir Assets

echo "=== 2. merged demonstrations (18.9 GB) -> $LEHOME_DATA/Datasets/example ==="
$HF download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example

if [ "${FETCH_DEPTH:-0}" = "1" ]; then
    echo "=== 3. per-garment demonstrations with depth (24.2 GB) ==="
    $HF download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/per_garment
else
    echo "=== 3. skipping depth dataset (FETCH_DEPTH=1 to pull it) ==="
fi

mkdir -p "$LEHOME_DATA/outputs"

echo
echo "=== what landed ==="
du -sh "$LEHOME_DATA"/* 2>/dev/null || true

echo
echo "=== the split that must never be trained on ==="
# 10 Seen + 2 Unseen per type. The 2 public Unseen garments per type are the
# validation split and the 8 private Holdout ones per type never shipped --
# consistent with the leaderboard having 20 per type, 10 of them exposed.
find "$LEHOME_DATA/Assets/objects/Challenge_Garment/Release" -maxdepth 1 -type d 2>/dev/null | while read -r d; do
    [ "$d" = "$LEHOME_DATA/Assets/objects/Challenge_Garment/Release" ] && continue
    n=$(basename "$d")
    seen=$(find "$d" -maxdepth 1 -name "*_Seen_*" | wc -l)
    unseen=$(find "$d" -maxdepth 1 -name "*_Unseen_*" | wc -l)
    echo "  $n: seen=$seen unseen=$unseen"
done
