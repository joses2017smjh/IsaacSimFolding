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

# The Hub rate-limits by IP, and this cluster shares one: the first attempt got
# 946 MB of assets and 8.2 GB of the demonstrations, then
#   429 Too Many Requests ... "We had to rate limit your IP (128.193.8.41).
#    create a HF account and pass a HF_TOKEN"
# A token lifts the anonymous limit and is the real fix; without one this backs
# off and resumes, because `hf download` is resumable and only re-fetches what
# is missing.
[ -n "${HF_TOKEN:-}" ] && echo "--- using HF_TOKEN ---" ||     echo "--- no HF_TOKEN: anonymous, IP rate limits apply ---"

pull() {   # pull <repo> <local-dir>
    local repo=$1 dest=$2 try=0
    until $HF download "$repo" --repo-type dataset --local-dir "$dest"; do
        try=$((try + 1))
        if [ $try -ge 8 ]; then
            echo "FATAL: $repo failed after $try attempts" >&2
            return 1
        fi
        local wait=$((try * 120))
        echo "--- $repo attempt $try failed; sleeping ${wait}s and resuming ---"
        sleep $wait
    done
}

echo "=== 1. garment assets (1.0 GB) -> $LEHOME_DATA/Assets ==="
pull lehome/asset_challenge Assets

echo "=== 2. merged demonstrations (18.9 GB) -> $LEHOME_DATA/Datasets/example ==="
pull lehome/dataset_challenge_merged Datasets/example

if [ "${FETCH_DEPTH:-0}" = "1" ]; then
    echo "=== 3. per-garment demonstrations with depth (24.2 GB) ==="
    pull lehome/dataset_challenge Datasets/per_garment
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
