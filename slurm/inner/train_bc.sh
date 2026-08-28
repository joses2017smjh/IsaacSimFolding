#!/bin/bash
# lerobot-train, run from the LeHome checkout so dataset paths resolve.
set -euo pipefail
cd "$LEHOME"
CFG="$REPO/$CONFIG"
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG" >&2; exit 2; }
BASE=$(basename "$CONFIG" .yaml)
OUT="$LEHOME_DATA/outputs/train/${BASE}_seed${SEED}"
echo "--- config $CFG"
echo "--- output $OUT"
"$PY" -c "import lerobot,torch;print('lerobot',lerobot.__version__,'torch',torch.__version__)"
exec "$PY" -m lerobot.scripts.train \
    --config_path="$CFG" \
    --output_dir="$OUT" \
    --seed="$SEED"
