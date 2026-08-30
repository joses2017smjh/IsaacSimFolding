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
# lerobot 0.4.3 renamed this: lerobot.scripts.train does not exist,
# it is lerobot.scripts.lerobot_train (console script: lerobot-train).
# -u / PYTHONUNBUFFERED: lerobot logs progress to stdout, which Python buffers
# when it is not a tty. The first run showed the stderr dataloader warnings and
# then nothing for 13 minutes -- indistinguishable from a hang, on a 6 h job.
# Being able to see step/loss is worth more than the buffering.
export PYTHONUNBUFFERED=1
exec "$PY" -u -m lerobot.scripts.lerobot_train \
    --config_path="$CFG" \
    --output_dir="$OUT" \
    --seed="$SEED"
