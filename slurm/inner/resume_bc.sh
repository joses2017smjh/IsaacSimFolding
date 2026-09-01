#!/bin/bash
# Resume lerobot-train from the last checkpoint, run from the LeHome checkout
# so dataset paths resolve exactly as they did for the original run.
set -euo pipefail
cd "$LEHOME"
OUT="$LEHOME_DATA/outputs/train/bc_smolvla_seed${SEED:-0}"
CKPT="$OUT/checkpoints/last/pretrained_model/train_config.json"
[ -f "$CKPT" ] || { echo "FATAL: no train_config.json at $CKPT" >&2; exit 2; }
echo "--- resuming from $(readlink -f "$OUT/checkpoints/last")"
"$PY" -c "
import json; d=json.load(open('$CKPT'))
print('--- target steps:', d['steps'], '| batch', d['batch_size'], '| save_freq', d['save_freq'])"
export PYTHONUNBUFFERED=1
# --resume=true continues optimizer and scheduler state from training_state/,
# so this is a continuation of the same schedule, not a fresh run warm-started
# from the weights.
exec "$PY" -u -m lerobot.scripts.lerobot_train --config_path="$CKPT" --resume=true
