#!/bin/bash
set -euo pipefail
cd "$LEHOME"
exec "$PY" -u "$REPO/scripts/finetune_rasterised.py" \
    --policy_path "$BASE_CKPT" \
    --capture_glob "$CAP_GLOB" \
    --out "$FT_OUT" \
    --steps "$FT_STEPS" \
    --unfreeze "$FT_UNFREEZE"
