#!/bin/bash
set -euo pipefail
cd "$LEHOME"
exec "$PY" "$REPO/scripts/trainer_loop.py" \
    --shared_dir "$SHARED_DIR" --rollout_dir "$ROLLOUT_DIR" \
    --policy_path "$POLICY_PATH" --value_path "$VALUE_PATH" \
    --dataset_root "${DATASET_ROOT:-Datasets/example/four_types_merged}" \
    --out "$STAGE_OUT" \
    --max_lag "${MAX_LAG:-2}" --beta "${BETA:-1.0}" \
    --use_awr "${USE_AWR:-1}" --use_recap "${USE_RECAP:-1}" \
    --dry_run "${DRY_RUN:-0}"
