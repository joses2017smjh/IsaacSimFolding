#!/bin/bash
set -euo pipefail
exec "$PY" "$REPO/scripts/rollout_worker.py" \
    --shared_dir "$SHARED_DIR" --rollout_dir "$ROLLOUT_DIR" \
    --lehome "$LEHOME" --repo "$REPO" --worker_id "$WORKER_ID" \
    --dataset_root "${DATASET_ROOT:-Datasets/example/four_types_merged}" \
    --policy_type "${POLICY_TYPE:-recap}" \
    --value_path "${VALUE_PATH:-}" --feature_path "${FEATURE_PATH:-}" \
    --episodes_per_batch "${NUM_EPISODES:-4}" \
    --garment_types "${GARMENT_TYPE:-top_long,top_short,pant_long,pant_short}"
