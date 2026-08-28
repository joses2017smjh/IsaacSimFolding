#!/bin/bash
set -euo pipefail
cd "$LEHOME"
exec "$PY" "$REPO/scripts/tune_inference.py" \
    --policy_path "$POLICY_PATH" --value_path "$VALUE_PATH" \
    --dataset_root "${DATASET_ROOT:-Datasets/example/four_types_merged}" \
    --lehome "$LEHOME" --repo "$REPO" \
    --garment_type "${GARMENT_TYPE:-top_long}" \
    --feature_path "${FEATURE_PATH:-}" \
    --budget "${BUDGET:-400}" \
    --out "$REPO/results/stage4_thompson.json"
