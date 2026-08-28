#!/bin/bash
set -euo pipefail
cd "$LEHOME"
exec "$PY" "$REPO/scripts/probe_backbone.py" \
    --policy_path "$POLICY_PATH" \
    --dataset_root "${DATASET_ROOT:-Datasets/example/four_types_merged}" \
    --device "${DEVICE:-cpu}"
