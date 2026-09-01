#!/bin/bash
# Action fidelity on in-distribution frames. Run from the LeHome checkout so
# dataset paths resolve the same way they do for training.
set -euo pipefail
cd "$LEHOME"
exec "$PY" -u "$REPO/scripts/eval/action_fidelity.py" \
    --policy_path "$CKPT" \
    --dataset_root "$LEHOME_DATA/Datasets/example/four_types_merged" \
    --episodes "${EPISODES:-25,26,2,250,500,750}" \
    --samples "${SAMPLES:-120}" \
    --out "$REPO/results/action_fidelity_${TAG}.json"
