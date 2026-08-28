#!/bin/bash
set -euo pipefail
exec "$PY" "$REPO/scripts/check_calibration.py" \
    --predictions "$REPO/results/value_val_predictions.npz" \
    --out "$REPO/results/g2_calibration.txt"
