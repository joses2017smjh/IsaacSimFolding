#!/bin/bash
set -euo pipefail
cd "$LEHOME"
ARGS=(--policy_path "$POLICY_PATH"
      --dataset_root "${DATASET_ROOT:-Datasets/example/four_types_merged}"
      --feature_path "$FEATURE_PATH" --hidden_dim "$HIDDEN_DIM"
      --out "$STAGE_OUT"
      --dump_val_predictions "$REPO/results/value_val_predictions.npz"
      --steps "${STAGE_STEPS:-3000}" --eval_only "${EVAL_ONLY:-0}"
      --device "${DEVICE:-cuda}" --seed "${SEED:-0}")
[ -n "${ROLLOUT_DIR:-}" ] && ARGS+=(--rollout_dir "$ROLLOUT_DIR")
exec "$PY" "$REPO/scripts/train_value.py" "${ARGS[@]}"
