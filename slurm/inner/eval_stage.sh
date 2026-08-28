#!/bin/bash
set -euo pipefail
cd "$LEHOME"
ARGS=(--policy_type "$POLICY_TYPE" --garment_type "$GARMENT_TYPE"
      --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS"
      --device cpu --enable_cameras --headless)
[ -n "${POLICY_PATH:-}" ] && ARGS+=(--policy_path "$POLICY_PATH")
[ -n "${DATASET_ROOT:-}" ] && ARGS+=(--dataset_root "$DATASET_ROOT")
exec "$PY" "$REPO/scripts/run_eval.py" "${ARGS[@]}"
