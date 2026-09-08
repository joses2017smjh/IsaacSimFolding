#!/bin/bash
set -euo pipefail
cd "$LEHOME"
# Storm and --enable_cameras are mutually exclusive. AppLauncher builds the
# Isaac Lab render product at LAUNCH when that flag is set, and 5.1's RTX
# delegate segfaults against this cluster's driver at 586 ms -- before any
# camera exists, so no amount of camera stubbing can help. When Storm is
# supplying the pixels the flag must be absent, not merely unused.
CAM=(--enable_cameras)
[ "${LH_STORM_EVAL:-0}" = "1" ] && CAM=()
ARGS=(--policy_type "$POLICY_TYPE" --garment_type "$GARMENT_TYPE"
      --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS"
      --device cpu "${CAM[@]}" --headless)
[ -n "${POLICY_PATH:-}" ] && ARGS+=(--policy_path "$POLICY_PATH")
[ -n "${DATASET_ROOT:-}" ] && ARGS+=(--dataset_root "$DATASET_ROOT")
exec "$PY" "$REPO/scripts/run_eval.py" "${ARGS[@]}"
