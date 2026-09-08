#!/bin/bash
set -euo pipefail
cd "$LEHOME"
# Storm and --enable_cameras are mutually exclusive. AppLauncher builds the
# Isaac Lab render product at LAUNCH when that flag is set, and 5.1's RTX
# delegate segfaults against this cluster's driver at 586 ms -- before any
# camera exists, so no amount of camera stubbing can help. When Storm is
# supplying the pixels the flag must be absent, not merely unused.
# LeHome's README calls --device the INFERENCE device and says only cpu, but
# evaluation.py feeds it to parse_env_cfg, which sets the SIMULATION device.
# PhysX particle cloth needs GPU dynamics -- policy_rollout51.py has carried
# the comment "particle cloth needs GPU dynamics" and --sim_device cuda:0 all
# along, and those are the runs that actually folded. Under --device cpu every
# episode ends with the garment exactly where it spawned: identical success
# distances across 27 episodes, two policies and two different garments.
# Parameterised so the two can be compared rather than argued about.
CAM=(--enable_cameras)
[ "${LH_STORM_EVAL:-0}" = "1" ] && CAM=()
ARGS=(--policy_type "$POLICY_TYPE" --garment_type "$GARMENT_TYPE"
      --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS"
      --device "${SIM_DEVICE:-cuda:0}" "${CAM[@]}" --headless)
[ -n "${POLICY_PATH:-}" ] && ARGS+=(--policy_path "$POLICY_PATH")
[ -n "${DATASET_ROOT:-}" ] && ARGS+=(--dataset_root "$DATASET_ROOT")
exec "$PY" "$REPO/scripts/run_eval.py" "${ARGS[@]}"
