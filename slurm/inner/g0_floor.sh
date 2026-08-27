#!/bin/bash
# Inner half of 05_g0_floor.sbatch. Runs inside the container.
#
# Calls the OFFICIAL evaluator. scripts/g0_eval.py registers the two floor
# policies into LeHome's own PolicyRegistry and then hands control to
# `scripts.eval` unchanged -- no fork of the eval loop, no reimplemented
# scorer, which is the only way the resulting number means what the
# leaderboard's number meant.
set -euo pipefail

# cwd must be the checkout root: LeHome resolves Assets/ and its particle
# config relative to it, and `scripts` has to import as their package.
cd "$LEHOME"

echo "--- interpreter: $PY"
"$PY" -c "import sys; print('python', sys.version.split()[0])"
"$PY" -c "import isaaclab, isaacsim, lerobot; print('isaaclab + isaacsim + lerobot import OK')"

echo "--- assets visible from cwd=$PWD ---"
ls Assets/objects/Challenge_Garment/Release 2>/dev/null || {
    echo "FATAL: Assets/ not visible at $LEHOME." >&2
    echo "  On LH_ROUTE=official this means the bind mount did not take." >&2
    echo "  Run 02_fetch_data.sbatch, or fall back to LH_ROUTE=source." >&2
    exit 5; }

# --headless and --enable_cameras are both required: headless because there is
# no display on a compute node, and enable_cameras because the policy consumes
# RGB and without it the observation dict comes back without images.
"$PY" "$REPO/scripts/g0_eval.py" \
    --policy_type "$POLICY_TYPE" \
    --garment_type "$GARMENT_TYPE" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --device cpu \
    --enable_cameras \
    --headless
