#!/bin/bash
# Inner half of 02_rtx_probe.sbatch. Runs inside the container.
set -uo pipefail

echo "--- interpreter: $PY"
"$PY" -c "import sys; print('python', sys.version.split()[0])"
"$PY" -c "import isaaclab, isaacsim; print('isaaclab + isaacsim import OK')" || {
    echo "FATAL: isaaclab/isaacsim not importable in $PY" >&2; exit 4; }

cd "$REPO"
# Not `exec`: the sbatch wrapper wants this script's exit status, and a
# segfaulting child is the signal it is looking for.
"$PY" "$REPO/scripts/probe_rtx_tiled.py"
rc=$?
echo "--- probe exited $rc"
exit $rc
