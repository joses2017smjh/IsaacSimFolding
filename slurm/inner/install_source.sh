#!/bin/bash
# Inner half of 04_install_source.sbatch. Runs inside the container.
#
# The editable route. Follows docs/installation.md from the challenge repo,
# which is NOT just `uv sync`: Isaac Lab comes from a FORK, installed through
# isaaclab.sh, and the lehome package is installed editable on top.
#
#   github.com/lehome-official/IsaacLab  (fork of isaac-sim/IsaacLab, BSD-3)
#
# Getting isaaclab from PyPI instead would give the stock 2.3.x without
# whatever the fork changed, and the garment task is the reason the fork exists.
set -euo pipefail

cd "$LEHOME"

echo "=== 1. resolving lehome/pyproject.toml into $UV_PROJECT_ENVIRONMENT ==="
uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
uv sync --frozen

echo "=== 2. forked Isaac Lab into third_party ==="
if [ ! -d third_party/IsaacLab/.git ]; then
    git clone --depth 1 https://github.com/lehome-official/IsaacLab.git third_party/IsaacLab
else
    echo "already present"
fi

echo "=== 3. installing Isaac Lab (-i none: no RL library, we bring our own) ==="
VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" ./third_party/IsaacLab/isaaclab.sh -i none

echo "=== 4. installing lehome editable ==="
VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --python "$PY" -e ./source/lehome

echo "=== 5. verifying ==="
"$PY" - <<'PYV'
import importlib.metadata as md
for p in ("isaacsim", "isaaclab", "lerobot", "torch", "lehome"):
    try:
        print(f"  {p:10s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:10s} (absent)")
import lehome  # noqa: F401
print("  lehome imports OK")
PYV
echo "venv size: $(du -sh "$UV_PROJECT_ENVIRONMENT" | cut -f1)"
