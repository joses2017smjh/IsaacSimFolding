#!/bin/bash
# lerobot + torch only. Pinned to LeHome's versions so a checkpoint trained
# here loads in their evaluator without a version argument.
set -euo pipefail
uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --python "$PY" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "torch==2.7.0" "torchvision==0.22.0" "lerobot==0.4.3" \
    "transformers>=4.57.6" "num2words>=0.5.14" "pyarrow" "pandas"
"$PY" - <<'PYV'
import importlib.metadata as md
for p in ("torch", "lerobot", "transformers"):
    try: print(f"  {p:14s} {md.version(p)}")
    except md.PackageNotFoundError: print(f"  {p:14s} (absent)")
import torch; print("  cuda available:", torch.cuda.is_available())
from lerobot.policies.factory import make_policy  # noqa: F401
print("  lerobot factory imports OK")
PYV
echo "venv size: $(du -sh "$UV_PROJECT_ENVIRONMENT" | cut -f1)"
