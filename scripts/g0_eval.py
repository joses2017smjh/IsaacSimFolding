"""Run the OFFICIAL LeHome evaluator with the G0 floor policies registered.

The point of G0 is that "success rate" means the number the leaderboard meant.
That rules out reimplementing the eval loop, so this does not: it registers two
extra policies into LeHome's own `PolicyRegistry` and then hands control to
`scripts.eval` unchanged, arguments and all.

`--policy_type` is a free-form string in LeHome's parser (no `choices=`), so a
newly registered name is accepted without patching the submodule. Nothing under
external/lehome-challenge is modified by this file, which is the property that
keeps the harness comparable.

Run from the lehome-challenge checkout, so that `scripts` resolves to theirs:

    cd external/lehome-challenge
    python /path/to/g0_eval.py --policy_type g0_uniform --garment_type top_long ...
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

# `scripts.eval_policy` is LeHome's package; importing it runs the decorators
# that register lerobot/custom/docker. Our module has to be imported as part of
# that same package so its relative imports (`.base_policy`) resolve.
HERE = Path(__file__).resolve().parent
import scripts.eval_policy as ep  # noqa: E402

_TARGET = Path(ep.__file__).parent / "_g0_policies.py"
# Copy rather than symlink: the submodule may be checked out on a filesystem
# where a dangling link across mounts is awkward, and this file is disposable.
shutil.copyfile(HERE / "g0_policies.py", _TARGET)

import importlib  # noqa: E402

importlib.import_module("scripts.eval_policy._g0_policies")
print(f"[g0_eval] registered: {ep.PolicyRegistry.list_policies()}", flush=True)

try:
    runpy.run_module("scripts.eval", run_name="__main__")
finally:
    # Leave the submodule exactly as it was found.
    try:
        os.unlink(_TARGET)
    except OSError:
        pass
