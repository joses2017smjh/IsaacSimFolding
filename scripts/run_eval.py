"""Run the OFFICIAL LeHome evaluator with this repo's policies registered.

The point of every gate here is that "success rate" means the number the
leaderboard meant. That rules out reimplementing the eval loop, so this does
not: it injects our policy modules into LeHome's own `scripts.eval_policy`
package, lets their decorators register them, and then hands control to
`scripts.eval` unchanged -- same arguments, same loop, same success checker.

Registered by this entry point:
    g0_uniform, g0_hold      floors            (scripts/g0_policies.py)
    candidate, recap         Stage 2 and 3     (scripts/stage_policies.py)
LeHome's own lerobot / custom / docker stay registered as usual, so Stage 1 is
evaluated with `--policy_type lerobot` and needs nothing from this file beyond
being able to run.

`--policy_type` is a free-form string in LeHome's parser (no `choices=`), so a
newly registered name is accepted without patching the submodule. The injected
files are removed on exit, so nothing under external/ is modified -- that
property is what keeps the harness comparable and it is worth protecting.

Run from the lehome-challenge checkout, so `scripts` resolves to theirs:

    cd $LEHOME
    python $REPO/scripts/run_eval.py --policy_type candidate ...
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The repo's own package has to be importable from inside LeHome's process.
sys.path.insert(0, str(HERE.parent / "src"))

import scripts.eval_policy as ep  # noqa: E402

_PKG = Path(ep.__file__).parent
_INJECTED = ("g0_policies", "stage_policies")
_copied: list[Path] = []

for name in _INJECTED:
    src = HERE / f"{name}.py"
    if not src.exists():
        continue
    dst = _PKG / f"_{name}.py"
    # Copy rather than symlink: the submodule may sit on a filesystem where a
    # cross-mount link is awkward, and these files are disposable.
    shutil.copyfile(src, dst)
    _copied.append(dst)

import importlib  # noqa: E402

for name in _INJECTED:
    dst = _PKG / f"_{name}.py"
    if not dst.exists():
        continue
    try:
        importlib.import_module(f"scripts.eval_policy._{name}")
    except Exception as exc:  # noqa: BLE001
        # A Stage 2/3 module that cannot import (missing value head, wrong
        # lerobot version) must not take the G0 floors down with it. Report and
        # continue -- but never silently, or a typo becomes "policy not found".
        print(f"[run_eval] WARNING: {name} did not register: {exc!r}", flush=True)

print(f"[run_eval] registered: {sorted(ep.PolicyRegistry.list_policies())}", flush=True)

try:
    runpy.run_module("scripts.eval", run_name="__main__")
finally:
    for dst in _copied:
        try:
            os.unlink(dst)
        except OSError:
            pass
