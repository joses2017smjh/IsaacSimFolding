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

# Storm-backed cameras, when asked for.
#
# LeHome's eval builds TiledCamera render products, and 5.1's RTX delegate
# segfaults against this cluster's driver -- verified, not inferred: a probe
# reached Isaac Sim and died with "Segmentation fault (core dumped)". 6.0
# renders but has no particle cloth. LH_STORM_EVAL=1 swaps the cameras for
# Storm-backed ones so the challenge's OWN eval loop runs unmodified and the
# success numbers still come from its checker.
if os.environ.get("LH_STORM_EVAL") == "1":
    try:
        from lehome_fold.storm_camera import DEPTH_IS_SYNTHETIC
        from lehome_fold.storm_eval import enable_when_imported
        from lehome_fold.storm_obs import StormObserver, StormObsConfig

        _obs = StormObserver(StormObsConfig(
            assets=os.environ["LH_STORM_ASSETS"],
            garment_dir=os.environ["LH_STORM_GARMENT_DIR"],
            workdir=os.environ.get("LH_STORM_WORKDIR", ""), extra={}))
        # Deferred: the env module imports isaaclab_tasks, which does not exist
        # until AppLauncher has run, and LeHome's eval owns the launch. A
        # post-import hook is the only point both after the app exists and
        # before the env is built.
        enable_when_imported("lehome.tasks.bedroom.garment_bi_v2", _obs,
                             device=os.environ.get("LH_STORM_DEVICE", "cuda"))
        if DEPTH_IS_SYNTHETIC:
            print("[run_eval] NOTE: Storm supplies colour; depth is synthetic. "
                  "The success checker reads particle positions, not depth.",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        # Refuse rather than fall through to RTX, which would segfault and
        # look like a different problem entirely.
        raise SystemExit(f"[run_eval] LH_STORM_EVAL=1 but setup failed: {exc!r}")

try:
    runpy.run_module("scripts.eval", run_name="__main__")
finally:
    for dst in _copied:
        try:
            os.unlink(dst)
        except OSError:
            pass
