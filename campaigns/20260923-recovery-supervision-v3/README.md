# Recovery supervision v3

Successor to `20260922-closed-loop-training-v2`, which is a completed negative
result: three episode-weighted AWR iterations did not improve H10, because
the policy's own rollouts contain too few successes to imitate. v3 does not
repeat that recipe. It builds a better supervision source.

```
H10 student on training-only garments
  -> snapshot student-visited roots (steps 50/150/250/350)
  -> from every root, execute a FIXED set of 4 candidate continuations
     (2 at H10, 2 at H50, fixed seeds) to the end + 60-step settle
  -> record every attempt; keep only settled successes as labels
     (rendered observation at a state + actions actually executed from it)
  -> coverage gate: >= 6 successes, >= 4 roots, >= 3 rows, or no training
  -> supervised training with the original BC anchor, v2's optimisation unchanged
  -> reload -> screen (H10 x1) -> confirm (H10 x2 + H50) -> untouched final set
```

Evaluation is repeated and preregistered in `manifest.json`: the baseline has
two H10 runs, and a candidate needs two H10 runs, one H50 run and a Fisher
test before any claim. The retention guard uses whole held-out
demonstration episodes of garments the anchor never sees; v2's guard is kept
only as an anchor-fit metric.

`STATUS.md` is the live page. `scripts/driver.py` owns every submission:

```bash
C=$PWD; PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
$PY scripts/driver.py --campaign $C tick --dry-run
$PY scripts/driver.py --campaign $C tick
$PY -m pytest tests -q
```
