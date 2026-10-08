"""Post hoc: grade v9 (fresh seeds) and v5-v8 (seeds 200-207) under every metric.

Written 2026-09-30, after v9 finished. Nothing here was preregistered: v2-v9
gate on SETTLED. It exists because the challenge scores a different metric
(src/lehome_fold/challenge_metric.py documents and tests that metric).

Metrics, per episode:
  settled     terminal_success: all 4 conditions after the 60-step terminal settle (the campaigns' metric)
  runner      official_ever_success: the runner's replica of the native checker. It is sampled every
              25 steps from step 20, twice the challenge's rate, so it reads high on transient folds.
              It is NOT the challenge's metric.
  phase27     the challenge's sampler at a fresh process's phase (steps 27, 77, ..., 577),
              applied to the runner's per-step geometric trace
  phase_avg   the same sampler averaged over its 50 equally likely phases
  reach       any policy step with all 4 conditions (geometric_ever_success): an upper bound

The phase27 and phase_avg rows apply the challenge's sampling to this runner's
trajectories. The challenge loop itself would produce different ones.

    PYTHONPATH=src python campaigns/20260928-fresh-seed-remeasure-v9/analysis/challenge-metric/regrade.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from lehome_fold import challenge_metric as M  # noqa: E402
import analysis as A  # noqa: E402  (v9's frozen analysis: exact Fisher)

CAMPAIGNS = REPO / "campaigns"
HISTORY = {"v5": ("20260924-matched-comparison-v5", "a2-step000250"),
           "v6": ("20260924-pose-balanced-v6", "pb1-step000200"),
           "v7": ("20260925-depth-qualified-v7", "dq1-step000200"),
           "v8": ("20260926-anchor-diagnostic-v8", "an1-step000300")}
METRICS = ("settled", "runner", "phase27", "phase_avg", "reach")


def grade(rollout: dict) -> dict:
    trace = M.trace_from_rollout(rollout)
    return {"settled": float(bool(rollout["terminal_success"])),
            "runner": float(bool(rollout["official_ever_success"])),
            "phase27": float(M.first_sampled_success(trace, M.FRESH_PROCESS_PHASE) is not None),
            "phase_avg": M.phase_average(trace),
            "reach": float(bool(rollout["geometric_ever_success"]))}


def tally(dirs) -> dict:
    total = collections.Counter()
    n = 0
    for d in dirs:
        g = grade(json.loads((d / "rollout.json").read_text()))
        total.update(g)
        n += 1
    return {"n": n, **{m: round(total[m], 2) for m in METRICS}}


def compare(cand: dict, base: dict) -> dict:
    out = {"candidate": cand, "baseline": base, "p_candidate_better": {}, "p_baseline_better": {}}
    for m in ("settled", "runner", "phase27", "reach"):
        c, b = int(cand[m]), int(base[m])
        out["p_candidate_better"][m] = round(A.fisher_one_sided(c, cand["n"], b, base["n"]), 4)
        out["p_baseline_better"][m] = round(A.fisher_one_sided(b, base["n"], c, cand["n"]), 4)
    return out


def main() -> int:
    v9 = CAMPAIGNS / "20260928-fresh-seed-remeasure-v9/evaluation"
    report = {"note": __doc__.split("\n\n")[0], "v9_fresh_seeds": {}, "reused_seeds_200_207": {}}
    for h in ("h10", "h50"):
        for block in ("a40", "rtx8000", "pooled"):
            blocks = ("a40", "rtx8000") if block == "pooled" else (block,)
            dirs = lambda pol: [d for b in blocks for d in sorted((v9 / f"{pol}-{b}/remeasure").glob(f"dev0?_{h}_*"))]  # noqa: E731
            report["v9_fresh_seeds"][f"{h}_{block}"] = compare(tally(dirs("an1-step000300")), tally(dirs("baseline")))
    for tag, (camp, label) in HISTORY.items():
        ev = CAMPAIGNS / camp / "evaluation"
        for h in ("h10", "h50"):
            dirs = lambda lab: sorted(d for r in ("r1", "r2") for d in (ev / f"{lab}-{r}/benchmark").glob(f"dev0?_{h}_*"))  # noqa: E731
            report["reused_seeds_200_207"][f"{tag}_{label}_{h}"] = compare(tally(dirs(label)), tally(dirs("baseline")))
    out = HERE / "regrade.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for section in ("v9_fresh_seeds", "reused_seeds_200_207"):
        print(f"\n{section}  (candidate vs baseline: settled | runner | phase27 | phase_avg | reach)")
        for k, v in report[section].items():
            c, b = v["candidate"], v["baseline"]
            print(f"  {k:32s} " + " | ".join(f"{c[m]:g} vs {b[m]:g}" for m in METRICS) + f"   (n {c['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
