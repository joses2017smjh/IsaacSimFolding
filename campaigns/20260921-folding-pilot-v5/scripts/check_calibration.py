"""Gate G2: is the success head calibrated?

Consumes the predictions written by train_value.py --dump_val_predictions and
emits a reliability table, ECE/MCE/Brier, and a pass/fail with reasons.

Exit codes are meaningful so a Slurm job can gate on them:
    0  G2 passes
    2  G2 fails (thresholds, or degenerate labels/predictions)
    3  could not evaluate at all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from lehome_fold import calibration as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="npz with arrays 'probs' and 'labels'")
    ap.add_argument("--out", default=None, help="write the report here too")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--max_ece", type=float, default=0.10)
    ap.add_argument("--min_pred_std", type=float, default=1e-2)
    args = ap.parse_args()

    p = Path(args.predictions)
    if not p.exists():
        print(f"G2: no predictions at {p}", file=sys.stderr)
        return 3
    d = np.load(p)
    for k in ("probs", "labels"):
        if k not in d:
            print(f"G2: {p} has no '{k}' array (found {list(d)})", file=sys.stderr)
            return 3

    try:
        rel = C.evaluate(d["probs"], d["labels"], n_bins=args.n_bins)
    except ValueError as exc:
        print(f"G2: cannot evaluate -- {exc}", file=sys.stderr)
        return 3

    ok, reasons = C.gate(rel, max_ece=args.max_ece, min_pred_std=args.min_pred_std)

    lines = [rel.table()]
    for n in rel.notes:
        lines.append(f"NOTE: {n}")
    lines.append("")
    lines.append(f"G2 thresholds: ECE <= {args.max_ece}, pred_std >= {args.min_pred_std}")
    lines.append(f"G2 {'PASS' if ok else 'FAIL'}")
    for r in reasons:
        lines.append(f"  - {r}")
    report = "\n".join(lines)

    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
        summary = {
            "ece": rel.ece, "mce": rel.mce, "brier": rel.brier,
            "base_rate": rel.base_rate, "pred_std": rel.pred_std,
            "pass": ok, "reasons": reasons, "n": int(rel.counts.sum()),
        }
        Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
