"""Turn official eval logs into the per-type table every gate from G1 needs.

Reporting rules this enforces rather than trusts:
  - per garment type, always -- long pants and long-sleeved tops are not the
    same task as shorts, and one averaged number hides that
  - seen and unseen reported separately, never pooled
  - garment count stated, because this repo can only reach 48 of the
    leaderboard's 80 (the 8 private per type never shipped)
  - seeds counted, because a single-seed number is not a result

Reads one or more log files, writes a CSV and prints the table.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lehome_fold import eval_log as E  # noqa: E402
from lehome_fold.splits import CATEGORIES  # noqa: E402

SEED_RE = re.compile(r"seed[_-]?(\d+)", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="eval log files")
    ap.add_argument("--label", default="run", help="what this run was")
    ap.add_argument("--out", default="results/eval_summary.csv")
    args = ap.parse_args()

    rows = []
    for path in args.logs:
        p = Path(path)
        text = p.read_text(errors="replace")
        seed_m = SEED_RE.search(p.name)
        seed = int(seed_m.group(1)) if seed_m else 0
        # Prove the log is whole before reading a number out of it. The
        # evaluator prints the run twice -- per episode and per garment -- and
        # a truncated log still yields a plausible aggregate.
        try:
            chk = E.cross_check(text)
            if not chk["aggregates_agree"]:
                print(f"  SKIP {p.name}: truncated or interleaved -- {chk}",
                      file=sys.stderr)
                continue
        except E.NoEpisodes:
            print(f"  SKIP {p.name}: no scored episodes (a crash, not a zero)",
                  file=sys.stderr)
            continue

        garments = E.parse_garments(text)
        if not garments:
            print(f"  (no per-garment summary in {p.name})", file=sys.stderr)
            continue
        for (cat, split), agg in E.per_category(garments).items():
            rows.append({
                "label": args.label, "seed": seed, "category": cat, "split": split,
                "n_garments": agg["n_garments"],
                "success_rate": round(agg["success_rate"], 4),
                "avg_return": round(agg["avg_return"], 3),
                "log": p.name,
            })

    if not rows:
        print("no results parsed -- the evaluations produced no per-garment summary",
              file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'category':12s} {'split':7s} {'seeds':>5s} {'garments':>8s} {'success':>9s}")
    for cat in CATEGORIES:
        for split in ("Seen", "Unseen"):
            sel = [r for r in rows if r["category"] == cat and r["split"] == split]
            if not sel:
                continue
            seeds = len({r["seed"] for r in sel})
            sr = sum(r["success_rate"] for r in sel) / len(sel)
            ng = sel[0]["n_garments"]
            flag = "" if seeds >= 2 else "   <- single seed, not a result"
            print(f"{cat:12s} {split:7s} {seeds:>5d} {ng:>8d} {sr:>8.1%}{flag}")

    print(f"\nwrote {out}")
    print("NOTE: 48 public garments, not the leaderboard's 80 -- the 8 private "
          "per category never shipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
