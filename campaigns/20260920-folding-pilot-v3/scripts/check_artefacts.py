"""Verify rendered artefacts are FRESH and non-blank, not merely present.

Two failure modes have both bitten this repo and neither is caught by looking
at a file list:

  stale  -- a re-run crashes and leaves the previous run's GIF in place, so the
            directory looks correct and describes an episode that never ran
  blank  -- the garment mesh fails to render, so the file is fresh, valid, and
            shows an empty table under a caption claiming a successful fold

`--newer-than` covers the first by comparing mtimes against a marker file, the
way a video guard counts files newer than a marker rather than counting files.
The garment check covers the second.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--newer-than", default=None,
                    help="marker file; artefacts older than it are stale")
    ap.add_argument("--min-garment", type=float, default=0.3,
                    help="percent of pixels that must look like garment")
    args = ap.parse_args()

    import numpy as np
    import imageio.v2 as imageio

    files = [f for p in args.paths for f in sorted(glob.glob(p))]
    if not files:
        print(f"FAIL: no artefacts matched {args.paths}", file=sys.stderr)
        return 1

    cutoff = os.path.getmtime(args.newer_than) if args.newer_than else None
    stale, blank = [], []
    for f in files:
        if cutoff is not None and os.path.getmtime(f) < cutoff:
            stale.append(os.path.basename(f))
            continue
        rd = imageio.get_reader(f)
        fr = [np.asarray(x)[:, :, :3] for x in rd]
        rd.close()
        m = fr[len(fr) // 2]
        r, g, b = m[:, :, 0].astype(int), m[:, :, 1].astype(int), m[:, :, 2].astype(int)
        pct = float(((r - g > 25) & (r - b > 25)).mean() * 100)
        if pct < args.min_garment:
            blank.append((os.path.basename(f), pct))

    print(f"checked {len(files)} artefacts")
    for n in stale:
        print(f"  STALE  {n}", file=sys.stderr)
    for n, p in blank:
        print(f"  BLANK  {n}  garment={p:.2f}%", file=sys.stderr)
    if stale or blank:
        print(f"FAIL: {len(stale)} stale, {len(blank)} blank", file=sys.stderr)
        return 1
    print("OK: all fresh and non-blank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
