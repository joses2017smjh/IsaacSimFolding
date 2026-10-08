"""Build the failure/fix comparison figure from the rendered gallery.

Each pair is a bug this renderer actually shipped, reintroduced by
`render_scene.py --defect <name>` so the gallery regenerates from source and
cannot drift from what it claims.

Captions carry a MEASURED difference against the corrected frame, not an
adjective. Two candidate defects -- a single-sided cloth mesh and a missing
180 deg yaw -- turned out to be byte-identical no-ops and were dropped rather
than captioned as if they mattered. That is why there are four here and not six.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

BG, FG, MUTED, BAD, GOOD = "#0d1117", "#e6edf3", "#8b949e", "#f85149", "#3fb950"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "font.family": "DejaVu Sans",
})

PAIRS = [
    ("exposure", "lights 4x too bright",
     "frame mean 221/255, 0.3% already clipped to white"),
    ("framing", "orbit at 0.92 m, 0.34 m high",
     "grazes the tabletop: half the frame is wood, no robot"),
    ("garment_scale", "the USD's own xformOp:scale = 0.01 left in place",
     "a 0.95 m mesh renders at 4 mm -- the garment vanishes"),
    ("gravity", "physics stepped with an unanchored base",
     "the arm slumps off the pose the demonstration commanded"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="docs/img/gallery")
    ap.add_argument("--out", default="docs/img/render_failures.png")
    args = ap.parse_args()
    g = Path(args.gallery)

    fixed = imageio.imread(g / "fixed_hero.png").astype(np.float32)[..., :3]
    rows = [p for p in PAIRS if (g / f"defect_{p[0]}_hero.png").exists()]
    if not rows:
        raise SystemExit("no defect frames found")

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 3.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, what, why) in enumerate(rows):
        bad = imageio.imread(g / f"defect_{name}_hero.png").astype(np.float32)[..., :3]
        d = np.abs(bad - fixed)
        pct = (d.max(axis=2) > 8).mean() * 100

        axes[r, 0].imshow(bad.astype(np.uint8))
        axes[r, 0].set_title(f"✗  {what}", color=BAD, fontsize=11,
                             fontweight="bold", loc="left")
        axes[r, 0].set_xlabel(f"{why}\n{pct:.0f}% of pixels differ from the fix",
                              color=MUTED, fontsize=9, loc="left")

        axes[r, 1].imshow(fixed.astype(np.uint8))
        axes[r, 1].set_title("✓  corrected", color=GOOD, fontsize=11,
                             fontweight="bold", loc="left")
        axes[r, 1].set_xlabel("same scene, same camera, same seed",
                              color=MUTED, fontsize=9, loc="left")
        for c in (0, 1):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_color(BAD if c == 0 else GOOD)
                sp.set_linewidth(1.6)

    fig.suptitle("Isaac Sim scene bugs, and what each one actually looked like",
                 color=FG, fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  {out}  ({len(rows)} pairs, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
