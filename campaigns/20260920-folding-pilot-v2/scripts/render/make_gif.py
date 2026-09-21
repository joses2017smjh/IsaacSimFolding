"""Assemble rendered frames into the README GIFs.

Kept separate from the render so a reruns costs no GPU: the frames are the
expensive artefact, and cropping/pacing decisions should not require another
allocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def build(frames: list[Path], out: Path, fps: int, width: int | None) -> None:
    if not frames:
        raise SystemExit(f"no frames for {out.name}")
    imgs = []
    for f in frames:
        a = imageio.imread(f)
        if a.ndim == 3 and a.shape[-1] == 4:
            a = a[..., :3]
        imgs.append(a)
    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    if width and w > width:
        from PIL import Image
        scale = width / w
        imgs = [np.asarray(Image.fromarray(i).resize(
            (width, int(h * scale)), Image.LANCZOS)) for i in imgs]
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, imgs, duration=1.0 / fps, loop=0)
    mb = out.stat().st_size / 1e6
    print(f"  {out.name}  {len(imgs)} frames  {imgs[0].shape[1]}x{imgs[0].shape[0]}  {mb:.1f} MB")
    # GitHub will happily serve a 40 MB GIF and the README will feel broken.
    if mb > 12:
        print(f"    NOTE: {mb:.1f} MB is large for a README; drop --fps or --width")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=880)
    args = ap.parse_args()

    src = Path(args.frames)
    out = Path(args.out)
    for tag, w in (("hero", args.width), ("top", 640)):
        fs = sorted(src.glob(f"{tag}_*.png"))
        if not fs:
            print(f"  (no {tag} frames)", file=sys.stderr)
            continue
        build(fs, out / f"isaacsim_{tag}.gif", args.fps, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
