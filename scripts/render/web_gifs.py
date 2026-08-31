"""Downscale rollout GIFs to something a README can actually load.

The rollouts write 1920x480 (three 640x480 cameras side by side) at 100
frames, which lands around 13 MB each -- far too heavy for GitHub, where a
reader gives a page a second or two before scrolling past a spinner.

This keeps the three-camera layout, since seeing both wrist views next to the
top-down is the point, and buys the size back with width and frame count. The
verdict stays in the filename: these are labelled by the challenge's own
success checker and that label must survive every transformation.
"""
from __future__ import annotations

import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(args.results, "rollout_*.gif")))
    if not srcs:
        print(f"no rollout GIFs in {args.results}")
        return 1

    for src in srcs:
        name = os.path.basename(src)
        rd = imageio.get_reader(src)
        frames = []
        for i, fr in enumerate(rd):
            if i % args.every:
                continue
            im = Image.fromarray(np.asarray(fr)[:, :, :3])
            h = round(im.height * args.width / im.width)
            frames.append(np.asarray(im.resize((args.width, h), Image.LANCZOS)))
        rd.close()
        dst = os.path.join(args.out, name)
        imageio.mimsave(dst, frames, duration=1.0 / args.fps, loop=0)
        mb_in = os.path.getsize(src) / 1e6
        mb_out = os.path.getsize(dst) / 1e6
        print(f"{name}: {mb_in:.1f} MB -> {mb_out:.1f} MB "
              f"({len(frames)} frames, {args.width}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
