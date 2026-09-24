"""Post-closure, outside the frozen campaign: make the gallery committable.

Replaces each full-size top-camera GIF copy under media/ with a
width-limited version (default 320 px) and records the resized file's
digest next to the original's in media/INDEX.json. The full-size originals
stay where the runner wrote them under evaluation/ and keep their hashes;
the final triptych PNGs are left untouched. Idempotent: an entry that
already carries `web_copy` is skipped.

This script is NOT an executed campaign source (it is not in the manifest
and the driver never calls it); it ran once after the campaign closed. Do
not re-run build_gallery.py afterwards without --no-copy: it would restore
the full-size copies.

    web_media.py --campaign ROOT [--width 320]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageSequence


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resize_gif(src: Path, dst: Path, width: int) -> int:
    im = Image.open(src)
    frames, durations = [], []
    for frame in ImageSequence.Iterator(im):
        durations.append(int(frame.info.get("duration", im.info.get("duration", 100))))
        rgb = frame.convert("RGB")
        height = max(1, round(rgb.height * width / rgb.width))
        frames.append(rgb.resize((width, height), Image.LANCZOS))
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()
    index_path = args.campaign.resolve() / "media/INDEX.json"
    index = json.loads(index_path.read_text())
    before = after = 0
    done = skipped = 0
    for entry in index["entries"]:
        for name, f in entry["files"].items():
            if not f.get("copied_to") or not name.endswith("_top.gif"):
                continue
            path = Path(f["copied_to"])
            if f.get("web_copy") or not path.is_file():
                skipped += 1
                continue
            before += path.stat().st_size
            tmp = path.with_name(path.stem + ".web.gif")
            frames = resize_gif(path, tmp, args.width)
            tmp.replace(path)
            after += path.stat().st_size
            f["web_copy"] = {"width": args.width, "frames": frames, "bytes": path.stat().st_size,
                             "sha256": sha256(path),
                             "note": ("the committed copy is resized; 'sha256' and 'bytes' above "
                                      "describe the full-size original under evaluation/")}
            done += 1
    index["web_copies"] = {"script": "scripts/web_media.py", "width": args.width, "resized": done}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"resized": done, "skipped": skipped, "bytes_before": before, "bytes_after": after}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
