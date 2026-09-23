"""Preflight: does the code about to run match the code the manifest froze?

Run before every production submission and inside the clean-checkout proof.
Checks the executed sources against the manifest's committed-blob hashes, the
baseline checkpoint against its pinned hashes, and the asset/pose metadata the
rows were derived from. Fails closed and names every mismatch -- the horizon
pilot's manifest drifted silently precisely because nothing ran this check
between campaigns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-checkpoint", action="store_true",
                    help="clean-checkout proof: sources only, no 1.2 GB weights needed")
    args = ap.parse_args()
    root = args.campaign.resolve()
    repo = (args.repo or root.parents[1]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())

    problems, checked = [], 0
    for rel, expected in manifest["executed_sources"].items():
        path = repo / rel
        if not path.is_file():
            problems.append(f"missing executed source: {rel}")
            continue
        actual = file_sha(path)
        checked += 1
        if actual != expected:
            problems.append(f"source changed: {rel}\n    manifest {expected}\n    on disk  {actual}")

    if not args.skip_checkpoint:
        ckpt = Path(manifest["baseline_checkpoint"]["path"])
        for name, expected in manifest["baseline_checkpoint"]["sha256"].items():
            path = ckpt / name
            if not path.is_file():
                problems.append(f"baseline checkpoint missing {name}")
            elif file_sha(path) != expected:
                problems.append(f"baseline checkpoint {name} changed -- it is declared immutable")
        pose = Path(manifest["pose_metadata"]["path"])
        if not pose.is_file():
            problems.append("pose metadata is unavailable")
        elif file_sha(pose) != manifest["pose_metadata"]["sha256"]:
            problems.append("pose metadata changed; the frozen rows were derived from it")

    report = {
        "campaign": str(root),
        "commit": manifest["git"]["commit"],
        "sources_declared": len(manifest["executed_sources"]),
        "sources_checked": checked,
        "checkpoint_checked": not args.skip_checkpoint,
        "passed": not problems,
        "problems": problems,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not problems else 5


if __name__ == "__main__":
    sys.exit(main())
