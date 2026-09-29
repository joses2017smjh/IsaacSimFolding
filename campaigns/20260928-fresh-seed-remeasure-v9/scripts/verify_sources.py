"""Preflight: does the code about to run match the code the manifest froze?

v5: both pinned checkpoints (baseline and candidate) are verified.

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


def tree_digest(base: Path, pattern: str) -> tuple[str, int]:
    """SHA-256 over the sorted (relative path, file SHA-256) list of base.rglob(pattern)."""
    h = hashlib.sha256()
    files = sorted(p for p in base.rglob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        h.update(f"{path.relative_to(base)}\0{file_sha(path)}\n".encode())
    return h.hexdigest(), len(files)


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
        # v5 pins TWO immutable checkpoints; both are checked file by file.
        for which in ("baseline_checkpoint", "candidate_checkpoint"):
            block = manifest.get(which)
            if not block:
                continue
            ckpt = Path(block["path"])
            for name, expected in block["sha256"].items():
                path = ckpt / name
                if not path.is_file():
                    problems.append(f"{which} missing {name}")
                elif file_sha(path) != expected:
                    problems.append(f"{which} {name} changed -- it is declared immutable")
        pose = Path(manifest["pose_metadata"]["path"])
        if not pose.is_file():
            problems.append("pose metadata is unavailable")
        elif file_sha(pose) != manifest["pose_metadata"]["sha256"]:
            problems.append("pose metadata changed; the frozen rows were derived from it")

    # v8: the campaign's one changed variable, the anchor, is verified at train
    # time too -- the trainer's glob must select exactly the pinned files, byte
    # for byte. Not under --skip-checkpoint: it guards the experiment itself.
    anchor = manifest.get("training", {}).get("anchor", {})
    if anchor.get("records"):
        import glob as _glob
        selected = [str(Path(p).resolve()) for p in sorted(_glob.glob(anchor["glob"]))[:int(anchor["files"])]]
        pinned = sorted(r["target"] for r in anchor["records"])
        if selected != pinned:
            problems.append(f"anchor glob selects {len(selected)} files that differ from the pinned records")
        for r in anchor["records"]:
            if not Path(r["target"]).is_file() or file_sha(Path(r["target"])) != r["sha256"]:
                problems.append(f"anchor file changed: {r['target']}")

    # v9: installed packages on the executed path that are not repository
    # sources (lerobot in lehome51-site) are pinned by a tree digest.
    for name, rec in (manifest.get("environment_digests") or {}).items():
        actual, count = tree_digest(Path(rec["path"]), rec["glob"])
        if actual != rec["sha256"] or count != rec["files"]:
            problems.append(f"environment {name} changed: {count} files, digest {actual[:12]} "
                            f"(manifest {rec['files']} files, {rec['sha256'][:12]})")

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
