#!/usr/bin/env python3
"""Audit an EXISTING production split without decompressing capture RGB arrays.

Only the small garment/episode/success arrays are loaded with numpy. Array
shapes come from NPY headers inside the ZIP archive. File size/mtime and a
metadata digest are recorded; the digest is NOT a hash of image/action data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np


GARMENT = re.compile(r"^(Top_Short|Top_Long|Pant_Short|Pant_Long)_(Seen|Unseen)_\d+$")


def array_shape(archive: zipfile.ZipFile, key: str) -> tuple[int, ...]:
    with archive.open(f"{key}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version {version} for {key}")
    if dtype.hasobject:
        raise ValueError(f"{key} uses an unsafe object dtype")
    return shape


def read_capture(path: Path, chunk_size: int) -> dict:
    with np.load(path, allow_pickle=False) as capture:
        garment = capture["garment"].item()
        episode = capture["episode"].item()
        success = capture["success"].item()
    if isinstance(garment, bytes):
        garment = garment.decode("utf-8")
    match = GARMENT.fullmatch(garment) if isinstance(garment, str) else None
    if not match:
        raise ValueError(f"unexpected garment name {garment!r}")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError(f"episode must be a nonnegative integer, got {episode!r}")
    if not isinstance(success, bool):
        raise ValueError(f"success must be a scalar boolean, got {success!r}")
    with zipfile.ZipFile(path) as archive:
        shapes = {key: list(array_shape(archive, key))
                  for key in ("images", "state", "action")}
    images, state, action = (shapes[key] for key in ("images", "state", "action"))
    if len(images) != 5 or images[1] != 3 or images[-1] != 3 or min(images) < 1:
        raise ValueError(f"expected images (N,3,H,W,3), got {images}")
    n = images[0]
    if state != [n, 12] or action != [n, chunk_size, 12]:
        raise ValueError(f"expected state ({n},12), action ({n},{chunk_size},12); "
                         f"got {state}, {action}")
    stat = path.stat()
    metadata = {"garment": garment, "episode": episode, "success": success,
                "shapes": shapes}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return {"path": str(path.resolve()), "class": match[1], "split": match[2],
            **metadata, "frames": n, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "metadata_sha256": digest}


def build_manifest(paths: list[Path], *, reference: Path, chunk_size: int) -> dict:
    records = [read_capture(path, chunk_size) for path in sorted(paths)]
    source_bytes = reference.read_bytes()
    production = json.loads(source_bytes)
    keys = [(r["garment"], r["episode"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate garment/episode captures; refusing ambiguous identity")
    current = {key: record for key, record in zip(keys, records)}
    groups, assigned = {}, set()
    for name in ("train", "validation", "excluded_failed_replays"):
        rows = []
        for old in production[name]:
            key = (old["garment"], old["episode"])
            if key in assigned or key not in current:
                raise ValueError(f"duplicate or missing production episode {key}")
            assigned.add(key)
            row = current[key]
            if row["split"] != "Seen":
                raise ValueError(f"production split includes Unseen garment {key}")
            if row["success"] != (name != "excluded_failed_replays"):
                raise ValueError(f"outcome disagrees with production split: {key}")
            # The existing production manifest pins path, byte size and mtime.
            if (Path(old["path"]).resolve() != Path(row["path"])
                    or old["bytes"] != row["size_bytes"]
                    or old["mtime_ns"] != row["mtime_ns"]
                    or old["frames"] != row["frames"]):
                raise ValueError(f"capture changed since production manifest: {key}")
            rows.append(row)
        groups[name] = rows
    if assigned != set(current):
        raise ValueError("captures do not exactly match the existing production split")
    if not groups["train"] or not groups["validation"]:
        raise ValueError("production train and validation must both be nonempty")
    train_garments = {r["garment"] for r in groups["train"]}
    val_garments = {r["garment"] for r in groups["validation"]}
    if not train_garments.isdisjoint(val_garments):
        raise ValueError("production train/validation share garment identities")

    def counts(rows: list[dict]) -> dict:
        return {"files": len(rows), "frames": sum(r["frames"] for r in rows),
                "by_class": dict(sorted(Counter(r["class"] for r in rows).items()))}

    return {"schema_version": 1, "purpose": "diagnostic audit of EXISTING production split",
            "reference_manifest": str(reference.resolve()),
            "reference_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "production_split_sha256": production.get("sha256"),
            "chunk_size": chunk_size, "split_unit": production["split_unit"],
            "heldout_garments": production.get("heldout_garments"),
            "identity_note": "metadata_sha256 hashes scalar metadata and array shapes only; "
                             "size/mtime describe files; image/action payloads are NOT hashed",
            "inspection_note": "loaded garment/episode/success only; RGB/state/action shapes "
                               "read from NPY headers without decompressing payloads; "
                               "finite action/state values are checked by production code",
            "source_counts": counts(records),
            "counts": {name: counts(rows) for name, rows in groups.items()},
            **{name: sorted(rows, key=lambda r: r["path"]) for name, rows in groups.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference_manifest", required=True,
                    help="existing production data_split.json; no new split is generated")
    ap.add_argument("--chunk_size", type=int, default=50)
    args = ap.parse_args()
    if args.chunk_size < 1:
        ap.error("chunk_size must be positive")
    paths = sorted({Path(p).resolve() for p in glob.glob(args.capture_glob)})
    if not paths:
        ap.error(f"no capture files match {args.capture_glob!r}")
    try:
        manifest = build_manifest(paths, reference=Path(args.reference_manifest),
                                  chunk_size=args.chunk_size)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        ap.error(str(exc))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
