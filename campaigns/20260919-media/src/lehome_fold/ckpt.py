"""Gate G3: the trainer and the rollout workers must agree on which policy ran.

The failure this exists to prevent has a specific shape. In an async pipeline
the trainer keeps updating while N workers keep rolling out. If a worker is
still executing checkpoint v7 while the trainer has reached v12, the advantages
computed for those rollouts are labelled against the wrong baseline. Nothing
crashes. The buffer quietly fills with mislabelled data and the run degrades
over hours instead of failing in seconds.

So every rollout carries the version that produced it, and the trainer refuses
data that is too far behind rather than trusting the directory listing.

Concurrency assumptions, since this runs across Slurm jobs on Lustre:
  - one writer (the trainer), many readers (the workers)
  - manifest writes are atomic via write-temp-then-rename, so a reader never
    sees a half-written file; POSIX rename within a directory is atomic and
    Lustre honours it
  - readers never write the manifest, so no locking is needed
  - a reader that catches the file mid-replace retries rather than failing,
    because NFS/Lustre attribute caching can briefly surface a stale handle
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST = "checkpoint_manifest.json"


@dataclass(frozen=True)
class CheckpointRef:
    version: int          # monotonic, incremented once per published checkpoint
    step: int             # optimiser step it was written at
    path: str             # absolute path to the checkpoint directory
    digest: str           # sha256 over the weight file(s), truncated
    written_at: float     # unix time

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def digest_path(path: Path, *, chunk: int = 1 << 20, max_bytes: int = 256 << 20) -> str:
    """sha256 over a checkpoint's weight files.

    Truncated at max_bytes per file: this is an identity check between two
    processes reading the same shared filesystem, not tamper evidence, and
    hashing several GB on every publish would cost more than it protects.
    Files are hashed in sorted order so the digest is stable.
    """
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*")
                   if p.is_file() and p.suffix in {".safetensors", ".bin", ".pt"})
    if not files:
        raise FileNotFoundError(f"no weight files under {path}")
    for f in files:
        h.update(f.relative_to(path).as_posix().encode())
        read = 0
        with open(f, "rb") as fh:
            while read < max_bytes:
                b = fh.read(min(chunk, max_bytes - read))
                if not b:
                    break
                h.update(b)
                read += len(b)
    return h.hexdigest()[:16]


def publish(shared_dir: Path, ckpt_dir: Path, *, version: int, step: int) -> CheckpointRef:
    """Trainer side: make a new checkpoint visible to the workers, atomically."""
    shared_dir, ckpt_dir = Path(shared_dir), Path(ckpt_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    ref = CheckpointRef(
        version=version, step=step, path=str(ckpt_dir.resolve()),
        digest=digest_path(ckpt_dir), written_at=time.time(),
    )
    final = shared_dir / MANIFEST
    tmp = shared_dir / f".{MANIFEST}.{os.getpid()}.tmp"
    tmp.write_text(ref.to_json())
    # fsync before rename: without it a crash can leave the manifest pointing
    # at content that never reached disk, and the workers would then load a
    # checkpoint the trainer does not think it published.
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return ref


def read(shared_dir: Path, *, retries: int = 5, delay: float = 0.2) -> CheckpointRef:
    """Worker side: what should I be running?"""
    p = Path(shared_dir) / MANIFEST
    last: Exception | None = None
    for _ in range(retries):
        try:
            return CheckpointRef(**json.loads(p.read_text()))
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            last = e
            time.sleep(delay)
    raise RuntimeError(f"could not read {p} after {retries} attempts: {last}")


class StaleCheckpoint(RuntimeError):
    """A rollout was produced by a policy too far behind the trainer."""


def check_lag(rollout_version: int, current_version: int, *, max_lag: int = 2) -> int:
    """Trainer side: how far behind was the policy that produced this rollout?

    `max_lag` is a policy decision, not a constant of nature. Zero would be
    correct and would also stall every worker on every publish; 2 lets a worker
    finish the episode it is in the middle of. What must never happen is an
    UNBOUNDED lag, which is the silent-degradation case.
    """
    if rollout_version > current_version:
        raise StaleCheckpoint(
            f"rollout claims version {rollout_version} but the trainer is at "
            f"{current_version} -- versions are not monotonic, which means two "
            f"trainers are writing the same manifest"
        )
    lag = current_version - rollout_version
    if lag > max_lag:
        raise StaleCheckpoint(
            f"rollout is {lag} versions behind (max_lag={max_lag}): produced by "
            f"v{rollout_version}, trainer at v{current_version}. Its advantage "
            f"labels are computed against the wrong baseline."
        )
    return lag


def stamp(record: dict, ref: CheckpointRef) -> dict:
    """Worker side: attach provenance to a rollout before writing it out."""
    out = dict(record)
    out["_ckpt_version"] = ref.version
    out["_ckpt_digest"] = ref.digest
    out["_ckpt_step"] = ref.step
    return out


def verify_stamp(record: dict, ref: CheckpointRef, *, max_lag: int = 2) -> int:
    """Trainer side: full check on one rollout. Returns the lag."""
    for k in ("_ckpt_version", "_ckpt_digest"):
        if k not in record:
            raise StaleCheckpoint(
                f"rollout carries no {k}: it was written by a worker that "
                f"predates checkpoint stamping, so its provenance is unknown"
            )
    lag = check_lag(int(record["_ckpt_version"]), ref.version, max_lag=max_lag)
    if lag == 0 and record["_ckpt_digest"] != ref.digest:
        raise StaleCheckpoint(
            f"rollout claims current version {ref.version} but digest "
            f"{record['_ckpt_digest']} != {ref.digest} -- the checkpoint "
            f"directory was overwritten in place instead of versioned"
        )
    return lag
