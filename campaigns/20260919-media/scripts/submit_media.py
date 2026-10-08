"""Submit only this additive media graph; never modify an existing Slurm job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def clean_scheduler_env():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if len(manifest["tasks"]) != 16:
        raise RuntimeError("reviewed campaign must contain exactly 16 media episodes")
    frozen = json.loads((root / "source_hashes.json").read_text())
    for relative, expected in frozen.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"reviewed source changed: {relative}")
    parents = manifest["existing_jobs_do_not_modify"]
    if parents != ["21359573", "21359574", "21359575"]:
        raise RuntimeError("existing dependency identities differ from reviewed plan")
    receipts_path = root / "submissions.jsonl"
    receipts = {}
    if receipts_path.exists():
        for line in receipts_path.read_text().splitlines():
            row = json.loads(line)
            receipts[row["role"]] = row
    lock_path = root / ".submission.lock"
    lock_fd = None
    if args.submit:
        # A failed or interrupted submission retains this lock for review;
        # do not blindly retry when scheduler acceptance could be ambiguous.
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.write(lock_fd, str(os.getpid()).encode())
        (root / "logs").mkdir(exist_ok=True)
    env = clean_scheduler_env()

    def submit(role, script, dependency, extra=(), script_args=()):
        if role in receipts:
            return receipts[role]["job_id"]
        cmd = ["sbatch", "--parsable", "--export=NONE", f"--chdir={root}",
               f"--output={root}/logs/%x-%A_%a.out", f"--error={root}/logs/%x-%A_%a.out",
               f"--dependency={dependency}", *extra, str(root / "slurm" / script),
               str(root), *script_args]
        if not args.submit:
            print(json.dumps({"role": role, "command": cmd}))
            return "NEW_" + role.upper()
        response = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        job_id = response.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"ambiguous scheduler response; inspect before retry: {response.stdout!r}")
        row = {"role": role, "job_id": job_id, "submitted_utc": datetime.now(timezone.utc).isoformat(),
               "dependency": dependency, "command": cmd,
               "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
               "stderr": response.stderr.strip()}
        with receipts_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipts[role] = row
        print(json.dumps(row), flush=True)
        return job_id

    gate = submit("gate", "media.sbatch", "afterany:" + ":".join(parents),
                  extra=("--job-name=lh-media-gate",), script_args=("0",))
    rest = submit("episodes", "media.sbatch", "afterok:" + gate,
                  extra=("--array=1-15%1", "--job-name=lh-media"))
    # If the gate crashes, later tasks can remain DependencyNeverSatisfied;
    # the immediate report below still shows that truth. A subsequent report
    # runs once the full media array reaches a terminal state.
    submit("report", "gallery.sbatch", "afterany:" + rest)
    if args.submit:
        subprocess.run(["python3", str(root / "scripts/media_report.py"), "--campaign", str(root)], check=True)
        status = ["# Folding session status — 19 September 2026", "",
                  "The correct project is IsaacSimFolding / lehome-fold-repro. Newer training and "
                  "strict evaluations live in the sibling bhl-robustness-ladder weekend campaign.", "",
                  "Two adapted SmolVLA checkpoints are complete (training seeds 0 and 1). "
                  "Their existing baseline/adaptation evaluations 21359573, 21359574, 21359575 were preserved. "
                  "Historical 6/24 is invalid as a visual-policy baseline because observations became stale.", "",
                  "New media submissions:", ""]
        status.extend(f"- {row['role']}: `{row['job_id']}`; dependency `{row['dependency']}`."
                      for row in receipts.values())
        status.extend(["", "Sixteen new pinned-pose episodes cover two checkpoints, four classes, two poses each. "
                       "Each saves top, left wrist, right wrist and combined GIFs with actual checker outcomes. "
                       "Maximum one of these GPU jobs at a time, thirty minutes each (eight GPU-hours maximum). "
                       "The first episode is a gate. Existing jobs are never changed.", "",
                       "Read [the full audit](campaigns/20260919-media/docs/AUDIT_2026-09-19.md), "
                       "[media status](campaigns/20260919-media/MEDIA_STATUS.md), and "
                       "[submission receipts](campaigns/20260919-media/submissions.jsonl) before next-session work. "
                       "The [gallery](campaigns/20260919-media/gallery.html) includes eight historical demonstration "
                       "replay controls, explicitly labelled; new policy media is pending. "
                       "Do not claim a success for a class unless its own policy episode passed.", "",
                       "Legacy candidate selection has an action-independent score; RECAP's gradient-update path "
                       "is unimplemented. Do not relaunch either as a functioning new method. Next research step "
                       "after results is more valid long-pants data and corrective rollout trajectories.", ""])
        # This is a NEW root note, created only after submission, not an edit to
        # any original source or document used by queued jobs.
        note = root.parent.parent / "SESSION_STATUS_2026-09-19.md"
        with note.open("x") as stream:
            stream.write("\n".join(status))
        with (root / "docs/AUDIT_2026-09-19.md").open("a") as stream:
            stream.write("\n## Final submission receipt\n\n")
            stream.write("All requested submissions succeeded; original jobs were untouched.\n\n")
            for row in receipts.values():
                stream.write(f"- `{row['role']}`: `{row['job_id']}`, `{row['dependency']}`.\n")
            stream.write("\nRead `../submissions.jsonl` and `../MEDIA_STATUS.md` for precise paths/status. "
                         "New outputs are pending; this session ends after queueing.\n")
        os.close(lock_fd)
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
