"""Replace every ledger entry's submission-time state with its real outcome.

Entries are appended when a job is submitted and read "submitted" forever
after. A handoff ledger has to say what actually happened: final Slurm state
per task, exit codes, elapsed time and GPU-hours, from sacct. Also records the
support jobs -- CPU probes and driver ticks -- that never had ledger entries,
so the ledger accounts for every job this campaign ran.

    reconcile_ledger.py --campaign ROOT
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sacct(job_ids: list[str]) -> dict[str, list[dict]]:
    out = subprocess.run(
        ["sacct", "-n", "-X", "-P", "-j", ",".join(job_ids),
         "--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Start,End"],
        capture_output=True, text=True).stdout
    tasks: dict[str, list[dict]] = collections.defaultdict(list)
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jid, name, state, exit_code, elapsed, tres, start, end = parts[:8]
        gpus = 0
        for item in tres.split(","):
            if item.startswith("gres/gpu="):
                gpus = int(item.split("=")[1])
        tasks[jid.split("_")[0]].append({
            "task": jid, "name": name, "state": state.split()[0], "exit": exit_code,
            "elapsed_s": int(elapsed or 0), "gpu_hours": gpus * int(elapsed or 0) / 3600.0,
            "start": start, "end": end})
    return tasks


def verdict(states: list[str]) -> str:
    active = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING"}
    if any(s in active for s in states):
        return "active"
    if states and all(s == "COMPLETED" for s in states):
        return "completed"
    return "failed" if states else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    root = ap.parse_args().campaign.resolve()
    path = root / "ledger/slurm-jobs.json"
    ledger = json.loads(path.read_text())

    ids = [j["job_id"] for j in ledger["jobs"]]
    tasks = sacct(ids)
    for job in ledger["jobs"]:
        rows = tasks.get(job["job_id"], [])
        states = [r["state"] for r in rows]
        job["state"] = verdict(states)
        job["final"] = {
            "tasks": len(rows),
            "states": dict(collections.Counter(states)),
            "nonzero_exits": sorted({r["exit"] for r in rows if r["exit"] not in ("0:0", "")}),
            "elapsed_seconds_total": sum(r["elapsed_s"] for r in rows),
            "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 4),
            "start": min((r["start"] for r in rows if r["start"] not in ("", "Unknown")), default=None),
            "end": max((r["end"] for r in rows if r["end"] not in ("", "Unknown")), default=None),
        }

    # Support jobs with no ledger entry: driver ticks (from their log names)
    # and the sbatch-from-a-job probes.
    support = set()
    for f in (root / "ledger/ticks").glob("*.out*") if (root / "ledger/ticks").is_dir() else []:
        m = re.search(r"-(\d+)\.out", f.name)
        if m:
            support.add(m.group(1))
    state = json.loads((root / "ledger/driver_state.json").read_text())
    for key in ("chain", "watchdog"):
        if state["ticks"].get(key):
            support.add(state["ticks"][key])
    probe_dir = root / "runtime/probe"
    for f in probe_dir.glob("*-*.out") if probe_dir.is_dir() else []:
        m = re.search(r"-(\d+)\.out", f.name)
        if m:
            support.add(m.group(1))
    support -= set(ids)
    support_tasks = sacct(sorted(support)) if support else {}
    ledger["support_jobs"] = [
        {"job_id": jid, "name": (rows[0]["name"] if rows else None),
         "state": verdict([r["state"] for r in rows]),
         "gpu_hours": round(sum(r["gpu_hours"] for r in rows), 4)}
        for jid, rows in sorted(support_tasks.items())]

    # Jobs submitted before the driver existed carry no gpu_tasks field; these
    # are their true task counts (the same mapping the driver budgets with).
    legacy_tasks = {"smoke": 1, "collect:iteration1": 8, "train:iteration1": 1}
    legacy_group = {"smoke": "smoke", "collect:iteration1": "iter1",
                    "compile:iteration1": "iter1", "train:iteration1": "iter1"}
    by_group = collections.defaultdict(lambda: {"jobs": 0, "gpu_tasks": 0, "gpu_hours": 0.0})
    for job in ledger["jobs"]:
        phase = job["phase"]
        group = legacy_group.get(phase) or phase.split(".")[0].split(":")[0]
        tasks = int(job["gpu_tasks"]) if "gpu_tasks" in job else legacy_tasks.get(phase, 0)
        job.setdefault("gpu_tasks", tasks)
        by_group[group]["jobs"] += 1
        by_group[group]["gpu_tasks"] += tasks
        by_group[group]["gpu_hours"] = round(by_group[group]["gpu_hours"] + job["final"]["gpu_hours"], 4)
    by_stage = by_group
    ledger["reconciled"] = {
        "by": "scripts/reconcile_ledger.py from sacct",
        "stage_jobs": len(ledger["jobs"]),
        "support_jobs": len(ledger["support_jobs"]),
        "gpu_hours_total": round(sum(j["final"]["gpu_hours"] for j in ledger["jobs"]), 3),
        "outcomes": dict(collections.Counter(j["state"] for j in ledger["jobs"])),
        "by_iteration_or_stage": dict(by_stage),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n")
    os.replace(tmp, path)
    print(json.dumps(ledger["reconciled"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
