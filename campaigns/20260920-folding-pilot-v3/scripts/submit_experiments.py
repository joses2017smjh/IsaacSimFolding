"""Submit validated new work only; preserve all existing Slurm jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def submit(kind, command):
    receipt = ROOT / "submissions.jsonl"
    old = [json.loads(line) for line in receipt.read_text().splitlines()] if receipt.exists() else []
    existing = next((r for r in old if r["kind"] == kind), None)
    if existing:
        print(json.dumps(existing))
        return existing["job_id"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}")
    row = {"kind": kind, "job_id": job, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "command": command, "output_root": str(ROOT / "outputs"), "stdout": result.stdout.strip()}
    with receipt.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "pilot"])
    args = parser.parse_args()
    static = json.loads((ROOT / "audit/phase_a.json").read_text())
    if not static["phase_a_complete"] or not static["required_tests_passed"]:
        raise ValueError("Static audit and CPU checks have not passed")
    script = str(ROOT / "slurm/improvement.sbatch")
    logs = str(ROOT / "logs")
    if args.phase == "smoke":
        job = submit("smoke_array", ["sbatch", "--parsable", "--array=0-4%1",
            f"--output={logs}/smoke-%A_%a.out", f"--error={logs}/smoke-%A_%a.err",
            script, str(ROOT), "smoke_tasks"])
        submit("switch_diagnostic", ["sbatch", "--parsable", "--job-name=lh-switch-diag",
            f"--dependency=afterany:{job}", f"--output={logs}/switch-%j.out", f"--error={logs}/switch-%j.err",
            script, str(ROOT), "smoke_tasks", "5"])
    else:
        gate = json.loads((ROOT / "audit/smoke_gate.json").read_text())
        if gate.get("passed") is not True:
            raise ValueError("Do not submit pilot before all five experiment-level smokes pass")
        submit("horizon_pilot", ["sbatch", "--parsable", "--job-name=lh-horizon-dev", "--array=0-23%1",
            f"--output={logs}/pilot-%A_%a.out", f"--error={logs}/pilot-%A_%a.err",
            script, str(ROOT), "pilot_tasks"])


if __name__ == "__main__":
    main()
