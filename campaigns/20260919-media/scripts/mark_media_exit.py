"""Persist even an Apptainer startup failure, then refresh the campaign report."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--campaign", type=Path, required=True)
p.add_argument("--index", type=int, required=True)
p.add_argument("--returncode", type=int, required=True)
args = p.parse_args()
root = args.campaign.resolve()
row = json.loads((root / "manifest.json").read_text())["tasks"][args.index]
dest = root / "outputs" / row["id"]
dest.mkdir(parents=True, exist_ok=True)
status = dest / "status.json"
if not status.exists():
    status.write_text(json.dumps({"task": row["id"], "state": "infrastructure_error",
                                 "returncode": args.returncode,
                                 "error": "job ended before producing a completed episode status; inspect scheduler log"},
                                indent=2) + "\n")
subprocess.run(["python3", str(root / "scripts/media_report.py"), "--campaign", str(root)], check=True)
