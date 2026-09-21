"""Require actual isolated four-class smoke results before the horizon pilot."""
import json
from pathlib import Path
from run_improvement_task import validate_result

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = []
    for index, row in enumerate(manifest["smoke_tasks"][:5]):
        directory = ROOT / "outputs" / row["id"]
        try:
            status = json.loads((directory / "status.json").read_text())
            if status["state"] != "completed":
                raise ValueError(status["state"])
            result = json.loads((directory / "rollout.json").read_text())
            validate_result(result, row)
            rows.append({"index": index, "passed": True, "garment": row["garment"],
                         "child_pid": status["child_pid"], "job": status["slurm_job_id"]})
        except (OSError, ValueError, KeyError) as exc:
            rows.append({"index": index, "passed": False, "garment": row["garment"], "reason": str(exc)})
    passed = all(r["passed"] for r in rows)
    result = {"passed": passed, "status": "passed" if passed else "incomplete_or_invalid",
              "rows": rows, "policy_success_is_not_required": True}
    (ROOT / "audit/smoke_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
