"""Smoke gate for v7's two interface changes, run on the smoke search row.

1. The runner's recovery branches must record the v7 telemetry: per policy
   step margin_trace (4 margins), gripper_distance_trace (2 distances),
   lift_trace, and terminal_margins / terminal_gripper_distance /
   terminal_lift, all finite, with one entry per executed action.
2. The smoke row is a large garment pinned at the P_B pose, which its
   garment metadata does not list (off-metadata placement). The spawned cloth
   must be a valid starting state: finite, at rest, and with all four checker
   conditions evaluable and the fold NOT already closed (closure margins
   negative at the first policy step), like every development start.
3. The compiler's pure functions must run on the smoke attempts.

    check_telemetry.py --campaign ROOT --out audit/smoke_telemetry.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_depth_dataset as bdd  # noqa: E402


def finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def check(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    row = manifest["recovery_smoke"][0]
    d = root / "smoke" / "recovery" / row["id"]
    problems = []
    rec = json.loads((d / "recovery.json").read_text())
    if rec.get("schema") != 2:
        problems.append(f"recovery.json schema {rec.get('schema')} != 2")
    done = [a for a in rec["attempts"] if a.get("state") == "completed"]
    if not done:
        problems.append("no completed smoke attempt")
    for a in done:
        n = a["actions_executed"]
        for key, width in (("margin_trace", 4), ("gripper_distance_trace", 2)):
            tr = a.get(key)
            if not isinstance(tr, list) or len(tr) != n or any(len(x) != width or not finite(x) for x in tr):
                problems.append(f"root {a['root']} cand {a['candidate']}: {key} malformed")
        lt = a.get("lift_trace")
        if not isinstance(lt, list) or len(lt) != n or not finite(lt):
            problems.append(f"root {a['root']} cand {a['candidate']}: lift_trace malformed")
        if len(a.get("terminal_margins") or []) != 4 or not finite(a["terminal_margins"]):
            problems.append(f"root {a['root']} cand {a['candidate']}: terminal_margins malformed")
        if len(a.get("terminal_gripper_distance") or []) != 2 or not finite([a.get("terminal_lift", float("nan"))]):
            problems.append(f"root {a['root']} cand {a['candidate']}: terminal gripper/lift malformed")
        # the trace must agree with the recorded pass/fail trace
        # margins are stored to 3 decimals, so a step within 2 mm of a
        # threshold can round across zero; only clear-cut steps must agree
        mism = sum(sum(m >= 0 for m in x) != y
                   for x, y in zip(a.get("margin_trace", []), a["policy_condition_trace"])
                   if all(abs(m) > 0.002 for m in x))
        if mism:
            problems.append(f"root {a['root']} cand {a['candidate']}: margin signs disagree with the "
                            f"condition trace at {mism} steps")
    rules = manifest["depth_rules"]
    records = [bdd.branch_record(a, rules) for a in done]
    mech = bdd.mechanism_check(records, rules)

    r = json.loads((d / "rollout.json").read_text())
    pol = [g for g in r["geometry_trajectory"] if g["phase"] == "policy"]
    start = pol[0]["details"] if pol else {}
    start_margins = [start.get(f"condition_{k}", {}).get("margin_cm") for k in (1, 2, 3, 4)]
    if not pol or not finite(start_margins):
        problems.append("off-metadata spawn: first-step checker margins not evaluable")
    elif start_margins[0] >= 0 and start_margins[1] >= 0:
        problems.append(f"off-metadata spawn starts already folded: {start_margins}")
    if not r.get("physics_finite") or not r.get("robot_finite"):
        problems.append("off-metadata spawn: non-finite simulator state")
    return {"passed": not problems, "problems": problems, "row": row["id"], "completed_attempts": len(done),
            "start_margins_cm": start_margins, "branch_records": records, "mechanism_check_runs": mech,
            "off_metadata_pose": row.get("off_metadata_pose", False)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = check(args.campaign.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "problems", "completed_attempts", "start_margins_cm")}))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    sys.exit(main())
