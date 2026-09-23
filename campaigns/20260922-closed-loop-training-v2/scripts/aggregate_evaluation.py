"""Score a candidate against the baseline under the preregistered targets.

Reads completed rollout results, reports terminal SETTLED geometric success at
each horizon, and states plainly whether the frozen targets were met. It does
not choose a threshold, a subset or a metric -- all of those are in the
manifest and were fixed before collection.

Two things it refuses to do: count an infrastructure failure as a policy
failure, and report the latched "ever" verdict as the headline. The official
checker latches on first pass and never unfires, so an episode that folds and
then unfolds reads as a success; the settled terminal verdict is what the
targets are stated against and both are printed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HORIZONS = (10, 50)


def load_rows(root: Path, label: str, phase: str) -> list[dict]:
    rows = []
    for directory in sorted((root / "evaluation" / label / phase).glob("*")):
        status_path, result_path = directory / "status.json", directory / "rollout.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        row = {"id": directory.name, "state": status["state"],
               "wall_seconds": status.get("wall_seconds")}
        if status["state"] == "completed" and result_path.is_file():
            result = json.loads(result_path.read_text())
            terminal = result.get("terminal_checker", {})
            row.update({
                "garment": result.get("garment"),
                "seed": result.get("seed"),
                "horizon": result.get("effective_n_action_steps"),
                "ever_success": bool(result.get("success")),
                "terminal_success": bool(result.get("terminal_success")),
                "conditions_passed": terminal.get("conditions_passed"),
                "conditions_total": terminal.get("conditions_total"),
                "infrastructure_valid": True,
            })
        else:
            row["infrastructure_valid"] = False
        rows.append(row)
    return rows


def tally(rows: list[dict], horizon: int | None = None) -> dict:
    valid = [r for r in rows if r["infrastructure_valid"]
             and (horizon is None or r.get("horizon") == horizon)]
    invalid = [r for r in rows if not r["infrastructure_valid"]
               and (horizon is None or True)]
    if not valid:
        return {"valid": 0, "invalid": len(invalid), "terminal_success": 0,
                "ever_success": 0, "mean_conditions": None, "rate": None}
    terminal = sum(r["terminal_success"] for r in valid)
    return {
        "valid": len(valid),
        "invalid": len(invalid),
        "terminal_success": terminal,
        "ever_success": sum(r["ever_success"] for r in valid),
        "mean_conditions": sum(r["conditions_passed"] for r in valid) / len(valid),
        "rate": terminal / len(valid),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--label", required=True, help="candidate checkpoint label")
    ap.add_argument("--baseline-label", default=None,
                    help="label of a freshly run baseline, if one exists for this phase")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    targets = manifest["targets"]

    dev = load_rows(root, args.label, "benchmark")
    test = load_rows(root, args.label, "frozen_test")
    base_test = load_rows(root, args.baseline_label, "frozen_test") if args.baseline_label else []

    dev_by_h = {h: tally(dev, h) for h in HORIZONS}
    summary = {
        "campaign": manifest["campaign"],
        "manifest_commit": manifest["git"]["commit"],
        "candidate_label": args.label,
        "development": {f"h{h}": dev_by_h[h] for h in HORIZONS},
        "development_baseline": targets["baseline_development"],
        "frozen_test": {"candidate": tally(test, 10),
                        "baseline": tally(base_test, 10) if base_test else None},
        "rows": {"development": dev, "frozen_test": test, "frozen_test_baseline": base_test},
    }

    # Preregistered targets. Baseline development numbers are the pilot's, and
    # the justification for reusing them is recorded in the manifest.
    base_h10 = int(targets["baseline_development"]["h10"].split("/")[0])
    base_h50 = int(targets["baseline_development"]["h50"].split("/")[0])
    cand_h10, cand_h50 = dev_by_h[10]["terminal_success"], dev_by_h[50]["terminal_success"]
    verdict = {
        "primary_h10_ge_4": cand_h10 >= 4,
        "stretch_h10_ge_6": cand_h10 >= 6,
        "no_h50_regression": cand_h50 >= base_h50,
        "improves_development_over_baseline": cand_h10 > base_h10,
        "baseline_h10": base_h10, "baseline_h50": base_h50,
        "candidate_h10": cand_h10, "candidate_h50": cand_h50,
    }
    training = root / "training" / "iteration1" / "training.json"
    if training.is_file():
        data = json.loads(training.read_text())
        final = data["checkpoints"][-1] if data.get("checkpoints") else {}
        verdict["raster_retention_guard"] = final.get("heldout_gate")
        verdict["heldout_loss"] = final.get("heldout_loss")
        verdict["heldout_loss_baseline"] = final.get("heldout_loss_baseline")
    verdict["second_iteration_earned"] = bool(
        verdict["improves_development_over_baseline"]
        and verdict.get("raster_retention_guard", False))
    summary["verdict"] = verdict

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["id", "garment", "seed", "horizon", "state", "infrastructure_valid",
              "ever_success", "terminal_success", "conditions_passed", "conditions_total"]
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dev + test + base_test)

    lines = [
        f"# Closed-loop evaluation — {args.label}", "",
        f"Manifest `{manifest['git']['commit'][:12]}`. Targets were frozen before collection.",
        "", "## Development set (8 poses, matched horizons)", "",
        "Development data, not an untouched test: its H10 failure is what motivated",
        "this campaign, so a gain here is not an unbiased estimate.", "",
        "| horizon | baseline settled | candidate settled | candidate mean conditions | valid |",
        "|---|---|---|---|---|",
        f"| H10 | {base_h10}/8 | **{cand_h10}/8** | "
        f"{dev_by_h[10]['mean_conditions']} | {dev_by_h[10]['valid']} |",
        f"| H50 | {base_h50}/8 | **{cand_h50}/8** | "
        f"{dev_by_h[50]['mean_conditions']} | {dev_by_h[50]['valid']} |",
        "", "## Preregistered targets", "",
        f"- primary, development H10 >= 4/8: **{'MET' if verdict['primary_h10_ge_4'] else 'NOT MET'}**",
        f"- stretch, development H10 >= 6/8: **{'MET' if verdict['stretch_h10_ge_6'] else 'NOT MET'}**",
        f"- no H50 regression below {base_h50}/8: **{'MET' if verdict['no_h50_regression'] else 'NOT MET'}**",
        f"- raster retention guard: **{verdict.get('raster_retention_guard')}**", "",
    ]
    if summary["frozen_test"]["candidate"]["valid"]:
        base = summary["frozen_test"]["baseline"]
        lines += ["## Frozen test set (excluded from training and selection)", "",
                  "| checkpoint | settled | valid |", "|---|---|---|",
                  f"| baseline | {base['terminal_success'] if base else 'not run'}/"
                  f"{base['valid'] if base else '-'} | {base['valid'] if base else '-'} |",
                  f"| {args.label} | {summary['frozen_test']['candidate']['terminal_success']}/"
                  f"{summary['frozen_test']['candidate']['valid']} | "
                  f"{summary['frozen_test']['candidate']['valid']} |", ""]
    lines += [f"Second collect-train iteration earned: "
              f"**{verdict['second_iteration_earned']}** "
              f"(requires a development improvement AND the retention guard).", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
