"""Build an outcome-labelled local gallery without treating missing jobs as failures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path


def summarize(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for task in manifest["tasks"]:
        directory = root / "outputs" / task["id"]
        row = {**task, "state": "pending", "success": None, "gifs": {}}
        status_path = directory / "status.json"
        if status_path.exists():
            row.update(json.loads(status_path.read_text()))
        elif directory.exists():
            row["state"] = "started_without_completion_record"
        result_path = directory / "rollout.json"
        if row["state"] == "completed" and result_path.exists():
            result = json.loads(result_path.read_text())
            row["success"] = bool(result["success"])
            row["terminal_success"] = result.get("terminal_success")
            row["first_success_step"] = result.get("first_success_step")
            for name, value in result.get("gif_views", {}).items():
                path = Path(value)
                if not path.is_absolute():
                    path = directory / path
                if path.is_file():
                    row["gifs"][name] = str(path.relative_to(root))
            if len(row["gifs"]) != 4:
                row["media_incomplete"] = True
        rows.append(row)
    if rows and rows[0]["state"] == "infrastructure_error":
        for row in rows[1:]:
            if row["state"] == "pending":
                row["state"] = "blocked_by_failed_gate"
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    args = p.parse_args()
    root = args.campaign.resolve()
    rows = summarize(root)
    stamp = datetime.now(timezone.utc).isoformat()
    report = {"generated_utc": stamp, "scope": "Illustrative pinned development poses; not the official benchmark",
              "tasks": rows, "expected_episodes": len(rows),
              "completed_episodes": sum(r["state"] == "completed" for r in rows),
              "successes": sum(r["state"] == "completed" and r["success"] for r in rows),
              "failures": sum(r["state"] == "completed" and not r["success"] for r in rows)}
    tmp = root / "MEDIA_STATUS.tmp.json"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(root / "MEDIA_STATUS.json")
    md = ["# Folding media status", "", f"Generated {stamp}.", "",
          "This gate runs one existing SmolVLA adaptation checkpoint. "
          "These recorded development poses are for illustrating behavior. Use the separately queued official "
          "evaluations for performance claims. A success means the unchanged checker fired at least once; "
          "terminal success is reported separately. Missing/crashed runs have no fold verdict.", "",
          "| Policy | Garment / pose | State | Checker ever / terminal | Camera GIFs |",
          "|---|---|---|---|---|"]
    cards = []
    for row in rows:
        verdict = "pending / unknown"
        if row["state"] == "completed":
            verdict = ("success" if row["success"] else "failure") + f" / {row.get('terminal_success')}"
        links = " · ".join(f"[{name}]({path})" for name, path in row["gifs"].items())
        md.append(f"| {row['policy_variant']} | {row['garment']} / {row['pose_source_episode']} | "
                  f"{row['state']} | {verdict} | {links or 'awaiting media'} |")
        images = "".join(f'<figure><figcaption>{html.escape(name)}</figcaption>'
                         f'<img loading="lazy" src="{html.escape(path)}" alt="{html.escape(name)} camera during '
                         f'{html.escape(row["garment"])} policy episode"></figure>'
                         for name, path in row["gifs"].items())
        cards.append(f'<article><h2>{html.escape(row["policy_variant"])} · {html.escape(row["garment"])} '
                     f'· pose {row["pose_source_episode"]}</h2><p>{html.escape(row["state"])}; '
                     f'checker ever / terminal: {html.escape(verdict)}</p><div class="views">{images}</div></article>')
    controls_path = root / "reference_replays/manifest.json"
    if controls_path.exists():
        md.extend(["", "## Historical demonstration replay controls", "",
                   "These are existing recorded-action replays, not either new trained policy. "
                   "Their historical animation timing is preserved; the current media jobs record timing separately.", ""])
        reference_cards = []
        for control in json.loads(controls_path.read_text()):
            label = f"{control['garment']} — replay {'success' if control['success'] else 'failure'}"
            md.append(f"- [{label}]({control['gif']}) ([scorer record]({control['result']}))")
            reference_cards.append(f'<article><h2>{html.escape(label)}</h2><p>Historical demonstration replay; '
                                   'not a new-policy result.</p>'
                                   f'<img loading="lazy" src="{html.escape(control["gif"])}" '
                                   f'alt="{html.escape(label)}"></article>')
        cards.append('<h1>Historical demonstration replay controls</h1>' + "".join(reference_cards))
    md.extend(["", "## Next session", "",
               "Read README.md, gate_submission.json and manifest.json first. "
               "Inspect only the new media jobs recorded in gate_submission.json. Do not cancel, requeue, "
               "release or modify any old jobs. Regenerate this report with "
               "`python3 scripts/media_report.py --campaign .`. "
               "If a class has no policy success, say so; do not substitute a replay under a policy caption.", ""])
    (root / "MEDIA_STATUS.md").write_text("\n".join(md))
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Garment folding — recorded camera views</title><style>'
            'body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1300px;background:#101820;color:#edf3f7}'
            'article{background:#1c2b36;padding:1rem;margin:1rem 0;border-radius:.6rem}'
            'h2{font-size:1.1rem}.views{display:flex;flex-wrap:wrap;gap:.8rem}'
            'figure{margin:0;max-width:100%}img{max-width:100%;height:auto}figcaption{padding:.4rem 0;color:#b6d9ed}'
            '</style><h1>Garment folding: two adapted policy checkpoints</h1>'
            '<p>Top, left wrist and right wrist observations from the simulator. Outcomes come from the '
            'LeHome geometric checker. Pinned development poses are illustrations; benchmark evaluations '
            'are reported separately. A missing success remains missing.</p>' + "".join(cards) + '</html>')
    (root / "gallery.html").write_text(page)
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
