"""Gallery of SETTLED folding successes with provenance; failures counted alongside.

    build_gallery.py --campaign ROOT [--out ROOT/gallery.html] [--media ROOT/media]

Scans evaluation/<label>/<phase>/<row>/ for completed rows. A row enters the
gallery only if the checker's settled terminal verdict is success. "Ever"
successes -- folds the latched checker saw that later came undone -- are
counted in the header and never shown as successes. For every settled
success the top-camera GIF and the final triptych PNG are copied under
media/<label>/<phase>/<row>/ (small enough to commit); the wrist GIFs, the
MP4 and the snapshots stay where the runner wrote them and are hashed into
media/INDEX.json so every published frame traces to a job, a row, a seed and
a rollout.json digest.

The runner names media by the EVER verdict (`rollout_policy_success_*` for
any episode the latched checker ever passed), which is why the settled flag
is read from rollout.json, never inferred from a filename.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

COPY = ("top.gif", "final_triptych.png")
MEDIA_SUFFIXES = (".gif", ".mp4", ".png")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(root: Path, phases=("benchmark", "final_test")) -> list[dict]:
    records = []
    for phase in phases:
        for rollout in sorted((root / "evaluation").glob(f"*/{phase}/*/rollout.json")):
            d = rollout.parent
            status_path = d / "status.json"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            if status.get("state") != "completed":
                continue
            r = json.loads(rollout.read_text())
            request = json.loads((d / "request.json").read_text()) if (d / "request.json").is_file() else {}
            tc = r.get("terminal_checker", {})
            ever = bool(r.get("success"))
            records.append({
                "label": d.parents[1].name, "phase": phase, "row": d.name,
                "policy": request.get("policy") or status.get("policy"),
                "run": request.get("run") or status.get("run"),
                "garment": r.get("garment"), "seed": r.get("seed"),
                "horizon": r.get("effective_n_action_steps"),
                "settled": bool(r.get("terminal_success")), "ever": ever,
                "conditions": f"{tc.get('conditions_passed')}/{tc.get('conditions_total')}",
                "slurm_job_id": status.get("slurm_job_id"),
                "slurm_array_task_id": status.get("slurm_array_task_id"),
                "wall_seconds": status.get("wall_seconds"),
                "checkpoint_sha256_model": (request.get("checkpoint", {}).get("sha256", {}) or {}).get("model.safetensors"),
                "rollout_sha256": sha256(rollout),
                "tag": "success" if ever else "failure",
                "dir": str(d),
            })
    return records


def counts(records: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for rec in records:
        key = (rec["label"], rec["phase"], rec["horizon"])
        g = groups.setdefault(key, {"label": key[0], "phase": key[1], "horizon": key[2],
                                    "valid": 0, "settled": 0, "ever": 0})
        g["valid"] += 1
        g["settled"] += int(rec["settled"])
        g["ever"] += int(rec["ever"])
    return [groups[k] for k in sorted(groups, key=lambda k: (k[0], k[1], k[2] or 0))]


def index_media(rec: dict, media_dir: Path, copy: bool) -> dict:
    d = Path(rec["dir"])
    prefix = f"rollout_policy_{rec['tag']}"
    files = {}
    target_dir = media_dir / rec["label"] / rec["phase"] / rec["row"]
    for path in sorted(d.iterdir()):
        if not path.is_file() or path.suffix not in MEDIA_SUFFIXES or not path.name.startswith(prefix):
            continue
        entry = {"source": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "copied_to": None}
        if copy and rec["settled"] and any(path.name.endswith(f"_{s}") for s in COPY):
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name
            if not target.is_file() or sha256(target) != entry["sha256"]:
                shutil.copy2(path, target)
            entry["copied_to"] = str(target)
        files[path.name] = entry
    return files


def render_html(campaign: str, tallies: list[dict], shown: list[dict], html_dir: Path) -> str:
    def rel(p: str) -> str:
        return html.escape(os.path.relpath(p, html_dir))
    rows = "".join(
        f"<tr><td>{html.escape(str(t['label']))}</td><td>{html.escape(t['phase'])}</td><td>H{t['horizon']}</td>"
        f"<td>{t['settled']}/{t['valid']}</td><td>{t['ever']}/{t['valid']}</td></tr>" for t in tallies)
    articles = []
    for rec in shown:
        files = rec["files"]
        gif = next((f for n, f in files.items() if n.endswith("_top.gif") and f["copied_to"]), None)
        tri = next((f for n, f in files.items() if n.endswith("_final_triptych.png") and f["copied_to"]), None)
        caption = (f"{rec['label']} · {rec['phase']} · {rec['row']} · {rec['garment']} · seed {rec['seed']} · "
                   f"H{rec['horizon']} · terminal conditions {rec['conditions']} · settled success · "
                   f"Slurm {rec['slurm_job_id']}_{rec['slurm_array_task_id']} · rollout.json sha256 "
                   f"{(rec['rollout_sha256'] or '')[:12]}…")
        figs = ""
        if gif:
            figs += (f'<figure><figcaption>top camera</figcaption><img loading="lazy" src="{rel(gif["copied_to"])}" '
                     f'alt="top camera, {html.escape(rec["row"])}"></figure>')
        if tri:
            figs += (f'<figure><figcaption>final triptych: left wrist / top / right wrist</figcaption>'
                     f'<img loading="lazy" src="{rel(tri["copied_to"])}" alt="final triptych, {html.escape(rec["row"])}"></figure>')
        articles.append(f"<article><h2>{html.escape(rec['label'])} · {html.escape(rec['row'])}</h2>"
                        f"<p>{html.escape(caption)}</p><div class=\"views\">{figs}</div></article>")
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(campaign)} — settled folding successes</title>"
            "<style>body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1300px;background:#101820;color:#edf3f7}"
            "article{background:#1c2b36;padding:1rem;margin:1rem 0;border-radius:.6rem}h2{font-size:1.1rem}"
            ".views{display:flex;flex-wrap:wrap;gap:.8rem}figure{margin:0;max-width:100%}img{max-width:100%;height:auto}"
            "figcaption{padding:.4rem 0;color:#b6d9ed}table{border-collapse:collapse}td,th{padding:.3rem .8rem;border-bottom:1px solid #2e4656}</style>"
            f"<h1>{html.escape(campaign)}: settled folding successes</h1>"
            "<p>Every episode the campaign ran is counted in the table; only episodes whose <em>settled</em> terminal "
            "verdict from the LeHome checker is success are shown below. \"Ever\" counts folds the latched checker "
            "passed at some step, including ones that came undone before the settle check; they are not shown as "
            "successes. Each caption names the Slurm task, seed and the digest of the rollout record it comes from; "
            "media/INDEX.json hashes every file.</p>"
            f"<table><tr><th>policy · run</th><th>phase</th><th>horizon</th><th>settled</th><th>ever</th></tr>{rows}</table>"
            + "".join(articles) + "</html>")


def build(root: Path, out_html: Path | None = None, media_dir: Path | None = None,
          phases=("benchmark", "final_test"), copy: bool = True) -> dict:
    root = root.resolve()
    out_html = out_html or root / "gallery.html"
    media_dir = media_dir or root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    records = scan(root, phases)
    tallies = counts(records)
    for rec in records:
        rec["files"] = index_media(rec, media_dir, copy)
    shown = [r for r in records if r["settled"]]
    campaign = root.name
    index = {
        "campaign": campaign,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": "shown iff rollout.json terminal_success is true; ever-success counted, never shown as success",
        "counts": tallies,
        "entries": [{k: v for k, v in r.items() if k != "dir"} | {"dir": r["dir"]} for r in records],
    }
    (media_dir / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    out_html.write_text(render_html(campaign, tallies, shown, out_html.parent))
    return {"valid": len(records), "settled": len(shown), "ever": sum(r["ever"] for r in records),
            "counts": tallies, "html": str(out_html), "index": str(media_dir / "INDEX.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--no-copy", action="store_true", help="index only; copy nothing")
    args = ap.parse_args()
    summary = build(args.campaign, args.out, args.media, copy=not args.no_copy)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
