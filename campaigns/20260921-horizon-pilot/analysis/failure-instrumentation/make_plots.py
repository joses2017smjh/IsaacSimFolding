"""Create compact, offline matched-pose diagnostics from pilot telemetry."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/failure-instrumentation"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def load(row):
    directory = ROOT / "outputs" / row["id"]
    result = json.loads((directory / "rollout.json").read_text())
    records = [json.loads(line) for line in Path(result["behavior_telemetry"]["path"]).read_text().splitlines()]
    return directory, result, records


def first(records, predicate):
    return next((int(r["action"]) for r in records if predicate(r)), None)


def polyline(points, x0, y0, width, height, xmin, xmax, ymin, ymax, color, dash=""):
    def xy(x, y):
        px = x0 + (x - xmin) / (xmax - xmin) * width
        py = y0 + height - (y - ymin) / (ymax - ymin) * height
        return f"{px:.1f},{py:.1f}"
    attrs = f'fill="none" stroke="{color}" stroke-width="1.3"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    return f'<polyline points="{" ".join(xy(x,y) for x,y in points)}" {attrs}/>'


def text(x, y, value, size=12, anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" fill="#222">{value}</text>'


def panel(svg, x, y, w, h, title, traces, ymin, ymax, ylabel, thresholds=()):
    left, top, width, height = x + 58, y + 28, w - 72, h - 54
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="#bbb"/>')
    svg.append(text(x + 6, y + 17, title, 12))
    svg.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#555"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#555"/>')
    for val, label in thresholds:
        py = top + height - (val-ymin)/(ymax-ymin)*height
        svg.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+width}" y2="{py:.1f}" stroke="#888" stroke-dasharray="4,3"/>')
        svg.append(text(left + width + 3, py + 4, label, 9))
    for trace in traces:
        svg.append(polyline(trace["points"], left, top, width, height, 0, 600/90, ymin, ymax, trace["color"], trace.get("dash", "")))
        for boundary in trace["boundaries"]:
            px = left + boundary / 90 / (600/90) * width
            svg.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+height}" stroke="{trace["color"]}" stroke-opacity=".12"/>')
        if trace.get("divergence") is not None:
            px = left + trace["divergence"] / 90 / (600/90) * width
            svg.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+height}" stroke="{trace["color"]}" stroke-dasharray="2,2"/>')
    svg.append(text(x + 7, y + h/2, ylabel, 10))


def main():
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = {(int(r["development_pose_slot"]), int(r["horizon"])): r for r in MANIFEST["pilot_tasks"]}
    for pose in (1, 3, 5, 7):
        series = {}
        for horizon in (50, 10, 5):
            directory, result, records = load(rows[(pose, horizon)])
            divergence = None
            if horizon != 50:
                _, _, base = load(rows[(pose, 50)])
                for a, b in zip(records, base):
                    diff = sum((float(x) - float(y)) ** 2 for x, y in zip(a["executed_action_rad"], b["executed_action_rad"])) ** .5
                    if diff > .25:
                        divergence = int(a["action"])
                        break
            series[horizon] = (directory, result, records, divergence)

        colors = {50: "#1769aa", 10: "#d95f02", 5: "#7570b3"}
        traces = {"distance": [], "conditions": [], "command_left": [], "command_right": [], "lift": []}
        for horizon in (50, 10, 5):
            _, result, records, divergence = series[horizon]
            t = [float(r["action"]) / 90.0 for r in records]
            d = [min(float(x) for x in r["gripper_link_origin_distance_m"].values()) * 100.0 for r in records]
            c = [int(r["conditions_passed"]) for r in records]
            cmd_l = [float(r["gripper_target_rad"]["left"]) for r in records]
            cmd_r = [float(r["gripper_target_rad"]["right"]) for r in records]
            lift = [float(r["maximum_particle_lift_m"]) * 100.0 for r in records]
            label = f"H{horizon}"
            boundaries = [int(r["action"]) for r in records if r.get("replan_boundary") and int(r["action"]) > 1]
            common = {"color": colors[horizon], "boundaries": boundaries, "divergence": divergence}
            traces["distance"].append({**common, "points": list(zip(t, d))})
            traces["conditions"].append({**common, "points": list(zip(t, c))})
            traces["command_left"].append({**common, "points": list(zip(t, cmd_l))})
            traces["command_right"].append({**common, "points": list(zip(t, cmd_r)), "dash": "4,3"})
            traces["lift"].append({**common, "points": list(zip(t, lift))})

        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">', '<rect width="1200" height="720" fill="#f7f7f7"/>']
        svg.append(text(600, 24, f"Matched pose {pose}: H50/H10/H5; vertical lines=replans, dotted=first action divergence >0.25 rad", 15, "middle"))
        panel(svg, 20, 40, 570, 300, "distance; circles 5 cm, x 3 cm", traces["distance"], 0, 15, "cm", ((5, "5"), (3, "3")))
        panel(svg, 610, 40, 570, 300, "geometric conditions passed", traces["conditions"], 0, 4, "count")
        panel(svg, 20, 370, 570, 300, "raw gripper target (solid=left, dashed=right)", traces["command_left"] + traces["command_right"], -2, 2, "rad")
        panel(svg, 610, 370, 570, 300, "maximum particle lift displacement proxy", traces["lift"], 0, 40, "cm")
        svg.append(text(1150, 695, "H50 blue   H10 orange   H5 purple", 11, "end"))
        svg.append('</svg>')
        (plots / f"pose{pose}_comparison.svg").write_text("\n".join(svg))
    print(json.dumps({"plots": 4, "directory": str(plots), "format": "svg"}))


if __name__ == "__main__":
    main()
