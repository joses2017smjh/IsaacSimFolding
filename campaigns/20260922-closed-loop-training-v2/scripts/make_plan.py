"""Write plans/iterationK.json from a compact spec, refusing unsafe rows.

A plan is where an iteration's single changed factor is decided, so it is
also where train/evaluation separation is easiest to break by accident. This
builder derives every row's initial pose from the same demonstration metadata
the manifest used, and fails closed if a row:

  - uses a frozen-test garment (the test set must stay unseen by training);
  - reproduces a development row's (garment, pose key), unless the spec says
    so explicitly and gives a reason that ends up in the plan;
  - reuses any seed from a preregistered block, or repeats a seed.

    make_plan.py --campaign ROOT --spec spec.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_rows(spec_rows: list[dict], demos: dict, inventory: dict, *, prefix: str) -> list[dict]:
    rows = []
    for r in spec_rows:
        garment, key = r["garment"], int(r["pose_key"])
        demo = demos[garment][str(key)]
        if len(set(demo["scale"])) != 1:
            raise SystemExit(f"{garment} key {key} has anisotropic scale")
        horizon = int(r.get("horizon", 10))
        rows.append({
            "id": f"{prefix}_{garment.lower()}_key{key}_s{int(r['seed'])}_h{horizon}",
            "garment": garment,
            "asset_config": inventory[garment]["config"],
            "pose_key": key,
            "match_pose": ":".join(format(v, ".10g") for v in demo["object_initial_pose"]),
            "match_scale": demo["scale"][0],
            "seed": int(r["seed"]),
            "steps": int(r.get("steps", 600)),
            "horizon": horizon,
            "trajectory_every": int(r.get("trajectory_every", 5)),
        })
    return rows


def separation_problems(rows: list[dict], manifest: dict, *, allow_dev_overlap: str | None) -> list[str]:
    problems = []
    test_garments = {r["garment"] for r in manifest["frozen_test"]}
    dev_pairs = {(r["garment"], int(r["pose_key"])) for r in manifest["benchmark"]}
    reserved = {int(r["seed"]) for key in ("benchmark", "frozen_test", "collection",
                                           "collection_expansion", "smoke")
                for r in manifest[key]}
    seen = set()
    for r in rows:
        if r["garment"] in test_garments:
            problems.append(f"{r['id']}: {r['garment']} is a frozen-test garment")
        if (r["garment"], r["pose_key"]) in dev_pairs and not allow_dev_overlap:
            problems.append(f"{r['id']}: reproduces development (garment, pose key) "
                            f"({r['garment']}, {r['pose_key']})")
        if r["seed"] in reserved:
            problems.append(f"{r['id']}: seed {r['seed']} belongs to a preregistered block")
        if r["seed"] in seen:
            problems.append(f"{r['id']}: seed {r['seed']} repeated within the plan")
        seen.add(r["seed"])
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    spec = json.loads(args.spec.read_text())
    k = int(spec["iteration"])
    out = root / "plans" / f"iteration{k}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; plans are frozen once written")

    pilot = root.parent / "20260921-horizon-pilot"
    inventory = {g["garment_id"]: g for g in
                 json.loads((pilot / "audit/garment_inventory.json").read_text())["garments"]}
    demos = json.loads(Path(manifest["pose_metadata"]["path"]).read_text())

    collection = build_rows(spec["collection"], demos, inventory, prefix=f"it{k}")
    expansion = build_rows(spec.get("collection_expansion", []), demos, inventory, prefix=f"it{k}x")
    problems = separation_problems(collection + expansion, manifest,
                                   allow_dev_overlap=spec.get("allow_dev_overlap"))
    if problems:
        raise SystemExit("unsafe plan:\n  " + "\n  ".join(problems))
    horizons = sorted({r["horizon"] for r in collection + expansion})

    plan = {
        "iteration": k,
        "factor_changed": spec["factor_changed"],
        "justification": spec["justification"],
        "evidence": spec.get("evidence"),
        "amendment": spec.get("amendment"),
        "collection_checkpoint": spec["collection_checkpoint"],
        "init_checkpoint": spec["init_checkpoint"],
        "collection_horizons": horizons,
        "collection": collection,
        "collection_expansion": expansion,
        "training": spec.get("training", {}),
        "allow_dev_overlap": spec.get("allow_dev_overlap"),
        "separation": {
            "frozen_test_garments_excluded": True,
            "development_pairs_excluded": not spec.get("allow_dev_overlap"),
            "preregistered_seed_blocks_excluded": True,
        },
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(out), "collection": len(collection), "expansion": len(expansion),
                      "horizons": horizons, "factor": plan["factor_changed"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
