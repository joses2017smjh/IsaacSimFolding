"""Preserve official geometric predicates across render-vertex/particle welding.

Garment JSON landmarks address the authored USD mesh. GPU PhysX welds UV-seam
duplicates. Mapping is validated against PhysX cooked vertex-remapping attributes;
an unverified correspondence must fail before an episode can be scored.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def authored_mesh(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f"Expected one unambiguous cloth mesh in {path}: {len(meshes)}")
    mesh = meshes[0]
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    return points, str(mesh.GetPath())


def weld_correspondence(points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid authored mesh points")
    _, first, inverse = np.unique(np.round(points, 6), axis=0,
                                  return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    mapping = rank[inverse]
    unique = points[first[order]]
    if not np.allclose(unique[mapping], points, atol=1e-6, rtol=0):
        raise ValueError("Weld would merge distinct authored positions")
    return mapping, unique


def validate_indices(indices, count):
    if len(indices) != 6 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                               or i < 0 or i >= count for i in indices):
        raise ValueError(f"Invalid six landmark indices {indices} for {count} points")


def verify_initial_correspondence(actual, expected, tolerance=1e-4):
    """Allow uniform gravity translation, never an arbitrary particle permutation."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Initial particle shape/state mismatch: {actual.shape}/{expected.shape}")
    delta = actual - expected
    translation = np.median(delta, axis=0)
    residual = float(np.linalg.norm(delta - translation, axis=1).max())
    if residual > tolerance:
        raise ValueError(f"Unverified particle ordering: rest residual {residual:.6g} m > {tolerance}")
    return {"max_rest_correspondence_error_m": residual,
            "uniform_initial_translation_m": translation.tolist(),
            "tolerance_m": tolerance}


def verify_cooked_rest(rest, unique, live_count):
    rest, unique = np.asarray(rest), np.asarray(unique)
    if rest.shape != unique.shape or rest.shape != (live_count, 3):
        raise ValueError(f"Cooked rest topology mismatch: {rest.shape}, {unique.shape}, {live_count}")
    error = float(np.linalg.norm(rest - unique, axis=1).max())
    if not np.isfinite(rest).all() or error > 1e-6:
        raise ValueError(f"Cooked rest ordering does not match authored weld: {error} asset units")
    return {"proof_source": "PhysxParticleClothAPI.restPoints in cooked simulation topology",
            "max_rest_correspondence_error_asset_units": error, "tolerance_asset_units": 1e-6}


def verify_physx_remap(authored, rest, to_weld, to_orig, live_count):
    """Validate the solver's own maps; never infer its particle ordering."""
    authored, rest = np.asarray(authored), np.asarray(rest)
    to_weld, to_orig = np.asarray(to_weld), np.asarray(to_orig)
    if rest.shape != authored.shape or not np.isfinite(rest).all():
        raise ValueError("Cooked authored rest topology mismatch")
    if not np.allclose(rest, authored, atol=1e-6, rtol=0):
        raise ValueError("Cooked rest positions differ from source mesh")
    if (to_weld.shape != (len(authored),) or to_orig.shape != (live_count,)
            or to_weld.dtype.kind not in 'iu' or to_orig.dtype.kind not in 'iu'):
        raise ValueError(f"Cooked remap shape/type mismatch: {to_weld.shape}/{to_orig.shape}/{live_count}")
    if (np.any(to_weld < 0) or np.any(to_weld >= live_count)
            or np.any(to_orig < 0) or np.any(to_orig >= len(authored))):
        raise ValueError("Cooked remap index out of bounds")
    if not np.array_equal(to_weld[to_orig], np.arange(live_count)):
        raise ValueError("Cooked forward/reverse maps are inconsistent")
    error = float(np.linalg.norm(rest[to_orig][to_weld] - authored, axis=1).max())
    if error > 1e-6:
        raise ValueError(f"Cooked weld merged distinct source positions: {error}")
    return to_weld.astype(np.int64), {
        "proof_source": "PhysX weldedVerticesRemapToWeld and weldedVerticesRemapToOrig; authored restPoints",
        "max_rest_correspondence_error_asset_units": error, "tolerance_asset_units": 1e-6}


def bind_landmarks(obj, mesh_path):
    from pxr import PhysxSchema
    authored, prim = authored_mesh(mesh_path)
    original = list(obj.check_points)
    validate_indices(original, len(authored))
    live = obj._cloth_prim_view.get_world_positions().squeeze(0).detach().cpu().numpy()
    rest_value = PhysxSchema.PhysxParticleClothAPI(obj._prim).GetRestPointsAttr().Get()
    rest = np.asarray(rest_value, dtype=np.float64) if rest_value is not None else np.empty((0, 3))
    to_weld = np.asarray(obj._prim.GetAttribute("physxParticle:weldedVerticesRemapToWeld").Get())
    to_orig = np.asarray(obj._prim.GetAttribute("physxParticle:weldedVerticesRemapToOrig").Get())
    print("[landmark] rest", rest.shape, "live", live.shape, "authored", authored.shape,
          "to_weld", to_weld.shape, "to_orig", to_orig.shape, flush=True)
    mapping, proof = verify_physx_remap(authored, rest, to_weld, to_orig, len(live))
    if live.shape != (len(to_orig), 3) or not np.isfinite(live).all():
        raise ValueError("Invalid live physics points")

    mapped = mapping[original]
    validate_indices(mapped.tolist(), len(live))
    obj._folding_vertex_to_particle = mapping
    obj._folding_particle_count = len(live)
    obj._folding_landmark_proof = {
        "mesh_path": str(mesh_path), "mesh_prim": prim,
        "mesh_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
        "authored_vertex_count": len(authored), "particle_count": len(live),
        "authored_indices": original, "particle_indices": mapped.tolist(),
        "mapping_changed": bool(np.any(mapped != original)),
        "runtime_correspondence_verified": True, **proof}
    return obj._folding_landmark_proof


def mapped_positions_cm(obj, indices):
    if not hasattr(obj, "_folding_vertex_to_particle"):
        raise ValueError("Scorer landmark correspondence not bound and verified")
    mapping = obj._folding_vertex_to_particle
    validate_indices(indices, len(mapping))
    live = obj._cloth_prim_view.get_world_positions().squeeze(0).detach().cpu().numpy()
    if live.shape != (obj._folding_particle_count, 3) or not np.isfinite(live).all():
        raise ValueError("Invalid live cloth particles for scorer")
    return (live[mapping[indices]] * 100).tolist()


def fresh_geometry(obj, garment_type, official):
    """Use the official pure predicates, bypassing only the 50-call throttle."""
    predicates = {"top-short-sleeve": official.check_top_sleeve,
                  "top-long-sleeve": official.check_top_sleeve,
                  "short-pant": official.check_pant_short,
                  "long-pant": official.check_pant_long}
    thresholds = [float(d) * float(obj.init_scale[0]) for d in obj.success_distance]
    points = mapped_positions_cm(obj, obj.check_points)
    passed, details = predicates[garment_type](points, thresholds)
    expected_count = 5 if garment_type.startswith("top-") else 4
    if len(details) != expected_count:
        raise ValueError("Official geometric predicate count changed")
    clean = {}
    for name, d in details.items():
        operator = "<=" if "<=" in d["description"] else ">="
        value, threshold = float(d["value"]), float(d["threshold"])
        if not np.isfinite([value, threshold]).all():
            raise ValueError("Nonfinite scorer distance or threshold")
        clean[name] = {"value_cm": value, "threshold_cm": threshold,
                       "operator": operator, "passed": bool(d["passed"]),
                       "margin_cm": threshold - value if operator == "<=" else value - threshold}
    return {"success": bool(passed), "garment_type": garment_type,
            "conditions_passed": sum(d["passed"] for d in clean.values()),
            "conditions_total": expected_count, "details": clean,
            "semantics": "unchanged official predicates; authored landmarks mapped to welded particles"}
