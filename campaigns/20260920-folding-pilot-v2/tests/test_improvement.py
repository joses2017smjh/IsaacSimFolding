"""Regression tests against actual assets and installed upstream implementations."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/render")]
from lehome_fold.folding_geometry import (authored_mesh, weld_correspondence,
    validate_indices, verify_initial_correspondence, verify_cooked_rest, fresh_geometry, mapped_positions_cm)
from rollout_audit_utils import configure_action_queue


def official_functions():
    path = ROOT / "external/lehome-challenge/source/lehome/lehome/utils/success_checker_chanllege.py"
    names = {"calculate_distance", "step_interval", "check_top_sleeve", "check_pant_long",
             "check_pant_short", "success_checker_garment_fold"}
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "get_object_particle_position": mapped_positions_cm}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def obj_for(points):
    obj = SimpleNamespace(check_points=list(range(6)), success_distance=[4, 4, 20, 20],
                          init_scale=[1, 1, 1], _folding_vertex_to_particle=np.arange(6),
                          _folding_particle_count=6)
    obj._cloth_prim_view = SimpleNamespace(get_world_positions=lambda: torch.tensor(points).unsqueeze(0))
    return obj


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((ROOT / "audit/garment_inventory.json").read_text())

    def test_exact_release_counts_identities_and_paths(self):
        rows = self.inventory["garments"]
        self.assertEqual(len(rows), 48)
        for category in ("Pant_Short", "Pant_Long", "Top_Short", "Top_Long"):
            actual = {r["garment_id"] for r in rows if r["category"] == category}
            expected = {f"{category}_Seen_{i}" for i in range(10)} | {f"{category}_Unseen_{i}" for i in range(2)}
            self.assertEqual(actual, expected)
        for r in rows:
            self.assertTrue(Path(r["mesh_path"]).is_file())
            self.assertTrue(all(Path(p).is_file() for p in r["material_paths"]))

    def test_all_scorer_landmarks_preserve_authored_rest_positions(self):
        for r in self.inventory["garments"]:
            with self.subTest(garment=r["garment_id"]):
                points, _ = authored_mesh(r["mesh_path"])
                mapping, unique = weld_correspondence(points)
                indices = r["authored_landmarks"]
                validate_indices(indices, len(points))
                validate_indices(mapping[indices].tolist(), len(unique))
                np.testing.assert_allclose(unique[mapping[indices]], points[indices], atol=1e-6, rtol=0)

    def test_reproduces_actual_top_short_3_index_error(self):
        row = next(r for r in self.inventory["garments"] if r["garment_id"] == "Top_Short_Seen_3")
        points, _ = authored_mesh(row["mesh_path"])
        mapping, particles = weld_correspondence(points)
        self.assertEqual((len(points), len(particles)), (11381, 10869))
        with self.assertRaises(IndexError):
            particles[row["authored_landmarks"]]
        self.assertEqual(particles[mapping[row["authored_landmarks"]]].shape, (6, 3))

    def test_runtime_ordering_rejects_permutation_and_nonfinite(self):
        rest = np.arange(18).reshape(6, 3).astype(float)
        self.assertLess(verify_initial_correspondence(rest + [0, 0, -.002], rest)["max_rest_correspondence_error_m"], 1e-10)
        with self.assertRaises(ValueError):
            verify_initial_correspondence(rest[::-1], rest)
        with self.assertRaises(ValueError):
            verify_initial_correspondence(rest * np.nan, rest)

    def test_native_throttle_false_is_not_terminal_geometry(self):
        official = official_functions()
        points = np.array([[0, 0, 0], [.01, 0, 0], [0, .1, 0], [.4, .1, 0], [.4, 0, 0], [.41, 0, 0]])
        obj = obj_for(points)
        self.assertIs(official.success_checker_garment_fold(obj, "short-pant"), False)
        self.assertTrue(fresh_geometry(obj, "short-pant", official)["success"])
        # A later unfurled state must not inherit an earlier success.
        points[4] = [.02, 0, 0]
        points[5] = [.03, 0, 0]
        self.assertFalse(fresh_geometry(obj, "short-pant", official)["success"])

    def test_cooked_rest_proof_rejects_deformation_wrong_count_and_permutation(self):
        unique = np.arange(18, dtype=float).reshape(6, 3)
        self.assertEqual(verify_cooked_rest(unique, unique, 6)["max_rest_correspondence_error_asset_units"], 0)
        for bad, count in [(unique[::-1], 6), (unique + .003, 6), (unique, 5)]:
            with self.assertRaises(ValueError):
                verify_cooked_rest(bad, unique, count)

    def test_official_condition_counts_and_signed_margins(self):
        official = official_functions()
        obj = obj_for(np.zeros((6, 3)))
        for garment_type, count in [("short-pant", 4), ("long-pant", 4), ("top-long-sleeve", 5), ("top-short-sleeve", 5)]:
            obj.success_distance = [10] * count
            result = fresh_geometry(obj, garment_type, official)
            self.assertEqual(result["conditions_total"], count)
            self.assertTrue(all(d["passed"] == (d["margin_cm"] >= 0) for d in result["details"].values()))

    def test_unbound_or_nonfinite_scorer_fails_closed(self):
        obj = obj_for(np.full((6, 3), np.nan))
        with self.assertRaises(ValueError):
            mapped_positions_cm(obj, obj.check_points)
        del obj._folding_vertex_to_particle
        with self.assertRaises(ValueError):
            mapped_positions_cm(obj, obj.check_points)

    def test_installed_smolvla_queue_replans_from_new_observation(self):
        # Execute the installed select_action/reset code with a deterministic
        # lightweight prediction function, rather than reimplement its queue.
        path = ROOT.parents[1].parent / "lehome51-site/lerobot/policies/smolvla/modeling_smolvla.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmolVLAPolicy")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"reset", "select_action", "_check_get_actions_condition"}]
        for node in methods:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args + node.args.kwonlyargs:
                arg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
        shell = ast.ClassDef(name="QueueHarness", bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = {"ACTION": "action", "deque": deque, "populate_queues": lambda q, batch, **kw: q}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])), str(path), "exec"), namespace)
        for horizon in (50, 10, 5):
            policy = namespace["QueueHarness"]()
            policy.config = SimpleNamespace(n_action_steps=50, chunk_size=50)
            policy.eval = lambda: None
            policy._rtc_enabled = lambda: False
            policy._prepare_batch = lambda batch: batch
            predictions = []
            def predict(batch, noise):
                predictions.append(batch["step"])
                return (torch.arange(50) + 1000 * batch["step"]).reshape(1, 50, 1)
            policy._get_action_chunk = predict
            configure_action_queue(policy, horizon)
            outputs = [policy.select_action({"step": step}).item() for step in range(60)]
            self.assertEqual(predictions, list(range(0, 60, horizon)))
            self.assertEqual(outputs, [1000 * (step // horizon * horizon) + step % horizon for step in range(60)])
            configure_action_queue(policy, 5)
            self.assertEqual(len(policy._queues["action"]), 0)
            self.assertEqual(policy.config.chunk_size, 50)

    def test_training_split_identity_and_whole_episode_separation(self):
        split = json.loads(Path(self.inventory["adaptation_split"]).read_text())
        train = {(r["garment"], r["episode"]) for r in split["train"]}
        val = {(r["garment"], r["episode"]) for r in split["validation"]}
        self.assertEqual((len(train), len(val)), (47, 8))
        self.assertFalse(train & val)
        self.assertFalse({g for g, _ in train} & {g for g, _ in val})
        self.assertFalse(any("_Unseen_" in g for g, _ in train | val))

    def test_baseline_raw_success_episodes_and_missing_not_zero(self):
        records = json.loads((ROOT / "audit/baseline_episodes.json").read_text())["cells"]
        for variant, count in [("baseline", 8), ("adapt1", 3)]:
            row = next(r for r in records if r["class"] == "Pant_Short" and r["policy_variant"] == variant)
            self.assertEqual((row["completed_recorded_episodes"], row["ever_successes"]), (24, count))
            self.assertIsNone(row["terminal_settled_successes"])
        self.assertTrue(any(r["unrecorded_episodes"] > 0 for r in records))

    def test_stale_camera_file_cannot_satisfy_false_record_success(self):
        from PIL import Image
        from pxr import Usd
        from lehome_fold.strict_observer import StrictStormObserver
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "top_rgb.png"
            Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(stale)
            observer = object.__new__(StrictStormObserver)
            observer.workdir = tmp
            observer._cams = {"top_rgb": None}
            observer._stage = Usd.Stage.CreateInMemory()
            observer._rec = SimpleNamespace(Record=lambda *args: True)
            with self.assertRaises(FileNotFoundError):
                observer.render()
            self.assertFalse(stale.exists())

    def test_mp4_is_encoded_from_rgb_with_policy_caption(self):
        from lehome_fold.policy_media import write_mp4
        with tempfile.TemporaryDirectory() as tmp:
            frames = {i: {k: np.full((16, 16, 3), 40 + i, dtype=np.uint8)
                           for k in ("left_rgb", "right_rgb", "top_rgb")} for i in (0, 6)}
            dest = Path(tmp) / "policy.mp4"
            result = write_mp4(frames, dest, 1 / 90, "POLICY failure; test fixture")
            self.assertGreater(dest.stat().st_size, 0)
            self.assertIn("POLICY failure", result["caption"])


if __name__ == "__main__":
    unittest.main()
