"""Failure handling and provenance checks for queued camera-media work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner, report = module("run_media_task"), module("media_report")


class MediaChecks(unittest.TestCase):
    def test_gate_requires_all_views_and_exact_episode_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            views = {}
            for name in ("top", "left_wrist", "right_wrist", "triptych"):
                path = directory / (name + ".gif")
                path.write_bytes(b"GIF89a")
                views[name] = str(path)
            row = {"steps": 600, "garment": "Top_Short_Seen_0", "seed": 100,
                   "policy_variant": "adapt_s0"}
            result = {**row, "completed_steps": 600, "mode": "policy", "policy": "/checkpoint",
                      "success": False, "terminal_success": False, "gif_views": views,
                      "n_rendered": 601, "render_integrity": {"validated_render_calls": 601}}
            runner.validate_result(result, row, Path("/checkpoint"), directory)
            for bad in ({"seed": 101}, {"steps": 12}, {"success": "False"},
                        {"gif_views": {}}, {"n_rendered": 4}, {"policy": "/wrong"}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    runner.validate_result({**result, **bad}, row, Path("/checkpoint"), directory)
            Path(views["right_wrist"]).unlink()
            with self.assertRaises(ValueError):
                runner.validate_result(result, row, Path("/checkpoint"), directory)

    def test_crash_has_no_fold_outcome_and_blocks_remaining_gate_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"tasks": [{"id": "gate"}, {"id": "later"}]}))
            directory = root / "outputs/gate"
            directory.mkdir(parents=True)
            (directory / "status.json").write_text(json.dumps({"state": "infrastructure_error", "returncode": 3}))
            rows = report.summarize(root)
            self.assertIsNone(rows[0]["success"])
            self.assertEqual(rows[1]["state"], "blocked_by_failed_gate")

    def test_interrupted_episode_remains_unscored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"tasks": [{"id": "gate"}]}))
            (root / "outputs/gate").mkdir(parents=True)
            rows = report.summarize(root)
            self.assertEqual(rows[0]["state"], "started_without_completion_record")
            self.assertIsNone(rows[0]["success"])


if __name__ == "__main__":
    unittest.main()
