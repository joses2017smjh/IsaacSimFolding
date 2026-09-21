"""CPU checks for truthful verdicts, execution queues, and camera artifacts."""

from collections import deque
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

RENDER = Path(__file__).resolve().parents[1] / "scripts" / "render"
sys.path.insert(0, str(RENDER))
from rollout_audit_utils import (EpisodeAudit, configure_action_queue,
                                 gif_durations_ms, parse_pose,
                                 validate_camera_images, write_camera_artifacts)


class RolloutAuditTests(unittest.TestCase):
    def test_transient_success_does_not_become_terminal_success(self):
        audit = EpisodeAudit(4)
        self.assertFalse(audit.record_step(False))
        self.assertTrue(audit.record_step(True))
        audit.record_step(None)
        audit.record_step(None)
        verdict = audit.finish(False)
        self.assertTrue(verdict["success"])
        self.assertFalse(verdict["terminal_success"])
        self.assertEqual(verdict["first_success_step"], 2)
        self.assertEqual(verdict["first_success_step_index"], 1)
        self.assertEqual(verdict["completed_steps"], 4)

    def test_failure_and_incomplete_runs_are_distinct(self):
        audit = EpisodeAudit(2)
        audit.record_step(False)
        with self.assertRaises(ValueError):
            audit.finish(False)
        audit.record_step(False)
        verdict = audit.finish(False)
        self.assertFalse(verdict["success"])
        self.assertIsNone(verdict["first_success_step"])
        self.assertTrue(verdict["episode_complete"])

    def test_execution_override_rebuilds_queue_without_changing_prediction_size(self):
        class Policy:
            config = SimpleNamespace(n_action_steps=50, chunk_size=50)

            def reset(self):
                self.queue = deque(maxlen=self.config.n_action_steps)

        policy = Policy()
        original = configure_action_queue(policy, 0)
        self.assertEqual(original["effective_n_action_steps"], 50)
        overridden = configure_action_queue(policy, 5)
        self.assertEqual(overridden["effective_n_action_steps"], 5)
        self.assertEqual(policy.queue.maxlen, 5)
        self.assertEqual(policy.config.chunk_size, 50)
        with self.assertRaises(ValueError):
            configure_action_queue(policy, 51)
        self.assertEqual(policy.queue.maxlen, 5)

    def test_pose_validation_rejects_nonfinite_and_incomplete_values(self):
        self.assertEqual(parse_pose("1:2:3:4:5:6"), [1, 2, 3, 4, 5, 6])
        for bad in ("1:2", "1:2:3:4:5:nan", "1:2:3:4:5:inf"):
            with self.assertRaises(ValueError):
                parse_pose(bad)

    def test_cpu_guard_fires_before_importing_simulator(self):
        result = subprocess.run(
            [sys.executable, str(RENDER / "policy_rollout51.py"),
             "--lehome", "/tmp", "--policy_path", "/tmp/unused-policy",
             "--garment_dir", "/tmp/unused-garment", "--assets", "/tmp",
             "--sim_device", "cpu"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("CPU would freeze it", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_off_stride_success_preserves_simulation_timing(self):
        durations = gif_durations_ms([0, 3, 5, 9], 1 / 90, 3)
        self.assertEqual(durations, [30, 30, 40, 30])
        self.assertLessEqual(abs(sum(durations) / 1000 - 12 / 90), 0.01)
        with self.assertRaises(ValueError):
            gif_durations_ms([0, 3, 3], 1 / 90, 3)

    @staticmethod
    def camera_frames(step):
        pattern = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        return {key: ((pattern.astype(np.int16) + offset + step) % 256).astype(np.uint8)
                for key, offset in (("top_rgb", 0), ("left_rgb", 30), ("right_rgb", 80))}

    def test_bad_camera_shapes_and_blank_images_are_rejected(self):
        frames = self.camera_frames(0)
        self.assertEqual(validate_camera_images(frames, 8, 6)["top_rgb"]["shape"], [6, 8, 3])
        frames["left_rgb"] = np.zeros((6, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_camera_images(frames, 8, 6)
        frames["left_rgb"][:] = [200, 0, 0]  # colored but still spatially blank
        with self.assertRaises(ValueError):
            validate_camera_images(frames, 8, 6)
        frames["left_rgb"] = np.zeros((8, 6, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_camera_images(frames, 8, 6)

    def test_encoded_views_preserve_camera_mapping_events_and_timing(self):
        frames = {step: self.camera_frames(step) for step in (0, 3, 5, 9)}
        with tempfile.TemporaryDirectory(prefix="rollout-audit-test-") as output:
            media = write_camera_artifacts(frames, str(Path(output) / "policy_success"),
                                           1 / 90, 3, 5)
            self.assertEqual(set(media["gifs"]), {"top", "left_wrist", "right_wrist", "triptych"})
            for view, path in media["gifs"].items():
                with Image.open(path) as gif:
                    self.assertEqual(gif.size, (24 if view == "triptych" else 8, 6))
                    total_duration = 0
                    for index in range(gif.n_frames):
                        gif.seek(index)
                        total_duration += gif.info["duration"]
                    self.assertEqual(total_duration, sum(media["frame_durations_ms"]))
            for event, step in (("first_success", 5), ("final", 9)):
                for view, key in media["camera_keys"].items():
                    with Image.open(media["snapshots"][event][view]) as snapshot:
                        np.testing.assert_array_equal(np.asarray(snapshot), frames[step][key])
                with Image.open(media["snapshots"][event]["triptych"]) as snapshot:
                    pixels = np.asarray(snapshot)
                    np.testing.assert_array_equal(pixels[:, :8], frames[step]["left_rgb"])
                    np.testing.assert_array_equal(pixels[:, 8:16], frames[step]["top_rgb"])
                    np.testing.assert_array_equal(pixels[:, 16:], frames[step]["right_rgb"])


if __name__ == "__main__":
    unittest.main()
