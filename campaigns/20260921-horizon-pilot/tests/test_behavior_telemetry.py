import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lehome_fold.behavior_telemetry import BehaviorTelemetry


def stripped_controller(path):
    class StripTelemetry(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == 'lehome_fold.behavior_telemetry' else node
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'behavior' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'behavior':
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if not isinstance(k, ast.Constant) or k.value != 'behavior_telemetry']
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return self.generic_visit(node)
    return ast.dump(StripTelemetry().visit(ast.parse(path.read_text())), include_attributes=False)


class TelemetryTests(unittest.TestCase):
    def test_active_controller_order_is_preserved_with_explicit_diagnostic_hooks(self):
        source = (ROOT / 'scripts/render/policy_rollout51.py').read_text()
        inference = source.index('with torch.inference_mode():\n                act = policy.select_action(batch)')
        step = source.index('env.step(torch.from_numpy(a.astype(np.float32))', inference)
        self.assertLess(inference, step)
        self.assertIn('queue_diagnostic', source)
        self.assertIn('prediction_timing_audit', source)

    def test_exact_boundary_metrics_no_contact_claims_or_input_mutation(self):
        points = np.array([[0., 0., 0.], [0.1, 0, 0]])
        links = {side: (np.array([[0., 0, .01], [.3, 0, 0]]), np.zeros((2, 4))) for side in ('left', 'right')}
        geom = {'conditions_passed': 1, 'conditions_total': 4, 'success': False, 'details': {}}
        with tempfile.TemporaryDirectory() as tmp:
            writer = BehaviorTelemetry(Path(tmp) / 'behavior.jsonl', points, 2, 1 / 90,
                {side: ['gripper', 'jaw'] for side in links})
            state = np.random.get_state()
            originals = points.copy(), links['left'][0].copy()
            for step in range(1, 7):
                action = np.full(12, float(step // 3))
                images = {k: np.full((16, 16, 3), step * 10, dtype=np.uint8) for k in ['top_rgb', 'left_rgb', 'right_rgb']}
                writer.record(step, action, np.zeros(12), images, points, links, geom)
                np.testing.assert_array_equal(action, np.full(12, float(step // 3)))
            result = writer.finish()
            self.assertEqual([r['action'] for r in result['replan_boundary_jumps']], [3, 5])
            self.assertAlmostEqual(result['replan_boundary_jumps'][0]['l2_rad'], np.sqrt(12))
            self.assertEqual(result['steps'], 6)
            self.assertIsNone(result['first_cloth_contact'])
            self.assertIsNone(result['successful_cloth_acquisition'])
            self.assertEqual(result['first_link_origin_proximity_under_3cm_action']['left'], 1)
            self.assertEqual(len(Path(result['path']).read_text().splitlines()), 6)
            self.assertEqual(len(result['camera_changes_between_replans']), 2)
            np.testing.assert_array_equal(points, originals[0])
            np.testing.assert_array_equal(links['left'][0], originals[1])
            after = np.random.get_state()
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])

if __name__ == '__main__':
    unittest.main()
