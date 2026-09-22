"""Focused schema checks for the pose-3 boundary replay diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "finetune_rasterised.py"
SPEC = importlib.util.spec_from_file_location("boundary_finetune", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_capture(path: Path):
    actions = np.arange(2 * 50 * 12, dtype=np.float32).reshape(2, 50, 12)
    actions[0, 45:] = actions[0, 44]
    actions[1, 40:] = actions[1, 39]
    mask = np.zeros((2, 50), dtype=np.bool_)
    mask[0, :45] = True
    mask[1, :40] = True
    indices = np.full((2, 50), -1, dtype=np.int32)
    indices[0, :45] = np.arange(5, 50)
    indices[1, :40] = np.arange(10, 50)
    np.savez(
        path,
        images=np.zeros((2, 3, 480, 640, 3), dtype=np.uint8),
        state=np.zeros((2, 12), dtype=np.float32),
        target_actions_rad=actions,
        target_valid_mask=mask,
        target_chunk_indices=indices,
        boundaries=np.array([5, 10], dtype=np.int32),
        task=np.array(["fold the garment on the table"] * 2),
        torch_rng=np.zeros((2, 8), dtype=np.uint8),
        cuda_rng=np.zeros((2, 1, 8), dtype=np.uint8),
    )


def test_boundary_capture_loader_enforces_two_suffix_examples(tmp_path):
    path = tmp_path / "capture.npz"
    _write_capture(path)
    X, S, A, M, boundaries, tasks = MODULE.load_boundary_capture(str(path))
    assert X.shape == (2, 3, 480, 640, 3)
    assert S.shape == (2, 12)
    assert A.shape == (2, 50, 12)
    assert M.tolist() == [[True] * 45 + [False] * 5,
                          [True] * 40 + [False] * 10]
    assert boundaries.tolist() == [5, 10]
    assert tasks == ["fold the garment on the table"] * 2


def test_boundary_capture_rejects_misaligned_target_indices(tmp_path):
    path = tmp_path / "capture.npz"
    _write_capture(path)
    data = dict(np.load(path, allow_pickle=False))
    data["target_chunk_indices"] = data["target_chunk_indices"].copy()
    data["target_chunk_indices"][0, 0] = 4
    np.savez(path, **data)
    with pytest.raises(SystemExit, match="target indices"):
        MODULE.load_boundary_capture(str(path))
