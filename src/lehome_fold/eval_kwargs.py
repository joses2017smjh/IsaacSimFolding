"""Translate LeHome's policy_kwargs into what LeRobotPolicy actually accepts.

`scripts/utils/evaluation.py` fills policy_path / dataset_root /
task_description only when `policy_type == "lerobot"`. Every other registered
type -- `candidate` and `recap` among them -- is constructed with

    {"device": ..., "model_path": args.policy_path}

and nothing else. Both of ours subclass LeRobotPolicy, which requires all
three, so the gap is bridged here rather than inside the policy constructor,
where it could not be tested without loading a 450M-parameter checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

BASE_TASK_FALLBACK = "fold the garment on the table"


def translate_policy_kwargs(kwargs: Mapping[str, Any],
                            env: Mapping[str, str],
                            *, base_task: str = BASE_TASK_FALLBACK,
                            ) -> Dict[str, Any]:
    """Return kwargs bindable to `LeRobotPolicy.__init__`.

    `env` supplies the two arguments evaluation.py withholds. run_eval.py
    stashes the evaluator's own argv into LH_EVAL_* so dataset_root cannot
    drift from the dataset being evaluated; DATASET_ROOT is the fallback.
    """
    out = dict(kwargs)
    if "model_path" in out and "policy_path" not in out:
        out["policy_path"] = out.pop("model_path")
    if not out.get("dataset_root"):
        out["dataset_root"] = (env.get("LH_EVAL_DATASET_ROOT")
                               or env.get("DATASET_ROOT", ""))
    if not out.get("task_description"):
        out["task_description"] = (env.get("LH_EVAL_TASK_DESCRIPTION")
                                   or base_task)
    if not out.get("policy_path"):
        raise ValueError(
            "no checkpoint given: evaluation.py passes --policy_path as "
            "model_path for non-lerobot policy types, and neither arrived")
    if not out.get("dataset_root"):
        raise ValueError(
            "dataset_root is required (LeRobotPolicy reads the action shape "
            "from its metadata); pass --dataset_root to the evaluator")
    return out
