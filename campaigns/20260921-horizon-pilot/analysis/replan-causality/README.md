# Replan causality diagnostic

This directory contains the action-5/action-10 branch diagnostic for the four matched diagnostic cases previously successful under H50 (pose slots 1, 3, 5, and 7). It is an inference-time replay study; it does not change model weights and does not launch RL, RECAP, AWR, DAgger, BC, or finetuning.

The GPU capture was added to `scripts/render/policy_rollout51.py`. At each boundary it snapshots cloth particle positions and velocities, arm joint state and targets, camera frames, episode counters, and Python/NumPy/Torch/CUDA RNG state. It then restores the snapshot and runs:

* **Branch A:** the cached H50 executed suffix from the completed pilot;
* **Branch B:** a fresh prediction from the restored observation, executed with H5 or H10 semantics;
* **Branch C:** repeated predictions at the same restored observation, first with the same RNG state and then with four changed Torch seeds.

Immediate restoration errors are recorded separately from replay drift. A long Branch A replay can diverge in a deformable simulation even when the snapshot restoration is exact; those outcomes are marked inconclusive in the report. `events.jsonl` from the earlier instrumentation remains the raw per-action telemetry source; the files here contain the causal branch data.

Files:

* `branch-results.csv`: one row per pose and boundary, with outcomes, restoration checks, prefix action distances, and repeatability measurements.
* `action-plan-comparison.json`: raw first-50 action prefixes for cached, fresh, deterministic-repeat, and varied-seed predictions, plus all prefix metrics.
* `plots/pose-{1,3,5,7}.svg`: compact old-versus-fresh action and gripper-command diagnostics.
* `reference-inference-audit.md`: checked-out reference and local queue behavior.
* `REPORT.md`: evidence classification, matched results, limitations, and the next experiment recommendation.

The raw simulator captures are retained under the ignored `outputs/causal_*` directories and are the provenance for the aggregate artifacts.
