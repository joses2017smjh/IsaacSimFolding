# Horizon semantics audit

The pilot's `horizon` is the installed LeRobot SmolVLA `config.n_action_steps` queue length. The checkpoint remains configured with `chunk_size=50`; the rollout calls `configure_action_queue(policy, horizon)`, which changes only `policy.config.n_action_steps` and resets the action queue. The worker then asserts that every `_get_action_chunk` result still has shape `(1, 50, 12)`.

At each simulator action step the runner renders the current state, calls `policy.select_action(batch)`, executes exactly the single action returned, and advances Isaac by one environment step. SmolVLA's `select_action` calls `_get_action_chunk` only when its action deque is empty, extends that deque with the first `n_action_steps` rows of the 50-row prediction, and pops one row. Therefore:

| Setting | Predicted rows per inference | Executed rows per inference | Observation refresh | Inferences in 600 actions | Simulated time per inference |
|---:|---:|---:|---|---:|---:|
| H50 | 50 | 50 | every 50 simulator steps | 12 | 0.5556 s |
| H10 | 50 | 10 | every 10 simulator steps | 60 | 0.1111 s |
| H5 | 50 | 5 | every 5 simulator steps | 120 | 0.0556 s |

The first action is inferred at simulator time zero after the fixed 60-step initial settle. One action advances the frozen environment by `dt * decimation = 1/90 s`. The inference wall time is outside simulated time: the process waits for the model call before executing the next action, so more frequent inference increases wall time but does not add physics steps. The pilot measured mean total policy inference time of 5.04 s, 22.79 s and 42.91 s for H50/H10/H5.

There is no action overlap, temporal blending, interpolation, inpainting, action-server buffering, or asynchronous policy execution in this runner. The action queue is emptied before a new chunk is requested. A new chunk starts at the next action with no soft transition from the old chunk. The telemetry's boundary jump is the L2 difference between the last action of the previous queue and the first action of the next queue.

The local path matches the ordinary upstream LeRobot policy abstraction: the reference LeHome evaluator calls `policy.select_action(observation)` once per environment step, converts the returned single action to a tensor, and calls `env.step(action)`. Its `lerobot_policy.py` wrapper delegates directly to `self.policy.select_action(batch_obs)`. The reference challenge also contains a separate Docker policy that caches whatever chunk length a server returns, but that is not the installed SmolVLA path and was not used here. No reference implementation in the checked repository performs action-plan overlap or soft inpainting for this experiment.

The horizon is therefore an execution/re-observation interval, not a change to the model's predicted horizon. The independent variable is how many leading rows of each unchanged 50-row prediction are executed before the next observation and fresh 50-row prediction.

The audit was performed from the frozen v5 source, the installed `modeling_smolvla.py`, `policies/utils.py`, and the upstream LeHome `scripts/utils/evaluation.py` and `scripts/eval_policy/lerobot_policy.py`. Source paths and exact hashes are preserved in the pilot manifest.

## Active-controller timing check

The active controller in `scripts/render/policy_rollout51.py` has the following ordering for every ordinary policy action: render/build the observation, call `policy.select_action(batch)` (the inference call is inside the `torch.inference_mode()` block), validate the returned 12-joint target, then call `env.step(...)`. The `env.step` call is after inference in the same loop; there is no background worker or simulator tick between those statements. The causal diagnostic uses the same boundary: `_predict_observed()` samples the simulator and episode counters before and after `_get_action_chunk`, synchronizes CUDA, and fails if either counter changes.

Each timing record stores `sim_step_delta`, `episode_length_delta`, `rng_before_digest`, `rng_after_digest`, and `inference_blocks_before_env_step`. The required dynamic audit result is therefore `sim_step_delta == 0` and `episode_length_delta == 0` for every prediction call. Inference wall time may increase, but zero simulator actions elapse while the model is blocked.

## Hard queue diagnostic semantics

The new `--queue_diagnostic` branch is inference-only and is restricted by its launcher to the exact pose-3 snapshots at action 5 and action 10. It preserves the cached H50 suffix and same-reconstructed-RNG fresh H10 controls. For each delay `d` in `{2,5,10}`, it generates a postprocessed 50-row fresh chunk from the saved boundary observation and reconstructed RNG stream, executes exactly `d` rows from the current previous-plan remainder, discards fresh rows `[0:d]`, executes fresh rows `[d:10]`, and carries fresh rows `[10:50]` as the current previous-plan remainder at the next H10 boundary. Every executed row records its source and index; the run fails closed if retained identity, fresh suffix indexing, snapshot restoration, RNG reconstruction, or zero-step inference checks fail. RTC is rejected for this branch.
