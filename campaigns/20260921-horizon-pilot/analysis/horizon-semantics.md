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
