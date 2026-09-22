# Inference and action-plan audit

## Local runner

`rollout_audit_utils.configure_action_queue` reads the checkpoint `chunk_size` (50), changes `policy.config.n_action_steps`, and calls `policy.reset()` so the internal action deque is rebuilt. `policy_rollout51.py` therefore requests a 50-action prediction but executes only the first 50, 10, or 5 actions before collecting a new observation and calling the policy again. The shorter setting changes execution length and observation refresh interval; it does not change the model's predicted chunk length. There is no action overlap, retained-prefix insertion, soft inpainting, interpolation, or action-plan blending in this path. One policy call represents one observation and one predicted 50-action chunk; in simulation each executed target is applied for one physics control step. Policy calls occur at action 1, 51, ... for H50, at 1, 11, ... for H10, and at 1, 6, ... for H5.

The causal runner captures both the raw normalized model chunk and the postprocessed command chunk. Branch A uses the completed pilot's executed H50 command stream, while Branch B postprocesses a newly predicted chunk before executing it. This distinction avoids accidentally replaying normalized model outputs as simulator joint targets.

## Checked reference checkout

The checked `external/lehome-challenge` implementation has a simple `select_action(observation)` contract. `scripts/eval_policy/lerobot_policy.py` preprocesses the observation, calls the wrapped policy once, postprocesses the result, and returns the action. The official evaluator calls this interface per environment step. The Docker adapter is the only checked reference adapter that explicitly caches a returned chunk: it consumes the server's actions until exhausted and then requests another chunk.

The repository search found no implementation or configuration named `actions_to_keep`, `actions_to_execute`, `soft_inpainting`, `inpainting`, or action overlap in the checked LeHome checkout or this campaign. Consequently, this campaign does not claim to reproduce a reference soft-inpainting mechanism. Any future continuity intervention must first identify a compatible upstream implementation and test its exact semantics.

## Consequence for interpretation

The local H50/H10/H5 comparison is a receding-horizon execution comparison, not a comparison of differently trained models or differently sized predictions. It changes how many leading commands survive before replanning. A difference at the first command after a boundary can arise from a genuinely observation-conditioned new prediction or from stochastic inference; it cannot be called a physics contact failure without simulator contact evidence.
