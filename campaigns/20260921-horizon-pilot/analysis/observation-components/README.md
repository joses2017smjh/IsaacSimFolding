# Observation-component causal diagnostic

This is a pose-3-only, inference-only diagnostic using the exact in-process
H50 snapshots after actions 5 and 10. It preserves two controls at each
boundary: the cached H50 suffix and a normal fresh H10 prediction from the
reconstructed Torch/CUDA RNG stream.

The active observation is `observation.state`, `top_rgb`, `left_rgb`,
`right_rgb`, plus the unchanged task. The diagnostic captures the original
action-0 H50 observation and evaluates current-all, stale-image, stale-state,
single-wrist, and both-wrist counterfactuals. Stale and hybrid inputs are
explicitly attribution probes, not deployable controllers.

Every prediction restores the exact saved simulator snapshot and a reconstructed
Torch/CUDA RNG state. The active controller records simulator and episode
counters around inference and fails closed if either advances before
`env.step()`.

The preregistered ranking gate is a reduction of at least 0.05 rad in cached
H50-suffix first-10 RMS at both action-5 and action-10 boundaries, relative to
same-RNG current-all. At most the best two probes can be executed, and no
larger validation is submitted by this diagnostic.
