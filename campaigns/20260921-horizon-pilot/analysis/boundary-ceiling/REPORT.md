# Existing-checkpoint closed-loop ceiling diagnostic

This diagnostic compared the retained boundary-only 300-step checkpoint and mixed-replay `step_000300` from the exact validated pose-3 action-5/action-10 snapshots.
Fresh branches used ordinary H10 replanning. Cached H50 was retained only as a control and to reconstruct the snapshots.

- Boundary-only 300 settled 4/4 at both boundaries: `False` (diagnostic upper bound; non-deployable because held-out retention failed).
- Mixed replay step 300 settled 4/4 at both boundaries: `False`.
- Snapshot restoration checks: `True`.
- Zero simulator steps during prediction: `True`.
- Reconstructed RNG stream invariant across baseline and both checkpoints: `True`.
- Decision: `stop_static_boundary_suffix_adaptation`.

Do not launch PEFT or parameter anchoring. Run only a small on-policy corrective-data experiment: roll the selected H10 student from the exact pose-3 snapshots; at each student-visited H10 replan state capture the exact simulator snapshot and observation, query a fresh H50 plan from that actual state, branch-execute each H50 candidate, and retain labels only when the H50 branch demonstrably recovers the target condition. Do not reuse the original time-aligned cached H50 suffix after state divergence.

Controls and raw trajectories are in `ceiling-evaluation.json`; tabular trajectories are in `condition-trajectories.csv` and `settled-outcomes.csv`.
