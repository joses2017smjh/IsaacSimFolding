# 20260922 closed-loop training v1

Opened from `0e83a7b6f4aba81d55785774d8456b7a58d0ecfc`. The horizon-pilot
diagnostic campaign is closed: fixed RNG, RTC, retained queues, observation
substitution, static H50 suffix training, mixed replay, and fresh-H50 rescue
were insufficient; the fresh-H50 rescue oracle recovered 0/6 student-visited
roots. H50 plans are therefore prohibited as training labels here.

## Protocol

- Immutable starting checkpoint: `bc_smolvla_raster_ft_full`, pinned in
  `manifest.json` by model/config/normalizer SHA-256.
- First gate: one autonomous H10 Isaac rollout, persisted full executed-action
  stream plus sampled RGB/state observations, compiler alignment checks, one
  AWR-resampled mixed update, checkpoint save, and evaluator-style reload.
- Full iteration if the smoke passes: eight predeclared 600-action ordinary-H10
  rollouts spanning Pant Short, Pant Long, Top Short, and Top Long, two real
  training poses each. Failures are retained.
- The reward is the existing official terminal geometric
  `conditions_passed / conditions_total`. Its fixed-collection mean is the
  advantage baseline; no invented value prediction is claimed. Existing
  `lehome_fold.awr.weights` is implemented through importance resampling,
  because SmolVLA exposes a scalar flow-matching batch loss.
- Every batch is 50% AWR-resampled rollout data and 50% original raster BC
  anchors. The legacy held-out raster loss is a guard only, not a target.
- Candidate selection is by terminal settled geometric successes on the same
  eight fixed H10 benchmark rows. The exact pose-3 action-5/action-10 snapshot
  branches are also run under ordinary fresh H10 semantics for every candidate.

## Current state

The campaign scaffold and focused static checks are complete. The smoke Slurm
job is the next authorized action; no full collection or training has started.
Teleoperation is not available in the current headless setup: the repository
stubs the GUI listener and contains no usable simulator recovery collector, so
this does not block the autonomous iteration.
