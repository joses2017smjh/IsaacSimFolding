# Boundary-state successful-H50 replay diagnostic

This diagnostic started from the validated pose-3 `Pant_Short_Seen_3` H50 success. It used exactly two corrective examples: the real post-action-5 and post-action-10 observations from that episode, with all three current RGB cameras, current `observation.state`, and the unchanged task. Targets were the corresponding cached H50 suffixes `[5:50]` and `[10:50]`; no fresh replan, stale/hybrid input, demonstration state, RTC output, or interpolation was used.

The active rollout controller performs policy inference before the subsequent `env.step()`. Its prediction audit therefore treats inference as blocking and fails closed if simulator or episode counters change during prediction. The final snapshot-only evaluator restored both exact simulator snapshots before every baseline/trained prediction, restored the same reconstructed Torch/CUDA RNG state for both models, and passed all four checks: snapshot restore, exact RNG restore, zero simulator advancement, and same-RNG pairing.

## Fixed experiment

The frozen baseline checkpoint was not modified. The one micro-fine-tune used seed `4242`, `300` optimizer steps, batch size `2`, learning rate `1e-5`, and the smallest existing plan-conditioning path (`99,880,992` parameters: action expert and action/state projections). The large VLM/vision path remained frozen. The NPZ loader asserted the two boundary shapes, valid masks, target indices, cached-action equality, task, garment/pose, checkpoint hashes, and unchanged normalization files.

The fixed held-out raster sample contained the first eight frames of four deterministic `storm_capture100` files. Its loss increased from `0.133254` to `0.213044` (`heldout_gate=false`), so the preregistered no-substantial-regression condition failed.

## Snapshot inference gate

All values are radians and compare each predicted suffix to the successful cached H50 suffix.

| boundary | model | first-action RMS | first-action max | first-10 RMS | first-10 max | relative first-10 RMS reduction |
|---:|---|---:|---:|---:|---:|---:|
| 5 | baseline same-RNG fresh | 0.092820 | 0.191196 | 0.106624 | 0.329396 | — |
| 5 | trained | 0.039395 | 0.090518 | 0.028604 | 0.090518 | 73.17% |
| 10 | baseline same-RNG fresh | 0.133385 | 0.237960 | 0.295612 | 1.023399 | — |
| 10 | trained | 0.015095 | 0.028021 | 0.030279 | 0.142734 | 89.76% |

The inference requirement of at least 50% first-10 RMS reduction passes at both boundaries. The combined gate fails because the held-out regression requirement fails. No normal fresh H10 closed-loop execution was launched, so there is no trained-controller settled-4/4 result to report and no larger pose 1/3/5/7 dataset or validation submission was authorized. The source cached H50 control itself remains a settled `4/4` success.

## Interpretation and stop decision

This is not evidence for dataset plumbing, target alignment, or normalization as the primary problem: the schema and source assertions passed, both target suffixes exactly matched the cached H50 stream, and unchanged checkpoint preprocessing was used. The action/state path has enough capacity to move the two boundary predictions substantially toward the target. The failure is instead consistent with optimization/generalization overfit to two corrective examples; the fixed held-out sample worsened despite the boundary memorization. Insufficient trainable capacity is not indicated by this result.

Stop this intervention. Recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment with an objective that preserves held-out raster behavior. Do not launch another training intervention, inference intervention, RTC/queue branch, or broad validation from this result.

Machine-readable outputs:

- `pose3-boundary-replay.npz` and `.json`: two aligned training examples and provenance.
- `inference-deviations.csv`: first-action/first-10 RMS and max metrics, restore/RNG/timing flags.
- `first10-action-differences.csv`: per-action, per-joint deviations for both boundaries and models.
- `condition-trajectories.csv`: validated cached-H50 condition trajectory; trained closed-loop was not run by gate design.
- `provenance.json` and `slurm-jobs.json`: source/checkpoint hashes, controls, gates, diagnosis and Slurm ledger.

The generated trained checkpoint remains at `trained_checkpoint/`; the repository’s existing `*.safetensors` ignore rule keeps the 1.2 GB model binary out of Git, while its paths and hashes are retained in provenance. The original baseline checkpoint remains untouched.
