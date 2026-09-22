# Fixed-RNG replanning report

## Protocol

At actions 5 and 10, the in-process simulator snapshot is restored. The cached H50 suffix is compared with a fresh first chunk generated from the reconstructed original sampler state. The same-RNG branch executes that first chunk with the H5 or H10 execution interval and then continues with ordinary fresh replanning. A four-seed panel (101, 102, 103, 104) is generated from the same restored observation for descriptive plan sensitivity. Poses 5 and 7 are plan-comparison cases only unless cached replay again reproduces their original success.

## Directly measured facts

Results are in `plan-comparisons.csv`; raw chunks and seed metadata are in `seed-panel.json`.

* Immediate simulator restoration is exact for all eight boundary cases: joint and cloth RMS are zero.
* The reconstructed original H50 50-action prediction matches the first 50 recorded baseline actions exactly for all four poses (50 actions compared, maximum absolute error 0).
* Cached H50 reaches 4/4 at both boundaries for poses 3, 5 and 7, and at pose 1 action 5. Pose 1 action 10's cached branch reaches only 3/4 in this replay and is therefore not a causal comparison.
* Same-RNG fresh branches reach 3/4 and fail geometric success for pose 1/action 5 and pose 3/actions 5 and 10. They match the normal fresh branch outcomes in all eight rows.
* The same-RNG first-action L2 difference from the reconstructed H50 suffix is 0.32–0.93 rad for pose 3 and 0.52–0.93 rad for pose 1; maximum joint differences over the first ten commands range from 0.29 to 1.28 rad in the supported cases.
* The four predetermined seeds produce materially different first-action plans. Across the eight boundaries, first-action L2 distances relative to the cached suffix range approximately 0.19–1.13 rad and maximum joint differences range approximately 0.26–1.40 rad. No seed was selected by outcome and these plans were not treated as a population estimate.

## Causal interpretation rules

* Cached success with same-RNG fresh failure means an inference seed change is not necessary for the observed boundary failure; observation-conditioned replanning remains supported.
* Same-RNG success with normal fresh failure would strengthen sampling as a mechanism.
* Large outcome variation in the fixed four-seed panel is descriptive evidence of a multimodal boundary distribution, not a tuned success rate.
* Poses 5 and 7 cannot be used for cached-versus-fresh causal success claims while their cached replay does not reproduce the original success.

The original H50 pilot did not save its pre-call RNG bytes. Consequently, this diagnostic uses a reproducible reconstruction from each frozen episode seed. The exact state bytes used here are recorded, and their effective output is verified against the historical action stream, but the historical bytes themselves are not recoverable. That limitation is part of the result.

## Causal result

For pose 3 at action 5 and action 10, cached H50 reaches 4/4 while the same-RNG fresh branch reaches only 3/4. Because the sampler state is held fixed at the reconstructed original call state, a changed inference seed is not necessary for the observed replanning failure. Pose 1 at action 5 shows the same direction, but its action-10 cached branch is inconclusive. This supports observation-conditioned replanning or loss of previous-plan context as the leading mechanism in the validated cases.

The normal fresh branch and same-RNG branch have the same outcomes in this diagnostic. Thus fixing the reconstructed RNG state did not rescue the validated fresh branches. The seed panel demonstrates plan sensitivity, but it was not executed to outcome and cannot establish a seed-conditioned success distribution. Poses 5 and 7 are retained as descriptive comparisons only because long deformable replay drift remains a limitation, even though their cached branches reached 4/4 in this run.

## Recommendation

The evidence-backed next experiment is one small inference-only temporal-continuity diagnostic on pose 3 at the action-5 and action-10 boundaries, preserving the cached H50 prefix and testing a reference-compatible retained-prefix/previous-plan mechanism if one can be identified. Do not start training, RL, or a full H10 rollout from this result.
