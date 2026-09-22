# Replan causality report

## Scope and protocol

The diagnostic examined poses 1, 3, 5, and 7, which are matched diagnostic cases previously successful under H50. At simulator actions 5 and 10, the runner restored a full simulator/policy snapshot and compared a cached H50 command suffix (Branch A), a fresh H5/H10 replan (Branch B), and repeated predictions at the same observation (Branch C). The full 600-action branch budget was used where the capture completed. The raw captures are under the ignored `outputs/causal_*` directories; aggregate values are in `branch-results.csv` and `action-plan-comparison.json`.

## Directly measured simulator and policy facts

* The snapshot restore itself is exact at the checked boundary in all eight cases: immediate cloth RMS and joint RMS are zero in `branch-results.csv`.
* Same-observation, same-RNG predictions are bitwise identical in all eight cases (`deterministic_repeat_max_abs = 0`).
* Changing the Torch inference seed changes predicted actions materially: the maximum pairwise action difference is approximately 2.46–3.37 radians across the eight cases.
* The cached-versus-fresh first-action L2 distance ranges from 0.32 to 3.14 radians. Pose 1 and pose 3 have high directional agreement for the first command (cosine about 0.995–0.998) despite nonzero magnitude differences; poses 5 and 7 have larger first-command differences and lower directional agreement.
* Branch A reaches 4/4 geometric conditions while Branch B reaches 3/4 for pose 1 at boundary 10 and for pose 3 at boundaries 5 and 10. The other branch pairs do not produce a supported success separation: the cached replay itself does not reach 4/4 in the captured branch.
* Cached suffix replays accumulate trajectory drift in some long branches. For example, pose 1 boundary 10 reaches 4/4 but has approximately 0.009 m maximum cloth-centroid replay error relative to the original trace; pose 3 has larger joint replay drift by the end of the branch. These are replay-fidelity limits, not contact measurements.

## Derived measurements and heuristic events

`branch-results.csv` reports L2 differences after the first 1, 3, 5, and 10 actions, gripper-command deltas, direction cosine, and repeatability. These are derived from command arrays; they are not simulator-native contact or grasp labels. The earlier `failure-instrumentation/events.jsonl` contains nearest-particle distance and motion-coupling proxies. The installed deformable-body path did not expose a validated binary finger/cloth contact stream, so contact start/end, acquisition, retention, release, and drop remain unknown.

## Hypotheses tested

**H1 — inference sampling:** supported as a source of plan variation. Identical restored observations are deterministic when RNG is restored, while changed seeds produce large action changes. This does not establish that sampling alone caused the task failures.

**H2 — fresh observation-conditioned replanning:** supported in the strongest paired cases. For pose 3, Branch A reaches 4/4 and Branch B reaches only 3/4 at both boundaries. Pose 1 shows the same separation at boundary 10. The fresh plan differs immediately from the cached suffix, so the shorter-horizon result is not explained by merely truncating an identical command stream.

**H3 — boundary discontinuity or temporal blending:** not isolated by this capture. The branch exposes the old and fresh command sequences and their discontinuity, but no compatible `actions_to_keep` or soft-inpainting reference implementation was found in the checked repository. No continuity intervention was launched.

**H4 — manipulation physics/contact:** inconclusive. The branch state contains cloth particles, joint state, end-effector state, and cloth displacement, but no validated native finger/cloth contact signal. The data cannot distinguish failed acquisition from an upstream command decision in the ambiguous cases.

**H5 — latency:** not isolated here. Inference timing is present in the earlier pilot telemetry, but the branch experiment does not hold wall-clock scheduling constant while replacing a fresh prediction with a cached suffix.

## Matched interpretation

Pose 3 is the cleanest causal result: exact boundary restoration, deterministic repeated prediction, materially different fresh plan, cached H50 suffix reaching 4/4, and fresh H10/H5 branch reaching 3/4. Pose 1 supplies a second supported separation at action 10. Pose 1 at action 5 and both boundaries for poses 5 and 7 are explicitly inconclusive because the cached replay did not itself reproduce the complete success or accumulated replay drift. They remain analyzed cases, not failures of the policy.

Across the supported separations, the earliest measurable difference is in the newly selected command at the replan boundary, not a demonstrated contact event. The evidence therefore favors a combination of fresh observation-conditioned replanning and inference sampling over a demonstrated cloth-physics mechanism. It does not prove whether the policy's semantic decision or the absence of an overlap/blending mechanism is the final cause.

## Recommendation

Preserve H50 as the baseline. Run at most one paired inference-only diagnostic on poses 1, 3, 5, and 7 with a fixed inference RNG and the same seeds, comparing H10/H5 fresh replans against the current cached suffix; this directly tests the supported H1 contribution without changing weights. Do not implement action blending or launch training until a compatible reference `actions_to_keep`/soft-inpainting design is located and the fixed-RNG result is available. Do not make a public performance claim from this diagnostic.
