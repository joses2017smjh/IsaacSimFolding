# Horizon pilot failure instrumentation report

This report analyzes the completed, process-isolated 24-rollout horizon pilot. It does not replay episodes, change model weights, launch training, or make a public performance claim. The source pilot result remains in [`../pilot-results/REPORT.md`](../pilot-results/REPORT.md).

## Result and scope

All 24 rollouts were infrastructure-valid and produced 600 action records plus terminal settling. H50 had 4/8 official ever-successes and 4/8 settled terminal successes; H10 and H5 had 0/8 for both outcomes. The H50 successes were matched poses 1, 3, 5 and 7. H10 and H5 both lost all four of those paired successes and produced no new success. This confirms the established result but does not by itself identify a manipulation mechanism.

The new [`events.jsonl`](events.jsonl) contains 14,400 per-action records. [`matched-comparison.csv`](matched-comparison.csv) and [`matched-comparison.json`](matched-comparison.json) contain the paired measurements. The four synchronized plots are [`pose1_comparison.svg`](plots/pose1_comparison.svg), [`pose3_comparison.svg`](plots/pose3_comparison.svg), [`pose5_comparison.svg`](plots/pose5_comparison.svg), and [`pose7_comparison.svg`](plots/pose7_comparison.svg). Vertical lines show replan boundaries; dotted lines show the first action whose L2 difference from the independent H50 action exceeds 0.25 rad. The existing per-rollout MP4/GIF/RGB artifacts remain under `../../outputs/<episode-id>/`.

## Horizon audit

The setting called `horizon` is `policy.config.n_action_steps`, the number of leading rows consumed from each prediction before the action queue is refilled. The checkpoint still predicts a `(1, 50, 12)` chunk in all three cases. The runner calls `select_action` once per simulator action and executes exactly one returned action.

| setting | rows predicted per call | rows executed before next call | simulator observation refresh | calls in 600 actions | simulated time per call |
|---:|---:|---:|---|---:|---:|
| H50 | 50 | 50 | every 50 actions | 12 | 0.5556 s |
| H10 | 50 | 10 | every 10 actions | 60 | 0.1111 s |
| H5 | 50 | 5 | every 5 actions | 120 | 0.0556 s |

The environment advances by 1/90 s per action. Inference is synchronous and outside simulator time. There is no action overlap, interpolation, temporal blending, inpainting, asynchronous action server, or buffered playback in this local path. The queue is empty before each new chunk is requested, so the first action of a new chunk follows the previous chunk with no soft transition. The exact source comparison is documented in [`../horizon-semantics.md`](../horizon-semantics.md); it also records the upstream LeHome evaluator path, which delegates one action at a time to the policy abstraction rather than changing the installed SmolVLA chunk size.

## Directly measured simulator and policy facts

The raw stream records the executed 12-joint target, pre-action measured joint state, simulator time, replan boundary, boundary target jump, post-action named gripper/jaw-link origin, nearest cloth particle index and position, per-action particle lift summary, condition values and margins, gripper target and target delta, and camera-change samples. Finite differences of the recorded simulator positions add gripper-link and nearest-particle speeds plus a velocity-alignment value; these are derived from simulator positions and are not contact measurements.

The frozen pilot did not instantiate a validated Isaac Lab `ContactSensor` for the particle cloth and did not expose a validated cloth-to-rigid contact-pair stream. Particle positions and velocities alone cannot establish contact. Therefore `native_contact`, contact start/end, contacting gripper, acquisition, retention, release/drop and binary grasp labels are null/unknown in the event file. No ambiguous episode has been relabeled as contact or grasp.

The paired informative poses are summarized below. Empty approach fields mean that the named gripper-origin did not reach the stated proxy threshold.

| pose | horizon | terminal | first <=5 cm proxy | first <=3 cm proxy | minimum named-origin distance (L/R m) | best / terminal conditions | first divergence vs H50 | boundary mean jump (rad) |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 1 | 50 | success | — | — | 0.0528 / 0.0561 | 4 / 4 | — | 0.331 |
| 1 | 10 | fail | — | — | 0.0694 / 0.0527 | 3 / 3 | action 11, 0.508 rad | 0.219 |
| 1 | 5 | fail | 286 | — | 0.0315 / 0.0658 | 3 / 2 | action 6, 0.818 rad | 0.232 |
| 3 | 50 | success | — | — | 0.0624 / 0.0581 | 4 / 4 | — | 0.149 |
| 3 | 10 | fail | — | — | 0.0632 / 0.0681 | 3 / 3 | action 11, 0.483 rad | 0.214 |
| 3 | 5 | fail | 599 | — | 0.0493 / 0.0525 | 3 / 1 | action 6, 0.460 rad | 0.270 |
| 5 | 50 | success |  — | — | 0.0554 / 0.0546 | 4 / 4 | — | 0.377 |
| 5 | 10 | fail | 444 | — | 0.0664 / 0.0479 | 3 / 3 | action 11, 0.535 rad | 0.170 |
| 5 | 5 | fail | — | — | 0.0527 / 0.0663 | 2 / 2 | action 6, 0.422 rad | 0.184 |
| 7 | 50 | success | 213 | — | 0.0683 / 0.0467 | 4 / 4 | — | 0.277 |
| 7 | 10 | fail | — | — | 0.0653 / 0.0656 | 4 / 3 | action 11, 0.679 rad | 0.236 |
| 7 | 5 | fail | — | — | 0.0506 / 0.0614 | 3 / 2 | action 6, 0.924 rad | 0.268 |

The 5 cm and 3 cm columns are proximity proxies, not contact events. No episode produced the conservative five-sample motion-coupling proxy (near 3 cm, both motions at least 1 cm/s, velocity cosine at least 0.5). That absence is insufficient evidence, not evidence of no physical contact.

## Derived stage taxonomy

The offline classifier keeps the raw stream separate from labels. It uses `A` when neither named gripper origin reaches 5 cm, `B` when 5 cm is reached without a coupling proxy, `C` only for the conservative five-sample coupling proxy, `F` for a local/partial geometric regression, and `G` for official settled success. The 24 episodes were labeled 7 A, 3 B, 0 C, 9 partial-regression F, 1 local-success-then-terminal-failure F, and 4 G. These are evidence-status labels; they do not claim a physical contact or grasp. The detailed episode labels are in [`episode-summary.json`](episode-summary.json).

## Earliest divergence and mechanism assessment

Every H10 matched rollout first exceeded the 0.25-rad action-difference threshold at action 11, its first replan boundary. Every H5 matched rollout first exceeded it at action 6, also its first replan boundary. H50's first subsequent replan is action 51. This is direct evidence that shorter execution intervals do not preserve the H50 action sequence after the first fresh chunk. It is not a same-trajectory causal comparison: each pilot rollout independently calls the policy, receives a new observation, and produces a new chunk.

The boundary-jump means do not increase systematically as the horizon is shortened: across all episodes they are approximately 0.268 rad (H50), 0.230 rad (H10), and 0.245 rad (H5). Thus the pilot supports a **fresh-chunk/action-plan divergence at the first shorter-horizon replan**, but it does not support the narrower claim that a larger commanded discontinuity alone caused the failure.

Simulator time remains regular at one 1/90-second step per action, so there is no observed physics-time drift. Wall-clock and inference costs do change: mean inference time is about 5.04 s (H50), 22.79 s (H10), and 42.91 s (H5), while mean worker wall time is about 296.8 s, 310.2 s and 329.4 s. The extra calls are therefore a real runtime confound, but the current telemetry cannot show that latency changed a policy observation or physics state; it only shows that synchronous waiting increased.

The evidence is **inconclusive between policy decision change and chunk/replanning mechanics**. The local implementation has no overlap or soft inpainting to preserve temporal coherence, and fresh inference at every shorter boundary is the earliest reproducible change. However, the independent policy calls also receive new observations and any inference-time sampling state, so action divergence cannot be assigned to a mechanical boundary jump, an observation-conditioned decision, or sampling variability from this pilot alone. Missing native contact and gripper force/contact telemetry prevents a physics-level attribution to contact, acquisition, retention or release.

## Recommendation

Preserve H50 as the baseline and do not start RL, AWR, RECAP, DAgger or model training. The smallest next inference-time diagnostic is a paired four-pose pilot with the same seeds and checkpoint that records the H10/H5 fresh chunks while applying a short, explicitly logged linear blend from the prior chunk into the first few actions at each replan; compare it with unblended H10 and H50. This directly tests the only execution-mechanics intervention implicated by the audit, while the event schema should first be extended with a validated simulator contact sensor if the Isaac Sim scene exposes one. Do not treat that recommendation as evidence that blending will improve success, and do not launch it automatically from this report.
