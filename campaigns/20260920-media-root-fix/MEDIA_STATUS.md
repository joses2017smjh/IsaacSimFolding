# Folding media status

Generated 2026-09-20T17:35:15.251122+00:00.

This gate runs one existing SmolVLA adaptation checkpoint. These recorded development poses are for illustrating behavior. Use the separately queued official evaluations for performance claims. A success means the unchanged checker fired at least once; terminal success is reported separately. Missing/crashed runs have no fold verdict.

| Policy | Garment / pose | State | Checker ever / terminal | Camera GIFs |
|---|---|---|---|---|
| adapt_s0 | Top_Short_Seen_0 / 2 | completed | failure / False | [top](outputs/adapt_s0_Top_Short_Seen_0_pose2/rollout_policy_failure_top.gif) · [left_wrist](outputs/adapt_s0_Top_Short_Seen_0_pose2/rollout_policy_failure_left_wrist.gif) · [right_wrist](outputs/adapt_s0_Top_Short_Seen_0_pose2/rollout_policy_failure_right_wrist.gif) · [triptych](outputs/adapt_s0_Top_Short_Seen_0_pose2/rollout_policy_failure_triptych.gif) |

## Next session

Read README.md, gate_submission.json and manifest.json first. Inspect only the new media jobs recorded in gate_submission.json. Do not cancel, requeue, release or modify any old jobs. Regenerate this report with `python3 scripts/media_report.py --campaign .`. If a class has no policy success, say so; do not substitute a replay under a policy caption.
