# Folding media status

Generated 2026-09-20T15:23:47.785764+00:00.

The two policies are independently trained SmolVLA adaptation seeds, not different architectures. These recorded development poses are for illustrating behavior. Use the separately queued official evaluations for performance claims. A success means the unchanged checker fired at least once; terminal success is reported separately. Missing/crashed runs have no fold verdict.

| Policy | Garment / pose | State | Checker ever / terminal | Camera GIFs |
|---|---|---|---|---|
| adapt_s0 | Top_Short_Seen_0 / 2 | infrastructure_error | pending / unknown | awaiting media |
| adapt_s1 | Top_Short_Seen_0 / 2 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Top_Short_Seen_0 / 1 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Top_Short_Seen_0 / 1 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Top_Long_Seen_0 / 251 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Top_Long_Seen_0 / 251 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Top_Long_Seen_0 / 250 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Top_Long_Seen_0 / 250 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Pant_Short_Seen_0 / 509 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Pant_Short_Seen_0 / 509 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Pant_Short_Seen_0 / 502 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Pant_Short_Seen_0 / 502 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Pant_Long_Seen_0 / 750 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Pant_Long_Seen_0 / 750 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s0 | Pant_Long_Seen_0 / 751 | blocked_by_failed_gate | pending / unknown | awaiting media |
| adapt_s1 | Pant_Long_Seen_0 / 751 | blocked_by_failed_gate | pending / unknown | awaiting media |

## Historical demonstration replay controls

These are existing recorded-action replays, not either new trained policy. Their historical animation timing is preserved; the current media jobs record timing separately.

- [Top_Short_Seen_0 — replay success](reference_replays/rollout_Top_Short_Seen_0_replay_ep2_success.gif) ([scorer record](reference_replays/rollout_Top_Short_Seen_0_replay_ep2.json))
- [Top_Short_Seen_0 — replay failure](reference_replays/rollout_Top_Short_Seen_0_replay_ep1_failure.gif) ([scorer record](reference_replays/rollout_Top_Short_Seen_0_replay_ep1.json))
- [Top_Long_Seen_0 — replay success](reference_replays/rollout_Top_Long_Seen_0_v2_ep251_replay_ep251_success.gif) ([scorer record](reference_replays/rollout_Top_Long_Seen_0_v2_ep251.json))
- [Top_Long_Seen_0 — replay failure](reference_replays/rollout_Top_Long_Seen_0_v2_ep250_replay_ep250_failure.gif) ([scorer record](reference_replays/rollout_Top_Long_Seen_0_v2_ep250.json))
- [Pant_Short_Seen_0 — replay success](reference_replays/rollout_Pant_Short_Seen_0_v2_ep501_replay_ep501_success.gif) ([scorer record](reference_replays/rollout_Pant_Short_Seen_0_v2_ep501.json))
- [Pant_Short_Seen_0 — replay failure](reference_replays/rollout_Pant_Short_Seen_0_v2_ep502_replay_ep502_failure.gif) ([scorer record](reference_replays/rollout_Pant_Short_Seen_0_v2_ep502.json))
- [Pant_Long_Seen_0 — replay success](reference_replays/rollout_Pant_Long_Seen_0_replay_ep750_success.gif) ([scorer record](reference_replays/rollout_Pant_Long_Seen_0_replay_ep750.json))
- [Pant_Long_Seen_0 — replay failure](reference_replays/rollout_Pant_Long_Seen_0_replay_ep751_failure.gif) ([scorer record](reference_replays/rollout_Pant_Long_Seen_0_replay_ep751.json))

## Next session

Read docs/AUDIT_2026-09-19.md, submissions.jsonl and manifest.json first. Inspect only the new media jobs recorded in submissions.jsonl. Do not cancel, requeue, release or modify any old jobs. Regenerate this report with `python3 scripts/media_report.py --campaign .`. If a class has no policy success, say so; do not substitute a replay under a policy caption.
