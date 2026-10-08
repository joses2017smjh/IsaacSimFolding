# Folding session status — 19 September 2026

The correct project is IsaacSimFolding / lehome-fold-repro. Newer training and strict evaluations live in the sibling bhl-robustness-ladder weekend campaign.

Two adapted SmolVLA checkpoints are complete (training seeds 0 and 1). Their existing baseline/adaptation evaluations 21359573, 21359574, 21359575 were preserved. Historical 6/24 is invalid as a visual-policy baseline because observations became stale.

New media submissions:

- gate: `21360435`; dependency `afterany:21359573:21359574:21359575`.
- episodes: `21360436`; dependency `afterok:21360435`.
- report: `21360437`; dependency `afterany:21360436`.

Sixteen new pinned-pose episodes cover two checkpoints, four classes, two poses each. Each saves top, left wrist, right wrist and combined GIFs with actual checker outcomes. Maximum one of these GPU jobs at a time, thirty minutes each (eight GPU-hours maximum). The first episode is a gate. Existing jobs are never changed.

Read [the full audit](campaigns/20260919-media/docs/AUDIT_2026-09-19.md), [media status](campaigns/20260919-media/MEDIA_STATUS.md), and [submission receipts](campaigns/20260919-media/submissions.jsonl) before next-session work. The [gallery](campaigns/20260919-media/gallery.html) includes eight historical demonstration replay controls, explicitly labelled; new policy media is pending. Do not claim a success for a class unless its own policy episode passed.

Legacy candidate selection has an action-independent score; RECAP's gradient-update path is unimplemented. Do not relaunch either as a functioning new method. Next research step after results is more valid long-pants data and corrective rollout trajectories.
