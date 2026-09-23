# Internal folding analysis

Infrastructure-valid results: **1/24**. Missing and invalid episodes are not policy failures.

No direct comparison to historical 8/24; contacts/acquisition remain unmeasured. No public improvement claim.

| Horizon | Valid | Invalid/missing | Ever-success | Settled terminal |
|---|---:|---:|---:|---:|
| 50 | 0 | 8 | 0 | 0 |
| 10 | 1 | 7 | 0 | 0 |
| 5 | 0 | 8 | 0 | 0 |

Cells: ever / terminal; best → terminal conditions; worst margin cm; replans; wall seconds. Full displacement, proximity, all margins, camera/boundary metrics and paired deltas are in summary.json.

| Pose | Garment / local pose | H50 | H10 | H5 |
|---|---|---|---|---|
| 0 | Pant_Short_Seen_0 / 0 | INVALID / pending | 0/0; 2→1/4; -22.16; 60; 315.5 | INVALID / pending |
| 1 | Pant_Short_Seen_0 / 1 | INVALID / pending | INVALID / pending | INVALID / pending |
| 2 | Pant_Short_Seen_3 / 0 | INVALID / pending | INVALID / pending | INVALID / pending |
| 3 | Pant_Short_Seen_3 / 1 | INVALID / pending | INVALID / pending | INVALID / pending |
| 4 | Pant_Short_Seen_7 / 0 | INVALID / pending | INVALID / pending | INVALID / pending |
| 5 | Pant_Short_Seen_7 / 1 | INVALID / pending | INVALID / pending | INVALID / pending |
| 6 | Pant_Short_Seen_9 / 0 | INVALID / pending | INVALID / pending | INVALID / pending |
| 7 | Pant_Short_Seen_9 / 1 | INVALID / pending | INVALID / pending | INVALID / pending |

Matched behavior cells: closest-threshold margin cm; mean terminal cloth displacement m; minimum gripper-origin distance m; boundary jump mean rad; total inference seconds. Contacts and grasps remain unknown.

| Pose | H50 | H10 | H5 |
|---|---|---|---|
| 0 | INVALID / pending | 5.26; 0.0284; 0.0660; 0.247; 19.7 | INVALID / pending |
| 1 | INVALID / pending | INVALID / pending | INVALID / pending |
| 2 | INVALID / pending | INVALID / pending | INVALID / pending |
| 3 | INVALID / pending | INVALID / pending | INVALID / pending |
| 4 | INVALID / pending | INVALID / pending | INVALID / pending |
| 5 | INVALID / pending | INVALID / pending | INVALID / pending |
| 6 | INVALID / pending | INVALID / pending | INVALID / pending |
| 7 | INVALID / pending | INVALID / pending | INVALID / pending |

Wait for all results or repair infrastructure; do not select a horizon from incomplete pairs.

Condition CSVs/PNGs include the terminal settling period. Proximity and decreasing command events are proxies, not contact, grasp or failed-close ground truth.
