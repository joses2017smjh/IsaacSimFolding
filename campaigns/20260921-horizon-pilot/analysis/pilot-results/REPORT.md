# Internal folding analysis

Infrastructure-valid results: **24/24**. Missing and invalid episodes are not policy failures.

No direct comparison to historical 8/24; contacts/acquisition remain unmeasured. No public improvement claim.

| Horizon | Valid | Invalid/missing | Ever-success | Settled terminal |
|---|---:|---:|---:|---:|
| 50 | 8 | 0 | 4 | 4 |
| 10 | 8 | 0 | 0 | 0 |
| 5 | 8 | 0 | 0 | 0 |

Cells: ever / terminal; best → terminal conditions; worst margin cm; replans; wall seconds. Full displacement, proximity, all margins, camera/boundary metrics and paired deltas are in summary.json.

| Pose | Garment / local pose | H50 | H10 | H5 |
|---|---|---|---|---|
| 0 | Pant_Short_Seen_0 / 0 | 0/0; 3→2/4; -23.62; 12; 325.9 | 0/0; 2→1/4; -22.16; 60; 315.5 | 0/0; 3→2/4; -4.96; 120; 332.1 |
| 1 | Pant_Short_Seen_0 / 1 | 1/1; 4→4/4; 1.68; 12; 303.5 | 0/0; 3→3/4; -6.53; 60; 310.9 | 0/0; 3→2/4; -7.25; 120; 329.2 |
| 2 | Pant_Short_Seen_3 / 0 | 0/0; 2→2/4; -6.57; 12; 287.1 | 0/0; 3→1/4; -4.28; 60; 308.6 | 0/0; 3→3/4; -6.31; 120; 324.3 |
| 3 | Pant_Short_Seen_3 / 1 | 1/1; 4→4/4; 4.57; 12; 286.3 | 0/0; 3→3/4; -0.29; 60; 304.2 | 0/0; 3→1/4; -3.04; 120; 327.1 |
| 4 | Pant_Short_Seen_7 / 0 | 0/0; 3→1/4; -9.46; 12; 290.4 | 0/0; 2→2/4; -3.79; 60; 308.9 | 0/0; 3→2/4; -2.74; 120; 332.1 |
| 5 | Pant_Short_Seen_7 / 1 | 1/1; 4→4/4; 2.86; 12; 296.2 | 0/0; 3→3/4; -2.51; 60; 310.2 | 0/0; 2→2/4; -4.19; 120; 333.0 |
| 6 | Pant_Short_Seen_9 / 0 | 0/0; 3→3/4; -5.09; 12; 291.3 | 0/0; 3→3/4; -5.43; 60; 312.7 | 0/0; 2→2/4; -4.70; 120; 327.3 |
| 7 | Pant_Short_Seen_9 / 1 | 1/1; 4→4/4; 1.58; 12; 294.0 | 0/0; 4→3/4; -1.88; 60; 310.6 | 0/0; 3→2/4; -11.61; 120; 330.2 |

Matched behavior cells: closest-threshold margin cm; mean terminal cloth displacement m; minimum gripper-origin distance m; boundary jump mean rad; total inference seconds. Contacts and grasps remain unknown.

| Pose | H50 | H10 | H5 |
|---|---|---|---|
| 0 | 0.17; 0.0745; 0.0519; 0.247; 6.1 | 5.26; 0.0284; 0.0660; 0.247; 19.7 | 1.19; 0.0191; 0.0498; 0.277; 43.5 |
| 1 | 1.68; 0.0482; 0.0528; 0.331; 4.9 | 4.51; 0.0275; 0.0527; 0.219; 23.4 | 1.70; 0.0728; 0.0315; 0.232; 42.4 |
| 2 | -2.37; 0.0041; 0.0712; 0.190; 4.8 | -0.01; 0.0335; 0.0488; 0.274; 23.3 | 0.43; 0.0665; 0.0310; 0.256; 42.9 |
| 3 | 4.57; 0.0504; 0.0581; 0.149; 4.8 | -0.29; 0.0166; 0.0632; 0.214; 23.2 | 0.15; 0.0504; 0.0493; 0.270; 42.9 |
| 4 | -2.45; 0.0254; 0.0475; 0.296; 4.9 | -2.62; 0.0040; 0.0657; 0.231; 23.3 | -0.08; 0.0161; 0.0464; 0.279; 42.9 |
| 5 | 2.86; 0.0613; 0.0546; 0.377; 4.9 | -2.51; 0.0324; 0.0479; 0.170; 23.2 | 0.06; 0.0131; 0.0527; 0.184; 43.2 |
| 6 | 0.04; 0.0099; 0.0544; 0.279; 5.0 | 4.81; 0.0749; 0.0456; 0.246; 23.1 | -0.06; 0.0063; 0.0743; 0.193; 42.6 |
| 7 | 1.58; 0.0604; 0.0467; 0.277; 4.9 | -1.88; 0.0433; 0.0653; 0.236; 23.2 | -1.00; 0.0434; 0.0506; 0.268; 42.9 |

Inspect matched terminal/ever wins and margins, action boundaries and footage before freezing any formal evaluation horizon.

Condition CSVs/PNGs include the terminal settling period. Proximity and decreasing command events are proxies, not contact, grasp or failed-close ground truth.
