# Hard retained-prefix queue diagnostic

Source: `outputs/queue_dev03_h50_Pant_Short_Seen_3/replan-causality.json`.
The diagnostic uses the exact pose-3 action-5 and action-10 in-process snapshots, cached H50 and same-reconstructed-RNG fresh H10 controls, and no RTC guidance.

## Timing and invariance gate

Prediction records: 918; simulator/episode-counter failures: 0.
Every queue prediction must have zero simulator steps and zero episode-length delta; every first fresh chunk must exactly match the same-RNG control. Snapshot restore, retained-action identity, fresh-prefix discard and suffix indexing are fail-closed checks in the raw JSON.

## Results

| delay | action-5 settled | action-10 settled | both-boundary gate |
|---:|:---:|:---:|:---:|
| 2 | 2/4 | 2/4 | FAIL |
| 5 | 3/4 | 2/4 | FAIL |
| 10 | 2/4 | 3/4 | FAIL |

Smallest passing delay: **none**.

No delay passed both pose-3 boundaries, or a required timing check failed. Stop here and recommend an observation-component ablation rather than another continuity intervention. No larger validation was submitted.

Detailed condition trajectories are in `condition-trajectories.csv`; first-ten action differences versus the cached H50 suffix are in `first10-action-differences.csv`; divergence at local actions 2/10/20 and gate fields are in `comparison.csv`.
