# Observation-component causal diagnostic

Source: `outputs/observation_dev03_h50_Pant_Short_Seen_3/replan-causality.json`.
This pose-3-only diagnostic uses the exact action-5 and action-10 H50 snapshots, cached H50 and same-reconstructed-RNG fresh H10 controls, and no RTC guidance.

## Timing and restoration gate

Prediction records: 576; simulator/episode-counter failures: 0.
All component probes restored within tolerance, matched the current-all same-RNG control, and had zero simulator advancement: **PASS**.
The original action-0 H50 observation and unchanged task are retained in the raw provenance JSON. The existing passive prefix-feature hook is reported there when available.

## Probe ranking

Stale and hybrid rows are attribution probes only; they are not deployable controllers.
Meaningful gate: at least 0.05 rad first-10 RMS improvement at both boundaries relative to same-RNG current-all.

| probe | action-5 improvement | action-10 improvement | both-boundary gate |
|---|---:|---:|:---:|
| current-images-old-state | 0.067168 | 0.178181 | PASS |
| old-all-images-current-state | 0.035476 | 0.096766 | FAIL |
| old-both-wrists | 0.035445 | 0.100364 | FAIL |
| old-left-only | 0.032181 | 0.096105 | FAIL |
| old-top-only | -0.001134 | -0.001613 | FAIL |
| old-right-only | 0.012243 | -0.010351 | FAIL |

Selected for execution (at most two): **current-images-old-state**.

- `current-images-old-state` settled 4/4 at both boundaries: **False**.

No executed candidate recovered settled 4/4 at both boundaries. Stop inference interventions and recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment. No larger validation was submitted.

Condition trajectories are in `condition-trajectories.csv`; first-ten action differences and per-joint RMS deltas are in `first10-action-differences.csv` and `per-joint-deltas.csv`; machine-readable gates and provenance are in `execution-gate.json` and `provenance.json`.
