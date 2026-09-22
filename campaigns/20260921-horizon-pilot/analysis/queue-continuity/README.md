# Hard retained-prefix queue diagnostic

This directory records the inference-only pose-3 continuity gate. The raw GPU capture uses the exact action-5 and action-10 simulator snapshots reconstructed by the active causal controller, with cached H50 and same-reconstructed-RNG fresh H10 controls. New branches use only the hard rule `previous_remainder[:d]`, discard `fresh_chunk[:d]`, execute `fresh_chunk[d:10]`, and carry `fresh_chunk[10:50]` to the next H10 replan. RTC is disabled.

Run the CPU aggregation after the pose-3 Slurm job:

```bash
python analysis/queue-continuity/build_report.py
```

`validation-gate.json` is the only authorization input for the conditional validation launcher. If both pose-3 boundaries pass for one or more delays and all invariance checks pass, `scripts/submit_queue_validation.py --submit` selects the smallest passing delay and submits exactly pose slots 1, 3, 5 and 7. Otherwise it records a stop recommendation for an observation-component ablation and submits nothing.
