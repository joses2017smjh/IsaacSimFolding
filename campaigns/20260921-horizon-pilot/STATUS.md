# Internal horizon pilot — 21 September 2026

This is a local campaign record. No public claims or README were updated and no training, RECAP or AWR work was launched.

## GPU certification

All five original v5 smokes passed full experiment validation, including 120 executed actions, finite robot and cloth states, fresh visible RGB, verified scorer binding, 60 terminal settling steps, measurable terminal geometry and consistent result/status JSON. Slurm completion alone was not used.

| Garment | Worker job | Ever | Settled terminal | Conditions | Minimum visible top pixels |
|---|---|---|---|---|---:|
| Top_Short_Seen_3 | 21383706 | False | False | 2/5 | 5080 |
| Pant_Short_Seen_0 | 21383707 | False | False | 1/4 | 6138 |
| Pant_Long_Seen_0 | 21383748 | False | False | 2/4 | 6265 |
| Top_Long_Seen_0 | 21383779 | False | False | 2/5 | 9165 |
| Top_Short_Seen_0 | 21383691 | False | False | 2/5 | 8219 |

Gate: **5/5 passed**. Switch job **21383692** returned in **1.182 s**. This does not reproduce the historical hang or certify a second policy episode/camera retarget after switching. All pilot episodes remain process-isolated.

[Certification and evidence hashes](audit/v5_certification.json). [Five-smoke timelines and plots](analysis/smoke-results/REPORT.md) cover the four actual garment classes; these were 120-action smoke episodes, not new 600-action class benchmarks.

## Released matrix

Pilot **21384164**, array **0-23%1**, was submitted after the gate passed; no outstanding Slurm dependency. Each task uses one GPU on the previously validated `gpu,ampere` partitions with `a40|rtx8000`, eight CPUs, 64 GB RAM, a 15-minute allocation and a bounded child process. It runs independently of the desktop.

| Slot | Garment | Local pose key | Seed | Array indices H50 / H10 / H5 |
|---:|---|---:|---:|---|
| 0 | Pant_Short_Seen_0 | 0 | 200 | 0 / 1 / 2 |
| 1 | Pant_Short_Seen_0 | 1 | 201 | 3 / 4 / 5 |
| 2 | Pant_Short_Seen_3 | 0 | 202 | 6 / 7 / 8 |
| 3 | Pant_Short_Seen_3 | 1 | 203 | 9 / 10 / 11 |
| 4 | Pant_Short_Seen_7 | 0 | 204 | 12 / 13 / 14 |
| 5 | Pant_Short_Seen_7 | 1 | 205 | 15 / 16 / 17 |
| 6 | Pant_Short_Seen_9 | 0 | 206 | 18 / 19 / 20 |
| 7 | Pant_Short_Seen_9 | 1 | 207 | 21 / 22 / 23 |

All 24 rows and every original protocol field exactly match the frozen v5 manifest: fixed checkpoint, garment IDs, poses/scales, seeds, camera pipeline, official scorer, action normalization, physics timestep, 600-action budget and 60-step terminal settling. The predicted chunk stays 50; only executed-prefix length varies. Expected replans are 12 / 60 / 120. The model SHA-256 is `4a37d4dce3d96e392044ba86295220467513e70ca37f8458edc855edb238ba77`.

The separate execution snapshot adds passive recording hooks requested for this experiment. An AST regression verifies that stripping those hooks yields the exact v5 controller AST. Telemetry tests verify no action/particle input mutation or NumPy RNG consumption. The 142 execution sources are frozen; workers recheck checkpoint, source, garment and original v5 result hashes. This is observational instrumentation of the v5 experiment, not a changed policy protocol. GPU telemetry completion is verified per result before analysis accepts it.

## Measurements and their limits

Every pilot result keeps official ever-triggered success and fresh settled terminal success separate, plus first-success action/time, geometric persistence, every condition margin, full condition trajectories, best policy and terminal condition counts, cloth displacement, inference timings, replans, runtime, camera freshness and labeled policy MP4/GIF views.

`rollout.json.behavior.jsonl` streams all 600 executed targets and pre-action joint states, replan boundary changes, stride-8 RGB change statistics between replans, post-action named gripper/jaw-origin distances and nearest particle IDs, gripper command deltas, and maximum/p95 particle lift. Input images correspond to the state before that action; geometry and proximity correspond to the resulting state. The recorder does not render an extra frame, step physics, sample an RNG or alter control. Measurement overhead is recorded.

Actual contact, contacting gripper, first true close, acquisition, failed grasp, retention, release and drop remain **unknown**. The installed cloth view exposes particle positions/velocities but no directly used, validated cloth-to-gripper contact stream. Rigid-body contact APIs were not assumed to measure particle cloth. Gripper aperture/contact calibration was not changed for this experiment. Decreasing joint-target events are explicitly labeled direction proxies, not calibrated close commands. Link-origin proximity is not surface contact.

Failure analysis conservatively reports geometric success then unfolding, no approach supported by the proximity proxy, or terminal partial-fold N/4. It does not invent wrong-region grasps, drops or over-pulling from insufficient evidence. Infrastructure-invalid and missing runs receive no policy verdict.

## Automatic analysis and diagnostics

CPU job **21384190**, dependency **afterany:21384164**, completed successfully. It analyzed all 24 completed results. No additional GPU diagnostic jobs were submitted. Existing five-smoke offline timelines/plots have already been generated.

The analysis produces per-pose H50/H10/H5 tables, matched win/loss and metric deltas, episode CSV, condition timeline CSV/PNG, boundary vs nonboundary target jumps, inference/runtime summaries, gripper proximity evidence, and wrist/top camera changes. Camera changes during p95 lift >2 cm are a manipulation proxy, not a proof that the policy uses visual correction.

Outputs:

- `outputs/<manifest task id>/`: request, worker status, simulator log, streamed behavior, final result and learned-policy media.
- `analysis/pilot-results/REPORT.md`: final matched comparison, written by the dependent CPU job.
- `analysis/pilot-results/summary.json`: detailed per-pose metrics and matched deltas.
- `analysis/pilot-results/episodes.csv` and `*_conditions.csv/png`: tabular and visual timelines.
- `analysis/pilot-progress/`: separately generated intermediate analysis, never a completed comparison.
- [Submission receipts](submissions.jsonl) and [job ledger](audit/job_ledger.json).

## Failure instrumentation result

The requested post-pilot instrumentation is complete in [`analysis/failure-instrumentation/`](analysis/failure-instrumentation/). It contains 14,400 raw per-action simulator/policy records in `events.jsonl`, paired H50/H10/H5 measurements in `matched-comparison.csv` and `matched-comparison.json`, conservative episode stages in `episode-summary.json`, and synchronized pose 1/3/5/7 SVG diagnostics in `plots/`. Raw simulator positions, measured pre-action joints, action targets, condition values, cloth particle positions and replan boundaries are kept separate from derived 5 cm/3 cm proximity and motion-coupling proxies.

The audit confirms that H50 predicts 50 rows and executes all 50 before re-observation; H10 and H5 still predict 50 rows but execute 10 or 5 leading rows. There is no overlap or soft inpainting in this runner. Every H10 matched trajectory first diverges from its independent H50 action sequence at action 11, and every H5 trajectory at action 6, the first shorter-horizon replan. Aggregate boundary jumps are not larger for shorter horizons, so the evidence supports fresh-chunk divergence but is inconclusive between action-boundary mechanics, observation-conditioned policy decisions and inference-time sampling. Native particle-cloth contact/grasp signals were not available in the frozen scene; contact, acquisition, retention and drop remain unknown.

No follow-up pilot, training, RL, AWR, RECAP or DAgger job was launched after the diagnosis. The recommended next step is a small paired inference-time blending diagnostic on poses 1, 3, 5 and 7, retaining H50 as baseline and adding validated contact sensing if exposed by the simulator.

## Final pilot result

The 24/24 rollout results and dependent analysis are complete. Every rollout was infrastructure-valid; no missing result was counted as a policy failure.

| Execution horizon | Valid rollouts | Official ever-success | Settled terminal success | Mean best conditions | Mean terminal conditions |
|---:|---:|---:|---:|---:|---:|
| 50 | 8 | 4/8 | 4/8 | 3.375 | 3.000 |
| 10 | 8 | 0/8 | 0/8 | 2.875 | 2.375 |
| 5 | 8 | 0/8 | 0/8 | 2.750 | 2.000 |

The four H50 successes were poses 1, 3, 5 and 7, all with local demonstration pose key 1. Each reached 4/4 conditions and remained successful after settling. H10 and H5 lost all four corresponding H50 successes, and neither shorter horizon produced a new success. At matched pose level, H10 had 0/8 terminal wins and 4/8 losses against H50; H5 had 0/8 wins and 4/8 losses.

Shorter replanning did not improve this checkpoint. Mean worker wall time was approximately 296.8 s for H50, 310.2 s for H10 and 329.4 s for H5. Mean policy inference time rose from 5.04 s to 22.79 s to 42.91 s. Mean terminal cloth displacement was 0.1079 m, 0.0995 m and 0.1051 m respectively. Mean replan-boundary action jumps were 0.268, 0.230 and 0.245 radians; the observed differences do not support a claim that boundary discontinuity alone explains the shorter-horizon failures.

Failure labels were conservative: 10 `never_approached_cloth_proxy`, 3 `partial_fold_1_of_4`, 3 `partial_fold_2_of_4`, 3 `partial_fold_3_of_4`, and 1 `success_then_unfolded`. These labels use geometry and named gripper/jaw-origin proximity. Actual contact, grasp acquisition, retention and drop remain unmeasured.

The full matched table, per-pose deltas, every condition margin, condition timelines, camera-change summaries, replan-boundary statistics and result hashes are in [analysis/pilot-results/REPORT.md](analysis/pilot-results/REPORT.md) and [summary.json](analysis/pilot-results/summary.json). This is an exploratory result under the repaired scorer and cannot be compared directly with historical 8/24.

The completed diagnosis is in [analysis/failure-instrumentation/REPORT.md](analysis/failure-instrumentation/REPORT.md). Preserve H50 as the baseline, do not freeze H10 or H5 for formal evaluation, and do not start RECAP/AWR or new BC training from this pilot alone.

```bash
squeue -j 21384164,21384190 -o "%.18i %.12P %.24j %.10T %.12M %.30R"
sacct -j 21384164,21384190 --format=JobID,State,ExitCode,Elapsed -X
find campaigns/20260921-horizon-pilot/outputs -name status.json -print -exec cat {} \;
cat campaigns/20260921-horizon-pilot/analysis/pilot-results/REPORT.md
```

## First live pilot result

At 2026-09-21 21:30 UTC, array task 1 (slot 0, H10) completed 600 actions and 60 terminal settling steps with 601 fresh camera acquisitions, 60 replans, 59 replan-boundary samples and 600 streamed behavior records. The worker accepted the telemetry and all infrastructure checks. Official ever and settled terminal success were false. This intermediate note is superseded by the completed 24-rollout table above.
