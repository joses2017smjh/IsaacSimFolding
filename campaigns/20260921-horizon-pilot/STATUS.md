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

## Replan causality diagnostic

The action-5/action-10 causal diagnostic is complete for the four matched diagnostic cases previously successful under H50 (pose slots 1, 3, 5 and 7). The runner snapshots simulator cloth particles and velocities, arm joints/targets, cameras, counters and RNG state, then compares a cached H50 suffix, a fresh H5/H10 replan and repeated same-observation predictions. The implementation is in `scripts/render/policy_rollout51.py`; the GPU captures were produced by diagnostic array **21386674** plus the earlier validation captures.

The snapshot restore is exact at the boundary (zero immediate cloth and joint RMS in all eight rows). Same-observation predictions with the same RNG state are bitwise identical. Changing the Torch inference seed changes actions by roughly 2.46–3.37 radians at maximum pairwise difference. A cached suffix reaches 4/4 while a fresh branch reaches 3/4 for pose 3 at both boundaries and pose 1 at action 10. These are the supported paired separations. Pose 1 at action 5 and both boundaries for poses 5 and 7 are inconclusive because the cached replay does not reproduce the complete success or accumulates deformable-simulation replay drift.

The evidence supports fresh observation-conditioned replanning and inference sampling as contributors to the shorter-horizon divergence. It does not isolate a cloth contact/grasp mechanism: no validated native finger/cloth contact signal was available, so acquisition, retention, release and drop remain unknown. A repository audit found no compatible `actions_to_keep`, `actions_to_execute` or soft-inpainting implementation in the checked reference checkout; no continuity intervention was launched.

Deliverables are in [`analysis/replan-causality/`](analysis/replan-causality/): `REPORT.md`, `README.md`, `branch-results.csv`, `action-plan-comparison.json`, `reference-inference-audit.md` and one SVG plot per pose. Preserve H50 as baseline. The single recommended next experiment is a paired inference-only fixed-RNG diagnostic on the same four poses; do not start training, RL, RECAP, AWR, DAgger or BC from this result.

## Fixed-RNG replanning diagnostic

The four-pose fixed-RNG array **21387703** completed successfully. The runner stored Torch/CUDA generator bytes immediately after the frozen episode seed, reconstructed the original H50 first prediction, ran a same-RNG first-replan branch at actions 5 and 10, and generated the preregistered seed panel 101/102/103/104. The reconstructed 50-action command sequence matches the recorded original H50 action stream exactly for every pose (50 actions compared, maximum absolute error 0).

Same-RNG fresh replanning still reaches only 3/4 while cached H50 reaches 4/4 for pose 3 at both boundaries and for pose 1 at action 5. Normal fresh and same-RNG fresh outcomes match in all eight rows. This means a changed inference seed is not necessary for the supported replanning failures; observation-conditioned replanning or loss of previous-plan context is now the leading mechanism. The seed panel shows substantial plan variation, but it was descriptive and not outcome-tuned. Pose 1/action 10 is inconclusive because its cached replay did not reach 4/4 in this run; poses 5 and 7 remain descriptive because deformable replay drift limits causal claims.

Deliverables are in [`analysis/fixed-rng-replanning/`](analysis/fixed-rng-replanning/): `REPORT.md`, `README.md`, `plan-comparisons.csv`, `seed-panel.json`, `rng-audit.md` and the aggregation script. No conditional full rollout or training was launched. The next recommended experiment is a small pose-3 temporal-continuity diagnostic using a reference-compatible retained-prefix/previous-plan mechanism, if one can be identified.

## Pose-3 RTC temporal-continuity diagnostic

The minimal inference-only RTC diagnostic ran on pose 3 at action-5 and action-10 snapshots under Slurm job **21396459**. Job **21396457** failed before simulator startup from an argument-ordering error and produced no data. The completed run used the installed LeRobot SmolVLA RTCProcessor directly: execution horizon 10, EXP prefix schedule, guidance weight 10, and offline inference delay 0. It supplied the unconsumed raw H50 chunk as previous-plan context and carried each generated raw suffix into the next replan.

RTC with no previous chunk exactly reproduced ordinary inference: raw/postprocessed maximum absolute differences were zero, post-call RNG digests matched, and the simulator did not step. With previous-plan guidance, cached H50 reached 4/4 at both boundaries, same-RNG fresh reached 3/4 at both, RTC reached 2/4 at action 5 and 3/4 at action 10. Neither boundary recovered the preregistered criterion, so the conditional full validation on poses 1, 3, 5 and 7 was not submitted.

Results are in [`analysis/rtc-continuity/`](analysis/rtc-continuity/): `REPORT.md`, `README.md` and `comparison.csv`. No environment upgrade, training, RL, RECAP, AWR, DAgger or finetuning was launched.

## Hard retained-prefix queue diagnostic

The active-controller timing audit is now documented in [`analysis/horizon-semantics.md`](analysis/horizon-semantics.md). `policy_rollout51.py` performs policy inference before the following `env.step()`; the diagnostic records simulator and episode counters around every prediction and fails closed if either advances. In the completed pose-3 capture, all **918** prediction records had zero simulator-step delta and zero episode-length delta. This verifies that inference wall time does not consume simulator actions.

The deterministic hard queue diagnostic ran under Slurm job **21397662** on only development pose slot 3, using the exact action-5 and action-10 in-process snapshots, cached H50 and same-reconstructed-RNG fresh H10 controls. RTC guidance was disabled. For delays `d={2,5,10}`, each H10 handoff retained exactly `d` actions from the current previous-plan remainder, discarded fresh rows `[0:d]`, executed fresh rows `[d:10]`, and carried fresh rows `[10:50]` into the next replan. Snapshot restore was zero joint/cloth RMS at both boundaries; first fresh chunks matched same-RNG fresh exactly; all retained-action identity, fresh-prefix discard, suffix-indexing and zero-step prediction checks passed.

| delay | action-5 settled | action-10 settled | both-boundary gate |
|---:|---:|---:|:---:|
| 2 | 2/4 | 2/4 | fail |
| 5 | 3/4 | 2/4 | fail |
| 10 | 2/4 | 3/4 | fail |

Condition trajectories, settled outcomes, divergence from cached H50 at local actions 2/10/20, first-ten action differences, provenance and the raw gate are in [`analysis/queue-continuity/`](analysis/queue-continuity/). No delay passed both pose-3 boundaries, so the smallest passing delay is none and the conditional pose 1/3/5/7 validation was **not submitted**. Stop and recommend an observation-component ablation rather than another continuity intervention. No training, RL, RECAP, AWR, DAgger, finetuning, environment upgrade or broad rollout was launched.

## Observation-component causal diagnostic

The follow-up implementation is in [`analysis/observation-components/`](analysis/observation-components/) and is restricted to development pose slot 3. It preserves the validated action-5/action-10 snapshots, cached H50 suffix and same-reconstructed-RNG fresh H10 controls. The active observation keys are `observation.state`, `top_rgb`, `left_rgb`, `right_rgb`, and the unchanged task. The original action-0 H50 observation is retained in the raw provenance.

The diagnostic evaluates current-all, old-all-images/current-state, current-images/old-state, old-top-only, old-left-only, old-right-only and old-both-wrists before executing any counterfactual. Stale and hybrid inputs are labeled attribution probes, not deployable controllers. The repository’s existing passive `model.embed_prefix` feature tap is used only when available and returns the original forward output unchanged.

The preregistered ranking gate is a minimum **0.05 rad** reduction in cached-H50-suffix first-10 RMS at both boundaries relative to same-RNG current-all. At most the best two probes may execute; each would need settled 4/4 at both boundaries before any larger validation. No broad pose validation is authorized by this request, so no validation submission is made. If no candidate meets the two-boundary settled criterion, stop inference interventions and recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment. The Slurm job, result hashes, timing/restoration checks, per-joint deltas, first-10 action differences and condition trajectories are recorded under `analysis/observation-components/`.

The diagnostic completed under Slurm job **21398036** in **15:30** with exit `0:0` (the earlier path-only submission **21398016** failed before simulator startup and is recorded as superseded). All **576** prediction timing records had zero simulator-step and episode-counter deltas. Both boundaries passed snapshot restoration, exact reconstructed RNG restoration, and current-all/same-RNG identity checks; the passive `model.embed_prefix` feature hook was available.

Only `current-images-old-state` met the preregistered first-10 RMS improvement gate, improving by **0.067168 rad** at action 5 and **0.178181 rad** at action 10. It was executed as the sole candidate, but settled at **2/4** at both boundaries, so the two-boundary 4/4 gate failed. No larger validation was submitted. Stop inference interventions and recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment.

## Boundary-state successful-H50 replay micro-fine-tune diagnostic

Starting from commit `7e23b0c`, this diagnostic used only the validated pose-3 `Pant_Short_Seen_3` H50 success. Replay-only Slurm job **21398513** captured exactly two real post-action observations (after actions 5 and 10): all three current RGB cameras, current 12-DoF state, unchanged task, exact cached H50 suffix targets `[5:50]` and `[10:50]`, and reconstructed Torch/CUDA RNG bytes. Assertions passed for observation/action alignment, target indices and equality, checkpoint SHA-256, normalization files, garment/pose identity and source successful-H50 episode. The cached H50 source settled at **4/4**.

The one fixed-seed micro-fine-tune was Slurm job **21398555**: seed 4242, 300 steps, batch size 2, learning rate 1e-5, and 99,880,992 trainable action/state-path parameters with the large VLM/vision path frozen. The original checkpoint remained read-only. Fixed held-out raster loss worsened from **0.133254** to **0.213044**, so the held-out no-regression gate failed.

Snapshot-only baseline/trained inference was completed by job **21399086** after two loader-plumbing retries. For both boundaries, exact simulator restore, exact same-RNG restoration, same-RNG pairing and zero simulator advancement during prediction passed. Trained first-10 RMS deviation from the cached suffix fell from **0.106624 to 0.028604** at action 5 (73.17%) and **0.295612 to 0.030279** at action 10 (89.76%), so the 50% inference gate passed at both boundaries. The combined gate failed because held-out behavior regressed. No trained closed-loop H10 execution was launched; therefore no trained settled-4/4 result, larger pose 1/3/5/7 boundary dataset, mixed fine-tune launch, or validation submission was authorized.

The result is consistent with optimization/generalization overfit to two corrective examples, not target plumbing, alignment, normalization, or insufficient action/state-path capacity. Stop this intervention and recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment with a held-out-preserving objective. Full report and machine-readable provenance are in [`analysis/boundary-replay/REPORT.md`](analysis/boundary-replay/REPORT.md). Jobs, failed retries, controls, source/checkpoint hashes, first-10 per-joint differences and the cached-H50 condition trajectory are in `analysis/boundary-replay/`.

## Mixed corrective/raster replay checkpoint diagnostic

Continuing from commit `413e4e0`, the retained-checkpoint audit found no intermediate checkpoints from the prior run. The only retained 300-step boundary-only checkpoint was re-evaluated under the exact pose-3 action-5/action-10 snapshot protocol: it retained the prior 73.17%/89.76% first-10 RMS reductions but failed the unchanged held-out raster gate.

The fixed mixed replay then started from the untouched baseline checkpoint under Slurm job **21399191**. It used seed `4242`, exactly 300 optimizer steps, one corrective boundary example plus four fixed frames from four disjoint `storm_capture/` raster-training files per batch (20%/80%), the same 1e-5 learning rate, unchanged normalization and the same state/action trainable path, with the vision/VLM frozen. Full checkpoints were saved at steps 25 through 300. The exact held-out set remained out of training; all twelve checkpoints passed raster retention.

The snapshot-only panel was Slurm job **21399222**. Every checkpoint passed exact snapshot restore, reconstructed same-RNG pairing and zero simulator advancement. The earliest unchanged combined-gate pass was `step_000150`: boundary-5 first-10 RMS reduction **50.546%**, boundary-10 **65.899%**, and held-out loss **0.079753** versus baseline **0.133254**. The complete boundary-versus-retention Pareto trajectory is in [`analysis/mixed-replay/REPORT.md`](analysis/mixed-replay/REPORT.md), with CSV/JSON metrics and checkpoint provenance.

The selected `step_000150` checkpoint was run under Slurm job **21399265** from the exact two snapshots using ordinary fresh H10 replanning only. The baseline cached H50 stream was used only to reconstruct the snapshots; no cached H50 actions, RTC, retained-prefix queue, stale/hybrid observation or other inference modification was used in either branch. Both 120-action branches reached at most and settled at **2/4**, so the required two-boundary settled `4/4` gate failed. Stop and do not launch larger pose validation or construct the larger pose 1/3/5/7 boundary dataset. The next recommendation is parameter-anchored or parameter-efficient adaptation rather than increasing replay duration.

The complete Slurm ledger is [`analysis/mixed-replay/slurm-jobs.json`](analysis/mixed-replay/slurm-jobs.json); selected execution trajectories are in `analysis/mixed-replay/condition-trajectories.csv`, and the original baseline checkpoint remains untouched.

## Existing-checkpoint closed-loop ceiling diagnostic

Continuing from commit `d6512952fd1d5e51145965b0517043f9a21a7d25`, the minimal ceiling diagnostic compared the retained 300-step boundary-only checkpoint and mixed-replay `step_000300` from the exact validated pose-3 action-5/action-10 snapshots. Fresh branches used ordinary H10 replanning only. Cached H50 was preserved as a control and snapshot reconstruction source; no RTC, retained-prefix queue, stale/hybrid observation, cached-H50 execution assistance or new training was used in fresh branches.

The completed diagnostic is Slurm job **21399454**. Cached H50 settled **4/4 at both boundaries**. Baseline fresh H10, boundary-only 300 and mixed step300 all failed the settled gate at both boundaries; both trained branches reached at most **2/4**. The boundary-only checkpoint remains a diagnostic upper bound only because its held-out retention gate had already failed. Exact snapshot restoration passed, the reconstructed RNG stream was invariant across baseline and both checkpoints at every H10 prediction, and all fresh predictions had zero simulator-step and episode-counter deltas.

| controller | action 5 | action 10 | both-boundary settled gate |
|---|---:|---:|:---:|
| cached H50 | 4/4 | 4/4 | pass |
| baseline fresh H10 | 3/4 terminal | 2/4 terminal | fail |
| boundary-only 300 | 2/4 terminal | 2/4 terminal | fail |
| mixed step300 | 2/4 terminal | 2/4 terminal | fail |

The decision is to **stop static boundary-suffix adaptation** and not launch PEFT or parameter anchoring. The next recommended diagnostic is a small on-policy corrective-data experiment: roll the selected H10 student from the exact snapshots, capture the exact simulator snapshot and observation at every subsequent H10 replan state, query a fresh H50 plan from that actual student-visited state, branch-execute each H50 candidate, and retain labels only when that branch demonstrably recovers the target condition. Do not reuse the original time-aligned cached H50 suffix after student-state divergence.

Raw output, condition trajectories, settled outcomes, provenance and the complete job ledger are in [`analysis/boundary-ceiling/`](analysis/boundary-ceiling/). Jobs **21399451** and **21399452** are recorded as non-result wrapper/diagnostic failures; neither launched training or produced a policy result. No poses 1/3/5/7 validation was submitted.

## On-policy H50 rescue-oracle diagnostic

Starting from commit `2ed1deb`, the inference-only rescue-oracle audit used the untouched original baseline checkpoint and the exact validated pose-3 action-5/action-10 snapshots. From each snapshot, ordinary fresh H10 replanning reached actual student-visited roots after **+10, +20 and +30** executed actions. Each of the six roots was restored exactly and branched into normal H10 continuation versus one deterministic fresh H50 chunk executed open-loop without replanning.

| initial boundary | student root | H10 terminal | fresh H50 terminal | H50 rescue |
|---:|---:|---:|---:|:---:|
| 5 | +10 | 2/4 | 2/4 | no |
| 5 | +20 | 1/4 | 2/4 | no |
| 5 | +30 | 1/4 | 2/4 | no |
| 10 | +10 | 2/4 | 2/4 | no |
| 10 | +20 | 2/4 | 2/4 | no |
| 10 | +30 | 2/4 | 2/4 | no |

All six roots had exact restoration/equivalence, exact reconstructed RNG restoration, zero simulator advancement during prediction, and identical H10-root/H50-generated chunks. Fresh H50 recovered **0/6** roots and produced **0** validated intermediate observation→unconsumed-plan-suffix examples at offsets 10/20/30/40. The reliable all-root recovery gate therefore failed.

Do not launch self-distillation or another static suffix intervention. The next recovery source should be validated simulator DAgger/teleoperation, a privileged scripted recovery if available, or a separately preregistered branch-evaluated candidate search. No training, PEFT/LoRA, RTC, broad DAgger collection, or multi-pose validation was launched.

The completed audit is Slurm job **21399500**. Root snapshots/observations are in `analysis/onpolicy-h50-oracle/student-roots.npz`; successful-H50 example storage is `successful-h50-examples.npz` and is empty by gate; raw outcomes, trajectories, CSV summaries and provenance are in [`analysis/onpolicy-h50-oracle/`](analysis/onpolicy-h50-oracle/).
