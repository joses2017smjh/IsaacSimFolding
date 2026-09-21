> Latest: [21 September verified execution status](FOLDING_STATUS_2026-09-21.md).

> Continuation: actual asset verification, repairs and Slurm submissions are recorded in [FOLDING_EXECUTION_2026-09-20.md](FOLDING_EXECUTION_2026-09-20.md). This earlier assessment used documentation only.

# Folding improvement assessment — 20 September 2026

**No new improvement is demonstrated. This session completed a documentation-based assessment and an experiment plan; it did not execute the requested research campaign.** The user's restriction to this repository's `.md`, `.txt`, and NOTICE files was applied literally. Code, result JSON, dataset metadata, meshes, checkpoints, and video contents were not inspected. Consequently, this report does not certify a true baseline, a complete asset inventory, infrastructure fixes, or new policy performance.

The attachment requests primary-artifact audits, code repairs, tests, and GPU experiments that require those additional file types. Those parts remain unperformed. Repository documents describe earlier work outside this checkout; their links were not followed into sibling repositories. No external websites were consulted. Git and read-only Slurm queries were performed because the attachment explicitly requires them. Existing media paths were checked for file existence only.

## Repository and current jobs

- Requested directory: `/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/lehome-fold-repro`.
- Resolved Git root: `/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro` — the same checkout through the filesystem alias.
- Remote: `https://github.com/joses2017smjh/IsaacSimFolding.git`.
- Branch: `main`; commit: `34a424c0a0a1e1d52cbd43beca4a4ad24738bc3b`.
- Pre-existing modified `results/stage4_thompson.json`, untracked session notes, and campaign outputs were preserved. No source, checkpoint, historical result, README, NOTICE, or existing campaign was changed.

At **2026-09-21 00:15:19 UTC / 2026-09-20 17:15:19 PDT**, `squeue` showed only `21369598` (`ood-advanced`) and `21369796` (`m7-learn-both-s0`) running. Neither is identified as a folding job. No folding jobs were running or pending in that snapshot. No jobs were submitted, cancelled, requeued, released, or otherwise modified by this session.

The initial sandboxed scheduler queries could not create sockets; read-only queries succeeded with the required sandbox escalation. The [scheduler snapshot](../campaigns/20260920-folding-improvement-audit/scheduler_snapshot.txt) records the final commands and results.

## Evidence and deliverables

The most relevant sources are [SESSION_STATUS.md](../SESSION_STATUS.md), the [September 19 audit](../campaigns/20260919-media/docs/AUDIT_2026-09-19.md), [STAGE0](STAGE0.md), the [completed media gate](MEDIA_GATE_2026-09-20.md), and the [latest media status](../campaigns/20260920-media-root-fix/MEDIA_STATUS.md). Older README, PLAN, and job-ledger statements are historical where these later records explicitly supersede them.

Machine-readable outputs:

- [Experiment manifest](../campaigns/20260920-folding-improvement-audit/experiment_manifest.json): 12 baseline rows, 24 planned horizon rollouts, gate requirements, deferred phases, and source hashes.
- [Partial garment inventory](../campaigns/20260920-folding-improvement-audit/garment_inventory.json): identities explicitly named by the selected documents, source lines, documented attributes, and nulls for unavailable facts.
- [Clip manifest](../campaigns/20260920-folding-improvement-audit/clip_manifest.json): six documented policy references and eight separate replay controls, with provenance gaps visible.

`null` means unknown, not zero or failure. Every experiment gate remains `passed: false` / `not_evaluated`. The manifest is a plan, not a runnable submission configuration or experiment result.

## Garments and demonstrations

The following counts are **documented historical counts**, not a new filesystem inventory. [STAGE0](STAGE0.md) reports counting the released asset listing and reading demonstration metadata.

| Benchmark class | Seen | Public Unseen | Total | Demonstrations |
|---|---:|---:|---:|---:|
| Pant_Short | 10 | 2 | 12 | 250 |
| Pant_Long | 10 | 2 | 12 | 250 |
| Top_Short | 10 | 2 | 12 | 250 |
| Top_Long | 10 | 2 | 12 | 250 |
| Total | 40 | 8 | 48 | 1,000 |

The documents report 265,798 demonstration frames and 25 demonstrations per Seen garment. They describe no demonstration outcome column; successful original demonstrations must not be conflated with successful simulator replays.

The documented explanation for **48 versus 80** is that `Release` contains 10 Seen plus 2 public Unseen garments per class, while eight additional private `Holdout` garments per class were not included in that release: `4 × 8 = 32`. This establishes what the repository records about that release, not current upstream availability. No missing assets were downloaded.

The selected documentation explicitly names **12 unique garment identities**, all Seen. The inventory records those 12; it does not manufacture the other 36 names or claim to enumerate any public Unseen identities. It includes the documented mesh/particle counts for four identities: Pant_Short_Seen_0 (11,573 / 11,385), Top_Long_Seen_0 (14,746 / 14,544), Top_Short_Seen_1 (9,774 / 9,774), and Top_Long_Seen_1 (10,410 / 10,410). These values come from README's historical UV-seam diagnosis, not a new mesh inspection. Mesh paths, most particle/vertex counts, colors, materials, and most per-identity capture counts remain unknown.

**Sweaters:** the documented benchmark has the four classes above and no explicit `Sweater` class. No visual inspection was performed, so no particular Top_Long asset is certified as sweater-like. A later visual inspection may justify the caption “long-sleeve top / sweater-like garment,” while retaining `Top_Long` as its class.

## Successful captures and leakage

The [September 19 audit](../campaigns/20260919-media/docs/AUDIT_2026-09-19.md) reports these counts from the 91 historical capture records and the newer production split:

| Class | Successful captures | Failed captures | Adaptation train episodes | Validation episodes | Adaptation-held-out garment |
|---|---:|---:|---:|---:|---|
| Pant_Short | 24 | 1 | 21 | 3 | Pant_Short_Seen_9 |
| Pant_Long | 3 | 21 | 2 | 1 | Pant_Long_Seen_5 |
| Top_Short | 11 | 7 | 8 | 3 | Top_Short_Seen_7 |
| Top_Long | 17 | 7 | 16 | 1 | Top_Long_Seen_9 |
| Total | 55 | 36 | 47 | 8 | Four Seen identities |

The original raster adaptation reportedly trained on successful and failed replays and split adjacent frames randomly. The newer adaptation excluded the 36 failed replays, balanced classes, and held out entire garment identities. The audit records production split SHA-256 `16fda1afe6eef9f89c7db1a294238718cd7ed021d68e63bded353d91c306a1d6`; it was not recalculated here.

| Cohort | Base pretraining | Newer adaptation training | Adaptation validation | Evaluation interpretation |
|---|---|---|---|---|
| Four named Seen holdouts | Documented as available to pretraining | Excluded | Included | Adaptation holdouts; not wholly unseen |
| Other Seen garments | Documented as available to pretraining | Exact participating identities unavailable here | Excluded by documented identity split | Official Seen |
| Eight public Unseen garments | Documented excluded | Documented excluded | Not the four adaptation holdouts | Stronger official generalization test |

This is a leakage assessment of documentary claims. Frame manifests, full episode identities, and actual training inputs still need verification. Class balancing cannot turn two long-pants episodes into broad long-pants coverage.

## Baseline ledger

The two populated rows below come from [SESSION_STATUS.md](../SESSION_STATUS.md). Seen counts are arithmetic derived from total minus the documented Unseen counts. Primary result JSON and camera audits were not read, so this table is **not an independently verified baseline**.

| Policy | Class | Seen success/episodes | Unseen success/episodes | Completed episodes | Ever-success | Terminal success | Infrastructure status |
|---|---|---|---|---|---|---|---|
| Historical raster baseline | Pant_Short | 8/20 | 0/4 | 24 | 8/24 | Unknown | Completion reported; validity not rechecked |
| Historical raster baseline | Pant_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Historical raster baseline | Top_Short | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Historical raster baseline | Top_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 0 | Pant_Short | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 0 | Pant_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 0 | Top_Short | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 0 | Top_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 1 | Pant_Short | 3/20 | 0/4 | 24 | 3/24 | Unknown | Completion reported; validity not rechecked |
| Adapt seed 1 | Pant_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 1 | Top_Short | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |
| Adapt seed 1 | Top_Long | Unknown | Unknown | Unknown | Unknown | Unknown | Class detail unavailable in allowed sources |

The documented short-pants rates are 33.3% and 12.5%; these do not support an improvement. The observed Seen-to-Unseen gap is 8/20 versus 0/4 for the baseline and 3/20 versus 0/4 for seed 1. Four Unseen episodes are too few for a precise generalization estimate.

Do not pool these with the older 2/8 short-pants pose subset, older 0/8 or 0/16 BC cohorts, or demonstration replay results. In particular, the September 19 audit invalidates job `21214241`'s old 6/24 visual-policy baseline because of swallowed rendering errors and stale observations. The older ledger still calls that run valid; the later audit supersedes that interpretation.

The latest media gate is a separate development example: adaptation seed 0 on Top_Short_Seen_0, recorded pose 2, 600 actions and 601 validated render calls, **ever false / terminal false**. It is not an extra episode to add to a strict evaluation denominator. The documents do not establish a terminal-settling protocol for the historical successes.

## Why historical evaluations are incomplete

Fresh accounting confirms **5 completed, 2 failed, and 5 timed-out Slurm tasks** in the 12-cell evaluation campaign. Scheduler completion is not independently verified experiment completion. The permitted documents do not establish the class-to-array-index mapping, so it is not guessed here.

| Policy job | Task 0 | Task 1 | Task 2 | Task 3 |
|---|---|---|---|---|
| 21359573, baseline | Completed | Failed, 1:0, 20m22s | Timeout, 6h00m19s | Timeout, 6h00m06s |
| 21359574, adaptation seed 0 | Timeout, 6h00m23s | Failed, 1:0, 27m03s | Completed | Completed |
| 21359575, adaptation seed 1 | Completed | Timeout, 6h00m03s | Timeout, 6h00m07s | Completed |

The documentation names a scorer-index error and a garment-switch cleanup stall as blockers. Without per-task logs, it does **not** establish which exact operation caused each failed or timed-out task. Valid completed episodes inside interrupted tasks must be recovered separately; their absence is not zero policy success.

Media accounting updates the earlier notes: `21360435` failed because of the inferred robot asset path; `21367674` failed after the robot repair exposed the same asset-root problem for the scene; `21367715` completed. The original dependent array `21360436` and report `21360437` are now **CANCELLED** in accounting, rather than still waiting as the historical notes say. This session did not cancel them and did not determine who or what did.

## Infrastructure diagnosis and repair requirements

**Scorer:** SESSION_STATUS records index 11029 against 10,869 particles. This is out of bounds, but that fact alone does not identify whether the index belongs to a different mesh, render vertices, a stale garment, or mismatched scorer/assets. The attachment names Top_Short_Seen_3; the allowed repository notes do not independently establish that identity. The scorer's documented entry point is `success_checker_garment_fold`, in upstream `source/lehome/lehome/utils/success_checker_chanllege.py`.

The historical render-vertex/particle deduplication fix is not evidence that the same transformation repairs scorer keypoints. Before GPU evaluation, resolve every Release mesh and every scorer reference, record both ordering conventions and asset/scorer hashes, and prove correspondence of each landmark. Any correction must preserve the five geometric predicates. Clamping, deleting a predicate, skipping an affected garment, or treating an exception as a failed fold is unacceptable. No scorer repair or preflight implementation was made here.

**Garment switching:** the last printed message “Old garment object deleted” does not identify the blocking call. The older `app.close()` shutdown hang and stale camera-layer bugs are distinct documented problems; neither proves the cause of the current switch stall. Timestamp entry and exit around cloth removal, USD cleanup, particle-system cleanup, physics-handle release, observer retargeting, scene synchronization, garbage collection, garment creation, physics initialization, camera reattachment, and first render. Capture a stack if a bounded operation stalls. Compare the same garments in one process versus one fresh Isaac process per garment. Prefer isolation if the comparison demonstrates reliability. No such comparison ran here.

**Previously implemented repairs, documented but not reimplemented here:** GPU cloth dynamics; Storm rendering without the RTX launch path; wrist-camera attachment and coordinate convention; UV-seam vertex mapping; camera-retargeting and error propagation; corrected action-chunk capture; explicit robot and scene asset roots. The completed asset-root gate demonstrates one rendered episode, not four-class scorer or switch reliability.

**Required episode validity:** record simulation step and render identifiers, acquisition freshness for every consumed camera view, RGB validity, garment visibility, finite robot and particle state, particle motion, and scorer outcomes. A render-call count or changing pixel hash alone cannot certify freshness. Record infrastructure-invalid episodes separately, with no policy-failure denominator entry.

**Required terminal evaluation:** keep the official latched ever-success result, first success action/time, and persistence trajectory. After a fixed terminal settling period, re-evaluate the geometric predicates on current particles without reusing the latch. Save all distances, operators, thresholds, margins, and conditions. Historical latched successes remain terminal-unknown.

## Controlled experiment matrix — planned, not run

| Phase | Experiment | Size / gate | Current status |
|---|---|---|---|
| A | Full inventory, scorer preflight, leakage audit, required tests | Every available garment | Not run; documentary assessment only |
| B | Fresh-process single-garment Isaac smoke | One garment per class; all observation and terminal checks | Not run |
| C | Fixed-checkpoint execution horizons 50 / 10 / 5 | Eight identical pose/seed pairs across multiple short-pants garments: 24 rollouts | Plan recorded; IDs, poses, seeds and hashes not frozen |
| D | Default versus one globally selected shorter horizon | Conditional plan: 12 garments × 2 poses × 2 seeds × 2 horizons = 96 rollouts | Blocked until A–C pass |
| Transfer screen | Matched small screen on all four classes | Same frozen candidate and protocol | Conditional; no candidate selected |
| E | Ordinary versus grasp-focused sampling | Same class distribution, checkpoint, optimizer, LR, steps, split; two seeds if promising | Deferred |
| F | Historical raster baseline versus best new candidate | Matched four-class, Seen/Unseen protocol | Deferred; no new candidate exists |

For Phase C the predicted action chunk stays **50**, candidate count stays **1**, and the proposed budget is **600 executed actions** at every horizon. The existing best raster-adapted checkpoint remains fixed. Replanning must discard unused actions and use newly acquired observations. Freeze checkpoint/preprocessing/camera/scorer/asset hashes, initial-state protocol, timestep, normalization, seeds, physics, and both initial and terminal settling protocols before launch. No extra horizon 3 is planned in this assessment.

Choose development poses before results are observed and reserve independent formal poses/seeds. Compare paired terminal successes first, then ever-success and condition margins, while reporting inference and wall-clock cost. Stop if shorter execution clearly degrades; otherwise freeze one candidate globally before formal evaluation. Report raw matched counts and uncertainty that respects shared garment identities. A one-episode difference is inconclusive.

For each valid rollout save the task, manipulation, geometry, perception, robot, and performance metrics requested in the attachment, including per-condition distance trajectories. Diagnose the earliest supported manipulation failure and keep camera/physics/scorer errors in infrastructure accounting. Relevant documentary hypotheses are missed grasps/hovering, pulling one fold axis apart while satisfying others, and losing a transient success. **No current failure-mode distribution or dominant mechanism was measured here.**

Grasp-focused training is **not justified for launch yet**. The documented data imbalance justifies auditing successful long-pants coverage. After reliable evaluation, compare 50% ordinary samples plus 50% validated pre-grasp/grasp-commitment samples against the existing control, preserving class proportions and whole-episode/garment splits. Additional captures require validated successful trajectories; repeating identical frames is not extra coverage. Failed trajectories without expert corrections are not DAgger.

The two earlier adaptations reportedly completed 1,500 updates, selecting step 1200 at loss 0.062234 for seed 0 and step 1000 at loss 0.059912 for seed 1. Those losses do not establish improved folding. The audit also reports that the legacy candidate score ignores candidate actions and that the RECAP gradient update is unimplemented. RECAP/AWR/Thompson experiments remain deferred. No training ran this session.

## Tests and storage

The completed media-gate document reports **66 historical CPU checks**: 4 asset regressions, 8 camera/episode checks, 3 media-pipeline checks, and 51 pure checks. They were not rerun. The attachment's 15 required test categories are each marked `not_run` in the manifest: scorer indices, asset resolution, exact counts, split mapping, camera freshness, switching/isolation, terminal checking, queue reset, deterministic horizons, whole-episode splits, garment leakage, class balancing, grasp sampling, finite cloth, and media provenance. No expensive job was submitted with these gates unverified.

This session passed **20 consistency checks on the new documentation artifacts**, including 52 source-line references, source hashes, count arithmetic, the paired horizon matrix, explicit unknown results, unpassed experiment gates, and existing media paths. The [validation record](../campaigns/20260920-folding-improvement-audit/validation.json) distinguishes these checks from policy or infrastructure tests. `git diff --check` also passed for tracked changes; the pre-existing result edit was not altered by this session.

Only small text/JSON artifacts were generated in this repository. Before any capture or retraining, measure the home and HPC-share usage using the attachment's `du`/`df` commands. Retaining raw RGB for the development screen would take `24 × 601 × 3 × 480 × 640 × 3 = 39,879,475,200` bytes, about **39.88 GB / 37.14 GiB**, before other data or overhead. This is an estimate, not generated data. Use HPC share or node-local scratch, measure a pilot, and stream MP4 where practical. No datasets or caches were created in home storage.

## Media coverage

Six existing policy references are documented below. Their files exist, but their contents, checkpoint hashes, execution horizons, evaluation seeds, and complete overlay provenance were not audited. These heterogeneous historical examples do not satisfy a new matched eight-clip policy collection.

| Class / identity | Documented policy outcome | GIF | MP4 | Terminal |
|---|---|---|---|---|
| Pant_Short_Seen_0 | Older raster-adapted success | [Historical policy](demo/POLICY_fold_success.gif) | Not established here | Unknown |
| Pant_Short_Seen_0 | Older BC failure | [Failure](gifs/rollout_Pant_Short_Seen_0_policy_failure.gif) | Not established here | Unknown |
| Pant_Long_Seen_0 | Older BC failure | [Failure](gifs/rollout_Pant_Long_Seen_0_policy_failure.gif) | Not established here | Unknown |
| Top_Long_Seen_0 | Older BC failure | [Failure](gifs/rollout_Top_Long_Seen_0_policy_failure.gif) | Not established here | Unknown |
| Top_Short_Seen_0, pose 2 | Adaptation seed 0 failure | [Newer gate GIF](demo/adapt-s0-failure.gif) | [Gate MP4](demo/adapt-s0-failure.mp4) | Documented false |
| Top_Short_Seen_0 | Older BC failure | [Failure](gifs/rollout_Top_Short_Seen_0_policy_failure.gif) | Not established here | Unknown |

There is no verified policy success for tops or long pants in the reviewed documents. A second policy reference for Pant_Long and Top_Long remains missing from this selection. No new videos or GIFs were generated. MP4 should be the authoritative output of the future collection; overlays must include policy/checkpoint, exact garment/class/split, evaluation seed, execution horizon, ever/terminal verdicts, condition count, and first supported failure category.

The eight separately documented replay controls below are **DEMONSTRATION REPLAY — NOT POLICY SUCCESS**. “Success” is their documented historical checker verdict; terminal settled success is unknown. Their existence does not fill a policy-success or policy-coverage slot.

| Class | Replay success reference | Replay failure reference |
|---|---|---|
| Pant_Short | [Seen_0, ep501](../campaigns/20260919-media/reference_replays/rollout_Pant_Short_Seen_0_v2_ep501_replay_ep501_success.gif) | [Seen_0, ep502](../campaigns/20260919-media/reference_replays/rollout_Pant_Short_Seen_0_v2_ep502_replay_ep502_failure.gif) |
| Pant_Long | [Seen_0, ep750](../campaigns/20260919-media/reference_replays/rollout_Pant_Long_Seen_0_replay_ep750_success.gif) | [Seen_0, ep751](../campaigns/20260919-media/reference_replays/rollout_Pant_Long_Seen_0_replay_ep751_failure.gif) |
| Top_Short | [Seen_0, ep2](../campaigns/20260919-media/reference_replays/rollout_Top_Short_Seen_0_replay_ep2_success.gif) | [Seen_0, ep1](../campaigns/20260919-media/reference_replays/rollout_Top_Short_Seen_0_replay_ep1_failure.gif) |
| Top_Long | [Seen_0, ep251](../campaigns/20260919-media/reference_replays/rollout_Top_Long_Seen_0_v2_ep251_replay_ep251_success.gif) | [Seen_0, ep250](../campaigns/20260919-media/reference_replays/rollout_Top_Long_Seen_0_v2_ep250_replay_ep250_failure.gif) |

## Remaining work and commands

The next executable step is Phase A: inspect the actual Release lists/meshes, scorer source and per-episode artifacts, then implement and test the index preflight and fresh-process evaluation. This requires reading files outside the current `.md`/`.txt`/NOTICE restriction. Exact baseline reconstruction, stall diagnosis, new horizon results, terminal-success rates, failure distributions, and eight new representative policy clips remain outstanding. The report and manifests make those omissions explicit rather than supplying assumed values.

From the existing directory, these commands inspect status without submitting or modifying jobs:

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
git remote -v
squeue -u "$USER" -o "%.18i %.12P %.24j %.10T %.12M %.30R %.20b"
sacct -u "$USER" --starttime 2026-09-18 \
  -j 21359573,21359574,21359575,21360435,21360436,21360437,21367674,21367715 \
  --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed -X
cat docs/FOLDING_IMPROVEMENT_2026-09-20.md
cat campaigns/20260920-folding-improvement-audit/scheduler_snapshot.txt
python3 -m json.tool campaigns/20260920-folding-improvement-audit/experiment_manifest.json
python3 -m json.tool campaigns/20260920-folding-improvement-audit/garment_inventory.json
python3 -m json.tool campaigns/20260920-folding-improvement-audit/clip_manifest.json
```

If the source-file restriction is later broadened, the existing gate's primary result can be inspected with `python3 -m json.tool docs/evidence/media-gate-2026-09-20/21367715-rollout.json`. That command is provided for follow-up; its input was not read in this session. Existing gate records should be audited before any new submission, and an exit code of zero must never substitute for an experiment-level validity gate.
