> Current status: [21 September findings, remaining defect, replacement jobs and next steps](FOLDING_STATUS_2026-09-21.md).

> Update: [v4 partition check and unwelded-mesh repair](../campaigns/20260920-folding-pilot-v4/PARTITION_CHECK.md). Corrected smoke jobs 21370102/21370103 are submitted. V3 long-pants subsequently failed its missing-remap guard; the results below retain their recorded v3 provenance.

# Folding execution — 20 September 2026

The continuation authorized primary-artifact inspection and execution. This report supersedes the documentation-only assessment for the findings below. No improved folding result is claimed.

All new work remains in `lehome-fold-repro`, remote `https://github.com/joses2017smjh/IsaacSimFolding.git`, branch `main`, commit `34a424c0a0a1e1d52cbd43beca4a4ad24738bc3b`. Shared folding assets, checkpoints and the historical folding-only result directory were read without modifying their source files. Unrelated repositories, training, and jobs were not changed.

## Verified inventory

[Machine-readable inventory](../campaigns/20260920-folding-pilot-v3/audit/garment_inventory.json) was generated from the actual four Release lists, 48 garment JSONs and USD meshes, demonstration metadata, 91 NPZ capture headers/scalars, production split files, raw result JSONs, and existing media paths. It records mesh/config hashes, authored vertices, statically welded particle counts, six landmark indices and their mapped counterparts, material paths, per-garment demonstrations/captures/splits/results/media. Runtime particle counts for every garment are not yet verified. Demonstration keys in `garment_info.json` are local within each garment, not global merged-dataset episode IDs.

| Class | Exact Seen IDs | Exact public Unseen IDs | Total | Demonstrations |
|---|---|---|---:|---:|
| Pant_Short | Pant_Short_Seen_0, Pant_Short_Seen_1, Pant_Short_Seen_2, Pant_Short_Seen_3, Pant_Short_Seen_4, Pant_Short_Seen_5, Pant_Short_Seen_6, Pant_Short_Seen_7, Pant_Short_Seen_8, Pant_Short_Seen_9 | Pant_Short_Unseen_0, Pant_Short_Unseen_1 | 12 | 250 |
| Pant_Long | Pant_Long_Seen_0, Pant_Long_Seen_1, Pant_Long_Seen_2, Pant_Long_Seen_3, Pant_Long_Seen_4, Pant_Long_Seen_5, Pant_Long_Seen_6, Pant_Long_Seen_7, Pant_Long_Seen_8, Pant_Long_Seen_9 | Pant_Long_Unseen_0, Pant_Long_Unseen_1 | 12 | 250 |
| Top_Short | Top_Short_Seen_0, Top_Short_Seen_1, Top_Short_Seen_2, Top_Short_Seen_3, Top_Short_Seen_4, Top_Short_Seen_5, Top_Short_Seen_6, Top_Short_Seen_7, Top_Short_Seen_8, Top_Short_Seen_9 | Top_Short_Unseen_0, Top_Short_Unseen_1 | 12 | 250 |
| Top_Long | Top_Long_Seen_0, Top_Long_Seen_1, Top_Long_Seen_2, Top_Long_Seen_3, Top_Long_Seen_4, Top_Long_Seen_5, Top_Long_Seen_6, Top_Long_Seen_7, Top_Long_Seen_8, Top_Long_Seen_9 | Top_Long_Unseen_0, Top_Long_Unseen_1 | 12 | 250 |

**Verified totals: 48 garments = 40 Seen + 8 public Unseen; 1,000 demonstrations across the 40 Seen identities, 25 each.** The released local pack supplies 12 per class. The documented leaderboard difference is eight private Holdout garments per class, or 32 total; no missing assets were downloaded or current upstream availability assumed. No explicit Sweater category exists in the four actual class lists. Specific sweater-like appearance has not been visually certified.

| Class | Successful/failed captures | Adaptation train | Validation |
|---|---:|---:|
| Pant_Short | 24/1 | 21 | 3 |
| Pant_Long | 3/21 | 2 | 1 |
| Top_Short | 11/7 | 8 | 3 |
| Top_Long | 17/7 | 16 | 1 |

Both adaptation seeds have the same actual split: 47 training episodes and 8 validation episodes, with disjoint whole episodes and garment identities; no official Unseen identity participates. The per-identity membership is in the inventory. Capture success labels are historical scorer outcomes; labels on garments affected by the landmark bug need corrected revalidation before another training experiment.

## Verified legacy baseline episodes

[Raw-artifact baseline ledger](../campaigns/20260920-folding-pilot-v3/audit/baseline_episodes.json) includes all 12 cells and every returned episode, source hashes, health records, log paths, and interruption evidence. The 8/24 and 3/24 counts are confirmed. They are native official ever-triggered outcomes under the legacy landmark selection, **not terminal settled successes or a verified corrected-scorer baseline**.

| Policy | Class | Seen ever/completed | Unseen ever/completed | All ever/completed | Unrecorded of 24 | Protocol status |
|---|---|---|---|---|---:|---|
| adapt0 | Pant_Long | 0/20 | 0/4 | 0/24 | 0 | completed |
| adapt0 | Pant_Short | 0/14 | 0/0 | 0/14 | 10 | garment_switch_timeout |
| adapt0 | Top_Long | 0/20 | 0/4 | 0/24 | 0 | completed |
| adapt0 | Top_Short | 0/6 | 0/0 | 0/6 | 18 | scorer_index_error |
| adapt1 | Pant_Long | 0/20 | 0/4 | 0/24 | 0 | completed |
| adapt1 | Pant_Short | 3/20 | 0/4 | 3/24 | 0 | completed |
| adapt1 | Top_Long | 0/2 | 0/0 | 0/2 | 22 | garment_switch_timeout |
| adapt1 | Top_Short | 0/2 | 0/0 | 0/2 | 22 | garment_switch_timeout |
| baseline | Pant_Long | 0/2 | 0/0 | 0/2 | 22 | garment_switch_timeout |
| baseline | Pant_Short | 8/20 | 0/4 | 8/24 | 0 | completed |
| baseline | Top_Long | 0/4 | 0/0 | 0/4 | 20 | garment_switch_timeout |
| baseline | Top_Short | 0/6 | 0/0 | 0/6 | 18 | scorer_index_error |

`0/0` means no returned episodes for that split, not a zero-percent rate. Valid historical returned episodes inside interrupted jobs are retained. Missing episodes, scorer exceptions and switch timeouts are not policy failures. Historical infrastructure-invalid episode counts cannot be reconstructed precisely from these aggregate/partial records, so the ledger records the failed operation and unrecorded count separately.

| Policy | Successful garment | Episode within garment (1-based) | Reported pre-success length | First success action (derived, 1-based) | Landmark correction affects garment? |
|---|---|---:|---:|---:|---|
| adapt1 | Pant_Short_Seen_3 | 1 | 186 | 187 | Yes |
| adapt1 | Pant_Short_Seen_5 | 2 | 333 | 334 | Yes |
| adapt1 | Pant_Short_Seen_7 | 1 | 313 | 314 | No |
| baseline | Pant_Short_Seen_2 | 2 | 326 | 327 | No |
| baseline | Pant_Short_Seen_3 | 1 | 243 | 244 | Yes |
| baseline | Pant_Short_Seen_3 | 2 | 293 | 294 | Yes |
| baseline | Pant_Short_Seen_4 | 2 | 503 | 504 | Yes |
| baseline | Pant_Short_Seen_7 | 1 | 233 | 234 | No |
| baseline | Pant_Short_Seen_8 | 2 | 113 | 114 | No |
| baseline | Pant_Short_Seen_9 | 1 | 293 | 294 | No |
| baseline | Pant_Short_Seen_9 | 2 | 243 | 244 | No |

Evaluation seed is 100 for these records. The upstream loop excludes the triggering action from its `length`, then executes a 50-action tail including that triggering action while retaining the success flag. Thus first-success action is `length + 1`, and total policy actions are `min(600, length + 50)` for these successful episodes. Neither continuing for that tail nor preserving the flag is a terminal recheck. All historical terminal settled outcomes remain unknown. The v1 audit field named `executed_actions` incorrectly transcribed the raw `length`; the v2/v3 ledgers explicitly corrects its semantics.

## Scorer diagnosis and repair

The JSON `check_point` values address authored USD vertices. GPU PhysX welds coincident UV-seam vertices into fewer particles, while `get_current_mesh_points()` returns the welded GPU particle array. The scorer indexed that array directly with authored-vertex indices.

For **Top_Short_Seen_3**, the actual USD has **11,381 vertices / 10,869 unique positions**. Its authored indices `[10309, 11029, 7217, 4539, 11266, 11378]` map under static first-occurrence welding to `[9845, 10565, 7217, 4539, 10776, 10866]` (actual runtime indices are read from the PhysX maps). Indexing the original list into the 10,869-point array reproduces the reported `IndexError` in a CPU regression. Top_Short_Seen_4 and Top_Short_Unseen_1 have the same out-of-range issue.

Ten garments have changed landmark selection, including seven that would not raise an index error:

Pant_Long_Seen_1, Pant_Long_Unseen_1, Pant_Short_Seen_3, Pant_Short_Seen_4, Pant_Short_Seen_5, Top_Long_Seen_4, Top_Short_Seen_3, Top_Short_Seen_4, Top_Short_Unseen_0, Top_Short_Unseen_1.

Three of the eight baseline short-pants successes and two of the three seed-1 successes involve affected garments. Their raw verdicts are preserved; corrected geometric success is unknown. Comparing a new corrected-scorer rate directly against 8/24 would confound scorer repair with policy performance.

The [repair](../campaigns/20260920-folding-pilot-v3/src/lehome_fold/folding_geometry.py) maps authored landmarks to welded particles and calls the existing official geometric predicate functions without deleting or clamping conditions. Static tests verify that mapped landmarks retain the exact authored rest positions for every available garment. Runtime evaluation uses PhysX’s own `weldedVerticesRemapToWeld` and `weldedVerticesRemapToOrig`, cross-validates both maps and their source rest positions, and requires a finite live particle array of the expected size. It accepts solver reordering rather than assuming first-occurrence welding order. An unverified mapping stops the run as infrastructure-invalid.

**Official condition counts are four for pants and five for tops.** The new result schema records both passed and total conditions rather than inventing a fifth pants condition.

The terminal bug is also reproduced: the official 50-call decorator returns `False` on skipped calls even for a geometrically successful state. The new runner separately records native official sampling events, fresh geometric success after each action, and fresh terminal geometry after 60 settling steps holding the last target. It does not use the throttled return or an earlier success flag as a terminal verdict.

## Garment-switch diagnosis

The raw timeout logs stop after “Old garment object deleted” while switching to Top_Long_Seen_2 (baseline), Pant_Long_Seen_1 (baseline), Pant_Short_Seen_7 (seed 0), Top_Short_Seen_1 (seed 1), and Top_Long_Seen_1 (seed 1). Code inspection places configuration reload and 20 calls to `self.sim.step(render=True)` immediately after that message, before new garment creation. The precise blocking call is not yet reproduced by the new diagnostic.

The [trace](../campaigns/20260920-folding-pilot-v3/src/lehome_fold/switch_trace.py) records timestamped calls, lines, returns and exceptions in switching, deletion, creation, config loading, scene stepping/rendering, initialization and observer retargeting, with repeated stack dumps after 45 seconds. A separate child-process timeout bounds the entire diagnostic. All policy evaluation uses one fresh Isaac process per garment/pose; no benchmark relies on successful in-process switching.

## Tests, smoke jobs and current gate

**14 new regression tests passed**, including real asset/index reproduction, preservation of all 48 sets of authored landmarks, rest-topology rejection tests, terminal-throttle regression, true installed SmolVLA queue execution at 50/10/5, stale camera-file rejection, MP4 encoding, counts/splits and raw baseline accounting. **51 existing pure tests passed**. This does not certify GPU smoke success. Test details are in [improvement_tests.txt](../campaigns/20260920-folding-pilot-v3/audit/improvement_tests.txt) and [validation.json](../campaigns/20260920-folding-pilot-v3/audit/validation.json).

The first new attempt, `21370048` with dependent diagnostic `21370049`, exposed a setup defect in the additional runtime guard: Isaac had already deformed startup particles by approximately 3.28 mm, so comparing current world particles to an undeformed rest shape was invalid. It stopped before policy actions. Those records are preserved, and only these newly submitted related jobs were cancelled to avoid repeating the defect. No unrelated job was cancelled.

The second attempt, `21370072` with diagnostic `21370073`, established that PhysX `restPoints` retain all 11,381 authored vertices while the live array contains 10,869 particles. The diagnostic printed the actual cooked forward/reverse remap attributes. This also stopped before policy actions; only these related jobs were cancelled. The v3 repair uses those authoritative maps, with inverse-map, source-position, bounds, shape and finite-state checks. CPU regression includes a valid solver permutation and rejects corrupted maps.

**Corrected smoke array `21370081` and dependent switch diagnostic `21370082` were submitted through Slurm at 2026-09-21 01:05:30 UTC**, independent of the desktop session. GPU outcomes remain pending until their artifacts pass [smoke_gate.json](../campaigns/20260920-folding-pilot-v3/audit/smoke_gate.json). Exact receipts: [submissions.jsonl](../campaigns/20260920-folding-pilot-v3/submissions.jsonl). Earlier snapshots and errors remain intact.

The first corrected GPU smoke **passed its infrastructure gate**: `21370081_0`, Top_Short_Seen_3, completed 120 policy actions and 60 terminal settling steps in 114.77 child-process seconds. Actual forward/reverse maps verified all six landmarks with zero rest-position error. All 121 three-camera acquisitions were fresh; the minimum garment contribution was 5,080 top-camera pixels. Cloth and robot states remained finite, with three replans. Official ever-success and terminal success were both false; terminal geometry passed **2/5** conditions. The conservative `never_approached_cloth` label uses last-link proximity, not measured contact. This is a legitimate smoke-budget policy failure, not an improvement result.

[Result JSON](../campaigns/20260920-folding-pilot-v3/outputs/smoke_Top_Short_Seen_3/rollout.json), [policy failure MP4](../campaigns/20260920-folding-pilot-v3/outputs/smoke_Top_Short_Seen_3/rollout_policy_failure.mp4), [triptych GIF](../campaigns/20260920-folding-pilot-v3/outputs/smoke_Top_Short_Seen_3/rollout_policy_failure_triptych.gif), [runtime landmark proof](../campaigns/20260920-folding-pilot-v3/audit/first_live_landmark_proof.json).

Final scheduler snapshot: `21370081_0` completed, `21370081_1` running, indices 2–4 pending on the single-task array limit, `21370082` pending on its dependency. See [scheduler snapshot](../campaigns/20260920-folding-pilot-v3/audit/scheduler_snapshot.txt). The experiment gate is 1/5 passed and remains closed for the pilot.

Before cancellation, v2 also completed `21370072_2` on Pant_Long_Seen_0, whose rest topology passed that version’s stricter shape check. It is preserved as a 120-action policy failure with terminal **2/4**, not mixed into the v3 smoke gate. [Long-pants result](../campaigns/20260920-folding-pilot-v2/outputs/smoke_Pant_Long_Seen_0/rollout.json) and [policy failure MP4](../campaigns/20260920-folding-pilot-v2/outputs/smoke_Pant_Long_Seen_0/rollout_policy_failure.mp4) provide a second class’s actual policy footage. Earlier cancelled workers can retain `state: running` if interrupted before their final status write; Slurm accounting and `attempt_status.json` take precedence for those cancelled attempts.

| Smoke index | Garment | Purpose | Budget |
|---:|---|---|---|
| 0 | Top_Short_Seen_3 | isolated policy, fresh RGB, scorer and terminal checks | 120 policy + 60 initial settle + 60 terminal settle |
| 1 | Pant_Short_Seen_0 | isolated policy, fresh RGB, scorer and terminal checks | 120 policy + 60 initial settle + 60 terminal settle |
| 2 | Pant_Long_Seen_0 | isolated policy, fresh RGB, scorer and terminal checks | 120 policy + 60 initial settle + 60 terminal settle |
| 3 | Top_Long_Seen_0 | isolated policy, fresh RGB, scorer and terminal checks | 120 policy + 60 initial settle + 60 terminal settle |
| 4 | Top_Short_Seen_0 | isolated policy, fresh RGB, scorer and terminal checks | 120 policy + 60 initial settle + 60 terminal settle |
| 5 | Top_Long_Seen_0 | bounded switch to Top_Long_Seen_1 | 120 policy + 60 initial settle + 60 terminal settle |

Array indices 0–4 cover all four classes plus the known scorer-failing short top. Index 5 is the separately submitted switch diagnostic. Each requests one GPU, eight CPUs, 64 GB RAM and 15 minutes, with at most one of these GPU tasks running at a time. Each child has an eight-minute wall-clock bound. Missing/crashed runs have no policy verdict.

## Frozen execution-horizon pilot

The [manifest](../campaigns/20260920-folding-pilot-v3/manifest.json) fixes the existing `bc_smolvla_raster_ft_full` checkpoint, model SHA-256 `4a37d4dce3d96e392044ba86295220467513e70ca37f8458edc855edb238ba77`, plus 140 source/config/script hashes. Only the executed prefix length changes. Prediction remains 50 actions; normalization, prompt, cameras, physics, initial/terminal settling and 600-action budget are shared.

| Development slot | Garment | Local demonstration pose key | Evaluation seed | Horizons |
|---:|---|---:|---:|---|
| 0 | Pant_Short_Seen_0 | 0 | 200 | 50, 10, 5 |
| 1 | Pant_Short_Seen_0 | 1 | 201 | 50, 10, 5 |
| 2 | Pant_Short_Seen_3 | 0 | 202 | 50, 10, 5 |
| 3 | Pant_Short_Seen_3 | 1 | 203 | 50, 10, 5 |
| 4 | Pant_Short_Seen_7 | 0 | 204 | 50, 10, 5 |
| 5 | Pant_Short_Seen_7 | 1 | 205 | 50, 10, 5 |
| 6 | Pant_Short_Seen_9 | 0 | 206 | 50, 10, 5 |
| 7 | Pant_Short_Seen_9 | 1 | 207 | 50, 10, 5 |

**8 fixed pose/seed pairs × 3 horizons = 24 rollouts. The pilot has not yet been submitted.** The submission and worker both require the experiment-level smoke gate, not just a Slurm exit code. No retraining or formal comparison has been launched.

Each rollout records native official ever-success, fresh per-action geometry, settled terminal geometry, condition values/margins, particle displacement, finite states, gripper-link proximity and commands, replanning/inference cost, action discontinuity, frame acquisition identifiers, actual garment visibility, runtime, and a conservative failure category. Ground-truth contacts, first grasp and failed-grasp counts remain null where unavailable; proximity is explicitly labeled as a proxy. Wrong-region grasp and dropped-cloth claims are not inferred from unsupported signals.

Freshness rejects reused files before each render. Garment visibility is measured by comparing top RGB with a same-state render in which only the cloth is hidden. MP4 is encoded from retained policy RGB with checkpoint/garment/seed/horizon/outcome/condition/failure captions, plus four GIF views. Smoke videos are plumbing evidence, not an estimate of 600-action policy success. Four-class representative policy footage is being collected; eight qualified new clips do not yet exist, and replay is not substituted.

## Storage and progress commands

Pre-capture measurements: home `du` = 28G; home `df` = 25G total / 22G used / 3.4G available; user HPC-share usage = 1.4T. These different filesystem measurements are recorded as returned. The previous 600-action media output occupies 47M. Projected smoke outputs are under 1 GB and the 24-rollout media/results under 3 GB; raw RGB training datasets are not being saved. All new outputs/caches are within the HPC-share campaign, with the rest of the shared workspace mounted read-only in the container.

Run from the existing repository directory:

```bash
squeue -u "$USER" -o "%.18i %.12P %.24j %.10T %.12M %.30R %.20b"
sacct -j 21370048,21370049,21370072,21370073,21370081,21370082 --format=JobID,JobName,State,ExitCode,Elapsed -X
cat campaigns/20260920-folding-pilot-v3/submissions.jsonl
find campaigns/20260920-folding-pilot-v3/outputs -name status.json -print -exec cat {} \;
python3 campaigns/20260920-folding-pilot-v3/scripts/check_smoke_gate.py
```

Once the smoke gate passes, the guarded pilot submission command is:

```bash
python3 campaigns/20260920-folding-pilot-v3/scripts/submit_experiments.py pilot
```

Outputs are `campaigns/20260920-folding-pilot-v3/outputs/<manifest task id>/`, with `request.json`, `status.json`, `rollout.log`, and on valid completion `rollout.json`, MP4 and GIF views. The switch trace is `rollout.json.switch_trace.jsonl` in its diagnostic output directory. Source hashes prevent a running job from silently using edited code.

No current result supports improved folding. The next gate is runtime verification of the repaired scorer and fresh isolated four-class evaluation; the shorter-horizon hypothesis remains untested.
