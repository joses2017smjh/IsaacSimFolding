# Closed-loop training v2 — live status

<!-- The four fields below are the contract. Update them, in place, after
     every submission and every completed job. Everything under "Record" is
     append-only history. -->

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21400710` iter1.reload.retry1, `21400812` iter2.collect |
| **Latest result** | matched baseline dev H10 2/8, H50 4/8; iter 1 `iter1-step000300` H10 0/8, H50 4/8 |
| **Blocker** | none |
| **Next milestone** | advance the current iteration; final frozen test after the loop |

_Updated 2026-09-23T07:58:32Z by scripts/driver.py._
<!-- driver:status:end -->

## What this campaign is

One bounded rollout-driven AWR iteration from the untouched BC baseline,
measured on closed-loop folding rather than offline loss.

The horizon-pilot campaign is closed at `0e83a7b`. It established that fixed
RNG, RTC, retained queues, observation substitution, static H50-suffix
training, mixed replay and a fresh-H50 rescue oracle all fail to repair the
H10 replanning collapse — the oracle recovered **0/6** student-visited roots.
H50 plans are therefore prohibited as training labels here.

`20260922-closed-loop-training-v1` was a draft of this campaign. It was
reviewed, found to have five correctness defects, and never launched. This
campaign is its repair, rebuilt from committed sources.

## Frozen before submission

Everything decidable in advance is in `manifest.json` and was fixed before any
job ran. The point is that no threshold, seed or selection rule can be chosen
after an outcome is visible.

- **Budget** — 44 GPU tasks for iteration 1 (1 smoke, 8 collect, 1 train, 16
  development, 16 frozen test, 1 boundary, plus 1 CPU compile). 8 more if the
  preregistered expansion fires; 30 more only if a second iteration is earned.
- **Collection** — 4 garment classes x 2 real demonstration poses, seeds
  4200-4207. Every trajectory is retained whatever it scores. No seed
  selection, no dropped failures.
- **Expansion** — a second 8-row draw (seeds 4300-4307, poses 2-3) declared
  now, so a degenerate first collection has a fixed response rather than one
  chosen once the outcomes are known.
- **AWR** — beta 1.0, w_max 3.0, w_min 1e-6. Importance *resampling* with an
  unchanged supervised loss, because SmolVLA's flow-matching forward exposes
  one scalar batch loss and no per-example likelihood. RECAP conditioning is
  **not** used: its gradient update is unimplemented in this repository.
- **Gates** — >=3 distinct rewards, advantage std > 0.02, episode ESS <= n-1
  and >= 3, sample ESS fraction <= 0.95. Two-sided on purpose: ESS is
  *maximised* by uniform weights, so "ESS is high" is equally consistent with
  a strong signal and with no signal at all.
- **Training** — 300 steps, batch 4, 50% AWR-resampled rollout + 50% raster BC
  anchor, lr 1e-5, AdamW, boundary unfreeze, seed 4242, checkpoint every 100.
- **Selection** — evaluate `step_000300` on the full development set; fall
  back to `step_000200` only if `step_000300` breaches the raster retention
  guard. One candidate, decided before training.
- **Targets** — development H10 >= 4/8 primary, >= 6/8 stretch with no H50
  regression below the baseline 4/8, then confirmation on the frozen test set.

## Evaluation design

The eight-pose set is **development data, not an untouched test**. Its H10
failure is what motivated this campaign, so a gain on it is not an unbiased
estimate of anything.

| set | rows | role |
|---|---|---|
| development | 8 poses x {H10, H50} | matched comparison; targets are stated against it |
| frozen test | 8 garments, seeds 9000-9007, H10 | excluded from collection, expansion, development and selection |
| raster held-out | 4 files x 8 frames | regression guard only, never an optimisation target |

Baseline development numbers are **reused** from the horizon pilot: H10 0/8,
H50 4/8. Identical garments, poses, seeds 200-207, checkpoint SHA, 600/60/60
protocol and runner. The only runner change since is trajectory capture, which
is inert unless `--trajectory_out` is passed and is not passed for benchmark
rows. Baseline on the **frozen test set is run fresh** under this campaign,
for both checkpoints.

## Reproducibility

`manifest.json` records the git commit and the SHA-256 of all 64 executed
sources taken from the **committed blob**, not the working tree.
`build_manifest.py` refuses to run against a dirty tree;
`verify_sources.py` re-checks every source and the immutable baseline before
each production submission.

This exists because the horizon pilot's frozen manifest drifted: it recorded
hashes for `policy_rollout51.py` matching neither HEAD nor the working tree,
and `run_improvement_task.py` hard-fails on that mismatch today. That drift is
recorded in `CLAUDE.md` §6.1 and has **not** been hidden by regenerating the
historical manifest.

## Record

append-only; newest last

### 2026-09-22 — campaign frozen and launched

Manifest frozen at `f2422ef`: 64 executed sources pinned by committed-blob
SHA-256, immutable baseline checkpoint pinned, every protocol decision
preregistered. Pushed to `origin/main` at `d6ae7fc` before any job was
submitted.

Clean-checkout proof (`audit/clean_checkout_proof.json`): cloned from origin
into scratch, verified all 64 sources and the baseline checkpoint, ran 13
campaign tests and 56 root tests, and resolved the smoke launch path. The
clone carries the runner, its import graph and the LeHome checkout, so the
campaign is reconstructible from the commit alone.

Smoke job **21400590** submitted.

### 2026-09-22 — smoke 21400590 failed, cause fixed

`OSError: [Errno 30] Read-only file system` on
`.../20260921-horizon-pilot/external/lehome-challenge/logs/...`. LeHome's
logger opens a log file at import time; this campaign binds
`/nfs/hpc/share/sanchej7` read-only with only its own directory writable, so
the frozen pilot checkout that `$LEHOME` points at was not writable. The
pilot never hit this because it bound its *own* campaign directory rw.

Fixed by binding a per-job writable directory over that log path rather than
relaxing the read-only mount, so the frozen campaign stays immutable.

The same commit makes `build_manifest.py` hash `.sh` files.
`slurm/_common.sh` holds the apptainer invocation, the bind mounts and
`PYTHONPATH` — the exact file this bug was in — and was not covered by the
source pin. Manifest refrozen at `a313e4a` with 66 sources.

Smoke retry **21400605** submitted.

### 2026-09-23 — smoke 21400605: every saved checkpoint was unloadable

Stages 1-4 passed. The rollout ran, the trajectory compiled (1 episode, 16
samples, gate correctly degenerate under `--allow-degenerate`), one AWR
update ran and a checkpoint was written. Stage 5 then failed reloading it:

    draccus.utils.ParsingError: Expected a dict with a 'type' key for
    <class 'lerobot.configs.policies.PreTrainedConfig'>

`SmolVLAPolicy.save_pretrained` writes 48 of the baseline's 49 config keys
and omits `type`, draccus's choice-class discriminator. **Every checkpoint
this trainer saves was unloadable by the evaluator path** — and it only shows
up on reload, which is after all the collection and training GPU time has
been spent. This is precisely what the smoke exists to catch.

The other 48 keys were verified byte-identical to the baseline's, and
fine-tuning changes weights rather than architecture, so the baseline config
is restored wholesale. Second benefit: the boundary evaluator symlinks every
candidate file *except* `config.json` and substitutes the baseline's, which
is now a provable no-op. `restore_config_discriminator` refuses to run if any
shared key genuinely differs, so it cannot mislabel a changed architecture.

Manifest refrozen at `d3422c9`, 68 sources. 16 campaign tests pass.
Smoke **21400623** submitted.

### 2026-09-23 — smoke passed, iteration 1 launched

Smoke **21400623** completed all five stages at manifest `d3422c9`:

| stage | result |
|---|---|
| verify sources | 68/68 against committed blobs |
| autonomous H10 rollout | 80 actions, trajectory persisted |
| compile | 1 episode, 16 samples, gate correctly degenerate |
| AWR update + save | 99,880,992 trainable params, discriminator restored 48 → 49 keys |
| reload | `config_matches_baseline: true`, finite action, CUDA |

The two defects the smoke caught — a read-only LeHome log path and an
unloadable saved checkpoint — were both invisible to static review and to the
CPU suite, and each would have surfaced only after the full collection and
training budget had been spent.

Iteration 1 submitted as a chain: collect **21400641** (array 0-7) →
compile **21400642** (`afterany`) → train **21400643** (`afterok`).
`compile` depends `afterany` on purpose so a partial collection reaches the
compiler and exits 2 with an explicit count, rather than leaving the job
silently unscheduled.

### 2026-09-23 — iteration 1 trained; the campaign now drives itself

**Mandate A1 recorded before any candidate existed.** Targets raised to
development H10 settled >= 6/8 (stretch 8/8) with no H50 regression against a
*matched* baseline; up to three collect-train-evaluate iterations; the frozen
test set reserved for the final selected checkpoint only; finite caps of 170
GPU tasks, 45 GPU-hours, no new iteration after 2026-09-24T18:00Z, and a
60 GB campaign / 40 GB filesystem-floor storage limit. `/nfs/hpc/share` is a
shared filesystem at 90% used, so the storage cap is deliberately tight.
See `amendments/A1-autonomous-mandate.json`.

**Iteration 1 gate passed.** Rewards `[0.75 x4, 0.6, 0.4 x3]` (pants reached
3/4 conditions, tops 2-3/5; no settled success): 3 distinct values, advantage
std 0.162, episode ESS 5.36 of 8, sample ESS fraction 0.67. The cap did not
bind on this batch (top z = 0.93 < ln 3), and the gate report says so.

**Iteration 1 training** (`21400643`, 4 min 49 s): all three checkpoints pass
the retention guard, and the raster held-out loss *fell* from 0.1333 to
0.0823 at step 300. The sampled advantage was +0.11 throughout, confirming
the resampling favoured the better episodes. The preregistered rule selects
`step_000300`.

**The driver** (`scripts/driver.py`) now owns progression. It adopted the
three iteration-1 jobs from the ledger rather than resubmitting them, and
submitted the matched baseline benchmark `21400694`, the reload gate
`21400695`, a dependent chain tick `21400696` and a watchdog `21400697`.

Building it surfaced a real hazard: the site `sbatch` is a wrapper that
expands `$@` **unquoted**, word-splitting and glob-expanding every argument.
Earlier submissions worked only because no argument had a space. The driver
calls the real binary and adds the wrapper's one flag itself.

Manifest refrozen as protocol v2 at `b319349` (70 sources); the iteration-1
manifest is archived at `manifests/manifest_d3422c9.json` because the
collection, compile and train jobs recorded that commit.

### 2026-09-23 — iteration 1: no improvement, and H10 got worse

**Matched baseline** (this campaign, identical 16 rows): H10 settled **2/8**,
H50 settled **4/8**. The horizon pilot measured the *same* seeds, checkpoint
and controller at H10 0/8 and H50 4/8 — two H10 rows flipped between runs.
Pooled: H10 2/16, H50 8/16. A development difference of one or two episodes
is within observed run-to-run variation, and every comparison below is read
with that in mind.

**Iteration 1** `iter1-step000300`:

| | H10 settled | H10 mean conditions | H50 settled | H50 mean conditions |
|---|---|---|---|---|
| matched baseline | 2/8 | 3.000 | 4/8 | 3.125 |
| iter1-step000300 | **0/8** | **2.125** | 4/8 | 3.250 |

Eligible (retention guard, clean reload, no H50 regression, all rows valid),
but no improvement — and a systematic H10 degradation: 7 of 8 rows collapsed
to exactly 2/4 and none improved. **Diagnosis:** AWR resampling of H10
*failure* trajectories reinforced the H10 failure mode, and the advantage it
used measured garment class rather than action quality (every pants episode
3/4, every top 2-3/5). The gate passed because a between-class confound has
spread and is not concentrated — which is exactly what the gate checks.

**Iteration 2** (amendment A2, recorded before any iteration-2 data): the
changed factor is the *success density* of the collection. H10 data from this
baseline is ~87% failure; H50 on the target class is the one configuration
where it is measured to succeed (8/10 on P_B/P_C poses across both runs).
16 H50 rollouts on Pant_Short garments 4/5/6/8 — no development or test
garment — mirroring the development pose mix, from the untouched baseline,
training otherwise identical. A2 names the cached-suffix issue the pilot
closed at two boundary states, how full-trajectory outcome-weighted training
differs, and the discriminating prediction: an H10 gain on P_B/P_C rows
supports plan-continuity transfer; no gain rules the factor out.

Launched by the driver from the committed plan: collection **21400812**.
Along the way the driver needed four fixes, each now a regression test: a
tick that `scancel`'d itself (ledger adoption closes the window), a stale
lock from a killed tick (holders are now checked), `--begin` read in local
time (now relative), and a budget-skipped test that would have been scored
0/0 (now reported "not run").
