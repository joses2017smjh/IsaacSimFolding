> **Note (2026-09-03):** several GIFs below render an empty table. The observer copied topology
> from the first mesh in each garment USD, and some assets carry more than one, so the cloth mesh
> was invalid and drew nothing. Physics and scoring were unaffected. Fixed in `storm_obs.py`;
> these GIFs predate the fix. See the Known issue section in the main README.

<h1 align="center">Isaac Sim Folding</h1>

<p align="center">
  Reproducing <a href="https://arxiv.org/abs/2606.27163"><b>Learning to Fold</b></a> —
  1st of 62 at the LeHome Challenge 2026 — on a shared HPC cluster.<br>
  <sub>Bimanual SO-ARM101 · flow-matching VLA · a policy that is its own value function</sub>
</p>

<p align="center">
  <img src="docs/img/cloth_fold.gif" width="820" alt="PhysX particle cloth settling onto a table over 364 simulation steps, rendered with OpenUSD Storm on Isaac Sim 5.1">
</p>

<p align="center"><sub><b>Real PhysX particle cloth.</b> 9,774 particles over 364 steps of
episode 0 of the released demonstrations, on Isaac&nbsp;Sim&nbsp;5.1 — the stack that still has
particle cloth — rasterised through OpenUSD Storm, which never loads the RTX delegate that
segfaults on this cluster. The geometry is simulation output, not an authored pose.</sub></p>

<p align="center">
  <img src="docs/img/cloth_start.png" width="400" alt="Garment bunched at the start of the episode">
  <img src="docs/img/cloth_settled.png" width="400" alt="The same garment draped flat on the table with sleeves visible">
</p>

<p align="center"><sub>Start, and 90 frames later: the garment drapes from bunched to flat, sleeves out.
The top falls <b>0.178 m</b> and total particle displacement is <b>0.330 m</b>.</sub></p>

<p align="center">
  <b>15 verified folds across all four garment classes</b> &nbsp;·&nbsp;
  every verdict from the challenge's own scorer &nbsp;·&nbsp;
  <a href="#the-cause-measured">the failure cause, measured</a>
</p>

<table align="center">
<tr><td align="center" width="33%"><b>What works</b></td>
    <td align="center" width="33%"><b>What does not</b></td>
    <td align="center" width="33%"><b>Why, measured</b></td></tr>
<tr>
<td>Particle cloth on Isaac&nbsp;Sim&nbsp;5.1, Storm rendering in-process, a 450M VLA driving the
env closed-loop, and LeHome's scorer labelling every episode.</td>
<td>The trained policy folds nothing — <b>0 of 8</b> — while demonstration replay through the
identical pipeline folds <b>15 of 21</b>.</td>
<td>Not the policy and not the training budget. A <b>0.902 skill drop</b> caused by the renderer
alone, with the state trajectory held fixed.</td>
</tr>
</table>

---

## The three cameras the policy actually sees

<p align="center">
  <img src="docs/img/three_camera_view.png" width="900" alt="Three camera views side by side: left wrist showing gripper jaws over red cloth, top-down showing both arms and the garment, right wrist">
</p>

<p align="center"><sub>Left wrist · top-down · right wrist, rendered by OpenUSD Storm inside a live
Kit process. The wrist cameras ride the grippers, matching the challenge's own rig
(<code>/Left_Robot/gripper/left_wrist_camera</code>).<br>
They did not always. They were pinned to fixed world poses aimed ~0.37&nbsp;m from where the garment
rests, so both rendered empty table and <b>the policy ran on two dead frames out of three</b>.
Measured before the fix: 14% of side-panel pixels changed between first and last frame but
<b>0.00%</b> changed by more than 60 — shading drift, no object motion. After: <b>13.1%</b> and
<b>17.4%</b> above 60.</sub></p>

---

## The first job I ran killed the plan

The build order said this needed Isaac Sim 6.0. **38 seconds of GPU time proved the opposite** — and proved the version it *does* need is broken here:

```
Episode: probe_rtx_tiled.py            exit 139
omni::usd::UsdManager::createHydraEngine
  → libomni.hydra.rtx.plugin.so
  → libcarb.scenerenderer-rtx.plugin.so
  → librtx.scenedb.plugin.so :: carbOnPluginStartup
node cn-r-2 · NVIDIA A40 · driver 595.71.05
```

The benchmark **pins `isaacsim==5.1.0`** and needs three RGB cameras. 5.1's RTX plugins segfault on
this cluster's drivers. 6.0's don't — same node class, same probe, renders fine.

Not missing RT cores (an A40 has them). Not fixable by `--constraint` (every driver here is 595 or 610).
Not fixable by a container (`--nv` injects the *host* driver). **It's cluster-wide, and I found it before
downloading 25 GB or writing a line of training code.**

### The 6.0 escape hatch is closed — but 5.1 was never the problem

Standing the task up on 6.0 gets through every import, builds the config,
reaches scene setup — then:

```
SingleClothPrim is no longer available. Omniverse PhysX removed the
deprecated particle-based cloth features.
```

So 6.0 renders and has no cloth. I concluded no stack here could do both.
**That was wrong**, and the error was in the other half: *"5.1 cannot render"*
is not true of 5.1. What segfaults is `librtx.scenedb.plugin.so`, the **RTX
delegate**. 5.1 physics is fine, and the same wheels ship OpenUSD's **Storm**
rasteriser, which never loads it.

<p align="center">
  <img src="docs/img/storm51_scene.png" width="820" alt="The LeHome scene rendered on Isaac Sim 5.1 by OpenUSD Storm: table, textured garment, two SO-101 arms">
</p>

<p align="center"><sub>The real challenge scene on <b>Isaac Sim 5.1</b>, drawn by
<b>Storm</b> — no Kit, no SimulationApp, no RTX. The stack that <i>has</i> the particle cloth.</sub></p>

| approach | physics | scorer | images |
|---|---|---|---|
| port the cloth to 6.0 | **changed** | official | RTX |
| **Storm on 5.1** | **official** | **official** | rasterised |

The official checker is **geometric over particle positions** — it does not
care which rasteriser drew the pixels. So this keeps official physics *and* the
official scorer and deviates only in what the policy sees.

**The caveat is real and is not cropped out.** Storm evaluates only
`UsdPreviewSurface`; Omniverse assets ship MDL. The table and garment shade
correctly because this repo authors PreviewSurface for them — **the robots
render flat white**. So the domain gap is larger than "rasteriser vs path
tracer", and it has to be measured before any success rate is reported from
this path. Full write-up, including the five things that had to line up and the
two failed rebinding attempts: [`docs/STORM.md`](docs/STORM.md).

> Then I split the stack so it stopped mattering. Stages 1–2 read the released LeRobot dataset and
> **never open a simulator** — `LH_ROUTE=train` runs them today on a 7.6 GB venv, and the 18 GB of
> demonstrations are downloaded. Only evaluation and Stage 3 rollouts are blocked.

---

## Components, running

Every figure below is generated by executing the code in [`src/lehome_fold/`](src/lehome_fold) —
`python scripts/make_figures.py`. Synthetic outcomes with **known ground truth**, because that's
the only way to show an estimator recovers the right answer.

### The gate that catches a silent failure

<img src="docs/img/g2_calibration.png" width="880" alt="Three reliability diagrams: calibrated passes G2, miscalibrated fails on ECE, degenerate fails because every label is 1">

An uncalibrated success head doesn't crash — it makes Stage 3's advantages *noise*, and the run
degrades over hours. So calibration is a gate with a number on it. **The right-hand panel is the
one that matters:** the released demonstrations contain no failures at all, so a success head fit
on them is accurate, useless, and would sail past a careless check.

<sub>These three panels are **synthetic**, built to show the gate distinguishing a calibrated head
from a miscalibrated and a degenerate one. The gate has since been run for real — see
[Stage 2](#stage-2-a-calibrated-value-head-gated-on-real-outcomes), where a head trained on 20
scored rollouts passes at `ECE=0.0717`. The right-hand panel stopped being hypothetical: the
demonstrations really were degenerate, and rollout outcomes are what fixed it.</sub>

### Why AWR needs a guard

<img src="docs/img/awr_ess.png" width="880" alt="Effective sample size collapsing from 256 to 1 as the AWR temperature beta shrinks">

Unclipped exponential weights let one episode own the batch. **ESS 256 → 1** as β shrinks, and the
loss curve stays smooth the whole way down. Logged every step.

### The stale-checkpoint bug, caught by construction

<img src="docs/img/g3_staleness.png" width="880" alt="Histogram of rollouts by checkpoint lag, with those beyond max_lag rejected">

In an async pipeline a worker running a 5-versions-old policy produces advantage labels against the
wrong baseline. Nothing crashes. Every rollout is stamped with the checkpoint version **and digest**
that produced it; anything past `max_lag` is dropped and counted.

```
[trainer] cycle 0: +80 episodes  lag0=80
[trainer]   G3 DROPPED 12: rollout is 5 versions behind (max_lag=2)
[trainer]   G3 DROPPED  1: rollout carries no _ckpt_version
[trainer] cycle 0: n=80 success=0.400 adv[-0.873,+0.890] recap_positive=0.400 AWR_ESS=40.8/80
```

---

## Four bugs, and what each one actually looked like

<img src="docs/img/render_failures.png" width="880" alt="Four side-by-side pairs: over-exposure, bad framing, a 4 mm garment, and an arm slumping under gravity, each against the corrected render">

Every defect is reintroduced by `render_scene.py --defect <name>`, so the gallery
regenerates from source and cannot drift from what it claims. Captions carry a
measured pixel difference, not an adjective.

Two candidates were **cut for being no-ops**: a single-sided cloth mesh and a
missing 180° yaw both produced byte-identical frames, so they are not shown —
and the backface-culling story I had written for the invisible garment was
wrong. It was invisible for exactly one reason: the USD authors
`xformOp:scale = 0.01`, so a 0.95 m mesh rendered at 4 mm.

---

## 34 tests, no GPU, no simulator

The paper's method lives in a half that needs neither — so it stays testable while the environment is broken.

```
$ python tests/test_pure.py
  ok  episode_mode_on_demo_data_is_degenerate_and_says_so
  ok  splits_rejects_held_out_garments_in_training
  ok  g3_rejects_stale_unstamped_and_mismatched_rollouts
  ok  feature_tap_captures_embed_prefix_with_the_real_signatures
  ok  eval_log_cross_check_catches_a_truncated_log
  ...
32 passed, 0 failed
```

**Three bugs installing the real LeRobot found**, that reading could not:
`embed_prefix` takes model-specific positional args on both π0.5 and SmolVLA and accepts no batch
dict — so the wrapper now *taps* the method mid-forward-pass instead of calling it. The optimiser
was being handed `None.parameters()`. And the rollout worker never recorded the value head's own
prediction, which would have silently degraded the advantage to a batch mean — the exact claim the
paper makes, quietly discarded.

---

## What it implements

| | | |
|---|---|---|
| **S1** | BC fine-tune, π0.5 or SmolVLA | trainable now |
| **S2** | value head — success, progress, futures, on a **frozen** backbone so it ablates cleanly | partly trainable |
| **S3** | RECAP advantage conditioning + AWR, async trainer + N workers | G3 verified |
| **S4** | Thompson sampling at inference | implemented |

`src/lehome_fold/` — [splits](src/lehome_fold/splits.py) · [labels](src/lehome_fold/labels.py) ·
[value_head](src/lehome_fold/value_head.py) · [awr](src/lehome_fold/awr.py) ·
[recap](src/lehome_fold/recap.py) · [calibration](src/lehome_fold/calibration.py) ·
[thompson](src/lehome_fold/thompson.py) · [ckpt](src/lehome_fold/ckpt.py) · [eval_log](src/lehome_fold/eval_log.py)

**Nothing under `external/` is modified.** `run_eval.py` injects our policies into the challenge's own
registry, defers to its evaluator unchanged, and cleans up on exit — so a success rate here means
what the leaderboard's meant.

---

## Rollouts, scored by the challenge's own checker

A policy now runs closed-loop inside Isaac Sim: the environment steps, OpenUSD Storm rasterises three
camera views in-process, SmolVLA consumes them, its action goes back into the environment, and every
episode is labelled by LeHome's `success_checker_garment_fold` — the same function the challenge
scores with. No verdict below is self-assessed, and the filenames carry the verdict so a label
cannot drift from the run that produced it.

<table>
<tr>
<td align="center"><b>Demonstration replay — SUCCESS</b></td>
<td align="center"><b>Trained BC policy — failure</b></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Top_Short_Seen_1_replay_ep25_success.gif" width="420" alt="Bimanual arms folding a short-sleeve top; the official checker returns success"></td>
<td><img src="docs/gifs/rollout_Top_Short_Seen_1_policy_failure.gif" width="420" alt="The same garment under the trained policy; the arms move but never contact the cloth"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Top_Short_Seen_0_replay_ep2_success.gif" width="420" alt="Short-sleeve top folded successfully by demonstration replay"></td>
<td><img src="docs/gifs/rollout_Top_Short_Seen_0_policy_failure.gif" width="420" alt="The identical garment under the trained policy, left untouched"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Top_Long_Seen_0_replay_ep251_success.gif" width="420" alt="Long-sleeve top folded successfully by demonstration replay"></td>
<td><img src="docs/gifs/rollout_Top_Long_Seen_0_policy_failure.gif" width="420" alt="Long-sleeve top under the policy, garment untouched"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Pant_Long_Seen_0_replay_ep750_success.gif" width="420" alt="Long pants folded successfully by demonstration replay"></td>
<td><img src="docs/gifs/rollout_Pant_Long_Seen_0_policy_failure.gif" width="420" alt="Long pants under the policy, garment untouched"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Pant_Short_Seen_0_replay_ep500_success.gif" width="420" alt="Short pants folded successfully by demonstration replay"></td>
<td><img src="docs/gifs/rollout_Pant_Short_Seen_0_policy_failure.gif" width="420" alt="Short pants under the policy, garment untouched"></td>
</tr>
</table>

<p align="center"><sub>All four garment classes, one row each after the first: short top, long top,
long pants, short pants.</sub></p>

<p align="center"><sub>The top two rows are the <b>same garment</b> in both columns.
Left column: recorded demonstrations replayed at their own spawn pose.
Right column: the trained BC policy. Both columns are the same pipeline, the same renderer and the
same scorer — only the thing choosing the actions differs.</sub></p>

| driver | episodes | succeeded |
|---|---|---|
| demonstration replay | 21 | **15** |
| trained BC policy | 8 | **0** |

Verified folds now cover **all four garment classes**, which matters because each class has its
own thresholds — a short top passes `[9.45, 12.15, 9.0, 13.05, 8.55]`, a long top
`[11.7, 10.8, 10.8, 9.9, 9.0]` — so a result on one says little about the others:

| class | matched-pose replays | succeeded |
|---|---|---|
| Top_Short | 11 | **8** |
| Top_Long | 4 | **3** |
| Pant_Short | 3 | **2** |
| Pant_Long | 3 | **2** |

Per-episode detail, including which of the five fold conditions each run passed, is in
[`results/summary.md`](results/summary.md).

### Replay succeeds and fails on the same garment

Open-loop replay is not a guaranteed success: it drives a recorded action sequence at a cloth that
diverges from the recording, so it lands 15 times in 21. Both outcomes below are the same garment
class, the same pipeline and the same scorer — which is what makes the success meaningful rather
than cherry-picked.

<table>
<tr><td align="center"><b>SUCCESS — 5/5 conditions</b></td>
    <td align="center"><b>failure — drift, not a scene problem</b></td></tr>
<tr>
<td><img src="docs/gifs/rollout_Top_Long_Seen_0_replay_ep252_success.gif" width="420" alt="Long-sleeve top folded, all five conditions passed"></td>
<td><img src="docs/gifs/rollout_Top_Long_Seen_0_replay_ep250_failure.gif" width="420" alt="Long-sleeve top, same class, open-loop replay drifts and misses"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Pant_Short_Seen_0_replay_ep501_success.gif" width="420" alt="Short pants folded successfully"></td>
<td><img src="docs/gifs/rollout_Pant_Short_Seen_0_replay_ep502_failure.gif" width="420" alt="Short pants, same class, replay misses"></td>
</tr>
<tr>
<td><img src="docs/gifs/rollout_Pant_Long_Seen_0_replay_ep752_success.gif" width="420" alt="Long pants folded successfully"></td>
<td><img src="docs/gifs/rollout_Pant_Long_Seen_0_replay_ep751_failure.gif" width="420" alt="Long pants, same class, replay misses"></td>
</tr>
</table>

<p align="center"><sub>Every filename carries the verdict the challenge's checker returned, so a
label cannot drift from the run that produced it. That guard exists because a replay once
overwrote a policy GIF of the same garment and verdict.</sub></p>

### What separates the two columns

The success criterion for a short-sleeve top is five distances, three that must close and two that
must stay open. An untouched garment passes the two "stay open" conditions for free, which is exactly
what the policy scores:

| condition | required | replay (ep2) | policy |
|---|---|---|---|
| `dist(p0,p4)` | ≤ 9.45 | **5.06** ✓ | 29.04 ✗ |
| `dist(p2,p3)` | ≤ 12.15 | **5.65** ✓ | 38.88 ✗ |
| `dist(p1,p5)` | ≤ 9.00 | **5.75** ✓ | 29.82 ✗ |
| `dist(p0,p1)` | ≥ 13.05 | 20.30 ✓ | 18.63 ✓ |
| `dist(p4,p5)` | ≥ 8.55 | 16.47 ✓ | 21.61 ✓ |

Instrumenting the failure rules out the easy explanations. The policy is not degenerate — its actions
vary step to step and the arms sweep ~1.5 rad. The action path is faithful: `action_scale = 1.0` and
targets reach `set_joint_position_target` with the correct 6/6 split. The end effectors descend from
~0.78 m to ~0.65 m and then **hover 12–14 cm above a garment resting at 0.529 m**, never closing.
With the garment settled before the policy acts, its own contribution to cloth motion over 300 steps is **~1.4 cm** — enough to perturb the cloth, nowhere near a fold. (An earlier figure of 0.18 cm came from a run with no settle phase, where the visible motion was gravity rather than the policy.)

The environment is not the problem, and the replay column is what proves it — same scene, same
physics, same renderer, real folds.

<a name="the-cause-measured"></a>

### The cause, measured

The policy is not undertrained and not mean-collapsed. Run on the **path-traced frames it was
trained on** — no simulator, no Storm — it reproduces demonstrated actions almost exactly. Run on
**Storm frames of the identical states**, it collapses to barely better than guessing the average
action:

| frames | skill vs mean-action baseline | MSE | output variance / demos |
|---|---|---|---|
| path-traced (training distribution) | **+0.966** | 0.01028 | 0.985 |
| Storm rasterised (what rollouts feed it) | **+0.063** | 0.18345 | **0.275** |

Same policy, same states, **only the renderer differs** — a 0.902 skill drop attributable to
rasterisation alone. The state trajectory is pinned by demonstration replay, so this isolates
perception from compounding closed-loop drift, which was the competing explanation.

That accounts for every symptom: the end effectors hovering ~12 cm above the cloth, the 0/8 rollout
failures, and why fixing a genuine wrist-camera bug changed the policy's behaviour almost not at all
— correcting camera *geometry* does nothing about a rendering-*style* mismatch. Mean-collapse is the
symptom of the blindness, not the disease.

**The bottleneck is the renderer, not the policy or the training budget.** Closing it means
path-traced observations at rollout time — which on this cluster means the RTX delegate that
[segfaults on Isaac Sim 5.1](docs/STORM.md) — or fine-tuning on rasterised frames so the two
distributions meet.

## Stage 2: a calibrated value head, gated on real outcomes

The success head could never pass its gate while the only labels were demonstrations — they are all
successes, so `labels.class_balance` reported them **degenerate** and there was no negative signal to
learn from. Twenty matched-pose replays now supply **15 success / 5 failure** across 6,066 frames,
and with that the gate passes:

| metric | value | threshold |
|---|---|---|
| ECE | **0.0717** | ≤ 0.10 ✓ |
| pred_std | **0.1813** | ≥ 0.01 ✓ |
| Brier | 0.1144 | — |
| base rate | 0.772 | — |
| MCE | 0.4383 | — |

**G2 PASS.** Read with two caveats, both in [`results/g2_calibration.txt`](results/g2_calibration.txt)
with the full reliability table: the 0.772 base rate flatters ECE, so `pred_std` is what shows the
head is not near-constant; and `MCE=0.438` reflects sparse low-confidence bins — 1–7 samples each,
against 199 in `[0.93, 1.00)`. The head is calibrated where the data is, not across the whole range.

## Honest status

**What works.** Particle cloth simulates on Isaac&nbsp;Sim&nbsp;5.1; Storm rasterises in-process
alongside a live Kit application; a 450M SmolVLA checkpoint drives the environment closed-loop; and
LeHome's own scorer labels every episode. Demonstration replay folds garments across **all four
classes** — 15 of 21, confirmed by the official checker.

**What does not.** The trained policy folds nothing, 0 of 8. The successes here are
**demonstration replays, not the policy**, and every filename says `replay` for that reason.

**Why, and it is now measured rather than guessed.** The policy reproduces demonstrated actions
almost exactly on the path-traced frames it was trained on (skill **+0.966** against a mean-action
baseline) and collapses to **+0.063** on Storm frames of the *identical* states. The bottleneck is
the renderer. Three earlier explanations were asserted and then refuted by measurement — a
mis-stated version of the domain gap, the wrist cameras, and mean-collapse — so the reasoning arc
is recorded in [`SLURM_JOBS.md`](SLURM_JOBS.md) alongside the results.

**Caveats that bound every number above.**

- The paper used **π0.5, not SmolVLA**. π0.5 is currently **blocked, not merely unfinished**:
  lerobot 0.4.3's implementation probes for `transformers.models.siglip.check`, a module from a
  patched transformers fork that no declared extra installs. Forcing one risks the working SmolVLA
  pipeline, so it was not attempted.
- The BC schedule is **incomplete**: 25K of 30K steps, resumed across wall clocks. Loss was still
  descending (0.067 → 0.059) when the last one expired.
- Replay is **open-loop**, so its 6 failures are drift, not a claim about the demonstrations.
- Every number is on **48 garments, not the leaderboard's 80** — the other 32 never shipped.
- The measured skill gap is **single-step and teacher-forced on replay states**. It isolates
  perception from compounding drift, which is what it was built to do, but it is not a closed-loop
  success measurement.

Job ledger: [`SLURM_JOBS.md`](SLURM_JOBS.md) · Full build order: [`docs/PLAN.md`](docs/PLAN.md)

<sub>Environment, assets and scorer © the LeHome Challenge organizers (Apache-2.0).
Method: <a href="https://ilialarchenko.com/projects/lehome2026/">Ilia Larchenko</a>.</sub>
