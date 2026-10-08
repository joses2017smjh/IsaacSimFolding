# Folding status and next steps — start here

Updated 20 September 2026. The user designated **lehome-fold-repro** as the
starting point for folding status and things to do. The GitHub repository is
[IsaacSimFolding](https://github.com/joses2017smjh/IsaacSimFolding).

## Read these records first

1. [Current media gate and repository boundaries](docs/MEDIA_GATE_2026-09-20.md).
2. [Latest recorded media status](campaigns/20260920-media-root-fix/MEDIA_STATUS.md),
   [JSON](campaigns/20260920-media-root-fix/MEDIA_STATUS.json), and
   [replacement snapshot README](campaigns/20260920-media-root-fix/README.md).
3. [Original audit and unfinished documentation fixes](campaigns/20260919-media/docs/AUDIT_2026-09-19.md).
4. [Strict evaluation results](../bhl-robustness-ladder/docs/CLOTH_FOLDING_WEEKEND.md)
   and [publication handoff](../bhl-robustness-ladder/docs/PUBLICATION_HANDOFF_2026-09-20.md).
   The sibling BHL campaign owns the newer adaptation training and strict evaluations;
   this folding repository owns the garment media.

The [September 19 session note](SESSION_STATUS_2026-09-19.md) and original
campaign reports remain historical records. Their “pending” statements are
superseded only where newer completed artifacts exist. Do not use old PLAN,
STORM or README snapshots as current execution instructions without checking
the audit and newer results.

## Latest recorded outcomes

- Two SmolVLA adaptations already finished 1,500 updates each. Selected
  checkpoints: seed 0, step 1200; seed 1, step 1000. They are two training
  seeds of one model family, not two newly implemented policy algorithms.
- Strict short-pants evaluation: historical raster-adapted baseline **8/24**,
  adapted seed 1 **3/24**; both **0/4 Unseen**. Only **5/12** class/checkpoint
  cells completed. No improvement is demonstrated; incomplete cells do not
  count as completed failed folds.
- Latest media report, **2026-09-20 17:35 UTC**: replacement gate **21367715**
  completed the seed-0 short-top pose-2 episode: **600 actions, 601 validated
  render calls**, overhead/left-wrist/right-wrist/triptych GIFs. Checker ever
  and terminal verdicts are both false. Rendering passed; the fold failed.
- Original gate **21360435** failed before a policy outcome; its dependent
  array **21360436** and report **21360437** were preserved. A replacement gate
  cannot satisfy the old array's dependency. **No replacement 15-episode array
  is recorded in these notes.** This is a recorded snapshot, not a fresh
  scheduler query.
- Earlier learned-policy success/failure clips are published alongside the
  new failure and both wrist views: [gallery and provenance](../bhl-robustness-ladder/docs/FOLDING_MEDIA.md),
  [live page](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-folding/).
  The successful clip uses an older checkpoint and a latched checker verdict;
  no new adapted-policy successful GIF or stable terminal success is verified.

## Things still to do, in order

1. **Read existing results before scheduling.** If resuming execution, inspect
   relevant scheduler records read-only and verify source JSON, missing
   episodes, camera freshness and physics. Do not rerun completed adaptation
   training or equate a successful batch job with a successful fold.
2. **Finish media coverage only as a separately requested continuation.**
   The repaired gate establishes one valid failed episode. A new immutable
   replacement array would be needed for the remaining 15 planned policy/pose
   recordings, using the repaired asset root and pinned checkpoints. Preserve
   the original array and dependencies. Save every actual outcome and all
   camera views; report missing class successes instead of substituting replay.
3. **Repair evaluation blockers before broad comparisons.** Diagnose the
   garment-switch cleanup stall and scorer/mesh index mismatch (index 11029
   with 10869 particles). Complete matched class/checkpoint protocols with
   explicit incomplete-run accounting and Seen/Unseen separation.
4. **Test bounded improvements after valid evaluation gates.** The
   [latest research directions](../bhl-robustness-ladder/docs/FOLDING_NEXT_EXPERIMENTS.md)
   propose fixed-checkpoint execution horizons 50/10/5, then a controlled
   uniform-versus-grasp-window sampling comparison. First expand the sparse
   long-pants supervision while retaining garment holdouts. These proposals
   are not queued work or proven improvements.
5. **Keep legacy implementation limits visible.** The candidate scorer is
   action-independent; RECAP's gradient update is unimplemented and its workers
   do not save full training trajectories. Do not launch them as working new
   methods. Corrective data must contain expert actions at visited states.
6. **Finish the documentation corrections listed in the original audit.**
   The public README's contradictory policy-success and completed-RECAP claims
   were qualified in commit `a154097`. Remaining checks include stale CPU/RTX
   advice in PLAN/STAGE0, historical banners in STORM/RENDER/README_long,
   the overly strong paired-renderer causal claim, and NOTICE's reproduction
   scope wording. Preserve copyright and license attribution. Do not overwrite
   historical measurements or describe SmolVLA as a full pi0.5 reproduction.

## Publication and local-workspace boundary

Public portfolio and README changes are already pushed; exact commits and
validation are recorded in the linked BHL publication handoff. They were
published from isolated checkouts. Local research checkouts may have older
HEADs and unrelated dirty files: inspect Git status before updating, and do not
reset, force-push or automatically pull over active work.

**No jobs were submitted, cancelled, requeued, released or changed while this
index was written.** The user's constraint to leave existing jobs alone remains
in force. This file links status and pending work; it does not authorize or
claim a new experiment.
