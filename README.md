# Bimanual Garment Folding in Isaac Sim

An Isaac Sim port and evaluation workflow for camera-driven garment folding: rasterized observations, recovery-supervised flow-matching policies, and independently checked settled folds.

[![Historical closed-loop policy fold in Isaac Sim](docs/demo/POLICY_fold_success.gif)](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-folding/)

*Historical raster-adapted policy on one selected short-pants pose. The challenge checker passed during the episode; this clip does not establish a stable final fold or the latest fine-tune's performance.*

[Case study](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-folding/) · [Latest completed study](campaigns/20260926-anchor-diagnostic-v8/REPORT.md) · [Recorded policy failures](docs/MEDIA_GATE_2026-09-20.md)

## Problem and contribution

A convincing folding clip can hide open-loop replay, a transient checker pass, or a rendering failure. I built the Isaac port, three-camera Storm observation path, policy rollout instrumentation, and source-pinned experiment drivers to distinguish those cases.

The robot, garment assets, challenge checker, and SmolVLA backbone are upstream components. My contribution is their integration, renderer-domain diagnosis, training adaptations, and reproducible evaluation. [Upstream challenge](https://github.com/lehome-official/lehome-challenge) · [Provenance](NOTICE).

## Result and limits

**Latest completed comparison: v8, September 27, 2026.** The candidate `an1-step000300` and the immutable raster-adapted baseline ran in the same interleaved Slurm arrays.

- **H10 settled folds:** candidate 9/16; baseline 2/16; one-sided Fisher p = 0.0117.
- **H50 settled folds:** candidate 6/16; baseline 8/16. The preregistered non-regression gate failed.
- **Validity:** 64/64 episodes completed 600 policy actions plus a 60-step settle. Retention, reload, and fit checks passed.
- **Decision:** retain the baseline. No fine-tune has passed the full screen; the confirmatory final set remains unspent.

H10 replans every 10 actions of a 50-action chunk; H50 executes the full chunk and is the shipped configuration. The short-horizon result is a development finding, not an improved deployed policy. The eight poses are reused across studies, repeats share seeds, and GPU model changes outcomes. The planned final set also overlaps a development pose and does not cover every pose cluster.

[Report and caveats](campaigns/20260926-anchor-diagnostic-v8/REPORT.md) · [Scored ledger](campaigns/20260926-anchor-diagnostic-v8/ledger/driver_state.json) · [Manifest](campaigns/20260926-anchor-diagnostic-v8/manifest.json). The [v9 hardware-pinned remeasurement](campaigns/20260928-fresh-seed-remeasure-v9/STATUS.md) has no published outcome in the audited snapshot; its last status update is September 29.

## How it works

1. PhysX particle cloth and bimanual joint state feed three rasterized RGB views: overhead and both wrists.
2. SmolVLA predicts joint-target action chunks; no demonstration actions are injected into policy rollouts.
3. Recovery search branches from baseline states on training garments. Only continuations that finish with a settled fold become supervision.
4. Fine-tuning mixes recovery labels with a behavior-cloning anchor. A held-out demonstration loss guard filters checkpoints.
5. The driver evaluates candidate and baseline together, then checks all four fold conditions after settling. An earlier latched checker pass is reported separately.

[Observer](src/lehome_fold/storm_obs.py) · [Executed rollout](campaigns/20260926-anchor-diagnostic-v8/scripts/render/policy_rollout51.py) · [Trainer](campaigns/20260926-anchor-diagnostic-v8/scripts/rollout_weighted_finetune.py) · [Decision rule](campaigns/20260926-anchor-diagnostic-v8/scripts/driver.py).

## Engineering decisions

- **Storm observations:** bypass the Isaac Sim 5.1 RTX crash on the cluster; diagnose and adapt to the renderer gap instead of attributing it to control.
- **fp32 master weights:** repair an AdamW update that changed only 12.9% of bf16 expert weights over 300 steps. [Optimizer diagnosis](campaigns/20260923-recovery-supervision-v3/REPORT.md).
- **Matched evaluation:** avoid pooling favorable baselines across campaigns. Source hashes, checkpoint hashes, manifests, and a restartable job ledger make each comparison inspectable.
- **Anchor diagnostic:** changing the BC anchor removed a shared early-H50 behavior shift, but also changed capture source and selected training step. Composition alone is not isolated.

The value head, RECAP, and bandit modules are research prototypes; they are not part of the executed v3–v8 recovery-training loop. Detailed experiment history stays in the [campaign reports](campaigns/) and [historical notes](docs/README_long.md).

## Run locally

The CPU modules require Python 3.10+, NumPy, and PyTorch; no simulator is needed. On Linux or macOS:

```bash
git clone https://github.com/joses2017smjh/IsaacSimFolding.git
cd IsaacSimFolding
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tests/test_pure.py
```

The test runner covers labels, calibration, weighting, provenance, evaluation parsing, and model heads. It does not validate live cloth physics or the LeRobot interface. [CPU CI](.github/workflows/tests.yml).

GPU rollouts additionally need the challenge assets, pinned Isaac Sim 5.1 / LeRobot environment, Apptainer, and Slurm. Each campaign's driver owns submissions and verifies source hashes; campaign ledgers record outcomes. See [container](container/lehome.def), [campaign setup](campaigns/20260926-anchor-diagnostic-v8/README.md), and [media gate](docs/MEDIA_GATE_2026-09-20.md). No universal GPU quickstart is claimed.

## Stack

Python · PyTorch · SmolVLA / flow matching · Isaac Sim 5.1 · PhysX particle cloth · OpenUSD Storm · LeRobot · NumPy · Slurm · Apptainer.

Jose Sanchez · [Portfolio](https://jose-sanchez-portfolio-com.vercel.app/) · [GitHub](https://github.com/joses2017smjh)
