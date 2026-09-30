# scripts/ — stage-era entry points

These are the entry points of the stage-era pipeline: BC, value head, rollout, Thompson and the
official-evaluator probes. They are driven by `slurm/`. **Nothing a campaign runs executes from here.**
Every campaign since 2026-09-19 runs campaign copies instead, and pins their SHA-256 in its manifest:
either its own snapshot in `campaigns/<campaign>/scripts/`, or an earlier campaign's pinned copy (v2 ran
the horizon pilot's runner).

Checked on 2026-09-30 against the latest campaign that carries each file:
- 24 of 26 tracked scripts here are byte-identical to that copy.
- 2 are stale forks. Use the campaign copy for anything current:

| root file | lines | executed copy | lines |
|---|---|---|---|
| `scripts/finetune_rasterised.py` | 234 | `campaigns/20260921-horizon-pilot/scripts/finetune_rasterised.py` | 519 |
| `scripts/render/policy_rollout51.py` | 600 | `campaigns/20260928-fresh-seed-remeasure-v9/scripts/render/policy_rollout51.py` | 3504 |

The rollout runner kept growing inside the campaigns:
- the horizon pilot added telemetry, geometry and media;
- v7 added passive branch telemetry.
The fine-tune script's current form is the horizon pilot's. The v2-v8 trainers are separate campaign
scripts (`rollout_weighted_finetune.py`).

The root copies are left unchanged on purpose, because the stage-era `slurm/` jobs still point at them.
Start new work from the newest campaign snapshot, not from here.
