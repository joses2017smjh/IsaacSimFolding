# Closed-loop training v2

Rollout-driven AWR fine-tuning of the LeHome folding policy, starting from the
untouched BC baseline.

```
baseline BC
  -> 8 predeclared autonomous H10 rollouts (failures retained)
  -> official terminal geometric reward, conditions_passed / conditions_total
  -> advantage vs the fixed-collection mean, AWR importance resampling
  -> 300-step update, 50% rollout + 50% raster BC anchor
  -> closed-loop evaluation: development H10/H50, then a frozen test set
```

`STATUS.md` is the live page: active job, latest result, blocker, next
milestone. `manifest.json` is the frozen protocol. Do not change either after
a production job has been submitted against them.

## Order of operations

```bash
C=.../campaigns/20260922-closed-loop-training-v2

sbatch $C/slurm/smoke.sbatch "$C"                      # launch gate, end to end
sbatch --array=0-7  $C/slurm/collect.sbatch "$C"       # 8 autonomous rollouts
sbatch $C/slurm/compile.sbatch "$C" iteration1         # gates; exit 4 = degenerate
sbatch $C/slurm/train.sbatch   "$C" iteration1         # refuses unless the gate passed
sbatch --array=0-15 $C/slurm/benchmark.sbatch "$C" <ckpt> <label>
sbatch --array=0-7  $C/slurm/test.sbatch      "$C" <ckpt> <label>
sbatch $C/slurm/boundary.sbatch "$C" <ckpt> <label>
```

If the collection gate fails, the preregistered response is one expansion
draw and a recompile over both, not a threshold change:

```bash
sbatch --array=0-7 $C/slurm/expand.sbatch "$C"
sbatch $C/slurm/compile.sbatch "$C" expanded
```

If it still fails, **stop** and report the unmet requirement. Training on
uniformly weighted failures is the outcome this campaign exists to avoid.

## Tests

```bash
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest tests/ -q   # 13, ~1s
```

They run on a login node and cover the gate logic, the AWR cap, the
frozen-set disjointness and the manifest's preregistration — the decisions
that determine whether 44 GPU tasks are worth spending.

## Artifacts

Rollouts, trajectories, checkpoints and simulator media are local
(`.gitignore`); their hashes and the small CSV/JSON provenance are committed.
