# Replacement folding media gate — 20 September 2026

Gate `21360435` failed before simulation because LeHome inferred its asset root
from the containing Git repository. The USD exists under the shared asset pack,
not `lehome-fold-repro/Assets`.

The first repair (21367674) loaded the robot but exposed the same inferred root
for the apartment scene. This snapshot sets `LEHOME_ASSETS_ROOT` before package
registration and also binds **both robot spawn configurations** to the explicit
`--assets` root. CPU preflight checks the robot, garment references, frozen code,
and checkpoint hashes. It does not prove that GPU rendering succeeds.

Only **one** replacement job is submitted: adaptation seed 0, short top pose 2,
600 policy actions after 60 settling steps, 1 GPU, 8 CPUs, 64 GB, 30 minutes.
It must finish the episode and save overhead, left wrist, right wrist and triptych
GIFs. A valid failed fold passes the media gate; a crash does not.

The original snapshot and jobs `21360436` and `21360437` are unchanged.
The original array still depends on failed gate `21360435`; a new gate cannot
satisfy that dependency. No replacement array is submitted in this repair.

## Reproduce and inspect

These commands require the shared HPC asset pack, checkpoint and container paths
in `manifest.json` and `slurm/media.sbatch`. This is an execution snapshot, not a
portable installation of Isaac Sim. Large assets and weights are not committed.
After cloning, link each of `objects`, `robots`, `textures`, and `scenes` from the
manifest's asset root under `external/lehome-challenge/Assets/`.

```bash
python3 scripts/preflight_assets.py --campaign "$PWD"
python3 scripts/submit_gate.py --campaign "$PWD"          # verify only
python3 scripts/submit_gate.py --campaign "$PWD" --submit # one job; receipt prevents repeats
python3 scripts/media_report.py --campaign "$PWD"
python3 -m unittest discover -s tests -p test_asset_paths.py
```

Submission identity and checked robot SHA-256 live in `gate_submission.json`.
Read `outputs/*/status.json` and scheduler logs before interpreting any outcome.
Historical GIFs in the parent repository are not output from this new gate.

The frozen upstream environment and renderer retain their original license notices.
Only `asset_paths.py`, its tests, preflight/submission, and the robot binding are
new behavior relative to the 19 September media snapshot.
