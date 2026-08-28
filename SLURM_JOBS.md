# Job ledger

Every job submitted from this repo, updated at submission. Kept because a
deliverable went missing once when it was not.

| job | name | date | state | what it answered |
|---|---|---|---|---|
| 21067439 | `lh-rtx51` | 2026-08-27 | **SEGFAULT (139)** | **G0-a: RTX on Isaac Sim 5.1. It does not render on this cluster.** A40 / driver 595.71.05, `cn-r-2`, died in `createHydraEngine` -> `librtx.scenedb.plugin.so`. Verdict in `results/rtx51_probe_borrowed.txt`. |
| 21067529 | `lh-image` | 2026-08-27 | FAILED | Download of the official image died at 5.7/24.8 GB, `curl: (18)`. `--retry` does not cover exit 18. |
| 21067642 | `lh-image` | 2026-08-27 | FAILED | Retry loop pulled 25 GB but the file was not a valid gzip: the CDN answered the resume with 200 not 206, curl threw away 12 GB client-side and spliced the rest onto the partial. Size matched, content did not. |
| 21067643 | `lh-trainenv` | 2026-08-27 | FAILED | Wanted `lehome.sif`, which is not built. The train route needs no Isaac Sim, so it now reuses `bhl.sif`. |
| 21078863 | `lh-trainenv` | 2026-08-28 | **COMPLETED** | torch 2.7.0+cu128, lerobot 0.4.3, transformers 4.57.6, CUDA available, 7.6 GB. **Stages 1-2 are unblocked.** |
| 21078864 | `lh-image` | 2026-08-28 | pending | Same pull via `hf` instead of curl, plus a `gzip -t` check before the 25 GB conversion. Tests the one remaining hope that the official image renders where the pip stack does not. |

## What the ledger says about the project

The renderer is the root of everything still blocked. `21067439` settled that
Isaac Sim 5.1 cannot render here and that it is a driver-version problem, not
hardware and not node selection — every GPU driver on this cluster is 595 or
610, and the 6.0 probe that worked ran on the same node class.

`21078863` is the one that matters for making progress anyway: Stage 1 BC and
Stage 2 value-head training read the released LeRobot dataset and never open a
simulator, so they run today on `LH_ROUTE=train`.

`21078864` is the last branch before the fallback. If the organizers' image
also segfaults — likely, since `--nv` injects the host driver — the remaining
option is porting LeHome to the 6.0 stack as a stated deviation.
