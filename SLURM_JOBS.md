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
| 21078864 | `lh-image` | 2026-08-28 | FAILED | Download was fine this time (26,676,771,349 bytes, `gzip -t` clean). `apptainer build` still refused it: **the tarball is an OCI layout (`blobs/sha256/...`), not a `docker save` archive**, and apptainer reports that mismatch as "gzip: invalid header" — which sends you hunting a corrupt download that isn't corrupt. Needs `oci-archive://`. |
| 21079091 | `lh-image` | 2026-08-28 | pending | Same 26 GB file, `oci-archive://`. No re-download. |
| 21079092 | `lh-data` | 2026-08-28 | pending | Retries with backoff; picks up from the 946 MB + 8.2 GB already on disk. |
| ~~21078864~~ | | | | *(superseded)* | Same pull via `hf` instead of curl, plus a `gzip -t` check before the 25 GB conversion. Tests the one remaining hope that the official image renders where the pip stack does not. |
| 21078925 | `lh-data` | 2026-08-28 | FAILED | Got the assets (946 MB, complete) and 8.2 of 18.9 GB of demonstrations, then **HTTP 429: the Hub rate-limits by IP and this cluster shares one**. A `HF_TOKEN` lifts the anonymous limit and is the real fix; the job now backs off and resumes without one. |

| 21079138 | `lh-render` | 2026-08-28 | queued | **The Isaac Sim render.** Real SO-101 USD at both bimanual poses, a real Release garment with its fabric texture, the challenge's own overhead camera, arms driven by episode 0 of the released demos — on **6.0**, which renders here. |
| 21079173 | `lh-port` | 2026-08-28 | queued | **How big is the 6.0 port?** All 15 isaaclab symbols the task imports resolve on 3.0.0b2, and the particle cloth is raw PhysX/USD, so it is version-independent. This stands the env up and reports the first real failure. |

| 21083423 | `lh-storm51` | 2026-08-28 | queued | **Is the pincer actually a pincer?** What dies on 5.1 is the RTX *delegate*, not 5.1. Physics on 5.1 is fine and the wheels also ship OpenUSD's Storm rasteriser. If Storm draws, the deviation becomes "different renderer" instead of "different physics" — and the geometric success checker is untouched. |

| 21083432-511 | `lh-storm51` | 2026-08-28 | **RESOLVED** | **5.1 CAN render.** Storm draws without the RTX delegate. Five probes: plugin path, `Parameters.rendererPluginId`, EGL context, compatibility profile, `FrameRecorder` readback. |
| 21083532-582 | `lh-stormscene` | 2026-08-28 | **COMPLETED** | The real challenge scene — robots, textured garment, table — rendered on 5.1 via Storm. Caveat: Storm cannot read MDL, so the robots render flat white. See `docs/STORM.md`. |

## What the ledger says about the project

The renderer is the root of everything still blocked. `21067439` settled that
Isaac Sim 5.1 cannot render here and that it is a driver-version problem, not
hardware and not node selection — every GPU driver on this cluster is 595 or
610, and the 6.0 probe that worked ran on the same node class.

`21078863` is the one that matters for making progress anyway: Stage 1 BC and
Stage 2 value-head training read the released LeRobot dataset and never open a
simulator, so they run today on `LH_ROUTE=train`.

**The renderer conclusion has been overturned.** `21083511` established that
5.1 renders through OpenUSD's Storm rasteriser — what segfaults is the RTX
delegate, not 5.1. Since 5.1 is the stack that still has PhysX particle cloth,
the project has a path that keeps official physics AND the official geometric
scorer, deviating only in the images. The remaining question is the size of
that domain gap, and it is measurable.

**Open ask — one command, needs your HF account.** The Hub rate-limits
anonymous traffic by IP and every job on this cluster shares one address. A
read-scoped token lifts it. I cannot create one for you; the jobs are already
wired to pick it up from a file:

```bash
printf %s 'hf_xxxxxxxxxxxx' > /nfs/hpc/share/$USER/Humanoid_Lite/.hf_token
chmod 600 /nfs/hpc/share/$USER/Humanoid_Lite/.hf_token
```

Token at <https://huggingface.co/settings/tokens>, read scope. `slurm/_env.sh`
reads that file into `HF_TOKEN` and forwards it into the container. It is a
FILE rather than a job-script variable so the credential never lands in git, in
a job log, or in `scontrol show job` output — and `.hf_token` is gitignored.
Nothing fails without it; the pull just falls back to anonymous with backoff.
