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

| 21090714-916 | `lh-cloth51` | 2026-08-29 | diagnostic chain | Standing the cloth env up on 5.1. Six distinct blockers: camera CLASS stub, zero-image stub, `open3d`, `_get_initial_info()` ordering, `pynput`, and a repeat of the Kit-swallows-`sys.exit` false success. |
| 21090922 | `lh-cloth51` | 2026-08-29 | **CLOTH SIMULATES** | `sim.device=cuda:0`. 0.330 m displacement, garment top fell 0.178 m, official checker executed. PhysX particle cloth needs GPU dynamics; `--device cpu` is policy inference. |
| 21090939-941 | `lh-fold` | 2026-08-29 | **COMPLETED** | 183 frames of the simulated cloth rendered through Storm → `docs/img/cloth_fold.gif`. 20.7% of pixels change first-vs-middle, measured. |
| 21091167 | `lh-cloth51` | 2026-08-29 | COMPLETED | Replay `action` not `observation.state` — 0.225 m, checker `False`. |
| 21091179 | `lh-cloth51` | 2026-08-29 | COMPLETED | Actions + garment pinned to its recorded `object_initial_pose` — 0.325 m, checker `False`. Open-loop replay does not reproduce the fold; see `docs/CLOTH.md`. |

| 21093704 | `lh-cloth51` | 2026-08-29 | COMPLETED | Re-ran the sim recording **robot link poses** (365x7x3 per arm) so the animation can show the arms, not just a garment moving itself. |
| 21093740 | `lh-fold` | 2026-08-29 | FAILED | Link names were written as a pickled object array and would not read back. Fixed on the read side with the known body order rather than re-running a 45 s simulator boot for a constant. |
| 21093748 | `lh-fold` | 2026-08-29 | COMPLETED | **Robot + cloth animation.** 14 link transforms across two SO-101s. Each link references `/visuals/<link>` explicitly — the USD's meshes are a *sibling* of its defaultPrim, which is why the arms had been rendering flat white. |
| 21093756 | `lh-fold` | 2026-08-29 | COMPLETED | Reframed on the whole workspace; a camera fitted to the cloth bounds was clipping an arm out of shot. → `docs/img/robot_fold.gif` |
| 21093764 | `lh-sweep` | 2026-08-29 | COMPLETED | **Four garments, four checker-verified failures.** Cloth moved 0.22–0.33 m each. Open-loop replay does not fold; see `docs/CLOTH.md`. |
| 21093929/30 | `lh-bc` | 2026-08-29 | FAILED | First Stage 1 attempt. Two bugs: `lerobot.scripts.train` does not exist in 0.4.3 (it is `lerobot_train`), and **torchcodec cannot load** — `libnppicc.so.12` missing, no CUDA NPP runtime. RGB is video-encoded so a decoder is not optional. |
| 21093962/63 | `lh-bc` | 2026-08-29 | CANCELLED | Relaunched with `video_backend: pyav`, then cancelled: two 24 h requests reserve 2,880 GPU-min and parked both behind `MaxGRESRunMinsPerAccount` — which was also blocking the user's own `v2-train` array. |
| 21093972 | `lh-bc` | 2026-08-29 | **RUNNING** | **Stage 1 BC, SmolVLA.** 6 h wall clock (queue reason dropped to plain `Resources`). 265,798 frames / 1,000 episodes loaded, 450M total / 100M trainable, batch 32. Checkpoints every 5k steps. |
| 21094000 | `lh-sik` | 2026-08-29 | **COMPLETED** | **Storm renders inside a live Kit process.** `GL 4.6.0 alongside Kit`, `Record() -> True`. So the observation pipeline is feasible in-process — a trained policy can actually be rolled out here. |

## What the ledger says about the project

The renderer is the root of everything still blocked. `21067439` settled that
Isaac Sim 5.1 cannot render here and that it is a driver-version problem, not
hardware and not node selection — every GPU driver on this cluster is 595 or
610, and the 6.0 probe that worked ran on the same node class.

`21078863` is the one that matters for making progress anyway: Stage 1 BC and
Stage 2 value-head training read the released LeRobot dataset and never open a
simulator, so they run today on `LH_ROUTE=train`.

**The evaluation path is now open.** `21094000` showed Storm rendering inside a
live Kit process, which is the missing piece: the env feeds the policy three RGB
images, the RTX cameras segfault on 5.1, and the current stub returns zeros — a
VLA looking at black frames cannot fold anything. In-process Storm means
official physics, the official geometric scorer, and only the images deviating.

**Two conclusions in this ledger were overturned by later jobs**, and both are
worth reading in order rather than trusting the earliest confident statement:
5.1 renders (via Storm), and 5.1 simulates cloth (on GPU physics). The
"no stack can do both" claim was wrong on both halves.

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
