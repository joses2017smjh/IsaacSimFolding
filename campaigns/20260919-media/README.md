# Folding media campaign — 19 September 2026

This is a private source snapshot for new media jobs. Original scripts, active
evaluations, checkpoints and shared environments remain untouched.

Start with [the audit and handoff](docs/AUDIT_2026-09-19.md),
[current media status](MEDIA_STATUS.md), [local gallery](gallery.html), and
[research directions](docs/RESEARCH_2026-09-19.md).

The two policies are existing SmolVLA adaptations trained with different seeds,
not two architectures. Their chosen checkpoints (steps 1200 and 1000) and file
hashes are frozen in `manifest.json`. Sixteen new single-garment episodes use
both checkpoints, four classes and two recorded development poses per class.
Each executes 600 policy actions after 60 settling steps, uses CUDA particle
physics, and saves true overhead/left-wrist/right-wrist views plus a triptych.
The task prompt matches adaptation training. The checkpoint's 50-action queue
is retained; no untested candidate-ranking or RECAP prompt is introduced.

These illustrative poses are not an unbiased performance benchmark. The
existing jobs `21359573`, `21359574`, `21359575` evaluate the baseline and both
adaptations on their separate matched 96-episode protocol. New media jobs wait
for those three arrays to finish. Only one new GPU task runs at a time, each
limited to 30 minutes (eight GPU-hours maximum across all 16). The first full
episode gates the remaining 15; a failed fold passes the infrastructure gate,
while a crash or missing camera output blocks it. No existing job is edited.

`reference_replays/` preserves eight historical GIFs and their scorer JSONs:
one success and one failure per class. They are explicitly demonstration
replays. New policy successes are never assumed or substituted with replays.

Each new episode writes `outputs/<task>/request.json`, `rollout.log`,
`rollout.json`, `status.json`, four GIFs, and final/first-success PNGs when
applicable. First success and terminal success are separate. Rendering errors
and interrupted runs never count as failed folds. A shell exit handler refreshes
the report even if the container fails to start; a Slurm SIGKILL/node failure
may prevent that handler, so read scheduler logs for incomplete tasks.

The snapshot includes the original project source and a small private challenge
checkout. Large assets, dependencies and checkpoints are mounted read-only;
only this new campaign's outputs and private runtime caches are writable.

Submission receipts live in `submissions.jsonl` after submission. Regenerate the
report with `python3 scripts/media_report.py --campaign .`. The submission
helper defaults to a dry run. Never blindly resubmit if `.submission.lock`
exists: a scheduler submission may have succeeded before the receipt was saved.

CPU validation covers camera identity/timing in decoded GIFs, first-vs-terminal
success, seeded action-queue setup, invalid frame/device inputs, exact result
provenance, missing media, and crash/blocked-job reporting. Full GPU execution
is deliberately deferred to the queued gate.
