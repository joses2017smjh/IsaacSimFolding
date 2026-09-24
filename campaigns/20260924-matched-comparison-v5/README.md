# Matched comparison v5

v4 ended on one open question. Its candidate `a2-step000250` beat the
baseline at H10 (8/16 vs 2/16, p = 0.027) but missed H50 non-regression by
one episode (3/8 vs 4/8) — against baseline numbers measured in **earlier**
campaigns, on a simulator whose cloth physics is not run-to-run
reproducible. v5 answers it the only fair way: both policies, same rows,
same runner, same Slurm array, same hour.

```
policies   immutable baseline  |  v4 candidate a2-step000250   (both pinned by SHA-256)
rows       v2's 16 development rows (8 poses x H10/H50, seeds 200-207), verbatim
runs       2 per policy per horizon: r1 and r2, each ONE array of 32 tasks in
           which even indices run the baseline and odd the candidate on the
           same row, throttled %8 so every concurrent batch runs four of each;
           both arrays are submitted by the same driver tick
rule       pooled H10 16 vs 16: margin >= 4 AND one-sided Fisher p < 0.05
           pooled H50 16 vs 16: candidate >= baseline (non-regression)
           retention guard and reload: as recorded in v4 for this checkpoint
           all 64 development episodes valid
N          fixed. No third run, whatever r1 and r2 show.
final set  v3's untouched 8 rows, both policies, only if the rule holds
media      the runner writes per-camera GIFs, an MP4 and snapshots for every
           episode; gallery.html shows SETTLED successes only, with every
           other episode counted alongside and every file hashed in
           media/INDEX.json
```

Evaluation only: no training, no search, no new checkpoint. Budget caps in
the manifest are anti-runaway bounds; the GPU-hour constraint was lifted on
2026-09-24.

`scripts/driver.py` owns every submission (restartable, duplicate-safe).
`STATUS.md` is the live page; `REPORT.md` appears when the campaign closes.
