# Recovery supervision v4 — final report

**Outcome: no preregistered improvement. The untouched baseline checkpoint
remains the deliverable; the final test set was not spent.** Attempt 2 met
every clause of the improvement rule except H50 non-regression, which it
missed by one episode.

## Result

| Measure (8 development poses, settled terminal success) | Baseline | Attempt 2 (`a2-step000250`) |
|---|---|---|
| H10, run 1 | 2/8 | **4/8** |
| H10, run 2 | 0/8 | **4/8** |
| H10 pooled | 2/16 | **8/16** — margin 6, one-sided Fisher p = 0.027 ✓ |
| H50 | 4/8 (mean conditions 3.125) | **3/8** (mean 3.000) ✗ |
| Retention guard, reload, all rows valid | — | ✓ |

The deciding H50 row (dev07) reached the full fold (latched success) and then
relaxed to 3/4 before the settle check. H10 gains were concentrated on the
P_C/P_B poses (dev02, dev05, dev07 settled in both runs); P_A (dev00) stayed
unsolved, as it is for the baseline at every horizon.

## What changed from v3 (all preregistered before any v4 result)

1. **Supervision breadth, 3x.** 384 simulator-validated H10 recovery
   continuations on training-only garments → **81 settled successes (21%)**,
   2,990 labels, coverage gate passed (one row contributed 37 of 81).
2. **Whole-episode anchor.** Anchor frames spread over each demonstration
   instead of its first 8 frames. It worked as designed: anchor fit no longer
   collapsed and retention breached late rather than immediately.
3. **Learning rate as the single attempt-2 factor** (1e-5 → 3.3e-6). The
   guard-passing frontier moved from step 100 to step 250 (≈0.54 → 1.34
   passes over the supervision), and the screen moved from 2/8 to 4/8.

Attempt 1 (lr 1e-5): screen 2/8 — at baseline parity, below the 4/8 bar.

## Amendments (each dated, recorded before the result it could affect)

- `2026-09-23-fit-gate-instrument.md` — the fit gate's "≥1 norm gain changed"
  clause was unpassable at lr 3.3e-6 (measured deltas 0.18 of a bf16
  half-ULP); repair is now verified by the changed fraction (59.4%) alone.
- `2026-09-23-task-proxy.md` — the task-count proxy could not fund a
  two-attempt success path (attempt 2's own plan summed to 62 > 60); raised to
  72. GPU-hours stayed at 28.0 and no evidence gate changed.

## Limitations

- **The baseline comparison is cross-campaign.** Both baseline H10 runs and
  the baseline H50 run were measured in v2/v3, not alongside the candidate.
  Cloth physics is not bitwise reproducible (CLAUDE.md §5), so the one-episode
  H50 gap is within run-to-run noise in either direction. That cuts both
  ways; the preregistered rule stands as ruled.
- 8-pose development set; results are for Pant_Short only.

## Recommended next step (not run)

A fresh campaign that measures **baseline and `a2-step000250` side by side**
(H10 ×2 and H50 ×2 each, same wave), with the rule unchanged, then the v3/v4
unseen final set on a qualifier. Re-running only the failed H50 clause on
this checkpoint would be selective re-testing and is not recommended.

## Budget and provenance

v4: 47 GPU tasks, **7.85 of 28.0 GPU-hours** (all 13 jobs completed).
Cumulative v2–v4: ≈20.0 of 45 GPU-hours. Manifests `ccdc6f5` → `ce834e5` →
`7a0b667`, archived under `manifests/`; ledger in `ledger/slurm-jobs.json`;
driver state and decision log in `ledger/driver_state.json`; bulk
trajectories stay local with hashes in the dataset provenance.
