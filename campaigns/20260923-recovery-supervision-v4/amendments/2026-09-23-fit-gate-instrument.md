# Amendment: fit-gate instrument correction (2026-09-23, ~23:05Z)

Recorded before any attempt-2 screen, confirmation or final-set result
exists; the only attempt-2 evidence seen is the training trajectory, the
reload, and the fit-gate report this amendment corrects.

## Finding

Attempt 2 was concluded by the fit gate's repair-verification clause
"≥ 1 RMSNorm gain changed on the reloaded candidate". Measured from the
saved fp32 tensors (`attempts/fit_gate_a2-step000250.instrument-artifact.json`
holds the original report):

- 23,750 / 23,760 norm gains **moved in fp32** — the optimizer updates them;
- their maximum |Δ| is 3.5e-4 = **0.18 of a bf16 half-ULP** at magnitude ~1.0,
  so the re-round at evaluator load erases them for ANY correctly working
  optimizer at lr 3.3e-6 × 250 steps — the clause is unpassable by
  construction at this learning rate, the same instrument-artifact class the
  v3 review struck from the original "strictly better single-seed RMS" gate;
- the clause's actual purpose — catching the bf16-frozen signature (11.5%
  changed, as in v3 attempt 1 and the raster fine-tune) — is decided by the
  changed fraction, which measured **59.4% ≥ 50%** (71.6% of ordinary weights
  exceeded their half-ULP in the fp32 file);
- the gate's other clause, multi-seed non-inferiority, **passed**.

## Change

`offline_fit.py`: repair verification = reloaded bf16 changed fraction ≥ 50%
alone; the norm-gain count remains reported. No evidence gate changes: the
≥ 4/8 screen, the confirmation rules, the retention guard, the selection
rule and the final-set reservation are untouched.

## Reopening

The driver had finalized on the baseline as a consequence of the misfiring
clause. `ledger/driver_state.json` is reopened by hand (done=False; the
finalize block and attempt-2 conclusion removed; the `a2.fit` stage cleared
so the corrected gate re-runs, +1 GPU task). Recorded here and in STATUS.
