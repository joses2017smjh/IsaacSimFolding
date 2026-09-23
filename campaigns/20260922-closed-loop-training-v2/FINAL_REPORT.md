# Final report — closed-loop training v2

Generated 2026-09-23T09:27:12Z by scripts/driver.py from ledger/driver_state.json.

## Development benchmark (8 poses; development data, used for selection)

| checkpoint | H10 settled | H50 settled | H10 mean conditions | eligible | notes |
|---|---|---|---|---|---|
| baseline (matched, this campaign) | 2/8 | 4/8 | 3.0 | — | |
| iter1-step000300 | 0/8 | 4/8 | 2.125 | yes | eligible |
| iter2-step000300 | 0/8 | 3/8 | 2.5 | no | H50 regression: 3 < matched baseline 4 |
| iter3-step000300 | 1/8 | 4/8 | 2.375 | yes | eligible |

## Selected checkpoint

`baseline` — no candidate qualified; the untouched baseline is retained

`/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_raster_ft_full`

## Frozen test set (reserved; excluded from all training and selection)

- baseline: H10 settled 0/8 (mean conditions 3.0)

## Target

- primary: development H10 settled >= 6/8 — **NOT MET**
- stretch: development H10 settled = 8/8 — **NOT MET**

