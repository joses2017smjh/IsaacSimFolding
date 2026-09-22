# Mixed corrective/raster replay diagnostic

The prior retained-checkpoint audit found no intermediate checkpoints. The only retained 300-step boundary-only checkpoint passed both boundary RMS reductions but failed held-out raster retention.

The mixed run used fixed seed `4242`, `300` steps, one corrective and four fixed raster TRAIN examples per batch, and checkpoints every `25` steps. The vision/VLM remained frozen.

## Pareto trajectory

| checkpoint | step | held-out loss | held-out gate | boundary-5 reduction | boundary-10 reduction | both gates |
|---|---:|---:|---|---:|---:|---|
| step_000025 | 25 | 0.094863 | True | 39.806% | 49.569% | False |
| step_000050 | 50 | 0.085269 | True | 33.444% | 52.489% | False |
| step_000075 | 75 | 0.083661 | True | 30.607% | 55.181% | False |
| step_000100 | 100 | 0.084261 | True | 40.056% | 57.813% | False |
| step_000125 | 125 | 0.082406 | True | 43.350% | 61.700% | False |
| step_000150 | 150 | 0.079753 | True | 50.546% | 65.899% | True |
| step_000175 | 175 | 0.078138 | True | 55.384% | 71.490% | True |
| step_000200 | 200 | 0.079280 | True | 54.547% | 75.659% | True |
| step_000225 | 225 | 0.082715 | True | 45.557% | 79.949% | False |
| step_000250 | 250 | 0.083438 | True | 33.413% | 78.345% | False |
| step_000275 | 275 | 0.082783 | True | 55.234% | 83.786% | True |
| step_000300 | 300 | 0.081291 | True | 56.106% | 83.408% | True |

Earliest checkpoint satisfying both unchanged gates: `step_000150`.

Every boundary prediction used exact snapshot restoration, reconstructed same-RNG inference, and no simulator stepping.

The earliest passing checkpoint was executed with ordinary fresh H10 branches, but settled 4/4 was not recovered at both boundaries. Stop and do not launch larger validation.
