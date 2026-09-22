# Fixed-RNG replanning diagnostic

This directory isolates inference sampling from observation-conditioned replanning at the first action-5 and action-10 boundaries. It uses the same four matched diagnostic cases previously successful under H50. It does not change model weights and does not launch training or a broad sweep.

The capture records the Torch and CUDA sampler states immediately after the frozen episode seed is applied, restores each boundary snapshot, generates a first fresh chunk from that state, executes the same-RNG branch, and records a small predetermined seed panel (101, 102, 103, 104). The cached H50 executed suffix remains the control. The original pilot did not persist its pre-call RNG bytes, so the report labels this state as a reproducible reconstruction from the episode seed rather than claiming exact recovery of the historical byte state.

* `plan-comparisons.csv` contains one row per pose and boundary.
* `seed-panel.json` contains the raw same-RNG and four-seed action chunks and metrics.
* `rng-audit.md` documents the PRNG and SmolVLA flow-matching noise path.
* `REPORT.md` separates measured facts, causal evidence, descriptive seed sensitivity, and unresolved limitations.
