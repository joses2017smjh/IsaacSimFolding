# Closed-loop training v1

This campaign starts at immutable baseline commit `0e83a7b` after the horizon
pilot established that inference continuity and static H50-suffix correction do
not repair the H10 failure. It uses no H50 action labels.

The bounded first iteration is:

`baseline BC → predeclared H10 autonomous rollouts → official terminal geometric reward → AWR importance resampling + raster-BC anchor → retained checkpoints → fixed H10 and pose-3-boundary evaluation`.

The smoke gate must complete an Isaac rollout, compile its trajectory, perform
one update, save a checkpoint, and load that checkpoint for inference before
the eight-episode collection is submitted. Raw simulator media, trajectories,
and checkpoint tensors are intentionally local artifacts rather than Git
objects; their hashes and small CSV/JSON provenance are retained in `analysis/`.
