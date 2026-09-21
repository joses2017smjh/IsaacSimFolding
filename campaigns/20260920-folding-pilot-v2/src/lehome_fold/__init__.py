"""Reproduction of Larchenko (2026), "Learning to Fold", on the LeHome Challenge.

The package deliberately splits into two halves:

  pure       splits, labels, awr, recap, calibration, thompson, ckpt
             No torch, no lerobot, no Isaac Sim. This is where the paper's
             actual ideas live, and it is unit-tested.

  glue       value_head, policy_wrap
             Needs torch and lerobot. Thin by design -- the less algorithm
             there is in the half that cannot be tested without a GPU and a
             30 GB environment, the better.
"""

__version__ = "0.1.0"
