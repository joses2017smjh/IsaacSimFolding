# Anchor diagnostic v8

A single-variable diagnostic chosen by the verified v7 diagnosis
(`../20260925-depth-qualified-v7/analysis/diagnosis/diagnosis.json`).

**The failure it targets.** Every fine-tune so far (a2, pb1, dq1) loses P_B
at H50 (0/12 vs the baseline's 9/12), and that single cell decides the H50
clause. The cause is a shift all fine-tunes share in the first open-loop H50
chunk: from the identical first observation they open only the right
gripper (38/48 episodes vs the baseline's 6/48) and lift the cloth 150-200
actions later; at dev03 they pinch inboard and never lift. The one component
every fine-tune shares byte for byte is the **BC anchor**, half of every
batch: 12 of its 16 demonstrations are tops, and every right-only frame-0
target in it comes from a top.

**The change, and only this change.**

```
anchor    16 pants demonstrations (8 Pant_Short + 8 Pant_Long, the lowest-
          numbered storm_capture100 episode per garment, excluding retention,
          final-set and anchor-fit episodes) instead of v7's 12 tops + 4 pants;
          same size, frame schedule (whole-episode, 12 frames) and 50% share
corpus    v7's depth-qualified corpus, byte for byte (no search, no compile)
recipe    v7's verbatim from the untouched baseline; latest guard-passing
          checkpoint; reload; fit gate
```

**How it is judged.**

1. **Mechanistic endpoints first** (`scripts/endpoints.py`, preregistered):
   the anchor hypothesis is supported iff the candidate opens both grippers
   in the first H50 chunk on >= 5 of 8 H50 rows AND lifts dev03 above 5 cm by
   action 150 (run r1). Anything else refutes the anchor as the driver.
2. **Development screen:** v5's matched rule on the 16 development rows,
   baseline and candidate interleaved, two runs each. Four fine-tunes were
   already scored on these rows, so this is a screen only.
3. **Confirmatory gate, only if the screen passes:** the untouched final set
   (8 poses, mesh PS_049, 7 of 8 poses absent from dev and every search) at
   H10 and H50 with 3 fresh seeds per policy in one interleaved array (96
   episodes); a positive claim needs the same rule on 24 vs 24. Its power is
   stated in advance as low (at most ~27%); it is spent once.

The baseline stays the deliverable unless both the screen and the
confirmatory gate pass. No GPU smoke is needed: the runner and trainer are
v7's unchanged; the anchor is data and the driver/runner-phase changes are
covered by `tests/test_v8.py`. `STATUS.md` is the live page.
