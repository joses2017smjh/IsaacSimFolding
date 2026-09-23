# Recovery supervision v4

Successor to v3, targeting exactly the limitation its report names:
**supervision breadth** paired with a **retention-compatible recipe**.

```
recovery search, 3x v3:  8 rows x 6 roots x 8 fixed H10 candidate seeds
  = 384 executed continuations, every attempt recorded, settled-only labels
  coverage gate scaled with it: >= 18 successes, >= 10 roots, >= 4 rows
training: fp32 master weights (v3's repair), effective batch 32,
  THE recipe factor: whole-episode anchor (frames spanning approach/grasp/
  fold/release) replacing the episode-start snap that protected nothing the
  retention guard measures; checkpoints every 25 steps, preregistered rule:
  LATEST guard-passing checkpoint, else no candidate
evaluation: pooled preregistered baseline (2/16 over two matched runs -- no
  new baseline spend) -> fit gate -> screen >= 4/8 -> confirm (2nd H10 + H50)
  -> v3's still-unseen final set, spent only on a qualifying candidate
budget: HOURS-PRIMARY -- 28.0 of the 32.8 GPU-hours remaining in the
  original 45-hour allocation; 60 tasks as an anti-runaway proxy, with the
  cumulative task overage vs the informal 170 declared upfront
```

H10-only candidates by design: v3 measured H10 and H50 branches equally
successful (16/64 each), and H10 labels are the deployment protocol's own
closed-loop behaviour — no stale open-loop suffixes. v3's mixed-horizon
labels are deliberately not imported.

`STATUS.md` is the live page; `scripts/driver.py` owns every submission.
