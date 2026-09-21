# Partition check and unwelded-topology follow-up

Checked on 2026-09-21 UTC from the existing repository directory.

- `dgx2` is UP, V100, maximum 7 days; `dgxh` is UP, H100/H200 (including H100 40 GB resources), maximum 2 days.
- Both allow all accounts at the partition level. The user's association is `coehpc|eecs||normal`; both `sbatch --test-only` requests with account eecs were accepted. These were scheduling previews, not job submissions; their printed IDs 21370096 and 21370097 are not campaign jobs.
- Idle CPUs do not imply free GPUs. Most GPUs were allocated; dgx2-5 showed 13/14 allocated and some dgxh nodes had nominally unallocated GPUs but planned allocations. Test-only start estimates did not establish a faster immediately available slot.
- Folding array 21370081 was running on cn-gpu5 in gpu. Pending siblings reported `JobArrayTaskLimit`, and the diagnostic reported `Dependency`, not resource scarcity. No partition change or unrelated cancellation was made.
- Current frozen requests constrain GPUs to `a40|rtx8000`. Adding dgx2/dgxh without changing this would not make their V100/H100/H200 resources eligible. Their physics + Storm rendering compatibility has not been tested. The repository documents a Storm renderer workaround; an RTX-only compatibility argument would not establish whether this exact pipeline works there.

During this check, v3 long-pants index 2 failed before policy actions because PhysX omits both mapping attributes on unwelded cloth. The other completed smokes remain valid evidence. V4 accepts identity mapping only when both remaps are absent, source vertices are all unique, cooked rest positions equal source positions, and live counts match. Partially missing maps, duplicates, count mismatch and source/rest mismatch fail closed. All 15 regression tests passed.

Corrected smoke array **21370102** waits for the preserved v3 diagnostic **21370082**. V4 diagnostic **21370103** waits for that smoke array. This preserves the single-GPU concurrency bound. No additional partition capacity is currently required by this sequence.

The v4 manifest freezes the same checkpoint and 24 matched pilot rows. The pilot remains gated on v4 experiment-level smoke results; old gates/results are not silently transferred.

```bash
squeue -u "$USER" -o "%.18i %.14P %.24j %.10T %.12M %.30R"
cat campaigns/20260920-folding-pilot-v4/submissions.jsonl
python3 campaigns/20260920-folding-pilot-v4/scripts/check_smoke_gate.py
# After the experiment-level gate passes:
python3 campaigns/20260920-folding-pilot-v4/scripts/submit_experiments.py pilot
```
