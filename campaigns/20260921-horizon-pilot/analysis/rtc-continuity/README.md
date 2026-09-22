# Pose-3 RTC temporal-continuity diagnostic

This is the smallest inference-only temporal-continuity test after the fixed-RNG result. It runs only pose 3 at the validated action-5 and action-10 snapshots. Cached H50 and same-RNG fresh replanning remain controls.

The implementation uses the installed LeRobot SmolVLA RTC reference directly through `SmolVLAPolicy.init_rtc_processor()`. It uses the leftover raw H50 action chunk as `prev_chunk_left_over`, execution horizon 10, EXP prefix weights, guidance weight 10, and zero offline inference delay. No weights or environment packages were changed.

`comparison.csv` is the compact result table. The raw branch capture is retained at `outputs/rtc_dev03_h50_Pant_Short_Seen_3/replan-causality.json` (ignored runtime output). The completed job was Slurm `21396459`; `21396457` failed before simulator startup because of an argument-ordering error and produced no experiment data.
