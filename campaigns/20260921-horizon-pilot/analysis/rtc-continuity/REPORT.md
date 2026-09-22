# RTC temporal-continuity report

## Protocol and provenance

Only pose 3 was tested, at action 5 and action 10. Each branch restored the existing simulator snapshot, preserved the same reconstructed H50 sampler state, and used the same checkpoint, seed, garment, camera path and 600-action branch budget. Controls were the cached H50 suffix and same-RNG fresh replanning.

The RTC branch calls the installed LeRobot SmolVLA flow-matching path with:

* `execution_horizon=10`
* `prefix_attention_schedule=EXP`
* `max_guidance_weight=10.0`
* `inference_delay=0` (offline branch; no actions elapse during the policy call)
* `prev_chunk_left_over` equal to the unconsumed raw H50 chunk at the first boundary, then the unconsumed raw suffix of each RTC-generated chunk at later replans.

The reference source is the existing environment at `/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome51-site/lerobot/policies/rtc/` and the SmolVLA call path in `policies/smolvla/modeling_smolvla.py`. No environment upgrade or separate vendored RTC implementation was needed; the campaign adds only a thin diagnostic adapter and provenance recording.

## Directly measured facts

`comparison.csv` contains the two boundary rows.

* RTC with `prev_chunk_left_over=None` exactly reproduces ordinary inference: raw and postprocessed maximum absolute differences are 0, the post-call Torch/CUDA RNG digests are equal, and the simulator did not step.
* Boundary snapshots restore with zero joint and cloth RMS.
* At action 5, cached H50 reaches 4/4, same-RNG fresh reaches 3/4, and RTC-guided fresh reaches 2/4.
* At action 10, cached H50 reaches 4/4, same-RNG fresh reaches 3/4, and RTC-guided fresh reaches 3/4.
* RTC's first action is close to the cached suffix (L2 0.0414 rad at action 5 and 0.0342 rad at action 10), but deviation grows over the first ten actions (mean absolute 0.0270 and 0.0592 rad; maximum joint errors 0.1804 and 0.5150 rad).

## Decision

Neither pose-3 boundary recovered the preregistered 4/4 criterion under RTC. The conditional full rollout on poses 1, 3, 5 and 7 was therefore not submitted. This pilot does not establish that RTC improves H10 reliability; at action 5 it produced a lower condition score than the same-RNG fresh control.

The next recommendation is to stop this RTC branch and inspect the exact online inference-delay/queue semantics before any further intervention. Do not launch a full rollout, training, RL, RECAP, AWR, DAgger or finetuning from this result.
