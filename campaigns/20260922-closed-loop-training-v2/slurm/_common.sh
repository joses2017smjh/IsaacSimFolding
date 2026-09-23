# Shared apptainer invocation for this campaign. Source it; do not execute it.
#
# These jobs deliberately do NOT use slurm/_env.sh. That file carries four
# environment routes and a large forwarded-variable list; only the `isaac`
# route has ever produced a working rollout here, so the one combination that
# works is spelled out explicitly instead of selected at runtime.
workspace=/nfs/hpc/share/sanchej7/Humanoid_Lite
repo="$workspace/lehome-fold-repro"
pilot="$repo/campaigns/20260921-horizon-pilot"

lh_runtime() {
  runtime="$campaign/runtime/${SLURM_JOB_ID}_${1}"
  mkdir -p "$runtime/home" "$runtime/cache" "$runtime/ov" "$runtime/nv" "$runtime/tmp"
  export APPTAINER_CACHEDIR="$runtime/apptainer-cache"
  libs=("$workspace"/venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*)
  mask=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then mask+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
}

# Full Isaac Sim environment: needs the lehome checkout and USD plugin path.
lh_isaac() {
  apptainer exec --nv --cleanenv --home "$runtime/home" \
    --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
    --bind "$campaign:$campaign:rw" \
    --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
    --env "XDG_CACHE_HOME=$runtime/cache" --env "OV_CACHE=$runtime/ov" \
    --env "CUDA_CACHE_PATH=$runtime/nv" --env "TMPDIR=$runtime/tmp" \
    --env "HF_HOME=$workspace/.cache/huggingface" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
    --env "PYTHONPATH=$workspace/lehome51-site:$pilot/external/lehome-challenge/source/lehome:$pilot/external/lehome-challenge:$pilot/src:$repo/src" \
    --env "PXR_PLUGINPATH_NAME=${libs[0]}/bin/usd" "${mask[@]}" \
    "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u "$@"
}

# Torch/lerobot only: training, compilation and checkpoint loading.
lh_torch() {
  apptainer exec --nv --cleanenv --home "$runtime/home" \
    --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
    --bind "$campaign:$campaign:rw" \
    --env "XDG_CACHE_HOME=$runtime/cache" --env "CUDA_CACHE_PATH=$runtime/nv" \
    --env "TMPDIR=$runtime/tmp" --env "HF_HOME=$workspace/.cache/huggingface" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "PYTHONPATH=$workspace/lehome51-site:$repo/src" "${mask[@]}" \
    "$workspace/container/bhl.sif" "$workspace/venv/bin/python" -u "$@"
}
