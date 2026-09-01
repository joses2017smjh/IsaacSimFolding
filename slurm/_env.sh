#!/bin/bash
# Shared paths + container invocation for every job in this project.
# Source this; do not execute it.
#
# Lifted wholesale from bhl-robustness-ladder/slurm/_env.sh, because the
# Slurm/apptainer discipline is the one thing that carries over between these
# two projects. The robot, the simulator stack and the learning paradigm do not.

WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro

# Two ways to get the LeHome environment, and the choice is not cosmetic.
#
#   LH_ROUTE=official (default)
#     The organizers' own Docker image, converted to apptainer:
#       huggingface.co/datasets/lehome/docker  ->  lehome-challenge.tar.gz  (26.7 GB)
#       apptainer build lehome-official.sif docker-archive://<tarball>
#     The venv is BAKED IN at /opt/lehome-challenge/.venv. This is the exact
#     environment the leaderboard was scored in, which is the whole point of
#     reproducing this paper rather than reimplementing it -- it removes every
#     resolver question (the forked IsaacLab, the pinned isaacsim 5.1.0, the
#     numpy/packaging overrides) in one step.
#
#   LH_ROUTE=source
#     `uv sync` against lehome-challenge/pyproject.toml into a venv on Lustre,
#     plus the FORKED Isaac Lab that the official install requires:
#       third_party/IsaacLab <- github.com/lehome-official/IsaacLab (BSD-3)
#     Slower to stand up and it can drift from the image, but it is the only
#     route where the environment code is editable, which Stage 3 will need for
#     the async rollout workers.
#
# Start official, prove G0 on it, and only move to source when something has to
# be edited. Any number produced on `source` must be re-checked against
# `official` before it is reported.
LH_ROUTE=${LH_ROUTE:-official}
case "$LH_ROUTE" in
    official)
        SIF=$WORKSPACE/container/lehome-official.sif
        LEHOME=/opt/lehome-challenge
        PY=/opt/lehome-challenge/.venv/bin/python
        export UV_PROJECT_ENVIRONMENT=/opt/lehome-challenge/.venv
        ;;
    train)
        # Training-only. lerobot + torch, NO isaacsim.
        #
        # This exists because of what the renderer probe found. Isaac Sim 5.1
        # cannot render on this cluster, which blocks EVALUATION and Stage 3
        # rollout collection -- but Stage 1 BC and Stage 2 value-head training
        # read the released LeRobot dataset and never open a simulator at all.
        # Splitting the stack means that block costs the training half nothing.
        #
        # A checkpoint trained here is a normal LeRobot checkpoint and is
        # scored, later, by the official harness on whichever route can render.
        #
        # Reuses the ALREADY BUILT bhl.sif rather than needing lehome.sif. It
        # provides glibc 2.35, the CUDA/GL userspace and uv, and this route
        # installs no Isaac Sim -- so there is nothing lehome.sif would add.
        # One less 2-hour container build between here and a trained policy.
        SIF=${LH_TRAIN_SIF:-$WORKSPACE/container/bhl.sif}
        LEHOME=$REPO/external/lehome-challenge
        PY=$WORKSPACE/venv-lehome-train/bin/python
        export UV_PROJECT_ENVIRONMENT=$WORKSPACE/venv-lehome-train
        ;;
    source)
        SIF=$WORKSPACE/container/lehome.sif
        LEHOME=$REPO/external/lehome-challenge
        PY=$WORKSPACE/venv-lehome/bin/python
        export UV_PROJECT_ENVIRONMENT=$WORKSPACE/venv-lehome
        ;;
    *) echo "unknown LH_ROUTE=$LH_ROUTE (expected official, source or train)" >&2; return 1 2>/dev/null || exit 1 ;;
esac

# Assets and datasets live on Lustre and are bind-mounted in, NOT baked into
# the image: the garment USDs and the 265,798-frame demonstration set are data,
# they are big, and re-pulling them on every image rebuild is wasted bandwidth.
# LeHome resolves them relative to cwd, so jobs cd into $LEHOME_DATA.
LEHOME_DATA=$WORKSPACE/lehome-data

# Isaac Sim 5.1 on Python 3.11 -- the SAME stack as the locked BHL v51 venv,
# NOT the 6.0 stack. See docs/STAGE0.md; this inversion is the central finding
# of Stage 0 and the central risk of the project, because 5.1 is the version
# whose RTX renderer segfaults on this cluster and LeHome needs RGB.

# /nfs/stak home has ~15GB free. Every cache goes to Lustre instead.
export UV_CACHE_DIR=$WORKSPACE/.uv-cache
export UV_PYTHON_INSTALL_DIR=$WORKSPACE/.uv-python
export XDG_CACHE_HOME=$WORKSPACE/.cache
export HOME_OVERRIDE=$WORKSPACE/.home
export HF_HOME=$WORKSPACE/.cache/huggingface

# Hugging Face token, if one has been dropped in. The Hub rate-limits
# anonymous traffic BY IP and every job on this cluster shares one address --
# the first data pull died with HTTP 429 after 8.2 of 18.9 GB. A read-scoped
# token lifts that.
#
# It is read from a file rather than baked into a job script so it never lands
# in git, in a log, or in `scontrol show job` output:
#
#   printf %s 'hf_xxxxxxxx' > $WORKSPACE/.hf_token && chmod 600 $WORKSPACE/.hf_token
#
# Nothing here fails without it; the pull just falls back to anonymous with
# backoff.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$WORKSPACE/.hf_token" ]; then
    HF_TOKEN=$(tr -d '[:space:]' < "$WORKSPACE/.hf_token")
    export HF_TOKEN
fi

mkdir -p "$LEHOME_DATA" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" \
         "$HOME_OVERRIDE" "$HF_HOME"

# Isaac Sim writes large shader/ext caches at runtime. On Lustre these are
# painfully slow, so point them at node-local scratch inside each job. Not every
# partition has /scratch; falling back to Lustre is slower but works.
setup_node_cache() {
    local base=/scratch/$USER
    if mkdir -p "$base/ov-cache" "$base/nv-computecache" 2>/dev/null; then
        export OV_CACHE=$base/ov-cache
        export CUDA_CACHE_PATH=$base/nv-computecache
    else
        echo "note: /scratch/$USER unavailable on $(hostname), caching to Lustre" >&2
        export OV_CACHE=$XDG_CACHE_HOME/ov
        export CUDA_CACHE_PATH=$XDG_CACHE_HOME/nv
        mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
    fi
}

# Because --cleanenv wipes the host environment, ANY variable an inner script
# needs must be forwarded explicitly.
LH_FORWARD_VARS="STAGE_STEPS HF_TOKEN FETCH_DEPTH CONFIG VALUE_PATH FEATURE_PATH HIDDEN_DIM POLICY_PATH_OVR ROLLOUT_DIR SHARED_DIR WORKER_ID N_CANDIDATES STAGE_OUT GARMENT_TYPE NUM_EPISODES MAX_STEPS POLICY_TYPE POLICY_PATH DATASET_ROOT PROBE_OUT OUT_CSV SEED DEVICE ENABLE_CAMERAS HEADLESS HF_HOME STEP_HZ RUN_NAME"

lh_exec() {
    local envargs=()
    local v
    for v in $LH_FORWARD_VARS; do
        # Passing an empty value is not the same as not passing it.
        [ -n "${!v:-}" ] && envargs+=(--env "$v=${!v}")
    done
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && envargs+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    [ -n "${SLURM_JOB_GPUS:-}" ] && envargs+=(--env "SLURM_JOB_GPUS=$SLURM_JOB_GPUS")

    local scratchbind=()
    [ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")

    # LeHome resolves BOTH the garment assets and its own particle config
    # relative to cwd:
    #   garment_cfg_base_path = "Assets/objects/Challenge_Garment"
    #   particle_cfg_path     = "source/lehome/.../particle_garment_cfg.yaml"
    # so cwd has to be the checkout root AND the data has to appear inside it.
    # On the official route that root is read-only inside the image, so the
    # data is bind-mounted into place rather than copied or symlinked. Keeping
    # 20 GB of demonstrations out of a 25 GB image is the point.
    local databind=()
    local d
    for d in Assets Datasets outputs; do
        [ -d "$LEHOME_DATA/$d" ] && databind+=(--bind "$LEHOME_DATA/$d:$LEHOME/$d")
    done

    apptainer exec --nv --cleanenv \
        --home "$HOME_OVERRIDE" \
        --bind /nfs/hpc/share/$USER \
        "${scratchbind[@]}" \
        "${databind[@]}" \
        --env UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
        --env UV_CACHE_DIR="$UV_CACHE_DIR" \
        --env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        --env HF_HOME="$HF_HOME" \
        --env OV_CACHE="${OV_CACHE:-$XDG_CACHE_HOME/ov}" \
        --env CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv}" \
        --env LEHOME="$LEHOME" \
        --env LEHOME_DATA="$LEHOME_DATA" \
        --env REPO="$REPO" \
        --env WORKSPACE="$WORKSPACE" \
        --env PY="$PY" \
        "${envargs[@]}" \
        "$SIF" bash "$@"
}

# Strip inherited Slurm job context so sbatch from inside an allocation submits
# new work rather than trying to run as a step of the current job.
slurm_clean() {
    env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NODEID \
        -u SLURM_TASKS_PER_NODE -u SLURM_CPUS_ON_NODE -u SLURM_JOB_CPUS_PER_NODE \
        -u SLURM_TRES_PER_TASK -u SLURM_JOB_GPUS -u SLURM_GPUS_ON_NODE \
        -u SLURM_JOB_NUM_NODES -u SLURM_MEM_PER_NODE -u SLURM_JOB_PARTITION \
        -u SLURM_EXPORT_ENV -u SLURM_TASK_PID -u SLURM_LOCALID -u SLURM_PROCID \
        -u SLURM_STEP_ID -u SLURM_STEPID -u SLURM_SUBMIT_DIR -u SLURMD_NODENAME \
        "$@"
}
