#!/bin/bash
#SBATCH --job-name=lh-pi05
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="l40s|h100|h200|h200-70g|a40"
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Stage 1 BC on pi0.5 -- THE PAPER'S ACTUAL BASE MODEL.
#
# Everything scored so far used SmolVLA (450M), an affordable stand-in. pi0.5
# is ~2.3B, so the config drops batch_size to 8; at that size 30,000 steps will
# not fit in one wall clock. This is SEGMENT 1. It checkpoints every 5,000
# steps and `slurm/74_resume_bc.sbatch` resumes any checkpointing run, so the
# schedule continues across jobs rather than restarting.
#
# Needs the network on first run: the pi0.5 base weights are not in the cache
# (unlike SmolVLM2, which is). HF_TOKEN comes from $WORKSPACE/.hf_token via
# _env.sh -- a file, never a job-script variable, so the credential stays out
# of git, out of logs and out of `scontrol show job`. So no HF_HUB_OFFLINE here,
# unlike the rollout jobs.
#
# Sequencing note: this is the most expensive item queued, and its value
# partly depends on whether the SmolVLA resume (21124825) shows that more
# training changes behaviour at all. It is queued anyway because the paper's
# base is worth having independently -- but if GPU minutes get tight, this is
# the one to cancel first.

set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
export CONFIG=${CONFIG:-configs/bc_pi05.yaml}
export SEED=${SEED:-0}
echo "=== node: $(hostname) | $CONFIG seed=$SEED | $(date) ==="
echo "    HF_TOKEN present: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo NO)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
lh_exec "$REPO/slurm/inner/train_bc.sh"
echo "=== done: $(date) ==="
