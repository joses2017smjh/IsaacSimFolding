#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --job-name=lh-value
#SBATCH --cpus-per-task=12
# 48G, down from 96G. This trains a 0.76M head against a FROZEN 450M backbone
# over a 6k-frame subset -- it does not need a training-sized allocation, and
# at 96G it sat PENDING on QOSMaxMemoryPerUser behind this project's own BC
# resume. Asking for what the job needs is what lets it start.
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Stage 2: the value heads, on a FROZEN Stage 1 backbone.
#
# Frozen so that Stage 2 vs Stage 1 is a clean ablation -- identical action
# weights, one mechanism added.
#
# Without ROLLOUT_DIR this trains the progress and future heads and produces a
# success head that CANNOT pass G2: every released demonstration succeeded, so
# the success labels are single-class. That is not a bug in this job, it is a
# property of the released data, and train_value.py says so loudly rather than
# letting G2 discover it.
set -euo pipefail
export LH_ROUTE=${LH_ROUTE:-train}
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh
setup_node_cache
: "${POLICY_PATH:?set POLICY_PATH to the Stage 1 checkpoint}"
# Verified defaults for lerobot 0.4.3: model.embed_prefix exists on both pi05
# and smolvla, and hidden_dim=0 infers the width from the first captured
# feature so it cannot disagree with the model. Override only if
# 12_probe_backbone says this checkpoint is laid out differently.
export FEATURE_PATH=${FEATURE_PATH:-model.embed_prefix}
export HIDDEN_DIM=${HIDDEN_DIM:-0}
# 20,000 steps is the script default and is wrong for this data. Each step
# runs a full 450M forward to drive a 0.76M head over only 6,066 scored
# frames; measured throughput was under 28 steps/min, so 20k needs ~12 h
# against any sane wall clock -- and the save happens AFTER the loop, so a
# TIMEOUT loses the entire run rather than degrading it.
export STAGE_STEPS=${STAGE_STEPS:-3000}
export STAGE_OUT=${STAGE_OUT:-$LEHOME_DATA/outputs/value/$(basename $POLICY_PATH)}
echo "=== node: $(hostname) | $POLICY_PATH -> $STAGE_OUT | $(date) ==="
lh_exec "$REPO/slurm/inner/train_value.sh"
echo "=== done: $(date) ==="
