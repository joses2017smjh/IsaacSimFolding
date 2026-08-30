#!/bin/bash
# One garment of the outcome sweep. Split out so the sweep body stays readable.
set -uo pipefail
G=$1; NPZ=$2
WORKSPACE=/nfs/hpc/share/$USER/Humanoid_Lite
REPO=$WORKSPACE/lehome-fold-repro
LEHOME=$REPO/external/lehome-challenge
SIF=$WORKSPACE/container/bhl.sif
VENV=$WORKSPACE/venv
SITE=$WORKSPACE/lehome51-site
export XDG_CACHE_HOME=$WORKSPACE/.cache HOME_OVERRIDE=$WORKSPACE/.home
if mkdir -p "/scratch/$USER/ov-cache" 2>/dev/null; then OV=/scratch/$USER/ov-cache
else OV=$XDG_CACHE_HOME/ov; mkdir -p "$OV"; fi
scratchbind=(); [ -d "/scratch/$USER" ] && scratchbind=(--bind "/scratch/$USER")
apptainer exec --nv --cleanenv --home "$HOME_OVERRIDE" \
  --bind /nfs/hpc/share/$USER "${scratchbind[@]}" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env OV_CACHE="$OV" \
  --env PYTHONPATH="$SITE:$LEHOME/source/lehome:$LEHOME" \
  "$SIF" "$VENV/bin/python" "$REPO/scripts/render/cloth_sim51.py" \
    --lehome "$LEHOME" --garment "$G" --steps "${STEPS:-364}" \
    --out "$NPZ" --traj "${TRAJ}"
