#!/bin/bash
#SBATCH --job-name=lh-image
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Pull the organizers' Docker image and convert it to apptainer.
#
# This is the route that makes "success rate" mean the leaderboard's number
# rather than one this repo resolved for itself. The image is public and
# ungated on the Hub and carries the forked IsaacLab, the pinned isaacsim
# 5.1.0 and the numpy/packaging overrides already resolved:
#
#   huggingface.co/datasets/lehome/docker/lehome-challenge.tar.gz   26.7 GB
#
# Disk: ~27 GB for the tarball plus ~25 GB for the .sif, so budget ~52 GB peak
# and ~25 GB steady once the tarball is deleted. /nfs/hpc/share had 448 GB free
# and no per-user block quota when this was written -- the constraint is the
# shared filesystem, not an allocation.
#
# No GPU: this only downloads and repacks.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro/slurm/_env.sh

TARBALL=$WORKSPACE/container/lehome-challenge.tar.gz
URL=https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz
# Exact size from the Hub API; the loop below runs until the file is this big.
EXPECT_BYTES=26676771349
export APPTAINER_CACHEDIR=$XDG_CACHE_HOME/apptainer
export APPTAINER_TMPDIR=$WORKSPACE/.cache/apptainer-tmp
mkdir -p "$WORKSPACE/container" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "=== node: $(hostname) | $(date) ==="
df -h /nfs/hpc/share/$USER | tail -1

# Download with `hf`, not curl.
#
# The first attempt used curl -C - and produced a 25 GB file that was the right
# SIZE and not a valid gzip. The Hub redirects to a CDN that answered the
# resume request with 200 instead of 206, so curl skipped 12 GB client-side
# ("Throwing away 12003866524 bytes") and spliced the remainder onto the
# earlier partial. A length check cannot catch that; only the content can.
#
# huggingface_hub does ranged, chunked, hash-verified downloads and resumes
# correctly, which is the whole problem solved by using the tool that knows the
# protocol. uv is baked into bhl.sif, so uvx supplies it without an install.
echo "--- downloading via huggingface_hub ---"
apptainer exec --cleanenv --home "$HOME_OVERRIDE" --bind /nfs/hpc/share/$USER \
    --env HF_HOME="$HF_HOME" --env UV_CACHE_DIR="$UV_CACHE_DIR" \
    --env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
    "$WORKSPACE/container/bhl.sif" \
    uvx --from huggingface_hub hf download lehome/docker lehome-challenge.tar.gz \
        --repo-type dataset --local-dir "$WORKSPACE/container"

got=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
if [ "$got" -ne "$EXPECT_BYTES" ]; then
    echo "FATAL: got $got bytes, expected $EXPECT_BYTES" >&2
    exit 1
fi
echo "--- $got bytes, size matches ---"

# Verify it is actually a gzip stream before handing 25 GB to apptainer. The
# previous failure got all the way to `apptainer build` before anything noticed.
if ! gzip -t "$TARBALL" 2>/dev/null; then
    echo "FATAL: $TARBALL is not a valid gzip stream" >&2
    exit 1
fi
echo "--- gzip integrity OK ---"

# oci-archive, NOT docker-archive. The tarball is an OCI layout --
# `blobs/sha256/...` at the root, not the manifest.json + layer tarballs a
# `docker save` produces. apptainer reports the mismatch as "gzip: invalid
# header", which sends you looking for a corrupt download; the file is a
# perfectly valid gzip of the wrong archive shape.
echo "--- inspecting archive shape ---"
# `head` is what stops tar early -- without it, listing this archive
# decompresses all 26 GB and takes minutes. But `set -o pipefail` is on, and
# head closing the pipe kills tar with SIGPIPE (141), so the pipeline reports
# failure however the grep went. That is why the previous run sniffed an OCI
# archive and still chose docker-archive://. Suspend pipefail for this one
# line: SIGPIPE here is the mechanism working, not an error.
set +o pipefail
FIRST=$(tar tzf "$TARBALL" 2>/dev/null | head -5)
set -o pipefail
echo "$FIRST" | sed 's/^/    /'
if echo "$FIRST" | grep -q "^blobs/"; then
    SRC="oci-archive://$TARBALL"
else
    SRC="docker-archive://$TARBALL"
fi
echo "--- converting $SRC -> sif ---"
apptainer build --force "$SIF" "$SRC"
echo "built: $SIF ($(du -h "$SIF" | cut -f1))"

echo "--- verifying the baked venv ---"
apptainer exec "$SIF" /opt/lehome-challenge/.venv/bin/python -c "
import importlib.metadata as m
for p in ('isaacsim','isaaclab','lerobot','torch'):
    try: print(f'  {p:10s}', m.version(p))
    except Exception: print(f'  {p:10s} (absent)')
"

echo
echo "Tarball kept at $TARBALL. Delete it to reclaim 27 GB once the .sif is proven:"
echo "  rm $TARBALL"
echo "=== done: $(date) ==="
