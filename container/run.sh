#!/usr/bin/env bash
# Run a Python script inside halogal.sif using THIS checkout's halogal code
# (the copy baked into the image is only as new as the last build).
#
# Usage (from anywhere):
#   container/run.sh examples/fit_uvlf.py --z 7 --processes 8
#
# Env overrides:
#   HALOGAL_SIF   path to the image   (default: container/halogal.sif)
#   HALOGAL_BIND  extra --bind spec   (e.g. /raid/scratch:/raid/scratch)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${HALOGAL_SIF:-$REPO/container/halogal.sif}"
RUNTIME="$(command -v apptainer || command -v singularity)" || {
    echo "Neither apptainer nor singularity found on $(hostname)" >&2; exit 1; }

binds=(--bind "$REPO")
[[ -n "${HALOGAL_BIND:-}" ]] && binds+=(--bind "$HALOGAL_BIND")

# ~/.cache may be a symlink to a scratch disk that isn't mounted in the
# container (dangling link -> arviz/platformdirs crash on import). Bind its
# real target if this host has it; otherwise point the cache at /tmp.
envs=(--env PYTHONPATH="$REPO")
cache_real="$(readlink -f "$HOME/.cache" 2>/dev/null || true)"
if [[ -L "$HOME/.cache" && -d "$cache_real" ]]; then
    binds+=(--bind "$cache_real")
elif [[ ! -d "$HOME/.cache" ]]; then
    mkdir -p "/tmp/$USER/cache"
    envs+=(--env XDG_CACHE_HOME="/tmp/$USER/cache")
fi

cd "$REPO"
exec "$RUNTIME" exec "${binds[@]}" "${envs[@]}" "$SIF" python "$@"
