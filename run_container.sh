#!/usr/bin/env bash
# Build (once) and run the simulator in the env_isaaclab container.
#   ./run_container.sh                       # Go2, flat terrain
#   ./run_container.sh main.py --robot g1 --terrain flat
#   ./run_container.sh main.py --headless --enable_cameras
#   ./run_container.sh bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
IMAGE=env_isaaclab
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
xhost +local:root >/dev/null 2>&1 || true
CACHE="${HOME}/.cache/env_isaaclab"; mkdir -p "$CACHE"/{kit,ov,pip,glcache,compute}
exec docker run --rm -it --gpus all --network=host \
    -e DISPLAY -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/workspace/go2_omniverse \
    -v "$CACHE/kit":/isaac-sim/kit/cache \
    -v "$CACHE/ov":/root/.cache/ov \
    -v "$CACHE/pip":/root/.cache/pip \
    -v "$CACHE/glcache":/root/.cache/nvidia/GLCache \
    -v "$CACHE/compute":/root/.nv/ComputeCache \
    "$IMAGE" "$@"
