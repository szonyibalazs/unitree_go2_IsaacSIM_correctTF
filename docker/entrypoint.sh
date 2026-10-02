#!/usr/bin/env bash
# Usage: entrypoint.sh <script.py> [args...]   (or `bash` for a shell)
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/unitree_ws/install/setup.bash
if [[ "$1" == *.py ]]; then exec /isaac-sim/python.sh -u "$@"; fi
exec "$@"
