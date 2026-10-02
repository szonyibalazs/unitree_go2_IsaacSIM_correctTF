#!/usr/bin/env bash
# Sim indítása a conda env_isaaclab-ban (Isaac Sim 4.x, Py3.10) + rendszer ROS2 Humble + new_ws.
# A valós go2_core/go2_description stack ugyanígy, módosítás nélkül fut mellette:
#   ros2 launch go2_core go2_base.launch.py urdf:=true
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
source /opt/ros/humble/setup.bash
source "${GO2_WS:-$HOME/unitree/new_ws}/install/setup.bash"   # unitree_go / unitree_api msgs
export PYTHONNOUSERSITE=1   # a ~/.local torch ütközik a conda-sal
export OMNI_KIT_ACCEPT_EULA=YES RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python -u main.py --robot_amount 1 --robot go2 --terrain flat "$@"
