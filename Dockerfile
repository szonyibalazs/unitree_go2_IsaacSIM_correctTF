# Isaac Sim 4.5.0 + Isaac Lab (pinned) + ROS 2 Humble  ==  the "env_isaaclab" stack
FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES \
    ISAACSIM_PATH=/isaac-sim ISAACLAB_PATH=/opt/IsaacLab \
    ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# system ROS 2 Humble (rclpy + colcon, needed for the unitree_go / unitree_api messages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release git cmake build-essential && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base ros-humble-rosidl-generator-dds-idl python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*

# Isaac Lab, pinned to the commit the project was developed against
ARG ISAACLAB_COMMIT=b4c3210
RUN git clone https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} && \
    git -C ${ISAACLAB_PATH} checkout ${ISAACLAB_COMMIT} && \
    ln -s ${ISAACSIM_PATH} ${ISAACLAB_PATH}/_isaac_sim && \
    cd ${ISAACLAB_PATH} && TERM=xterm ./isaaclab.sh --install rsl_rl

# Unitree ROS 2 messages (unitree_go, unitree_api)
ARG TOOLBOX_COMMIT=cd11d74797b22b1a972fcce5aad4ed0f96ca76f7
RUN git clone https://github.com/andy-zhuo-02/go2_ros2_toolbox.git /tmp/toolbox && \
    git -C /tmp/toolbox checkout ${TOOLBOX_COMMIT} && \
    mkdir -p /opt/unitree_ws/src && cp -r /tmp/toolbox/unitree_pkgs/unitree_go /tmp/toolbox/unitree_pkgs/unitree_api /opt/unitree_ws/src/ && \
    rm -rf /tmp/toolbox && \
    bash -c "source /opt/ros/humble/setup.bash && cd /opt/unitree_ws && colcon build"

WORKDIR /workspace/go2_omniverse
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["main.py", "--robot_amount", "1", "--robot", "go2", "--terrain", "flat"]
