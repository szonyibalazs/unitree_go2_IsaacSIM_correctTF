import array
import struct
import time
import numpy as np
import os
import shutil
import omni.kit.app

from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Header
from unitree_go.msg import LowState

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
from isaacsim.ros2.bridge import collect_namespace, read_camera_info

# FONTOS: az isaaclab.sensors.Camera (CameraCfg alapú) render product
# útvonala lusta módon (lazy) inicializálódik, és megbízhatóan csak több
# env.step() lefutása UTÁN kérdezhető le. Ehelyett a natív, alacsonyabb
# szintű isaacsim.sensors.camera.Camera API-t használjuk - ugyanazt, amit
# a másik (működő) projekt SensorManager-je is használ. Ennek
# camera.initialize() hívása UTÁN azonnal, megbízhatóan elérhető a
# camera._render_product_path attribútum.
from isaacsim.sensors.camera import Camera as IsaacSimCamera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.rtx import LidarRtx
import omni.replicator.core as rep
import omni.graph.core as og
import omni.syntheticdata._syntheticdata as sd
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# A VALÓS Go2 INTERFÉSZ (go2_core / go2_description, módosítás nélkül)
# ---------------------------------------------------------------------------
# A szimuláció a valós robot topicjait és frame_id-it adja (a bag:
# rosbag2_2026_09_15-15_01_34), így a go2_base(_joy).launch.py, a
# robot_state_publisher + az URDF, a SLAM/nav stack változtatás nélkül fut:
#
#   KI (sim -> ROS2)                  tip.                 frame_id
#   /utlidar/robot_pose               PoseStamped          odom      (go2_base ebből csinál odom->base_link TF-et)
#   /utlidar/robot_odom               Odometry             odom -> base_link
#   /utlidar/imu                      Imu                  utlidar_imu
#   /utlidar/cloud                    PointCloud2          utlidar_lidar
#   /utlidar/cloud_base               PointCloud2          base_link
#   /utlidar/cloud_deskewed           PointCloud2          odom
#   /utlidar/range_info               PointStamped         base_link
#   /lowstate                         unitree_go/LowState  (joint_state_from_lowstate.py ebből csinál /joint_states-t)
#   BE: /api/sport/request            unitree_api/Request  (go2_base ebbe fordítja a /cmd_vel-t; lásd omniverse_sim.py)
#
# A szimuláció SZÁNDÉKOSAN NEM publikál /tf-et, /odom-ot, /joint_states-t és
# /cmd_vel-t hallgat: ezeket a valós launch (go2_base, robot_state_publisher,
# joint_state_from_lowstate) adja, pont mint a valós roboton.
# Több robotnál a 0. robot a valós (prefix nélküli) neveket kapja, az i. robot
# a /robot{i} névteret.
# ---------------------------------------------------------------------------

IMU_FRAME_ID = "utlidar_imu"          # a valós bagben is ez (az URDF-ben nincs)
LIDAR_FRAME_ID = "utlidar_lidar"
BASE_FRAME_ID = "base_link"
ODOM_FRAME_ID = "odom"

# go2_description/xacro/robot.xacro "lidar_joint" (trunk == base_link, identity)
LIDAR_XYZ = (0.29515, -0.00003, -0.06597)
LIDAR_RPY = (0.0, -0.2, 0.0)
# front_camera_joint
CAMERA_XYZ = (0.32715, -0.00003, 0.04297)
# NOTE: a lidar szenzor-tengelyei az Isaac Unitree_L1 profilban nem feltétlen egyeznek az
# utlidar_lidar-ral; ha a talaj nem z~0-n látszik odom-ban, itt kalibrálj (x,y,z,w).
LIDAR_CALIB_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

# Unitree motor_state[] sorrend (== joint_state_from_lowstate.py JOINT_NAMES)
UNITREE_JOINTS = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]
UNITREE_FEET = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']  # foot_force[] sorrend

_LIDAR_R = Rotation.from_euler('xyz', LIDAR_RPY)          # base <- lidar
_LIDAR_T = np.array(LIDAR_XYZ)
_LIDAR_CALIB = Rotation.from_quat(LIDAR_CALIB_QUAT_XYZW)
_GRAV = np.array([0.0, 0.0, 9.81])


def _ns(i):
    return "" if i == 0 else f"/robot{i}"


def _to_numpy(arr):
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _create_point_cloud2(header, points, intensity=None, ring_time=True):
    """A valós utlidar felhők formátuma: point_step 32, x,y,z @0/4/8, intensity @16
    (+ ring uint16 @20, time f32 @24 a /utlidar/cloud és /cloud_base esetén)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    buf = np.zeros((n, 8), dtype=np.float32)
    buf[:, :3] = pts
    buf[:, 4] = 255.0 if intensity is None else intensity
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [PointField(name=k, offset=o, datatype=PointField.FLOAT32, count=1)
                  for k, o in (('x', 0), ('y', 4), ('z', 8), ('intensity', 16))]
    if ring_time:
        msg.fields += [PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1),
                       PointField(name='time', offset=24, datatype=PointField.FLOAT32, count=1)]
    msg.is_bigendian = False
    msg.point_step = 32
    msg.row_step = 32 * n
    msg.is_dense = True
    # array('B'): az rclpy setter bytes/list-nél bájtonként validál Pythonban (~50 ms/felhő)
    msg.data = array.array('B', buf.tobytes())
    return msg


def add_rtx_lidar(num_envs, robot_type, debug=False):
    source_json = os.path.abspath("data/sensors/lidar/Unitree_L1.json")
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = ext_manager.get_enabled_extension_id("isaacsim.sensors.rtx")

    if ext_id and os.path.exists(source_json):
        target_dir = os.path.join(ext_manager.get_extension_path(ext_id), "data", "lidar_configs")
        os.makedirs(target_dir, exist_ok=True)
        target_json = os.path.join(target_dir, "Unitree_L1.json")
        try:
            shutil.copy(source_json, target_json)
        except Exception:
            pass

    annotator_lst = []

    for i in range(num_envs):
        if robot_type == "g1":
            lidar_sensor = LidarRtx(f'/World/envs/env_{i}/Robot/head_link/lidar_sensor',
                                    rotation_frequency=200,
                                    pulse_time=1,
                                    translation=(0.0, 0.0, 0.0),
                                    orientation=(1.0, 0.0, 0.15, 0.0),
                                    config_file_name="Unitree_L1")
        else:
            # a pozíció/orientáció az URDF lidar_joint-ja (base_link -> utlidar_lidar)
            q = (_LIDAR_R * Rotation.from_quat(LIDAR_CALIB_QUAT_XYZW)).as_quat()  # x,y,z,w
            lidar_sensor = LidarRtx(f'/World/envs/env_{i}/Robot/base/lidar_sensor',
                                    rotation_frequency=200,
                                    pulse_time=1,
                                    translation=LIDAR_XYZ,
                                    orientation=(q[3], q[0], q[1], q[2]),
                                    config_file_name="Unitree_L1")

        if debug:
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([lidar_sensor.get_render_product_path()])

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        annotator.attach(lidar_sensor.get_render_product_path())
        annotator_lst.append(annotator)
    return annotator_lst


def add_camera(num_envs, robot_type, freq=None):
    """Létrehozza a kamera szenzorokat minden environmenthez.

    A natív isaacsim.sensors.camera.Camera API-t használja (nem az
    isaaclab.sensors.CameraCfg-t), mert ennek a render product útvonala
    azonnal elérhető camera.initialize() után, nem kell env.step()-re
    várni. Ugyanez az API, amit a másik (működő) projekt SensorManager-je
    is használ.

    freq=None esetén a kamera a szimuláció rendering frekvenciáját követi
    automatikusan. Ha explicit értéket adsz meg, annak OSZTHATÓNAK kell
    lennie a rendering frekvenciával, különben
    "frequency of the camera sensor needs to be a divisible by the
    rendering frequency" hibát kapsz.

    Visszaadja a létrehozott, már inicializált Camera objektumok listáját.
    A camera._render_product_path ezután azonnal elérhető, és az
    add_camera_ros2_publisher()-nek erre van szüksége.
    """
    camera_lst = []
    for i in range(num_envs):
        if robot_type == "g1":
            prim_path = f"/World/envs/env_{i}/Robot/head_link/front_cam"
            translation = np.array([0.0, 0.0, 0.0])
        else:
            prim_path = f"/World/envs/env_{i}/Robot/base/front_cam"
            translation = np.array(CAMERA_XYZ)  # URDF front_camera_joint

        camera = IsaacSimCamera(
            prim_path=prim_path,
            translation=translation,
            frequency=freq,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        )
        camera.initialize()
        camera_lst.append(camera)
    return camera_lst


def add_camera_ros2_publisher(num_envs, camera_lst, step_size=1):
    """Natívan (rep.writers-en keresztül) bekapcsolja az RGB, Depth és
    CameraInfo publikálást minden robot kamerájára.

    Topicok (0. robot, prefix nélkül; i>0 esetén /robot{i} előtaggal):
      /camera/image_raw      (== a go2_base.launch image_topic default)
      /camera/depth_image
      /camera/camera_info

    Ezt egyszer kell meghívni, miután add_camera() lefutott (a
    camera.initialize() hívás miatt a render_product azonnal elérhető,
    nincs szükség env.step()-re várni előtte).

    Visszaadja: True, ha minden kamerához sikerült publishert kötni.
    """
    node_namespace = ""
    queue_size = 1
    all_succeeded = True

    for i in range(num_envs):
        cam = camera_lst[i]
        render_product = getattr(cam, "_render_product_path", None)

        if not render_product:
            print(
                f"[WARN] Camera {i}: _render_product_path még nem elérhető "
                f"(camera.initialize() lefutott, de a render product még "
                f"üres). Kamera {i} ROS2 publisher kihagyva ebben a körben."
            )
            all_succeeded = False
            continue

        frame_id = "front_camera"  # URDF link

        # ---- RGB (color) ----
        color_topic = f"{_ns(i)}/camera/image_raw"  # == go2_base.launch image_topic
        rv_color = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        color_writer = rep.writers.get(rv_color + "ROS2PublishImage")
        color_writer.initialize(
            frameId=frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=color_topic,
        )
        color_writer.attach([render_product])

        color_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_color + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(color_gate_path + ".inputs:step").set(step_size)

        # ---- Depth ---- (a valós go2_core stackben nincs depth; GO2_DEPTH=1-gyel kérhető)
        if os.environ.get('GO2_DEPTH'):
            depth_topic = f"{_ns(i)}/camera/depth_image"
            rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )
            depth_writer = rep.writers.get(rv_depth + "ROS2PublishImage")
            depth_writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=depth_topic,
            )
            depth_writer.attach([render_product])

            depth_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv_depth + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(depth_gate_path + ".inputs:step").set(step_size)

        # ---- Camera Info ----
        info_topic = f"{_ns(i)}/camera/camera_info"
        camera_info = read_camera_info(render_product_path=render_product)
        info_writer = rep.writers.get("ROS2PublishCameraInfo")
        info_writer.initialize(
            frameId=frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=info_topic,
            width=camera_info["width"],
            height=camera_info["height"],
            projectionType=camera_info["projectionType"],
            k=camera_info["k"].reshape([1, 9]),
            r=camera_info["r"].reshape([1, 9]),
            p=camera_info["p"].reshape([1, 12]),
            physicalDistortionModel=camera_info["physicalDistortionModel"],
            physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
        )
        info_writer.attach([render_product])

        info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(info_gate_path + ".inputs:step").set(step_size)

    return all_succeeded



def pub_robo_data_ros2(robot_type, num_envs, base_node, env, annotator_lst, start_time):
    scene = env.unwrapped.scene
    robot_data = scene["robot"].data
    root_state = _to_numpy(robot_data.root_state_w)
    lin_vel_b = _to_numpy(robot_data.root_lin_vel_b)
    ang_vel_b = _to_numpy(robot_data.root_ang_vel_b)
    lin_acc_w = _to_numpy(robot_data.body_lin_acc_w)[:, 0]
    joint_pos = _to_numpy(robot_data.joint_pos)
    joint_vel = _to_numpy(robot_data.joint_vel)
    joint_tau = _to_numpy(robot_data.applied_torque)

    if not hasattr(base_node, "joint_idx"):
        names = list(robot_data.joint_names)
        base_node.joint_idx = [names.index(n) for n in UNITREE_JOINTS]
        base_node.foot_idx = None
        if robot_type == "go2":
            bodies = list(scene["contact_forces"].body_names)
            low = [b.lower() for b in bodies]
            base_node.foot_idx = [low.index(f.lower()) for f in UNITREE_FEET]
    ji = base_node.joint_idx

    # EGY közös, wall-clock időbélyeg: a go2_base.cpp az odom->base_link TF-et a saját
    # now()-jával bélyegzi, ahogy a valós roboton is.
    stamp = base_node.get_clock().now().to_msg()

    poses = []
    for i in range(num_envs):
        pos, quat = root_state[i, :3], root_state[i, 3:7]  # quat: w,x,y,z
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        poses.append((pos, rot))
        # IMU = valódi gyorsulásmérő: R^T (a_world + g), a gravitációt is méri.
        acc_b = rot.inv().apply(lin_acc_w[i] + _GRAV)
        foot = None
        if base_node.foot_idx is not None:
            foot = _to_numpy(scene["contact_forces"].data.net_forces_w)[i][base_node.foot_idx, 2]
        base_node.publish_state(i, stamp, pos, rot, lin_vel_b[i], ang_vel_b[i], acc_b,
                                joint_pos[i][ji], joint_vel[i][ji], joint_tau[i][ji], foot)

    # A lidar/kamera csak renderelt lépésben frissül (render_interval), ekkor olvassuk ki
    # (GPU->CPU szinkron, drága); köztes lépésben duplikált felhőt kapnánk.
    cfg = env.unwrapped.cfg
    base_node.step_n = getattr(base_node, "step_n", 0) + 1
    if base_node.step_n % max(1, cfg.sim.render_interval // cfg.decimation) == 0:
        for j in range(min(num_envs, len(annotator_lst))):
            try:
                data = annotator_lst[j].get_data()
                if data and 'data' in data:
                    pts = _to_numpy(data['data'])
                    if pts is not None and len(pts) > 0:
                        base_node.publish_lidar(pts, j, stamp, *poses[j])
            except Exception as e:
                print(f"[WARN] Lidar error: {e}")


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__('go2_driver_node')
        q = QoSProfile(depth=10)
        self.pose_pub, self.odom_pub, self.imu_pub, self.low_pub = [], [], [], []
        self.cloud_pub, self.cloud_base_pub, self.cloud_odom_pub, self.range_pub = [], [], [], []
        for i in range(num_envs):
            n = _ns(i)
            self.pose_pub.append(self.create_publisher(PoseStamped, f'{n}/utlidar/robot_pose', q))
            self.odom_pub.append(self.create_publisher(Odometry, f'{n}/utlidar/robot_odom', q))
            self.imu_pub.append(self.create_publisher(Imu, f'{n}/utlidar/imu', q))
            self.low_pub.append(self.create_publisher(LowState, f'{n}/lowstate', q))
            self.cloud_pub.append(self.create_publisher(PointCloud2, f'{n}/utlidar/cloud', q))
            self.cloud_base_pub.append(self.create_publisher(PointCloud2, f'{n}/utlidar/cloud_base', q))
            self.cloud_odom_pub.append(self.create_publisher(PointCloud2, f'{n}/utlidar/cloud_deskewed', q))
            self.range_pub.append(self.create_publisher(PointStamped, f'{n}/utlidar/range_info', q))
        # NEM publikálunk /tf-et, /odom-ot, /joint_states-t (lásd a fájl tetején).

    def publish_state(self, i, stamp, pos, rot, lin_vel_b, ang_vel_b, acc_b, q, dq, tau, foot):
        x, y, z, w = rot.as_quat()
        pos = [float(v) for v in pos]

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = ODOM_FRAME_ID
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
        pose.pose.orientation.x, pose.pose.orientation.y = float(x), float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = float(z), float(w)
        self.pose_pub[i].publish(pose)

        odom = Odometry()
        odom.header = pose.header
        odom.child_frame_id = BASE_FRAME_ID
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = (float(v) for v in lin_vel_b)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = (float(v) for v in ang_vel_b)
        self.odom_pub[i].publish(odom)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = IMU_FRAME_ID
        imu.orientation = pose.pose.orientation
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in ang_vel_b)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(v) for v in acc_b)
        self.imu_pub[i].publish(imu)

        low = LowState()
        for k in range(len(UNITREE_JOINTS)):
            low.motor_state[k].q = float(q[k])
            low.motor_state[k].dq = float(dq[k])
            low.motor_state[k].tau_est = float(tau[k])
        low.imu_state.quaternion = [float(w), float(x), float(y), float(z)]  # Unitree: w,x,y,z
        low.imu_state.gyroscope = [float(v) for v in ang_vel_b]
        low.imu_state.accelerometer = [float(v) for v in acc_b]
        low.imu_state.rpy = [float(v) for v in rot.as_euler('xyz')]
        if foot is not None:
            low.foot_force = [int(max(0, min(32767, v))) for v in foot]
        self.low_pub[i].publish(low)

    def publish_lidar(self, points, i, stamp, pos, rot):
        """A pontfelhő 3 frame-ben, mint a valós robotnál: utlidar_lidar (szenzor-lokális),
        base_link és odom. A base_link/odom változatot az URDF lidar_joint-jával és az
        aktuális póz-szal számoljuk, így konzisztens a go2_base + robot_state_publisher TF-fel."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        pts = _LIDAR_CALIB.apply(pts[np.isfinite(pts).all(axis=1)])  # szenzor-tengelyek -> utlidar_lidar
        pts_base = _LIDAR_R.apply(pts) + _LIDAR_T
        pts_odom = rot.apply(pts_base) + np.asarray(pos)
        for pub, p, frame, rt in ((self.cloud_pub[i], pts, LIDAR_FRAME_ID, True),
                                  (self.cloud_base_pub[i], pts_base, BASE_FRAME_ID, True),
                                  (self.cloud_odom_pub[i], pts_odom, ODOM_FRAME_ID, False)):
            pub.publish(_create_point_cloud2(Header(stamp=stamp, frame_id=frame), p, ring_time=rt))
        rng = PointStamped()  # a valós robot konstans értéke a bagben
        rng.header.stamp, rng.header.frame_id = stamp, BASE_FRAME_ID
        rng.point.x, rng.point.y, rng.point.z = 0.95, 0.5, 0.35
        self.range_pub[i].publish(rng)
