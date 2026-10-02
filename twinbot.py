import struct
import time
import numpy as np
import os
import shutil
import omni.kit.app

from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header, Float32MultiArray

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
# ÖSSZHANG A go2_core / go2_base.cpp ÉS A HOZZÁ TARTOZÓ LAUNCH FÁJLLAL
# ---------------------------------------------------------------------------
# A go2_base.cpp (C++ node) a valós robotot vezérlő "go2_core" csomag
# node-ja. Ez:
#   - feliratkozik a /utlidar/robot_pose (geometry_msgs/PoseStamped) topicra,
#     és ABBÓL saját maga állítja elő:
#       * az odom -> base_link TF-et (msg->header.stamp időbélyeggel!)
#       * a /odom (nav_msgs/Odometry) topicot
#   - feliratkozik a /cmd_vel-re és sport API hívásokat generál belőle.
#
# A launch fájl emellett elindítja a robot_state_publisher-t is, ami az
# URDF (go2_description/xacro/robot.xacro, use_nav2_links:=true) alapján
# adja ki a base_link -> imu_link, base_link -> utlidar_lidar (stb.)
# TF-eket (STATIKUS, a robot geometriájából, nem szimulációs adatból).
#
# EBBŐL KÖVETKEZIK, hogy ennek a szimulációs (Isaac Sim) node-nak:
#   1) NEM SZABAD saját maga odom->base_link TF-et publikálnia
#      (ezt a go2_base.cpp csinálja a /utlidar/robot_pose-ból; két
#      versengő forrás ugráló/villódzó TF-et okozna).
#   2) NEM SZABAD közvetlenül /odom-ra publikálnia
#      (ugyanezen okból - a go2_base.cpp állítja elő a /odom-ot is).
#   3) A /utlidar/robot_pose-t KELL publikálnia, header.frame_id="odom"-mal,
#      mert ez a go2_base.cpp egyetlen bemenete a pózhoz.
#   4) A szenzor (IMU, lidar) üzenetek frame_id-jét a robot_state_publisher
#      által ismert link-nevekhez KELL igazítani (pl. "imu_link",
#      "utlidar_lidar"), NEM "base_link"-hez vagy "odom"-hoz, különben a
#      TF fában nincs "szenzor -> base_link" lánc, és a downstream (pl.
#      SLAM) lookupTransform()-ja identity-re esik vissza.
#   5) A lidar pontfelhőt a SZENZOR SAJÁT (lokális) frame-jében kell
#      küldeni, NEM világ/odom-koordinátákba előre transzformálva -
#      a világkoordinátás elhelyezést a TF fa (odom -> base_link ->
#      utlidar_lidar, a go2_base + robot_state_publisher együtteséből)
#      végzi el a fogadó oldalon (pl. my_slam-ben).
# ---------------------------------------------------------------------------

# A robot_state_publisher URDF-jében (go2_description xacro,
# use_nav2_links:=true) használt link-nevek. Ha az URDF-ben más neveket
# használsz, ezt a két konstanst kell módosítani.
IMU_FRAME_ID = "imu_link"
LIDAR_FRAME_ID = "utlidar_lidar"

# ---------------------------------------------------------------------------
# REALSENSE D435 SZENZOR-SZIMULÁCIÓ
# ---------------------------------------------------------------------------
# Az alábbi konstansok a valódi Intel RealSense D435 gyári specifikációját
# követik (RGB és depth optikai modul), hogy a szimulált kamera kimenete
# (felbontás, látószög, tartomány, topic-elnevezés) a lehető legjobban
# megfeleljen egy valódi D435-ös driver (realsense-ros / librealsense)
# kimenetének.
#
# RGB modul (color):
#   - natív felbontás: 1920x1080, mi 1280x720-at használunk (D435 alap
#     "Color" profil), vízszintes FOV ~69°, függőleges FOV ~42°
# Depth modul (sztereó, aktív IR):
#   - natív felbontás: 1280x720 (mi is ezt használjuk), vízszintes FOV
#     ~87°, függőleges FOV ~58°
#   - működési tartomány kb. 0.1 m - 10 m (ideális esetben; a gyakorlati
#     megbízható tartomány ~0.3-3m, de a szimulált clipping range-et a
#     gyári min/max-hoz igazítjuk)
# Mindkét stream 30 Hz-en fut alapból.
#
# FONTOS EGYSZERŰSÍTÉS: a D435-ön a color és depth optikai tengelye
# fizikailag ~15mm-rel el van tolva egymástól (extrinsics), és a valódi
# driver ezt a "depth -> color" transzformációval veszi figyelembe a
# regisztrált (aligned) képeknél. Ebben a szimulációban EGY KÖZÖS
# render product-ot (egyetlen kamera prim-et) használunk mind a színhez,
# mind a mélységhez - ez azt jelenti, hogy a color és depth kép már eleve
# pixel-pontosan illeszkedik (mintha "align_depth"-elt/registered
# módban futna a valódi driver). Ha valakinek kifejezetten a nyers,
# el nem tolt sztereó-depth kimenetre van szüksége (pl. bal/jobb IR
# képekkel), az egy külön depth-only kamera prim hozzáadásával oldható
# meg, ugyanígy a REALSENSE_DEPTH_* konstansok felhasználásával.
REALSENSE_COLOR_RESOLUTION = (1280, 720)   # (width, height), D435 "Color" alapprofil
REALSENSE_COLOR_HFOV_DEG = 69.4            # gyári RGB modul vízszintes FOV
REALSENSE_COLOR_VFOV_DEG = 42.5            # gyári RGB modul függőleges FOV

REALSENSE_DEPTH_RESOLUTION = (1280, 720)   # D435 depth alapprofil (848x480 is gyakori, de 1280x720-at is támogat)
REALSENSE_DEPTH_HFOV_DEG = 87.0            # gyári sztereó depth modul vízszintes FOV
REALSENSE_DEPTH_VFOV_DEG = 58.0            # gyári sztereó depth modul függőleges FOV

REALSENSE_DEPTH_MIN_RANGE_M = 0.1          # gyári min. mélységtartomány
REALSENSE_DEPTH_MAX_RANGE_M = 10.0         # gyári max. mélységtartomány (jó fényviszonyok mellett)

REALSENSE_FPS = 30.0                       # mindkét stream alapértelmezett frekvenciája

# A librealsense/realsense-ros driver optikai frame-elnevezési konvenciója.
# A robot_state_publisher URDF-jébe innentől ILYEN NÉVVEL kell felvenni a
# kamera linket (pl. static_transform_publisher-rel base_link -> ezekhez),
# hogy a TF fa illeszkedjen a valódi RealSense driver konvenciójához.
REALSENSE_COLOR_OPTICAL_FRAME = "camera_color_optical_frame"
REALSENSE_DEPTH_OPTICAL_FRAME = "camera_depth_optical_frame"


def _to_numpy(arr):
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _create_point_cloud2(header, points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3).astype(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
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
                                    orientation=(1.0, 0.0, 0.0, 0.0),
                                    config_file_name="Unitree_L1")
        else:
            lidar_sensor = LidarRtx(f'/World/envs/env_{i}/Robot/base/lidar_sensor',
                                    rotation_frequency=200,
                                    pulse_time=1,
                                    translation=(0.0, 0, 0.4),
                                    orientation=(1.0, 0.0, 0.0, 0.0),
                                    config_file_name="Unitree_L1")

        if debug:
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([lidar_sensor.get_render_product_path()])

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        annotator.attach(lidar_sensor.get_render_product_path())
        annotator_lst.append(annotator)
    return annotator_lst


def _get_rendering_frequency_hz(default_hz: float = 60.0) -> float:
    """Lekérdezi a FUTÓ szimuláció tényleges rendering frekvenciáját (Hz).

    Az isaacsim.sensors.camera.Camera 'frequency' paramétere KÖTELEZŐEN
    osztója kell legyen a rendering frekvenciának (lásd IsaacSimCamera
    forrás: "frequency of the camera sensor needs to be a divisible by
    the rendering frequency" hiba), ami viszont a task cfg-től függ
    (pl. sim.dt=0.005, render_interval=4 -> rendering_dt=0.02 -> 50 Hz),
    NEM feltétlenül 60 Hz. Ezért ezt sosem szabad statikusan feltételezni
    (pl. fixen 30 Hz-et beállítani), hanem a ténylegesen futó
    SimulationContext-ből kell kiolvasni.
    """
    try:
        from isaaclab.sim import SimulationContext
        sim_context = SimulationContext.instance()
        if sim_context is not None:
            rendering_dt = sim_context.get_rendering_dt()
            if rendering_dt and rendering_dt > 0:
                return 1.0 / rendering_dt
    except Exception:
        pass

    try:
        from isaacsim.core.rendering_manager import RenderingManager
        rendering_dt = RenderingManager.get_dt()
        if rendering_dt and rendering_dt > 0:
            return 1.0 / rendering_dt
    except Exception:
        pass

    print(
        f"[WARN] Nem sikerült lekérdezni a tényleges rendering frekvenciát, "
        f"visszaesés {default_hz} Hz-re."
    )
    return default_hz


def _pick_camera_frequency_hz(target_hz: float = REALSENSE_FPS) -> float:
    """Kiválasztja a kívánt (D435 natív, 30 Hz-es) frekvenciához legközelebbi,
    a TÉNYLEGES rendering frekvenciával osztható értéket.

    Enélkül a kamera.initialize() a valódi szimulációs cfg-től függően
    "frequency of the camera sensor needs to be a divisible by the
    rendering frequency" hibával elszáll (pl. 50 Hz rendering mellett a
    fix 30 Hz nem osztható vele: 50 % 30 != 0).
    """
    rendering_hz = _get_rendering_frequency_hz()

    if rendering_hz <= 0:
        return target_hz

    # Csak egész osztók jöhetnek szóba (rendering_hz / n a kamera frekvenciája,
    # n = 1, 2, 3, ...). Azt az osztót választjuk, amelyik a target_hz-hez
    # (30 Hz) a legközelebb esik, de nem haladja meg azt jelentősen.
    best_freq = None
    best_diff = None
    max_divisor = max(1, int(round(rendering_hz / 1.0)))  # ne menjünk 1 Hz alá értelmetlenül
    for n in range(1, max_divisor + 1):
        candidate = rendering_hz / n
        if candidate < 1.0:
            break
        diff = abs(candidate - target_hz)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_freq = candidate

    if best_freq is None:
        best_freq = rendering_hz

    if abs(best_freq - target_hz) > 1e-6:
        print(
            f"[INFO] Kamera frekvencia: a kért {target_hz} Hz (D435 natív) nem "
            f"osztója a {rendering_hz} Hz rendering frekvenciának, helyette "
            f"{best_freq} Hz kerül beállításra (a legközelebbi érvényes osztó)."
        )

    return best_freq


def _apply_realsense_d435_intrinsics(camera: IsaacSimCamera):
    """Beállítja a kamera prim-en a horizontal/vertical aperture és a
    clipping range értékeket úgy, hogy a resituló FOV és mélységtartomány
    a valódi RealSense D435 depth modulját kövesse.

    A horizontal_aperture/vertical_aperture (Omniverse "film gate", cm-ben)
    és a focal_length (szintén cm-ben) együtt határozzák meg a FOV-ot:
        fov = 2 * atan(aperture / (2 * focal_length))
    Rögzített, valósághű 24mm-es focal_length mellett visszaszámoljuk az
    apertúrát a kívánt FOV-ból - ez pontosan illeszkedő intrinsics mátrixot
    ad (fx, fy, cx, cy), amit a ROS2PublishCameraInfo writer helyesen fog
    kiolvasni és publikálni.
    """
    focal_length_cm = 2.4  # tetszőleges rögzített érték, csak arányossághoz kell

    h_fov_rad = np.deg2rad(REALSENSE_DEPTH_HFOV_DEG)
    v_fov_rad = np.deg2rad(REALSENSE_DEPTH_VFOV_DEG)

    horizontal_aperture = 2.0 * focal_length_cm * np.tan(h_fov_rad / 2.0)
    vertical_aperture = 2.0 * focal_length_cm * np.tan(v_fov_rad / 2.0)

    camera.set_focal_length(focal_length_cm)
    camera.set_horizontal_aperture(horizontal_aperture)
    camera.set_vertical_aperture(vertical_aperture)
    # A gyári D435 depth tartomány (0.1 - 10 m) mint clipping range - így
    # a DistanceToImagePlane annotator sem ad vissza fizikailag
    # értelmetlen (a valódi szenzornál sosem mérhető) távolságokat.
    camera.set_clipping_range(REALSENSE_DEPTH_MIN_RANGE_M, REALSENSE_DEPTH_MAX_RANGE_M)


def add_camera(num_envs, robot_type, freq=None):
    """Létrehozza a kamera szenzorokat minden environmenthez, RealSense
    D435-kompatibilis beállításokkal (felbontás, FOV, mélységtartomány).

    A natív isaacsim.sensors.camera.Camera API-t használja (nem az
    isaaclab.sensors.CameraCfg-t), mert ennek a render product útvonala
    azonnal elérhető camera.initialize() után, nem kell env.step()-re
    várni. Ugyanez az API, amit a másik (működő) projekt SensorManager-je
    is használ.

    freq=None esetén a kódnak MEG KELL kérdeznie a ténylegesen futó
    szimuláció rendering frekvenciáját, és az ahhoz a legközelebb eső,
    azzal OSZTHATÓ frekvenciát kell választania a D435 natív 30 Hz-éhez
    közelítve (lásd _pick_camera_frequency_hz()) - fix 30 Hz-et
    beállítani NEM biztonságos, mert az IsaacSimCamera API megköveteli,
    hogy a kamera 'frequency'-je OSZTHATÓ legyen a rendering
    frekvenciával, ami viszont a task cfg-től függ (pl. Go2 rough env:
    sim.dt=0.005, render_interval=4 -> 50 Hz rendering, aminek 30 NEM
    osztója -> "frequency of the camera sensor needs to be a divisible
    by the rendering frequency" hibával elszáll a camera.initialize()).
    Ha freq explicit meg van adva, azt fixen használjuk (a hívó
    felelőssége, hogy az osztható legyen).

    A felbontás a REALSENSE_DEPTH_RESOLUTION-t követi (1280x720), mivel
    egyetlen közös render product szolgáltatja mind a color, mind a depth
    (és ebből a point cloud) kimenetet - lásd a modul tetején lévő
    megjegyzést az "aligned/registered" egyszerűsítésről.

    Visszaadja a létrehozott, már inicializált Camera objektumok listáját.
    A camera._render_product_path ezután azonnal elérhető, és az
    add_camera_ros2_publisher()-nek erre van szüksége.
    """
    camera_lst = []
    stream_freq = freq if freq is not None else _pick_camera_frequency_hz(REALSENSE_FPS)

    for i in range(num_envs):
        if robot_type == "g1":
            prim_path = f"/World/envs/env_{i}/Robot/head_link/front_cam"
            translation = np.array([0.0, 0.0, 0.0])
        else:
            prim_path = f"/World/envs/env_{i}/Robot/base/front_cam"
            translation = np.array([0.32487, -0.00095, 0.05362])

        # A camera.initialize() a frequency validáció miatt ("frequency of
        # the camera sensor needs to be a divisible by the rendering
        # frequency") kivételt dobhat, ha a hívó explicit, a tényleges
        # rendering frekvenciával nem osztható freq-et adott meg. Ilyenkor
        # a félig konstruált Camera objektum __del__-je is hibát dobhat
        # (AttributeError: '_supported_annotators'), ezért a hibát itt,
        # kamerénként elkapjuk, hogy egy rossz env ne hagyjon csendben
        # félkész objektumokat a listában, és a többi env kamerája
        # (helyes frekvenciával) továbbra is létrejöjjön.
        try:
            camera = IsaacSimCamera(
                prim_path=prim_path,
                translation=translation,
                frequency=stream_freq,
                resolution=REALSENSE_DEPTH_RESOLUTION,
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )
            camera.initialize()
            _apply_realsense_d435_intrinsics(camera)
            camera_lst.append(camera)
        except Exception as e:
            print(
                f"[WARN] Camera {i} inicializálás sikertelen "
                f"(frequency={stream_freq} Hz): {type(e).__name__}: {e}. "
                f"Ez az env kamera nélkül fut tovább."
            )
    return camera_lst


def add_camera_ros2_publisher(num_envs, camera_lst, step_size=1):
    """Bekapcsolja az RGB, Depth, CameraInfo (color + depth) ÉS a
    Point Cloud publikálást minden robot kamerájára, RealSense D435
    driver-szerű topic-elnevezéssel (realsense-ros konvenció).

    Topicok robotonként (a robot namespace-e "robot{i}"):
      robot{i}/camera/color/image_raw          (sensor_msgs/Image, RGB8)
      robot{i}/camera/color/camera_info         (sensor_msgs/CameraInfo)
      robot{i}/camera/depth/image_rect_raw      (sensor_msgs/Image, mélységkép)
      robot{i}/camera/depth/camera_info         (sensor_msgs/CameraInfo)
      robot{i}/camera/depth/color/points        (sensor_msgs/PointCloud2)

    A point cloud generálása a hivatalos Isaac Sim
    "<rendervar>ROS2PublishPointCloud" writer-rel történik (GPU-n, a
    Replicator/RTX pipeline része), amely a depth képet a kamera
    intrinsics-e alapján 3D pontfelhővé alakítja - ugyanígy működik, mint
    a valódi RealSense driver "depth/color/points" topicja
    (pointcloud.enable:=true esetén).

    Mivel a color és a depth ugyanabból a render productból (ugyanabból a
    kamera prim-ből) származik, a depth/color/points topic pontfelhője már
    eleve a color képpel regisztrált (aligned) - ez felel meg a valódi
    driver "align_depth:=true" üzemmódjának.

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

        # A realsense-ros driver konvenciója szerinti optikai frame-ek.
        # Mivel egyetlen közös render product szolgálja ki a color és a
        # depth streamet is (lásd add_camera() megjegyzését), mindkettőt
        # ugyanarra a fizikai szenzorpozícióra vetítjük - a color és
        # depth optical frame-et éppen ezért KÜLÖN névvel adjuk ki, hogy a
        # robot_state_publisher URDF-jében definiált (akár identikus
        # pózú) linkekhez illeszkedjen a downstream (pl. RViz, SLAM)
        # feldolgozás, a valódi driver TF-konvencióját követve.
        color_frame_id = f"robot{i}/{REALSENSE_COLOR_OPTICAL_FRAME}"
        depth_frame_id = f"robot{i}/{REALSENSE_DEPTH_OPTICAL_FRAME}"

        # ---- RGB (color) image ----
        color_topic = f"robot{i}/camera/color/image_raw"
        rv_color = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        color_writer = rep.writers.get(rv_color + "ROS2PublishImage")
        color_writer.initialize(
            frameId=color_frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=color_topic,
        )
        color_writer.attach([render_product])

        color_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_color + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(color_gate_path + ".inputs:step").set(step_size)

        # ---- Depth image ----
        # A realsense-ros "depth/image_rect_raw" topicnak megfelelő
        # elnevezés (16UC1/32FC1 mélységkép, méterben/mm-ben a driver
        # beállításától függően; itt a natív Isaac Sim float32 méter
        # formátumban publikál).
        depth_topic = f"robot{i}/camera/depth/image_rect_raw"
        rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        depth_writer = rep.writers.get(rv_depth + "ROS2PublishImage")
        depth_writer.initialize(
            frameId=depth_frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=depth_topic,
        )
        depth_writer.attach([render_product])

        depth_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_depth + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(depth_gate_path + ".inputs:step").set(step_size)

        # ---- Color Camera Info ----
        color_info_topic = f"robot{i}/camera/color/camera_info"
        camera_info = read_camera_info(render_product_path=render_product)
        color_info_writer = rep.writers.get("ROS2PublishCameraInfo")
        color_info_writer.initialize(
            frameId=color_frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=color_info_topic,
            width=camera_info["width"],
            height=camera_info["height"],
            projectionType=camera_info["projectionType"],
            k=camera_info["k"].reshape([1, 9]),
            r=camera_info["r"].reshape([1, 9]),
            p=camera_info["p"].reshape([1, 12]),
            physicalDistortionModel=camera_info["physicalDistortionModel"],
            physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
        )
        color_info_writer.attach([render_product])

        color_info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(color_info_gate_path + ".inputs:step").set(step_size)

        # ---- Depth Camera Info ----
        # A valódi RealSense driver a depth és a color streamhez KÜLÖN
        # CameraInfo-t ad ki (eltérő extrinsics/intrinsics lehet közöttük).
        # Itt egyetlen közös render productból dolgozunk, ezért az
        # intrinsics mátrix azonos, de a topic és a frame_id külön van,
        # hogy a downstream node-ok (pl. depth_image_proc,
        # rtabmap, RViz) a megszokott két külön topicot találják meg.
        depth_info_topic = f"robot{i}/camera/depth/camera_info"
        depth_info_writer = rep.writers.get("ROS2PublishCameraInfo")
        depth_info_writer.initialize(
            frameId=depth_frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=depth_info_topic,
            width=camera_info["width"],
            height=camera_info["height"],
            projectionType=camera_info["projectionType"],
            k=camera_info["k"].reshape([1, 9]),
            r=camera_info["r"].reshape([1, 9]),
            p=camera_info["p"].reshape([1, 12]),
            physicalDistortionModel=camera_info["physicalDistortionModel"],
            physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
        )
        depth_info_writer.attach([render_product])

        depth_info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(depth_info_gate_path + ".inputs:step").set(step_size)

        # ---- Point Cloud (depth -> 3D pontfelhő, GPU-n, RTX/Replicator) ----
        # A hivatalos Isaac Sim minta szerinti "<rendervar>ROS2PublishPointCloud"
        # writer: a depth képet a kamera intrinsics-e alapján konvertálja
        # sensor_msgs/PointCloud2 üzenetté, ugyanazon render product-ból,
        # amit a depth image writer is használ. Ez felel meg a valódi
        # RealSense driver "depth/color/points" topicjának.
        pointcloud_topic = f"robot{i}/camera/depth/color/points"
        pointcloud_writer = rep.writers.get(rv_depth + "ROS2PublishPointCloud")
        pointcloud_writer.initialize(
            frameId=depth_frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=pointcloud_topic,
        )
        pointcloud_writer.attach([render_product])

        pointcloud_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_depth + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(pointcloud_gate_path + ".inputs:step").set(step_size)

    return all_succeeded


def pub_robo_data_ros2(robot_type, num_envs, base_node, env, annotator_lst, start_time):
    robot_data = env.unwrapped.scene["robot"].data
    joint_pos = _to_numpy(robot_data.joint_pos)
    root_state = _to_numpy(robot_data.root_state_w)
    lin_vel_b = _to_numpy(robot_data.root_lin_vel_b)
    ang_vel_b = _to_numpy(robot_data.root_ang_vel_b)

    # Egyetlen közös időbélyeg ebben a lépésben minden topichoz/TF-hez.
    # A go2_base.cpp kifejezetten a bejövő /utlidar/robot_pose header
    # stampjét használja az odom->base_link TF-hez és a /odom-hoz (NEM
    # a saját now()-ját), ezért itt is fontos, hogy a robot_pose és az
    # ezzel egy pillanathoz tartozó egyéb üzenetek (imu, joint_states,
    # opcionális info-odom) UGYANAZT az időbélyeget kapják, különben a
    # TF fa és a szenzoradatok között csúszás keletkezik.
    stamp = base_node.get_clock().now().to_msg()

    for i in range(num_envs):
        base_node.publish_joints(robot_data.joint_names, joint_pos[i], i, stamp)
        # A go2_base.cpp EGYEDÜLI póz-bemenete: ebből épül fel az
        # odom -> base_link TF és a /odom topic a C++ oldalon.
        base_node.publish_robot_pose(root_state[i, :3], root_state[i, 3:7], i, stamp)
        # Tájékoztató jellegű, nem-TF odom topic (nem ütközik a
        # go2_base.cpp által kiadott /odom-mal, mert más néven fut).
        base_node.publish_odom_info(root_state[i, :3], root_state[i, 3:7], i, stamp)
        base_node.publish_imu(root_state[i, 3:7], lin_vel_b[i, :], ang_vel_b[i, :], i, stamp)

        if robot_type == "go2":
            net_forces = _to_numpy(env.unwrapped.scene["contact_forces"].data.net_forces_w)
            base_node.publish_robot_state([
                net_forces[i][4][2],
                net_forces[i][8][2],
                net_forces[i][14][2],
                net_forces[i][18][2],
            ], i)

    if not hasattr(base_node, 'last_lidar_time'):
        base_node.last_lidar_time = time.time()

    if (time.time() - base_node.last_lidar_time) > 0.05:
        base_node.last_lidar_time = time.time()
        for j in range(num_envs):
            try:
                data = annotator_lst[j].get_data()
                if data and 'data' in data:
                    pts = _to_numpy(data['data'])
                    if pts is not None and len(pts) > 0:
                        # A pontfelhőt a lidar SAJÁT (lokális) frame-jében
                        # küldjük, világkoordinátás transzformáció NÉLKÜL.
                        # A world-be helyezést a TF fa végzi (odom ->
                        # base_link a go2_base.cpp-ből, base_link ->
                        # utlidar_lidar a robot_state_publisher URDF-ből).
                        base_node.publish_lidar(pts, j, stamp)
            except Exception as e:
                print(f"[WARN] Lidar error: {e}")


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__('go2_driver_node')
        qos_profile = QoSProfile(depth=10)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_pub = []
        self.odom_info_pub = []
        self.imu_pub = []
        self.pose_pub = []

        for i in range(num_envs):
            self.joint_pub.append(self.create_publisher(JointState, f'robot{i}/joint_states', qos_profile))
            self.go2_state_pub.append(self.create_publisher(Float32MultiArray, f'robot{i}/foot_force', qos_profile))
            self.go2_lidar_pub.append(self.create_publisher(PointCloud2, f'/utlidar/cloud_deskewed', qos_profile))
            # Csak tájékoztató jellegű topic. FIGYELEM: EZ NEM a
            # go2_base.cpp által kiadott /odom - azt szándékosan nem
            # dupláljuk innen, hogy ne legyen két versengő forrás
            # ugyanarra a topicra.
            self.odom_info_pub.append(self.create_publisher(Odometry, f'/utlidar/robot_odom', qos_profile))
            self.imu_pub.append(self.create_publisher(Imu, f'robot{i}/imu', qos_profile))
            # A go2_base.cpp EGYEDÜLI bemenete: ebből számolja ki saját
            # maga az odom->base_link TF-et és a /odom-ot.
            self.pose_pub.append(self.create_publisher(PoseStamped, f'/utlidar/robot_pose', qos_profile))

        # A kamera (RGB, depth, camera_info, point cloud) topicokat NEM
        # ez a node hozza létre - azokat a natív rep.writers/omni.graph
        # ROS2 publisher node-jai adják ki közvetlenül (lásd
        # add_camera_ros2_publisher() a fájl elején), ugyanúgy, ahogy a
        # LIDAR annotátor adatait is ez a Python node publikálja tovább.

        # FONTOS: ez a node SZÁNDÉKOSAN NEM hoz létre /tf publisher-t és
        # NEM publikál odom->base_link TF-et. Azt a go2_base.cpp végzi a
        # /utlidar/robot_pose alapján (a helyes, msg->header.stamp
        # időbélyeggel). A base_link -> imu_link / base_link ->
        # utlidar_lidar (stb.) ágakat pedig a robot_state_publisher adja
        # az URDF-ből. Ha itt is publikálnánk odom->base_link-et, két
        # forrás versengene ugyanazon a TF élen -> villódzó/ugráló póz.

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num, stamp):
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = [f"robot{robot_num}/{n}" for n in joint_names_lst]
        joint_state.position = [float(v.item()) for v in joint_state_lst]
        self.joint_pub[robot_num].publish(joint_state)

    def publish_robot_pose(self, base_pos, base_rot, robot_num, stamp):
        """A go2_base.cpp (go2_core) EGYEDÜLI póz-bemenete.

        A go2_base.cpp ebből az üzenetből állítja elő saját maga:
          - az odom -> base_link TF-et (a msg fejlécének stampjével,
            NEM a sajátjával - ezért kritikus, hogy itt a helyes,
            szimulációs lépéshez tartozó stamp menjen ki)
          - a /odom (nav_msgs/Odometry) topicot
        Ezért itt SEM /tf-re, SEM /odom-ra nem publikálunk közvetlenül.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        self.pose_pub[robot_num].publish(pose_msg)

    def publish_odom_info(self, base_pos, base_rot, robot_num, stamp):
        """Kizárólag tájékoztató jellegű Odometry topic (/utlidar/robot_odom).

        Nem TF-et, nem is a /odom topicot adja ki - azokat a go2_base.cpp
        állítja elő a /utlidar/robot_pose-ból. Ez a topic csak azoknak a
        (opcionális) fogyasztóknak szól, akik közvetlenül szeretnék látni
        a szimulációs odometria adatot twist nélkül.
        """
        odom_topic = Odometry()
        odom_topic.header.stamp = stamp
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = "base_link"
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        self.odom_info_pub[robot_num].publish(odom_topic)

    def publish_imu(self, base_rot, base_lin_vel, base_ang_vel, robot_num, stamp):
        """Az IMU frame_id-je a robot_state_publisher URDF-jében definiált
        imu_link (a valós robotnál a launch fájl egy static_transform_
        publisher "híddal" köti össze a gyári 'utlidar_imu' nevet az
        URDF 'imu_link' nevével - a szimulációban egyenesen a helyes
        'imu_link'-et küldjük, így a híd nélkül is illeszkedik a TF fába).
        """
        imu_trans = Imu()
        imu_trans.header.stamp = stamp
        imu_trans.header.frame_id = IMU_FRAME_ID
        imu_trans.linear_acceleration.x = base_lin_vel[0].item()
        imu_trans.linear_acceleration.y = base_lin_vel[1].item()
        imu_trans.linear_acceleration.z = base_lin_vel[2].item()
        imu_trans.angular_velocity.x = base_ang_vel[0].item()
        imu_trans.angular_velocity.y = base_ang_vel[1].item()
        imu_trans.angular_velocity.z = base_ang_vel[2].item()
        imu_trans.orientation.x = base_rot[1].item()
        imu_trans.orientation.y = base_rot[2].item()
        imu_trans.orientation.z = base_rot[3].item()
        imu_trans.orientation.w = base_rot[0].item()
        self.imu_pub[robot_num].publish(imu_trans)

    def publish_robot_state(self, foot_force_lst, robot_num):
        msg = Float32MultiArray()
        msg.data = [float(v.item()) for v in foot_force_lst]
        self.go2_state_pub[robot_num].publish(msg)

    def publish_lidar(self, points, robot_num, stamp):
        """A pontfelhőt a lidar SAJÁT (lokális szenzor-) frame-jében
        küldjük ki, semmilyen world/odom transzformáció NÉLKÜL. A
        helyes elhelyezést a TF fa adja: odom -> base_link (go2_base.cpp,
        a /utlidar/robot_pose-ból) -> utlidar_lidar (robot_state_publisher,
        az URDF statikus geometriájából).
        """
        header = Header(frame_id=LIDAR_FRAME_ID)
        header.stamp = stamp
        self.go2_lidar_pub[robot_num].publish(_create_point_cloud2(header, points))