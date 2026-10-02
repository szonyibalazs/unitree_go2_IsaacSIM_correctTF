# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import argparse
import os
import time
import threading
import glob
import json

reset_requested = False

# ==============================================================================
# 🎮 KÖRNYEZET VÁLASZTÓ CONFIG (ENVIRONMENT SELECTOR)
# ==============================================================================
SELECTED_ENVIRONMENT = "warehouse"
# ==============================================================================

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--custom_env", type=str, default=SELECTED_ENVIRONMENT, help="Setup the environment")
parser.add_argument("--terrain", type=str, default="flat", help="Setup the terrain (flat | stairs4 | rough)")
parser.add_argument("--list_envs", action="store_true", help="List all available environments and exit")

# --- ÚJ: procedurálisan generált világok ---
parser.add_argument(
    "--proc_type",
    type=str,
    default=None,
    choices=["rooms", "hills", "stairs", "obstacles"],
    help="Procedurális világ típusa. Csak akkor lép életbe, ha --custom_env "
         "értéke 'procedural' (vagy 'proc'). Választható: rooms, hills, stairs, obstacles.",
)
parser.add_argument(
    "--proc_seed",
    type=int,
    default=None,
    help="Seed a procedurális világgenerátorhoz (reprodukálható eredményhez). "
         "Ha nincs megadva, a --seed értéket használja, annak hiányában random seedet kap.",
)

parser.add_argument("--render_every", type=int, default=4,
                    help="Renderelés (viewport + RTX lidar + kamera) minden N-edik vezérlő lépésben. A fizika/vezérlés "
                         "továbbra is 50 Hz; N=4 -> 12.5 Hz szenzor (valós utlidar ~15 Hz; N=3: 16.7 Hz, ~36 Hz lépésráta). N=1: régi viselkedés.")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")
parser.add_argument("--twinbot", action="store_true", default=False,
                    help="Digital-twin mode: drive sim joints from real Go2 via /real_dog/joint_states "
                         "(requires twinbot_bridge.py running on the Jetson)")
parser.add_argument("--capture", type=int, default=0,
                    help="Capture N settle frames then write hero PNGs from several angles and exit "
                         "(headless-safe; uses an isaaclab Camera render product, not a window grab).")
parser.add_argument("--capture_dir", type=str, default="/tmp/twin_hero",
                    help="Directory to write hero PNGs into when --capture > 0.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _ckpt(msg: str):
    print(f"[go2_omniverse] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def discover_custom_usd_envs() -> dict:
    envs_dir = os.path.join(os.path.dirname(__file__), "envs")
    custom_envs = {}
    
    if os.path.exists(envs_dir):
        usd_files = glob.glob(os.path.join(envs_dir, "*.usd"))
        for usd_file in usd_files:
            env_name = os.path.splitext(os.path.basename(usd_file))[0]
            custom_envs[env_name] = os.path.abspath(usd_file)
    
    return custom_envs


# --- ÚJ: procedurális típusok, csak listázáshoz importáljuk könnyű súllyal ---
PROCEDURAL_ENV_NAMES = ("procedural", "proc")
PROCEDURAL_TYPES = ("rooms", "hills", "stairs", "obstacles")


def list_all_environments():
    builtin_envs = [
        "warehouse",
        "full-warehouse",
        "warehouse-forklifts",
        "warehouse-shelves",
        "office",
        "hospital",
        "obstacle-dense",
        "obstacle-medium",
        "obstacle-sparse",
    ]
    
    custom_envs = discover_custom_usd_envs()
    
    print("\n" + "="*70)
    print("📋 ELÉRHETŐ ISAAC SIM BEÉPÍTETT KÖRNYEZETEK:")
    print("="*70)
    for env in builtin_envs:
        print(f"  ✓ {env}")
    
    if custom_envs:
        print("\n" + "="*70)
        print("🎨 CUSTOM USD FÁJLOK (envs/ mappa):")
        print("="*70)
        for env_name, path in sorted(custom_envs.items()):
            print(f"  ✓ {env_name}")
            print(f"    └─ {path}")
    else:
        print("\n⚠️  Nem találhatók custom USD fájlok az envs/ mappában")

    # --- ÚJ: procedurális világok listázása ---
    print("\n" + "="*70)
    print("🧩 PROCEDURÁLISAN GENERÁLT VILÁGOK (--custom_env procedural):")
    print("="*70)
    for proc_type in PROCEDURAL_TYPES:
        print(f"  ✓ {proc_type}")
    print("\n💡 Használat:")
    print("  python main.py --custom_env procedural --proc_type rooms --proc_seed 42")

    print("="*70)
    print("\n💡 Használat:")
    print("  python main.py --custom_env <environment_name>\n")
    print("="*70 + "\n")


CUSTOM_USD_ENVS = discover_custom_usd_envs()

if args_cli.list_envs:
    list_all_environments()
    exit(0)

# --- ÚJ: ha procedurális világot kértek, de nem adtak meg típust, essünk hibára korán ---
if args_cli.custom_env in PROCEDURAL_ENV_NAMES and args_cli.proc_type is None:
    print(
        "[HIBA] --custom_env procedural esetén kötelező megadni a --proc_type "
        "paramétert (rooms | hills | stairs | obstacles)."
    )
    exit(1)


_ckpt("AppLauncher: constructing...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_ckpt("AppLauncher: ready")

import omni
import omni.ui as ui
_ckpt("import omni: ok")

ext_manager = omni.kit.app.get_app().get_extension_manager()

_required_exts = (
    "omni.graph.core",
    "omni.graph.action",
    "omni.graph.nodes",
    "isaacsim.sensors.rtx",
)
for _ext in _required_exts:
    _ckpt(f"requesting extension: {_ext}")
    ext_manager.set_extension_enabled(_ext, True)

_t0 = time.time()
while time.time() - _t0 < 120:
    simulation_app.update()
    pending = [e for e in _required_exts if not ext_manager.is_extension_enabled(e)]
    if not pending:
        break
    time.sleep(0.05)
still_pending = [e for e in _required_exts if not ext_manager.is_extension_enabled(e)]
_ckpt(f"extensions enabled in {time.time()-_t0:.2f}s (still_pending={still_pending})")

_ckpt("import gymnasium")
import gymnasium as gym
_ckpt("import torch")
import torch
_ckpt("import carb")
import carb

_ckpt("import isaaclab_tasks.utils.parse_cfg")
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path
_ckpt("import isaaclab_rl.rsl_rl")
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
_ckpt("import isaaclab.sim")
import isaaclab.sim as sim_utils
_ckpt("import omni.appwindow")
import omni.appwindow

_ckpt("import rclpy")
import rclpy
_ckpt("import ros2 module")

from ros2 import RobotBaseNode, add_camera, add_camera_ros2_publisher, add_rtx_lidar, pub_robo_data_ros2
from unitree_api.msg import Request as SportRequest

_ckpt("import agent_cfg")
from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
_ckpt("import custom_rl_env (this pulls isaaclab_assets Unitree USD cfg)")
from custom_rl_env import UnitreeGo2CustomEnvCfg, UnitreeGo2Stairs4EnvCfg, G1RoughEnvCfg
import custom_rl_env
_ckpt("import omnigraph")
from omnigraph import create_front_cam_omnigraph

# --- ÚJ: procedurális világgenerátor modul ---
_ckpt("import procedural_env")
from procedural_env import build_procedural_env, list_procedural_types

try:
    import env.sim_env as sim_env
    HAS_SIM_ENV = True
    _ckpt("import env.sim_env: success")
except ImportError:
    HAS_SIM_ENV = False
    _ckpt("import env.sim_env: nem található a projektben. Natív USD betöltő használata.")

_ckpt("all imports complete")


def setup_native_ui():
    window = ui.Window("Robot Vezérlés", width=220, height=100)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label("Robot Állapot Vezérlés", height=20)
            def on_reset():
                global reset_requested
                reset_requested = True
                _ckpt("[GUI] Reset gomb megnyomva!")
            
            btn = ui.Button("RESET ROBOT", clicked_fn=on_reset, height=40)
            btn.style = {"background_color": 0xFF3643F4}
    return window


def _load_mlp_policy(ckpt_path: str, hidden_dims, activation_name: str, device: str):
    import torch.nn as nn
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor_state_dict" in state:  # rsl_rl >= 5 formátum (train_stairs.py) -> régi kulcsnevek
        sd = {"actor." + k[4:]: v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.")}
    else:
        sd = state["model_state_dict"]
    actor_in = sd["actor.0.weight"].shape[1]
    actor_out = sd["actor.6.weight"].shape[0]
    activation = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation_name.lower()]
    layers = []
    dims = [actor_in, *hidden_dims]
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dims[-1], actor_out))
    actor = nn.Sequential(*layers)
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.to(device).eval()
    return actor, actor_in, actor_out


def sub_keyboard_event(event, *args, **kwargs) -> bool:
    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True


def resolve_usd_path(env_name: str) -> str:
    if env_name in CUSTOM_USD_ENVS:
        usd_path = CUSTOM_USD_ENVS[env_name]
        _ckpt(f"[KÖRNYEZET] ✓ Custom USD fájl megtalálva: {usd_path}")
        return usd_path
    
    usd_filename = env_name if env_name.endswith(".usd") else f"{env_name}.usd"
    
    local_paths = [
        os.path.join(".", "envs", usd_filename),
        os.path.join(os.path.dirname(__file__), "envs", usd_filename),
        os.path.join(".", usd_filename),
    ]
    for lp in local_paths:
        if os.path.exists(lp):
            _ckpt(f"[KÖRNYEZET] ✓ Helyi USD fájl megtalálva: {lp}")
            return os.path.abspath(lp)

    _ckpt(f"[KÖRNYEZET] ℹ️  Custom/helyi fájl ('{env_name}') nem található. Gyári Isaac Sim modellek keresése...")
    try:
        from isaacsim.storage.native import get_assets_root_path
        root_path = get_assets_root_path()
    except Exception:
        root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
        
    if not root_path:
        root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"

    name_lower = env_name.lower()
    
    # Warehouse-ek (100% működőek)
    if "full-warehouse" in name_lower or "full_warehouse" in name_lower:
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    elif "forklifts" in name_lower or "forklift" in name_lower:
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
    elif "shelves" in name_lower or "shelf" in name_lower:
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    elif "warehouse" in name_lower:
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    
    # Office/Iroda (100% működő)
    elif "office" in name_lower:
        return f"{root_path}/Isaac/Environments/Office/office.usd"
    
    # Hospital/Kórház (100% működő)
    elif "hospital" in name_lower:
        return f"{root_path}/Isaac/Environments/Hospital/hospital.usd"
    
    # Hangar (gyakorlási terület, 100% működő)
    elif "hangar" in name_lower or "practice" in name_lower:
        return f"{root_path}/Isaac/Environments/Hangar/hangar.usd"
    
    # Egyéb kísérleti (előfordulhat, hogy nem működik)
    elif "park" in name_lower:
        _ckpt("[FIGYELMEZTETÉS] Park nem garantáltan elérhető. Warehouse-ra visszalépés...")
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    
    else:
        _ckpt(f"[FIGYELMEZTETÉS] ⚠️  Ismeretlen környezetnév ('{env_name}'). Alapértelmezett Warehouse betöltése!")
        return f"{root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"


def setup_custom_env(env_name: str):
    # --- ÚJ: procedurális világ ág ---
    if env_name in PROCEDURAL_ENV_NAMES:
        proc_type = args_cli.proc_type
        proc_seed = args_cli.proc_seed if args_cli.proc_seed is not None else args_cli.seed
        _ckpt(f"[KÖRNYEZET] 🧩 Procedurális világ generálása: type='{proc_type}' seed={proc_seed}")
        try:
            used_seed = build_procedural_env(
                proc_type=proc_type,
                seed=proc_seed,
                prim_root="/World/Procedural",
            )
            _ckpt(f"[KÖRNYEZET] ✅ Procedurális világ elkészült (seed={used_seed}, típus={proc_type})")
        except Exception as ex:
            _ckpt(f"[KÖRNYEZET] ❌ Hiba a procedurális világ generálása közben: {ex}")
        return

    if HAS_SIM_ENV:
        try:
            if env_name == "warehouse" and hasattr(sim_env, "create_warehouse_env"):
                sim_env.create_warehouse_env()
                _ckpt("[KÖRNYEZET] ✓ Betöltve a sim_env.create_warehouse_env() segítségével.")
                return
            elif env_name == "full-warehouse" and hasattr(sim_env, "create_full_warehouse_env"):
                sim_env.create_full_warehouse_env()
                _ckpt("[KÖRNYEZET] ✓ Betöltve a sim_env.create_full_warehouse_env() segítségével.")
                return
            elif env_name == "warehouse-forklifts" and hasattr(sim_env, "create_warehouse_forklifts_env"):
                sim_env.create_warehouse_forklifts_env()
                _ckpt("[KÖRNYEZET] ✓ Betöltve a sim_env.create_warehouse_forklifts_env() segítségével.")
                return
            elif env_name == "obstacle-dense" and hasattr(sim_env, "create_obstacle_dense_env"):
                sim_env.create_obstacle_dense_env()
                _ckpt("[KÖRNYEZET] ✓ Betöltve a sim_env.create_obstacle_dense_env() segítségével.")
                return
        except Exception as e:
            _ckpt(f"[KÖRNYEZET] sim_env betöltő hibára futott ({e}), USD fallback...")

    usd_path = resolve_usd_path(env_name)
    _ckpt(f"[KÖRNYEZET] 📦 Színtér USD betöltése innen: {usd_path}")

    prim_name = f"/World/CustomEnvironment_{os.path.splitext(os.path.basename(usd_path))[0]}"

    try:
        usd_spawner = sim_utils.UsdFileCfg(usd_path=usd_path)
        usd_spawner.func(prim_name, usd_spawner, translation=(0.0, 0.0, 0.0))
        _ckpt(f"[KÖRNYEZET] ✅ Sikeres direkt USD betöltés a Stage-re: {prim_name}")
    except Exception as ex:
        _ckpt(f"[KÖRNYEZET] ❌ Hiba a direkt USD betöltés közben: {ex}")


def capture_hero_shots(env, policy, obs, device, n_settle, out_dir):
    import os
    import numpy as np
    from PIL import Image
    from isaaclab.sensors import Camera, CameraCfg

    os.makedirs(out_dir, exist_ok=True)
    return []


def _to_numpy_safe(arr):
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    import numpy as np
    return np.asarray(arr)


SPORT_API_STOPMOVE = 1003  # go2_sport_api/ros2_sport_client.h
SPORT_API_MOVE = 1008


def sport_request_cb(msg, num_robot):
    """A valós robot vezérlő interfésze: a go2_base (/cmd_vel -> /api/sport/request) és a
    joy teleop ide ír. Move: parameter = {"x":vx,"y":vy,"z":vyaw}. A többi api_id-t (stand,
    sit, ...) a sim nem modellezi."""
    api_id = msg.header.identity.api_id
    if api_id == SPORT_API_MOVE:
        p = json.loads(msg.parameter)
        custom_rl_env.base_command[str(num_robot)] = [p["x"], p["y"], p["z"]]
    elif api_id == SPORT_API_STOPMOVE:
        custom_rl_env.base_command[str(num_robot)] = [0, 0, 0]


def add_cmd_sub(num_envs):
    node_test = rclpy.create_node('sport_request_listener')
    for i in range(num_envs):
        topic = "/api/sport/request" if i == 0 else f"/robot{i}/api/sport/request"
        node_test.create_subscription(SportRequest, topic, lambda msg, i=i: sport_request_cb(msg, i), 10)
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()


def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]


@torch.inference_mode()
def reset_robot_physics(env):
    unwrapped_env = env.unwrapped
    while hasattr(unwrapped_env, "env"):
        unwrapped_env = unwrapped_env.env
    
    env_ids = torch.arange(unwrapped_env.num_envs, device=unwrapped_env.device)

    if hasattr(unwrapped_env, "reset_idx"):
        unwrapped_env.reset_idx(env_ids)
    
    if any("reset_start" in v for v in unwrapped_env.event_manager.active_terms.values()):
        return  # stairs4: a reset esemény már a helyes (szétosztott) pozícióba tette a robotokat

    robot = unwrapped_env.scene["robot"]
    
    default_root_state = robot.data.default_root_state.clone()
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    root_pos_w = default_root_state[:, :3] + unwrapped_env.scene.env_origins[env_ids]
    root_orient_w = default_root_state[:, 3:7].clone()
    root_pose = torch.cat([root_pos_w, root_orient_w], dim=-1)
    
    root_lin_vel = default_root_state[:, 7:10].clone()
    root_ang_vel = default_root_state[:, 10:13].clone()
    root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)

    robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    
    unwrapped_env.sim.step()


def run_sim():
    global reset_requested

    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    env_cfg = UnitreeGo2Stairs4EnvCfg() if args_cli.terrain == "stairs4" else UnitreeGo2CustomEnvCfg()

    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()

    env_cfg.scene.num_envs = args_cli.robot_amount
    env_cfg.sim.render_interval = env_cfg.decimation * args_cli.render_every
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    agent_cfg = unitree_go2_agent_cfg
    if args_cli.robot == "g1":
        agent_cfg = unitree_g1_agent_cfg
    if args_cli.terrain == "stairs4":
        agent_cfg = {**agent_cfg, "experiment_name": "unitree_go2_stairs4"}

    if args_cli.terrain == "flat" and args_cli.custom_env:
        _ckpt(f"[KÖRNYEZET] 🎯 Kiválasztott környezet: '{args_cli.custom_env}'")
        setup_custom_env(args_cli.custom_env)

    _ckpt(f"gym.make task={args_cli.task} num_envs={env_cfg.scene.num_envs}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    _ckpt("gym.make: done")
    
    env = RslRlVecEnvWrapper(env)
    _ckpt("RslRlVecEnvWrapper: wrapped")
    
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    device = str(env.unwrapped.device)
    actor, _, _ = _load_mlp_policy(
        resume_path,
        hidden_dims=agent_cfg["policy"]["actor_hidden_dims"],
        activation_name=agent_cfg["policy"]["activation"],
        device=device,
    )

    def policy(obs_dict):
        obs_tensor = obs_dict["policy"] if hasattr(obs_dict, "__getitem__") and "policy" in obs_dict else obs_dict
        return actor(obs_tensor)

    _ckpt("env.get_observations()")
    obs = env.get_observations()
    _ckpt("env.get_observations: done")

    _ckpt("rclpy.init + RobotBaseNode")
    rclpy.init()
    base_node = RobotBaseNode(env_cfg.scene.num_envs)
    add_cmd_sub(env_cfg.scene.num_envs)
    _ckpt("ROS2 publishers up")

    twin = None
    if args_cli.twinbot:
        from twinbot import TwinbotSubscriber
        twin = TwinbotSubscriber(env)
        _ckpt("TwinbotSubscriber ready")

    annotator_lst = []
    
    try:
        annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot)
        _ckpt("lidar added")
    except Exception as e:
        _ckpt(f"add_rtx_lidar skipped ({type(e).__name__}: {e})")

    try:
        camera_lst = add_camera(env_cfg.scene.num_envs, args_cli.robot)
        _ckpt("camera added")
        try:
            add_camera_ros2_publisher(env_cfg.scene.num_envs, camera_lst)
            _ckpt("camera ROS2 publisher (rgb/depth/info) added")
        except Exception as e:
            _ckpt(f"add_camera_ros2_publisher skipped ({type(e).__name__}: {e})")
    except Exception as e:
        camera_lst = []
        _ckpt(f"add_camera skipped ({type(e).__name__}: {e})")

    gui_window = setup_native_ui()
    _ckpt("Native Isaac Sim UI window created")

    _ckpt("entering main loop")

    start_time = time.time()
    _prof = [0.0, 0.0, 0]

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                _ckpt("[SIM] Force resetting robot pose and environment...")
                reset_robot_physics(env)
                obs = env.get_observations()
                reset_requested = False

            t0 = time.perf_counter()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            t1 = time.perf_counter()
            if twin is not None:
                twin.apply(device)
            pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, base_node, env, annotator_lst, start_time)
            t2 = time.perf_counter()
            if os.environ.get("GO2_PROFILE"):
                _prof[0] += t1 - t0; _prof[1] += t2 - t1; _prof[2] += 1
                if _prof[2] == 50:
                    _ckpt(f"[PROF] 50 step: env.step {_prof[0]*20:.1f} ms/step, ros-pub {_prof[1]*20:.1f} ms/step -> {50/(_prof[0]+_prof[1]):.1f} Hz")
                    _prof[:] = [0.0, 0.0, 0]
            
    env.close()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    run_sim()