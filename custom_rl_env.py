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


import math
import torch
from dataclasses import MISSING
from typing import Literal
import argparse  

from isaaclab.terrains import TerrainImporterCfg, TerrainGenerator
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab_assets import UNITREE_GO2_CFG
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from terrain_cfg import ROUGH_TERRAINS_CFG
from stairs4 import (STAIRS4_TERRAINS_CFG, STAIRS4_CURRICULUM_CFG, Stairs4CommandCfg, reset_stairs_start, stairs_done,
                     stairs_levels, progress, feet_stumble, foot_clearance, stance_width, base_height_local,
                     edge_fear, fell_off)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from robots.g1.config import G1_CFG

from omniverse_sim import args_cli


# NOTE: resolve Isaac's bundled studio HDRI once at module load (a plain string, so
# it survives @configclass deepcopy — module objects in a class body cannot be pickled).
# Glob because the extscache path is version-pinned and shifts on Isaac upgrades; empty
# string falls back to a flat neutral dome if the asset ever moves.
def _find_studio_hdri():
    import glob, os
    import isaacsim
    hits = glob.glob(
        os.path.dirname(isaacsim.__file__) + "/**/domeLight/photo_studio_01_4k.hdr",
        recursive=True,
    )
    return hits[0] if hits else ""


_STUDIO_HDRI = _find_studio_hdri()


base_command = {}


def constant_commands(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    global base_command
    """The generated command from the command generator."""
    tensor_lst = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    for i in range(env.num_envs):
        tensor_lst[i] = torch.tensor(base_command[str(i)], dtype=torch.float32, device=env.device)
    return tensor_lst


# Biztosítjuk az IsaacLab kompatibilitást a hiányzó attribútumok pótlásával
if ROUGH_TERRAINS_CFG is not None:
    ROUGH_TERRAINS_CFG.class_type = TerrainGenerator
    if not hasattr(ROUGH_TERRAINS_CFG, "border_height"):
        ROUGH_TERRAINS_CFG.border_height = 0.0


class KeyboardVelocityCommand(mdp.UniformVelocityCommand):
    """A billentyű/ROS parancsot (base_command) tükrözi, hogy a robotok felett látszó nyíl a valós irányt mutassa."""
    def _update_command(self):
        self.vel_command_b[:] = constant_commands(self._env)


@configclass
class KeyboardVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = KeyboardVelocityCommand


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    if args_cli.terrain == "flat":
    
        # flat terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            debug_vis=False,
        )
    else:
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=STAIRS4_TERRAINS_CFG if args_cli.terrain == "stairs4" else ROUGH_TERRAINS_CFG,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

    # robots
    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # NOTE: light the dome with Isaac's bundled studio HDRI (see _STUDIO_HDRI above)
    # so the Go2's shells catch real reflections instead of reading as flat gray.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0), intensity=900.0,
            #texture_file=_STUDIO_HDRI or None,
            texture_file= None,
        ),
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1920, 1080)

    origin_type: Literal["world", "env", "asset_root"] = "world"

    env_index: int = 0

    asset_name: str | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = KeyboardVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg()
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class UnitreeGo2CustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 🚀 KEZDŐPOZÍCIÓ EMELÉSE (X, Y, Z) - pl. Z = 0.6 vagy 0.8 méter
        self.scene.robot.init_state.pos = (1.0, 1.0, 0.7)

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        # torque-t itt nem emelem: a lépcsőn nagyobb erő kell, csak a jutalomsúlyt hangolom (lásd Stairs4Train)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2Stairs4EnvCfg(UnitreeGo2CustomEnvCfg):
    """--terrain stairs4: lépcsős világ, billentyűs/ROS parancsokkal; a height scan (elevation pontfelhő) látszik."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.45)
        self.scene.height_scanner.debug_vis = True
        self.events.reset_start = EventTerm(func=reset_stairs_start, mode="reset", params={"probs": (1.0, 0.0, 0.0), "mirror": False, "spread": True, "rand_start": False})
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Stairs4CurriculumCfg:
    terrain_levels = CurrTerm(func=stairs_levels)


@configclass
class UnitreeGo2Stairs4TrainEnvCfg(UnitreeGo2Stairs4EnvCfg):
    """Tanításhoz: véletlen sebességparancs a keyboard helyett, lépcsőhöz hangolt jutalmak."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.debug_vis = False
        self.episode_length_s = 10.0
        # 1) lépcsőmagasság-curriculum: 5 -> 17 cm, 7 szint
        self.scene.terrain.terrain_generator = STAIRS4_CURRICULUM_CFG
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum = Stairs4CurriculumCfg()
        # UP nehezebb (felfelé kapaszkodás, teljes parancssebesség), DOWN könnyebb (gravitáció segít, 0.8x sebesség,
        # már a pihenőről indul) -> ugyanolyan reset-valószínűség mellett is a DOWN epizódok élnek tovább, ezért
        # egy adott pillanatban a robotok nagy része DOWN-ban látszik. Több UP-reset, hogy több gyakorlást kapjon.
        self.events.reset_start = EventTerm(func=reset_stairs_start, mode="reset", params={"probs": (0.55, 0.3, 0.15)})
        self.terminations.stairs_done = DoneTerm(func=stairs_done, time_out=True)
        self.terminations.fell_off = DoneTerm(func=fell_off)
        self.observations.policy.velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        self.commands.base_velocity = c = Stairs4CommandCfg(asset_name="robot", resampling_time_range=(4.0, 8.0))
        c.heading_command, c.rel_standing_envs, c.debug_vis = False, 0.0, False
        # 0.15-0.3 m/s túl lassú volt: a robot inkább megállt a lépcsőn, mint hogy nekimenjen; 0.4-0.6-nál viszont
        # "leszánkázott" a fokokon. Középút, hogy tényleg lépkedve, de megállás nélkül menjen fel/le.
        c.ranges = mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.3, 0.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3))
        self.rewards.lin_vel_z_l2.weight = -0.5
        # dölöngélés (roll/pitch billegés) ellen: alapból szinte büntetlen volt (flat_orientation_l2=0,
        # ang_vel_xy_l2 csak -0.05), ez engedte, hogy a robot oldalra dőlve is jó jutalmat kapjon.
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.2
        # alacsonyabb testhelyzet (a height_scanner-rel a lokális talajhoz igazítva, lépcsőn is helyes) +
        # szélesebb terpesz (a jelenlegi túl közeli lábállás ellen)
        self.rewards.base_height = RewTerm(func=base_height_local, weight=-5.0, params={"target_height": 0.33})
        self.rewards.stance_width = RewTerm(
            func=stance_width, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})
        # 3) lépcső-jutalmak
        self.rewards.feet_air_time.weight = 0.15  # sík/lépcső ritmusa: emelje fel a lábát, de csak lépéshez, ne mindig
        self.rewards.dof_torques_l2.weight = -0.0005
        self.rewards.progress = RewTerm(func=progress, weight=1.0)
        self.rewards.dof_pos_limits.weight = -2.0
        # comb/lábszár beakadás a lépcsőélen (hátrafelé lemenetnél a lábszár akad)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_(thigh|calf)"), "threshold": 1.0})
        self.rewards.feet_stumble = RewTerm(
            func=feet_stumble, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})
        # peremfélelem: ne közelítsen a lépcsőoszlop oldalához, ha már magasan van (leesés/oldalfal-mászás ellen)
        self.rewards.edge_fear = RewTerm(func=edge_fear, weight=-3.0)
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.9)


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        G1_MINIMAL_CFG = G1_CFG.copy()
        G1_MINIMAL_CFG.spawn.usd_path = "./robots/g1/g1.usd"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.undesired_contacts = None

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link"]