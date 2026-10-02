"""stairs4 világ (fokmagasság <= 19 cm, fokszélesség >= 25.4 cm, 1.5 m pihenő; eredetileg SDF) Isaac Lab mesh-terepként + reset esemény."""
import numpy as np
import torch
import trimesh
import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

X0 = 1.5  # a start (robot x=0) ennyire van a csempe szélétől
LANDING = 2.5  # hosszabb pihenő (sík felső rész), hogy elférjenek rajta a bevágások/lyukak
LEAD = 1.5  # sík szakasz a start és az első fok közt
TILE = (9.5, 3.0)
ROUGH_AMP = 0.02  # egyenetlenség a sík csempén [m]; kicsi, hogy a lépcső-analitika (jutalmak, reset) ne térjen el érdemben
HOLE_COLS, HOLE_ROWS, HOLE_PROB = 16, 8, 0.35  # még finomabb rács -> még kisebb lyukak; az első/utolsó oszlop mindig tömör


def _landing_holes(x0, h, tf):
    """A pihenő lyukas cellái: kis, egyenként izolált négyzetek maradnak ki (nincs két szomszédos lyuk, hogy sose
    nőjön össze egy nagy, átléphetetlen gödörré), jól szétszórva. Az első és utolsó oszlopcsík (ahol a lépcső
    csatlakozik) mindig tömör marad, hogy a pálya folytonos legyen."""
    cx, cy = LANDING / HOLE_COLS, 2.0 / HOLE_ROWS  # a 2.0 m a lépcsőoszlop szélessége (ld. HALF_W), nem a teljes csempe
    hole = np.zeros((HOLE_COLS, HOLE_ROWS), dtype=bool)
    cells = [(i, j) for i in range(1, HOLE_COLS - 1) for j in range(HOLE_ROWS)]
    np.random.shuffle(cells)
    for i, j in cells:
        neighbors = ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
        if np.random.rand() < HOLE_PROB and not any(0 <= ni < HOLE_COLS and 0 <= nj < HOLE_ROWS and hole[ni, nj] for ni, nj in neighbors):
            hole[i, j] = True
    boxes = []
    for i in range(HOLE_COLS):
        for j in range(HOLE_ROWS):
            if hole[i, j]:
                continue
            x, y = x0 + cx * (i + 0.5), (TILE[1] / 2 - 1.0) + cy * (j + 0.5)
            boxes.append(trimesh.creation.box((cx * 0.95, cy * 0.95, h), tf((x, y, h / 2))))
    return boxes


def _rough_bumps(tf, cell=0.25, amp=ROUGH_AMP):
    """Apró véletlen kiemelkedések az egész csempén, hogy a modell rögtön egyenetlen talajt is lásson
    (stabilitás), a lépcsőfokok geometriáját nem érinti (azok fölé kerülnek, elhanyagolható átfedéssel)."""
    xs, ys = np.arange(cell / 2, TILE[0], cell), np.arange(cell / 2, TILE[1], cell)
    boxes = []
    for x in xs:
        for y in ys:
            h = np.random.uniform(0, amp)
            if h > 0.002:
                boxes.append(trimesh.creation.box((cell * 0.9, cell * 0.9, h), tf((x, y, h / 2))))
    return boxes


def _going(riser):
    """Fokszélesség a fokmagasságból: 2*riser + going ~ 0.63 m, de min 25.4 cm (max 35 cm). Szabványos: 19/25.4 cm."""
    return np.clip(0.63 - 2 * riser, 0.254, 0.35) if isinstance(riser, (float, int)) else (0.63 - 2 * riser).clamp(0.254, 0.35)


def _mid(riser):
    """A pihenő közepe a pálya elejétől [m]; a pálya erre tükrös."""
    return LEAD + 3 * _going(riser) + LANDING / 2


def _stairs4(difficulty, cfg):
    level = min(int(difficulty * cfg.num_levels), cfg.num_levels - 1)
    r = _riser_of(cfg, level)
    g = _going(r)
    tf = trimesh.transformations.translation_matrix
    meshes = [trimesh.creation.box((TILE[0], TILE[1], 0.1), tf((TILE[0] / 2, TILE[1] / 2, -0.05)))]
    blocks = [(X0 + LEAD + (k - .5) * g, g, k * r) for k in (1, 2, 3)]                      # fel
    blocks += [(X0 + LEAD + 3 * g + LANDING + (3 - k + .5) * g, g, k * r) for k in (3, 2, 1)]  # le
    for x, sx, h in blocks:
        meshes.append(trimesh.creation.box((sx, 2.0, h), tf((x, TILE[1] / 2, h / 2))))
    meshes += _landing_holes(X0 + LEAD + 3 * g, 4 * r, tf)                                 # pihenő, lyukakkal
    meshes += _rough_bumps(tf)
    return meshes, np.array([X0, TILE[1] / 2, 0.0])


def _riser_of(cfg, level):
    lo, hi = cfg.riser_range
    return lo + (hi - lo) * level / max(cfg.num_levels - 1, 1)


@configclass
class Stairs4TerrainCfg(terrain_gen.SubTerrainBaseCfg):
    function = _stairs4
    riser_range: tuple = (0.19, 0.19)
    num_levels: int = 1  # = a generátor num_rows-a (curriculum szintek)


# játék: 1 csempe, szabványos 19 cm / 25.4 cm lépcső
STAIRS4_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=TILE, border_width=4.0, border_height=0.1, num_rows=1, num_cols=1, use_cache=False,
    sub_terrains={"stairs4": Stairs4TerrainCfg(size=TILE)},
)

# tanítás: 7 szint, 5 -> 19 cm (fokszélesség 35 -> 25.4 cm); a curriculum lépteti a robotokat a sorok közt
LEVELS = 7
STAIRS4_CURRICULUM_CFG = terrain_gen.TerrainGeneratorCfg(
    size=TILE, border_width=4.0, border_height=0.1, num_rows=LEVELS, num_cols=1, use_cache=False, curriculum=True,
    sub_terrains={"stairs4": Stairs4TerrainCfg(size=TILE, riser_range=(0.05, 0.19), num_levels=LEVELS)},
)


UP, DOWN, TURN = 0, 1, 2


def _riser(env, env_ids=slice(None)):
    """Az env-ek aktuális lépcsőmagassága (a terep sora = curriculum szint)."""
    sub = env.scene.terrain.cfg.terrain_generator.sub_terrains["stairs4"]
    return _riser_of(sub, env.scene.terrain.terrain_levels[env_ids].float())


def _course_h(u, riser):
    """Terepmagasság a pihenő közepétől mért u [m] távolságban (a pálya u=0-ra tükrös). riser u-val broadcastol."""
    k = (4 - torch.ceil((u.abs() - LANDING / 2) / _going(riser))).clamp(0, 4)
    return k * riser


def _max_h(u, riser, half=0.35, n=8):
    """Legmagasabb terep a [u-half, u+half] ablakban (spawn-magasság / lábemelés)."""
    off = torch.linspace(-half, half, n, device=u.device)
    return _course_h(u.unsqueeze(-1) + off, riser.unsqueeze(-1)).amax(-1)


def _state(env):
    if not hasattr(env, "_st_mode"):
        env._st_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._st_s = torch.ones(env.num_envs, device=env.device)
    return env._st_mode, env._st_s


def reset_stairs_start(env, env_ids, probs=(0.4, 0.4, 0.2), mirror=True, spread=False, rand_start=True):
    """Reset. Módok: UP = fejjel előre felmegy; DOWN = a pihenőről HÁTRAFELÉ (a robot -x felé néz, +x-be halad) lemegy;
    TURN = a pihenőn helyben fordul. mirror: a pálya tükrözése a pihenő közepére (visszaút), yaw + pi.
    rand_start: véletlen kezdőpont a szakaszon belül (a lépcsőfokokon is), hogy a nehéz rész több mintát kapjon."""
    mode_all, s_all = _state(env)
    n = len(env_ids)
    mode = torch.multinomial(torch.tensor(probs, device=env.device), n, replacement=True)
    s = torch.where(torch.rand(n, device=env.device) < 0.5, 1.0, -1.0) if mirror else torch.ones(n, device=env.device)
    mode_all[env_ids], s_all[env_ids] = mode, s
    riser = _riser(env, env_ids)
    mid = _mid(riser)
    r = torch.rand(n, device=env.device) if rand_start else torch.zeros(n, device=env.device)
    # UP: a lépcső előtt (-mid) .. a 3. fokig; DOWN: a pihenőről (-0.3) .. az utolsó lefelé vezető fokig; TURN: a pihenőn
    # (a "belenyúlás" a pihenő pereme után a lépcsőbe LANDING-hez képest skálázva, hogy hosszabb pihenőnél is a
    # stairs4 tréning-mezőnyt találja el, ne ragadjon a pihenőn)
    up_hi, down_hi = -(LANDING / 2 + 0.25), LANDING / 2 + 0.65
    u = torch.where(mode == UP, -mid + (mid + up_hi) * r, torch.where(mode == DOWN, -0.3 + (down_hi + 0.3) * r, -0.3 + 0.6 * r))
    yaw = torch.where(mode == DOWN, np.pi, 0.0) + torch.where(s < 0, np.pi, 0.0)
    yaw = yaw + torch.where(mode == TURN, (torch.rand(n, device=env.device) < 0.5).float() * np.pi, 0.0)
    robot = env.scene["robot"]
    st = robot.data.default_root_state[env_ids].clone()
    st[:, 0] += env.scene.env_origins[env_ids, 0] + mid + s * u
    y = (torch.rand(n, device=env.device) - 0.5) * 0.6
    if spread:  # játékmód: a robotok egymás mellett, ne ugyanarra a pontra
        y = (env_ids.float() - (env.num_envs - 1) / 2) * 1.6 / max(env.num_envs - 1, 1)
    st[:, 1] += env.scene.env_origins[env_ids, 1] + y
    st[:, 2] += env.scene.env_origins[env_ids, 2] + _max_h(u, riser)
    st[:, 3:7] = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], dim=-1)
    st[:, 7:] = 0.0
    robot.write_root_pose_to_sim(st[:, :7], env_ids)
    robot.write_root_velocity_to_sim(st[:, 7:], env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids)


def stairs_done(env):
    """Sikeres szakasz vége (time_out=True -> nincs büntetés): UP a pihenőn, DOWN a lépcső után."""
    mode, s = _state(env)
    riser = _riser(env)
    u = s * (env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - _mid(riser))
    return ((mode == UP) & (u > 0.4)) | ((mode == DOWN) & (u > LANDING / 2 + 3 * _going(riser) + 0.5))


def stairs_levels(env, env_ids):
    """Curriculum: sikeres szakasz -> magasabb lépcső, elesés -> alacsonyabb."""
    tm = env.termination_manager
    up, down = tm.get_term("stairs_done")[env_ids], tm.get_term("base_contact")[env_ids]
    env.scene.terrain.update_env_origins(env_ids, up, down)
    return _riser(env).mean()


# ---------------- jutalmak ----------------

def progress(env, vmax=0.5):
    """Haladás a szakasz célja felé (világ x, tükrözve), UP/DOWN módban; felülről vágva, hogy ne rohanjon."""
    mode, s = _state(env)
    v = s * env.scene["robot"].data.root_lin_vel_w[:, 0]
    return torch.where(mode == TURN, 0.0, v.clamp(max=vmax))


def feet_stumble(env, sensor_cfg):
    """Láb a lépcső élébe ütközik: a talperő vízszintes része sokkal nagyobb a függőlegesnél."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (f[..., :2].norm(dim=-1) > 4 * f[..., 2].abs()).any(dim=1).float()


def base_height_local(env, target_height=0.33):
    """Törzsmagasság-jutalom a HELYI talajhoz igazítva. A stairs4 analitikus geometriájából számol
    (nem a height_scanner raycastból), mert az utóbbi resetkor átmenetileg -inf-et adhat vissza
    (a sugár nem talál felületet), ami NaN-en át szétveri a tanítást (RuntimeError: normal std >= 0.0)."""
    robot = env.scene["robot"]
    riser = _riser(env)
    u = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - _mid(riser)
    ground = _course_h(u, riser)
    h = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - ground
    return torch.square(h - target_height)


HALF_W = 1.0  # a lépcsőoszlop y félszélessége (a step-blockok 2.0 m szélesek, TILE[1]/2 közepén, ld. _stairs4)


def edge_fear(env, margin=0.25, edge_height=0.03):
    """Büntetés, ha a törzs a lépcsőoszlop laterális pereme felé közelít, miközben alatta már emelkedés van:
    ez tanítja meg félni a magasból eséstől (ne menjen a szélre) és kerülni a lépcső oldalfalát
    (ne próbáljon arra felmászni), ahelyett hogy csak a leesés UTÁN kapna büntetést (base_contact/fell_off)."""
    robot = env.scene["robot"]
    riser = _riser(env)
    u = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - _mid(riser)
    h = _course_h(u, riser)
    y = (robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]) - TILE[1] / 2
    over = (y.abs() - (HALF_W - margin)).clamp(min=0.0)
    return (h > edge_height).float() * over


def fell_off(env, drop=0.15):
    """Valódi leesés: a törzs a HELYI (lépcső-analitikus) talajszint alatt van (leesett az oldalperemen vagy a fokról),
    nem a világ-origóhoz képesti abszolút magasságot nézi (az a lépcső tetején sosem futna le, l. korábbi
    mdp.root_height_below_minimum használat)."""
    robot = env.scene["robot"]
    riser = _riser(env)
    u = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - _mid(riser)
    ground = _course_h(u, riser)
    h = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - ground
    return h < -drop


def stance_width(env, asset_cfg, target=0.28):
    """Terpesz-jutalom: a lábak csípőhöz (törzsközépponthoz) képesti oldaltávolsága (törzs y tengelye, yaw-forgatva)
    legalább `target` méter legyen; e fölött nincs plusz jutalom (ne terpeszkedjen indokolatlanul szét)."""
    robot = env.scene[asset_cfg.name]
    yaw = yaw_quat(robot.data.root_quat_w)
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids]
    offset_w = foot_pos_w - robot.data.root_pos_w[:, None, :3]
    offset_b = quat_apply_inverse(yaw.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1), offset_w)
    return offset_b[..., 1].abs().mean(dim=1).clamp(max=target)


def foot_clearance(env, asset_cfg, target=0.05):
    """Lengőláb (vízszintes seb. > 0.2 m/s) ne menjen target-nél alacsonyabban a KÖZVETLENÜL ELŐTTE lévő fok fölött.
    Csak ott bünteti a keveset emelést, ahol tényleg fok jön (a talajszinttől eltérő pont a láb előtt +-0.1 m-en
    belül); sík szakaszon _max_h mindenhol 0, tehát ott a láb nem kap indokolatlan "emeld magasra" jelet."""
    robot = env.scene[asset_cfg.name]
    p = robot.data.body_pos_w[:, asset_cfg.body_ids]
    swing = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1) > 0.2
    riser = _riser(env)
    u = p[..., 0] - env.scene.env_origins[:, 0:1] - _mid(riser).unsqueeze(-1)
    h_ahead = _max_h(u + 0.06, riser.unsqueeze(-1).expand_as(u), half=0.06, n=3)  # a lépés iránya elé nézünk
    clear = p[..., 2] - env.scene.env_origins[:, 2:3] - h_ahead
    return (swing * (target - clear).clamp(0, target)).sum(dim=1)


class Stairs4Command(UniformVelocityCommand):
    """Sebességparancs módonként: UP előre, DOWN hátra (vx<0), TURN helyben forog."""
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        mode, _ = _state(self._env)
        m, c = mode[env_ids], self.vel_command_b
        vx = c[env_ids, 0].abs()
        c[env_ids, 0] = torch.where(m == UP, vx, torch.where(m == DOWN, -0.8 * vx, 0.0))
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, 1.0, -1.0)
        c[env_ids, 1] = torch.where(m == TURN, 0.0, c[env_ids, 1])
        c[env_ids, 2] = torch.where(m == TURN, 0.6 * sign, c[env_ids, 2])


@configclass
class Stairs4CommandCfg(UniformVelocityCommandCfg):
    class_type: type = Stairs4Command
