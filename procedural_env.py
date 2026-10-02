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
"""
procedural_env.py
==================

Procedurálisan generált Isaac Sim környezetek egyszerű primitívekből
(kocka, henger, gömb). Nem igényel külső USD fájlt: a `isaaclab.sim`
(sim_utils) spawner API-jával közvetlenül a Stage-re rajzolja az
objektumokat, ugyanúgy ahogy a main.py a `resolve_usd_path`/
`setup_custom_env` során az USD környezeteket tölti be.

Támogatott típusok (--proc_type):
    - "rooms"     : falakból álló szoba/labirintus rendszer
    - "hills"     : dombos, egyenetlen terep (kockákból "heightfield")
    - "stairs"    : lépcsős szintek
    - "obstacles" : akadálypálya (random dobozok/hengerek/gömbök)

Minden generátor determinisztikus egy adott --seed mellett.

Használat a main.py-ból:

    from procedural_env import build_procedural_env

    build_procedural_env(
        proc_type="rooms",
        seed=42,
        prim_root="/World/Procedural",
    )
"""

from __future__ import annotations

import random
import math
from typing import Optional

import isaaclab.sim as sim_utils


def _ckpt(msg: str):
    print(f"[procedural_env] {msg}", flush=True)


# ------------------------------------------------------------------------------------
# Segéd: primitív spawnolás
# ------------------------------------------------------------------------------------

def _spawn_box(prim_path: str, size, position, color=(0.6, 0.6, 0.6)):
    """Egy dobozt (fal, kocka, lépcsőfok, stb.) helyez el a Stage-en."""
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(prim_path, cfg, translation=position)


def _spawn_cylinder(prim_path: str, radius, height, position, color=(0.55, 0.4, 0.3)):
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(prim_path, cfg, translation=position)


def _spawn_sphere(prim_path: str, radius, position, color=(0.3, 0.5, 0.8)):
    cfg = sim_utils.SphereCfg(
        radius=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(prim_path, cfg, translation=position)


def _spawn_ground_plane(prim_path: str, size=(60.0, 60.0)):
    cfg = sim_utils.GroundPlaneCfg(size=size)
    cfg.func(prim_path, cfg)


# ------------------------------------------------------------------------------------
# 1) Szoba / labirintus generátor
# ------------------------------------------------------------------------------------

def _generate_rooms(rng: random.Random, root: str, grid_size: int = 6, cell: float = 4.0,
                     wall_height: float = 2.0, wall_thickness: float = 0.2):
    """
    Egyszerű grid-alapú labirintus: minden cella közötti falat egy adott
    valószínűséggel megtartunk vagy elbontunk (recursive-backtracker helyett
    egyszerű random törlés, ami könnyen bejárható, de mégis útvesztő-szerű).
    """
    _ckpt(f"[rooms] generálás: grid={grid_size}x{grid_size}, cell={cell}m")

    half = grid_size * cell / 2.0
    idx = 0

    # Külső keret + belső rácsfalak
    for i in range(grid_size + 1):
        for j in range(grid_size):
            # Függőleges (X irányra merőleges) falszakaszok
            if i == 0 or i == grid_size or rng.random() < 0.35:
                x = -half + i * cell
                y = -half + (j + 0.5) * cell
                _spawn_box(
                    f"{root}/wall_v_{idx}",
                    size=(wall_thickness, cell, wall_height),
                    position=(x, y, wall_height / 2.0),
                )
                idx += 1

    for j in range(grid_size + 1):
        for i in range(grid_size):
            if j == 0 or j == grid_size or rng.random() < 0.35:
                x = -half + (i + 0.5) * cell
                y = -half + j * cell
                _spawn_box(
                    f"{root}/wall_h_{idx}",
                    size=(cell, wall_thickness, wall_height),
                    position=(x, y, wall_height / 2.0),
                )
                idx += 1

    _ckpt(f"[rooms] kész, {idx} falszakasz generálva")


# ------------------------------------------------------------------------------------
# 2) Dombos terep generátor
# ------------------------------------------------------------------------------------

def _generate_hills(rng: random.Random, root: str, grid_n: int = 14, cell: float = 2.0,
                     max_height: float = 1.2, octaves: int = 3):
    """
    Egyszerű "value-noise"-szerű dombos terep: minden rácspontra egy
    magasságértéket generálunk (több frekvenciájú szinusz-zajok összegével),
    majd minden cellába egy megfelelő magasságú kockát helyezünk, hogy a
    tetejük egy folytonos-nak ható, egyenetlen felszínt adjon ki.
    """
    _ckpt(f"[hills] generálás: grid={grid_n}x{grid_n}, cell={cell}m")

    phases = [(rng.uniform(0, math.tau), rng.uniform(0, math.tau)) for _ in range(octaves)]
    half = grid_n * cell / 2.0

    def height_at(x, y):
        h = 0.0
        for k, (px, py) in enumerate(phases, start=1):
            freq = 0.15 * k
            amp = max_height / k
            h += amp * math.sin(freq * x + px) * math.cos(freq * y + py)
        return max(0.15, h + max_height / 2.0)

    count = 0
    for i in range(grid_n):
        for j in range(grid_n):
            cx = -half + (i + 0.5) * cell
            cy = -half + (j + 0.5) * cell
            h = height_at(cx, cy)
            _spawn_box(
                f"{root}/hill_{i}_{j}",
                size=(cell * 1.02, cell * 1.02, h),
                position=(cx, cy, h / 2.0),
                color=(0.35 + 0.1 * rng.random(), 0.55, 0.3),
            )
            count += 1

    _ckpt(f"[hills] kész, {count} terepblokk generálva")


# ------------------------------------------------------------------------------------
# 3) Lépcsős szint generátor
# ------------------------------------------------------------------------------------

def _generate_stairs(rng: random.Random, root: str, num_flights: int = 4,
                      steps_per_flight: int = 8, step_depth: float = 0.35,
                      step_height: float = 0.15, step_width: float = 3.0,
                      landing_gap: float = 1.5):
    """
    Több lépcsőkarból álló szint, köztük sík pihenőkkel. Minden lépcsőkar
    után az irány véletlenszerűen tükröződhet (balra/jobbra fordulás
    illúziója), hogy ne legyen teljesen egyenes folyosó.
    """
    _ckpt(f"[stairs] generálás: {num_flights} lépcsőkar, karonként {steps_per_flight} fok")

    x = 0.0
    z = 0.0
    idx = 0

    # Kiinduló pihenő platform
    _spawn_box(f"{root}/landing_start", size=(step_width, landing_gap, 0.1),
               position=(0.0, -landing_gap / 2.0, 0.05), color=(0.5, 0.5, 0.55))

    y = 0.0
    for f in range(num_flights):
        for s in range(steps_per_flight):
            z += step_height
            y += step_depth
            _spawn_box(
                f"{root}/step_{idx}",
                size=(step_width, step_depth, z),
                position=(x, y, z / 2.0),
                color=(0.5, 0.5, 0.55),
            )
            idx += 1

        # Pihenő a lépcsőkar tetején
        y += landing_gap / 2.0
        _spawn_box(
            f"{root}/landing_{f}",
            size=(step_width, landing_gap, 0.1),
            position=(x, y, z + 0.05),
            color=(0.45, 0.45, 0.5),
        )
        y += landing_gap / 2.0

        # Néha oldalirányba tolja a következő kart, hogy ne legyen monoton
        if rng.random() < 0.5:
            x += rng.choice([-1, 1]) * step_width * 0.8

    _ckpt(f"[stairs] kész, {idx} lépcsőfok + {num_flights + 1} pihenő generálva")


# ------------------------------------------------------------------------------------
# 4) Akadálypálya generátor
# ------------------------------------------------------------------------------------

def _generate_obstacles(rng: random.Random, root: str, num_obstacles: int = 40,
                         area: float = 20.0):
    """
    Random elszórt dobozok/hengerek/gömbök egy adott méretű területen,
    változatos méretekkel és magasságokkal — klasszikus RL akadálypálya.
    """
    _ckpt(f"[obstacles] generálás: {num_obstacles} akadály, terület={area}x{area}m")

    half = area / 2.0
    for i in range(num_obstacles):
        x = rng.uniform(-half, half)
        y = rng.uniform(-half, half)
        # Hagyjunk egy szabad kezdő zónát a robot(ok) spawn helyénél
        if abs(x) < 1.5 and abs(y) < 1.5:
            x += math.copysign(2.0, x if x != 0 else 1.0)

        shape = rng.choice(["box", "cylinder", "sphere"])
        if shape == "box":
            w = rng.uniform(0.3, 1.2)
            d = rng.uniform(0.3, 1.2)
            h = rng.uniform(0.2, 1.0)
            _spawn_box(f"{root}/obs_{i}", size=(w, d, h), position=(x, y, h / 2.0),
                       color=(0.7, 0.3, 0.2))
        elif shape == "cylinder":
            r = rng.uniform(0.2, 0.6)
            h = rng.uniform(0.3, 1.4)
            _spawn_cylinder(f"{root}/obs_{i}", radius=r, height=h, position=(x, y, h / 2.0),
                            color=(0.6, 0.45, 0.2))
        else:
            r = rng.uniform(0.25, 0.6)
            _spawn_sphere(f"{root}/obs_{i}", radius=r, position=(x, y, r), color=(0.2, 0.4, 0.7))

    _ckpt(f"[obstacles] kész, {num_obstacles} akadály generálva")


# ------------------------------------------------------------------------------------
# Publikus belépési pont
# ------------------------------------------------------------------------------------

_GENERATORS = {
    "rooms": _generate_rooms,
    "hills": _generate_hills,
    "stairs": _generate_stairs,
    "obstacles": _generate_obstacles,
}


def list_procedural_types() -> list:
    return list(_GENERATORS.keys())


def build_procedural_env(proc_type: str, seed: Optional[int] = None,
                          prim_root: str = "/World/Procedural",
                          add_ground: bool = True, **kwargs):
    """
    Felépíti a kiválasztott procedurális világot a Stage-en.

    Args:
        proc_type: "rooms" | "hills" | "stairs" | "obstacles"
        seed: opcionális egész szám a determinisztikus generáláshoz.
              Ha None, a rendszer órája alapján kap seedet (nem reprodukálható).
        prim_root: a generált objektumok szülő prim path-je.
        add_ground: helyezzen-e el alap talajsíkot a generátor előtt.
        **kwargs: a kiválasztott generátorfüggvénynek átadott extra paraméterek
                  (pl. grid_size, num_obstacles, stb.)
    """
    if proc_type not in _GENERATORS:
        raise ValueError(
            f"Ismeretlen procedurális típus: '{proc_type}'. "
            f"Választható: {list_procedural_types()}"
        )

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        _ckpt(f"[procedural] nincs megadva --seed, generált seed: {seed}")
    else:
        _ckpt(f"[procedural] seed: {seed}")

    rng = random.Random(seed)

    if add_ground:
        _spawn_ground_plane(f"{prim_root}/ground")

    generator_fn = _GENERATORS[proc_type]
    generator_fn(rng, prim_root, **kwargs)

    _ckpt(f"[procedural] '{proc_type}' típusú világ elkészült ({prim_root} alatt)")
    return seed