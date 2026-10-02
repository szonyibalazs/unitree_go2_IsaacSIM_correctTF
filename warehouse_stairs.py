"""--terrain warehouse_stairs: full warehouse + "stairs4" lépcsőpálya (SDF: fok 0.17 m, mélység 0.29 m, sík 1.5 m pihenő,
2 m széles, +x felé nő) a raktár középső, üres folyosóján.

A height scanner (RayCaster) csak EGY mesh-t lát, ezért a talajlap és a fokok egyetlen terep-mesh (mint stairs4.py)."""
import numpy as np
import trimesh
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

# A talajlap a raktár belsejét fedi (world x,y in [-30, 30]); a raktár saját padlója (GroundPlane/CollisionMesh) z=0-n van.
TILE = (60.0, 60.0)
# World koordináta, ahol a robot indul (SDF origó). A raktár középső folyosója x~[-17,-7], y~[-23,9] üres (felülnézeti térkép).
COURSE_XY = (-14.5, -8.0)
SLAB_TOP = -0.02  # a talajlap kicsit a raktárpadló alatt, hogy ne z-fightoljon vele
WIDTH = 2.0
# (közép x, méret x, felső z) az SDF-ből, a robot x=0-ban áll
STEPS = [(1.645, 0.29, 0.17), (1.935, 0.29, 0.34), (2.225, 0.29, 0.51), (3.12, 1.5, 0.68),
         (4.015, 0.29, 0.51), (4.305, 0.29, 0.34), (4.595, 0.29, 0.17)]


def _course(difficulty, cfg):
    tf = trimesh.transformations.translation_matrix
    ox, oy = COURSE_XY[0] + TILE[0] / 2, COURSE_XY[1] + TILE[1] / 2  # tile-lokális (a generátor a világ közepére tolja)
    meshes = [trimesh.creation.box((TILE[0], TILE[1], 0.1), tf((TILE[0] / 2, TILE[1] / 2, SLAB_TOP - 0.05)))]
    for cx, sx, top in STEPS:
        h = top - SLAB_TOP
        meshes.append(trimesh.creation.box((sx, WIDTH, h), tf((ox + cx, oy, SLAB_TOP + h / 2))))
    return meshes, np.array([ox, oy, 0.0])


@configclass
class WarehouseStairsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    function = _course


WAREHOUSE_STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=TILE, border_width=1.0, border_height=0.1, num_rows=1, num_cols=1, use_cache=False,
    sub_terrains={"course": WarehouseStairsTerrainCfg(size=TILE)},
)
