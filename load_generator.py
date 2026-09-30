"""Automatic load generation for CivilAI-FEA.

Implements, per the project brief:
  * dynamic wind pressure profile varying with height (power-law velocity
    profile, EC1-style dynamic pressure q = 0.613e-3 v(z)^2) applied as
    equivalent nodal forces on the windward face;
  * equivalent static seismic lateral forces (code-style base shear
    V = Cs*W distributed vertically proportional to W_i * z_i);
  * factored gravity (slab dead + live + facade + steel self-weight)
    applied as nodal loads via tributary areas.

All load functions return global (ndof,) force vectors in kN.
"""
from __future__ import annotations

import numpy as np

from fea_solver import RHO_STEEL_W

G_KN_M2 = 9.81


# ----------------------------------------------------------------- profiles
def wind_velocity(z, vb=40.0, alpha=0.16, z_ref=10.0):
    """Power-law mean wind velocity profile [m/s] at height z [m]."""
    z = np.maximum(np.asarray(z, dtype=float), 1.0)
    return vb * (z / z_ref) ** alpha


def wind_pressure(z, vb=40.0, alpha=0.16):
    """Dynamic wind pressure q(z) [kN/m2] = 0.613e-3 * v(z)^2."""
    v = wind_velocity(z, vb=vb, alpha=alpha)
    return 0.613e-3 * v * v


# ------------------------------------------------------------- gravity loads
def gravity_nodal_loads(model, sdl=3.0, ll=2.5, facade_knm2=2.0):
    """Unfactored slab/facade gravity loads per node.

    sdl  : superimposed dead load [kN/m2] on floor plate
    ll   : live load [kN/m2]
    facade_knm2 : facade weight per facade area [kN/m2], tributary to
                  perimeter nodes (story height times tributary length).
    Returns (Fg, Fq): dead (slab+facade) and live (ndof,) vectors, kN,
    negative Z (down). Levels 1..stories receive plate loads.
    """
    nn = len(model["nodes"])
    st, npx, npy = model["stories"], model["npx"], model["npy"]
    bx, by, h = model["bay_x"], model["bay_y"], model["story_h"]
    lvl, i_, j_ = model["node_level"], model["node_i"], model["node_j"]

    area = np.full(nn, bx * by)                       # interior tributary
    edge_x = (i_ == 0) | (i_ == model["nx"])
    edge_y = (j_ == 0) | (j_ == model["ny"])
    corner = edge_x & edge_y
    area[corner] = bx * by / 4.0
    area[(edge_x | edge_y) & ~corner] = bx * by / 2.0

    fac_len = np.zeros(nn)                            # facade tributary length
    perim = (edge_x | edge_y)
    fac_len[corner] = bx / 2.0 + by / 2.0
    # non-corner perimeter nodes border the facade on both sides along it,
    # so their tributary length is a full bay of the face they sit on
    fac_len[perim & ~corner] = np.where(
        (edge_x & ~corner)[perim & ~corner], by, bx)

    floor = lvl >= 1
    Fg = np.zeros(6 * nn)
    Fq = np.zeros(6 * nn)
    Fg[2::6] -= np.where(floor, area * sdl, 0.0)
    Fg[2::6] -= np.where(floor, fac_len * h * facade_knm2, 0.0)
    Fq[2::6] -= np.where(floor, area * ll, 0.0)
    return Fg, Fq


def steel_self_weight_nodal(model, sections_A):
    """Lump member self-weights to their end nodes (half each). Returns
    (ndof,) downward nodal load vector [kN]."""
    nn = len(model["nodes"])
    mi, mj = model["mi"], model["mj"]
    w = np.asarray(sections_A) * RHO_STEEL_W * model_L(model)
    F = np.zeros(6 * nn)
    np.add.at(F, 6 * mi + 2, -w / 2.0)     # uz DOF of node i (stride 6)
    np.add.at(F, 6 * mj + 2, -w / 2.0)     # uz DOF of node j
    return F


def model_L(model):
    """Member lengths (m) from node coordinates."""
    p = model["nodes"]
    return np.linalg.norm(p[model["mj"]] - p[model["mi"]], axis=1)


# ------------------------------------------------------------------ wind
def wind_nodal_loads(model, vb=40.0, cs_cd=1.3, direction="+x"):
    """Extreme wind: q(z) integrated over the windward face per story and
    distributed equally to the windward nodes of each floor.

    cs_cd : net pressure coefficient (windward + leeward suction + gust).
    Returns (ndof,) force vector [kN].
    """
    nn = len(model["nodes"])
    st, npx, npy = model["stories"], model["npx"], model["npy"]
    h = model["story_h"]
    width = model["bay_y"] * model["ny"] if direction in ("+x", "-x") \
        else model["bay_x"] * model["nx"]

    F = np.zeros(6 * nn)
    for lvl in range(1, st + 1):
        z = lvl * h
        F_level = wind_pressure(z, vb=vb) * width * h * cs_cd
        mask = (model["node_level"] == lvl)
        if direction in ("+x", "-x"):
            mask &= (model["node_i"] == (model["nx"] if direction == "+x" else 0))
            F[0::6][mask] += F_level / mask.sum() * (1 if direction == "+x" else -1)
        else:
            mask &= (model["node_j"] == (model["ny"] if direction == "+y" else 0))
            F[1::6][mask] += F_level / mask.sum() * (1 if direction == "+y" else -1)
    return F


# ------------------------------------------------------------------ seismic
def seismic_weight_per_node(model, Fg_dead, Fq_live, psi_q=0.3):
    """Seismic nodal weights W = G + psi*Q [kN] (from gravity load vectors)."""
    return -(Fg_dead[2::6] + psi_q * Fq_live[2::6])


def seismic_equivalent_static(model, W_node, cs=0.10, direction="+x"):
    """Equivalent static lateral force (IBC/EC8-style first-mode shape).

    V = cs * sum(W); F_i = V * W_i * z_i / sum(W_j z_j), applied at every
    node of floor i (symmetric grid -> no torsion).
    """
    nn = len(model["nodes"])
    st, h = model["stories"], model["story_h"]
    W = np.asarray(W_node, dtype=float)
    lvl = model["node_level"]
    z = lvl * h

    W_floor = np.array([W[lvl == l].sum() for l in range(1, st + 1)])
    z_floor = np.array([l * h for l in range(1, st + 1)])
    V = cs * W_floor.sum()
    F_floor = V * W_floor * z_floor / np.sum(W_floor * z_floor)

    F = np.zeros(6 * nn)
    for k, l in enumerate(range(1, st + 1)):
        mask = (lvl == l)
        comp = 0 if direction in ("+x", "-x") else 1
        sgn = 1.0 if direction in ("+x", "+y") else -1.0
        F[comp::6][mask] += sgn * F_floor[k] / mask.sum()
    return F


# ------------------------------------------------------------------ combos
def load_combinations(model, sections_A, sdl=3.0, ll=2.5, facade_knm2=2.0,
                      vb=40.0, cs=0.10, cs_cd=1.3):
    """Build the factored load-case dictionary (extreme-event set).

    LC GRAV : 1.35 G + 1.50 Q
    LC WINDX: 1.00 G + 0.50 Q + 1.50 Wx   (extreme hurricane profile)
    LC WINDY: 1.00 G + 0.50 Q + 1.50 Wy
    LC EQX  : 1.00 G + 0.50 Q + 1.00 Ex   (Cs = cs * W)
    LC EQY  : 1.00 G + 0.50 Q + 1.00 Ey
    """
    Fg, Fq = gravity_nodal_loads(model, sdl=sdl, ll=ll, facade_knm2=facade_knm2)
    Fsw = steel_self_weight_nodal(model, sections_A)
    G = Fg + Fsw
    Q = Fq
    Wx = wind_nodal_loads(model, vb=vb, cs_cd=cs_cd, direction="+x")
    Wy = wind_nodal_loads(model, vb=vb, cs_cd=cs_cd, direction="+y")
    W_node = seismic_weight_per_node(model, G, Q)
    Ex = seismic_equivalent_static(model, W_node, cs=cs, direction="+x")
    Ey = seismic_equivalent_static(model, W_node, cs=cs, direction="+y")

    return {
        "GRAV": 1.35 * G + 1.50 * Q,
        "WINDX": 1.00 * G + 0.50 * Q + 1.50 * Wx,
        "WINDY": 1.00 * G + 0.50 * Q + 1.50 * Wy,
        "EQX": 1.00 * G + 0.50 * Q + 1.00 * Ex,
        "EQY": 1.00 * G + 0.50 * Q + 1.00 * Ey,
        "_diag": {"V_base_wind_x": float(np.sum(Wx[0::6])),
                  "V_base_eq_x": float(np.sum(Ex[0::6])),
                  "W_seismic_total": float(W_node.sum())},
    }
