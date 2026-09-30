"""Eurocode 3 (EN 1993-1-1) member checks for CivilAI-FEA.

Basic member verification based on FEA output:
  * axial yield (EN 3-1-1 §6.2.3)      N_t,Rd = A fy / gamma_M0
  * flexural buckling (§6.3.1)         N_b,Rd = chi A fy / gamma_M1,
    buckling curve 'b' (alpha=0.34), Euler Ncr about the weaker axis,
    effective-length factors: columns 1.5, beams 1.2 (sway-frame approximation)
  * strong-axis bending with lateral-torsional buckling (§6.3.2):
    Mcr from the standard I-section formula, curve 'b' for chi_LT
  * weak-axis bending                  M_v,Rd = Zy fy / gamma_M0
  * shear (§6.2.6)                     V_pl,Rd = Av fy / (sqrt3 gamma_M0)
  * linear interaction (§6.2.1 simplified):  N/NRd + M3/Mb,Rd + M2/Mv,Rd

All functions are vectorized over members; units kN, m (fy in kN/m2).
"""
from __future__ import annotations

import numpy as np

from fea_solver import E_STEEL, G_STEEL

FY = 355.0e3          # S355 steel yield strength [kN/m2]
GAMMA_M0 = 1.00
GAMMA_M1 = 1.00
K_EFF_COLUMN = 1.5    # sway moment frame, unbraced (conservative)
K_EFF_BEAM = 1.2

UTIL_LIMIT = 0.90     # project constraint: members <= 90% of capacity


def chi_buckling(lam_bar, alpha=0.34):
    """EC3 reduction factor chi for the given relative slenderness
    (Perry-Robertson format of §6.3.1.2). Vectorized."""
    lam_bar = np.maximum(np.asarray(lam_bar, float), 1e-9)
    phi = 0.5 * (1.0 + alpha * (lam_bar - 0.2) + lam_bar ** 2)
    disc = np.maximum(phi * phi - lam_bar ** 2, 1e-12)
    return np.minimum(1.0 / (phi + np.sqrt(disc)), 1.0)


def member_utilization(N, V2, V3, M2, M3, sec, L, is_column):
    """Vectorized EC3 utilization for every member.

    N, V2, V3, M2, M3 : (nm,) design forces (N signed: tension > 0)
    sec  : dict of per-member property arrays (A, Ix, Iy, J, Zx, Zy, d, bf,
           tw, tf, Cw, hw)
    L    : (nm,) member lengths
    is_column : (nm,) bool -> effective-length factor selection

    Returns (util, parts) where util is the governing utilization in [0, inf)
    and parts carries the individual checks for reporting.
    """
    Lcr = np.where(is_column, K_EFF_COLUMN, K_EFF_BEAM) * L

    # --- axial ---------------------------------------------------------------
    Nt_Rd = sec["A"] * FY / GAMMA_M0
    Ncr = np.minimum(np.pi ** 2 * E_STEEL * sec["Ix"] / Lcr ** 2,
                     np.pi ** 2 * E_STEEL * sec["Iy"] / Lcr ** 2)
    lam = np.sqrt(sec["A"] * FY / np.maximum(Ncr, 1e-9))
    chi = chi_buckling(lam)
    Nc_Rd = chi * sec["A"] * FY / GAMMA_M1
    N_Rd = np.where(N >= 0.0, Nt_Rd, Nc_Rd)
    u_N = np.abs(N) / N_Rd

    # --- strong-axis bending with LTB ---------------------------------------
    Mcr = (1.13 * np.pi ** 2 * E_STEEL * sec["Iy"] / Lcr ** 2 *
           np.sqrt(np.maximum(sec["Cw"] / sec["Iy"], 1e-9) +
                   G_STEEL * sec["J"] * Lcr ** 2 /
                   (np.pi ** 2 * E_STEEL * sec["Iy"])))
    lam_LT = np.sqrt(sec["Zx"] * FY / np.maximum(Mcr, 1e-9))
    chi_LT = np.where(lam_LT <= 0.4, 1.0, chi_buckling(lam_LT, 0.34))
    Mb_Rd = chi_LT * sec["Zx"] * FY / GAMMA_M1
    Mv_Rd = sec["Zy"] * FY / GAMMA_M0
    u_M = M3 / Mb_Rd + M2 / Mv_Rd

    # --- shear ----------------------------------------------------------------
    Vpl2 = sec["hw"] * sec["tw"] * FY / np.sqrt(3.0)     # strong (web) shear
    Vpl3 = 2.0 * sec["bf"] * sec["tf"] * FY / np.sqrt(3.0)  # weak (flange) shear
    u_V = np.maximum(V2 / Vpl2, V3 / Vpl3)

    util = np.maximum(u_N + u_M, u_V)
    parts = {"u_N": u_N, "u_M": u_M, "u_V": u_V,
             "chi": chi, "chi_LT": chi_LT, "Nc_Rd": Nc_Rd, "Mb_Rd": Mb_Rd}
    return util, parts


def envelope_over_lcs(lcs_utils):
    """Max utilization across load cases: dict lc_name -> (nm,) util array."""
    names = list(lcs_utils.keys())
    env = np.max(np.stack([lcs_utils[n] for n in names]), axis=0)
    arg = np.argmax(np.stack([lcs_utils[n] for n in names]), axis=0)
    return env, {n: names[a] for n, a in zip(range(len(arg)), arg)}


def drift_checks(d, model, drift_limit_h=300.0, top_limit_H=400.0):
    """Inter-story drift ratio and top-drift ratios against limits.

    Returns dict with per-story ratios and pass flags.
    """
    from model_builder import story_drift_ratios
    rx, ry = story_drift_ratios(d, model)
    H = model["stories"] * model["story_h"]
    ux_top = np.abs(d[0::6][model["node_level"] == model["stories"]]).max()
    uy_top = np.abs(d[1::6][model["node_level"] == model["stories"]]).max()
    return {
        "story_ratio_max": float(max(rx.max(), ry.max())),
        "story_ratio_limit": 1.0 / drift_limit_h,
        "story_ok": bool(max(rx.max(), ry.max()) <= 1.0 / drift_limit_h),
        "top_drift_max": float(max(ux_top, uy_top) / H),
        "top_drift_limit": 1.0 / top_limit_H,
        "top_ok": bool(max(ux_top, uy_top) / H <= 1.0 / top_limit_H),
    }
