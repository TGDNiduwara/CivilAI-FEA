"""CivilAI-FEA :: autonomous stress-test pipeline (Step 4).

Builds a 20-story 3D steel moment frame (>1000 members), applies extreme
wind + equivalent-static seismic + factored gravity loads, runs the sparse
FEA solver for every load case, checks every member against Eurocode 3,
verifies inter-story drift limits and global equilibrium, and reports
staged execution latency.

Run:  python run_simulation.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from compliance import UTIL_LIMIT, member_utilization, drift_checks
from fea_solver import Structure, solve_load_cases
from load_generator import load_combinations
from model_builder import (build_grid_frame, group_index_arrays,
                           materialize_design, member_stats)
from sections import BEAM_POOL, COLUMN_POOL

# --- baseline design (pool indices) -----------------------------------------
BASELINE = {
    # corner / edge / interior columns, low-mid-high tiers
    "col_corner_low": 5, "col_corner_mid": 4, "col_corner_high": 3,
    "col_edge_low": 6, "col_edge_mid": 5, "col_edge_high": 4,
    "col_interior_low": 7, "col_interior_mid": 6, "col_interior_high": 5,
    # perimeter / interior floor beams
    "beam_x_perim_low": 6, "beam_x_perim_mid": 4, "beam_x_perim_high": 3,
    "beam_x_int_low": 6, "beam_x_int_mid": 6, "beam_x_int_high": 4,
    "beam_y_perim_low": 6, "beam_y_perim_mid": 4, "beam_y_perim_high": 3,
    "beam_y_int_low": 6, "beam_y_int_mid": 6, "beam_y_int_high": 4,
    # central X-braces (tension-only), W14 column pool
    "brace_x_low": 5, "brace_x_mid": 5, "brace_x_high": 4,
    "brace_y_low": 5, "brace_y_mid": 5, "brace_y_high": 4,
}

# extreme-event demand calibration
LOAD_KW = dict(vb=40.0, cs=0.05, cs_cd=1.3)

LATERAL_LCS = ("WINDX", "WINDY", "EQX", "EQY")


def run(nx=5, ny=4, stories=20, verbose=True):
    t0 = time.perf_counter()
    model = build_grid_frame(nx=nx, ny=ny, stories=stories)
    t_build = time.perf_counter() - t0

    design = materialize_design(model, BASELINE)
    st = Structure(model["nodes"], model["mi"], model["mj"], model["supports"],
                   diaphragms=model["diaphragms"])
    st.set_sections(design["A"], design["Ix"], design["Iy"], design["J"])

    _, _, is_col_group = group_index_arrays(model)
    mgroup = design["mgroup"]
    is_col_m = is_col_group[mgroup]

    t1 = time.perf_counter()
    LC = load_combinations(model, design["A"], **LOAD_KW)
    t_loads = time.perf_counter() - t1
    diag = LC.pop("_diag")

    if verbose:
        n_members = st.nm
        print("=" * 78)
        print("CIVILAI-FEA :: 3D FRAME STRESS TEST")
        print("=" * 78)
        print(f"grid            : {nx}x{ny} bays x {stories} stories "
              f"({nx*model['bay_x']:.0f} x {ny*model['bay_y']:.0f} x "
              f"{stories*model['story_h']:.0f} m)")
        print(f"nodes / DOFs    : {st.nn} / {st.ndof}")
        print(f"members         : {n_members}  (>1000 required: "
              f"{'YES' if n_members >= 1000 else 'NO'})")
        cnt = member_stats(model)
        cols = sum(v for k, v in cnt.items() if k.startswith("col_"))
        braces = sum(v for k, v in cnt.items() if k.startswith("brace_"))
        beams = n_members - cols - braces
        print(f"                  columns {cols}, beams {beams}, braces {braces}")
        print(f"baseline steel  : {st.weight():.1f} kN "
              f"({st.weight()/77.0:.1f} m3, ~{st.weight()/9.81:.0f} t)")
        print(f"wind base shear : {diag['V_base_wind_x']:.0f} kN (X, vb=40 m/s)")
        print(f"seismic         : V = {diag['V_base_eq_x']:.0f} kN, "
              f"W = {diag['W_seismic_total']:.0f} kN (Cs=0.05)")
        print(f"bracing         : {'X-braces (BRB behavior) in central bays'
              if model['braced'] else 'none (bare moment frame)'}")

    results = {"stages": {"build_s": t_build, "loads_s": t_loads},
               "n_members": st.nm, "n_nodes": st.nn, "ndof": st.ndof}
    lcs_util, per_lc = {}, {}
    is_col_arr = is_col_m

    for name, F in LC.items():
        t2 = time.perf_counter()
        d = st.solve(F)
        t_solve = time.perf_counter() - t2

        t3 = time.perf_counter()
        fl = st.member_forces(d)
        env = Structure.force_envelope(fl)
        util, parts = member_utilization(
            env["N"], env["V2"], env["V3"], env["M2"], env["M3"],
            design, st.L, is_col_arr)
        t_post = time.perf_counter() - t3

        R = st.reactions(F, d)
        fz_applied = float(F[2::6].sum())
        fz_react = float(R[2::6].sum())
        imax = int(np.argmax(util))

        drift = drift_checks(d, model) if name in LATERAL_LCS else None
        lcs_util[name] = util
        per_lc[name] = {
            "solve_s": t_solve, "post_s": t_post,
            "max_util": float(util[imax]), "util_member": int(imax),
            "util_group": model["groups"][imax],
            "util_section": design["names"][imax],
            "sum_Fz_applied_kN": fz_applied, "sum_Fz_reaction_kN": fz_react,
            "Fz_equilibrium_err": abs(fz_applied + fz_react),
            "drift": drift,
        }
        results["stages"][f"solve_{name}_s"] = t_solve
        if verbose:
            print("-" * 78)
            print(f"LC {name:6s} | solve {t_solve*1e3:7.1f} ms | "
                  f"post {t_post*1e3:6.1f} ms")
            print(f"  max util {util[imax]:6.3f} @ member {imax} "
                  f"[{model['groups'][imax]}, {design['names'][imax]}, "
                  f"z={model['nodes'][model['mi'][imax],2]:.1f} m]"
                  f" {'!! OVER ' + str(UTIL_LIMIT) if util[imax] > UTIL_LIMIT else ''}")
            print(f"  sum Fz: applied {fz_applied:12.1f} kN | "
                  f"reactions {fz_react:12.1f} kN | "
                  f"equilibrium err {abs(fz_applied+fz_react):.2e} kN")
            if drift:
                print(f"  drift: story max {drift['story_ratio_max']*100:.3f}% "
                      f"(limit {drift['story_ratio_limit']*100:.3f}%) "
                      f"{'OK' if drift['story_ok'] else 'VIOLATED'} | "
                      f"top {drift['top_drift_max']*100:.3f}% "
                      f"(limit {drift['top_drift_limit']*100:.3f}%) "
                      f"{'OK' if drift['top_ok'] else 'VIOLATED'}")

    env_util = np.max(np.stack(list(lcs_util.values())), axis=0)
    imax = int(np.argmax(env_util))
    n_over = int(np.sum(env_util > UTIL_LIMIT))
    results["envelope"] = {
        "max_util": float(env_util[imax]),
        "n_members_over_90pct": n_over,
        "worst_member": {"index": imax, "group": model["groups"][imax],
                         "section": design["names"][imax],
                         "lc": max(lcs_util, key=lambda k: float(lcs_util[k][imax]))},
        "steel_weight_kN": st.weight(),
    }
    results["per_lc"] = per_lc
    t_total = time.perf_counter() - t0
    results["stages"]["total_s"] = t_total

    if verbose:
        print("=" * 78)
        print(f"ENVELOPE: max util {env_util[imax]:.3f} "
              f"({n_over} members above {UTIL_LIMIT:.2f}) at "
              f"[{model['groups'][imax]} / {design['names'][imax]}]")
        print(f"total wall time {t_total:.2f} s")
    return model, st, design, results


if __name__ == "__main__":
    import os
    _results = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(_results, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",
                    default=os.path.join(_results, "baseline_results.json"))
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--stories", type=int, default=20)
    args = ap.parse_args()
    _, _, _, results = run(nx=args.nx, ny=args.ny, stories=args.stories)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {args.json}")
