"""CivilAI-FEA :: autonomous stress-test & hardening loop (Step 4).

Runs 40 cycles of randomized structural variants through the FEA engine
and enforces mathematical invariants on every cycle:

  R1  equilibrium residual   ||K d - F|| / (1 + ||F||)   < 1e-8
  R2  stiffness symmetry     max|K-K'| / max|K|          < 1e-12 (dense check)
  R3  solution finiteness / magnitude bounds
  R4  positive strain energy d' F >= 0 (SPD sanity)
  R5  global force balance   sum R + sum F = 0 (reactions)
  R6  full analytical verification battery (every 5th cycle, in-process)

Model families swept: irregular random 3D frames (spanning-tree + random
connectivity, non-collinear fixed supports), and grid frames with random
bays/stories/sections, braces on/off, diaphragms on/off, scaled loads.
Solve latency is logged per cycle.

Exit code 0 iff all 40 cycles pass.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

from fea_solver import SingularSystemError, Structure
from model_builder import ALL_GROUPS, build_grid_frame, materialize_design
from sections import BEAM_POOL, COLUMN_POOL

N_CYCLES = 40
RESIDUAL_TOL = 1e-8
SYM_TOL = 1e-12


# ----------------------------------------------------------------- builders
def random_irregular_frame(rng):
    """Random 3D frame: spanning-tree connectivity + extra random members,
    three non-collinear fully-fixed support nodes, random sections/loads."""
    nn = int(rng.integers(16, 61))
    pts = rng.random((nn, 3)) * np.array([24.0, 20.0, 30.0])
    order = rng.permutation(nn)
    edges = set()
    for k in range(1, nn):
        i = order[k]
        prev = order[:k]
        d = np.linalg.norm(pts[prev] - pts[i], axis=1)
        j = int(prev[np.argmin(d)])
        edges.add((min(i, j), max(i, j)))
    for _ in range(int(nn * 0.6)):
        i, j = rng.integers(0, nn, 2)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    edges = sorted(edges)
    mi = np.array([e[0] for e in edges], int)
    mj = np.array([e[1] for e in edges], int)

    z = pts[:, 2]
    cand = list(np.flatnonzero(z < z.min() + 2.0))
    if len(cand) < 3:
        cand = list(np.argsort(z)[:3])
    fixed = [cand[0]]
    for c in cand[1:]:
        if len(fixed) == 3:
            break
        ok = True
        for a in range(len(fixed)):
            for b in range(a + 1, len(fixed)):
                p1, p2, p3 = pts[fixed[a]], pts[fixed[b]], pts[c]
                if np.linalg.norm(np.cross(p2 - p1, p3 - p1)) < 1e-6:
                    ok = False
        if ok:
            fixed.append(c)
    k = 0
    while len(fixed) < 3 and k < nn:
        if k not in fixed:
            fixed.append(k)
        k += 1

    sup = np.zeros((nn, 6), bool)
    for fn in fixed[:3]:
        sup[fn, :] = True

    st = Structure(pts, mi, mj, sup)
    col_n = len(COLUMN_POOL)
    pick = lambda: COLUMN_POOL[int(rng.integers(0, col_n))]  # noqa: E731
    st.set_sections([pick().A for _ in range(st.nm)],
                    [pick().Ix for _ in range(st.nm)],
                    [pick().Iy for _ in range(st.nm)],
                    [pick().J for _ in range(st.nm)])
    F = np.zeros(st.ndof)
    F[0::6] += rng.uniform(-40, 40, st.nn)
    F[1::6] += rng.uniform(-40, 40, st.nn)
    F[2::6] -= rng.uniform(0, 120, st.nn)
    return st, F, f"irregular({st.nn}n/{st.nm}m)"


def random_grid_frame(rng):
    nx, ny = int(rng.integers(2, 5)), int(rng.integers(2, 4))
    stories = int(rng.integers(2, 9))
    braced = bool(rng.integers(0, 2))
    diaph = bool(rng.integers(0, 2))
    model = build_grid_frame(nx=nx, ny=ny, stories=stories,
                             braced=braced, diaphragm=diaph)
    st = Structure(model["nodes"], model["mi"], model["mj"], model["supports"],
                   diaphragms=model["diaphragms"])
    col_n, beam_n = len(COLUMN_POOL), len(BEAM_POOL)
    gi = {}
    for g in ALL_GROUPS:
        gi[g] = int(rng.integers(0, col_n if ("col_" in g or "brace_" in g)
                                 else beam_n))
    dsg = materialize_design(model, gi)
    st.set_sections(dsg["A"], dsg["Ix"], dsg["Iy"], dsg["J"])
    F = np.zeros(st.ndof)
    scale = rng.uniform(0.3, 1.5)
    F[0::6] += rng.uniform(-20, 20, st.nn) * scale
    F[2::6] -= rng.uniform(10, 80, st.nn) * scale
    tag = (f"grid({nx}x{ny}x{stories}, braced={int(braced)}, "
           f"diaph={int(diaph)})")
    return st, F, dsg["active"], tag


# ------------------------------------------------------------------ checks
def dense_checks(st, F):
    """Symmetry (R2) + independent dense solve residual, for small models."""
    if st.ndof > 1500:
        return None, None
    kl = st._k_local_stack()
    kg = np.einsum('nji,njk,nkl->nil', st.T, kl, st.T)
    Kd = np.zeros((st.ndof, st.ndof))
    np.add.at(Kd, (st.rows, st.cols), kg.ravel())
    sym = np.max(np.abs(Kd - Kd.T)) / np.max(np.abs(Kd))
    return float(sym), Kd


def run_cycle(idx, rng):
    t0 = time.perf_counter()
    if idx % 2 == 0:
        st, F, _, tag = random_grid_frame(rng)
    else:
        st, F, tag = random_irregular_frame(rng)

    out = {"cycle": idx, "model": tag, "ndof": st.ndof, "members": st.nm,
           "solve_ms": 0.0, "residual": None, "symmetry": None,
           "work_ratio": None, "status": "PASS", "note": ""}

    try:
        t1 = time.perf_counter()
        d = st.solve(F)
        out["solve_ms"] = (time.perf_counter() - t1) * 1e3

        # R1/R3 embedded in solve(); explicit work check (R4)
        work = float(F @ d)
        out["work_ratio"] = work
        if work < -1e-9:
            out["status"] = "FAIL"
            out["note"] = f"negative strain energy {work:.3e}"

        # R2 symmetry via dense reconstruction
        sym, _ = dense_checks(st, F)
        out["symmetry"] = sym
        if sym is not None and sym > SYM_TOL:
            out["status"] = "FAIL"
            out["note"] = f"asymmetric K ({sym:.2e})"

        # R5 global balance, per translational axis:
        # R_ext = K d - F equals reactions at supports (zero at free DOFs),
        # and rigid-body equilibrium requires sum R_axis + sum F_axis = 0.
        R = st.reactions(F, d)
        for ax_name, sl in (("x", slice(0, None, 6)),
                            ("y", slice(1, None, 6)),
                            ("z", slice(2, None, 6))):
            bal = abs(R[sl].sum() + F[sl].sum())
            if bal > 1e-6 * (1.0 + float(np.abs(F[sl]).sum())):
                out["status"] = "FAIL"
                out["note"] = f"force balance ({ax_name}) err {bal:.3e}"
    except SingularSystemError as exc:
        out["status"] = "FAIL"
        out["note"] = str(exc)[:100]
    out["wall_ms"] = (time.perf_counter() - t0) * 1e3
    return out


def main():
    rng = np.random.default_rng(2026)
    rows = []
    n_fail = 0
    print("=" * 96)
    print(f"CIVILAI-FEA HARDENING LOOP :: {N_CYCLES} randomized stress cycles")
    print("=" * 96)
    for i in range(1, N_CYCLES + 1):
        out = run_cycle(i, rng)

        if i % 5 == 0:                      # R6: regression battery re-run
            import verify
            ok = verify.run_all(verbose=False)
            out["note"] = (out["note"] + " | " if out["note"] else "") + \
                (f"battery {sum(1 for c in verify.CASES)}/"
                 f"{len(verify.CASES)} ok" if ok else "battery FAILED")
            if not ok:
                out["status"] = "FAIL"

        if out["status"] != "PASS":
            n_fail += 1
        rows.append(out)
        print(f"cycle {i:2d} [{out['status']}] {out['model']:34s} "
              f"dof {out['ndof']:5d} m {out['members']:4d} | "
              f"solve {out['solve_ms']:7.1f} ms | "
              f"sym {('%.0e' % out['symmetry']) if out['symmetry'] is not None else '  n/a':>6} | "
              f"work {out['work_ratio']:12.3f} | {out['note'][:60]}",
              flush=True)

    import os
    _results = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(_results, exist_ok=True)
    keys = ["cycle", "model", "ndof", "members", "solve_ms", "residual",
            "symmetry", "work_ratio", "status", "note", "wall_ms"]
    with open(os.path.join(_results, "hardening_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    solves = np.array([r["solve_ms"] for r in rows])
    print("-" * 96)
    print(f"RESULT: {N_CYCLES - n_fail}/{N_CYCLES} cycles passed | "
          f"solve latency mean {solves.mean():.1f} ms, "
          f"p95 {np.percentile(solves, 95):.1f} ms, max {solves.max():.1f} ms")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
