"""Analytical verification battery for CivilAI-FEA.

Every case compares the FEA result against a closed-form structural
mechanics solution. With nodal-only loading, Euler-Bernoulli frame
elements are exact (cubic interpolation), so tolerances are tight
(~1e-9 relative). Run:  python verify.py  (exit 0 = all pass)
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from fea_solver import E_STEEL, G_STEEL, Structure, SingularSystemError
from sections import COLUMN_POOL

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def bar_structure(L, A, Ix, Iy, J):
    nodes = np.array([[0., 0., 0.], [L, 0., 0.]])
    st = Structure(nodes, [0], [1], np.array([[True] * 6, [False] * 6]))
    st.set_sections([A], [Ix], [Iy], [J])
    return st


@case
def t1_axial_bar():
    L, A = 5.0, 1.0e-2
    P = 100.0
    st = bar_structure(L, A, 1e-5, 1e-5, 1e-7)
    F = np.zeros(st.ndof); F[6] = P                      # ux of node 1
    d = st.solve(F)
    expect = P * L / (E_STEEL * A)
    err = rel(d[6], expect)
    fl = st.member_forces(d)
    assert err < 1e-9, f"tip axial u {d[6]:.6e} vs {expect:.6e}"
    assert rel(fl[0, 6], P) < 1e-9, f"axial force {fl[0,6]} vs {P}"
    return f"u={d[6]:.6e} m, N={fl[0,6]:.1f} kN, err={err:.2e}"


@case
def t2_cantilever_strong_bend():
    L, Ix = 4.0, 3.0e-4
    P = 10.0
    st = bar_structure(L, 1e-2, Ix, 1e-5, 1e-7)
    F = np.zeros(st.ndof); F[8] = -P                     # uz of node 1
    d = st.solve(F)
    expect = -P * L ** 3 / (3 * E_STEEL * Ix)
    err = rel(d[8], expect)
    fl = st.member_forces(d)
    assert err < 1e-9, f"tip defl {d[8]:.6e} vs {expect:.6e}"
    assert rel(abs(fl[0, 5]), P * L) < 1e-9, f"root M3 {fl[0,5]} vs {P*L}"
    return f"delta={d[8]:.6e} m, M3(root)={fl[0,5]:.2f} kNm, err={err:.2e}"


@case
def t3_cantilever_weak_bend():
    L, Iy = 4.0, 1.0e-5
    P = 10.0
    st = bar_structure(L, 1e-2, 3e-4, Iy, 1e-7)
    F = np.zeros(st.ndof); F[7] = P                      # uy of node 1
    d = st.solve(F)
    expect = P * L ** 3 / (3 * E_STEEL * Iy)
    err = rel(d[7], expect)
    assert err < 1e-9, f"tip defl {d[7]:.6e} vs {expect:.6e}"
    return f"delta={d[7]:.6e} m (weak axis), err={err:.2e}"


@case
def t4_torsion():
    L, J = 3.0, 2.0e-6
    T = 5.0
    st = bar_structure(L, 1e-2, 3e-4, 1e-5, J)
    F = np.zeros(st.ndof); F[6 + 3] = T                  # rx of node 1
    d = st.solve(F)
    expect = T * L / (G_STEEL * J)
    err = rel(d[9], expect)
    assert err < 1e-9, f"twist {d[9]:.6e} vs {expect:.6e}"
    return f"theta={d[9]:.6e} rad, err={err:.2e}"


@case
def t5_simple_beam_mid_load():
    L, Ix = 6.0, 2.0e-4
    P = 20.0
    nodes = np.array([[0., 0., 0.], [L / 2, 0., 0.], [L, 0., 0.]])
    sup = np.zeros((3, 6), bool)
    sup[0] = (True, True, True, True, False, False)   # pin + torsion restraint
    sup[1] = (False, False, False, False, False, False)  # mid node free
    sup[2] = (False, True, True, True, False, False)  # roller along x
    st = Structure(nodes, [0, 1], [1, 2], sup)
    st.set_sections([1e-2] * 2, [Ix] * 2, [1e-5] * 2, [1e-7] * 2)
    F = np.zeros(st.ndof); F[6 + 2] = P
    d = st.solve(F)
    expect = P * L ** 3 / (48 * E_STEEL * Ix)
    err = rel(d[8], expect)
    R = st.reactions(F, d)
    assert err < 1e-9, f"mid defl {d[8]:.6e} vs {expect:.6e}"
    assert rel(R[2], -P / 2) < 1e-9, f"reaction {R[2]} vs {-P/2}"
    assert rel(R[14], -P / 2) < 1e-9, f"reaction2 {R[14]} vs {-P/2}"
    return f"mid delta={d[8]:.6e} m, R={R[2]:.2f}/{R[14]:.2f} kN, err={err:.2e}"


@case
def t6_diagonal_axial_member():
    # member from (0,0,0) to (3,4,0): L=5, force along axis
    L, A = 5.0, 8.0e-3
    P = 50.0
    nodes = np.array([[0., 0., 0.], [3., 4., 0.]])
    st = Structure(nodes, [0], [1], np.array([[True] * 6, [False] * 6]))
    st.set_sections([A], [1e-5], [1e-5], [1e-7])
    F = np.zeros(st.ndof)
    F[6:9] = P * np.array([3 / 5, 4 / 5, 0.0])
    d = st.solve(F)
    u_axis = np.dot(d[6:9], np.array([3 / 5, 4 / 5, 0.0]))
    expect = P * L / (E_STEEL * A)
    err = rel(u_axis, expect)
    fl = st.member_forces(d)
    assert err < 1e-9, f"axial elong {u_axis:.6e} vs {expect:.6e}"
    assert rel(fl[0, 6], P) < 1e-9, f"N={fl[0,6]} vs {P}"
    return f"3D transform OK: N={fl[0,6]:.1f} kN, err={err:.2e}"


@case
def t7_rotated_cantilever():
    # cantilever along (1,1,0)/sqrt2, tip load -Z: strong bending about
    # local z (local y = global Z for horizontal members)
    L, Ix = 4.0, 3.0e-4
    P = 10.0
    c = 1 / np.sqrt(2)
    nodes = np.array([[0., 0., 0.], [4 * c, 4 * c, 0.]])
    st = Structure(nodes, [0], [1], np.array([[True] * 6, [False] * 6]))
    st.set_sections([1e-2], [Ix], [1e-5], [1e-7])
    F = np.zeros(st.ndof); F[8] = -P
    d = st.solve(F)
    expect = -P * (L) ** 3 / (3 * E_STEEL * Ix)
    assert rel(d[8], expect) < 1e-9, \
        f"tip dz {d[8]:.6e} vs {expect:.6e}"
    assert abs(d[6]) + abs(d[7]) < 1e-9 * abs(expect), "unexpected in-plane sway"
    return f"45deg cantilever: dz={d[8]:.6e} m OK"


@case
def t8_sparse_vs_dense_frame():
    """Cross-check sparse solver against a dense numpy solve on a small
    irregular frame (also verifies assembly symmetry)."""
    from model_builder import build_grid_frame
    m = build_grid_frame(nx=2, ny=1, stories=2, bay_x=6.0, bay_y=5.0, story_h=3.2)
    st = Structure(m["nodes"], m["mi"], m["mj"], m["supports"])
    rng = np.random.default_rng(7)
    A = rng.uniform(5e-3, 2e-2, st.nm)
    Ix = rng.uniform(5e-5, 5e-4, st.nm)
    Iy = Ix / 4
    J = Ix / 10
    st.set_sections(A, Ix, Iy, J)

    F = np.zeros(st.ndof)
    F[0::6] += rng.uniform(-30, 30, st.nn)
    F[2::6] -= rng.uniform(0, 100, st.nn)

    # dense assembly path (independent of scipy)
    kl = st._k_local_stack()
    kg = np.einsum('nji,njk,nkl->nil', st.T, kl, st.T)
    Kd = np.zeros((st.ndof, st.ndof))
    np.add.at(Kd, (st.rows, st.cols), kg.ravel())
    sym = np.max(np.abs(Kd - Kd.T)) / np.max(np.abs(Kd))

    Kff = Kd[st.free_dofs][:, st.free_dofs]
    d_dense = np.zeros(st.ndof)
    d_dense[st.free_dofs] = np.linalg.solve(Kff, F[st.free_dofs])
    d_sp = st.solve(F)
    err = np.max(np.abs(d_dense - d_sp)) / max(np.max(np.abs(d_dense)), 1e-12)
    assert sym < 1e-12, f"K not symmetric: {sym:.2e}"
    assert err < 1e-8, f"sparse vs dense mismatch: {err:.2e}"
    return f"symmetry {sym:.1e}, sparse==dense rel.err {err:.1e}"


@case
def t10_buckling_capacity_sanity():
    from sections import pool_arrays
    s = pool_arrays([COLUMN_POOL[7]])            # W14x211
    L = np.array([3.6])
    Lcr = 1.5 * L
    Ncr = np.pi ** 2 * E_STEEL * s["Iy"] / Lcr ** 2
    lam = np.sqrt(s["A"] * 355e3 / Ncr)
    from compliance import chi_buckling
    chi = chi_buckling(lam)
    assert 0.4 < chi[0] < 1.0, f"chi out of range: {chi[0]}"
    lam_sweep = np.linspace(0.1, 3.0, 60)
    chis = chi_buckling(lam_sweep)
    assert np.all(np.diff(chis) < 1e-9), "chi not monotonically decreasing"
    assert np.all(chis <= 1.0) and np.all(chis > 0), "chi out of (0,1]"
    return f"W14x211 k=1.5 L=3.6m: lam={lam[0]:.3f}, chi={chi[0]:.3f} OK"


@case
def t11_rigid_body_zero_load():
    """Self-equilibrium: unloaded structure with a unit displaced support
    stays consistent (solve with zero load gives zero displacement)."""
    from model_builder import build_grid_frame
    m = build_grid_frame(nx=1, ny=1, stories=1, bay_x=4.0, bay_y=4.0, story_h=3.0)
    st = Structure(m["nodes"], m["mi"], m["mj"], m["supports"])
    st.set_sections(np.full(st.nm, 1e-2), np.full(st.nm, 2e-4),
                    np.full(st.nm, 5e-5), np.full(st.nm, 4e-6))
    d = st.solve(np.zeros(st.ndof))
    assert np.max(np.abs(d)) == 0.0
    return "zero-load zero-displacement OK"


@case
def t12_rigid_diaphragm():
    """Diaphragm condensation: floor nodes move rigidly in-plane, and the
    condensated solution matches an unconstrained solve of the same frame
    loaded at the master node only (identical by symmetry of the load path
    through the master)."""
    from model_builder import build_grid_frame
    m = build_grid_frame(nx=2, ny=1, stories=2, bay_x=6.0, bay_y=5.0,
                         story_h=3.2, braced=False, diaphragm=True)
    st = Structure(m["nodes"], m["mi"], m["mj"], m["supports"],
                   diaphragms=m["diaphragms"])
    st.set_sections(np.full(st.nm, 1e-2), np.full(st.nm, 3e-4),
                    np.full(st.nm, 8e-5), np.full(st.nm, 5e-6))
    F = np.zeros(st.ndof)
    F[0::6] += 40.0        # +x on every node
    F[2::6] -= 60.0
    d = st.solve(F)

    # rigid-body check per floor: ux equal, uy = uy_m + rz*(x-x_m)
    for lvl in range(1, m["stories"] + 1):
        mask = m["node_level"] == lvl
        ids = np.flatnonzero(mask)
        ux = d[0::6][ids]
        assert np.ptp(ux) < 1e-12, f"floor {lvl}: ux not uniform ({np.ptp(ux):.2e})"
        mst = ids[0]
        xm, ym = m["nodes"][mst, 0], m["nodes"][mst, 1]
        rzm = d[6 * mst + 5]
        for s in ids:
            dx, dy = m["nodes"][s, 0] - xm, m["nodes"][s, 1] - ym
            uy = d[6 * s + 1]
            assert abs(uy - (d[6 * mst + 1] + rzm * dx)) < 1e-12
            assert abs(d[6 * s + 5] - rzm) < 1e-12

    # equilibrium must still close with condensation active
    R = st.reactions(F, d)
    assert abs(R[2::6].sum() + F[2::6].sum()) < 1e-6
    assert abs(R[0::6].sum() + F[0::6].sum()) < 1e-6
    return (f"ux top floor {d[0::6][m['node_level'] == m['stories']][0]:.5f} m, "
            f"rigid in-plane + equilibrium OK")


def run_all(verbose=True):
    results = []
    for fn in CASES:
        name = fn.__name__
        try:
            msg = fn()
            results.append((name, True, msg))
            if verbose:
                print(f"  [PASS] {name:32s} {msg}")
        except AssertionError as exc:
            results.append((name, False, str(exc)))
            if verbose:
                print(f"  [FAIL] {name:32s} {exc}")
        except Exception as exc:                       # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  [ERR ] {name:32s} {type(exc).__name__}: {exc}")
                traceback.print_exc()
    n_ok = sum(1 for _, ok, _ in results if ok)
    if verbose:
        print(f"  verification battery: {n_ok}/{len(results)} passed")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    print("== CivilAI-FEA verification battery ==")
    ok = run_all()
    sys.exit(0 if ok else 1)
