"""CivilAI-FEA :: 3D frame finite element solver (Direct Stiffness Method).

Linear-elastic, small-displacement, Euler-Bernoulli frame elements with
6 degrees of freedom per node [ux, uy, uz, rx, ry, rz].

Units: kN, m (1 MPa = 1000 kN/m2).

Element local axes (right-handed, local -> global rotation R = [ex ey ez]):
  local 1 (x): member axis, node i -> node j
  local 2 (y): projection of global +Z perpendicular to x  ("up" direction
               for horizontal members; falls back to +X, then +Y for
               vertical members)
  local 3 (z): x cross y

Internal force sign convention (SAP-style output naming):
  N   axial (tension +)          T   torsion about local 1
  V2  shear along local 2        V3  shear along local 3
  M2  moment about local 2  (weak-axis bending for standard orientations)
  M3  moment about local 3  (strong-axis bending: vertical loads on
       horizontal members bend them about local 3)
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve

E_STEEL = 2.1e8      # Young's modulus, kN/m2 (210 GPa)
NU_STEEL = 0.3
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))   # 8.0769e7 kN/m2
RHO_STEEL_W = 77.0   # unit weight kN/m3 (7850 kg/m3)


class SingularSystemError(RuntimeError):
    """Raised when the global stiffness matrix is singular or the solution
    is not finite (floating node / mechanism / numerical breakdown)."""


class Structure:
    """3D frame structure with fixed topology.

    Geometry, connectivity, DOF maps and coordinate transformations are
    built once. Section properties can be swapped per member (used by the
    genetic optimizer) and members can be deactivated (topology removal)
    without rebuilding the bookkeeping.
    """

    def __init__(self, nodes, mi, mj, supports_mask, diaphragms=None):
        nodes = np.asarray(nodes, dtype=float)
        mi = np.asarray(mi, dtype=int)
        mj = np.asarray(mj, dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must be (n,3)")
        if mi.shape != mj.shape or np.any(mi == mj):
            raise ValueError("invalid member connectivity")
        if np.any(np.linalg.norm(nodes[mj] - nodes[mi], axis=1) <= 0.0):
            raise ValueError("zero-length member")

        self.nodes = nodes
        self.mi, self.mj = mi, mj
        self.nn, self.nm = len(nodes), len(mi)
        self.ndof = 6 * self.nn

        # --- local coordinate systems -------------------------------------
        d = nodes[mj] - nodes[mi]
        L = np.linalg.norm(d, axis=1)
        self.L = L
        ex = d / L[:, None]

        vref = np.tile(np.array([0.0, 0.0, 1.0]), (self.nm, 1))
        vertical = np.abs(ex[:, 2]) > 0.999
        vref[vertical] = (1.0, 0.0, 0.0)
        also_x = vertical & (np.abs(ex[:, 0]) > 0.999)
        vref[also_x] = (0.0, 1.0, 0.0)

        ey = vref - (vref * ex).sum(axis=1)[:, None] * ex
        ey /= np.linalg.norm(ey, axis=1)[:, None]
        ez = np.cross(ex, ey)
        self.R = np.stack((ex, ey, ez), axis=1)          # (nm,3,3), columns = local axes

        T = np.zeros((self.nm, 12, 12))
        for b in range(4):
            T[:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = self.R
        self.T = T                                        # u_local = T @ u_global

        # --- DOF bookkeeping ------------------------------------------------
        dofs = np.arange(self.ndof).reshape(self.nn, 6)
        self.dofmap = np.concatenate((dofs[mi], dofs[mj]), axis=1)   # (nm,12)
        rows = np.repeat(self.dofmap[:, :, None], 12, axis=2)
        cols = np.repeat(self.dofmap[:, None, :], 12, axis=1)
        self.rows, self.cols = rows.ravel(), cols.ravel()

        supports_mask = np.asarray(supports_mask, dtype=bool)
        if supports_mask.shape != (self.nn, 6):
            raise ValueError("supports_mask must be (n,6) bool")
        self.supports = supports_mask
        self.free_dofs = np.flatnonzero(~supports_mask.ravel())
        if len(self.free_dofs) == 0:
            raise ValueError("structure has no free DOFs")

        # --- rigid diaphragm condensation ---------------------------------
        # diaphragms: list of (master_node, [slave_nodes]); each ties the
        # in-plane DOFs (ux, uy, rz) of every slave to the master node of
        # that floor:  ux_s = ux_m - rz_m (y_s-y_m)
        #             uy_s = uy_m + rz_m (x_s-x_m),  rz_s = rz_m
        self.T_dg = None
        if diaphragms:
            self._build_diaphragms(diaphragms)

        # placeholder section properties (set via set_sections)
        self.set_sections(
            A=np.full(self.nm, 1e-2), Ix=np.full(self.nm, 1e-5),
            Iy=np.full(self.nm, 1e-5), J=np.full(self.nm, 1e-7))

    # ------------------------------------------------------------------ props
    def _build_diaphragms(self, diaphragms):
        """Build the kinematic transformation  d = T_dg @ d_reduced.

        Reduced DOFs = all physical DOFs except the in-plane DOFs of slave
        nodes (those are determined by their floor master node).
        """
        n = self.ndof
        slave_inplane = set()
        cols = {d: [(d, 1.0)] for d in range(n)}      # phys dof -> [(phys, coef)]
        # slave-keyed term table: d[p] = sum coef * d_reduced[col(master)]
        self._dof_terms = {d: [(d, 1.0)] for d in range(n)}
        for master, slaves in diaphragms:
            m6 = 6 * int(master)
            xm, ym = self.nodes[master, 0], self.nodes[master, 1]
            for s in slaves:
                s6 = 6 * int(s)
                dx, dy = self.nodes[s, 0] - xm, self.nodes[s, 1] - ym
                for off in (0, 1, 5):
                    slave_inplane.add(s6 + off)
                    cols[s6 + off] = []               # determined, not retained
                # rigid-body relations, expressed as entries of the MASTER's
                # reduced columns: d[phys] += coef * d_reduced[column]
                cols[m6 + 0].append((s6 + 0, 1.0))            # ux_s = ux_m
                cols[m6 + 1].append((s6 + 1, 1.0))            # uy_s = uy_m
                cols[m6 + 5].append((s6 + 5, 1.0))            # rz_s = rz_m
                cols[m6 + 5].append((s6 + 0, -dy))            # ux_s -= rz_m dy
                cols[m6 + 5].append((s6 + 1, dx))             # uy_s += rz_m dx
                # slave-keyed equivalent (for the reduced scatter pattern)
                self._dof_terms[s6 + 0] = [(m6 + 0, 1.0), (m6 + 5, -dy)]
                self._dof_terms[s6 + 1] = [(m6 + 1, 1.0), (m6 + 5, dx)]
                self._dof_terms[s6 + 5] = [(m6 + 5, 1.0)]
        rows, cidx, vals = [], [], []
        self.reduced_phys = []
        for r, d in enumerate(d for d in range(n) if d not in slave_inplane):
            self.reduced_phys.append(d)
            for phys, coef in cols[d]:
                rows.append(phys); cidx.append(r); vals.append(coef)
        self.T_dg = coo_matrix((vals, (rows, cidx)), shape=(n, len(self.reduced_phys)),
                               dtype=float).tocsr()
        rp = np.asarray(self.reduced_phys)
        # positions (not physical ids) of reduced DOFs that are free
        self.free_reduced = np.flatnonzero(np.isin(rp, self.free_dofs))
        self._prepare_reduced_scatter(rp)

    def _prepare_reduced_scatter(self, rp):
        """Precompute the direct reduced-space assembly pattern.

        Each physical DOF maps to 1..3 reduced columns (identity for
        retained DOFs, the master-node relation for diaphragm slaves).
        Every member entry K[i,j] expands to terms_i x terms_j reduced
        entries; the pattern/coefficients are topology-only and computed
        once, so per solve only the value array is rebuilt.
        """
        rmap = np.full(self.ndof, -1, dtype=np.int64)
        rmap[rp] = np.arange(len(rp))

        masters_map = np.tile(np.arange(self.ndof, dtype=np.int64)[:, None], (1, 3))
        coef_map = np.zeros((self.ndof, 3))
        coef_map[:, 0] = 1.0
        for phys, terms in self._dof_terms.items():
            masters_map[phys, :] = 0
            coef_map[phys, :] = 0.0
            for a, (q_dof, c) in enumerate(terms):
                masters_map[phys, a] = q_dof
                coef_map[phys, a] = c

        q3 = rmap[masters_map[self.dofmap]]             # (nm,12,3) reduced positions
        c3 = coef_map[self.dofmap]                      # (nm,12,3)
        rowsR = np.broadcast_to(q3[:, :, None, :, None], (self.nm, 12, 12, 3, 3))
        colsR = np.broadcast_to(q3[:, None, :, None, :], (self.nm, 12, 12, 3, 3))
        coefR = c3[:, :, None, :, None] * c3[:, None, :, None, :]
        mask = (rowsR >= 0) & (colsR >= 0)
        self._r_rows = np.ascontiguousarray(rowsR[mask])
        self._r_cols = np.ascontiguousarray(colsR[mask])
        self._r_mask = np.ascontiguousarray(mask)
        self._r_coef = np.ascontiguousarray(coefR)

    def _assemble_reduced(self, active=None):
        """Assemble K directly in the reduced (condensed) space."""
        kl = self._k_local_stack()
        kg = np.einsum('nji,njk,nkl->nil', self.T, kl, self.T)
        if active is not None and not bool(np.all(active)):
            kg = kg * np.asarray(active, dtype=float)[:, None, None]
        vals = (kg[:, :, :, None, None] * self._r_coef).ravel()[self._r_mask.ravel()]
        Kr = coo_matrix((vals, (self._r_rows, self._r_cols)),
                        shape=(len(self.reduced_phys),) * 2, dtype=float).tocsr()
        return Kr

    def factor(self, active=None):
        """Numeric factorization of the free-block of K (reduced space when
        diaphragms are active). Returns a FactoredSystem: one factorization,
        any number of right-hand sides -- used when several load cases share
        the same stiffness (linear elastic analysis).
        """
        if self.T_dg is not None:
            Kr = self._assemble_reduced(active)
            free = self.free_reduced
            reduced = True
        else:
            Kr = self.assemble(active)
            free = self.free_dofs
            reduced = False
        Kff = Kr[free][:, free].tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                lu = splu(Kff)
            except (RuntimeError, UserWarning, ValueError) as exc:
                raise SingularSystemError(
                    f"singular/ill-conditioned K: {exc}") from exc
        return FactoredSystem(self, lu, Kff, free, reduced)

    def solve(self, F, active=None, rtol_check=1e-8, max_disp=1e3):
        """Solve [K]{D} = {F}. Returns the full displacement vector (ndof,).
        Raises SingularSystemError on singular/ill-conditioned systems."""
        return self.factor(active).solve(F, rtol_check=rtol_check,
                                         max_disp=max_disp)

    # ------------------------------------------------------------------ props
    def set_sections(self, A, Ix, Iy, J):
        """Per-member section properties (arrays of length nm)."""
        A, Ix, Iy, J = (np.asarray(v, dtype=float) for v in (A, Ix, Iy, J))
        if any(v.shape != (self.nm,) for v in (A, Ix, Iy, J)):
            raise ValueError("section property arrays must have length nm")
        self.A, self.Ix, self.Iy, self.J = A, Ix, Iy, J
        self.EA = E_STEEL * A
        self.GJ = G_STEEL * J
        self.EI_major = E_STEEL * Ix   # bending in local x-y plane (about local z)
        self.EI_minor = E_STEEL * Iy   # bending in local x-z plane (about local y)

    # ------------------------------------------------------------- element K
    def _k_local_stack(self):
        """Vectorized 12x12 local stiffness matrices, shape (nm,12,12).

        DOF order per end: [Fx Fy Fz Mx My Mz]; end i = 0..5, end j = 6..11.
        """
        L, L2 = self.L, self.L ** 2
        k = np.zeros((self.nm, 12, 12))

        ia = self.EA / L                       # axial
        k[:, 0, 0] = k[:, 6, 6] = ia
        k[:, 0, 6] = k[:, 6, 0] = -ia

        it = self.GJ / L                       # torsion
        k[:, 3, 3] = k[:, 9, 9] = it
        k[:, 3, 9] = k[:, 9, 3] = -it

        # bending in x-y plane (v=1, rz=5 | v=7, rz=11), EI = EI_major
        c = self.EI_major / L ** 3
        for a_, b_, v in ((1, 1, 12 * c), (1, 5, 6 * L * c), (1, 7, -12 * c),
                          (1, 11, 6 * L * c), (5, 5, 4 * L2 * c),
                          (5, 7, -6 * L * c), (5, 11, 2 * L2 * c),
                          (7, 7, 12 * c), (7, 11, -6 * L * c),
                          (11, 11, 4 * L2 * c)):
            k[:, a_, b_] += v
            if a_ != b_:
                k[:, b_, a_] += v

        # bending in x-z plane (w=2, ry=4 | w=8, ry=10), EI = EI_minor
        c = self.EI_minor / L ** 3
        for a_, b_, v in ((2, 2, 12 * c), (2, 4, -6 * L * c), (2, 8, -12 * c),
                          (2, 10, -6 * L * c), (4, 4, 4 * L2 * c),
                          (4, 8, 6 * L * c), (4, 10, 2 * L2 * c),
                          (8, 8, 12 * c), (8, 10, 6 * L * c),
                          (10, 10, 4 * L2 * c)):
            k[:, a_, b_] += v
            if a_ != b_:
                k[:, b_, a_] += v
        return k

    def assemble(self, active=None):
        """Global stiffness matrix (scipy.sparse.csr) over all DOFs.

        Members flagged inactive in `active` contribute nothing (topology
        removal); their DOFs remain, held by the remaining structure.
        """
        kl = self._k_local_stack()
        kg = np.einsum('nji,njk,nkl->nil', self.T, kl, self.T)   # T' K T
        vals = kg.ravel()
        if active is None or bool(np.all(active)):
            rows, cols = self.rows, self.cols
        else:
            keep = np.repeat(np.asarray(active, dtype=bool), 144)
            rows, cols, vals = self.rows[keep], self.cols[keep], vals[keep]
        K = coo_matrix((vals, (rows, cols)),
                       shape=(self.ndof, self.ndof), dtype=float).tocsr()
        return K

    # ------------------------------------------------------------------ post
    def member_forces(self, d=None, active=None):
        """Local end forces f = k_local T u_global, shape (nm,12).

        Layout: [end i: Fx Fy Fz Mx My Mz | end j: Fx Fy Fz Mx My Mz].
        N (tension+) is f[:,6]; envelope extraction is done by
        `force_envelope`.
        """
        if d is None:
            d = self.d
        ug = d[self.dofmap]                                    # (nm,12)
        ul = np.einsum('nij,nj->ni', self.T, ug)
        fl = np.einsum('nij,nj->ni', self._k_local_stack(), ul)
        return fl

    @staticmethod
    def force_envelope(fl):
        """Per-member design force components from local end forces."""
        return {
            "N": fl[:, 6],                       # axial at end j, tension positive
            "V2": np.maximum(np.abs(fl[:, 1]), np.abs(fl[:, 7])),
            "V3": np.maximum(np.abs(fl[:, 2]), np.abs(fl[:, 8])),
            "T": np.maximum(np.abs(fl[:, 3]), np.abs(fl[:, 9])),
            "M2": np.maximum(np.abs(fl[:, 4]), np.abs(fl[:, 10])),
            "M3": np.maximum(np.abs(fl[:, 5]), np.abs(fl[:, 11])),
        }

    def reactions(self, F, d=None):
        """Support reactions (external), (ndof,); zero on free DOFs."""
        if d is None:
            d = self.d
        if self.K is None:
            self.K = self.assemble(self._last_active)
        return self.K @ d - np.asarray(F, dtype=float)

    def weight(self, active=None):
        """Self-weight of all (active) members in kN."""
        mask = np.ones(self.nm, bool) if active is None else np.asarray(active, bool)
        return float(np.sum(self.A[mask] * RHO_STEEL_W * self.L[mask]))


class FactoredSystem:
    """Factorized free-block of K: one numeric factorization, N right-hand
    sides (all load cases of a linear-elastic design share the stiffness)."""

    def __init__(self, st, lu, Kff, free, reduced, active=None):
        self.st, self.lu, self.Kff, self.free = st, lu, Kff, free
        self.reduced = reduced
        self.active = active

    def solve(self, F, rtol_check=1e-8, max_disp=1e3):
        st = self.st
        F = np.asarray(F, dtype=float)
        if F.shape != (st.ndof,):
            raise ValueError("F must have shape (ndof,)")
        Ff = (st.T_dg.T @ F)[self.free] if self.reduced else F[self.free]
        try:
            dff = self.lu.solve(Ff)
        except (RuntimeError, ValueError) as exc:
            raise SingularSystemError(f"factorization failed: {exc}") from exc
        if not np.all(np.isfinite(dff)):
            raise SingularSystemError("non-finite displacements")
        if np.max(np.abs(dff)) > max_disp:
            raise SingularSystemError(
                f"displacements blew up (max |d| = {np.max(np.abs(dff)):.3e} m) "
                "- mechanism or inadequate support")
        if np.linalg.norm(self.Kff @ dff - Ff) > rtol_check * (1.0 + np.linalg.norm(Ff)):
            raise SingularSystemError("equilibrium residual too large")
        if self.reduced:
            full = np.zeros(st.T_dg.shape[1])
            full[self.free] = dff
            d = st.T_dg @ full
        else:
            d = np.zeros(st.ndof)
            d[self.free] = dff
        st.d = d
        st.K = None                       # full matrix rebuilt lazily
        st._last_active = self.active
        return d


def solve_load_cases(st, F_map, tension_only=None, base_active=None,
                     max_iter=4, comp_tol=1e-3):
    """Solve a set of load cases with optional tension-only members.

    Slender X-braces buckle in compression and shed load; standard practice
    models them as tension-only. For each load case the structure is solved
    iteratively: braces that end up compressed beyond `comp_tol` are
    deactivated and the case re-solved until the active set is stable.

    st            : Structure
    F_map         : {name: F vector}
    tension_only  : (nm,) bool mask of tension-only members (braces)
    base_active   : (nm,) bool design activity mask (GA removals)

    Returns {name: (d, fl, active)} with local end forces zeroed for
    inactive members.
    """
    out = {}
    for name, F in F_map.items():
        active = np.ones(st.nm, bool) if base_active is None else base_active.copy()
        fl = None
        for _ in range(max_iter):
            d = st.solve(F, active)
            fl = st.member_forces(d)
            if tension_only is None:
                break
            comp = tension_only & active & (fl[:, 6] < -comp_tol)
            if not np.any(comp):
                break
            active[comp] = False
        fl[~active] = 0.0
        out[name] = (d, fl, active)
    return out
