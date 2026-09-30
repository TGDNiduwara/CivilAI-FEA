"""Grid-frame model builder for CivilAI-FEA.

Generates the node/member topology of a regular rectangular moment-frame
building, assigns every member to a design group (used by the GA
chromosome) and exposes grid bookkeeping (level / i / j per node).
"""
from __future__ import annotations

import numpy as np

# design groups: 9 column groups (class x story tier) + 12 beam groups
# (axis/orientation x story tier)
COLUMN_CLASSES = ("corner", "edge", "interior")
BEAM_CLASSES = ("beam_x_perim", "beam_x_int", "beam_y_perim", "beam_y_int")
TIERS = ("low", "mid", "high")          # low: 1-1/3, mid: 1/3-2/3, high: 2/3-1
COL_GROUPS = [f"col_{c}_{t}" for c in COLUMN_CLASSES for t in TIERS]
BEAM_GROUPS = [f"{b}_{t}" for b in BEAM_CLASSES for t in TIERS]
BRACE_GROUPS = [f"brace_{o}_{t}" for o in ("x", "y") for t in TIERS]
ALL_GROUPS = COL_GROUPS + BEAM_GROUPS + BRACE_GROUPS


def _tier(story_1based, stories):
    s = story_1based / stories
    return "low" if s <= 1 / 3 else ("mid" if s <= 2 / 3 else "high")


def _center_bays(n):
    """Symmetric central bay indices for n bays."""
    return (n // 2,) if n % 2 else (n // 2 - 1, n // 2)


def _center_lines(n):
    """Symmetric central grid-line indices for n bays (n+1 lines)."""
    return (n // 2,) if (n + 1) % 2 else (n // 2 - 1, n // 2 + 1)


def _brace_planes(n):
    """Symmetric brace plane pairs (a brace plane pair shares the overturning
    couple; a single central plane would overload its beams and columns)."""
    return (n // 2, n // 2 + 1) if n % 2 else (n // 2 - 1, n // 2 + 1)


def build_grid_frame(nx=5, ny=4, stories=20, bay_x=7.0, bay_y=7.0, story_h=3.6,
                     braced=True, diaphragm=True):
    """Rectangular grid moment frame with optional central X-bracing.

    nx, ny   : number of bays in X / Y
    stories  : number of elevated floors (base at z=0)
    braced   : add X-braces (both diagonals per bay per story) in the
               symmetric central bays/planes -- X-sway braces in planes
               _center_lines(ny), Y-sway braces in planes _center_lines(nx).
               Braces connect existing nodes only and are modeled with
               buckling-restrained (BRB) behavior: active in tension AND
               compression, sized through the EC3 buckling check.

    Returns a model dict with topology + bookkeeping.
    """
    npx, npy = nx + 1, ny + 1
    nn = npx * npy * (stories + 1)

    def nid(lvl, i, j):
        return lvl * npx * npy + j * npx + i

    nodes = np.zeros((nn, 3))
    lvl_arr = np.zeros(nn, int)
    i_arr = np.zeros(nn, int)
    j_arr = np.zeros(nn, int)
    for lvl in range(stories + 1):
        for j in range(npy):
            for i in range(npx):
                n = nid(lvl, i, j)
                nodes[n] = (i * bay_x, j * bay_y, lvl * story_h)
                lvl_arr[n], i_arr[n], j_arr[n] = lvl, i, j

    mi, mj, groups, m_axis = [], [], [], []
    # columns: every grid point, every story (never removed by the GA)
    for lvl in range(stories):
        story = lvl + 1
        tier = _tier(story, stories)
        for j in range(npy):
            for i in range(npx):
                corner = i in (0, nx) and j in (0, ny)
                edge = (i in (0, nx)) or (j in (0, ny))
                cls = "corner" if corner else ("edge" if edge else "interior")
                mi.append(nid(lvl, i, j)); mj.append(nid(lvl + 1, i, j))
                groups.append(f"col_{cls}_{tier}"); m_axis.append("z")
    # beams at every elevated floor
    for lvl in range(1, stories + 1):
        tier = _tier(lvl, stories)
        for j in range(npy):                       # beams along X
            for i in range(nx):
                per = j in (0, ny)
                mi.append(nid(lvl, i, j)); mj.append(nid(lvl, i + 1, j))
                groups.append(f"beam_x_{'perim' if per else 'int'}_{tier}")
                m_axis.append("x")
        for i in range(npx):                       # beams along Y
            for j in range(ny):
                per = i in (0, nx)
                mi.append(nid(lvl, i, j)); mj.append(nid(lvl, i, j + 1))
                groups.append(f"beam_y_{'perim' if per else 'int'}_{tier}")
                m_axis.append("y")

    # diagonal X-braces in the symmetric central bays (X-sway and Y-sway)
    if braced:
        for lvl in range(stories):
            story = lvl + 1
            tier = _tier(story, stories)
            # X-sway: bays along X at the symmetric brace plane pair in Y
            for jp in _brace_planes(ny):
                for bi in _center_bays(nx):
                    mi.append(nid(lvl, bi, jp)); mj.append(nid(lvl + 1, bi + 1, jp))
                    groups.append(f"brace_x_{tier}"); m_axis.append("diag")
                    mi.append(nid(lvl, bi + 1, jp)); mj.append(nid(lvl + 1, bi, jp))
                    groups.append(f"brace_x_{tier}"); m_axis.append("diag")
            # Y-sway: bays along Y at the symmetric brace plane pair in X
            for ip in _brace_planes(nx):
                for bj in _center_bays(ny):
                    mi.append(nid(lvl, ip, bj)); mj.append(nid(lvl + 1, ip, bj + 1))
                    groups.append(f"brace_y_{tier}"); m_axis.append("diag")
                    mi.append(nid(lvl, ip, bj + 1)); mj.append(nid(lvl + 1, ip, bj))
                    groups.append(f"brace_y_{tier}"); m_axis.append("diag")

    mi = np.array(mi, int); mj = np.array(mj, int)
    supports = np.zeros((nn, 6), bool)
    supports[lvl_arr == 0, :] = True               # fully fixed base

    # rigid floor diaphragms: every elevated floor tied to its first node
    diaphragms = []
    if diaphragm:
        for lvl in range(1, stories + 1):
            floor_nodes = np.flatnonzero(lvl_arr == lvl)
            diaphragms.append((int(floor_nodes[0]), floor_nodes[1:]))

    return {
        "nodes": nodes, "mi": mi, "mj": mj, "groups": groups, "m_axis": m_axis,
        "supports": supports, "nx": nx, "ny": ny, "stories": stories,
        "bay_x": bay_x, "bay_y": bay_y, "story_h": story_h,
        "npx": npx, "npy": npy, "braced": braced, "diaphragms": diaphragms,
        "node_level": lvl_arr, "node_i": i_arr, "node_j": j_arr,
        "nid": nid,
    }


def group_index_arrays(model):
    """Map each member to its design-group index; also return per-group
    member index lists and a column mask per group."""
    gidx = {g: k for k, g in enumerate(ALL_GROUPS)}
    mgroup = np.array([gidx[g] for g in model["groups"]], int)
    per_group = [np.flatnonzero(mgroup == k) for k in range(len(ALL_GROUPS))]
    is_col = np.array([g.startswith("col_") for g in ALL_GROUPS])
    return mgroup, per_group, is_col


def story_drift_ratios(d, model):
    """Max inter-story drift ratio per story (h_ref), for X and Y sway.

    d: (ndof,) displacement vector. Returns (rx, ry) arrays of length
    `stories` (ratio = max over all column lines of |Δ|/story_h).
    """
    st, npx, npy = model["stories"], model["npx"], model["npy"]
    h = model["story_h"]
    ux = d[0::6].reshape(st + 1, npy, npx)
    uy = d[1::6].reshape(st + 1, npy, npx)
    rx = np.abs(np.diff(ux, axis=0)).reshape(st, -1).max(axis=1) / h
    ry = np.abs(np.diff(uy, axis=0)).reshape(st, -1).max(axis=1) / h
    return rx, ry


def member_stats(model):
    """Human-readable member counts."""
    from collections import Counter
    return Counter(model["groups"])


def materialize_design(model, group_pool_idx, removed_groups=()):
    """Convert a per-group design (pool indices) into per-member arrays.

    group_pool_idx : {group_name: index into COLUMN_POOL for col_* groups,
                      index into BEAM_POOL for beam_* groups}
    removed_groups : group names whose members are deactivated (GA topology
                     removal). Columns are never removed.

    Returns dict with per-member property arrays (float64), `active` mask,
    `names` (section name per member) and `pool_idx` per member.
    """
    import numpy as np
    from sections import BEAM_POOL, COLUMN_POOL, pool_arrays

    nm = len(model["mi"])
    mgroup, per_group, is_col = group_index_arrays(model)
    col_arr, beam_arr = pool_arrays(COLUMN_POOL), pool_arrays(BEAM_POOL)
    prop_keys = ("A", "Ix", "Iy", "J", "Zx", "Zy", "d", "bf", "tw", "tf", "Cw", "hw")
    sec = {k: np.zeros(nm) for k in prop_keys}
    names = [""] * nm
    pool_idx = np.zeros(nm, int)
    active = np.ones(nm, bool)
    removed = set(removed_groups)

    for gi, g in enumerate(ALL_GROUPS):
        idxs = per_group[gi]
        if len(idxs) == 0:
            continue
        # columns and braces draw from the W14 column pool
        arr = col_arr if (is_col[gi] or g.startswith("brace_")) else beam_arr
        pi = int(group_pool_idx[g])
        if g in removed and not is_col[gi]:
            active[idxs] = False
        for k in prop_keys:
            sec[k][idxs] = arr[k][pi]
        pool_idx[idxs] = pi
        for i in idxs:
            names[i] = arr["names"][pi]
    sec["active"] = active
    sec["names"] = names
    sec["pool_idx"] = pool_idx
    sec["mgroup"] = mgroup
    return sec
