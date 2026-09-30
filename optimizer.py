"""CivilAI-FEA :: Genetic Algorithm for steel topology + section optimization
(Step 3).

Chromosome
  sizes[27] : section pool index per design group
              (col_*/brace_* groups index COLUMN_POOL, beam_* -> BEAM_POOL)
  bits[18]  : group-removal switches for the 12 beam groups and 6 brace
              groups (columns are structural and never removed)

Fitness (minimize)
  weight * (1 + 60*[util>0.90] + 60*[story-drift>h/300] + 60*[top>H/400])

Constraints evaluated on the full FEA (5 load cases, rigid diaphragms,
EC3 member checks). Every evaluation is cached; infeasible/singular
designs get infinite fitness.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from compliance import UTIL_LIMIT, drift_checks, member_utilization
from fea_solver import RHO_STEEL_W, SingularSystemError, Structure
from load_generator import (gravity_nodal_loads, seismic_equivalent_static,
                            seismic_weight_per_node, steel_self_weight_nodal,
                            wind_nodal_loads)
from model_builder import ALL_GROUPS, build_grid_frame, group_index_arrays, \
    materialize_design
from run_simulation import BASELINE, LOAD_KW, LATERAL_LCS
from sections import BEAM_POOL, COLUMN_POOL, pool_arrays

N_COL, N_BEAM, N_BRACE = 9, 12, 6           # group counts
REMOVABLE = list(range(N_COL, N_COL + N_BEAM + N_BRACE))
LATERAL = ("WINDX", "WINDY", "EQX", "EQY")
DRIFT_H, TOP_H = 300.0, 400.0               # drift limits (1/h)


class StructuralGA:
    def __init__(self, nx=5, ny=4, stories=20, pop=24, gens=100, seed=42,
                 p_cross=0.9, p_mut_size=0.25, p_mut_bit=0.10, elite=2,
                 log_csv=None):
        import os
        if log_csv is None:
            log_csv = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "results", "ga_log.csv")
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        self.pop, self.gens, self.elite = pop, gens, elite
        self.p_cross, self.p_ms, self.p_mb = p_cross, p_mut_size, p_mut_bit
        self.log_csv = log_csv

        self.model = build_grid_frame(nx=nx, ny=ny, stories=stories)
        self.st = Structure(self.model["nodes"], self.model["mi"],
                            self.model["mj"], self.model["supports"],
                            diaphragms=self.model["diaphragms"])
        self.nm = self.st.nm
        _, _, self.is_col_g = group_index_arrays(self.model)

        # design-space bookkeeping
        self.n_g = len(ALL_GROUPS)
        self.n_bits = len(REMOVABLE)
        self.base_sizes = np.array([BASELINE[g] for g in ALL_GROUPS], int)
        self.col_arr = pool_arrays(COLUMN_POOL)
        self.beam_arr = pool_arrays(BEAM_POOL)

        # fixed load components (design-independent except steel self-weight)
        self.Fg, self.Fq = gravity_nodal_loads(self.model)
        self.Wx = wind_nodal_loads(self.model, vb=LOAD_KW["vb"],
                                   cs_cd=LOAD_KW["cs_cd"], direction="+x")
        self.Wy = wind_nodal_loads(self.model, vb=LOAD_KW["vb"],
                                   cs_cd=LOAD_KW["cs_cd"], direction="+y")
        dsg0 = materialize_design(self.model, BASELINE)
        self.baseline_weight = float(
            np.sum(dsg0["A"] * RHO_STEEL_W * self.st.L))
        W_node = seismic_weight_per_node(
            self.model, self.Fg + steel_self_weight_nodal(self.model, dsg0["A"]),
            self.Fq)
        self.Ex = seismic_equivalent_static(self.model, W_node,
                                            cs=LOAD_KW["cs"], direction="+x")
        self.Ey = seismic_equivalent_static(self.model, W_node,
                                            cs=LOAD_KW["cs"], direction="+y")

        self.cache = {}
        self.rng = np.random.default_rng(seed)
        self.n_eval = 0
        self.t_eval = 0.0

    # ------------------------------------------------------------ chromosome
    def decode(self, sizes, bits):
        gi = {g: int(sizes[k]) for k, g in enumerate(ALL_GROUPS)}
        removed = [ALL_GROUPS[REMOVABLE[b]] for b in np.flatnonzero(bits)]
        return materialize_design(self.model, gi, removed)

    def random_chrom(self):
        sizes = self.base_sizes.copy()
        col_n, beam_n = len(COLUMN_POOL), len(BEAM_POOL)
        sizes[:N_COL + N_BRACE * 0] = self.rng.integers(0, col_n, N_COL)
        # brace groups sit at the END of ALL_GROUPS (indices 21..26)
        sizes[N_COL + N_BEAM:] = self.rng.integers(0, col_n, N_BRACE)
        sizes[N_COL:N_COL + N_BEAM] = self.rng.integers(0, beam_n, N_BEAM)
        bits = (self.rng.random(self.n_bits) < 0.15)
        return sizes, bits

    # ------------------------------------------------------------- evaluation
    def loads_for(self, A):
        G = self.Fg + steel_self_weight_nodal(self.model, A)
        Q05 = 0.5 * self.Fq
        return {
            "GRAV": 1.35 * G + 1.5 * self.Fq,
            "WINDX": G + Q05 + 1.5 * self.Wx,
            "WINDY": G + Q05 + 1.5 * self.Wy,
            "EQX": G + Q05 + self.Ex,
            "EQY": G + Q05 + self.Ey,
        }

    def evaluate(self, chrom):
        sizes, bits = chrom
        key = (sizes.tobytes(), bits.tobytes())
        if key in self.cache:
            return self.cache[key]
        t0 = time.perf_counter()
        self.n_eval += 1

        dsg = self.decode(sizes, bits)
        act = dsg["active"]
        is_col_m = self.is_col_g[dsg["mgroup"]]
        res = None
        if act.any():
            try:
                self.st.set_sections(dsg["A"], dsg["Ix"], dsg["Iy"], dsg["J"])
                fact = self.st.factor(act)        # one factorization, 5 LCs
                env_u = np.zeros(self.nm)
                story = top = 0.0
                for name, F in self.loads_for(dsg["A"]).items():
                    d = fact.solve(F)
                    e = Structure.force_envelope(self.st.member_forces(d))
                    u, _ = member_utilization(e["N"], e["V2"], e["V3"],
                                              e["M2"], e["M3"], dsg,
                                              self.st.L, is_col_m)
                    u[~act] = 0.0
                    env_u = np.maximum(env_u, u)
                    if name in LATERAL:
                        dc = drift_checks(d, self.model, DRIFT_H, TOP_H)
                        story = max(story, dc["story_ratio_max"])
                        top = max(top, dc["top_drift_max"])
                weight = self.st.weight(act)
                e_u = max(0.0, float(env_u.max()) - UTIL_LIMIT)
                e_d = max(0.0, story * DRIFT_H - 1.0)
                e_t = max(0.0, top * TOP_H - 1.0)
                fitness = weight * (1.0 + 60.0 * (e_u + e_d + e_t))
                res = {"fitness": fitness, "weight": weight,
                       "util": float(env_u.max()), "story": story, "top": top,
                       "n_removed": int((~act).sum())}
            except SingularSystemError:
                res = None
        if res is None:
            res = {"fitness": np.inf, "weight": np.inf, "util": np.inf,
                   "story": np.inf, "top": np.inf, "n_removed": int((~act).sum())}
        res["eval_ms"] = (time.perf_counter() - t0) * 1e3
        self.t_eval += res["eval_ms"]
        self.cache[key] = res
        return res

    # -------------------------------------------------------------------- GA
    def tournament(self, pop_fit, k=3):
        idx = self.rng.integers(0, len(pop_fit), k)
        return idx[np.argmin([pop_fit[i] for i in idx])]

    def crossover(self, a, b):
        sa, sb = a[0].copy(), b[0].copy()
        ba, bb = a[1].copy(), b[1].copy()
        m = self.rng.random(self.n_g) < 0.5
        sa[m], sb[m] = b[0][m], a[0][m]
        m = self.rng.random(self.n_bits) < 0.5
        ba[m], bb[m] = b[1][m], a[1][m]
        return (sa, ba), (sb, bb)

    def mutate(self, chrom):
        sizes, bits = chrom[0].copy(), chrom[1].copy()
        col_n, beam_n = len(COLUMN_POOL), len(BEAM_POOL)
        for k in range(self.n_g):
            if self.rng.random() < self.p_ms:
                lo = 0
                hi = col_n if (k < N_COL or k >= N_COL + N_BEAM) else beam_n
                jump = int(self.rng.integers(1, 4)) * self.rng.choice((-1, 1))
                sizes[k] = int(np.clip(sizes[k] + jump, lo, hi - 1))
        flip = self.rng.random(self.n_bits) < self.p_mb
        bits ^= flip
        return sizes, bits

    def ramped(self, k):
        sizes = self.base_sizes.copy()
        col_n, beam_n = len(COLUMN_POOL), len(BEAM_POOL)
        for g in range(self.n_g):
            hi = col_n if (g < N_COL or g >= N_COL + N_BEAM) else beam_n
            sizes[g] = int(min(sizes[g] + k, hi - 1))
        return sizes, np.zeros(self.n_bits, bool)

    def run(self, verbose=True):
        t0 = time.perf_counter()
        seed_pop = [self.ramped(k) for k in (0, 1, 2, 3)]
        pop = seed_pop + [self.random_chrom() for _ in range(self.pop - 4)]
        fits = [self.evaluate(c) for c in pop]
        order = np.argsort([f["fitness"] for f in fits])

        rows = []
        for gen in range(1, self.gens + 1):
            new = [pop[i] for i in order[:self.elite]]
            newfits = [fits[i] for i in order[:self.elite]]
            while len(new) < self.pop:
                pa = pop[self.tournament([f["fitness"] for f in fits])]
                pb = pop[self.tournament([f["fitness"] for f in fits])]
                ca, cb = self.crossover(pa, pb)
                new.append(self.mutate(ca))
                if len(new) < self.pop:
                    new.append(self.mutate(cb))
            pop = new
            fits = [self.evaluate(c) for c in pop]
            order = np.argsort([f["fitness"] for f in fits])
            best = fits[order[0]]
            rows.append({"gen": gen, "best_fitness": best["fitness"],
                         "best_weight_kN": best["weight"],
                         "mean_fitness": float(np.mean([f["fitness"] for f in fits
                                                        if np.isfinite(f["fitness"])])),
                         "best_util": best["util"], "best_story_drift": best["story"],
                         "best_top_drift": best["top"],
                         "n_removed": best["n_removed"]})
            if verbose and (gen % 5 == 0 or gen == 1):
                w = best["weight"]
                print(f"gen {gen:3d} | best w {w/9.81:7.1f} t | fit {best['fitness']:10.1f} | "
                      f"util {best['util']:.3f} | story {best['story']*100:.3f}% | "
                      f"top {best['top']*100:.3f}% | rm {best['n_removed']:3d} | "
                      f"evals {self.n_eval} ({self.t_eval/max(self.n_eval,1):.0f} ms)",
                      flush=True)

        with open(self.log_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        best_i = order[0]
        return pop[best_i], fits[best_i], rows, time.perf_counter() - t0


def section_name(group, idx):
    arr = COLUMN_POOL if (group.startswith("col_") or group.startswith("brace_")) \
        else BEAM_POOL
    return arr[idx].name


def main():
    import os
    _results = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(_results, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(_results, "best_design.json"))
    args = ap.parse_args()

    ga = StructuralGA(pop=args.pop, gens=args.gens, seed=args.seed)
    base_w = ga.baseline_weight
    print(f"GA start: pop={args.pop} gens={args.gens} | baseline steel "
          f"{base_w/9.81:.1f} t | {ga.n_g} sizing genes + {ga.n_bits} removal bits",
          flush=True)

    (bs, bb), best, rows, dt = ga.run()

    dsg = ga.decode(bs, bb)
    out = {
        "generations": args.gens, "population": args.pop, "seed": args.seed,
        "wall_time_s": dt, "n_evaluations": ga.n_eval,
        "baseline_weight_kN": base_w,
        "best_weight_kN": best["weight"],
        "reduction_pct": 100.0 * (1.0 - best["weight"] / base_w),
        "best_fitness": best["fitness"], "max_util": best["util"],
        "story_drift": best["story"], "top_drift": best["top"],
        "n_members_removed": best["n_removed"],
        "groups": {g: {"pool_index": int(bs[k]), "section": section_name(g, bs[k])}
                   for k, g in enumerate(ALL_GROUPS)},
        "removed_groups": [ALL_GROUPS[REMOVABLE[b]] for b in np.flatnonzero(bb)],
        "history_every10": rows[9::10],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 78)
    print(f"BEST: {best['weight']/9.81:.1f} t ({out['reduction_pct']:+.1f}% vs "
          f"baseline {base_w/9.81:.1f} t) | util {best['util']:.3f} | "
          f"story {best['story']*100:.3f}% | top {best['top']*100:.3f}% | "
          f"removed {best['n_removed']} members")
    print(f"design written to {args.out}, log in {ga.log_csv}")


if __name__ == "__main__":
    main()
