# CivilAI-FEA — Engineering Report

**Date:** 2026-09-30 · **Scope:** 3D FEA solver + genetic topology optimizer for a 20-story steel building, built from scratch in Python (NumPy/SciPy).

---

## 1. System architecture

| Module | Content |
|---|---|
| `fea_solver.py` | Direct Stiffness Method, 3D frame elements, 6 DOF/node. Vectorized 12×12 local stiffness, 3D transformation (Gram-Schmidt local axes), global assembly in reduced/condensed space (`scipy.sparse` COO→CSR), `splu` factorization shared across load cases, member end forces (N, V2, V3, T, M2, M3), reactions. Rigid-diaphragm kinematic condensation (3,780 → 2,040 DOF). |
| `sections.py` | W-shape library (9 columns incl. W14x233, 9 beams), SI units, derived Iy/J/Zy/Cw. |
| `model_builder.py` | 5×4-bay × 20-story grid (35×28×72 m), 1,820 members (600 columns, 980 beams, 240 X-braces in symmetric central bays), 27 design groups, rigid floor diaphragms. |
| `load_generator.py` | Wind: power-law profile v(z)=vb(z/10)^0.16, q=0.613e-3·v², vb=40 m/s, net Cp 1.3. Seismic: V=Cs·W, Cs=0.05, distributed ∝ W_i·z_i. Gravity: 1.35G+1.5Q tributary nodal loads incl. facade and steel self-weight. |
| `compliance.py` | EC3: tension yield, flexural buckling (curve b, χ from λ̄), strong-axis bending with LTB (standard I-Mcr, χLT), weak bending, shear, linear interaction; drift h/300 story, H/400 top. |
| `optimizer.py` | GA: 27 sizing genes + 18 group-removal bits (columns protected), tournament/uniform-crossover/jump-mutation, elitism, evaluation cache; fitness = weight × (1 + 60·violations). |
| `verify.py` | 11 analytical cases. `harden_loop.py` | 40-cycle randomized stress sweep with invariant checks. |

## 2. Verification (analytical, machine precision)

11/11 cases pass at ~1e-16 relative error: axial bar, strong/weak-axis cantilevers (PL³/3EI), torsion (TL/GJ), simple beam (PL³/48EI + reactions), diagonal member, 45°-rotated cantilever (3D transformation), sparse-vs-dense cross-solve, K symmetry, buckling-curve sanity, zero-load rigidity, diaphragm kinematics + equilibrium.

## 3. Autonomous hardening — real defects found and fixed

The verification/hardening loops caught **10 genuine defects**, each fixed and re-validated:

1. Bending terms assembled without the `EI/L³` factor + diagonal double-add in the symmetric fill (deflections EI-independent).
2. Section library `Iy` missing the `/12` (12× error → buckling checks wrong).
3. Steel self-weight used DOF stride 2 instead of 6 (loads landed in wrong DOFs, wrong sign).
4. Facade tributary length halved on non-corner perimeter nodes.
5. Duplicate `solve()` definition shadowed the diaphragm-condensation version (Python last-definition-wins).
6. Diaphragm rotation-coupling entries placed in wrong matrix rows/columns.
7. Reduced-space indexing mixed physical DOF ids with condensed positions (IndexError).
8. Force-balance invariant in the hardening loop summed forces and moments into one scalar (harness bug, per-axis rewrite).
9. Tension-only bracing model invalidated by gravity column-shortening precompression (92% of braces shed) → replaced with buckling-restrained (BRB) behavior, EC3 compression check governs.
10. Single central X-braced bay overloaded its beams (util 0.955 at max pool) → symmetric brace-plane pair; column pool extended with W14x233 → feasibility restored (max util 0.790 at all-max).

## 4. Baseline simulation (20 stories, 1,820 members)

- Baseline design (1,220 t): drift OK in all directions (0.222–0.304% vs 0.333% limit) except EQX top drift marginal (0.251% vs 0.250%); **134 members above 90% utilization** (max 1.44, high-tier braced-bay beams) — i.e. an economical but code-violating design.
- Equilibrium closed to ~1e-9 kN on every load case; solve latency ≈ 190 ms/load case (incl. condensation).

## 5. Genetic optimization (100 generations, pop 24, ~2,200 FEA evaluations)

Convergence (best design per generation):

| gen | weight | max util | story drift | top drift | members removed |
|----:|-------:|---------:|------------:|----------:|----------------:|
| 1   | 1,814.9 t | 0.811 | 0.198% | 0.166% | 0   |
| 11  | 1,553.8 t | 0.899 | 0.240% | 0.184% | 224 |
| 21  | **1,347.8 t** | 0.894 | 0.247% | 0.213% | 242 |
| 100 | **1,347.8 t** | 0.894 | 0.247% | 0.213% | 242 |

- **All constraints satisfied**: utilization 0.894 ≤ 0.90, story drift 0.247% ≤ h/300, top drift 0.213% ≤ H/400.
- **−25.7% weight within the run** (best feasible gen-1 → gen-21 optimum); **−35.9%** vs the all-max feasible reference (2,103 t); +10.5% vs the lighter but code-violating baseline — the price of full compliance.
- The GA removed 3 non-critical beam groups (242 members) and independently discovered the physically sensible pattern: heavy Y-braces at low tiers (W14x233), lighter braces up high, deep girders where brace forces concentrate, graduated column sizes.

## 6. Hardening sweep (final code)

**40/40 cycles passed** — irregular random 3D frames (spanning-tree connectivity, non-collinear fixed supports) and randomized grid variants (braces/diaphragms on/off, scaled loads). Invariants per cycle: equilibrium residual < 1e-8, stiffness symmetry ≤ 1e-16, positive strain energy, per-axis force balance, finite bounded displacements; full analytical battery re-run every 5th cycle (11/11 each time). Solve latency on random models: mean 8.9 ms, p95 25.5 ms, max 38.7 ms.

## 7. Known limitations

Linear-elastic (no P-Δ), nodal loading only, simplified EC3 interactions, fixed bases, approximate section properties, seismic weight tied to baseline steel inside the GA. See `README.md`.

**Artifacts:** `baseline_results.json`, `ga_log.csv`, `ga_output.txt`, `best_design.json`, `hardening_log.csv`.
