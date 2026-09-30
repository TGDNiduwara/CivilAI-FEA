# CivilAI-FEA

**A 3D Finite Element Analysis solver and Genetic Topology Optimizer for steel buildings — implemented from scratch in Python (NumPy/SciPy).**

CivilAI-FEA performs matrix structural analysis of 3D steel frames by the
Direct Stiffness Method, generates code-style wind/seismic/gravity loading
automatically, checks every member against Eurocode 3, and evolves the
structure with a Genetic Algorithm that minimizes steel weight subject to
strict strength and drift constraints.

---

## Highlights

- **3D frame FEA from first principles** — 6 DOF/node (ux, uy, uz, rx, ry, rz),
  vectorized 12×12 element stiffness matrices, 3D coordinate transformations,
  sparse global assembly (`scipy.sparse`), rigid-diaphragm condensation, and
  one factorization reused across all load cases.
- **Automatic extreme-load generation** — height-dependent dynamic wind
  pressure (power-law velocity profile), equivalent-static seismic base
  shear distributed by first-mode shape, and factored gravity via
  tributary areas including facade and steel self-weight.
- **Eurocode 3 verification** — tension yield, flexural buckling (curve b),
  strong-axis bending with lateral-torsional buckling, weak-axis bending,
  shear, and linear interaction; story-drift and top-drift serviceability.
- **Genetic topology optimization** — simultaneous section resizing
  (27 design groups) and member removal (18 removable groups); fitness =
  steel weight penalized for any utilization above 90% or any drift exceedance.
- **Verification-first engineering** — an 11-case analytical battery
  (machine-precision agreement) and a 40-cycle randomized stress-hardening
  sweep with per-cycle invariants.

## Results on the demo problem (20-story building)

| quantity | value |
|---|---|
| structure | 5×4 bays (35×28 m), 20 stories × 3.6 m, 72 m tall |
| members | 1,820 (600 columns, 980 beams, 240 X-braces) |
| DOFs | 3,780 modeled → 2,040 after diaphragm condensation |
| baseline design | 1,220 t, **134 members over 90% utilization** (infeasible) |
| GA optimum (100 generations) | **1,347.8 t, fully compliant**: util 0.894 ≤ 0.90, story drift 0.247% ≤ h/300, top drift 0.213% ≤ H/400 |
| GA effect | −25.7% within the run; −35.9% vs all-max feasible reference; +10.5% vs the lighter but code-violating baseline |
| verification battery | 11/11 analytical cases at ~1e-16 relative error |
| hardening sweep | 40/40 randomized cycles passed (equilibrium < 1e-8, symmetry ≤ 1e-16) |
| solve latency | ≈ 190 ms/load case (full building) · 8.9 ms mean on random models |

Full numbers and the generation-by-generation GA log: [ENGINEERING_REPORT.md](ENGINEERING_REPORT.md), [results/](results/).

## Installation

```bash
git clone https://github.com/<your-account>/CivilAI-FEA.git
cd CivilAI-FEA
pip install -r requirements.txt   # numpy, scipy
```

Python 3.10+ recommended (developed on 3.12, NumPy 2.5, SciPy 1.18).

## Usage

Run from the repository root (scripts use flat sibling imports, so invoke
them by path):

```bash
# 1. Analytical verification battery (all 11 cases must pass)
python civil_fea/verify.py

# 2. Baseline stress test: 20-story, 1,820-member frame, 5 load cases,
#    EC3 checks, drift, reactions, staged latency
python civil_fea/run_simulation.py            # writes results/baseline_results.json

# 3. Genetic optimization, 100 generations (logs ga_log.csv, best_design.json)
python civil_fea/optimizer.py --pop 24 --gens 100 --seed 42 --out best_design.json

# 4. Autonomous hardening sweep: 40 randomized structures with invariant checks
python civil_fea/harden_loop.py               # writes results/hardening_log.csv
```

Pre-generated outputs of the reference run are shipped in
[`results/`](results/) (`best_design.json` contains the winning design:
section per group + removed groups).

## Repository structure

```
CivilAI-FEA/
├── README.md                  this file
├── ENGINEERING_REPORT.md      full engineering report (results, defects found, logs)
├── LICENSE                    MIT
├── requirements.txt
├── .gitignore
├── results/                   artifacts of the reference run
└── civil_fea/                 the Python package
    ├── __init__.py
    ├── fea_solver.py          Direct Stiffness Method core (sparse, condensed)
    ├── sections.py            W-shape steel section library (SI units)
    ├── model_builder.py       grid moment-frame generator + brace/diaphragm layout
    ├── load_generator.py      wind / seismic / gravity load generation
    ├── compliance.py          Eurocode 3 member checks + drift serviceability
    ├── optimizer.py           genetic algorithm (sizing + topology removal)
    ├── run_simulation.py      20-story stress-test pipeline
    ├── verify.py              analytical verification battery (11 cases)
    └── harden_loop.py         40-cycle randomized stress-hardening sweep
```

## Engineering model

**Analysis.** Linear-elastic, small-displacement, Euler-Bernoulli frame
elements. Local axes: axis 1 = member axis; axis 2 = projected global +Z
("up" for horizontal members); axis 3 = 1×2 (right-handed). Force output
follows SAP-style naming: N (tension +), V2/V3 shear, T torsion, M2 (weak),
M3 (strong) — for standard orientations vertical load on beams produces M3.

**Structure.** Regular grid moment frame with X-braces (both diagonals,
buckling-restrained behavior — active in tension *and* compression, sized
by the EC3 compression check) in the symmetric central bays, and rigid
floor diaphragms (in-plane UX/UY/RZ of every slab tied to a per-floor
master node, condensed out of the solution).

**Loads (extreme-event set).**

| LC | combination |
|---|---|
| GRAV | 1.35 G + 1.50 Q |
| WINDX/WINDY | 1.0 G + 0.5 Q + 1.5 W (vb = 40 m/s, q(z) = 0.613e-3·v(z)², net Cp 1.3) |
| EQX/EQY | 1.0 G + 0.5 Q + 1.0 E (V = 0.05·W distributed ∝ W_i·z_i) |

**Design groups.** Columns (corner/edge/interior × low/mid/high tiers),
beams (x/y × perimeter/interior × tiers), braces (x/y × tiers). The GA
assigns a W-shape per group and may delete whole beam/brace groups;
columns are always retained (they guarantee connectivity and stability).

## Verification & validation

- `verify.py` — 11 closed-form cases at ~1e-16: axial bar, strong/weak-axis
  cantilevers, torsion, simple beam (deflection + reactions), diagonal
  member, 45°-rotated cantilever, sparse-vs-dense cross-solve, K symmetry,
  buckling-curve sanity, zero-load rigidity, diaphragm kinematics.
- `harden_loop.py` — 40 cycles over randomized irregular 3D frames
  (spanning-tree connectivity, non-collinear supports) and randomized grid
  variants (braces/diaphragms on/off, load scales). Invariants per cycle:
  equilibrium residual < 1e-8, stiffness symmetry ≤ 1e-16, positive strain
  energy, per-axis force balance, finite bounded displacements; the full
  analytical battery is re-run every 5th cycle.

These loops caught and fixed **10 real defects** during development
(stiffness factors, section-property errors, DOF-stride bugs, a shadowed
`solve()` definition, diaphragm matrix placement, and a physical modeling
correction: tension-only bracing invalidated by gravity column shortening —
replaced with buckling-restrained behavior). Details in
[ENGINEERING_REPORT.md](ENGINEERING_REPORT.md).

## Known limitations

- Linear elastic; no P-Δ / geometric nonlinearity.
- Nodal loading only (slab loads enter as equivalent nodal forces; no
  member-distributed loads inside elements).
- EC3 checks are deliberately simplified (linear interaction, curve b for
  buckling and LTB, k_eff 1.5 columns / 1.2 beams); not a substitute for a
  full code interaction design.
- Section properties derived from nominal dimensions without fillet
  contributions (few-percent accuracy) — **not for construction documents**.
- Fixed bases (no soil-structure interaction); seismic weight uses the
  baseline steel tonnage inside the GA loop.

## License

[MIT](LICENSE) — free to use, study, modify, and distribute.
