"""CivilAI-FEA: 3D Finite Element Analysis solver + Genetic Topology Optimizer.

Modules:
  fea_solver     Direct Stiffness Method, 3D frame elements (6 DOF/node)
  sections       W-shape steel section library
  model_builder  rectangular grid moment-frame generator
  load_generator wind / seismic / gravity load generation
  compliance     Eurocode 3 member checks
  optimizer      genetic algorithm for sizing + member removal
  run_simulation 20-story stress-test pipeline
  verify         analytical verification battery
  harden_loop    autonomous stress-hardening sweep
"""
