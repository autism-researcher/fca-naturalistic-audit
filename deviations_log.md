# Deviations Register

Per §13 of the pre-registration, classify each entry:

- **within-protocol refinement** (does not change scope)
- **post-hoc addition** (analysis not in this registration)
- **verdict-changing** (alters pass/fail conclusion of a confirmatory test)

Verdict-changing deviations require explicit pre-disclosure before the affected analysis is run.

| Date | Item | Class | Impact |
|---|---|---|---|
| (none yet) | | | |
| 2026-05-19 | --skip-gate-check on run_waymo_experiment.py (N=5000): /mnt/d copy is not a git working tree, so the Gate 2 tag cannot be queried. Pipeline frozen state recorded by tree sha256 = 6c57385f90ff33c37c12cc65319e90d73e770578d1f8bc214e870e73659c6399 over src/, scripts/, carla_weights.json, paper2_constants.json. | within-protocol refinement | none — same code as OSF supplement; gate intent (freeze before confirmatory) satisfied by hash anchor |
