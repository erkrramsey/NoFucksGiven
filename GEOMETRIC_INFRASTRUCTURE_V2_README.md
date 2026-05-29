# Geometric Infrastructure Optimization v2 Hard-Push Bundle

This bundle formalizes and tests locality-first compute fabric placement.

## Files

- `geometric_infrastructure_optimization_thesis_v2_hardpush.pdf` - rendered thesis PDF.
- `geometric_infrastructure_optimization_thesis_v2_hardpush.tex` - LaTeX source.
- `geometric_infrastructure_v2_hardpush_tests_executed.ipynb` - executed machine-precision notebook.
- `geometric_scheduler_v2_reference.py` - reference Python extracted from the notebook.
- `geometric_infra_v2_evidence.json` - deterministic evidence manifest with SHA-256 root.
- `geo_v2_figures/` - generated benchmark figures.

## Evidence root

`cff9b39527b2746771d855214e3d7e1d7aa3092bfb990aa7854c9393363afd2c`

## Core result

The system proves the exact integer swap-delta identity, tests float64 exactness within the 2^53 safety envelope, enumerates a true small-case optimum, benchmarks space-filling/ribbon layout against random placement, tests rack-aware cluster packing, simulates dynamic workload drift with migration cost, and verifies a uniform-communication null case where geometry gives no advantage.

## How to rerun

Open the executed notebook and rerun all cells, or run:

```bash
jupyter nbconvert --to notebook --execute geometric_infrastructure_v2_hardpush_tests.ipynb --output rerun.ipynb --ExecutePreprocessor.timeout=300
```

The notebook is self-contained. Major tests rebuild fixtures inside the test functions to avoid undefined-symbol failures from out-of-order execution.
