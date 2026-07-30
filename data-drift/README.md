# Data Drift

Notes and hands-on notebooks on **data drift in ML**: what it is, how it differs from neighboring
concepts (concept drift, prediction drift, training-serving skew, data quality, outliers), why it
matters for production models, and — most of the folder — four different techniques for actually
detecting it, each demonstrated end to end against the same synthetic dataset.

The concept notes are a distillation of (and all diagrams are copied from) Evidently AI's article
[*"What is data drift in ML, and how to detect and handle it"*](https://www.evidentlyai.com/ml-in-production/data-drift) —
thank you to the Evidently AI team for writing such a clear, freely available reference. Nothing in
this folder is original research; see `docs/README.md` for full source & credits.

## Folder contents

| Path | Description |
|---|---|
| [`docs/README.md`](docs/README.md) | **Start here for concepts.** Index of 6 short notes: what data drift is, how it compares to related concepts, why it matters, how to detect it, how to handle it, and the Evidently library. |
| `docs/01`–`06-*.md` | The concept notes themselves, in reading order, with diagrams under `docs/images/`. |
| [`notebooks/README.md`](notebooks/README.md) | **Start here for hands-on.** Purpose and "use this when..." guidance for each detection notebook. |
| `notebooks/01_summary_statistics.ipynb` | Means/quantiles, a two-sigma rule, min-max range compliance. |
| `notebooks/02_statistical_tests.ipynb` | Kolmogorov-Smirnov and Chi-square tests, plus a sample-size sensitivity experiment. |
| `notebooks/03_distance_metrics.ipynb` | Wasserstein distance, Jensen-Shannon divergence, a hand-implemented PSI. |
| `notebooks/04_rule_based_checks.ipynb` | A small rule engine for cheap, always-on threshold alerting. |
| [`src/drift_dataset.py`](src/drift_dataset.py) | Shared synthetic `reference`/`current` dataset generator all four notebooks analyze, so results are directly comparable across techniques. |
| `.notes-reference/` | Local, **untracked** (gitignored) saved copy of the source article, kept for offline reference — see the link above for the live version. |

## Setup

The notebooks need `numpy`, `pandas`, `scipy`, `matplotlib`, and `scikit-learn`, and were executed
under Python 3.10 via a dedicated Jupyter kernel:

```bash
python3.10 -m pip install numpy pandas scipy matplotlib scikit-learn ipykernel
python3.10 -m ipykernel install --user --name data-drift-py310 --display-name "Python 3.10 (data-drift)"
```

Every notebook is committed with its outputs already executed, so you can read them without running
anything. To re-run: open in Jupyter, select the `data-drift-py310` kernel, and run all cells — the
dataset is regenerated from a fixed seed, so results are deterministic.

## How to use this folder

1. Read `docs/README.md` and the six concept notes (01–06) for the "what" and "why."
2. Read or run `notebooks/README.md` and the four notebooks (01–04) for the "how" — each applies one
   detection technique in depth against the same drift scenario, so you can compare how differently
   each technique responds to the exact same shift.
3. Reuse `src/drift_dataset.py` directly if you want to try a new detection technique against the
   same reference/current data (see `COLUMN_DRIFT_GROUND_TRUTH` in that file for what each column is
   built to demonstrate).
