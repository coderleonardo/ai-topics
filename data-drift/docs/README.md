# Data Drift — Notes

Notes distilled from a single saved article — Evidently AI's *"What is data drift in ML, and how to detect and handle it"* (published at [evidentlyai.com](https://www.evidentlyai.com/ml-in-production/data-drift), saved locally 2025-01-09) — in reading order. Data drift is a shift in the distribution of the features an ML model receives in production, and these notes walk through what it is, how it differs from neighboring concepts (concept drift, prediction drift, training-serving skew, data quality, outliers), why it matters, how to detect it, and how to respond to it.

1. [What Is Data Drift?](01-what-is-data-drift.md)
2. [Data Drift vs. Related Concepts](02-data-drift-vs-related-concepts.md) — concept drift, prediction drift, training-serving skew, data quality, outlier detection
3. [Why Data Drift Matters](03-why-data-drift-matters.md) — model maintenance, feedback delay, model debugging
4. [How to Detect Data Drift](04-detecting-data-drift.md) — summary statistics, statistical tests, distance metrics, rule-based checks
5. [How to Handle Data Drift](05-handling-data-drift.md) — analysis, retraining, process intervention, model redesign
6. [Detecting Drift with Evidently](06-evidently-and-resources.md) — the OSS library, plus further reading

Diagrams referenced in these notes live in [`images/`](images/).

## Source & credits

All text here is a summarized distillation of, and all diagrams are copied from, the original article by the **Evidently AI** team — thank you to them for writing such a clear, freely available reference on this topic. Nothing in this folder is original research; if you want the authoritative version (with the newest updates, related articles, and the full site context), read it directly at [evidentlyai.com/ml-in-production/data-drift](https://www.evidentlyai.com/ml-in-production/data-drift). A local saved copy also lives in `../.notes-reference/` for offline reference — that folder is untracked, so it won't be present in a fresh clone of this repo.
