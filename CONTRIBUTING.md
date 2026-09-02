# Contributing

Issues and pull requests are welcome.

## Before opening a PR

```bash
python -m compileall -q vgl scripts paper_runs
python -m pytest tests/ -q
```

## If you change an evaluation metric

Every rule-based extractor in this project has a measured ceiling on ground-truth renders, and that
ceiling bounds any model accuracy computed with it. If you modify a metric, re-measure the ceiling
and report it in the PR. We have twice found that an apparent model failure was a metric that could
not read its own ground truth.

## If you change a dataset geometry

Regenerate the dataset, re-measure the relevant extractor ceiling, and say so in the PR. Model
numbers computed on a new geometry are not comparable to the published table.

## Reproducing before modifying

`paper_runs/` contains the paper-locked code paths. Please do not change the numbers those produce
without flagging it explicitly — they are the reference the published results are tied to.
