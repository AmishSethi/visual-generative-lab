# Table 2 Artifact

This directory is the canonical, paper-locked pipeline for the single-skill design-variation table.

What it does:
- generates exact 10k datasets for `size`, `position`, `rotation`, and `count`
- writes all new datasets, logs, Slurm scripts, and results under `$VGL_ROOT/MORE_SEEDS/table2`
- uses a single manifest for the baseline and the six non-redundant ablations
- exposes a parameter-matched `UNet-M` so the `+ UNet` row is not dominated by a parameter-count mismatch

Important note:
- the older checkpoints on `$VGL_ROOT/results` are useful provenance, but several of them were trained on undersized datasets such as `9216`, `9450`, or `2400` images rather than a canonical `10000`
- because of that, a clean reproducible artifact should rerun the table on the canonical datasets instead of averaging across the mixed legacy runs

Usage:

```bash
python paper_runs/table2/generate_canonical_datasets.py --force
python paper_runs/table2/submit_training.py --submit
```

To restrict submission to a subset:

```bash
python paper_runs/table2/submit_training.py --skills size position --variants baseline flow unet --seeds 0 1 2 --submit
```
