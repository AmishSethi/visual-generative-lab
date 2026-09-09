# Table 2 Artifact

This directory is the canonical, paper-locked pipeline for the single-skill design-variation table.

What it does:
- generates exact 10k datasets for `size`, `position`, `rotation`, and `count`
- writes all new datasets, logs, Slurm scripts, and results under `$VGL_ROOT/MORE_SEEDS/table2`
- uses a single manifest for the baseline and the seven ablations
- uses parameter-matched U-Nets (`UNet-DiT-S2-matched` for size, position, and rotation; `UNet-M` for count) so the `+ U-Net` row is not dominated by a parameter-count mismatch

Usage:

```bash
python paper_runs/table2/generate_canonical_datasets.py --force
python paper_runs/table2/submit_training.py --submit
```

To restrict submission to a subset:

```bash
python paper_runs/table2/submit_training.py --skills size position --variants baseline flow unet --seeds 0 1 2 --submit
```
