# Table 2 Artifact

This directory is the canonical, paper-locked pipeline for the single-skill design-variation table.

What it does:
- generates exact 10k datasets for `size`, `position`, `rotation`, and `count`
- writes all new datasets, logs, Slurm scripts, and results under `$VGL_ROOT/MORE_SEEDS/table2`
- uses a single manifest for the baseline and the seven ablations
- uses the parameter-matched `UNet-DiT-S2-matched` U-Net (~22M parameters) for every skill

Usage:

```bash
python paper_runs/table2/generate_canonical_datasets.py --force
python paper_runs/table2/submit_training.py --submit
```

The rotation baseline cell of Table 2 is `baseline_nz` over seeds 0-9:

```bash
python paper_runs/table2/submit_baseline_nz_extra_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9 --submit
```

To restrict submission to a subset:

```bash
python paper_runs/table2/submit_training.py --skills size position --variants baseline flow unet --seeds 0 1 2 --submit
```
