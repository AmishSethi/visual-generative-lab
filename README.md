# Visual Generative Lab (VGL)

Code for **"Do Diffusion Models Learn to Generalize Basic Visual Skills?"**

[Paper](https://arxiv.org/abs/XXXX.XXXXX) · [Datasets](#datasets) · [Pretrained checkpoints](#pretrained-checkpoints)

Diffusion models are usually evaluated with distribution-level scores (FID, IS) or human preference,
neither of which tells you whether a model has *learned a visual skill* or is reproducing training
patterns. VGL trains diffusion models from scratch on synthetic data where every skill value is
controlled by construction, then measures generalization with rule-based metrics that have no learned
components.

Six skills — **size**, **position**, **rotation**, **count**, **shape**, **color** — each evaluated on
three splits:

| split | meaning |
|---|---|
| **train** | held-out samples whose skill values appear in training |
| **interp** | values inside the training range but never seen (e.g. radius 12.5) |
| **extra** | values outside the training range entirely (e.g. radius 25) |

---

## Findings

1. **Extrapolation is limited; rotation is the exception.** Every skill fits its training support
   nearly perfectly, and only rotation continues to work far outside it.
2. **Coverage beats dataset size.** At a fixed image budget, spreading samples across more unique
   skill combinations beats concentrating them on fewer.
3. **Skills are learned jointly.** An out-of-range request in one skill degrades the others, even
   when those remain in range.

---

## Results

Accuracy (%) on the four numeric skills, mean over three seeds (ten for the rotation baseline).
Thresholds: IoU ≥ 0.90 (size), ≤ 2 px (position), ≤ 5° (rotation), exact match (count).

| Model | size train | size extra | pos train | pos extra | rot train | rot extra | count train | count extra |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **DiT-S/2 (baseline)** | 100.0 | 47.3 | 100.0 | 31.0 | 100.0 | 68.0 | 99.7 | 34.6 |
| + sinusoidal emb. | 100.0 | 7.1 | 100.0 | 0.0 | 99.7 | 26.6 | 99.2 | 19.2 |
| + rotary emb. | 100.0 | 10.4 | 100.0 | 0.0 | 100.0 | 37.6 | 99.7 | 34.2 |
| + AdaLN | 100.0 | 31.4 | 100.0 | 63.6 | 100.0 | 58.5 | 99.7 | 45.0 |
| + VAE latent | 82.8 | 6.1 | 99.8 | 0.1 | 99.9 | 21.2 | 66.9 | 45.8 |
| + flow matching | 49.5 | 24.0 | 100.0 | 12.1 | 100.0 | 57.8 | 10.3 | 5.0 |
| + U-Net (capacity-matched) | 100.0 | 19.9 | 100.0 | 50.0 | 99.7 | 11.2 | 100.0 | 66.7 |
| + DiT-L (capacity scaling) | 100.0 | 40.4 | 100.0 | 32.8 | 100.0 | 30.1 | 100.0 | 63.3 |

Every skill reaches ~100% on its training support and drops sharply outside it. Scaling to DiT-L
(~306M parameters, 14× the baseline) does not repair extrapolation.

Reproduce this table with [`paper_runs/table2/`](paper_runs/table2) — see
[Reproducing paper results](#reproducing-paper-results).


## Setup

```bash
git clone https://github.com/AmishSethi/visual-generative-lab.git
cd visual-generative-lab
conda env create -f environment.yml
conda activate vgl
```

Or with pip (assumes a working PyTorch ≥ 2.0 + CUDA install):

```bash
pip install -r requirements.txt
pip install -e .
```

Set the storage root for datasets, checkpoints and results (defaults to `~/vgl-data`):

```bash
export VGL_ROOT=/path/to/scratch     # needs tens of GB for full reproduction
```

Verify:

```bash
python -c "import vgl; from vgl.models import DiT_models_continuous as M; print(vgl.__version__, sorted(M))"
python -m pytest tests/ -q
```

---

## Quickstart

Generate a dataset, train, and evaluate one skill end to end. The paths below are the ones the
paper-locked evaluator expects, so step 3 finds what step 2 produced.

```bash
export VGL_ROOT=/path/to/scratch
T2=$VGL_ROOT/MORE_SEEDS/table2

# 1. build the size dataset (10k images, 64x64, radii 5-20)
python -m paper_runs.table2.generate_canonical_datasets --skills size --output-root $T2/datasets

# 2. train the baseline (single GPU; use --nproc_per_node=N for more)
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --data-path $T2/datasets/size/train --results-dir $T2/results/size/baseline/seed_0 \
  --model DiT-S/2 --image-size 64 --epochs 1000 --global-batch-size 128 --global-seed 0 \
  --radius-embedding-type linear --conditioning-method concat

# 3. evaluate on train / interp / extra with the paper's protocol
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0
```

On a SLURM cluster, `python -m paper_runs.table2.submit_training --skills size --seeds 0 --submit`
writes and submits the same command.

Each skill has its own training entry point, because the conditioning differs:

| skill | script | conditioning |
|---|---|---|
| size | `scripts/train.py` | scalar radius |
| position | `scripts/train_position.py` | 2D (x, y) |
| rotation | `scripts/train_rotation.py` | angle, periodic |
| count | `scripts/train_count.py` | discrete integer |
| multi-skill | `scripts/train_compositional.py` | `--include-properties` |

---

## Repository layout

```
vgl/                       importable library
  models.py                DiT, continuous scalar conditioning (size)
  models_position.py       DiT for 2D position
  models_rotation.py       DiT for angle conditioning
  models_compositional.py  multi-skill DiT
  unet_models*.py          U-Net and SongUNet backbones
  diffusion/               DDPM (adapted from OpenAI ADM)
  flow_matching.py         flow-matching objective
  eval_radius.py           size sampler + rule-based metric
  eval_position.py         position sampler + metric
  eval_rotation.py         rotation sampler + metric
  eval_count.py            count sampler + metric
  reproducibility_utils.py seeding, worker init, run-config logging

scripts/                   training entry points (one per skill)

paper_runs/              exact code that produced each paper table
  table2/                single-skill generalization (Table 2)
  table3/                compositional generalization (Table 3)
  appendix/              experiments behind the appendix: data scaling, resolution, token-matched
                         latent vs pixel, visually complex renders, text conditioning, the
                         text-to-image probe, coverage vs K, and count-metric cross-validation

tests/                   unit tests
docs/                    extended documentation
```

---

## Reproducing paper results

`paper_runs/` is the reproduction layer. Each subdirectory has a `manifest.py` holding the canonical
paths and per-skill configuration, a `submit_*.py` that emits SLURM scripts, and an `evaluate_*.py`
that is the **paper-locked evaluator** — the code that produced the published numbers.

```bash
# Table 2: single-skill generalization, all skills x all variants x 3 seeds
python -m paper_runs.table2.submit_training --skills size position rotation count --seeds 0 1 2
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0
python -m paper_runs.table2.aggregate_table2

# Table 3: compositional generalization
python -m paper_runs.table3.submit_training
```

Point `manifest.py` at your own storage before submitting — the paths there are the ones we used.


---

## Datasets

All datasets are synthetic, generated deterministically from a fixed seed, so they can be rebuilt:

```bash
python paper_runs/table2/generate_canonical_datasets.py --skills size position rotation count
```

Datasets are `ImageFolder`-structured with the skill value encoded in the directory name.

Compositional pairs that include rotation without shape render an arrow (the same polygon as the
single-skill rotation set), and the evaluator recentres the object before reading its angle, since the
rotation detector is not translation invariant (set `VGL_ROTATION_SHAPE=arrow` when evaluating them).

| skill | type | training domain | extrapolation values |
|---|---|---|---|
| size | continuous | radius ∈ [5, 20] px | {1,2,3,4} ∪ {21,…,30} px |
| position | continuous 2D | (x, y) ∈ [−18, 18]² px | L∞ ≥ 4 px outside the square |
| rotation | continuous | angle ∈ [45°, 315°] | {0°,…,44°} ∪ {316°,…,359°} |
| count | discrete | {2,…,7} objects | {0, 1, 8, 9} |
| shape | categorical | 4 classes | — |
| color | categorical | 8 classes | — |

---

## Pretrained checkpoints

The checkpoints behind Table 2 (all skills, all variants, every seed; 128 files, 98 GB) are on the
Hugging Face Hub at [ASethi04/vgl-checkpoints](https://huggingface.co/ASethi04/vgl-checkpoints), laid out as
`table2/<skill>/<variant>/seed_<n>/final.pt` with each run's `run_config.json` beside it:

```bash
huggingface-cli download ASethi04/vgl-checkpoints --include "table2/rotation/baseline/*" --local-dir checkpoints
```

See [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for the mapping from checkpoint to table row.

---

## Evaluation metrics

Every metric is rule-based, with no learned components, so scores are reproducible and interpretable:

| skill | measurement | correct when |
|---|---|---|
| size | Otsu threshold → mask area → r = √(area/π) | IoU ≥ 0.90 vs ground truth |
| position | centroid of the object mask | within 2 px |
| rotation | multi-scale template matching on the arrow | within 5° |
| count | distance transform + watershed, blobs > 30 px | exact match |
| shape | contour-based classification | class match |
| color | nearest-neighbour in RGB | class match |


---

## Citation

```bibtex
@article{sethi2026vgl,
  title   = {Do Diffusion Models Learn to Generalize Basic Visual Skills?},
  author  = {Sethi, Amish and Zeng, Boya and Chai, Wenhao and Liu, Zhuang},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgments

Built on [facebookresearch/DiT](https://github.com/facebookresearch/DiT). The diffusion
implementation is adapted from [OpenAI's ADM](https://github.com/openai/guided-diffusion).

## License

**[CC BY-NC 4.0](LICENSE)** — free to share and adapt with attribution, **non-commercial use only**.

This repository is a derivative of [DiT](https://github.com/facebookresearch/DiT), which Meta
released under CC BY-NC 4.0, so it inherits those terms including the non-commercial restriction.
`vgl/diffusion/` is adapted from [OpenAI ADM](https://github.com/openai/guided-diffusion) and retains
its original MIT license.
