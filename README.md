# Visual Generative Lab (VGL)

Code for **[Do Diffusion Models Learn to Generalize Basic Visual Skills?](https://arxiv.org/abs/2611.22427)** (NeurIPS 2026)

[Amish Sethi](https://amishsethi.github.io/), [Boya Zeng](https://boyazeng.github.io/), [Wenhao Chai](https://wenhaochai.com/), [Zhuang Liu](https://www.cs.princeton.edu/~zhuangl) · Princeton University

[Project page](https://amishsethi.github.io/visual-generative-lab/) · [Datasets](#datasets) · [Pretrained checkpoints](#pretrained-checkpoints)

Diffusion models are usually evaluated with distribution-level scores (FID, IS) or human preference,
neither of which tells you whether a model has *learned a visual skill* or is reproducing training
patterns. VGL trains diffusion models from scratch on synthetic data where every skill value is
controlled by construction, then measures generalization with rule-based metrics that have no learned
components.

Six skills — **size**, **position**, **rotation**, **count**, **shape**, **color**. The four numeric
skills are evaluated on up to three splits (count has no interpolation split):

| split | meaning |
|---|---|
| **train** | new samples for skill values that appear in training |
| **interp** | values inside the training range but never seen |
| **extra** | values outside the training range entirely |

---

## Findings

1. **Extrapolation is limited.** Every skill fits its training support nearly perfectly but
   extrapolates only slightly beyond it, and rotation extrapolation accuracy varies widely across
   seeds.
2. **The number of distinct combinations matters more than dataset size.** At a fixed image budget,
   spreading samples across more unique skill combinations beats concentrating them on fewer.
3. **Skills mostly fail independently, except position.** An out-of-range position can also lower
   accuracy on the skills that stay in range.

---

## Results

Accuracy (%) on the four numeric skills, mean over seeds.
Thresholds are listed under [Evaluation metrics](#evaluation-metrics).

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">size</th>
      <th colspan="2">pos</th>
      <th colspan="2">rot</th>
      <th colspan="2">count</th>
    </tr>
    <tr>
      <th>train</th><th>extra</th>
      <th>train</th><th>extra</th>
      <th>train</th><th>extra</th>
      <th>train</th><th>extra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DiT-S/2 (baseline)</strong></td>
      <td>100.0</td><td>51.5</td>
      <td>100.0</td><td>29.7</td>
      <td>100.0</td><td>68.0</td>
      <td>100.0</td><td>52.5</td>
    </tr>
    <tr>
      <td>+ sinusoidal emb.</td>
      <td>100.0</td><td>7.1</td>
      <td>100.0</td><td>0.0</td>
      <td>100.0</td><td>26.6</td>
      <td>98.6</td><td>20.4</td>
    </tr>
    <tr>
      <td>+ rotary emb.</td>
      <td>100.0</td><td>9.9</td>
      <td>100.0</td><td>0.0</td>
      <td>100.0</td><td>38.1</td>
      <td>100.0</td><td>36.3</td>
    </tr>
    <tr>
      <td>+ AdaLN</td>
      <td>100.0</td><td>31.0</td>
      <td>100.0</td><td>61.5</td>
      <td>100.0</td><td>58.5</td>
      <td>100.0</td><td>72.5</td>
    </tr>
    <tr>
      <td>+ VAE latent</td>
      <td>99.1</td><td>7.1</td>
      <td>100.0</td><td>0.0</td>
      <td>100.0</td><td>21.2</td>
      <td>93.3</td><td>48.8</td>
    </tr>
    <tr>
      <td>+ flow matching</td>
      <td>100.0</td><td>47.3</td>
      <td>100.0</td><td>27.8</td>
      <td>100.0</td><td>51.0</td>
      <td>100.0</td><td>24.2</td>
    </tr>
    <tr>
      <td>+ U-Net (matched #params)</td>
      <td>100.0</td><td>21.2</td>
      <td>100.0</td><td>42.7</td>
      <td>100.0</td><td>11.2</td>
      <td>99.7</td><td>79.6</td>
    </tr>
    <tr>
      <td>+ DiT-L (higher #params)</td>
      <td>100.0</td><td>44.5</td>
      <td>100.0</td><td>42.5</td>
      <td>100.0</td><td>62.0</td>
      <td>100.0</td><td>72.1</td>
    </tr>
  </tbody>
</table>

The baseline reaches ~100% on every skill's training support, and only rotation has seeds that
extrapolate across its whole out-of-range set. Scaling to DiT-L (~306M parameters, 14× the baseline) improves count and position
extrapolation but not size or rotation.

Reproduce this table with [`paper_runs/table2/`](paper_runs/table2) — see
[Reproducing paper results](#reproducing-paper-results).


## Setup

```bash
git clone https://github.com/AmishSethi/visual-generative-lab.git
cd visual-generative-lab
conda env create -f environment.yml
conda activate vgl
pip install -e .
```

Or with pip (install PyTorch 2.5.1 and torchvision 0.20.1 for your CUDA version first):

```bash
pip install -r requirements.txt
pip install -e .
```

The text-encoder and text-to-image appendix scripts (`paper_runs/appendix/build_text_tables.py`,
`t2i_*.py`) run in a separate environment: `pip install -r requirements-appendix.txt`.

Set the storage root for datasets, checkpoints and results (defaults to `~/vgl-data`):

```bash
export VGL_ROOT=/path/to/scratch     # needs tens of GB for full reproduction
```

Verify:

```bash
python -c "import vgl; from vgl.models import DiT_models_continuous as M; print(vgl.__version__, sorted(M))"
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
  --data-path $T2/datasets/size --results-dir $T2/results/size/baseline/seed_0 \
  --model DiT-S/2 --image-size 64 --epochs 1000 --global-batch-size 128 --global-seed 0 \
  --num-workers 4 --radius-embedding-type linear --conditioning-method concat

# 3. evaluate on train / interp / extra with the paper's protocol
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0
```

On a SLURM cluster, `python -m paper_runs.table2.submit_training --skills size --variants baseline --seeds 0 --submit`
writes and submits the same command.

Each skill has its own training entry point, because the conditioning differs:

| skill | script | conditioning |
|---|---|---|
| size | `scripts/train.py` | scalar radius |
| position | `scripts/train_position.py` | 2D (x, y) |
| rotation | `scripts/train_rotation.py` | angle, periodic |
| count | `scripts/train_count.py` | discrete integer |
| multi-skill | `scripts/train_compositional.py` | `--include-properties` |

In file names and flags the size skill is called radius (e.g. `--radius-embedding-type`).
The scripts' own defaults are not the paper's settings; use the commands above or the
`paper_runs/*/submit_training.py` scripts, which pass every flag.

---

## Repository layout

```
vgl/                       importable library
  models.py                DiT, continuous scalar conditioning (size)
  models_position.py       DiT for 2D position
  models_rotation.py       DiT for angle conditioning
  models_compositional.py  multi-skill DiT
  unet_models*.py          U-Net backbones
  diffusion/               DDPM (adapted from OpenAI ADM)
  flow_matching.py         flow-matching objective
  eval_radius.py           size sampler + rule-based metric
  eval_position.py         position sampler + metric
  eval_rotation.py         rotation sampler + metric
  eval_compositional.py    multi-skill test sets (and the metrics of `paper_runs/table3/evaluate_table3.py`)
  reproducibility_utils.py seeding, worker init, run-config logging

scripts/                   training entry points (one per skill) and the compositional dataset generators

paper_runs/              code behind the paper's tables
  table2/                single-skill generalization (Table 2)
  table3/                compositional generalization (Table 3, Figures 4(b) and 5)
  appendix/              experiments behind the appendix: data scaling, resolution, token-matched
                         latent vs pixel, visually complex renders, text conditioning, the
                         text-to-image probe, coverage vs K, and the three- and four-skill splits
```

---

## Reproducing paper results

Open a table or figure below for the commands we used. Run them from the repository root
with `VGL_ROOT` set as described in [Setup](#setup). The scripts in [`paper_runs/`](paper_runs/)
read the dataset paths and experiment settings from each directory’s `manifest.py`.

<details>
<summary><strong>Table 2 · How well does each skill extrapolate?</strong></summary>

Generate the four datasets and submit the training jobs. Most runs use three seeds. For the rotation baseline (`baseline_nz`), we use ten. To preview the Slurm scripts, leave off `--submit`.

```bash
python -m paper_runs.table2.generate_canonical_datasets --skills size position rotation count
python -m paper_runs.table2.submit_training --skills size position rotation count --seeds 0 1 2 --submit
python -m paper_runs.table2.submit_baseline_nz_extra_seeds --seeds 0 1 2 3 4 5 6 7 8 9 --submit
```

After training, evaluate each skill, variant, and seed. These examples cover the size and rotation baselines and DiT-L on position. For DiT-L and U-Net on position, we use a fixed subset of 100 conditions for training-range and interpolation tests and the full extrapolation set.

```bash
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0
python -m paper_runs.table2.evaluate_table2 --skill rotation --variant baseline_nz --seed 0
python -m paper_runs.table2.evaluate_table2 --skill position --variant dit_large --seed 0 \
  --conditions-file paper_runs/table2/position_subsample_100.json
```

Once you have scored the runs, combine the results for Table 2.

```bash
python -m paper_runs.table2.aggregate_table2
```

</details>

<details>
<summary><strong>Table 3 · Can the model combine two skills?</strong></summary>

Generate the datasets and train each model with six seeds.

```bash
python -m paper_runs.table3.generate_datasets
python -m paper_runs.table3.submit_training --seeds 0 1 2 3 4 5 --submit
```

For each skill pair, coverage level, and seed, draw 20 samples per combination from the final checkpoint’s online weights. Here is the color and shape run at 75% coverage with seed 0.

```bash
python -m paper_runs.table3.sample_table3 --pair color_shape --coverage 75 --seed 0
mkdir -p $VGL_ROOT/MORE_SEEDS/table3/scores
python -m paper_runs.table3.score_table3 --samples $VGL_ROOT/MORE_SEEDS/table3/samples/color_shape/cov75/seed_0 \
  --out $VGL_ROOT/MORE_SEEDS/table3/scores/color_shape_cov75_seed0.json
```

After scoring the runs with `score_table3.py`, combine their scores.

```bash
python -m paper_runs.table3.aggregate_scores
```

</details>

<details>
<summary><strong>Figures 4(b) and 5 · Color and shape at a fixed image budget</strong></summary>

Generate a dataset and train on it. This example uses 15,000 images and 25% of the possible color and shape pairs. The paper compares 25%, 50%, and 75% coverage at budgets of 5,000, 10,000, and 15,000 images.

```bash
python scripts/generate_color_shape_dataset.py --output-dir $VGL_ROOT/datasets/color_shape_composition_cov25_samples15k \
  --coverage 25 --train-samples 15000
torchrun --standalone --nproc_per_node=1 scripts/train_compositional.py --include-properties color shape \
  --data-path $VGL_ROOT/datasets/color_shape_composition_cov25_samples15k/train \
  --results-dir $VGL_ROOT/fig4/results/cov25_samples15k_seed0 --global-seed 0
```

Set `CHECKPOINT` to the path of the `checkpoints/final_*.pt` file from that run. Draw 20 samples per held-out combination using the exponential moving average (EMA) weights, then score them.

```bash
python -m paper_runs.table3.sample_spec --checkpoint "$CHECKPOINT" \
  --dataset $VGL_ROOT/datasets/color_shape_composition_cov25_samples15k --out $VGL_ROOT/fig4/samples/cov25_samples15k_seed0
python -m paper_runs.table3.score_spec --samples $VGL_ROOT/fig4/samples/cov25_samples15k_seed0 \
  --out $VGL_ROOT/fig4/scores/cov25_samples15k_seed0.json
```

</details>

<details>
<summary><strong>Figure 6 · How does guidance scale affect size?</strong></summary>

Train the size baseline with conditioning dropout of 0.1 and seed 0.

```bash
python -m paper_runs.appendix.submit_training --families cfg --seeds 0 --submit
```

After training finishes, sample the Table 2 interpolation and extrapolation radii at guidance scales from 1 to 100. Then score the samples and plot the results.

```bash
python -m paper_runs.appendix.cfg_sample
python -m paper_runs.appendix.cfg_score
python -m paper_runs.appendix.cfg_aggregate
python -m paper_runs.appendix.cfg_plot
```

</details>

<details>
<summary><strong>Table 8 · How does accuracy change with distance outside the training range?</strong></summary>

Start with the Table 2 baseline scores. This script groups the requested values by how far they lie outside the training range, measured as a fraction of the range’s width.

```bash
python -m paper_runs.appendix.matched_distance
```

</details>

<details>
<summary><strong>Appendix F · What happens when one skill goes out of range?</strong></summary>

The example below trains a model on size, position, and shape at 75% coverage, with 53 images per training combination. The four-skill experiment adds color and uses 9 images per combination.

```bash
python scripts/generate_compositional_dataset_coverage.py --coverage 0.75 --include-properties radius position shape \
  --samples-per-combination 53 --output-dir $VGL_ROOT/appendix/multi_skill/data_3skill
torchrun --standalone --nproc_per_node=1 scripts/train_compositional.py --include-properties radius position shape \
  --data-path $VGL_ROOT/appendix/multi_skill/data_3skill/train --results-dir $VGL_ROOT/appendix/multi_skill/results/3skill_seed0
```

Set `CHECKPOINT` to the path of the `checkpoints/final_*.pt` file from the three-skill run. Build the test splits, draw 20 samples per combination with the EMA weights, and score them.

```bash
python -m paper_runs.appendix.multi_skill_splits --out-dir $VGL_ROOT/appendix/multi_skill
python -m paper_runs.table3.sample_spec --checkpoint "$CHECKPOINT" \
  --splits $VGL_ROOT/appendix/multi_skill/3skill.json --out $VGL_ROOT/appendix/multi_skill/samples/3skill_seed0
python -m paper_runs.table3.score_spec --samples $VGL_ROOT/appendix/multi_skill/samples/3skill_seed0 \
  --out $VGL_ROOT/appendix/multi_skill/scores/3skill_seed0.json
```

</details>

---

## Datasets

All datasets are synthetic. The Table 2 datasets are rebuilt deterministically with:

```bash
python paper_runs/table2/generate_canonical_datasets.py --skills size position rotation count
```

For the Table 3 datasets the seed fixes the training combinations but not the colour jitter or pixel
noise.

Datasets are `ImageFolder`-structured with the skill value encoded in the directory name.

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

[Browse the checkpoints on Hugging Face](https://huggingface.co/ASethi04/vgl-checkpoints/tree/main).

With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), download the rotation baseline for seed 0 and its configuration in one command.

```bash
hf download ASethi04/vgl-checkpoints --include "table2/rotation/baseline_nz/seed_0/*" --local-dir checkpoints
```

To download the collection, run the same command without a folder filter.

```bash
hf download ASethi04/vgl-checkpoints --local-dir checkpoints
```

---

## Evaluation metrics

Every metric is rule-based, with no learned components, so scores are reproducible and interpretable:

| skill | measurement | correct when |
|---|---|---|
| size | Otsu threshold → mask area → r = √(area/π) | IoU ≥ 0.90 vs ground truth |
| position | centroid of the object mask | within 2 px |
| rotation | template matching on the arrow, coarse to fine over angle | within 5° |
| count | Otsu mask (blank frames, with classes < 20 grey levels apart, count 0), distance transform + watershed, blobs > 30 px | exact match |
| shape | template matching against shape renders (best fit over orientation) | class match |
| color | nearest-neighbour in RGB | class match |

The size, position, rotation and count rows are the single-skill (Table 2) metrics. Table 3 scores
every sample with `paper_runs/table3/score_table3.py`, whose docstring gives its rules (count, for
example, splits touching objects at necks).


---

## License

The code is released under CC BY-NC 4.0 ([LICENSE](LICENSE)), inherited from DiT; `vgl/diffusion/` keeps
the MIT license of ADM ([vgl/diffusion/LICENSE](vgl/diffusion/LICENSE)). The checkpoints are released
under CC BY-NC 4.0.

## Citation

```bibtex
@inproceedings{sethi2026vgl,
  title     = {Do Diffusion Models Learn to Generalize Basic Visual Skills?},
  author    = {Sethi, Amish and Zeng, Boya and Chai, Wenhao and Liu, Zhuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgments

Built on [facebookresearch/DiT](https://github.com/facebookresearch/DiT). The diffusion
implementation is adapted from [OpenAI's ADM](https://github.com/openai/guided-diffusion).
