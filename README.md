# Visual Generative Lab (VGL)

Code for **"Do Diffusion Models Learn to Generalize Basic Visual Skills?"** (NeurIPS 2026)

Amish Sethi, Boya Zeng, Wenhao Chai, Zhuang Liu · Princeton University

[Datasets](#datasets) · [Pretrained checkpoints](#pretrained-checkpoints) · [License](#license)

Diffusion models are usually evaluated with distribution-level scores (FID, IS) or human preference,
neither of which tells you whether a model has *learned a visual skill* or is reproducing training
patterns. VGL trains diffusion models from scratch on synthetic data where every skill value is
controlled by construction, then measures generalization with rule-based metrics that have no learned
components.

Six skills — **size**, **position**, **rotation**, **count**, **shape**, **color** — each evaluated on
three splits:

| split | meaning |
|---|---|
| **train** | new samples for skill values that appear in training |
| **interp** | values inside the training range but never seen |
| **extra** | values outside the training range entirely |

---

## Findings

1. **Extrapolation is limited; rotation is the exception.** Every skill fits its training support
   nearly perfectly, and only rotation continues to work far outside it.
2. **Coverage beats dataset size.** At a fixed image budget, spreading samples across more unique
   skill combinations beats concentrating them on fewer.
3. **Skills are learned jointly.** An out-of-range request in one skill can degrade the others, even
   when those remain in range.

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
      <td>100.0</td><td>31.0</td>
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
      <td>100.0</td><td>37.6</td>
      <td>100.0</td><td>36.3</td>
    </tr>
    <tr>
      <td>+ AdaLN</td>
      <td>100.0</td><td>31.0</td>
      <td>100.0</td><td>63.6</td>
      <td>100.0</td><td>58.5</td>
      <td>100.0</td><td>72.5</td>
    </tr>
    <tr>
      <td>+ VAE latent</td>
      <td>82.8</td><td>5.6</td>
      <td>99.8</td><td>0.1</td>
      <td>100.0</td><td>21.2</td>
      <td>71.7</td><td>43.8</td>
    </tr>
    <tr>
      <td>+ flow matching</td>
      <td>100.0</td><td>47.3</td>
      <td>100.0</td><td>26.6</td>
      <td>100.0</td><td>51.0</td>
      <td>100.0</td><td>24.2</td>
    </tr>
    <tr>
      <td>+ U-Net (matched #params)</td>
      <td>100.0</td><td>21.2</td>
      <td>100.0</td><td>39.9</td>
      <td>100.0</td><td>11.2</td>
      <td>99.7</td><td>79.6</td>
    </tr>
    <tr>
      <td>+ DiT-L (higher #params)</td>
      <td>100.0</td><td>44.5</td>
      <td>100.0</td><td>34.7</td>
      <td>100.0</td><td>62.0</td>
      <td>100.0</td><td>72.1</td>
    </tr>
  </tbody>
</table>

The baseline reaches ~100% on every skill's training support, and only rotation extrapolates well
beyond it. Scaling to DiT-L (~306M parameters, 14× the baseline) improves count extrapolation but
not size or rotation.

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
  reproducibility_utils.py seeding, worker init, run-config logging

scripts/                   training entry points (one per skill)

paper_runs/              exact code that produced each paper table
  table2/                single-skill generalization (Table 2)
  table3/                compositional generalization (Table 3)
  appendix/              experiments behind the appendix: data scaling, resolution, token-matched
                         latent vs pixel, visually complex renders, text conditioning, the
                         text-to-image probe, and coverage vs K
```

---

## Reproducing paper results

`paper_runs/` is the reproduction layer. Each subdirectory has a `manifest.py` holding the canonical
paths and per-skill configuration, `submit_*.py` scripts that emit SLURM jobs, and the **paper-locked
evaluators** (`evaluate_table2.py`, `score_table3.py`, and the `eval_*.py` scripts under `appendix/`),
the code that produced the published numbers.

```bash
# Table 2: single-skill generalization, all skills x all variants x 3 seeds
python -m paper_runs.table2.generate_canonical_datasets --skills size position rotation count
# without --submit, the submit_* scripts only write the Slurm scripts
python -m paper_runs.table2.submit_training --skills size position rotation count --seeds 0 1 2 --submit
# the rotation baseline cell is baseline_nz over seeds 0-9
python -m paper_runs.table2.submit_baseline_nz_extra_seeds --seeds 0 1 2 3 4 5 6 7 8 9 --submit
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0
python -m paper_runs.table2.evaluate_table2 --skill rotation --variant baseline_nz --seed 0
# DiT-L and U-Net position: train and interp on a fixed random 100-condition subset, extra in full
python -m paper_runs.table2.evaluate_table2 --skill position --variant dit_large --seed 0 \
  --conditions-file paper_runs/table2/position_subsample_100.json
python -m paper_runs.table2.aggregate_table2

# Table 3: compositional generalization, each cell over seeds 0-5
python -m paper_runs.table3.generate_datasets
python -m paper_runs.table3.submit_training --seeds 0 1 2 3 4 5 --submit
python -m paper_runs.table3.sample_table3 --pair color_shape --coverage 75 --seed 0
mkdir -p $VGL_ROOT/MORE_SEEDS/table3/scores
python -m paper_runs.table3.score_table3 --samples $VGL_ROOT/MORE_SEEDS/table3/samples/color_shape/cov75/seed_0 \
  --out $VGL_ROOT/MORE_SEEDS/table3/scores/color_shape_cov75_seed0.json
python -m paper_runs.table3.aggregate_scores
```

Everything is written under `$VGL_ROOT`; each `manifest.py` reads that variable.

Table 3 samples the online (non-EMA) weights of each run's final checkpoint, 20 samples per
combination, and scores every sample on its own. `evaluate_table3.py` and `aggregate_table3.py` are an
earlier scorer; the paper's Table 3 uses `score_table3.py`.


---

## Datasets

All datasets are synthetic, generated deterministically from a fixed seed, so they can be rebuilt:

```bash
python paper_runs/table2/generate_canonical_datasets.py --skills size position rotation count
```

Datasets are `ImageFolder`-structured with the skill value encoded in the directory name.

Compositional pairs that include rotation without shape render an arrow; set `VGL_ROTATION_SHAPE=arrow` when evaluating them.

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

The Table 2 checkpoints are on the Hugging Face Hub at
[ASethi04/vgl-checkpoints](https://huggingface.co/ASethi04/vgl-checkpoints) (128 files, 98 GB), except
those of the flow-matching row, the U-Net rotation cell and DiT-L size seed 2. Each run is at
`table2/<skill>/<variant>/seed_<n>/final.pt` with its `run_config.json` beside it:

```bash
huggingface-cli download ASethi04/vgl-checkpoints --include "table2/rotation/baseline_nz/*" --local-dir checkpoints
```

To evaluate a downloaded checkpoint with the paper's protocol, generate that skill's dataset first
(see [Datasets](#datasets); the evaluator reads the query grid from its metadata), then point the
evaluator at it:

```bash
python -m paper_runs.table2.evaluate_table2 --skill rotation --variant baseline_nz --seed 0 \
  --checkpoint checkpoints/table2/rotation/baseline_nz/seed_0/final.pt
```

See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the mapping from checkpoint to table row.

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
