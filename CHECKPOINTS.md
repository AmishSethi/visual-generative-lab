# Pretrained checkpoints

Checkpoints are not stored in this repository. The training commands are in the main README; the
checkpoints are provided so you can skip training.

## Download

Hosted on the Hugging Face Hub: **https://huggingface.co/ASethi04/vgl-checkpoints** (model repo, CC BY-NC 4.0).

```bash
pip install -U huggingface_hub
# one cell of Table 2 (rotation baseline, all seeds)
huggingface-cli download ASethi04/vgl-checkpoints --include "table2/rotation/baseline_nz/*" --local-dir checkpoints
# everything (98 GB)
huggingface-cli download ASethi04/vgl-checkpoints --local-dir checkpoints
```

## Layout

```
checkpoints/
  table2/{skill}/{variant}/seed_{n}/final.pt
  table2/{skill}/{variant}/seed_{n}/run_config.json
```

`skill` ∈ {size, position, rotation, count}; `variant` is `baseline`, `sinusoidal`, `rotary`, `adaln`,
`vae`, `flow`, `unet` or `dit_large`. `flow` holds the earlier flow-matching runs, which drew training
times as t = sigmoid(u) (`--flow-use-sigmoid-time`); the Table 2 row is `flow_tfix`. Rotation also has
`baseline_nz` (seeds 0-9), `baseline_nz_warmstart` (seeds 100-104) and
`{sinusoidal,rotary,adaln,vae,flow,unet}_nz`, where `_nz` marks a zero null embedding
(`--null-embedding-type zero`); `rotation/baseline` uses a learnable one.

Not on the Hub: the flow-matching row (`flow_tfix`), the U-Net rotation cell (`unet_sincos`) and
DiT-L size seed 2.

Each file contains `model`, `ema`, `opt`, `train_steps` and `epoch`. Evaluation uses the **EMA** weights.

## Mapping to the results table

| README row | `variant` | notes |
|---|---|---|
| DiT-S/2 (baseline) | `baseline` | linear embedding, concat conditioning, pixel space, DDPM; for rotation the table uses `baseline_nz` (seeds 0-9) |
| + sinusoidal emb. | `sinusoidal` | |
| + rotary emb. | `rotary` | |
| + AdaLN | `adaln` | AdaLN-Zero instead of concatenation |
| + VAE latent | `vae` | `--use-latent-diffusion` |
| + flow matching | `flow_tfix` | `--use-flow-matching`, t ~ U[0, 1] |
| + U-Net (matched #params) | `unet` | parameter-matched to DiT-S/2; for rotation the table uses `unet_sincos` ([sin, cos] angle input); `rotation/unet` takes the angle as a scalar |
| + DiT-L (higher #params) | `dit_large` | ~306M params |

Count checkpoints are trained for 3000 epochs; every other skill uses 1000, except
`rotation/rotary/seed_0` and `rotation/baseline/seed_1`, which stopped at 74,168 and 73,906 of 78,000
steps (about 950 epochs). Runs that resumed after an interruption skip the rest of the interrupted
epoch (and the next one, in `scripts/train.py` and `scripts/train_position.py`), so some end up to a
few hundred steps short.

## Evaluating a checkpoint

Generate that skill's dataset first (see Datasets in the main README); the evaluator reads the
query grid from its metadata.

```bash
python -m paper_runs.table2.evaluate_table2 --skill size --variant baseline --seed 0 \
  --checkpoint checkpoints/table2/size/baseline/seed_0/final.pt
```

## Loading a checkpoint

The constructor arguments are the training arguments saved beside each checkpoint:

```python
import json, torch
from vgl.models import DiT_models_continuous

run = "checkpoints/table2/size/baseline/seed_0"
args = json.load(open(f"{run}/run_config.json"))["args"]
model = DiT_models_continuous[args["model"]](
    input_size=args["image_size"], in_channels=3,
    radius_embedding_type=args["radius_embedding_type"],
    conditioning_method=args["conditioning_method"],
    null_embedding_type=args["null_embedding_type"])
model.load_state_dict(torch.load(f"{run}/final.pt", map_location="cpu")["ema"])
model.eval()
```
