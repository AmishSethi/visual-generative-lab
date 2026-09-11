# Pretrained checkpoints

Checkpoints are not stored in this repository. Every result in the paper can be reproduced from
scratch with the commands in the main README; the checkpoints are provided so you can skip training.

## Download

Hosted on the Hugging Face Hub: **https://huggingface.co/ASethi04/vgl-checkpoints** (model repo, CC BY-NC 4.0).

```bash
pip install -U huggingface_hub
# one cell of Table 2 (rotation baseline, all seeds)
huggingface-cli download ASethi04/vgl-checkpoints --include "table2/rotation/baseline/*" --local-dir checkpoints
# everything (98 GB)
huggingface-cli download ASethi04/vgl-checkpoints --local-dir checkpoints
```

## Layout

```
checkpoints/
  table2/{skill}/{variant}/seed_{n}/final.pt
  table2/{skill}/{variant}/seed_{n}/run_config.json
```

`skill` ∈ {size, position, rotation, count}; `variant` is a row of the results table
(`baseline`, `sinusoidal`, `rotary`, `adaln`, `vae`, `flow`, `unet`, `dit_large`).

Each file contains `model`, `ema`, `opt`, `train_steps` and `epoch`. Evaluation uses the **EMA** weights.

## Mapping to the results table

| README row | `variant` | notes |
|---|---|---|
| DiT-S/2 (baseline) | `baseline` | linear embedding, concat conditioning, pixel space, DDPM |
| + sinusoidal emb. | `sinusoidal` | |
| + rotary emb. | `rotary` | |
| + AdaLN | `adaln` | AdaLN-Zero instead of concatenation |
| + VAE latent | `vae` | `--use-latent-diffusion` |
| + flow matching | `flow` | `--use-flow-matching` |
| + U-Net (capacity-matched) | `unet` | parameter-matched to DiT-S/2 |
| + DiT-L (capacity scaling) | `dit_large` | ~306M params |

Count checkpoints are trained for 3000 epochs; every other skill uses 1000.

## Evaluating a checkpoint

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
