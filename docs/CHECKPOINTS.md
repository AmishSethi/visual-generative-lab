# Pretrained checkpoints

Checkpoints are not stored in this repository. Every result in the paper can be reproduced from
scratch with the commands in the main README; the checkpoints are provided so you can skip training.

## Download

Hosted on the Hugging Face Hub: **https://huggingface.co/ASethi04/vgl-checkpoints** (model repo, CC BY-NC 4.0).

```bash
pip install -U huggingface_hub
# one cell of Table 2 (rotation baseline, all ten seeds)
huggingface-cli download ASethi04/vgl-checkpoints --include "table2/rotation/baseline/*" --local-dir checkpoints
# everything (98 GB)
huggingface-cli download ASethi04/vgl-checkpoints --local-dir checkpoints
```

## Layout

```
checkpoints/
  {skill}/{variant}/seed_{n}/final_{step}.pt
```

`skill` ∈ {size, position, rotation, count}; `variant` is a row of the results table
(`baseline`, `sinusoidal`, `rotary`, `adaln`, `vae`, `flow`, `unet`, `dit_large`).

Each file contains `model`, `ema`, `opt`, `train_steps` and `epoch`. Evaluation uses the **EMA**
weights, which is what the paper reports.

## Mapping to the results table

| README row | `variant` | notes |
|---|---|---|
| DiT-S/2 (baseline) | `baseline` | linear embedding, concat conditioning, pixel space, DDPM |
| + sinusoidal emb. | `sinusoidal` | |
| + rotary emb. | `rotary` | |
| + AdaLN | `adaln` | AdaLN-Zero instead of concatenation |
| + VAE latent | `vae` | `--use-latent-diffusion` |
| + flow matching | `flow` | `--use-flow-matching` |
| + U-Net (capacity-matched) | `unet` | 22.96M params vs DiT-S/2's 22.20M |
| + DiT-L (capacity scaling) | `dit_large` | ~306M params |

Count checkpoints are trained for 3000 epochs; every other skill uses 1000. See the README section
"Count dataset".

## Loading a checkpoint

```python
import torch
from vgl.models import DiT_models_continuous

ckpt = torch.load("checkpoints/size/baseline/seed_0/final_0078000.pt", map_location="cpu")
model = DiT_models_continuous["DiT-S/2"](
    input_size=64, in_channels=3,
    radius_embedding_type="linear", conditioning_method="concat")
model.load_state_dict(ckpt["ema"])   # EMA weights are what the paper evaluates
model.eval()
```
