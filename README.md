## Bottleneck Transformers in JAX/Flax

An implementation of <a href="https://arxiv.org/abs/2101.11605">Bottleneck Transformers for Visual Recognition</a>, a hybrid architecture that replaces the spatial convolutions in the final three bottleneck blocks of a ResNet with global self-attention.

The code in this repository is limited to the image classification models and based on the <a href="https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2">authors' official code</a>.

## Known issues

It's worth noting that the models as made available in this repository do not perfectly match the number of parameters as presented in the paper. It is my belief however that the majority of the difference is due to an error in the script used by the authors to count the number of parameters: specifically, section 4.8.4 notes that Squeeze-and-Excite layers are only employed in ResNet bottleneck blocks, while the <a href="https://gist.github.com/aravindsrinivas/e8a9e33425e10ed0c69c1bf726b81495">official script</a> does not correctly take this into account. Updating the script (specifically line 111) leads to an almost perfect match.

The number of parameters are reported for clarity:

| Model | Parameters in this repo | Parameters after updating the script | Parameters in the paper
| :---: | :---: | :---: | :---: |
T3 | 30.4M | 30.4M | 33.5M
T4 | 51.6M | 51.5M | 54.7M
T5 | 69.0M | 68.8M | 75.1M
T6 | 47.8M | 47.7M | 53.9M
T7 | 69.0M | 68.8M | 75.1M

I am currently waiting for feedback from the authors about this issue and will update the repo once I know more about it.

## Install

```bash
$ pip install bottleneck-transformer-flax
```

## Usage

```python
from jax import random
from jax import numpy as jnp
from bottleneck_transformer_flax import BoTNet, BoTNetConfig

#example configuration for BoTNet-S1-128
config = BoTNetConfig(
    stage_sizes = [3, 4, 23, 12],
    num_classes = 1000,
    stride_one = True,
    se_ratio = 0.0625
)

rng = random.PRNGKey(seed=0)
model = BoTNet(config=config)
params = model.init(rng, jnp.ones((1, 256, 256, 3), dtype=config.dtype))
img = random.uniform(rng, (2, 256, 256, 3))
logits, updated_state = model.apply(params, img, mutable=['batch_stats']) # logits.shape is (2, 1000)
```

## Citation

```bibtex
@misc{srinivas2021bottleneck,
    title   = {Bottleneck Transformers for Visual Recognition}, 
    author  = {Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
    year    = {2021},
    eprint  = {2101.11605},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```