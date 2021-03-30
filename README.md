## Bottleneck Transformers in JAX/Flax

An implementation of <a href="https://arxiv.org/abs/2101.11605">Bottleneck Transformers for Visual Recognition</a>, a powerful hybrid architecture that combines a ResNet-like architecture with global relative position self-attention.

The code in this repository is limited to the image classification models and based on the <a href="https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2">authors' official code</a>.

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
    num_classes = 1000
)

rng = random.PRNGKey(seed=0)
model = BoTNet(config=config)
params = model.init(rng, jnp.ones((1, 256, 256, 3), dtype=config.dtype))
img = random.uniform(rng, (2, 256, 256, 3))
logits, updated_state = model.apply(params, img, mutable=['batch_stats']) # logits.shape is (2, 1000)
```

## Example Configurations

A BoTNet configuration has the following arguments:

```python
class BoTNetConfig:
    stage_sizes: Sequence[int]                                          # Stages sizes (as in Table 13)
    num_classes: int = 1000                                             # Number of classes
    stride_one: bool = True                                             # Whether the model is a BoTNet-S1
    se_ratio: float = 0.0625                                            # How much to squeeze
    activation_fn: ModuleDef = nn.swish                                 # Activation function
    num_heads: int = 4                                                  # Number of heads in multi head self attention
    head_dim: int = 128                                                 # Head dimension in multi head self attention
    initial_filters: int = 64                                           # Resnet stem output channels
    projection_factor: int = 4                                          # Ratio between block output and input channels
    bn_momentum: float = 0.9                                            # Batch normalization momentum
    bn_epsilon: float = 1e-5                                            # Batch normalization epsilon
    dtype: jnp.dtype = jnp.float32                                      # dtype of the computation
    precision: Any = jax.lax.Precision.DEFAULT                          # Numerical precision of the computation
    kernel_init: Callable = initializers.he_uniform()                   # Initializer function for the weight matrix
    bias_init: Callable = initializers.normal(stddev=1e-6)              # Initializer function for the bias
    posemb_init: Callable = initializers.normal(stddev=head_dim**-0.5)  # Initializer function for positional embeddings
```

Provided below are example configurations for all BoTNets.

### BoTNet T3

```python
config = BoTNetConfig(
    stage_sizes = [3, 4, 6, 6],
    num_classes = 1000
)
```

### BoTNet T4

```python
config = BoTNetConfig(
    stage_sizes = [3, 4, 23, 6],
    num_classes = 1000
)
```

### BoTNet T5

```python
config = BoTNetConfig(
    stage_sizes = [3, 4, 23, 12],
    num_classes = 1000
)
```

### BoTNet T6

```python
config = BoTNetConfig(
    stage_sizes = [3, 4, 6, 12],
    num_classes = 1000
)
```

### BoTNet T7

```python
config = BoTNetConfig(
    stage_sizes = [3, 4, 23, 12],
    num_classes = 1000
)
```

## Known issues

It's worth noting that the models as made available in this repository do not perfectly match the number of parameters as presented in the paper. The majority of the difference can however be explained by what I believe is an error in the script used by the authors to count the parameters: specifically, section 4.8.4 notes that Squeeze-and-Excite layers are only employed in ResNet bottleneck blocks, while the <a href="https://gist.github.com/aravindsrinivas/e8a9e33425e10ed0c69c1bf726b81495">official script</a> does not correctly take this into account. Updating the script (specifically line 111) leads to an almost perfect match.

The number of parameters for each model is reported for clarity:

| Model | This implementation | Updated script | Paper
| :---: | :---: | :---: | :---: |
T3 | 30.4M | 30.4M | 33.5M
T4 | 51.6M | 51.5M | 54.7M
T5 | 69.0M | 68.8M | 75.1M
T6 | 47.8M | 47.7M | 53.9M
T7 | 69.0M | 68.8M | 75.1M

I am currently waiting for feedback from the authors about this issue and will update the repo as soon as possible.

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