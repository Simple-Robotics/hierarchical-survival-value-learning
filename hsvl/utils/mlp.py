from typing import Any, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from hsvl.utils.flax_utils import default_init


class Identity(nn.Module):
    def __call__(self, x):
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class ResidualMLP(nn.Module):
    """Residual MLP.

    First projects input to `hidden_dims[0]` so the skip path is always identity-compatible.
    Pre-LN inside each block.
    """

    hidden_dims: Sequence[int] = (512, 512, 512)
    num_blocks: int = 3
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        if len(self.hidden_dims) == 0:
            return x

        width = int(self.hidden_dims[0])
        if x.shape[-1] != width:
            x = nn.Dense(width, kernel_init=self.kernel_init, name="in_proj")(x)
        for b in range(self.num_blocks):
            h = x
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"block{b}_preln")(h)
            for i, dim in enumerate(self.hidden_dims):
                is_last = i == len(self.hidden_dims) - 1
                out_dim = width if is_last else int(dim)
                h = nn.Dense(out_dim, kernel_init=self.kernel_init, name=f"block{b}_dense{i}")(h)
                if not is_last:
                    h = self.activations(h)
            x = x + h

        if self.use_layer_norm:
            x = nn.LayerNorm(name="final_ln")(x)
        return x


class LengthNormalize(nn.Module):
    """Normalize last dim to length sqrt(dim)."""

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution that supports `.mode()`."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())
