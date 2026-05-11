from __future__ import annotations

from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from hsvl.utils.flax_utils import default_init
from hsvl.utils.mlp import MLP, ResidualMLP, TransformedWithMode


class GCActor(nn.Module):
    """Goal-conditioned Gaussian actor."""

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: Optional[nn.Module] = None
    activation: Any = nn.gelu
    use_residual: bool = False
    num_blocks: Optional[int] = None
    layer_norm: bool = False

    def setup(self):
        if self.use_residual:
            self.actor_net = ResidualMLP(
                hidden_dims=self.hidden_dims,
                num_blocks=self.num_blocks,
                activations=self.activation,
                use_layer_norm=self.layer_norm,
            )
        else:
            self.actor_net = MLP(
                hidden_dims=self.hidden_dims,
                activations=self.activation,
                activate_final=True,
                layer_norm=self.layer_norm,
            )
        self.mean_net = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale),
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale),
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds",
                    nn.initializers.zeros,
                    (self.action_dim,),
                )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature,
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution,
                distrax.Block(distrax.Tanh(), ndims=1),
            )

        return distribution
