from __future__ import annotations

from typing import Any, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from hsvl.utils.flax_utils import default_init
from hsvl.utils.mlp import MLP, ResidualMLP


class SurvivalHazardHead(nn.Module):
    """Survival hazard head with conditioned basis selection.

    Outputs:
      logit0: (B,) logit for P(T=0)
      logits: (B, num_bins) hazard logits per bin
    """

    hidden_dims: Sequence[int]
    num_bins: int
    k_basis: int = 512
    num_basis_sets: int = 16
    layer_norm: bool = True
    activation: Any = nn.gelu
    use_residual: bool = True
    num_blocks: Optional[int] = None

    def setup(self) -> None:
        nb = 2 if self.num_blocks is None else int(self.num_blocks)

        if self.use_residual:
            self.backbone = ResidualMLP(
                hidden_dims=self.hidden_dims,
                num_blocks=nb,
                activations=self.activation,
                use_layer_norm=self.layer_norm,
            )
        else:
            self.backbone = MLP(
                hidden_dims=self.hidden_dims,
                activations=self.activation,
                activate_final=True,
                layer_norm=self.layer_norm,
            )

        self.p0 = nn.Dense(1, kernel_init=default_init(), name="p0")
        self.basis_selector = nn.Dense(
            self.num_basis_sets,
            kernel_init=default_init(),
            name="basis_selector",
        )
        self.coeffs_layer = nn.Dense(
            self.k_basis,
            kernel_init=default_init(),
            name="coeffs",
        )

        self.basis_library = self.param(
            "basis_library",
            nn.initializers.normal(stddev=0.02),
            (self.num_basis_sets, self.num_bins, self.k_basis),
        )
        self.time_bias = self.param(
            "time_bias",
            nn.initializers.zeros,
            (self.num_bins,),
        )

    def __call__(self, z: jnp.ndarray):
        h = self.backbone(z)

        logit0 = self.p0(h).squeeze(-1)
        w = nn.softmax(self.basis_selector(h), axis=-1)
        c = self.coeffs_layer(h)

        tmp = jnp.einsum("bk,shk->bsh", c, self.basis_library)
        logits = jnp.einsum("bs,bsh->bh", w, tmp)
        logits = logits + self.time_bias[None, :]
        return logit0, logits


class GCSurvivalValue(nn.Module):
    """Goal-conditioned survival value model.

    Returns (head1(z), head2(z)) when ensemble=True, else (head(z), head(z)).
    Each head returns (logit0, logits).
    """

    hidden_dims: Sequence[int]
    num_bins: int
    k_basis: int = 512
    num_basis_sets: int = 16
    layer_norm: bool = True
    gc_encoder: Optional[nn.Module] = None
    activation: Any = nn.gelu
    ensemble: bool = True
    use_residual: bool = True
    num_blocks: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
    ):
        if self.gc_encoder is not None:
            z_enc = self.gc_encoder(observations, goals)
        else:
            z_enc = jnp.concatenate([observations, goals], axis=-1)

        z = jnp.concatenate([z_enc, actions], axis=-1) if actions is not None else z_enc

        def make_head(name: str):
            return SurvivalHazardHead(
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                k_basis=self.k_basis,
                num_basis_sets=self.num_basis_sets,
                layer_norm=self.layer_norm,
                activation=self.activation,
                use_residual=self.use_residual,
                num_blocks=self.num_blocks,
                name=name,
            )

        if self.ensemble:
            head1 = make_head("head1")
            head2 = make_head("head2")
            return head1(z), head2(z)

        head = make_head("head")
        out = head(z)
        return out, out
