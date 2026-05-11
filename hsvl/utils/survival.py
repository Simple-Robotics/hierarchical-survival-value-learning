from __future__ import annotations

import jax
import jax.numpy as jnp


def get_bin_discount_factors(bins: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Compute D_k = sum_{t=b_k}^{b_{k+1}-1} gamma^t."""
    bins = jnp.asarray(bins, dtype=jnp.int32)
    starts = bins[:-1].astype(jnp.float32)
    lengths = (bins[1:] - bins[:-1]).astype(jnp.float32)

    if gamma == 1.0:
        return lengths

    gamma_start = jnp.power(gamma, starts)
    gamma_len = jnp.power(gamma, lengths)
    return gamma_start * (1.0 - gamma_len) / (1.0 - gamma)


def create_perbin_value_fn(
    bins: jnp.ndarray,
    gamma: float,
    add_tail: bool = True,
):
    """Per-bin hazard model value under S(t)=S(b_k) within each bin."""
    bins = jnp.asarray(bins, dtype=jnp.int32)
    K = int(bins.shape[0] - 1)
    H = int(bins[-1])

    bin_discounts = get_bin_discount_factors(bins, gamma)  # (K,)

    @jax.jit
    def value_from_logits(logit0: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
        B, K_logits = logits.shape
        assert K_logits == K

        log1mp0 = jax.nn.log_sigmoid(-logit0)
        log1mh = jax.nn.log_sigmoid(-logits)

        prefix = jnp.cumsum(log1mh, axis=-1)
        prefix_before = jnp.concatenate(
            [jnp.zeros((B, 1), dtype=logits.dtype), prefix[:, :-1]],
            axis=1,
        )

        logS_start = log1mp0[:, None] + prefix_before
        S_start = jnp.exp(logS_start)

        v = -jnp.sum(S_start * bin_discounts[None, :], axis=-1)

        if add_tail and gamma < 1.0:
            logS_end = log1mp0 + prefix[:, -1]
            S_end = jnp.exp(logS_end)
            v = v - (jnp.power(gamma, H) * S_end) / (1.0 - gamma)

        return v

    return value_from_logits


def create_perbin_nll_fn(bins: jnp.ndarray):
    """Per-bin hazard NLL.

    Convention:
      tau in [0..H]; tau=0 means event at time 0 via p0; tau>0: event belongs to bin containing tau (side='right').
      censor in [0..H]; censor=c credits survival through bins [0..edge-1] where edge = searchsorted(bins, c, 'left').
    """
    bins = jnp.asarray(bins, dtype=jnp.int32)
    K = int(bins.shape[0] - 1)

    @jax.jit
    def nll_from_logits(
        logit0: jnp.ndarray,
        logits: jnp.ndarray,
        is_event: jnp.ndarray,
        tau: jnp.ndarray,
        censor: jnp.ndarray,
    ) -> jnp.ndarray:
        B, K_logits = logits.shape
        assert K_logits == K
        bidx = jnp.arange(B)

        logp0 = jax.nn.log_sigmoid(logit0)
        log1mp0 = jax.nn.log_sigmoid(-logit0)

        logh = jax.nn.log_sigmoid(logits)
        log1mh = jax.nn.log_sigmoid(-logits)

        prefix = jnp.cumsum(log1mh, axis=-1)
        prefix_before = jnp.concatenate(
            [jnp.zeros((B, 1), dtype=logits.dtype), prefix[:, :-1]],
            axis=1,
        )

        tau = jnp.asarray(tau, dtype=jnp.int32)
        censor = jnp.asarray(censor, dtype=jnp.int32)

        # event log-likelihood
        ll_tau0 = logp0
        tau_bin = jnp.searchsorted(bins, tau, side="right") - 1
        tau_bin = jnp.clip(tau_bin, 0, K - 1).astype(jnp.int32)
        ll_event_pos = log1mp0 + prefix_before[bidx, tau_bin] + logh[bidx, tau_bin]
        ll_event = jnp.where(tau == 0, ll_tau0, ll_event_pos)

        # censor log-likelihood
        censor_edge = jnp.searchsorted(bins, censor, side="left")
        censor_edge = jnp.clip(censor_edge, 0, K).astype(jnp.int32)
        surv_bins = jnp.where(
            censor_edge == 0,
            jnp.zeros((B,), dtype=logits.dtype),
            prefix[bidx, censor_edge - 1],
        )
        ll_censor = log1mp0 + surv_bins

        ll = jnp.where(is_event > 0.5, ll_event, ll_censor)
        return -ll

    return nll_from_logits
