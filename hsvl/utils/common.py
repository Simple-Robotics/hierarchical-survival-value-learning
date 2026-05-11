from __future__ import annotations
import numpy as np
import random
import jax
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import flax.linen as nn
import wandb
import jax.numpy as jnp
from typing import Tuple

def get_activation_fn(name: str):
    if name == "gelu":
        return nn.gelu

    elif name == "relu":
        return nn.relu

    elif name == "swish":
        return nn.swish

    elif name == "elu":
        return nn.elu


def jit_static(*args, **kwargs):
    def decorator(fn):
        return jax.jit(fn, *args, **kwargs)

    return decorator


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def maybe_put_on_cpu(agent, enabled: bool):
    if not enabled:
        return agent
    return jax.device_put(agent, device=jax.devices("cpu")[0])


def make_save_dir(cfg: DictConfig, run_name: str) -> Path:
    return Path(os.path.join(cfg.save_dir, f"{cfg.wandb.project}_{cfg.wandb.group}", run_name))


def get_exp_name(cfg: DictConfig, seed):
    s = f"{cfg.agent.agent_name}_{cfg.env_name}_s{seed}"
    if "SLURM_JOB_ID" in os.environ:
        s += f'_slurm_{os.environ["SLURM_JOB_ID"]}'
    if "SLURM_PROCID" in os.environ:
        s += f'_{os.environ["SLURM_PROCID"]}'
    return s


def initialize_wandb(cfg: DictConfig, run_name: str):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    group = cfg.wandb.group
    wandb.init(
        project=cfg.wandb.project,
        group=group,
        name=run_name,
        tags=[group],
        config=config,
        mode=cfg.wandb.mode,
        reinit=True,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="train/step")


def create_log_bins(horizon: int, num_bins: int) -> Tuple[jnp.ndarray, int]:
    """
    Create bin boundaries b_0..b_K with b_0=0, b_K=horizon.

    Returns:
      bins: int32 array of shape (K+1,)
      K: number of bins
    """
    if num_bins <= 0 or num_bins >= horizon:
        bins = jnp.arange(horizon + 1, dtype=jnp.int32)
        return bins, horizon

    # Geometric spacing then de-dup.
    ratio = jnp.power(horizon / 1.0, 1.0 / num_bins)
    bins = jnp.round(jnp.power(ratio, jnp.arange(num_bins + 1))).astype(jnp.int32)
    bins = jnp.clip(bins, 0, horizon)

    bins = jnp.unique(bins)
    bins = jnp.unique(jnp.concatenate([jnp.array([0, horizon], dtype=jnp.int32), bins]))
    K = int(bins.shape[0] - 1)
    return bins, K