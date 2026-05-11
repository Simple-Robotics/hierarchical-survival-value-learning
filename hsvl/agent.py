from __future__ import annotations
from typing import Any, Dict, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from hsvl.utils.encoders import GCEncoder
from hsvl.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from hsvl.utils.mlp import MLP, Identity, LengthNormalize
from hsvl.utils.actors import GCActor
from hsvl.utils.critics import GCSurvivalValue
from hsvl.utils.survival import create_perbin_nll_fn, create_perbin_value_fn
from hsvl.utils.datasets import HGCDataset_sample
from hsvl.utils.common import get_activation_fn, jit_static, create_log_bins


class HSVL(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    _value_from_logits: Any = nonpytree_field()
    _nll_from_logits: Any = nonpytree_field()
    _bins: Any = nonpytree_field()

    # ---------- scalar V from (logit0, logits) ----------
    def _V(self, pair: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        logit0, logits = pair
        return self._value_from_logits(logit0, logits)

    def value_loss(self, batch, grad_params):
        is_event = batch["surv_is_event"]
        tau = batch["surv_time"].astype(jnp.int32)
        censor = batch["surv_censor_time"].astype(jnp.int32)
        valid = batch["surv_valid"]

        pair1, pair2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )

        nll1 = self._nll_from_logits(pair1[0], pair1[1], is_event, tau, censor)
        nll2 = self._nll_from_logits(pair2[0], pair2[1], is_event, tau, censor)

        valid_sum = jnp.maximum(valid.sum(), 1.0)
        value_loss = (valid * (nll1 + nll2)).sum() / valid_sum

        v = 0.5 * (self._V(jax.lax.stop_gradient(pair1)) + self._V(jax.lax.stop_gradient(pair2)))
        info = {
            "value_loss": value_loss,
            "surv/event_rate": (valid * is_event).sum() / valid_sum,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }
        return value_loss, info

    def low_actor_loss(self, batch, grad_params):
        p1, p2 = self.network.select("value")(batch["observations"], batch["low_actor_goals"])
        np1, np2 = self.network.select("value")(batch["next_observations"], batch["low_actor_goals"])

        if self.config["critic_agg"] == "min":
            v = jnp.minimum(self._V(p1), self._V(p2))
            nv = jnp.minimum(self._V(np1), self._V(np2))
        else:   
            v = 0.5 * (self._V(p1) + self._V(p2))
            nv = 0.5 * (self._V(np1) + self._V(np2))

        adv = nv - v

        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select("goal_rep")(
            jnp.concatenate([batch["observations"], batch["low_actor_goals"]], axis=-1),
            params=grad_params,
        )
        if not bool(self.config["low_actor_rep_grad"]):
            goal_reps = jax.lax.stop_gradient(goal_reps)

        dist = self.network.select("low_actor")(
            batch["observations"], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_prob).mean()

        info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }
        return actor_loss, info

    def high_actor_loss(self, batch, grad_params):
        p1, p2 = self.network.select("value")(batch["observations"], batch["high_actor_goals"])
        np1, np2 = self.network.select("value")(batch["high_actor_targets"], batch["high_actor_goals"])

        if self.config["critic_agg"] == "min":
            v = jnp.minimum(self._V(p1), self._V(p2))
            nv = jnp.minimum(self._V(np1), self._V(np2))
        else:
            v = 0.5 * (self._V(p1) + self._V(p2))
            nv = 0.5 * (self._V(np1) + self._V(np2))
            
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["high_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        target = self.network.select("goal_rep")(
            jnp.concatenate([batch["observations"], batch["high_actor_targets"]], axis=-1)
        )
        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - target) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        info: Dict[str, jnp.ndarray] = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v
 
        low_loss, low_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_info.items():
            info[f"low_actor/{k}"] = v

        high_loss, high_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_info.items():
            info[f"high_actor/{k}"] = v

        loss = value_loss + low_loss + high_loss
        info["total_loss"] = loss
        info["sucess_set_event"] = batch["number_set_event"]
        return loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = (
            goal_reps
            / (jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) + 1e-8)
            * jnp.sqrt(goal_reps.shape[-1])
        )

        low_dist = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jit_static(static_argnames=("batch_size", "dataset_size"))
    def sample_and_update(self, dataset, batch_size: int, dataset_size: int):
        batch, rng = HGCDataset_sample(self.rng, dataset, batch_size, dataset_size, self.config)
        rng, loss_rng = jax.random.split(rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        config = dict(config)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if ex_goals is None:
            ex_goals = ex_observations

        action_dim = int(ex_actions.shape[-1])
        config["action_dim"] = int(action_dim)

        activation_fn = get_activation_fn(config["activation_fn"])

        # -------- Goal representation phi([s; g]) --------
        goal_rep_def = nn.Sequential([
            MLP(
                hidden_dims=(*config["value_hidden_dims"], config["rep_dim"]),
                activations=activation_fn,
                activate_final=False,
                layer_norm=config["layer_norm"],
            ),
            LengthNormalize(),
        ])

        # -------- Encoders --------
        value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
        low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
        high_actor_encoder_def = None

        # -------- Survival bins + math --------
        H = int(config["surv_horizon"])
        K_req = int(config["num_log_bins"])
        bins, K = create_log_bins(H, K_req)

        value_from_logits = create_perbin_value_fn(bins, float(config["discount"]), config["add_tail"])
        nll_from_logits = create_perbin_nll_fn(bins)

        # -------- Value network --------
        value_def = GCSurvivalValue(
            hidden_dims=config["value_hidden_dims"],
            num_bins=int(K),
            k_basis=config["k_basis"],
            num_basis_sets=config["num_basis_sets"],
            layer_norm=config["layer_norm"],
            gc_encoder=value_encoder_def,
            activation=activation_fn,
            ensemble=config["ensemble"],
            use_residual=config["residual_critic"],
            num_blocks=config["critic_num_blocks"],
        )

        # -------- Actors --------
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config["const_std"],
            gc_encoder=low_actor_encoder_def,
            activation=activation_fn,
            use_residual=config["residual_actor"],
            num_blocks=config["actor_num_blocks"],
            layer_norm=config["layer_norm"],
        )

        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=int(config["rep_dim"]),
            state_dependent_std=False,
            const_std=config["const_std"],
            gc_encoder=high_actor_encoder_def,
            activation=activation_fn,
            use_residual=config["residual_actor"],
            num_blocks=config["actor_num_blocks"],
            layer_norm=config["layer_norm"],
        )

        # -------- ModuleDict init --------
        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1),)),
            value=(value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=float(config["lr"]))
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(model_def=network_def, params=network_params, tx=network_tx)

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            _value_from_logits=value_from_logits,
            _nll_from_logits=nll_from_logits,
            _bins=bins,
        )


agents = dict(
    hsvl=HSVL,
)
