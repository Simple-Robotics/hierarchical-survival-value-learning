from __future__ import annotations

import collections
from functools import partial
from typing import Any, Mapping

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from flax.core.frozen_dict import FrozenDict
from gymnasium.spaces import Box


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper that stacks the last `num_stack` observations along the last axis."""

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype,
        )

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name: str, frame_stack: int | None = None):
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name,
        compact_dataset=True,
    )

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()
    return env, train_dataset, val_dataset


def _ensure_last_is_terminal_locs(tloc: np.ndarray, n: int) -> np.ndarray:
    if tloc.size == 0:
        return np.array([n - 1], dtype=np.int64)
    if tloc[-1] != n - 1:
        tloc = np.concatenate([tloc, np.array([n - 1], dtype=tloc.dtype)], axis=0)
    return tloc


def prepare_hgc_dataset_for_jax(
    dataset: Mapping[str, Any],
    do_terminals: bool = True,
    *,
    device: jax.Device | None = None,  # type: ignore
    idx_dtype: jnp.dtype = jnp.int32,
) -> FrozenDict:
    """Build valid_idxs + terminal_locs on host and move everything to device once."""
    dataset = dict(**dataset)
    n = len(dataset["observations"])

    if "next_observations" not in dataset:
        # Compact dataset
        assert "valids" in dataset, "Compact dataset must include 'valids'."
        valids = np.asarray(dataset["valids"])

        (valid_idxs,) = np.nonzero(valids > 0)
        dataset["valid_idxs"] = valid_idxs.astype(np.int32)

        if do_terminals:
            (ends,) = np.nonzero(valids == 0)
            ends = _ensure_last_is_terminal_locs(ends.astype(np.int64), n)
            dataset["terminal_locs"] = ends.astype(np.int32)
            dataset["initial_locs"] = np.concatenate(
                [[0], dataset["terminal_locs"][:-1] + 1]
            ).astype(np.int32)

    else:
        if "terminals" in dataset:
            terminals = np.asarray(dataset["terminals"]).astype(bool)
            dataset["valid_idxs"] = np.nonzero(~terminals)[0].astype(np.int32)
        else:
            dataset["valid_idxs"] = np.arange(n, dtype=np.int32)

        if do_terminals:
            if "terminals" in dataset:
                (tloc,) = np.nonzero(np.asarray(dataset["terminals"]) > 0)
                tloc = _ensure_last_is_terminal_locs(tloc.astype(np.int64), n)
                dataset["terminal_locs"] = tloc.astype(np.int32)
                dataset["initial_locs"] = np.concatenate(
                    [[0], dataset["terminal_locs"][:-1] + 1]
                ).astype(np.int32)
            else:
                dataset["terminal_locs"] = np.array([n - 1], dtype=np.int32)
                dataset["initial_locs"] = np.array([0], dtype=np.int32)

    on_device = {}
    for k, v in dataset.items():
        arr = jnp.asarray(v)
        if k in ("valid_idxs", "terminal_locs", "initial_locs"):
            arr = arr.astype(idx_dtype)
        on_device[k] = arr

    if device is not None:
        on_device = jax.device_put(on_device, device)

    return FrozenDict(on_device)


def dataset_size_jax(dataset: FrozenDict) -> int:
    return int(dataset["valid_idxs"].shape[0])


@partial(jax.jit, static_argnames=("batch_size", "dataset_size"))
def Dataset_sample_idxs(rng, dataset: FrozenDict, batch_size: int, dataset_size: int):
    rng, sample_rng = jax.random.split(rng)
    batch_pos = jax.random.randint(sample_rng, (batch_size,), 0, dataset_size)
    batch_idxs = dataset["valid_idxs"][batch_pos]
    return batch_idxs, rng


def get_stacked_observations(dataset, idxs, frame_stack: int):
    """Return the frame-stacked observations for the given indices."""
    initial_state_idxs = dataset["initial_locs"][
        jnp.searchsorted(dataset["initial_locs"], idxs, side="right") - 1
    ]
    rets = []
    for i in reversed(range(frame_stack)):
        cur_idxs = jnp.maximum(idxs - i, initial_state_idxs)
        rets.append(
            jax.tree_util.tree_map(
                lambda arr: arr[cur_idxs],
                dataset["observations"],
            )
        )
    return jax.tree_util.tree_map(lambda *args: jnp.concatenate(args, axis=-1), *rets)


def Dataset_get_observations(idxs, dataset: FrozenDict, frame_stack: int | None = None):
    if frame_stack is not None:
        return get_stacked_observations(dataset, idxs, frame_stack)
    return dataset["observations"][idxs]


def Dataset_get_next_observations(
    idxs, dataset: FrozenDict, frame_stack: int | None = None
):
    if frame_stack is not None:
        return get_stacked_observations(dataset, idxs + 1, frame_stack)
    if "next_observations" in dataset:
        return dataset["next_observations"][idxs]
    return dataset["observations"][idxs + 1]


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def GCDataset_sample_goals(
    idxs,
    final_state_idxs,
    rng,
    dataset,
    batch_size,
    dataset_size,
    p_curgoal,
    p_trajgoal,
    p_randomgoal,
    geom_sample,
    geom_discount,
):
    if p_curgoal == 1.0:
        return idxs, rng

    rng, sample_rng = jax.random.split(rng)
    if geom_sample:
        offsets = jax.random.geometric(sample_rng, 1 - geom_discount, (batch_size,))
        goal_idxs = jnp.minimum(idxs + offsets, final_state_idxs)
    else:
        distances = jax.random.uniform(sample_rng, (batch_size,))
        goal_idxs = jnp.rint(
            jnp.minimum(idxs + 1, final_state_idxs) * distances
            + final_state_idxs * (1.0 - distances)
        ).astype(jnp.int32)

    if p_trajgoal != 1.0:
        rng, sample_rng = jax.random.split(rng)
        r_choice = jax.random.uniform(sample_rng, (batch_size,))
        if p_randomgoal != 0.0:
            random_goal_idxs, rng = Dataset_sample_idxs(
                rng, dataset, batch_size, dataset_size
            )
            goal_idxs = jnp.where(
                r_choice > 1.0 - p_randomgoal,
                random_goal_idxs,
                goal_idxs,
            )
        if p_curgoal != 0.0:
            goal_idxs = jnp.where(r_choice < p_curgoal, idxs, goal_idxs)

    return goal_idxs, rng


def _get_goal_reprs_or_obs(goal_idxs, dataset: FrozenDict, frame_stack: int | None):
    if "oracle_reps" in dataset:
        return dataset["oracle_reps"][goal_idxs]
    return Dataset_get_observations(goal_idxs, dataset, frame_stack)


@partial(jax.jit, static_argnames=("batch_size", "dataset_size", "config"))
def HGCDataset_sample(
    rng,
    dataset: FrozenDict,
    batch_size: int,
    dataset_size: int,
    config: Any,
):
    """JAX-jitted sampler producing observations, goals, and survival targets."""
    idxs, rng = Dataset_sample_idxs(rng, dataset, batch_size, dataset_size)
    actions = dataset["actions"][idxs]
    obs = Dataset_get_observations(idxs, dataset, config["frame_stack"])
    next_obs = Dataset_get_next_observations(idxs, dataset, config["frame_stack"])

    terminal_locs = dataset["terminal_locs"]
    final_state_idxs = terminal_locs[jnp.searchsorted(terminal_locs, idxs)]

    value_goal_idxs, rng = GCDataset_sample_goals(
        idxs,
        final_state_idxs,
        rng,
        dataset,
        batch_size,
        dataset_size,
        float(config["value_p_curgoal"]),
        float(config["value_p_trajgoal"]),
        float(config["value_p_randomgoal"]),
        bool(config["value_geom_sample"]),
        float(config["discount"]),
    )
    value_goals = _get_goal_reprs_or_obs(value_goal_idxs, dataset, config["frame_stack"])

    actor_goal_idxs, rng = GCDataset_sample_goals(
        idxs,
        final_state_idxs,
        rng,
        dataset,
        batch_size,
        dataset_size,
        float(config["actor_p_curgoal"]),
        float(config["actor_p_trajgoal"]),
        float(config["actor_p_randomgoal"]),
        bool(config["actor_geom_sample"]),
        float(config["discount"]),
    )
    actor_goals = _get_goal_reprs_or_obs(actor_goal_idxs, dataset, config["frame_stack"])

    low_goal_idxs = jnp.minimum(idxs + int(config["subgoal_steps"]), final_state_idxs)
    low_actor_goals = _get_goal_reprs_or_obs(low_goal_idxs, dataset, config["frame_stack"])

    rng, sample_rng = jax.random.split(rng)
    if bool(config["actor_geom_sample"]):
        offsets = jax.random.geometric(
            sample_rng,
            1.0 - float(config["discount"]),
            (batch_size,),
        )
        high_goal_idxs = jnp.minimum(idxs + offsets, final_state_idxs)
    else:
        distances = jax.random.uniform(sample_rng, (batch_size,))
        high_goal_idxs = jnp.rint(
            jnp.minimum(idxs + 1, final_state_idxs) * distances
            + final_state_idxs * (1.0 - distances)
        ).astype(jnp.int32)

    high_target_idxs = jnp.minimum(idxs + int(config["subgoal_steps"]), high_goal_idxs)

    if float(config["actor_p_randomgoal"]) != 0.0:
        random_goal_idxs, rng = Dataset_sample_idxs(
            rng, dataset, batch_size, dataset_size
        )
        rng, pick_rng = jax.random.split(rng)
        pick_random = (
            jax.random.uniform(pick_rng, (batch_size,))
            < float(config["actor_p_randomgoal"])
        )
        high_goal_idxs = jnp.where(pick_random, random_goal_idxs, high_goal_idxs)
        high_target_idxs = jnp.where(pick_random, low_goal_idxs, high_target_idxs)

    high_actor_goals = _get_goal_reprs_or_obs(high_goal_idxs, dataset, config["frame_stack"])
    high_actor_targets = _get_goal_reprs_or_obs(high_target_idxs, dataset, config["frame_stack"])

    if "oracle_reps" in dataset:
        high_actor_targets_state = Dataset_get_observations(
            high_target_idxs, dataset, config["frame_stack"]
        )
    else:
        high_actor_targets_state = None

    H = int(config["surv_horizon"])

    goal_final = terminal_locs[jnp.searchsorted(terminal_locs, value_goal_idxs)]
    same_traj = goal_final == final_state_idxs
    goal_ahead = value_goal_idxs >= idxs

    tau_exact = (value_goal_idxs - idxs).astype(jnp.int32)
    oracle_event = same_traj & goal_ahead & (tau_exact >= 0) & (tau_exact <= H)

    rem = (final_state_idxs - idxs).astype(jnp.int32)
    max_future = jnp.minimum(rem, H).astype(jnp.int32)
    has_future = max_future >= 1
    censor_time = max_future

    is_event = oracle_event
    tau = tau_exact
    fb_apply = jnp.zeros_like(oracle_event, dtype=bool)
    event_set = jnp.zeros_like(oracle_event, dtype=bool)

    tau = jnp.where(is_event, tau, jnp.minimum(max_future, 1))
    tau = jnp.clip(tau, 0, H).astype(jnp.int32)

    valid = (has_future | (is_event & (tau == 0))).astype(jnp.float32)

    batch = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "value_goals": value_goals,
        "value_goal_idxs": value_goal_idxs,
        "actor_goals": actor_goals,
        "actor_goal_idxs": actor_goal_idxs,
        "surv_is_event": is_event.astype(jnp.float32),
        "surv_time": tau,
        "surv_censor_time": censor_time,
        "surv_valid": valid,
        "surv_oracle_event": oracle_event.astype(jnp.float32),
        "low_actor_goals": low_actor_goals,
        "high_actor_goals": high_actor_goals,
        "high_actor_targets": high_actor_targets,
        "high_actor_targets_state": high_actor_targets_state,
        "number_set_event": jnp.mean(fb_apply & event_set).astype(jnp.float32),
    }

    return batch, rng
