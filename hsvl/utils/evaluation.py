from collections import defaultdict
import tqdm
from typing import Any, Dict
from omegaconf import DictConfig
from flax.core.frozen_dict import FrozenDict
import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get("goal")
        goal_frame = info.get("goal_rendered")
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def run_ogbench_eval(cfg: DictConfig, agent_cfg: FrozenDict, env: Any, eval_agent: Any) -> Dict[str, float]:
    """
    Thin wrapper around  utils.evaluation.evaluate.
    Keeps the metric keys stable across all offline agents.
    """
    overall = defaultdict(list)
    eval_metrics: Dict[str, float] = {}

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, "task_infos") else env.task_infos
    num_tasks = int(cfg.train.eval_tasks) if cfg.train.eval_tasks is not None else len(task_infos)

    for task_id in tqdm.trange(1, num_tasks + 1, desc="Eval tasks"):
        task_name = task_infos[task_id - 1]["task_name"]
        eval_info, trajs, cur_renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=agent_cfg,
            num_eval_episodes=int(cfg.train.eval_episodes),
            num_video_episodes=int(cfg.train.video_episodes),
            video_frame_skip=int(cfg.train.video_frame_skip),
            eval_temperature=float(cfg.train.eval_temperature),
            eval_gaussian=bool(cfg.train.eval_gaussian),
        )

        # Keep success-only for now, extend later
        if "success" in eval_info:
            v = float(eval_info["success"])
            eval_metrics[f"evaluation/{task_name}_success"] = v
            overall["success"].append(v)

    if len(overall["success"]) > 0:
        eval_metrics["evaluation/overall_success"] = float(np.mean(overall["success"]))

    return eval_metrics
