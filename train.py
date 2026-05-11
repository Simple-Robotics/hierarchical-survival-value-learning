from __future__ import annotations

import time
from pathlib import Path

import hydra
import jax
import numpy as np
import tqdm
import wandb
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf

from hsvl.agent import agents as AGENT_REGISTRY
from hsvl.utils.common import (
    get_exp_name,
    initialize_wandb,
    make_save_dir,
    maybe_put_on_cpu,
    set_global_seeds,
)
from hsvl.utils.datasets import (
    HGCDataset_sample,
    dataset_size_jax,
    make_env_and_datasets,
    prepare_hgc_dataset_for_jax,
)
from hsvl.utils.evaluation import run_ogbench_eval
from hsvl.utils.flax_utils import save_agent

OmegaConf.register_new_resolver("eval", lambda s: eval(s, globals()), replace=True)


def _write_resolved_config(save_dir: Path, cfg: DictConfig) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


@hydra.main(config_path="hsvl/config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed is not None else np.random.randint(2**31)
    set_global_seeds(int(seed))

    run_name = get_exp_name(cfg, seed)
    initialize_wandb(cfg, run_name)

    save_dir = make_save_dir(cfg, run_name)
    _write_resolved_config(save_dir, cfg)

    agent_cfg = FrozenDict(**cfg.agent)
    batch_size = int(agent_cfg["batch_size"])

    env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name,
        frame_stack=agent_cfg["frame_stack"],
    )
    train_dataset = prepare_hgc_dataset_for_jax(train_dataset, do_terminals=True)

    train_size = dataset_size_jax(train_dataset)

    rng_init = jax.random.PRNGKey(int(seed))
    example_batch, _ = HGCDataset_sample(rng_init, train_dataset, batch_size, train_size, agent_cfg)

    agent_class = AGENT_REGISTRY[agent_cfg["agent_name"]]
    agent = agent_class.create(
        seed=int(seed),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        ex_goals=example_batch["oracle_reps"] if "oracle_reps" in train_dataset.keys() else None,
        config=agent_cfg,
    )

    train_steps = int(cfg.train.train_steps)
    log_interval = int(cfg.train.log_interval)
    eval_interval = int(cfg.train.eval_interval)
    save_interval = int(cfg.train.save_interval)

    print("Warming up JIT compilation...")
    warmup_start = time.time()
    agent, _ = agent.sample_and_update(train_dataset, batch_size, train_size)
    jax.block_until_ready(agent)
    warmup_time = time.time() - warmup_start
    print(f"✓ Compilation complete in {warmup_time:.2f}s. Starting training...")

    try:
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Available devices: {jax.devices()}")
    except Exception:
        pass

    first_time = time.time()
    last_log_time = first_time
    last_log_step = 0

    final_overall_successes: list[float] = []
    last_eval_step: int | None = None

    tqdm_update_interval = log_interval
    sync_interval = log_interval

    step = 0
    steps_since_last_log = 0

    with tqdm.tqdm(total=train_steps, smoothing=0.1, dynamic_ncols=True) as pbar:
        while step < train_steps:
            for _ in range(sync_interval):
                if step >= train_steps:
                    break
                agent, update_info = agent.sample_and_update(train_dataset, batch_size, train_size)
                step += 1
                steps_since_last_log += 1

            if step % tqdm_update_interval == 0 or step >= train_steps:
                pbar.update(step - pbar.n)

            if step % log_interval == 0 or step >= train_steps:
                now = time.time()

                steps_since = step - last_log_step
                dt = now - last_log_time
                sps = (steps_since / dt) if dt > 0 else float("inf")

                train_metrics = {f"training/{k}": float(v) for k, v in update_info.items()}
                train_metrics["time/total_time"] = now - first_time
                train_metrics["time/steps_per_second"] = sps
                train_metrics["time/interval_seconds"] = dt
                train_metrics["time/interval_steps"] = steps_since

                wandb.log(train_metrics, step=step)

                last_log_time = now
                last_log_step = step

            do_eval = (
                (step % eval_interval == 0 and not bool(cfg.train.only_eval_three_lasts))
                or step >= train_steps
                or (
                    bool(cfg.train.only_eval_three_lasts)
                    and step
                    in {
                        int(cfg.train.train_steps) - 200_000,
                        int(cfg.train.train_steps) - 100_000,
                        int(cfg.train.train_steps),
                    }
                )
            )
            if do_eval:
                eval_agent = maybe_put_on_cpu(agent, bool(cfg.eval_on_cpu))
                eval_metrics = run_ogbench_eval(
                    cfg=cfg,
                    agent_cfg=agent_cfg,
                    env=env,
                    eval_agent=eval_agent,
                )

                if "evaluation/overall_success" in eval_metrics:
                    if bool(cfg.train.only_eval_three_lasts):
                        final_overall_successes.append(float(eval_metrics["evaluation/overall_success"]))
                        if step >= int(cfg.train.train_steps):
                            eval_metrics["evaluation/final_overall_success"] = float(
                                np.mean(final_overall_successes)
                            )

                wandb.log(eval_metrics, step=step)
                last_eval_step = step

            if step % save_interval == 0:
                save_agent(agent, save_dir, step)

    if last_eval_step != step:
        eval_agent = maybe_put_on_cpu(agent, bool(cfg.eval_on_cpu))
        eval_metrics = run_ogbench_eval(
            cfg=cfg,
            agent_cfg=agent_cfg,
            env=env,
            eval_agent=eval_agent,
        )
        if "evaluation/overall_success" in eval_metrics and bool(cfg.train.only_eval_three_lasts):
            final_overall_successes.append(float(eval_metrics["evaluation/overall_success"]))
            eval_metrics["evaluation/final_overall_success"] = float(np.mean(final_overall_successes))
        wandb.log(eval_metrics, step=step)

    (save_dir / "done.txt").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
