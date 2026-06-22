"""Train a PPO agent on the OT-2 Gym env (Task 11).

The CLI takes one hyperparameter set per run, e.g.

    python train_rl.py --learning_rate 3e-4 --batch_size 64 --n_steps 2048 \
        --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 10 \
        --total_timesteps 100000 --experiment_name "PPO_smoke"

TensorBoard logging is always on. W&B and ClearML are opt-in (see the
--wandb_* and --clearml flags). Checkpoints and the final model go to
weights/, and weights/latest.zip is refreshed each run so test_rl.py can
pick up the most recent model without being told which one.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ot2.envs.ot2_env import OT2Env


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = HERE / "weights"


def parse_args():
    p = argparse.ArgumentParser()
    # PPO hyperparameters
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--net_arch", type=str, default="64,64",
                   help="Comma-separated hidden layer sizes, e.g. '64,64' or '128,128,64'.")
    # Training
    p.add_argument("--total_timesteps", type=int, default=100_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # Env
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--success_threshold_mm", type=float, default=1.0)
    p.add_argument("--dwell_steps", type=int, default=20)
    # Logging
    p.add_argument("--experiment_name", type=str, default="PPO_smoke")
    p.add_argument("--wandb_key", type=str, default=None)
    p.add_argument("--wandb_offline", action="store_true", help="Run W&B in offline mode (no upload).")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B entirely.")
    p.add_argument("--clearml", action="store_true", help="Enable ClearML task tracking (reads creds from ~/clearml.conf or CLEARML_API_* env vars).")
    # Eval
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=5)
    return p.parse_args()


def make_env(seed: int, max_episode_steps: int, success_threshold: float,
             dwell_steps: int, rank: int = 0):
    def _thunk():
        env = OT2Env(
            seed=seed + rank,
            max_steps=max_episode_steps,
            success_threshold=success_threshold,
            dwell_steps=dwell_steps,
            randomise_start=True,
        )
        env = Monitor(env)
        return env
    return _thunk


def setup_clearml(experiment_name: str) -> Any:
    """Init ClearML.

    Credentials come from a local ~/clearml.conf if present. If not, you can
    pass them through the environment instead (CLEARML_API_HOST,
    CLEARML_WEB_HOST, CLEARML_FILES_HOST, CLEARML_API_ACCESS_KEY,
    CLEARML_API_SECRET_KEY). See .env.example for the names. No credentials
    are hardcoded here.
    """
    from clearml import Task

    home = Path(os.path.expanduser("~"))
    if not (home / "clearml.conf").exists():
        access_key = os.getenv("CLEARML_API_ACCESS_KEY")
        secret_key = os.getenv("CLEARML_API_SECRET_KEY")
        api_host = os.getenv("CLEARML_API_HOST")
        web_host = os.getenv("CLEARML_WEB_HOST")
        files_host = os.getenv("CLEARML_FILES_HOST")
        if access_key and secret_key and api_host:
            print("No ~/clearml.conf found; using credentials from environment.")
            Task.set_credentials(
                api_host=api_host,
                web_host=web_host,
                files_host=files_host,
                key=access_key,
                secret=secret_key,
            )
        else:
            print("No ~/clearml.conf and no ClearML env vars set; "
                  "skipping ClearML. Set CLEARML_API_* or add a clearml.conf "
                  "to enable it.")
            return None
    try:
        task = Task.init(project_name="OT2-RL", task_name=experiment_name, reuse_last_task_id=False)
    except Exception as e:
        print(f"ClearML init failed ({type(e).__name__}: {e}). Continuing without ClearML.")
        return None
    return task


def setup_wandb(experiment_name: str, args, key: str | None, offline: bool) -> Any:
    """Init W&B. Returns the run, or None if it's disabled or init fails."""
    import wandb

    if key is not None:
        os.environ["WANDB_API_KEY"] = key

    mode = "offline" if offline or os.environ.get("WANDB_API_KEY") is None else "online"
    if mode == "offline":
        print("W&B running in offline mode; sync later with `wandb sync runs/wandb/<run>`.")
    try:
        run = wandb.init(
            project="OT2-RL",
            name=experiment_name,
            config=vars(args),
            mode=mode,
            dir=str(RUNS_DIR / "wandb"),
            sync_tensorboard=True,
            reinit=True,
        )
    except Exception as e:
        print(f"W&B init failed ({type(e).__name__}: {e}). Continuing without W&B.")
        return None
    return run


class WandbStepCallback(BaseCallback):
    """Hook for pushing extra metrics to W&B mid-rollout."""
    def __init__(self, wandb_run):
        super().__init__()
        self.wandb_run = wandb_run

    def _on_step(self) -> bool:
        if self.wandb_run is None:
            return True
        # sync_tensorboard=True already mirrors the SB3 scalars into W&B, so
        # there's nothing to do here unless we add custom metrics later.
        return True


def main():
    args = parse_args()
    print(f"=== Training run: {args.experiment_name} ===")
    print(vars(args))

    RUNS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    tb_dir = RUNS_DIR / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    # ClearML has to go first so it auto-captures everything that follows.
    clearml_task = setup_clearml(args.experiment_name) if args.clearml else None

    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(args.experiment_name, args, args.wandb_key, args.wandb_offline)

    env_fns = [
        make_env(
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            success_threshold=args.success_threshold_mm / 1000.0,
            dwell_steps=args.dwell_steps,
            rank=i,
        )
        for i in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)

    # Dedicated single-env for eval; offset seed so it sees unseen targets.
    eval_env = DummyVecEnv([
        make_env(
            seed=args.seed + 1000,
            max_episode_steps=args.max_episode_steps,
            success_threshold=args.success_threshold_mm / 1000.0,
            dwell_steps=args.dwell_steps,
        )
    ])

    net_arch = [int(x) for x in args.net_arch.split(",")]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.total_timesteps // 5 // max(1, args.n_envs), 1),
            save_path=str(WEIGHTS_DIR),
            name_prefix=args.experiment_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(WEIGHTS_DIR / f"{args.experiment_name}_best"),
            log_path=str(RUNS_DIR / "eval" / args.experiment_name),
            eval_freq=max(args.eval_freq // max(1, args.n_envs), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
        WandbStepCallback(wandb_run),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.experiment_name,
    )

    final_path = WEIGHTS_DIR / f"{args.experiment_name}_final.zip"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    # test_rl.py defaults to latest.zip, so point it at this run.
    latest = WEIGHTS_DIR / "latest.zip"
    shutil.copyfile(final_path, latest)
    print(f"Updated {latest}")

    if wandb_run is not None:
        wandb_run.finish()
    if clearml_task is not None:
        clearml_task.close()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
