"""Resume PPO training from an existing checkpoint.

Loads `--resume_from` and trains for that many more timesteps, keeping the
same env and logging path. Defaults to 4 envs since SubprocVecEnv tends to
flake out with more on Windows.

Run:
    python task11_rl_controller\resume_train.py \
        --resume_from D:\Robotics\task11_rl_controller\weights\PPO_5M_v3_best\best_model.zip \
        --total_timesteps 3000000 \
        --experiment_name "PPO_resume_v3"
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ot2.envs.ot2_env import OT2Env


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = HERE / "weights"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume_from", type=str, required=True,
                   help="Path to .zip checkpoint to resume from.")
    p.add_argument("--total_timesteps", type=int, default=3_000_000)
    p.add_argument("--n_envs", type=int, default=4,
                   help="Lower than 8 for Windows PyBullet stability.")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--experiment_name", type=str, default="PPO_resume")
    p.add_argument("--max_episode_steps", type=int, default=600)
    p.add_argument("--success_threshold_mm", type=float, default=1.0)
    p.add_argument("--dwell_steps", type=int, default=10)
    p.add_argument("--eval_freq", type=int, default=100_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--checkpoint_freq", type=int, default=200_000,
                   help="Save a checkpoint every N timesteps (across all envs).")
    return p.parse_args()


def make_env(seed: int, max_episode_steps: int, success_threshold: float,
             dwell_steps: int, rank: int = 0):
    def _thunk():
        env = OT2Env(seed=seed + rank, max_steps=max_episode_steps,
                     success_threshold=success_threshold, dwell_steps=dwell_steps,
                     randomise_start=True)
        return Monitor(env)
    return _thunk


def main():
    args = parse_args()
    # Resolve this now: building OT2Env later calls ot2.sim.make_sim(), which
    # chdir's into the sim folder and would break a relative checkpoint path.
    resume = Path(args.resume_from).resolve()
    if not resume.exists():
        raise SystemExit(f"checkpoint not found: {resume}")
    print(f"=== Resuming from {resume} for {args.total_timesteps:,} more steps ===")
    print(vars(args))

    RUNS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    tb_dir = RUNS_DIR / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    env_fns = [
        make_env(seed=args.seed,
                 max_episode_steps=args.max_episode_steps,
                 success_threshold=args.success_threshold_mm / 1000.0,
                 dwell_steps=args.dwell_steps, rank=i)
        for i in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)

    eval_env = DummyVecEnv([
        make_env(seed=args.seed + 5000,
                 max_episode_steps=args.max_episode_steps,
                 success_threshold=args.success_threshold_mm / 1000.0,
                 dwell_steps=args.dwell_steps)
    ])

    # reload the checkpoint and attach the fresh vec env
    model = PPO.load(str(resume), env=env, device="auto",
                     tensorboard_log=str(tb_dir))
    print(f"loaded model: policy net = {model.policy}")

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(1, args.n_envs), 1),
            save_path=str(WEIGHTS_DIR),
            name_prefix=args.experiment_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(WEIGHTS_DIR / f"{args.experiment_name}_best"),
            log_path=str(RUNS_DIR / "eval" / args.experiment_name),
            eval_freq=max(args.eval_freq // max(1, args.n_envs), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True, render=False,
        ),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.experiment_name,
        reset_num_timesteps=False,  # keep the step counter going from the checkpoint
    )

    final = WEIGHTS_DIR / f"{args.experiment_name}_final.zip"
    model.save(final)
    print(f"\nSaved {final}")

    shutil.copyfile(final, WEIGHTS_DIR / "latest.zip")
    print(f"Updated {WEIGHTS_DIR / 'latest.zip'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
