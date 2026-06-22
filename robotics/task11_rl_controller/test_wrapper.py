"""Smoke test for the Gym wrapper (required by Task 11).

Drives the OT-2 env for 1000 random-action steps and prints per-episode
stats. No model and no training, just a check that the env API behaves.

Run from project root with the venv activated:
    python task11_rl_controller\test_wrapper.py
"""
from __future__ import annotations

import time

import numpy as np

from ot2_gym_wrapper import OT2Env


N_STEPS = 1000
SEED = 20260521


def main():
    env = OT2Env(seed=SEED, max_steps=200, randomise_start=True)
    obs, info = env.reset(seed=SEED)
    print(f"observation space: {env.observation_space}")
    print(f"action space:      {env.action_space}")
    print(f"initial obs (9,):  {obs}")
    print(f"initial info:      {info}")
    print()

    n_episodes = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    final_distances: list[float] = []
    cur_return = 0.0
    cur_length = 0
    cur_final_dist = None

    t0 = time.time()
    for step in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        cur_return += reward
        cur_length += 1
        cur_final_dist = info["distance_m"]

        if terminated or truncated:
            n_episodes += 1
            episode_returns.append(cur_return)
            episode_lengths.append(cur_length)
            final_distances.append(cur_final_dist or float("nan"))
            why = "TERM" if terminated else "TRUNC"
            print(f"  ep{n_episodes:3d}  step={step:4d}  len={cur_length:3d}  "
                  f"return={cur_return:+7.2f}  final_dist={cur_final_dist * 1000:6.2f} mm  [{why}]")
            cur_return = 0.0
            cur_length = 0
            obs, info = env.reset()

    elapsed = time.time() - t0
    env.close()

    print("\n=== summary ===")
    print(f"  total env steps:        {N_STEPS}")
    print(f"  wall time:              {elapsed:.2f} s ({N_STEPS / elapsed:.1f} steps/s)")
    print(f"  full episodes:          {n_episodes}")
    if n_episodes:
        print(f"  mean episode return:    {np.mean(episode_returns):+.2f} (random actions, expected to be negative)")
        print(f"  mean episode length:    {np.mean(episode_lengths):.1f} steps")
        print(f"  mean final distance:    {np.mean(final_distances) * 1000:.2f} mm")
    print("\nOK")


if __name__ == "__main__":
    main()
