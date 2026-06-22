"""Gymnasium environment for the OT-2 pipette-positioning task.

Where this departs from a stock Gym wrapper, it's because of what Task 9 turned up:

Targets and the start pose live in the practical envelope (z >= 0.170 m), not the
raw joint-limit box. The chassis floor blocks descent below ~0.1695 m except in a
narrow xy patch near the initial pose, so anything lower wouldn't be reachable from
an arbitrary start.

Randomised starts go through `ot2.teleport.teleport_pipette` rather than the
upstream `Simulation.set_start_position`, which has a kinematic sign error (see
ot2/teleport.py). teleport_pipette does the resetJointState with the correct inverse.

The reward is dense and deliberately smooth: a potential-based shaping term, a
couple of proximity wells, a dwell bonus, a terminal success bonus, and a small
time cost. Earlier versions used discontinuous "milestone" bonuses, which gave the
policy reward jumps it had trouble climbing.

The observation is 9-D: pipette position normalised to [-1, 1], joint velocities,
and the relative vector to the target. The target isn't passed explicitly; it shows
up in that relative vector.

Actions are 3-D continuous velocity commands in [-1, 1]. The sim's fourth `drop`
action is pinned to 0, since dispensing isn't part of the Task 11 objective.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ot2.envelope import (
    INITIAL_PIPETTE,
    PRACTICAL_ENVELOPE,
    Envelope,
    sample_target,
)
from ot2.sim import make_sim, precise_pipette_position, precise_joint_velocity
from ot2.teleport import teleport_pipette


class OT2Env(gym.Env):
    """Gymnasium env wrapping the OT-2 PyBullet sim.

    Observation (9,):
        [px, py, pz, vx, vy, vz, rx, ry, rz]

        - p* = pipette position in workspace coords, normalised to [-1, 1]
          against the practical envelope.
        - v* = joint velocity (rad/s for revolute, m/s for prismatic; sim's
          three joints are all prismatic so units are m/s). Not normalised
          because magnitudes are O(1) already.
        - r* = target - pipette in metres (un-normalised, so the agent gets
          true distance signal). Typical magnitude is well under 1 m.

    Action (3,):
        [vx_cmd, vy_cmd, vz_cmd] in [-1, 1]. Drop is forced to 0.

    Reward, per step (see _compute_reward for the exact constants):
        distance-improvement shaping, plus three exp() proximity wells at
        decreasing length scales (10 mm, 0.3 mm, 0.05 mm) so the pull keeps
        growing as the error shrinks, minus a velocity penalty that only bites
        inside 2 cm, minus a small per-step time cost.
    Terminal:
        large bonus once the pipette has dwelt inside success_threshold for
        dwell_steps; penalty if it leaves the safe zone.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 240}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 1000,
        success_threshold: float = 1.0e-3,  # 1 mm, matches assignment spec
        dwell_steps: int = 10,               # short dwell so success commits quickly
        envelope: Envelope = PRACTICAL_ENVELOPE,
        boundary_margin: float = 0.01,       # terminate if outside envelope - margin
        randomise_start: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.dwell_steps = dwell_steps
        self.envelope = envelope
        self.boundary_margin = boundary_margin
        self.randomise_start = randomise_start

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32,
        )

        render = render_mode == "human"
        rgb_array = render_mode == "rgb_array"
        self.sim = make_sim(num_agents=1, render=render, rgb_array=rgb_array)

        self._np_random = np.random.default_rng(seed)

        # Episode state.
        self._steps = 0
        self._dwell = 0
        self._target: np.ndarray = INITIAL_PIPETTE.copy()
        self._prev_distance: float | None = None
        self._robot_key: str = ""

    # ---------- Gymnasium API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Target
        if options and "target" in options:
            self._target = np.asarray(options["target"], dtype=np.float64)
        else:
            self._target = sample_target(self._np_random, envelope=self.envelope, margin=0.01)

        # Reset sim and teleport to a safe start. The sim's initial pose
        # (z=0.1195) is below the practical envelope, so an episode that
        # starts there would immediately fail the safe-zone check on step 1.
        # Always teleport: a random envelope-interior point if randomising,
        # else the envelope centre (a deterministic, eval-friendly start).
        self.sim.reset(num_agents=1)
        self.sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)  # let initial state settle
        self._robot_key = next(iter(self.sim.get_states()))

        if self.randomise_start:
            start = sample_target(self._np_random, envelope=self.envelope, margin=0.02)
        else:
            start = self.envelope.center
        teleport_pipette(self.sim, start)
        self.sim.run([[0.0, 0.0, 0.0, 0]], num_steps=1)  # settle after teleport

        self._steps = 0
        self._dwell = 0
        self._prev_distance = None

        obs = self._observe()
        info = self._info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        sim_action = [float(action[0]), float(action[1]), float(action[2]), 0]
        self.sim.run([sim_action], num_steps=1)
        self._steps += 1

        obs = self._observe()
        pipette = self._pipette()
        distance = float(np.linalg.norm(self._target - pipette))

        reward = self._compute_reward(pipette, distance)
        self._prev_distance = distance

        # Track dwell
        if distance < self.success_threshold:
            self._dwell += 1
        else:
            self._dwell = 0

        terminated = False
        if self._dwell >= self.dwell_steps:
            # Make the terminal bonus large enough that success clearly outweighs
            # the per-step proximity wells, otherwise the agent is happy to loiter
            # near the target and farm shaping reward instead of committing. Paired
            # with a tight success_threshold (training passes 0.5 mm) so it has to
            # actually be precise to collect it.
            reward += 500.0
            terminated = True
        elif not self._in_safe_zone(pipette):
            reward -= 10.0
            terminated = True

        truncated = self._steps >= self.max_steps

        info = self._info()
        info["distance_m"] = distance
        info["dwell"] = self._dwell
        info["pipette"] = pipette.tolist()
        info["target"] = self._target.tolist()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return getattr(self.sim, "current_frame", None)
        return None

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass

    # ---------- internals ----------

    def _pipette(self) -> np.ndarray:
        """float64 pose, bypassing the sim's 0.1 mm rounding (see ot2.sim)."""
        return precise_pipette_position(self.sim)

    def _velocity(self) -> np.ndarray:
        """High-precision joint velocity (float64) m/s on each axis."""
        return precise_joint_velocity(self.sim)

    def _observe(self) -> np.ndarray:
        pipette = self._pipette()
        velocity = self._velocity()
        rel = self._target - pipette
        # normalise pipette to [-1, 1] against the envelope
        low, high = self.envelope.low, self.envelope.high
        p_norm = 2.0 * (pipette - low) / (high - low) - 1.0
        return np.concatenate([p_norm, velocity, rel]).astype(np.float32)

    def _compute_reward(self, pipette: np.ndarray, distance: float) -> float:
        """Sub-mm shaping reward.

        An earlier version handed out a flat +1 per step while inside 1 mm, so
        once the agent could park inside 1 mm it saw no reason to do better. The
        fix is to replace that flat bonus with proximity wells at successively
        finer length scales, so there's always a gradient pointing closer right
        down to ~0.1 mm. The pieces below, in order: distance-improvement
        shaping to drive the approach, three exp() wells (10 mm / 0.3 mm /
        0.05 mm), a velocity penalty that only matters once you're inside 2 cm,
        and a small time cost.
        """
        reward = 0.0

        # Distance-improvement shaping (the main driver of the approach).
        if self._prev_distance is not None:
            reward += 30.0 * (self._prev_distance - distance)

        # Coarse proximity well, 10 mm scale: a gentle pull once roughly on target.
        reward += 0.2 * float(np.exp(-distance / 0.01))

        # Sub-mm well, 0.3 mm scale. Weight 5 keeps a useful gradient through
        # the 1 mm -> 0.1 mm range (e.g. ~0.18 at 1 mm, ~0.94 at 0.5 mm).
        reward += 5.0 * float(np.exp(-distance / 0.0003))

        # Ultra-fine well, 0.05 mm scale. Effectively flat above ~0.1 mm and
        # dominant below it, so it's what pulls the policy from sub-mm toward
        # sub-100 um (weight 20 at the target, ~2.7 at 0.1 mm).
        reward += 20.0 * float(np.exp(-distance / 0.00005))

        # Penalise speed near the target so it doesn't overshoot; ramps in
        # linearly from 2 cm down to 0.
        proximity_factor = max(0.0, 1.0 - distance / 0.02)
        vel_mag = float(np.linalg.norm(self._velocity()))
        reward -= 0.3 * proximity_factor * vel_mag

        # Small time cost to discourage dawdling.
        reward -= 0.02

        return reward

    def _in_safe_zone(self, pipette: np.ndarray) -> bool:
        return self.envelope.contains(pipette, margin=-self.boundary_margin)

    def _info(self) -> dict[str, Any]:
        return {
            "steps": self._steps,
            "dwell": self._dwell,
            "envelope": (self.envelope.low.tolist(), self.envelope.high.tolist()),
        }
