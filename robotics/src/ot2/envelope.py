"""Work envelope constants for the OT-2 sim.

Two envelopes are exposed:

- JOINT_LIMIT_ENVELOPE: the raw AABB from `find_envelope.py` (Task 9). This is
  the largest box reachable when driving one axis at a time to its joint stop.
- PRACTICAL_ENVELOPE: chassis-collision-constrained subset. z_min is raised
  from +0.1195 to +0.170 because the OT-2 chassis floor blocks pipette descent
  below ~0.1695 m except in a small zone near the initial pose. Use this for
  RL target sampling and PID demo targets so any goal is reachable from any
  starting pose.

See task09_robotics_environment/README.md for the empirical derivation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Envelope:
    """Axis-aligned bounding box in world frame, metres."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def low(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min], dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max, self.z_max], dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    @property
    def extent(self) -> np.ndarray:
        return self.high - self.low

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=np.float64)
        return bool(np.all(p >= self.low + margin) and np.all(p <= self.high - margin))


# Joint-limit AABB, world frame, metres. Source: find_envelope.py output (Task 9).
JOINT_LIMIT_ENVELOPE = Envelope(
    x_min=-0.1870, x_max=+0.2530,
    y_min=-0.1705, y_max=+0.2195,
    z_min=+0.1195, z_max=+0.2895,
)

# Practical envelope: same x/y, but z floor raised above the chassis interference.
PRACTICAL_ENVELOPE = Envelope(
    x_min=-0.1870, x_max=+0.2530,
    y_min=-0.1705, y_max=+0.2195,
    z_min=+0.1700, z_max=+0.2895,
)

# Pipette pose at sim reset (= sim.pipette_offset in sim_class.py).
INITIAL_PIPETTE = np.array([0.073, 0.0895, 0.1195], dtype=np.float64)


def sample_target(
    rng: np.random.Generator | None = None,
    envelope: Envelope = PRACTICAL_ENVELOPE,
    margin: float = 0.005,
) -> np.ndarray:
    """Uniform sample of a target point inside the envelope, with a margin.

    Args:
        rng: numpy Generator. If None, a fresh `default_rng()` is used.
        envelope: which envelope to sample from. Default is the practical one.
        margin: shrink each axis by this many metres on both sides (default 5 mm)
            so targets are never right at the wall.
    """
    rng = rng if rng is not None else np.random.default_rng()
    low = envelope.low + margin
    high = envelope.high - margin
    return rng.uniform(low, high).astype(np.float64)


def in_envelope(
    p: np.ndarray,
    envelope: Envelope = PRACTICAL_ENVELOPE,
    margin: float = 0.0,
) -> bool:
    """Convenience: is `p` inside `envelope` with an optional inset margin?"""
    return envelope.contains(np.asarray(p, dtype=np.float64), margin=margin)
