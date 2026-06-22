"""Wrapper around the upstream `sim_class.Simulation`.

The upstream sim loads `textures/` with a relative path in its constructor
(`os.listdir("textures")`), and the `meshes/` and `*.urdf` files are likewise
looked up relative to cwd. So we chdir into the sim folder before constructing
it and leave the cwd there, which lets us call it from any task folder.

    from ot2.sim import make_sim
    sim = make_sim(num_agents=1, render=False)   # headless
    state = sim.run([[0.0, 0.0, 0.0, 0]], num_steps=30)
    sim.close()
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator


# Upstream sim repo, expected three levels up from this file (next to src/).
SIM_DIR: Path = (Path(__file__).resolve().parent.parent.parent / "Y2B-2023-OT2_Twin").resolve()


def sim_dir() -> Path:
    """Absolute path to the upstream sim folder."""
    if not SIM_DIR.exists():
        raise FileNotFoundError(
            f"Upstream sim folder not found at {SIM_DIR}. "
            "Re-fetch from the BUas LMS zip and place it at this path."
        )
    return SIM_DIR


@contextlib.contextmanager
def with_sim_cwd() -> Iterator[Path]:
    """Context manager that chdirs into the sim folder and restores on exit.

    Use when calling anything that touches the sim's relative file paths
    (`p.loadURDF`, `p.loadTexture`, etc.) outside of constructor time.
    """
    prev = os.getcwd()
    os.chdir(sim_dir())
    try:
        yield sim_dir()
    finally:
        os.chdir(prev)


def _ensure_sim_importable() -> None:
    """Make `sim_class` importable from any caller."""
    sim_path = str(sim_dir())
    if sim_path not in sys.path:
        sys.path.insert(0, sim_path)


def make_sim(num_agents: int = 1, render: bool = False, rgb_array: bool = False):
    """Construct a `sim_class.Simulation` with the cwd correctly set.

    Args:
        num_agents: number of OT-2 robots in the world (we only ever use 1).
        render: True opens a PyBullet GUI window. False runs in p.DIRECT (headless).
        rgb_array: True enables per-step offscreen camera capture (slow, avoid).

    Returns:
        A `sim_class.Simulation` instance. The cwd remains in the sim folder so
        subsequent `sim.run(...)` / `sim.reset(...)` calls keep working.
    """
    _ensure_sim_importable()
    os.chdir(sim_dir())  # stay here so loadURDF calls inside Simulation resolve
    from sim_class import Simulation  # type: ignore[import-not-found]
    return Simulation(num_agents=num_agents, render=render, rgb_array=rgb_array)


def precise_pipette_position(sim, robot_index: int = 0):
    """Pipette pose at float64 precision, read straight from the joint states.

    `sim.get_states()['pipette_position']` rounds to 4 decimals (0.1 mm), which
    hides any error below the rounding floor. RL aiming at sub-mm accuracy needs
    to see those small errors, hence reading the joints directly here.

    Returns a numpy (3,) float64 array of world-frame coords in metres.
    """
    import numpy as np
    import pybullet as p
    robot_id = sim.robotIds[robot_index]
    base = p.getBasePositionAndOrientation(robot_id)[0]
    js = p.getJointStates(robot_id, [0, 1, 2])
    off = sim.pipette_offset
    return np.array([
        base[0] - js[0][0] + off[0],
        base[1] - js[1][0] + off[1],
        base[2] + js[2][0] + off[2],
    ], dtype=np.float64)


def precise_joint_velocity(sim, robot_index: int = 0):
    """Joint velocities (m/s) at float64 precision."""
    import numpy as np
    import pybullet as p
    robot_id = sim.robotIds[robot_index]
    js = p.getJointStates(robot_id, [0, 1, 2])
    return np.array([js[i][1] for i in range(3)], dtype=np.float64)
