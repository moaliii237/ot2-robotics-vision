"""Sign-bug-free pipette teleport.

The upstream `Simulation.set_start_position(x, y, z)` computes
    adjusted_x = x - robot_x - pipette_offset_x
    p.resetJointState(robotId, 0, adjusted_x)

But `get_pipette_position` returns
    pipette_x = robot_x - joint_0 + offset_x   (note the minus sign on joint_0)

So calling `set_start_position(target)` actually puts the pipette at
    2 * robot_x + 2 * offset_x - target_x

i.e. mirrored about `robot_x + offset_x`. This function inverts the sign so the
pipette ends up where the caller asked for.

See task09_robotics_environment/README.md, section "Key findings", point 6.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from .envelope import Envelope, PRACTICAL_ENVELOPE, in_envelope


def teleport_pipette(sim, target: np.ndarray, robot_index: int = 0) -> None:
    """Reset robot joint states so the pipette ends up exactly at `target`.

    Args:
        sim: a `sim_class.Simulation` instance.
        target: (3,) array of world-frame coordinates in metres.
        robot_index: which agent to teleport. Defaults to first robot.

    Raises:
        ValueError: if target is not finite or not in any usable envelope.
    """
    t = np.asarray(target, dtype=np.float64)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        raise ValueError(f"target must be a finite 3-vector, got {target!r}")

    robot_id = sim.robotIds[robot_index]
    robot_pos = p.getBasePositionAndOrientation(robot_id)[0]
    offset = sim.pipette_offset  # [x, y, z], metres

    # Forward model (sim_class.get_pipette_position):
    #   pipette = robot - [joint_0, joint_1, -joint_2] + offset
    # so the inverse is:
    #   joint_0 = robot_x + offset_x - pipette_x
    #   joint_1 = robot_y + offset_y - pipette_y
    #   joint_2 = pipette_z - robot_z - offset_z
    joint_0 = robot_pos[0] + offset[0] - t[0]
    joint_1 = robot_pos[1] + offset[1] - t[1]
    joint_2 = t[2] - robot_pos[2] - offset[2]

    p.resetJointState(robot_id, 0, targetValue=joint_0, targetVelocity=0.0)
    p.resetJointState(robot_id, 1, targetValue=joint_1, targetVelocity=0.0)
    p.resetJointState(robot_id, 2, targetValue=joint_2, targetVelocity=0.0)


def teleport_safe(sim, target: np.ndarray, envelope: Envelope = PRACTICAL_ENVELOPE,
                  robot_index: int = 0) -> None:
    """Like `teleport_pipette` but refuses to place the pipette outside `envelope`."""
    if not in_envelope(target, envelope=envelope):
        raise ValueError(
            f"target {target} is outside the {envelope}. "
            "Use teleport_pipette() if you know what you're doing."
        )
    teleport_pipette(sim, target, robot_index=robot_index)
