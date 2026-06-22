"""Quantify the reachable workspace volume.

Computes:
    V_joint_limit   = box volume of the joint-limit envelope (geometric).
    V_practical     = box volume of the practical envelope (chassis-aware).
    V_monte_carlo   = Monte Carlo estimate using N random samples in the
                      joint-limit box, attempting to teleport+drive to each
                      and counting the reachable fraction.

The Monte Carlo number should match V_practical / V_joint_limit (modulo the
sample noise) if the chassis is a clean horizontal plane at z = 0.1695 m.

Run:
    python task09_robotics_environment\workspace_volume.py

Writes:
    results/workspace_volume.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ot2.envelope import JOINT_LIMIT_ENVELOPE, PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DT = 1.0 / 240.0
N_SAMPLES = 200
MAX_DESCEND_STEPS = 200
SETTLE_EPS = 5e-6
REACHED_TOL_M = 1e-3   # 1 mm
Z_SAFE = 0.25


def can_reach(sim, target: np.ndarray) -> bool:
    """Teleport above the target xy at z_safe, descend, then check if we reached the target z."""
    sim.reset(num_agents=1)
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)
    teleport_pipette(sim, np.array([target[0], target[1], Z_SAFE]))
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=2)

    # Drive down with full-velocity command until stall, then check if we'd land near target z.
    prev_z = precise_pipette_position(sim)[2]
    stalled = 0
    z_min_reached = prev_z
    for _ in range(MAX_DESCEND_STEPS):
        sim.run([[0.0, 0.0, -1.0, 0]], num_steps=1)
        z = precise_pipette_position(sim)[2]
        z_min_reached = min(z_min_reached, z)
        if abs(z - prev_z) < SETTLE_EPS:
            stalled += 1
            if stalled >= 5:
                break
        else:
            stalled = 0
        prev_z = z

    # Was the target z above the stall height? If yes, reachable.
    return z_min_reached - REACHED_TOL_M <= target[2]


def main():
    v_joint = float(np.prod(JOINT_LIMIT_ENVELOPE.high - JOINT_LIMIT_ENVELOPE.low))
    v_prac = float(np.prod(PRACTICAL_ENVELOPE.high - PRACTICAL_ENVELOPE.low))
    print("Geometric envelope volumes:")
    print(f"  joint-limit AABB:  {v_joint*1e6:8.2f} cm^3")
    print(f"  practical:         {v_prac*1e6:8.2f} cm^3")
    print(f"  practical / joint = {v_prac / v_joint:.3f}")

    rng = np.random.default_rng(20260522)
    # Sample inside the joint-limit AABB.
    samples = rng.uniform(JOINT_LIMIT_ENVELOPE.low, JOINT_LIMIT_ENVELOPE.high, size=(N_SAMPLES, 3))

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    reached = np.zeros(N_SAMPLES, dtype=bool)
    for i, s in enumerate(samples):
        reached[i] = can_reach(sim, s)
        if (i + 1) % 20 == 0:
            print(f"  monte carlo {i+1:3d}/{N_SAMPLES} done, frac reachable so far = {reached[:i+1].mean():.3f}")
    sim.close()

    frac = float(reached.mean())
    v_mc = v_joint * frac
    print(f"\nMonte Carlo reachable fraction (N={N_SAMPLES}): {frac:.3f}")
    print(f"Monte Carlo reachable volume: {v_mc*1e6:.2f} cm^3 "
          f"(geometric practical: {v_prac*1e6:.2f} cm^3, error vs geometric: {(v_mc-v_prac)/v_prac*100:+.1f}%)")

    out = RESULTS / "workspace_volume.json"
    out.write_text(json.dumps({
        "joint_limit_volume_m3": v_joint,
        "practical_volume_m3": v_prac,
        "monte_carlo_volume_m3": v_mc,
        "monte_carlo_fraction_reachable": frac,
        "n_samples": N_SAMPLES,
        "geometric_vs_mc_relative_error": (v_mc - v_prac) / v_prac,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
