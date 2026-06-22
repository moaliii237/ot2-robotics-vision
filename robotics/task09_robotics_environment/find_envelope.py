"""Find the joint-limit work envelope by driving each axis into its stop.

Push one axis at full velocity until its coordinate stops moving between chunks,
do that for all six directions, and the six extremes give the AABB and its 8
corners. Also writes a practical envelope with z_min raised off the chassis
floor, since the downstream tasks want a box they can actually reach into.

Run from the project root:
    python task09_robotics_environment\find_envelope.py

Writes:
    results/envelope.json: joint-limit and practical envelopes + corners + raw sweeps
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

VEL = 1.0           # commanded velocity magnitude on the swept axis
CHUNK = 30          # sim steps per probe chunk
EPSILON = 1e-6      # if a chunk moves less than this, the joint has hit its stop
MAX_CHUNKS = 80     # bail out even if it never settles, so we can't loop forever
DT = 1.0 / 240.0


def drive_until_stop(sim, action, axis_index: int) -> tuple[float, int]:
    """Hold `action` until the coordinate on `axis_index` stops moving.

    Returns (extreme_coord, chunks_used).
    """
    last = None
    for chunk in range(MAX_CHUNKS):
        sim.run([action], num_steps=CHUNK)
        pos = float(precise_pipette_position(sim)[axis_index])
        if last is not None and abs(pos - last) < EPSILON:
            return pos, chunk + 1
        last = pos
    return last, MAX_CHUNKS  # hit the chunk cap without converging


def main():
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=30)
    p0 = precise_pipette_position(sim)
    print(f"start pipette: {p0}")

    sweeps = [
        ("axis_x_pos_action", [+VEL, 0, 0, 0], 0),
        ("axis_x_neg_action", [-VEL, 0, 0, 0], 0),
        ("axis_y_pos_action", [0, +VEL, 0, 0], 1),
        ("axis_y_neg_action", [0, -VEL, 0, 0], 1),
        ("axis_z_pos_action", [0, 0, +VEL, 0], 2),
        ("axis_z_neg_action", [0, 0, -VEL, 0], 2),
    ]
    extremes = {}
    for label, action, axis in sweeps:
        sim.reset(num_agents=1)
        sim.run([[0, 0, 0, 0]], num_steps=20)
        coord, chunks = drive_until_stop(sim, action, axis)
        extremes[label] = {"axis": "xyz"[axis], "action": action,
                           "extreme_coord_m": coord, "chunks_used": chunks}
        print(f"  {label:22s}  {'xyz'[axis]}={coord:+.6f} m  ({chunks} chunks)")

    x_lo = min(extremes["axis_x_pos_action"]["extreme_coord_m"], extremes["axis_x_neg_action"]["extreme_coord_m"])
    x_hi = max(extremes["axis_x_pos_action"]["extreme_coord_m"], extremes["axis_x_neg_action"]["extreme_coord_m"])
    y_lo = min(extremes["axis_y_pos_action"]["extreme_coord_m"], extremes["axis_y_neg_action"]["extreme_coord_m"])
    y_hi = max(extremes["axis_y_pos_action"]["extreme_coord_m"], extremes["axis_y_neg_action"]["extreme_coord_m"])
    z_lo = min(extremes["axis_z_pos_action"]["extreme_coord_m"], extremes["axis_z_neg_action"]["extreme_coord_m"])
    z_hi = max(extremes["axis_z_pos_action"]["extreme_coord_m"], extremes["axis_z_neg_action"]["extreme_coord_m"])

    joint_limit = {"x_min": x_lo, "x_max": x_hi, "y_min": y_lo, "y_max": y_hi, "z_min": z_lo, "z_max": z_hi}
    # Below ~0.170 m the pipette fouls the chassis floor (see reachability_map.py),
    # so clamp z_min there for the envelope the other tasks actually use.
    practical = dict(joint_limit)
    practical["z_min"] = max(joint_limit["z_min"], 0.170)

    corners = []
    for xi, x in enumerate([joint_limit["x_min"], joint_limit["x_max"]]):
        for yi, y in enumerate([joint_limit["y_min"], joint_limit["y_max"]]):
            for zi, z in enumerate([joint_limit["z_min"], joint_limit["z_max"]]):
                idx = xi * 4 + yi * 2 + zi
                corners.append({"index": idx, "xyz_m": [x, y, z]})

    print("\nJoint-limit envelope (m):")
    for k, v in joint_limit.items():
        print(f"  {k}: {v:+.4f}")
    print(f"\nPractical envelope (z_min raised to clear chassis):")
    print(f"  z_min raised from {joint_limit['z_min']:+.4f} to {practical['z_min']:+.4f}")

    out = RESULTS / "envelope.json"
    out.write_text(json.dumps({
        "settings": {"velocity_magnitude": VEL, "chunk_steps": CHUNK, "epsilon_m": EPSILON, "max_chunks": MAX_CHUNKS},
        "joint_limit_envelope": joint_limit,
        "practical_envelope": practical,
        "corners_joint_limit": corners,
        "raw_sweeps": extremes,
    }, indent=2))
    print(f"\nwrote {out}")
    sim.close()


if __name__ == "__main__":
    main()
