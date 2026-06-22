"""Smoke test for the sim API and the action sign convention.

Drives all 6 axis-aligned velocity commands (+/- x, y, z) and prints a table of
commanded velocity -> observed pipette displacement, so we know which sign moves
which way without trusting the docs.

Run from the project root:
    python task09_robotics_environment\smoke_test.py

Writes:
    results/smoke_test.json: per-command delta and inferred velocity gain
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ot2.sim import make_sim, precise_pipette_position, precise_joint_velocity


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
N_STEPS = 60                     # 0.25 s of motion per probe
VELOCITY_COMMAND = 0.5           # action magnitude, valid range is [-1, 1]


def probe(sim, axis: int, sign: float) -> dict:
    """Run one axis at the given sign for N_STEPS and return the resulting state."""
    sim.reset(num_agents=1)
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)  # let it settle before measuring

    p_start = precise_pipette_position(sim)

    action = [0.0, 0.0, 0.0, 0]
    action[axis] = sign * VELOCITY_COMMAND
    sim.run([action], num_steps=N_STEPS)

    p_end = precise_pipette_position(sim)
    v_end = precise_joint_velocity(sim)
    delta = p_end - p_start
    # m moved per second per unit of action, so probes at different commands compare
    gain = delta[axis] / (N_STEPS * DT) / (sign * VELOCITY_COMMAND)
    return {
        "axis": "xyz"[axis],
        "sign": sign,
        "action_component": sign * VELOCITY_COMMAND,
        "p_start": p_start.tolist(),
        "p_end": p_end.tolist(),
        "delta": delta.tolist(),
        "axis_delta_m": float(delta[axis]),
        "inferred_velocity_gain_m_per_s_per_unit_action": float(gain),
        "joint_velocity_at_end": v_end.tolist(),
    }


def main():
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)
    p0 = precise_pipette_position(sim)
    print(f"initial pipette pose: {p0}")
    print(f"sim.pipette_offset:   {sim.pipette_offset}\n")

    rows = []
    headers = ["axis", "sign", "action", "delta_x_mm", "delta_y_mm", "delta_z_mm", "gain_m/s/unit"]
    print(f"{headers[0]:>4} {headers[1]:>4} {headers[2]:>7} {headers[3]:>11} {headers[4]:>11} {headers[5]:>11} {headers[6]:>14}")
    for axis in (0, 1, 2):
        for sign in (+1.0, -1.0):
            r = probe(sim, axis, sign)
            rows.append(r)
            d = r["delta"]
            print(f"{r['axis']:>4} {r['sign']:>+4} {r['action_component']:>+7.2f} "
                  f"{d[0]*1000:>+11.3f} {d[1]*1000:>+11.3f} {d[2]*1000:>+11.3f} "
                  f"{r['inferred_velocity_gain_m_per_s_per_unit_action']:>+14.4f}")
    sim.close()

    out = RESULTS / "smoke_test.json"
    out.write_text(json.dumps({
        "settings": {"velocity_command": VELOCITY_COMMAND, "n_steps": N_STEPS, "dt": DT},
        "initial_pose": p0.tolist(),
        "probes": rows,
    }, indent=2))
    print(f"\nwrote {out}")

    print("\n=== sign convention verification ===")
    for r in rows:
        ok = "OK" if (r["axis_delta_m"] * r["sign"]) > 0 else "FAIL"
        print(f"  action[{r['axis']}]={r['action_component']:+.2f}  ->  pipette {r['axis']}={r['axis_delta_m']*1000:+.3f} mm   [{ok}]")


if __name__ == "__main__":
    main()
