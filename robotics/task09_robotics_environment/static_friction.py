"""Characterise the per-axis velocity-control dead zone (static friction).

Method: from a fresh reset and 20-step settling, apply a constant velocity
command of magnitude `v` for a short window and measure the pipette
displacement on that axis. Repeat for a sweep of `v` per axis per direction
and find the smallest magnitude that produces a non-trivial motion.

Run from the project root:
    python task09_robotics_environment\static_friction.py

Writes:
    results/static_friction.json: per-axis dead-zone characterisation
    results/static_friction.png:  plot of displacement vs commanded magnitude
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
PROBE_STEPS = 30
MOTION_THRESHOLD_M = 1e-5     # 10 um, "did the carriage move at all"
V_SWEEP = np.round(np.arange(0.0, 0.41, 0.02), 4)  # 0, 0.02, 0.04, ..., 0.40


def probe(sim, axis: int, sign: float, v_mag: float) -> float:
    """Apply a constant velocity command, return the per-axis displacement in m."""
    sim.reset(num_agents=1)
    sim.run([[0, 0, 0, 0]], num_steps=20)
    p0 = precise_pipette_position(sim)
    action = [0.0, 0.0, 0.0, 0]
    action[axis] = sign * v_mag
    sim.run([action], num_steps=PROBE_STEPS)
    p1 = precise_pipette_position(sim)
    return float(p1[axis] - p0[axis])


def main():
    sim = make_sim(num_agents=1, render=False)

    data: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, ax in zip((0, 1, 2), axes):
        label = "xyz"[axis]
        data[label] = {}
        for sign, color in [(+1.0, "#4C9AFF"), (-1.0, "#FF6B6B")]:
            displacements_mm = []
            min_moving = None
            for v in V_SWEEP:
                if v == 0:
                    displacements_mm.append(0.0)
                    continue
                d = probe(sim, axis, sign, v)
                displacements_mm.append(d * 1000.0)
                if min_moving is None and abs(d) > MOTION_THRESHOLD_M:
                    min_moving = float(v)
            data[label][f"sign_{int(sign):+d}"] = {
                "v_sweep": V_SWEEP.tolist(),
                "displacement_mm": displacements_mm,
                "min_moving_v": min_moving,
            }
            ax.plot(V_SWEEP, displacements_mm, "o-", color=color, label=f"sign={int(sign):+d}, dead-zone>={min_moving}")
            print(f"axis={label}  sign={int(sign):+d}  min-moving v = {min_moving}")
        ax.set_xlabel("|action|")
        if axis == 0:
            ax.set_ylabel(f"displacement after {PROBE_STEPS} steps (mm)")
        ax.set_title(f"axis {label}")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    sim.close()

    fig.suptitle(f"Static-friction sweep: pipette displacement after {PROBE_STEPS} sim steps (~{PROBE_STEPS*DT:.3f}s) at fixed |action|")
    fig.tight_layout()
    plot = RESULTS / "static_friction.png"
    fig.savefig(plot, dpi=140)
    plt.close(fig)
    print(f"wrote {plot}")

    out = RESULTS / "static_friction.json"
    out.write_text(json.dumps({
        "settings": {"probe_steps": PROBE_STEPS, "motion_threshold_m": MOTION_THRESHOLD_M, "v_sweep": V_SWEEP.tolist()},
        "axes": data,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
