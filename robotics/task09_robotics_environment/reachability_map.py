"""Quantify the chassis-collision constraint by sampling reachability at z=z_min.

Method (corrected): for each (x, y) on a grid at z = z_min, first teleport
the pipette to (x, y, z_safe) above the chassis, then drive it straight
down. Record the lowest z actually achieved. Where the chassis blocks
descent the final z will be well above z_min.

Run from project root:
    python task09_robotics_environment\reachability_map.py

Writes:
    results/reachability.json: grid + reachable mask
    results/reachability_map.png: heatmap of final-z per (x, y)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ot2.envelope import JOINT_LIMIT_ENVELOPE
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GRID_X = 11
GRID_Y = 11
MAX_STEPS = 400
DT = 1.0 / 240.0
REACHED_TOL_M = 5e-4   # within 0.5 mm of z_min counts as reached
Z_SAFE = 0.25          # known-clear height above chassis
DESCEND_VELOCITY = -1.0  # full negative-z action to push past any static friction
SETTLE_EPS = 5e-6      # |dz| between consecutive steps below this = "stopped"


def descend_until_stop(sim) -> float:
    """Drive z- at max action until the pipette stops moving on z. Return final z."""
    prev = precise_pipette_position(sim)[2]
    stalled = 0
    for step in range(MAX_STEPS):
        sim.run([[0.0, 0.0, DESCEND_VELOCITY, 0]], num_steps=1)
        z = precise_pipette_position(sim)[2]
        if abs(z - prev) < SETTLE_EPS:
            stalled += 1
            if stalled >= 5:  # need 5 consecutive stalled steps
                return float(z)
        else:
            stalled = 0
        prev = z
    return float(prev)


def main():
    env = JOINT_LIMIT_ENVELOPE
    z_target = env.z_min
    xs = np.linspace(env.x_min, env.x_max, GRID_X)
    ys = np.linspace(env.y_min, env.y_max, GRID_Y)

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    z_final = np.zeros((GRID_X, GRID_Y))
    reached = np.zeros((GRID_X, GRID_Y), dtype=bool)
    cells = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            # Teleport to (x, y, Z_SAFE) above the chassis, then drive down.
            teleport_pipette(sim, np.array([x, y, Z_SAFE]))
            sim.run([[0, 0, 0, 0]], num_steps=2)  # settle teleport
            z_min_reached = descend_until_stop(sim)
            z_final[i, j] = z_min_reached
            reached[i, j] = (z_min_reached - z_target) < REACHED_TOL_M
            cells.append({"x": float(x), "y": float(y), "z_target": float(z_target),
                          "z_min_reached_m": z_min_reached, "reached": bool(reached[i, j])})
        print(f"row x={x:+.3f} done  (z_min row range: {z_final[i].min()*1000:.2f}..{z_final[i].max()*1000:.2f} mm)")
    sim.close()

    out_json = RESULTS / "reachability.json"
    out_json.write_text(json.dumps({
        "settings": {"grid_x": GRID_X, "grid_y": GRID_Y, "z_target": z_target,
                     "z_safe": Z_SAFE, "descend_velocity": DESCEND_VELOCITY,
                     "reached_tol_m": REACHED_TOL_M, "max_steps": MAX_STEPS},
        "cells": cells,
        "summary": {
            "n_reached": int(reached.sum()),
            "n_total": int(reached.size),
            "fraction_reached": float(reached.mean()),
            "min_z_anywhere_m": float(z_final.min()),
            "max_z_anywhere_m": float(z_final.max()),
        },
    }, indent=2))
    print(f"\nwrote {out_json}")
    print(f"reached {reached.sum()}/{reached.size} ({reached.mean()*100:.1f}%)")
    print(f"max final z anywhere: {z_final.max():.4f} m (z_target = {z_target:.4f} m)")

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [env.y_min, env.y_max, env.x_min, env.x_max]
    im = ax.imshow(z_final * 1000, extent=extent, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"Final z (mm) when commanding z_target = {z_target*1000:.1f} mm\n"
                 f"Bright = stopped above target (chassis-blocked)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("final z (mm)")
    # mark the initial pose
    ax.plot(0.0895, 0.073, "rx", markersize=12, label="initial pipette xy")
    ax.legend()
    fig.tight_layout()
    out_png = RESULTS / "reachability_map.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
