"""Repeatability: drive to one fixed target from N random start poses, measure final-pose scatter.

The sim is fully deterministic for a given trajectory, so identical
(start, target) -> identical (final). The scientifically interesting
question is how much the final pose varies as a function of the *start*
pose, which is what affects automation accuracy in practice (the robot
doesn't always start from rest).

Method:
    1. Pick one fixed target (envelope centre).
    2. Sample N starts uniformly from the practical envelope (with margin).
    3. For each start, reset + teleport to that start + drive to the target
       with the tuned PID.
    4. Record the final pose at the moment of settle.
    5. Compute per-axis stddev, range, mean, max scalar distance from target.

Run:
    python task09_robotics_environment\repeatability.py

Writes:
    results/repeatability.json
    results/repeatability_scatter.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ot2.control.pid import XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DT = 1.0 / 240.0
N_TRIALS = 50
MAX_STEPS = 700
SETTLE_THRESHOLD = 1.0e-3
SETTLE_HOLD_STEPS = 15
SEED = 20260601

# Use the v3 final SIMC-tuned gains (loaded from Task 10 results if present).
T10_GAINS = HERE.parent / "task10_pid_controller" / "results" / "gains_final.json"


def make_pid() -> XYZPID:
    if T10_GAINS.exists():
        d = json.loads(T10_GAINS.read_text())
        return XYZPID.from_dict(d["pid"])
    # fallback if Task 10 v3 hasn't been run
    from ot2.control.pid import AxisPID
    return XYZPID(
        x=AxisPID(kp=75.0, ki=4.0, settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=75.0, ki=4.0, settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=52.5, ki=3.0, settle_threshold=0.0, min_output=0.0),
    )


def run_to_target(sim, pid: XYZPID, target: np.ndarray) -> dict:
    """Run the controller for the full MAX_STEPS so the integral term gets to
    sub-um precision. Records the step at which the 1 mm settle was first
    achieved (for stats) but does not exit at that point."""
    pid.reset()
    settle_step = None
    consecutive_under = 0
    final_pos = precise_pipette_position(sim)
    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        final_pos = pos
        err = float(np.linalg.norm(target - pos))
        if err < SETTLE_THRESHOLD:
            consecutive_under += 1
            if consecutive_under >= SETTLE_HOLD_STEPS and settle_step is None:
                settle_step = step
        else:
            consecutive_under = 0
        a = pid.step(target, pos, dt=DT)
        sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
    return {"final_pos": final_pos.tolist(),
            "settle_step": settle_step if settle_step is not None else MAX_STEPS,
            "settled": settle_step is not None,
            "err_mm": float(np.linalg.norm(target - final_pos)) * 1000.0}


def main():
    target = PRACTICAL_ENVELOPE.center
    rng = np.random.default_rng(SEED)

    pid = make_pid()
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    starts = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.02) for _ in range(N_TRIALS)]
    finals: list[np.ndarray] = []
    rows: list[dict] = []
    for i, start in enumerate(starts):
        sim.reset(num_agents=1)
        sim.run([[0, 0, 0, 0]], num_steps=20)
        teleport_pipette(sim, start)
        sim.run([[0, 0, 0, 0]], num_steps=2)
        res = run_to_target(sim, pid, target)
        finals.append(np.asarray(res["final_pos"]))
        rows.append({"trial": i, "start": start.tolist(), **res})
        if i % 10 == 0:
            print(f"  trial {i:3d}/{N_TRIALS}  err = {res['err_mm']:.4f} mm  settle = {res['settle_step']}")
    sim.close()

    F = np.asarray(finals)
    delta = F - target
    per_axis_mean_mm = delta.mean(axis=0) * 1000.0
    per_axis_std_mm = delta.std(axis=0, ddof=1) * 1000.0
    per_axis_max_mm = np.abs(delta).max(axis=0) * 1000.0
    scalar_err_mm = np.linalg.norm(F - target, axis=1) * 1000.0
    success = sum(1 for r in rows if r["settled"]) / len(rows)

    print(f"\n=== repeatability across {N_TRIALS} random starts ===")
    print(f"  success rate (<{SETTLE_THRESHOLD*1000} mm settle):    {success*100:.0f}%")
    print(f"  per-axis bias (mean error vs target):       x={per_axis_mean_mm[0]:+.5f}  y={per_axis_mean_mm[1]:+.5f}  z={per_axis_mean_mm[2]:+.5f}  mm")
    print(f"  per-axis stddev (1-sigma scatter):          x={per_axis_std_mm[0]:7.5f}  y={per_axis_std_mm[1]:7.5f}  z={per_axis_std_mm[2]:7.5f}  mm")
    print(f"  per-axis worst case (max |dev|):            x={per_axis_max_mm[0]:7.5f}  y={per_axis_max_mm[1]:7.5f}  z={per_axis_max_mm[2]:7.5f}  mm")
    print(f"  scalar final error: median={np.median(scalar_err_mm):.5f}  max={scalar_err_mm.max():.5f}  mm")

    out_json = RESULTS / "repeatability.json"
    out_json.write_text(json.dumps({
        "target": target.tolist(),
        "n_trials": N_TRIALS,
        "settle_threshold_m": SETTLE_THRESHOLD,
        "settle_hold_steps": SETTLE_HOLD_STEPS,
        "seed": SEED,
        "success_rate": success,
        "per_axis_bias_mm": per_axis_mean_mm.tolist(),
        "per_axis_std_mm": per_axis_std_mm.tolist(),
        "per_axis_max_mm": per_axis_max_mm.tolist(),
        "scalar_err_stats_mm": {
            "median": float(np.median(scalar_err_mm)),
            "mean": float(scalar_err_mm.mean()),
            "max": float(scalar_err_mm.max()),
            "min": float(scalar_err_mm.min()),
        },
        "trials": rows,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # 3-panel scatter
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, (label, ax) in enumerate(zip(["x vs y", "x vs z", "y vs z"], axes)):
        if i == 0:
            ax.scatter(delta[:, 0] * 1000, delta[:, 1] * 1000, alpha=0.6, color="#4C9AFF")
            ax.set_xlabel("delta x (mm)"); ax.set_ylabel("delta y (mm)")
        elif i == 1:
            ax.scatter(delta[:, 0] * 1000, delta[:, 2] * 1000, alpha=0.6, color="#FF6B6B")
            ax.set_xlabel("delta x (mm)"); ax.set_ylabel("delta z (mm)")
        else:
            ax.scatter(delta[:, 1] * 1000, delta[:, 2] * 1000, alpha=0.6, color="#54B567")
            ax.set_xlabel("delta y (mm)"); ax.set_ylabel("delta z (mm)")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(label)
    fig.suptitle(f"Repeatability: final-pose deviations from target across {N_TRIALS} random starts")
    fig.tight_layout()
    out_png = RESULTS / "repeatability_scatter.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
