"""Phase 1: PID gain grid search with high-precision pose feedback.

Unlike v1 this feeds the PID float64 pose from precise_pipette_position
instead of the 0.1 mm rounded readout, drops the settle-threshold and
min_output work-arounds v1 needed, and puts Kd > 0 back in the grid. v1's
Kd > 0 cases all failed on quantisation noise; with clean feedback the
derivative term has a chance.

Run:
    python task10_pid_controller\tune_pid.py
"""
from __future__ import annotations

import csv
import itertools
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ot2.control.pid import AxisPID, XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
MAX_STEPS = 700
SETTLE_THRESHOLD_FOR_SCORING = 1.0e-3   # 1 mm = client requirement
SETTLE_HOLD_STEPS = 15
N_TARGETS = 6
SEED = 20260521

KP_XY_GRID = [25.0, 35.0, 45.0]
KI_XY_GRID = [0.0, 2.0, 4.0, 8.0]
KD_XY_GRID = [0.0, 0.5]
Z_SCALE = {"kp": 0.7, "ki": 0.75, "kd": 0.75}


def make_pid(kp: float, ki: float, kd: float) -> XYZPID:
    return XYZPID(
        x=AxisPID(kp=kp, ki=ki, kd=kd, settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=kp, ki=ki, kd=kd, settle_threshold=0.0, min_output=0.0),
        z=AxisPID(
            kp=kp * Z_SCALE["kp"], ki=ki * Z_SCALE["ki"], kd=kd * Z_SCALE["kd"],
            settle_threshold=0.0, min_output=0.0,
        ),
    )


def run_episode(sim, pid: XYZPID, target: np.ndarray) -> dict:
    pid.reset()
    settle_step = None
    consecutive_under = 0
    final_err = float("inf")
    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        err = float(np.linalg.norm(target - pos))
        final_err = err
        if err < SETTLE_THRESHOLD_FOR_SCORING:
            consecutive_under += 1
            if consecutive_under >= SETTLE_HOLD_STEPS and settle_step is None:
                settle_step = step
        else:
            consecutive_under = 0
        action = pid.step(target, pos, dt=DT)
        sim.run([[float(action[0]), float(action[1]), float(action[2]), 0]], num_steps=1)
    return {"settled": settle_step is not None,
            "settle_step": settle_step if settle_step is not None else MAX_STEPS,
            "final_err_mm": final_err * 1000.0}


def aggregate(per_target):
    n = len(per_target)
    successes = sum(1 for r in per_target if r["settled"])
    errs = [r["final_err_mm"] for r in per_target]
    settles = [r["settle_step"] for r in per_target if r["settled"]]
    return {
        "n_targets": n,
        "success_rate": successes / n,
        "median_final_err_mm": statistics.median(errs),
        "max_final_err_mm": max(errs),
        "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
        "mean_settle_step": statistics.mean(settles) if settles else MAX_STEPS,
    }


def main():
    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]
    print("Targets:")
    for i, t in enumerate(targets):
        print(f"  T{i}: ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    grid = list(itertools.product(KP_XY_GRID, KI_XY_GRID, KD_XY_GRID))
    print(f"\nGrid: {len(grid)} combos x {N_TARGETS} targets = {len(grid)*N_TARGETS} episodes\n")

    rows: list[dict] = []
    t0 = time.time()
    for ci, (kp, ki, kd) in enumerate(grid, 1):
        pid = make_pid(kp, ki, kd)
        per_target = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            m = run_episode(sim, pid, target)
            per_target.append(m)
            rows.append({"kp_xy": kp, "ki_xy": ki, "kd_xy": kd, "target_idx": ti,
                         "target_x": float(target[0]), "target_y": float(target[1]), "target_z": float(target[2]),
                         **m})
        agg = aggregate(per_target)
        print(f"  [{ci:2d}/{len(grid)}]  Kp={kp:4.1f} Ki={ki:3.1f} Kd={kd:3.1f}  "
              f"succ={agg['success_rate']*100:5.1f}%  med_err={agg['median_final_err_mm']:5.3f} mm  "
              f"max_err={agg['max_final_err_mm']:5.3f} mm  med_settle={agg['median_settle_step']:5.0f} steps  "
              f"elapsed={time.time()-t0:5.1f}s")
    sim.close()

    csv_path = RESULTS / "tuning_log.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    by_combo: dict = {}
    for r in rows:
        by_combo.setdefault((r["kp_xy"], r["ki_xy"], r["kd_xy"]), []).append(r)
    scored = []
    for k, group in by_combo.items():
        per_target = [{"settled": r["settled"], "settle_step": r["settle_step"], "final_err_mm": r["final_err_mm"]} for r in group]
        scored.append((k, aggregate(per_target)))
    scored.sort(key=lambda kv: (-kv[1]["success_rate"], kv[1]["max_final_err_mm"], kv[1]["median_settle_step"]))
    best_k, best_agg = scored[0]
    print(f"\nBEST: Kp={best_k[0]} Ki={best_k[1]} Kd={best_k[2]}  {best_agg}")

    gains = {
        "search": {
            "kp_xy_grid": KP_XY_GRID, "ki_xy_grid": KI_XY_GRID, "kd_xy_grid": KD_XY_GRID,
            "z_scale": Z_SCALE, "n_targets": N_TARGETS, "max_steps": MAX_STEPS,
            "settle_threshold_m_scoring": SETTLE_THRESHOLD_FOR_SCORING,
            "settle_hold_steps": SETTLE_HOLD_STEPS, "seed": SEED,
        },
        "best_score": best_agg,
        "pid": make_pid(*best_k).to_dict(),
    }
    out = RESULTS / "gains_phase1.json"
    out.write_text(json.dumps(gains, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
