"""Phase 2: push Kp higher and add feed-forward on top of the phase-1 result.

Phase 1 landed on pure P at Kp=45 with 0.003 mm median error. Accuracy is
already there, so phase 2 only asks whether more Kp or feed-forward buys us
shorter settle time without giving the accuracy back.

Run:
    python task10_pid_controller\refine_pid.py
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

DT = 1.0 / 240.0
MAX_STEPS = 700
SETTLE_THRESHOLD_FOR_SCORING = 1.0e-3
SETTLE_HOLD_STEPS = 15
N_TARGETS = 8
SEED = 20260521

KP_GRID = [45.0, 55.0, 65.0, 75.0]
KFF_GRID = [0.0, 5.0, 10.0]
Z_SCALE = {"kp": 0.7, "kff": 0.7}


def make_pid(kp: float, kff: float) -> XYZPID:
    return XYZPID(
        x=AxisPID(kp=kp, ki=0.0, kd=0.0, kff=kff, settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=kp, ki=0.0, kd=0.0, kff=kff, settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=kp * Z_SCALE["kp"], ki=0.0, kd=0.0, kff=kff * Z_SCALE["kff"],
                  settle_threshold=0.0, min_output=0.0),
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
    return {"n_targets": n, "success_rate": successes / n,
            "median_final_err_mm": statistics.median(errs),
            "max_final_err_mm": max(errs),
            "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
            "mean_settle_step": statistics.mean(settles) if settles else MAX_STEPS}


def main():
    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    grid = list(itertools.product(KP_GRID, KFF_GRID))
    print(f"Phase 2 grid: {len(grid)} combos x {N_TARGETS} targets\n")

    rows: list[dict] = []
    scored: list[tuple] = []
    t0 = time.time()
    for ci, (kp, kff) in enumerate(grid, 1):
        pid = make_pid(kp, kff)
        per_target = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            m = run_episode(sim, pid, target)
            per_target.append(m)
            rows.append({"kp": kp, "kff": kff, "target_idx": ti, **m})
        agg = aggregate(per_target)
        scored.append(((kp, kff), agg))
        print(f"  [{ci}/{len(grid)}]  Kp={kp:4.1f} kff={kff:4.1f}  "
              f"succ={agg['success_rate']*100:5.1f}%  med_err={agg['median_final_err_mm']:6.4f} mm  "
              f"max_err={agg['max_final_err_mm']:6.4f} mm  med_settle={agg['median_settle_step']:5.0f} steps  "
              f"elapsed={time.time()-t0:5.1f}s")
    sim.close()

    csv_path = RESULTS / "refine_log.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    scored.sort(key=lambda kv: (-kv[1]["success_rate"], kv[1]["max_final_err_mm"], kv[1]["median_settle_step"]))
    best_k, best_agg = scored[0]
    print(f"\nBEST: Kp={best_k[0]} kff={best_k[1]}  {best_agg}")

    final = {
        "phase2_search": {"kp_grid": KP_GRID, "kff_grid": KFF_GRID, "z_scale": Z_SCALE,
                          "n_targets": N_TARGETS, "max_steps": MAX_STEPS, "seed": SEED},
        "phase2_best_score": best_agg,
        "pid": make_pid(*best_k).to_dict(),
    }
    out = RESULTS / "gains.json"
    out.write_text(json.dumps(final, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
