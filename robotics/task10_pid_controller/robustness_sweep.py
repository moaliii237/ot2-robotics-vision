"""Robustness sweep: validate the tuned PID across multiple random-target seeds.

Tests the final tuned gains (from `gains.json`) on N independent seeds, each
with N_TARGETS targets. Confirms the result is not seed-lucky.

Run:
    python task10_pid_controller\robustness_sweep.py
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ot2.control.pid import XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DT = 1.0 / 240.0
MAX_STEPS = 700
SETTLE_THRESHOLD_FOR_SCORING = 1.0e-3
SETTLE_HOLD_STEPS = 15
N_TARGETS_PER_SEED = 10
SEEDS = [10001, 20002, 30003]


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


def main():
    gains = json.loads((RESULTS / "gains.json").read_text())
    pid_template = gains["pid"]
    print("Using gains:")
    print(json.dumps(pid_template, indent=2))

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    all_results: list[dict] = []
    per_seed_agg: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS_PER_SEED)]
        pid = XYZPID.from_dict(pid_template)
        seed_results = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            r = run_episode(sim, pid, target)
            seed_results.append(r)
            all_results.append({"seed": seed, "target_idx": ti,
                                "target_x": float(target[0]), "target_y": float(target[1]), "target_z": float(target[2]),
                                **r})
        errs = [r["final_err_mm"] for r in seed_results]
        settles = [r["settle_step"] for r in seed_results if r["settled"]]
        agg = {
            "seed": seed,
            "n_targets": len(seed_results),
            "success_rate": sum(1 for r in seed_results if r["settled"]) / len(seed_results),
            "median_final_err_mm": statistics.median(errs),
            "mean_final_err_mm": statistics.mean(errs),
            "max_final_err_mm": max(errs),
            "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
        }
        per_seed_agg.append(agg)
        print(f"  seed {seed:5d}  succ={agg['success_rate']*100:5.1f}%  med_err={agg['median_final_err_mm']:7.4f} mm  "
              f"mean_err={agg['mean_final_err_mm']:7.4f} mm  max_err={agg['max_final_err_mm']:7.4f} mm  "
              f"med_settle={agg['median_settle_step']:5.0f} steps  elapsed={time.time()-t0:5.1f}s")
    sim.close()

    # Overall
    all_errs = [r["final_err_mm"] for r in all_results]
    all_settles = [r["settle_step"] for r in all_results if r["settled"]]
    overall = {
        "total_targets": len(all_results),
        "success_rate": sum(1 for r in all_results if r["settled"]) / len(all_results),
        "median_final_err_mm": statistics.median(all_errs),
        "mean_final_err_mm": statistics.mean(all_errs),
        "max_final_err_mm": max(all_errs),
        "median_settle_step": statistics.median(all_settles) if all_settles else MAX_STEPS,
    }
    print("\n=== overall across all seeds ===")
    print(f"  total targets: {overall['total_targets']}")
    print(f"  success rate (<1 mm, held 15 steps): {overall['success_rate']*100:.1f}%")
    print(f"  median final error: {overall['median_final_err_mm']:.4f} mm")
    print(f"  mean final error:   {overall['mean_final_err_mm']:.4f} mm")
    print(f"  max final error:    {overall['max_final_err_mm']:.4f} mm")
    print(f"  median settle:      {overall['median_settle_step']:.0f} steps")

    out_csv = RESULTS / "robustness_log.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nwrote {out_csv}")

    out_json = RESULTS / "robustness_summary.json"
    out_json.write_text(json.dumps({
        "seeds": SEEDS, "n_targets_per_seed": N_TARGETS_PER_SEED,
        "per_seed": per_seed_agg, "overall": overall,
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
