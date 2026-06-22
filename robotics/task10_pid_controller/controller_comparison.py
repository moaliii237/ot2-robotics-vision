"""Head-to-head comparison of PID variants on the same 10 random targets.

Variants:
    P_only         pure proportional, Kp=75
    P_ff           P + feed-forward (v2 phase-2 winner, no integral)
    PI_v1          PI with v1's old gains, for legacy comparison
    SIMC_tauc_10ms theoretical PI gains derived from sysid (lambda_c = 10 ms)
    SIMC_tauc_5ms  faster theoretical
    Optuna_best    result of optuna_tune.py (our best so far)

Reports per-controller: success rate, median/max final error, median settle
step, mean cumulative action effort (sum of |action| over the episode).

Run:
    python task10_pid_controller\controller_comparison.py
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
N_TARGETS = 10
SEED = 20260524           # fresh seed so no controller has an advantage


def simc_pi(K: float, tau: float, tau_c: float) -> tuple[float, float]:
    Kp = 1.0 / (K * (tau + tau_c))
    Ti = 4.0 * (tau + tau_c)
    Ki = Kp / Ti
    return Kp, Ki


def make_variants() -> dict[str, XYZPID]:
    """Build the named PID variants."""
    plant = json.loads((RESULTS / "plant_model.json").read_text())
    optuna_gains = json.loads((RESULTS / "optuna_tuning.json").read_text())["pid"]

    def simc_pid(tau_c: float) -> XYZPID:
        def ax(key):
            p = plant["axes"][f"axis_{key}"]
            Kp, Ki = simc_pi(p["K_m_per_s_per_unit"], p["tau_s"], tau_c)
            return AxisPID(kp=Kp, ki=Ki, kd=0.0, settle_threshold=0.0, min_output=0.0)
        return XYZPID(x=ax("x"), y=ax("y"), z=ax("z"))

    def p_uniform(kp: float, kff: float = 0.0) -> XYZPID:
        return XYZPID(
            x=AxisPID(kp=kp, kff=kff, settle_threshold=0.0, min_output=0.0),
            y=AxisPID(kp=kp, kff=kff, settle_threshold=0.0, min_output=0.0),
            z=AxisPID(kp=kp * 0.7, kff=kff * 0.7, settle_threshold=0.0, min_output=0.0),
        )

    return {
        "P_only        ": p_uniform(75.0, 0.0),
        "P_ff_v2winner ": p_uniform(75.0, 10.0),
        "PI_v1         ": XYZPID(
            x=AxisPID(kp=35.0, ki=4.0, kd=0.0, settle_threshold=0.0005, min_output=0.12),
            y=AxisPID(kp=35.0, ki=4.0, kd=0.0, settle_threshold=0.0005, min_output=0.12),
            z=AxisPID(kp=24.5, ki=3.0, kd=0.0, settle_threshold=0.0005, min_output=0.12),
        ),
        "SIMC_tauc_5ms ": simc_pid(0.005),
        "SIMC_tauc_10ms": simc_pid(0.010),
        "Optuna_best   ": XYZPID.from_dict(optuna_gains),
    }


def run_episode(sim, pid: XYZPID, target: np.ndarray) -> dict:
    pid.reset()
    settle_step = None
    consecutive_under = 0
    final_err = float("inf")
    effort = 0.0
    n_steps_run = 0
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
        effort += float(np.sum(np.abs(action)))
        sim.run([[float(action[0]), float(action[1]), float(action[2]), 0]], num_steps=1)
        n_steps_run += 1
    return {"settled": settle_step is not None,
            "settle_step": settle_step if settle_step is not None else MAX_STEPS,
            "final_err_mm": final_err * 1000.0,
            "control_effort": effort,
            "control_effort_per_step": effort / n_steps_run}


def aggregate(per_target: list[dict]) -> dict:
    errs = [r["final_err_mm"] for r in per_target]
    settles = [r["settle_step"] for r in per_target if r["settled"]]
    effort = [r["control_effort_per_step"] for r in per_target]
    return {
        "success_rate": sum(1 for r in per_target if r["settled"]) / len(per_target),
        "median_final_err_mm": statistics.median(errs),
        "max_final_err_mm": max(errs),
        "mean_final_err_mm": statistics.mean(errs),
        "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
        "mean_control_effort_per_step": statistics.mean(effort),
    }


def main():
    variants = make_variants()
    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    results: dict[str, dict] = {}
    print(f"\n{'controller':<16}  {'succ':>5}  {'median err':>12}  {'mean err':>12}  {'max err':>12}  "
          f"{'med settle':>10}  {'effort/step':>11}")
    print("-" * 100)
    t0 = time.time()
    for name, pid in variants.items():
        per_target = []
        for t in targets:
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            m = run_episode(sim, pid, t)
            per_target.append(m)
        agg = aggregate(per_target)
        results[name.strip()] = {"score": agg, "per_target": per_target, "gains": pid.to_dict()}
        print(f"{name}  {agg['success_rate']*100:>4.0f}%  "
              f"{agg['median_final_err_mm']:>12.5f}  {agg['mean_final_err_mm']:>12.5f}  "
              f"{agg['max_final_err_mm']:>12.5f}  {agg['median_settle_step']:>10.0f}  "
              f"{agg['mean_control_effort_per_step']:>11.4f}")
    sim.close()
    print(f"\nelapsed: {time.time()-t0:.1f}s")

    out_json = RESULTS / "controller_comparison.json"
    out_json.write_text(json.dumps({
        "n_targets": N_TARGETS, "seed": SEED,
        "settle_threshold_m": SETTLE_THRESHOLD_FOR_SCORING,
        "results": results,
    }, indent=2))
    print(f"wrote {out_json}")

    # Plot
    names = list(results.keys())
    med_errs = [results[n]["score"]["median_final_err_mm"] for n in names]
    max_errs = [results[n]["score"]["max_final_err_mm"] for n in names]
    settles = [results[n]["score"]["median_settle_step"] * DT * 1000 for n in names]
    efforts = [results[n]["score"]["mean_control_effort_per_step"] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x - 0.2, med_errs, width=0.4, label="median", color="#4C9AFF")
    axes[0].bar(x + 0.2, max_errs, width=0.4, label="max", color="#FF6B6B")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linestyle="--", label="1 mm spec")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("final error (mm, log)"); axes[0].set_title("Accuracy")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)

    axes[1].bar(x, settles, color="#54B567")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("median settle time (ms)"); axes[1].set_title("Speed")
    axes[1].grid(alpha=0.3)

    axes[2].bar(x, efforts, color="#B567E3")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("mean |action| per step"); axes[2].set_title("Control effort")
    axes[2].grid(alpha=0.3)

    fig.suptitle("PID variants head-to-head (10 random targets)")
    fig.tight_layout()
    out_png = RESULTS / "controller_comparison.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
