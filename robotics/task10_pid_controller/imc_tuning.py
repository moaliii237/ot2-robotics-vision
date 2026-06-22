"""Derive PID gains analytically from the identified plant via SIMC/IMC tuning.

The OT-2 plant from action to position is

    G(s) = K / (s * (tau*s + 1))

with K the velocity gain and tau the velocity-loop time constant, both from
sysid.py. Folding the tau lag into an effective delay with the SIMC half-rule
turns this into an integrator-plus-delay plant, G(s) ~= K*exp(-tau*s)/s, for
which the SIMC rules give a PI controller:

    Kp = 1 / (K * (tau + tau_c))
    Ti = 4 * (tau + tau_c)
    Ki = Kp / Ti

tau_c is the desired closed-loop time constant, i.e. the one knob you turn.

We sweep tau_c, validate each set of gains on a handful of random targets,
and run the empirical Optuna/grid gains through the same test. That tells us
whether theory lands where the search did, and shows the speed (small tau_c)
vs robustness (large tau_c) tradeoff.

Run:
    python task10_pid_controller\imc_tuning.py

Writes:
    results/imc_tuning.json  (theoretical gains + per-target validation results)
    results/imc_vs_empirical.png
"""
from __future__ import annotations

import csv
import json
import statistics
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
N_TARGETS = 6
SEED = 20260522

TAU_C_GRID = [0.005, 0.010, 0.020, 0.050, 0.100]   # desired closed-loop time constant (s)


def simc_pi_gains(K: float, tau: float, tau_c: float) -> tuple[float, float]:
    """Skogestad SIMC for integrator+delay-equivalent plant. Returns (Kp, Ki)."""
    Kp = 1.0 / (K * (tau + tau_c))
    Ti = 4.0 * (tau + tau_c)
    Ki = Kp / Ti
    return Kp, Ki


def make_pid_from_model(plant: dict, tau_c: float) -> XYZPID:
    """Build an XYZPID with PI gains derived from the per-axis plant model."""
    def axis(key):
        K = plant["axes"][f"axis_{key}"]["K_m_per_s_per_unit"]
        tau = plant["axes"][f"axis_{key}"]["tau_s"]
        Kp, Ki = simc_pi_gains(K, tau, tau_c)
        return AxisPID(kp=Kp, ki=Ki, kd=0.0, settle_threshold=0.0, min_output=0.0)
    return XYZPID(x=axis("x"), y=axis("y"), z=axis("z"))


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
    errs = [r["final_err_mm"] for r in per_target]
    settles = [r["settle_step"] for r in per_target if r["settled"]]
    return {"success_rate": sum(1 for r in per_target if r["settled"]) / len(per_target),
            "median_final_err_mm": statistics.median(errs),
            "max_final_err_mm": max(errs),
            "median_settle_step": statistics.median(settles) if settles else MAX_STEPS}


def main():
    plant_path = RESULTS / "plant_model.json"
    if not plant_path.exists():
        raise SystemExit(f"missing {plant_path}; run sysid.py first")
    plant = json.loads(plant_path.read_text())
    print("Identified plant per axis:")
    for ax, p in plant["axes"].items():
        print(f"  {ax}: K={p['K_m_per_s_per_unit']:.4f}  tau={p['tau_s']*1000:.3f} ms  R^2={p['r_squared']:.4f}")

    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    # IMC sweep
    imc_results = []
    print("\n=== SIMC PI gains derived from plant model ===")
    print(f"{'tau_c (ms)':>10}  {'Kp_x':>7}  {'Ki_x':>7}  {'Kp_y':>7}  {'Ki_y':>7}  {'Kp_z':>7}  {'Ki_z':>7}  "
          f"{'success':>7}  {'med err':>9}  {'max err':>9}  {'med settle':>10}")
    for tau_c in TAU_C_GRID:
        pid = make_pid_from_model(plant, tau_c)
        per_target = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            m = run_episode(sim, pid, target)
            per_target.append(m)
        agg = aggregate(per_target)
        gains = pid.to_dict()
        print(f"{tau_c*1000:>10.1f}  {gains['x']['kp']:>7.1f}  {gains['x']['ki']:>7.1f}  "
              f"{gains['y']['kp']:>7.1f}  {gains['y']['ki']:>7.1f}  "
              f"{gains['z']['kp']:>7.1f}  {gains['z']['ki']:>7.1f}  "
              f"{agg['success_rate']*100:>5.1f}%   {agg['median_final_err_mm']:>9.4f}  "
              f"{agg['max_final_err_mm']:>9.4f}  {agg['median_settle_step']:>10.0f}")
        imc_results.append({
            "tau_c_s": tau_c,
            "gains": gains,
            "score": agg,
            "per_target": per_target,
        })

    # Validate empirical gains too (load from existing gains.json if present)
    emp_path = RESULTS / "gains.json"
    emp_agg = None
    emp_gains = None
    if emp_path.exists():
        emp_gains = json.loads(emp_path.read_text())["pid"]
        pid = XYZPID.from_dict(emp_gains)
        per_target = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            m = run_episode(sim, pid, target)
            per_target.append(m)
        emp_agg = aggregate(per_target)
        print(f"\nempirical    "
              f"Kp_x={emp_gains['x']['kp']:5.1f}  Ki_x={emp_gains['x']['ki']:5.1f}  "
              f"-> succ={emp_agg['success_rate']*100:.0f}%  "
              f"med_err={emp_agg['median_final_err_mm']:.4f} mm  "
              f"max_err={emp_agg['max_final_err_mm']:.4f} mm  "
              f"med_settle={emp_agg['median_settle_step']:.0f} steps")

    sim.close()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tau_cs = [r["tau_c_s"] * 1000 for r in imc_results]
    med_errs = [r["score"]["median_final_err_mm"] for r in imc_results]
    max_errs = [r["score"]["max_final_err_mm"] for r in imc_results]
    settles = [r["score"]["median_settle_step"] * DT * 1000 for r in imc_results]
    axes[0].semilogy(tau_cs, med_errs, "o-", label="SIMC median err")
    axes[0].semilogy(tau_cs, max_errs, "s-", label="SIMC max err")
    if emp_agg is not None:
        axes[0].axhline(emp_agg["median_final_err_mm"], color="green", linestyle="--", label="empirical median err")
        axes[0].axhline(emp_agg["max_final_err_mm"], color="red", linestyle="--", label="empirical max err")
    axes[0].axhline(1.0, color="black", linestyle=":", label="1 mm spec")
    axes[0].set_xlabel("design tau_c (ms)")
    axes[0].set_ylabel("final error (mm)")
    axes[0].set_title("Accuracy vs SIMC design parameter")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)

    axes[1].plot(tau_cs, settles, "o-", label="SIMC settle")
    if emp_agg is not None:
        axes[1].axhline(emp_agg["median_settle_step"] * DT * 1000, color="green", linestyle="--", label="empirical settle")
    axes[1].set_xlabel("design tau_c (ms)")
    axes[1].set_ylabel("median settle time (ms)")
    axes[1].set_title("Speed vs SIMC design parameter")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_png = RESULTS / "imc_vs_empirical.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_png}")

    out_json = RESULTS / "imc_tuning.json"
    out_json.write_text(json.dumps({
        "plant": plant,
        "tau_c_grid_s": TAU_C_GRID,
        "imc_results": imc_results,
        "empirical_gains": emp_gains,
        "empirical_score": emp_agg,
        "n_targets": N_TARGETS,
        "seed": SEED,
        "notes": [
            "SIMC PI: Kp = 1/(K*(tau+tau_c)), Ti = 4*(tau+tau_c), Ki = Kp/Ti.",
            "Plant K and tau are per-axis from sysid.py.",
            "Empirical gains are the result of Optuna/grid search (gains.json).",
        ],
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
