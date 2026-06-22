"""Bayesian hyperparameter optimisation of the PID gains using Optuna.

A TPE sampler in place of the coarse grid: it lands on a good gain set in
O(50) trials versus the O(24) combos a grid needs for the same range, and it
can reach regimes the grid never sampled. The big one is very high Ki, which
the IMC result said should help and the grid couldn't have found.

Search space:
    Kp_xy  ~ LogUniform(10, 300)
    Ki_xy  ~ LogUniform(0.1, 5000)   # wide because SIMC showed Ki in the 1000s
    Kd_xy  ~ Uniform(0, 2.0)
    kff_xy ~ Uniform(0, 30)

z gains are scaled from xy by a fixed ratio (0.85 for Kp, 0.85 for Ki, 0.85 for kff)
that's roughly the ratio of the plant's per-axis time constants.

Objective (to minimise): max_final_err_mm * 100 + median_settle_step.
Accuracy dominates because that's the ILO metric; settle time is a soft tiebreak.

Run:
    python task10_pid_controller\optuna_tune.py [--n_trials 80]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from ot2.control.pid import AxisPID, XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
MAX_STEPS = 700
SETTLE_THRESHOLD_FOR_SCORING = 1.0e-3
SETTLE_HOLD_STEPS = 15
N_TARGETS = 6
SEED_TARGETS = 20260523       # different seed from grid + IMC so Optuna doesn't overfit
Z_SCALE = 0.85


def make_pid(kp: float, ki: float, kd: float, kff: float) -> XYZPID:
    return XYZPID(
        x=AxisPID(kp=kp, ki=ki, kd=kd, kff=kff, settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=kp, ki=ki, kd=kd, kff=kff, settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=kp * Z_SCALE, ki=ki * Z_SCALE, kd=kd * Z_SCALE, kff=kff * Z_SCALE,
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


class StudyContext:
    """Holds the persistent sim handle + the fixed target set."""
    def __init__(self, n_targets: int, seed: int):
        rng = np.random.default_rng(seed)
        self.targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(n_targets)]
        self.sim = make_sim(num_agents=1, render=False)
        self.sim.run([[0, 0, 0, 0]], num_steps=20)

    def evaluate(self, pid: XYZPID) -> dict:
        per_target = []
        for t in self.targets:
            self.sim.reset(num_agents=1)
            self.sim.run([[0, 0, 0, 0]], num_steps=20)
            per_target.append(run_episode(self.sim, pid, t))
        errs = [r["final_err_mm"] for r in per_target]
        settles = [r["settle_step"] for r in per_target if r["settled"]]
        return {
            "success_rate": sum(1 for r in per_target if r["settled"]) / len(per_target),
            "median_final_err_mm": statistics.median(errs),
            "max_final_err_mm": max(errs),
            "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=80)
    args = ap.parse_args()

    ctx = StudyContext(N_TARGETS, SEED_TARGETS)

    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("kp", 10.0, 300.0, log=True)
        ki = trial.suggest_float("ki", 0.1, 5000.0, log=True)
        kd = trial.suggest_float("kd", 0.0, 2.0)
        kff = trial.suggest_float("kff", 0.0, 30.0)
        pid = make_pid(kp, ki, kd, kff)
        agg = ctx.evaluate(pid)
        # Hard penalty if not 100% success
        if agg["success_rate"] < 1.0:
            return 1e6
        # Primary: max error (in mm). Secondary: settle time (in steps).
        return agg["max_final_err_mm"] * 100.0 + agg["median_settle_step"] * 1e-3

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Running Optuna TPE search, {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    ctx.sim.close()

    best = study.best_params
    print("\n=== BEST ===")
    print(f"  Kp={best['kp']:.2f}  Ki={best['ki']:.2f}  Kd={best['kd']:.4f}  kff={best['kff']:.2f}")
    print(f"  objective: {study.best_value:.5f}")

    # Re-evaluate the best on the same targets for a clean score
    ctx2 = StudyContext(N_TARGETS, SEED_TARGETS)
    pid = make_pid(**best)
    final_agg = ctx2.evaluate(pid)
    print(f"  validation: succ={final_agg['success_rate']*100:.0f}%  "
          f"med_err={final_agg['median_final_err_mm']:.4f} mm  "
          f"max_err={final_agg['max_final_err_mm']:.4f} mm  "
          f"med_settle={final_agg['median_settle_step']:.0f} steps")
    ctx2.sim.close()

    out_json = RESULTS / "optuna_tuning.json"
    out_json.write_text(json.dumps({
        "n_trials": args.n_trials,
        "sampler": "TPE",
        "z_scale": Z_SCALE,
        "best_params": best,
        "best_objective": study.best_value,
        "validation_score": final_agg,
        "pid": make_pid(**best).to_dict(),
        "trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            } for t in study.trials
        ],
    }, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
