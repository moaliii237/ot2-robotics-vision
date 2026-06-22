"""Compare PID (Task 10 v3) and the trained RL policy on the same random targets.

Task 11 asks for visualisations comparing the RL controller against a
traditional method across defined metrics. Here those metrics are final
positioning error, settle time, and control effort. Writes a summary table
plus a side-by-side plot.

Run:
    python task11_rl_controller\compare_pid_vs_rl.py --model task11_rl_controller\weights\latest.zip
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ot2.control.pid import AxisPID, XYZPID
from ot2.envs.ot2_env import OT2Env
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DT = 1.0 / 240.0
MAX_STEPS = 600
N_TARGETS = 15
SEED = 20260710
SETTLE_THRESH_MM = 1.0
DWELL_HOLD = 10


def build_pid_simc() -> XYZPID:
    """v3 SIMC baseline, tau_c = 10 ms, gains derived from the sysid fit.

    Per-axis gains are copied from task10_pid_controller/results/gains_final.json.
    Best PID we have: median final error 0 mm, max 0.175 um over 10 random targets.
    """
    return XYZPID(
        x=AxisPID(kp=72.73408306256486, ki=1323.153703458222, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=77.0519465340697,  ki=1485.2571151046516, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=86.29272019286083, ki=1860.0441691493995, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
    )


# old name some scripts still import
build_pid_v3 = build_pid_simc


def first_robot_key(state):
    return next(iter(state))


def run_pid_episode(sim, pid: XYZPID, target: np.ndarray) -> dict:
    pid.reset()
    settle = None
    consecutive = 0
    final_err = float("inf")
    effort = 0.0
    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        err = float(np.linalg.norm(target - pos))
        final_err = err
        if err < SETTLE_THRESH_MM / 1000:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle is None:
                settle = step
        else:
            consecutive = 0
        a = pid.step(target, pos, dt=DT)
        effort += float(np.sum(np.abs(a)))
        sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
    return {"final_err_mm": final_err * 1000.0,
            "settle_step": settle, "settled": settle is not None,
            "effort_per_step": effort / MAX_STEPS}


def run_rl_episode(env: OT2Env, model: PPO, target: np.ndarray) -> dict:
    obs, info = env.reset(options={"target": target})
    settle = None
    consecutive = 0
    final_err = float("inf")
    effort = 0.0
    n = 0
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        effort += float(np.sum(np.abs(action)))
        obs, reward, term, trunc, info = env.step(action)
        final_err = info["distance_m"] * 1000.0
        n += 1
        if info["distance_m"] < SETTLE_THRESH_MM / 1000:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle is None:
                settle = step
        else:
            consecutive = 0
        if term or trunc:
            break
    return {"final_err_mm": final_err, "settle_step": settle, "settled": settle is not None,
            "effort_per_step": effort / max(n, 1), "n_steps": n}


def aggregate(rows):
    errs = [r["final_err_mm"] for r in rows]
    settles = [r["settle_step"] for r in rows if r["settled"]]
    eff = [r["effort_per_step"] for r in rows]
    return {
        "n": len(rows),
        "median_err_mm": statistics.median(errs),
        "mean_err_mm": statistics.mean(errs),
        "max_err_mm": max(errs),
        "success_rate_lt_1mm": sum(1 for e in errs if e < 1) / len(rows),
        "median_settle_step": statistics.median(settles) if settles else MAX_STEPS,
        "mean_effort_per_step": statistics.mean(eff),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(HERE / "weights" / "latest.zip"))
    ap.add_argument("--out_dir", type=str, default=str(HERE / "results"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]

    # run the PID baseline first, then the RL policy below
    from ot2.teleport import teleport_pipette
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    def run_pid_variant(label: str, pid: XYZPID) -> list[dict]:
        print(f"\nrunning {label}...")
        rows = []
        for ti, target in enumerate(targets):
            sim.reset(num_agents=1)
            sim.run([[0, 0, 0, 0]], num_steps=20)
            teleport_pipette(sim, PRACTICAL_ENVELOPE.center)
            sim.run([[0, 0, 0, 0]], num_steps=2)
            r = run_pid_episode(sim, pid, target)
            rows.append(r)
            print(f"  T{ti:2d}  err={r['final_err_mm']:9.6f} mm  settle={r['settle_step']}")
        return rows

    pid_simc_rows = run_pid_variant("PID SIMC (Task 10 v3)", build_pid_simc())
    sim.close()

    model = PPO.load(args.model, device="cpu")
    env = OT2Env(seed=0, max_steps=MAX_STEPS, success_threshold=SETTLE_THRESH_MM / 1000,
                 dwell_steps=DWELL_HOLD, randomise_start=False)
    rl_rows = []
    print("\nrunning RL...")
    for ti, target in enumerate(targets):
        r = run_rl_episode(env, model, target)
        rl_rows.append(r)
        print(f"  T{ti:2d}  err={r['final_err_mm']:7.4f} mm  settle={r['settle_step']}")
    env.close()

    pid_simc_agg = aggregate(pid_simc_rows)
    rl_agg       = aggregate(rl_rows)

    variants = [
        ("PID SIMC", pid_simc_agg),
        ("RL PPO",   rl_agg),
    ]

    print("\n=== summary ===")
    print(f"{'controller':<10}  {'median err':>12}  {'mean err':>12}  {'max err':>12}  "
          f"{'success<1mm':>12}  {'med settle':>11}  {'effort/step':>12}")
    for name, agg in variants:
        print(f"{name:<10}  {agg['median_err_mm']:>12.6f}  {agg['mean_err_mm']:>12.6f}  "
              f"{agg['max_err_mm']:>12.6f}  {agg['success_rate_lt_1mm']*100:>11.1f}%  "
              f"{agg['median_settle_step']:>11.0f}  {agg['mean_effort_per_step']:>12.4f}")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [v[0] for v in variants]
    aggs  = [v[1] for v in variants]
    x = np.arange(len(variants))

    medians = [a["median_err_mm"] for a in aggs]
    maxes   = [a["max_err_mm"]    for a in aggs]
    # PID errors hit ~0, which a log axis can't draw, so floor them at 1e-4 mm
    medians_plot = [max(m, 1e-4) for m in medians]
    maxes_plot   = [max(m, 1e-4) for m in maxes]
    axes[0].bar(x - 0.2, medians_plot, width=0.4, label="median", color="#4C9AFF")
    axes[0].bar(x + 0.2, maxes_plot,   width=0.4, label="max",    color="#FF6B6B")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linestyle="--", label="1 mm spec")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("final error (mm, log)")
    axes[0].set_title("Accuracy")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()

    settles = [a["median_settle_step"] * DT * 1000 for a in aggs]
    axes[1].bar(x, settles, color="#54B567")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("median settle (ms)")
    axes[1].set_title("Speed")
    axes[1].grid(alpha=0.3)

    eff = [a["mean_effort_per_step"] for a in aggs]
    axes[2].bar(x, eff, color="#B567E3")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylabel("|action| per step")
    axes[2].set_title("Control effort")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"PID variants vs RL on {N_TARGETS} random targets")
    fig.tight_layout()
    out_png = out_dir / "pid_vs_rl.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"\nwrote {out_png}")

    out_json = out_dir / "pid_vs_rl.json"
    out_json.write_text(json.dumps({
        "n_targets": N_TARGETS, "seed": SEED,
        "max_steps": MAX_STEPS, "settle_thresh_mm": SETTLE_THRESH_MM, "dwell_hold": DWELL_HOLD,
        "pid_simc": {"aggregate": pid_simc_agg, "per_target": pid_simc_rows},
        "rl":       {"aggregate": rl_agg,       "per_target": rl_rows},
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
