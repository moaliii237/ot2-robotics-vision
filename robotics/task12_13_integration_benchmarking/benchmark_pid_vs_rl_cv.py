"""Task 13 benchmark: PID-SIMC vs RL-PPO on CV-derived targets.

For every test image (default: all 19 in the Kaggle dish set):
  1. Run CV (U-Net + plant detection + tip extraction).
  2. Calibrate tip pixels -> OT-2 world coords (sim_dish calibration).
  3. Drive the pipette to each tip with PID-SIMC AND with RL-PPO.
  4. Capture per-target metrics: final_err_mm, settle_step, effort/step,
     wall_time_s. No GIF rendering, just measurements.

Output:
  results/pid_vs_rl_cv_benchmark.json   aggregated + per-target data
  results/pid_vs_rl_cv_benchmark.png    side-by-side comparison plot

Run:
    python task12_13_integration_benchmarking\\benchmark_pid_vs_rl_cv.py \\
        --image_dir <path-to-dish-image-folder>

The Kaggle dish set is not in the repo (it lives on the team drive). Point
--image_dir at any folder of dish PNGs. It defaults to cv/samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cv"))

from stable_baselines3 import PPO

from ot2.control.pid import AxisPID, XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.envs.ot2_env import OT2Env
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette
from root_pipeline import detect_tips_in_image, get_dish_bbox

from calibration import Calibration, DEFAULT_DISPENSE_Z_M


DT = 1.0 / 240.0
MAX_STEPS_PER_TARGET = 600
SETTLE_THRESH_MM = 1.0
DWELL_HOLD = 10


def build_pid_simc() -> XYZPID:
    return XYZPID(
        x=AxisPID(kp=72.73408306256486, ki=1323.153703458222, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=77.0519465340697,  ki=1485.2571151046516, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=86.29272019286083, ki=1860.0441691493995, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
    )


def clip_to_envelope(world: np.ndarray) -> np.ndarray:
    return np.clip(world,
                   PRACTICAL_ENVELOPE.low + 0.005,
                   PRACTICAL_ENVELOPE.high - 0.005)


def drive_pid_episode(sim, pid: XYZPID, target: np.ndarray,
                      max_steps: int, home: np.ndarray) -> dict:
    """Drive PID to target, starting from a fixed home pose every call.

    Fairness: re-home to home (zeroed velocity) at the start of every target,
    exactly as the RL env does on reset(). If the sim were teleported just
    once before the loop, each PID target would inherit the previous target's
    final pose, so travel distance (and hence settle time / effort) would be
    shorter than RL's. Re-homing makes both travel the same home->target
    distance per target.

    Settle parity: the settle distance is read post-step (after sim.run),
    matching how the RL env reports distance_m. Reading it pre-step gives PID
    a 1-step head start over RL. The episode also counts DWELL_HOLD
    consecutive in-tolerance steps the same way the RL env does.
    """
    teleport_pipette(sim, home)
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=2)   # let it settle at home
    pid.reset()
    settle_step = None
    consecutive = 0
    effort = 0.0
    n_steps = 0
    t0 = time.perf_counter()
    for step in range(max_steps):
        pos = precise_pipette_position(sim)
        a = pid.step(target, pos, dt=DT)
        effort += float(np.sum(np.abs(a)))
        sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
        n_steps += 1
        post = precise_pipette_position(sim)            # read after the step, like RL
        err_m = float(np.linalg.norm(target - post))
        if err_m < SETTLE_THRESH_MM / 1000.0:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle_step is None:
                settle_step = step                       # note first settle, keep going
        else:
            consecutive = 0
        # Deliberately no break at settle. Both controllers run the full fixed
        # horizon from the same home, so final_err is each one's true
        # steady-state precision (PID's integral converges to sub-micron, RL
        # holds its learned ~0.5 mm). settle_step still shows how fast each
        # first hits the 1 mm spec, and effort is averaged over the same
        # horizon for both.
    t1 = time.perf_counter()
    final_pos = precise_pipette_position(sim)
    final_err = float(np.linalg.norm(target - final_pos)) * 1000.0
    return {"final_err_mm": final_err,
            "settle_step": settle_step,
            "settled": settle_step is not None,
            "effort_per_step": effort / max(n_steps, 1),
            "wall_time_s": t1 - t0,
            "n_steps": n_steps}


def drive_rl_episode(env: OT2Env, model: PPO, target: np.ndarray,
                     max_steps: int) -> dict:
    obs, info = env.reset(options={"target": target})
    settle_step = None
    consecutive = 0
    final_err = float("inf")
    n_steps = 0
    effort = 0.0
    t0 = time.perf_counter()
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        effort += float(np.sum(np.abs(action)))
        obs, reward, term, trunc, info = env.step(action)
        n_steps += 1
        final_err = info["distance_m"] * 1000.0
        if info["distance_m"] < SETTLE_THRESH_MM / 1000.0:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle_step is None:
                settle_step = step
        else:
            consecutive = 0
        if term or trunc:
            break
    t1 = time.perf_counter()
    return {"final_err_mm": final_err,
            "settle_step": settle_step,
            "settled": settle_step is not None,
            "effort_per_step": effort / max(n_steps, 1),
            "wall_time_s": t1 - t0,
            "n_steps": n_steps}


def aggregate(rows: list[dict]) -> dict:
    errs = [r["final_err_mm"] for r in rows]
    settles = [r["settle_step"] for r in rows if r["settled"] and r["settle_step"] is not None]
    efforts = [r["effort_per_step"] for r in rows]
    wall = [r["wall_time_s"] for r in rows]
    tier8 = sum(1 for e in errs if e < 1.0)
    tier6 = sum(1 for e in errs if 1.0 <= e < 5.0)
    tier4 = sum(1 for e in errs if 5.0 <= e < 10.0)
    tier0 = sum(1 for e in errs if e >= 10.0)
    return {
        "n": len(rows),
        "median_err_mm": statistics.median(errs) if errs else float("nan"),
        "mean_err_mm": statistics.mean(errs) if errs else float("nan"),
        "max_err_mm": max(errs) if errs else float("nan"),
        "min_err_mm": min(errs) if errs else float("nan"),
        "success_rate_lt_1mm": tier8 / max(len(rows), 1),
        "success_rate_lt_5mm": (tier8 + tier6) / max(len(rows), 1),
        "rubric_tier_8_count": tier8,
        "rubric_tier_6_count": tier6,
        "rubric_tier_4_count": tier4,
        "rubric_tier_0_count": tier0,
        "median_settle_step": statistics.median(settles) if settles else None,
        "mean_settle_step": statistics.mean(settles) if settles else None,
        "mean_effort_per_step": statistics.mean(efforts) if efforts else float("nan"),
        "median_wall_time_s": statistics.median(wall) if wall else float("nan"),
        "mean_wall_time_s": statistics.mean(wall) if wall else float("nan"),
        "total_wall_time_s": sum(wall),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", type=str,
                    default=str(HERE / "cv" / "samples"))
    ap.add_argument("--unet_weights", type=str,
                    default=str(HERE / "cv" / "weights" / "unet_256px.keras"))
    ap.add_argument("--rl_weights", type=str,
                    default=r"D:\Robotics\task11_rl_controller\weights\latest.zip")
    ap.add_argument("--out_json", type=str,
                    default=str(HERE / "results" / "pid_vs_rl_cv_benchmark.json"))
    ap.add_argument("--out_png", type=str,
                    default=str(HERE / "results" / "pid_vs_rl_cv_benchmark.png"))
    ap.add_argument("--max_steps_per_target", type=int,
                    default=MAX_STEPS_PER_TARGET)
    ap.add_argument("--n_images", type=int, default=None,
                    help="limit to first N images (debug)")
    args = ap.parse_args()
    args.unet_weights = str(Path(args.unet_weights).resolve())
    args.rl_weights = str(Path(args.rl_weights).resolve())
    args.out_json = str(Path(args.out_json).resolve())
    args.out_png = str(Path(args.out_png).resolve())

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if args.n_images is not None:
        images = images[: args.n_images]
    if not images:
        raise SystemExit(f"no images in {args.image_dir}")
    print(f"benchmarking on {len(images)} images...")

    # 1) CV pass: detect tips for every image up front, one TF session.
    all_detections: list[dict] = []
    for img_path in images:
        result, img_orig, _ = detect_tips_in_image(args.unet_weights, img_path)
        dish_bbox = get_dish_bbox(img_orig)
        cal = Calibration.sim_dish(dish_bbox, z_m=DEFAULT_DISPENSE_Z_M)
        targets_world = [
            np.asarray(cal.pixel_to_world(p.tip[0], p.tip[1]))
            for p in result.plants
        ]
        clipped = [clip_to_envelope(t) for t in targets_world]
        # Track how far the envelope clip moved each target, so a near-rim tip
        # that got displaced shows up instead of being silently clamped and
        # then scored against the moved point.
        clip_disp_mm = [
            float(np.linalg.norm(c - u) * 1000.0)
            for c, u in zip(clipped, targets_world)
        ]
        max_disp = max(clip_disp_mm) if clip_disp_mm else 0.0
        all_detections.append({
            "image_path": img_path,
            "image_stem": Path(img_path).stem,
            "n_plants": result.n_detected,
            "dish_bbox": list(dish_bbox),
            "tips_pixel": [list(p.tip) for p in result.plants],
            "tips_world_unclipped": [t.tolist() for t in targets_world],
            "tips_world": [t.tolist() for t in clipped],
            "clip_displacement_mm": clip_disp_mm,
            "calibration_label": cal.label,
        })
        warn = f"  [WARN clip moved a tip {max_disp:.1f} mm]" if max_disp > 0.1 else ""
        print(f"  {Path(img_path).name}: {result.n_detected} plants{warn}")

    # 2) PID pass: one sim, drive every target.
    pid_rows: list[dict] = []
    print("\nrunning PID-SIMC...")
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)
    home = PRACTICAL_ENVELOPE.center   # same per-target start for both controllers
    pid = build_pid_simc()
    t_start = time.perf_counter()
    for det in all_detections:
        for i, target in enumerate(det["tips_world"]):
            r = drive_pid_episode(sim, pid, np.asarray(target),
                                  args.max_steps_per_target, home=home)
            r.update({"image": det["image_stem"], "plant_index": i,
                      "target_world": target})
            pid_rows.append(r)
        last_n = pid_rows[-det["n_plants"]:]
        summary_strs = [f"{r['final_err_mm']:.2f}" for r in last_n]
        print(f"  PID {det['image_stem']}: [{', '.join(summary_strs)}]")
    sim.close()
    print(f"  PID total wall time: {time.perf_counter() - t_start:.1f} s")

    # 3) RL pass.
    rl_rows: list[dict] = []
    print("\nrunning RL-PPO...")
    # dwell_steps > max_steps means the env never terminates early on success,
    # so RL runs the same full horizon as PID and we get its true steady-state
    # precision, not just the pose at the 1 mm dwell.
    env = OT2Env(seed=0, max_steps=args.max_steps_per_target,
                 success_threshold=SETTLE_THRESH_MM / 1000.0,
                 dwell_steps=args.max_steps_per_target + 1, randomise_start=False)
    model = PPO.load(args.rl_weights, device="cpu")
    t_start = time.perf_counter()
    for det in all_detections:
        for i, target in enumerate(det["tips_world"]):
            r = drive_rl_episode(env, model, np.asarray(target),
                                 args.max_steps_per_target)
            r.update({"image": det["image_stem"], "plant_index": i,
                      "target_world": target})
            rl_rows.append(r)
        last_n = rl_rows[-det["n_plants"]:]
        summary_strs = [f"{r['final_err_mm']:.2f}" for r in last_n]
        print(f"  RL  {det['image_stem']}: [{', '.join(summary_strs)}]")
    env.close()
    print(f"  RL total wall time: {time.perf_counter() - t_start:.1f} s")

    pid_agg = aggregate(pid_rows)
    rl_agg = aggregate(rl_rows)

    # 4) Print summary
    print("\n=== summary ===")
    cols = ["n", "median_err_mm", "mean_err_mm", "max_err_mm",
            "success_rate_lt_1mm", "median_settle_step",
            "mean_effort_per_step", "mean_wall_time_s"]
    header = f"{'controller':<10} " + " ".join(f"{c:>20s}" for c in cols)
    print(header)
    for name, agg in [("PID-SIMC", pid_agg), ("RL-PPO", rl_agg)]:
        row = f"{name:<10} "
        for c in cols:
            v = agg[c]
            if v is None:
                row += f"{'-':>20}"
            elif isinstance(v, float):
                row += f"{v:>20.4f}"
            else:
                row += f"{v:>20}"
        print(row)

    # 5) Comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = ["PID-SIMC", "RL-PPO"]
    aggs = [pid_agg, rl_agg]
    x = np.arange(2)

    # 5a accuracy (log)
    medians = [max(a["median_err_mm"], 1e-4) for a in aggs]
    maxes = [max(a["max_err_mm"], 1e-4) for a in aggs]
    axes[0].bar(x - 0.2, medians, width=0.4, label="median", color="#4C9AFF")
    axes[0].bar(x + 0.2, maxes, width=0.4, label="max", color="#FF6B6B")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linestyle="--", label="1 mm spec")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("final error (mm, log)")
    axes[0].set_title("Accuracy")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()

    # 5b settle
    settles = [(a["median_settle_step"] or 0) * DT * 1000 for a in aggs]
    axes[1].bar(x, settles, color="#54B567")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("median settle (ms)")
    axes[1].set_title("Speed (settle time)")
    axes[1].grid(alpha=0.3)

    # 5c effort
    eff = [a["mean_effort_per_step"] for a in aggs]
    axes[2].bar(x, eff, color="#B567E3")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("|action| per step")
    axes[2].set_title("Control effort")
    axes[2].grid(alpha=0.3)

    # 5d success rate
    sr = [a["success_rate_lt_1mm"] * 100 for a in aggs]
    axes[3].bar(x, sr, color="#FFB050")
    axes[3].set_xticks(x); axes[3].set_xticklabels(names)
    axes[3].set_ylabel("success rate <1 mm (%)")
    axes[3].set_ylim(0, 105)
    axes[3].set_title("Success rate")
    axes[3].grid(alpha=0.3)

    n_targets = sum(d["n_plants"] for d in all_detections)
    fig.suptitle(f"PID-SIMC vs RL-PPO on {n_targets} CV-derived targets "
                 f"across {len(images)} dishes")
    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out_png}")

    # 6) JSON dump
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "image_dir": args.image_dir,
        "n_images": len(images),
        "n_targets": n_targets,
        "detections": all_detections,
        "pid": {"aggregate": pid_agg, "per_target": pid_rows},
        "rl":  {"aggregate": rl_agg,  "per_target": rl_rows},
    }, indent=2, default=str))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
