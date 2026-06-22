"""Heavier evaluation of a trained PPO model than test_rl.py.

Sweeps N seeds x K targets per seed (default 3 x 20 = 60), records settle
step at 0.5 / 1 / 5 mm, and lets every episode run the full MAX_STEPS instead
of stopping at first settle so the reported error is the true final-pose
precision. Dumps a summary JSON we can lift straight into the README.

Run:
    python task11_rl_controller\eval_robust.py --model task11_rl_controller\weights\latest.zip
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ot2.envs.ot2_env import OT2Env
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target


HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(HERE / "weights" / "latest.zip"))
    ap.add_argument("--log", type=str, default=str(HERE / "results" / "rl_robust_eval.json"))
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_targets_per_seed", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--dwell_steps", type=int, default=10)
    ap.add_argument("--success_threshold_mm", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10001, 20002, 30003])
    ap.add_argument("--deterministic", action="store_true", default=True)
    return ap.parse_args()


def run_episode(env: OT2Env, model: PPO, target: np.ndarray, max_steps: int,
                deterministic: bool) -> dict:
    obs, info = env.reset(options={"target": target})
    final_dist = float(info.get("envelope", [(0,0,0)])[0][0])  # gets overwritten on step 0
    final_pos = None
    settled_at_step: dict[float, int | None] = {0.0005: None, 0.001: None, 0.005: None}
    consecutive: dict[float, int] = {k: 0 for k in settled_at_step}
    HOLD = 10  # need this many consecutive in-threshold steps to count as settled

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        final_dist = info["distance_m"]
        final_pos = info["pipette"]
        for thr in settled_at_step:
            if final_dist < thr:
                consecutive[thr] += 1
                if consecutive[thr] >= HOLD and settled_at_step[thr] is None:
                    settled_at_step[thr] = step
            else:
                consecutive[thr] = 0
        if term or trunc:
            break
    return {
        "target": target.tolist(),
        "final_pos": final_pos,
        "final_err_mm": final_dist * 1000.0,
        "settled_at_step_0_5mm": settled_at_step[0.0005],
        "settled_at_step_1mm": settled_at_step[0.001],
        "settled_at_step_5mm": settled_at_step[0.005],
        "settled_under_1mm": settled_at_step[0.001] is not None,
    }


def aggregate(results: list[dict]) -> dict:
    errs = [r["final_err_mm"] for r in results]
    settle_1mm = [r["settled_at_step_1mm"] for r in results if r["settled_at_step_1mm"] is not None]
    settle_05 = [r["settled_at_step_0_5mm"] for r in results if r["settled_at_step_0_5mm"] is not None]
    settle_5mm = [r["settled_at_step_5mm"] for r in results if r["settled_at_step_5mm"] is not None]

    # bucket each error into the rubric's accuracy tiers
    tier_8 = sum(1 for e in errs if e < 1.0)
    tier_6 = sum(1 for e in errs if 1.0 <= e < 5.0)
    tier_4 = sum(1 for e in errs if 5.0 <= e < 10.0)
    tier_0 = sum(1 for e in errs if e >= 10.0)

    return {
        "n": len(results),
        "median_err_mm": statistics.median(errs),
        "mean_err_mm": statistics.mean(errs),
        "max_err_mm": max(errs),
        "min_err_mm": min(errs),
        "success_rate_lt_1mm": tier_8 / len(results),
        "success_rate_lt_5mm": (tier_8 + tier_6) / len(results),
        "rubric_tier_8_pts_count": tier_8,
        "rubric_tier_6_pts_count": tier_6,
        "rubric_tier_4_pts_count": tier_4,
        "rubric_tier_0_pts_count": tier_0,
        "median_settle_step_0_5mm": (statistics.median(settle_05) if settle_05 else None),
        "median_settle_step_1mm": (statistics.median(settle_1mm) if settle_1mm else None),
        "median_settle_step_5mm": (statistics.median(settle_5mm) if settle_5mm else None),
    }


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    print(f"loading {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    env = OT2Env(
        seed=0,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold_mm / 1000.0,
        dwell_steps=args.dwell_steps,
        randomise_start=False,
    )

    all_results: list[dict] = []
    per_seed_summary = []
    for seed in args.seeds[: args.n_seeds]:
        rng = np.random.default_rng(seed)
        targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01)
                   for _ in range(args.n_targets_per_seed)]
        seed_results = []
        for t in targets:
            r = run_episode(env, model, t, args.max_steps, args.deterministic)
            r["seed"] = seed
            seed_results.append(r)
            all_results.append(r)
        agg = aggregate(seed_results)
        per_seed_summary.append({"seed": seed, **agg})
        print(f"seed {seed}: max_err={agg['max_err_mm']:6.2f} mm  median={agg['median_err_mm']:6.2f} mm  "
              f"frac<1mm={agg['success_rate_lt_1mm']*100:5.1f}%  "
              f"tier8={agg['rubric_tier_8_pts_count']}/{agg['n']}")
    env.close()

    overall = aggregate(all_results)
    print("\n=== overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    out = Path(args.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": str(model_path),
        "args": vars(args),
        "per_seed": per_seed_summary,
        "overall": overall,
        "trials": all_results,
    }, indent=2))
    print(f"\nwrote {out}")

    print("\n=== Task 11 rubric projection ===")
    n = overall["n"]
    print(f"  out of {n} test targets,")
    print(f"    {overall['rubric_tier_8_pts_count']:3d} reached <1 mm   (rubric tier: 8 points)")
    print(f"    {overall['rubric_tier_6_pts_count']:3d} reached <5 mm   (rubric tier: 6 points)")
    print(f"    {overall['rubric_tier_4_pts_count']:3d} reached <10 mm  (rubric tier: 4 points)")
    print(f"    {overall['rubric_tier_0_pts_count']:3d} reached >=10 mm (rubric tier: 0 points)")
    worst_tier = (8 if overall["max_err_mm"] < 1 else
                  6 if overall["max_err_mm"] < 5 else
                  4 if overall["max_err_mm"] < 10 else 0)
    print(f"\n  Worst-case max-error tier (grader's likely view): {worst_tier} / 8 points")


if __name__ == "__main__":
    main()
