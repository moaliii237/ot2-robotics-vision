"""Evaluate a trained PPO model on the OT-2 Gym env, record a GIF and metrics.

Run from project root with the venv activated:
    python task11_rl_controller\test_rl.py --model weights\latest.zip
    python task11_rl_controller\test_rl.py --model weights\PPO_smoke_best\best_model.zip --gif rl_demo.gif --n_targets 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from ot2.envs.ot2_env import OT2Env
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target


HERE = Path(__file__).resolve().parent

try:
    _FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    _FONT_BIG = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 18)
except OSError:
    _FONT = ImageFont.load_default()
    _FONT_BIG = _FONT


def annotate(frame: np.ndarray, lines: list[tuple[str, bool]]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    box_w = 360
    box_h = 18 * len(lines) + 10
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 200))
    for i, (line, big) in enumerate(lines):
        font = _FONT_BIG if big else _FONT
        draw.text((6, 4 + i * 18), line, fill=(120, 220, 255), font=font)
    return np.array(img)


def grab_frame():
    w, h = 560, 420
    view = p.computeViewMatrix(
        cameraEyePosition=[0.55, -0.45, 0.45],
        cameraTargetPosition=[0.03, 0.025, 0.20],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    proj = p.computeProjectionMatrixFOV(fov=42, aspect=w / h, nearVal=0.05, farVal=3.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width=w, height=h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def parse_args():
    p_ = argparse.ArgumentParser()
    p_.add_argument("--model", type=str, default=str(HERE / "weights" / "latest.zip"))
    p_.add_argument("--gif", type=str, default=str(HERE / "rl_demo.gif"))
    p_.add_argument("--log", type=str, default=str(HERE / "results" / "rl_eval_log.json"))
    p_.add_argument("--n_targets", type=int, default=5)
    p_.add_argument("--max_steps", type=int, default=400)
    p_.add_argument("--success_threshold_mm", type=float, default=1.0)
    p_.add_argument("--dwell_steps", type=int, default=20)
    p_.add_argument("--seed", type=int, default=20260601)
    p_.add_argument("--deterministic", action="store_true", default=True)
    p_.add_argument("--frame_every", type=int, default=8)
    return p_.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}. Train one first with train_rl.py.")
    model = PPO.load(str(model_path), device="cpu")  # CPU is plenty for a single rollout and frees the GPU
    print(f"loaded {model_path}")

    env = OT2Env(
        seed=args.seed,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold_mm / 1000.0,
        dwell_steps=args.dwell_steps,
        randomise_start=False,  # fixed start makes the GIF easier to follow
    )

    frames: list[np.ndarray] = []
    results: list[dict] = []

    rng = np.random.default_rng(args.seed)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(args.n_targets)]

    for ti, target in enumerate(targets):
        obs, info = env.reset(options={"target": target})
        ep_reward = 0.0
        final_pos = info["envelope"][0]  # overwritten on the first step
        final_dist = float("inf")
        settled_step = None
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            final_pos = info["pipette"]
            final_dist = info["distance_m"]
            if step % args.frame_every == 0:
                frames.append(annotate(grab_frame(), [
                    (f"T{ti}  PPO  step {step:4d}", True),
                    (f"target = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m", False),
                    (f"pipette= ({final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}) m", False),
                    (f"error  = {final_dist*1000:6.2f} mm   reward={reward:+.2f}", False),
                ]))
            if terminated or truncated:
                if terminated and info["dwell"] >= args.dwell_steps:
                    settled_step = step
                break

        results.append({
            "target": target.tolist(),
            "final_pos": final_pos if isinstance(final_pos, list) else list(final_pos),
            "final_err_mm": final_dist * 1000.0,
            "ep_reward": ep_reward,
            "settled_step": settled_step,
            "settled": settled_step is not None,
        })
        # a few held frames so targets don't jump-cut in the GIF
        for _ in range(3):
            frames.append(annotate(grab_frame(), [(f"Next target after T{ti}...", True)]))

    env.close()

    print(f"\nwriting {args.gif} ({len(frames)} frames)")
    imageio.mimsave(args.gif, frames, duration=0.05, loop=0)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "model": str(model_path),
        "args": vars(args),
        "results": results,
    }, indent=2))
    print(f"wrote {log_path}")

    succ = sum(1 for r in results if r["settled"]) / max(len(results), 1)
    mean_err = float(np.mean([r["final_err_mm"] for r in results]))
    print(f"\n=== summary ===")
    for i, r in enumerate(results):
        ok = "OK" if r["settled"] else "MISS"
        print(f"  T{i}: target={r['target']}  err={r['final_err_mm']:6.2f} mm  settle_step={r['settled_step']}  [{ok}]")
    print(f"\n  success rate (<{args.success_threshold_mm} mm, held {args.dwell_steps} steps): {succ*100:.0f}%")
    print(f"  mean final error: {mean_err:.3f} mm")


if __name__ == "__main__":
    main()
