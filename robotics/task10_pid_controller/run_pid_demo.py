"""Demo run: drive the pipette to 5 random targets with the tuned PID and record a GIF.

Reads `results/gains.json` (final tuned gains from `refine_pid.py`).

Run:
    python task10_pid_controller\run_pid_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from ot2.control.pid import XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
GAINS = HERE / "results" / "gains.json"
GIF = HERE / "pid_demo.gif"
LOG = HERE / "results" / "demo_log.json"

DT = 1.0 / 240.0
N_TARGETS = 5
MAX_STEPS_PER_TARGET = 600
SETTLE_THRESHOLD = 1.0e-3   # 1 mm settle (matches assignment spec)
SETTLE_HOLD_STEPS = 15
FRAME_EVERY = 12
PRINT_EVERY = 60
SEED = 20260601


try:
    _F = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    _FB = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 18)
except OSError:
    _F = ImageFont.load_default(); _FB = _F


def annotate(frame, lines):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    box_w, box_h = 380, 18 * len(lines) + 10
    d.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 200))
    for i, (line, big) in enumerate(lines):
        d.text((6, 4 + i * 18), line, fill=(80, 230, 110), font=(_FB if big else _F))
    return np.array(img)


def grab():
    w, h = 560, 420
    v = p.computeViewMatrix([0.55, -0.45, 0.45], [0.03, 0.025, 0.20], [0, 0, 1])
    j = p.computeProjectionMatrixFOV(42, w / h, 0.05, 3.0)
    _, _, rgba, _, _ = p.getCameraImage(w, h, v, j, renderer=p.ER_TINY_RENDERER)
    return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def main():
    gains = json.loads(GAINS.read_text())
    pid = XYZPID.from_dict(gains["pid"])
    print("Gains:")
    print(json.dumps(gains["pid"], indent=2))

    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(N_TARGETS)]
    print("\nDemo targets:")
    for i, t in enumerate(targets):
        print(f"  T{i}: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    frames: list[np.ndarray] = []
    log: list[dict] = []
    results: list[dict] = []

    for ti, target in enumerate(targets):
        pid.reset()
        settle = None
        consecutive = 0
        final_err = float("inf")
        final_pos = precise_pipette_position(sim)
        print(f"\n--- T{ti}: target = {target.tolist()} ---")
        for step in range(MAX_STEPS_PER_TARGET):
            pos = precise_pipette_position(sim)
            err = float(np.linalg.norm(target - pos))
            final_err = err; final_pos = pos
            if step % PRINT_EVERY == 0:
                print(f"  [T{ti}] step={step:4d}  pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})  err={err*1000:8.4f} mm")
                log.append({"target": f"T{ti}", "step": step, "pos_m": pos.tolist(),
                            "target_m": target.tolist(), "err_mm": err * 1000})
            if step % FRAME_EVERY == 0:
                frames.append(annotate(grab(), [
                    (f"T{ti}  PID demo  step {step:4d}", True),
                    (f"target = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m", False),
                    (f"pipette= ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m", False),
                    (f"error  = {err*1000:8.4f} mm", False),
                ]))
            if err < SETTLE_THRESHOLD:
                consecutive += 1
                if consecutive >= SETTLE_HOLD_STEPS and settle is None:
                    settle = step
                    # hold a few frames at settled state
                    for _ in range(3):
                        frames.append(annotate(grab(), [
                            (f"T{ti}  SETTLED  step {settle}", True),
                            (f"target = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m", False),
                            (f"pipette= ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m", False),
                            (f"error  = {err*1000:8.4f} mm  OK", False),
                        ]))
                    break
            else:
                consecutive = 0
            a = pid.step(target, pos, dt=DT)
            sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
        results.append({"target": target.tolist(), "final_pos": final_pos.tolist(),
                        "final_err_mm": final_err * 1000.0,
                        "settled": settle is not None, "settle_step": settle})
        for _ in range(2):
            frames.append(annotate(grab(), [(f"Next target after T{ti}...", True)]))
    sim.close()

    print(f"\nwriting {GIF} ({len(frames)} frames)")
    imageio.mimsave(GIF, frames, duration=0.05, loop=0)
    LOG.write_text(json.dumps({"results": results, "trajectory": log,
                               "settings": {"settle_threshold_m": SETTLE_THRESHOLD,
                                            "settle_hold_steps": SETTLE_HOLD_STEPS,
                                            "max_steps": MAX_STEPS_PER_TARGET, "seed": SEED}}, indent=2))
    print(f"wrote {LOG}")

    print("\n=== summary ===")
    for i, r in enumerate(results):
        ok = "OK" if r["settled"] else "MISS"
        print(f"  T{i}: target={r['target']}  err={r['final_err_mm']:8.4f} mm  settle_step={r['settle_step']}  [{ok}]")
    succ = sum(1 for r in results if r["settled"]) / len(results)
    mean_err = float(np.mean([r["final_err_mm"] for r in results]))
    print(f"\n  success rate (<{SETTLE_THRESHOLD*1000} mm, held {SETTLE_HOLD_STEPS} steps): {succ*100:.0f}%")
    print(f"  mean final error: {mean_err:.4f} mm")


if __name__ == "__main__":
    main()
