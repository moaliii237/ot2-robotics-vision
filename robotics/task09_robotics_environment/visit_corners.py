"""Drive the PID to each of the 8 envelope corners and record what happens.

Writes an annotated GIF plus a per-step trajectory log. Only some corners are
reachable from the start pose (5 of 8 last time), the rest get marked BLOCKED.

Run from project root:
    python task09_robotics_environment\visit_corners.py

Reads:
    results/envelope.json     (corners)
Writes:
    corners.gif
    results/corners_log.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from ot2.control.pid import XYZPID, AxisPID
from ot2.sim import make_sim, precise_pipette_position


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DT = 1.0 / 240.0
MAX_STEPS = 800
SETTLE_THRESHOLD_M = 2e-3   # loose 2 mm settle just for a readable demo, PID goes tighter
PRINT_EVERY = 60
FRAME_EVERY = 16


try:
    _FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    _FONT_BIG = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 18)
except OSError:
    _FONT = ImageFont.load_default()
    _FONT_BIG = _FONT


def annotate(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    box_w, box_h = 360, 18 * len(lines) + 10
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 200))
    for i, (line, big) in enumerate(lines):
        font = _FONT_BIG if big else _FONT
        draw.text((6, 4 + i * 18), line, fill=(0, 230, 110), font=font)
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


def make_demo_pid() -> XYZPID:
    """Controller for the corner-visit demo. Same gains as Task 10, plus a small
    min_output floor so it can break through the static friction the reachability
    scans ran into near the joint stops."""
    return XYZPID(
        x=AxisPID(kp=30.0, ki=3.0, kd=0.0, min_output=0.08, settle_threshold=1e-4),
        y=AxisPID(kp=30.0, ki=3.0, kd=0.0, min_output=0.08, settle_threshold=1e-4),
        z=AxisPID(kp=22.0, ki=2.5, kd=0.0, min_output=0.08, settle_threshold=1e-4),
    )


def main():
    envelope_path = RESULTS / "envelope.json"
    if not envelope_path.exists():
        print(f"missing {envelope_path}; run find_envelope.py first.", file=sys.stderr)
        sys.exit(1)
    env = json.loads(envelope_path.read_text())
    corners = [c["xyz_m"] for c in env["corners_joint_limit"]]

    pid = make_demo_pid()
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=30)

    frames: list[np.ndarray] = []
    log: list[dict] = []
    results: list[dict] = []

    for i, corner in enumerate(corners):
        target = np.array(corner, dtype=np.float64)
        pid.reset()
        print(f"\n--- C{i}: target = {target.tolist()} ---")
        settled = None
        consecutive_under = 0
        final_pos = precise_pipette_position(sim)
        final_err = float("inf")
        for step in range(MAX_STEPS):
            pos = precise_pipette_position(sim)
            err = float(np.linalg.norm(target - pos))
            final_err = err
            final_pos = pos
            if step % PRINT_EVERY == 0:
                print(f"  [C{i}] step={step:4d}  pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) err={err*1000:6.2f} mm")
                log.append({"corner": i, "step": step, "pos_m": pos.tolist(),
                            "target_m": target.tolist(), "err_mm": err * 1000})
            if step % FRAME_EVERY == 0:
                frames.append(annotate(grab_frame(), [
                    (f"C{i}  step {step:4d}", True),
                    (f"target = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m", False),
                    (f"pipette= ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m", False),
                    (f"error  = {err * 1000:6.2f} mm", False),
                ]))
            if err < SETTLE_THRESHOLD_M:
                consecutive_under += 1
                if consecutive_under >= 10:
                    settled = step
                    frames.append(annotate(grab_frame(), [
                        (f"C{i}  SETTLED step {settled}", True),
                        (f"target = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m", False),
                        (f"pipette= ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m", False),
                        (f"error  = {err * 1000:6.2f} mm  OK", False),
                    ]))
                    break
            else:
                consecutive_under = 0
            action = pid.step(target, pos, dt=DT)
            sim.run([[float(action[0]), float(action[1]), float(action[2]), 0]], num_steps=1)
        results.append({"corner_index": i, "target_m": target.tolist(),
                        "final_pos_m": final_pos.tolist(),
                        "final_err_mm": final_err * 1000.0,
                        "settled_at_step": settled})
        # couple of filler frames so corners don't jump-cut in the GIF
        for _ in range(2):
            frames.append(annotate(grab_frame(), [(f"Next corner after C{i}...", True)]))

    sim.close()

    gif = HERE / "corners.gif"
    print(f"\nwriting {gif} ({len(frames)} frames)")
    imageio.mimsave(gif, frames, duration=0.05, loop=0)

    log_path = RESULTS / "corners_log.json"
    log_path.write_text(json.dumps({"results": results, "trajectory": log,
                                    "settings": {"settle_threshold_m": SETTLE_THRESHOLD_M,
                                                 "max_steps": MAX_STEPS}}, indent=2))
    print(f"wrote {log_path}")

    print("\n=== summary ===")
    for r in results:
        ok = "OK" if r["settled_at_step"] is not None else "BLOCKED"
        print(f"  C{r['corner_index']}: target={r['target_m']}  err={r['final_err_mm']:6.2f} mm  [{ok}]")


if __name__ == "__main__":
    main()
