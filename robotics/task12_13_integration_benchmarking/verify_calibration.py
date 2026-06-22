"""Calibration probe: checks the -pi/2 axis routing against the sim.

Two things to pin down: the calibration's -pi/2 derivation is described
loosely, and none of the error metrics ever check that the pipette lands over
the photographed tip (they all measure controller-vs-setpoint). This script
asks the sim directly instead of trusting the docstring:

    1. Build a deliberately asymmetric texture: four solid colour quadrants
       on a white plate (red TL, green TR, blue BL, yellow BR) with labels,
       so orientation is unambiguous.
    2. Treat the whole image as the dish (dish_bbox = full frame) and build
       the same Calibration.sim_dish the demo/benchmark use.
    3. Apply the texture to the specimen plate the same way the demo's
       PyBulletRenderer does (RGB->BGR, loadTexture, changeVisualShape).
    4. For each quadrant centre, get its world coord via pixel_to_world and
       drop a matching-coloured probe sphere there on the agar surface.
    5. Render from a near-overhead camera and the demo's angled camera, save
       the PNGs.

If each sphere sits on top of its matching colour, the calibration agrees
with the texture UV (pipette lands on the tip). If a sphere lands on the
opposite quadrant, the row/col signs are flipped.

Run:
    python task12_13_integration_benchmarking\\verify_calibration.py
Then look at results/calib_probe_overhead.png and results/calib_probe_angle.png.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cv"))

from ot2.sim import make_sim
from calibration import (
    AGAR_SURFACE_Z_M, DEFAULT_DISPENSE_Z_M, DISH_CENTER_X_M, DISH_CENTER_Y_M,
    Calibration,
)

S = 600  # synthetic square dish, pixels
H = S // 2

# Four solid quadrants. Each tuple: (label, row, col, RGB quadrant, RGBA
# sphere). Sampling at the quadrant centre so the sphere sits over its colour.
PROBES = [
    ("TL-red",     S // 4,      S // 4,      (220, 40, 40),  [0.90, 0.12, 0.12, 1.0]),
    ("TR-green",   S // 4,      3 * S // 4,  (40, 190, 70),  [0.15, 0.78, 0.25, 1.0]),
    ("BL-blue",    3 * S // 4,  S // 4,      (50, 90, 230),  [0.20, 0.35, 0.95, 1.0]),
    ("BR-yellow",  3 * S // 4,  3 * S // 4,  (235, 200, 30), [0.92, 0.80, 0.12, 1.0]),
]


def build_synthetic_texture() -> np.ndarray:
    """Plate in 4 solid colour quadrants. RGB uint8.

    Image-space layout (row down, col right):
        top-left  = RED      top-right = GREEN
        bot-left  = BLUE     bot-right = YELLOW
    """
    img = np.zeros((S, S, 3), dtype=np.uint8)
    img[0:H,   0:H]   = (220, 40, 40)    # TL red
    img[0:H,   H:S]   = (40, 190, 70)    # TR green
    img[H:S,   0:H]   = (50, 90, 230)    # BL blue
    img[H:S,   H:S]   = (235, 200, 30)   # BR yellow
    # Black cross on the quadrant seams to make orientation obvious.
    cv2.line(img, (H, 0), (H, S - 1), (0, 0, 0), 6)
    cv2.line(img, (0, H), (S - 1, H), (0, 0, 0), 6)
    for label, r, c, _, _ in PROBES:
        cv2.putText(img, label, (c - 70, r), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def apply_texture(specimen_id: int, rgb_img: np.ndarray) -> None:
    """Apply the texture the same way the demo's PyBulletRenderer does."""
    fd, tmppath = tempfile.mkstemp(prefix="calib_probe_tex_", suffix=".png")
    os.close(fd)
    cv2.imwrite(tmppath, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    tex_id = p.loadTexture(tmppath)
    p.changeVisualShape(specimen_id, -1, textureUniqueId=tex_id,
                        rgbaColor=[1.0, 1.0, 1.0, 1.0])


def grab(eye, target, up, w=900, h=900) -> np.ndarray:
    view = p.computeViewMatrix(cameraEyePosition=eye,
                               cameraTargetPosition=target, cameraUpVector=up)
    proj = p.computeProjectionMatrixFOV(fov=42, aspect=w / h,
                                        nearVal=0.05, farVal=3.0)
    _, _, rgba, _, _ = p.getCameraImage(width=w, height=h, viewMatrix=view,
                                        projectionMatrix=proj,
                                        renderer=p.ER_TINY_RENDERER)
    return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def main():
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    dish_bbox = (0, 0, S, S)            # the whole synthetic image is the dish
    cal = Calibration.sim_dish(dish_bbox, z_m=DEFAULT_DISPENSE_Z_M)
    print(f"calibration: {cal}")
    print("\nProbe pixel -> world mapping:")
    world_pts = []
    for label, r, c, _, rgba in PROBES:
        wx, wy, wz = cal.pixel_to_world(r, c)
        world_pts.append((label, wx, wy, rgba))
        print(f"  {label:11s} pixel ({r:3d},{c:3d}) -> world ({wx:+.4f}, {wy:+.4f}, {wz:.4f})")

    tex = build_synthetic_texture()

    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)
    spec_id = sim.specimenIds[0]
    apply_texture(spec_id, tex)

    # Drop a probe sphere at each quadrant-centre's calibrated world position.
    for label, wx, wy, rgba in world_pts:
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.009, rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                          basePosition=[wx, wy, AGAR_SURFACE_Z_M + 0.010])

    cx, cy = DISH_CENTER_X_M, DISH_CENTER_Y_M
    # Top-down from above the dish. Tiny -y offset so up=+y isn't degenerate.
    overhead = grab(eye=[cx, cy - 0.001, AGAR_SURFACE_Z_M + 0.62],
                    target=[cx, cy, AGAR_SURFACE_Z_M],
                    up=[1.0, 0.0, 0.0])
    cv2.imwrite(str(out_dir / "calib_probe_overhead.png"),
                cv2.cvtColor(overhead, cv2.COLOR_RGB2BGR))

    # Demo's angled camera.
    angle = grab(eye=[cx + 0.32, cy - 0.28, AGAR_SURFACE_Z_M + 0.45],
                 target=[cx, cy, AGAR_SURFACE_Z_M + 0.04],
                 up=[0.0, 0.0, 1.0])
    cv2.imwrite(str(out_dir / "calib_probe_angle.png"),
                cv2.cvtColor(angle, cv2.COLOR_RGB2BGR))

    sim.close()
    print(f"\nwrote {out_dir / 'calib_probe_overhead.png'}")
    print(f"wrote {out_dir / 'calib_probe_angle.png'}")

    # ---- numeric checks (independent of the render) ----
    # PyBullet cube-unwraps a GEOM_BOX texture, so it won't paint the whole
    # photo onto the top face and the render can't pin an absolute orientation.
    # The checks below stay within what the calibration alone guarantees:
    # round-trip inverse, dish centre -> plate centre, and that the four
    # quadrant centres land on distinct world points whose left/right
    # (col -> +y) and near/far (row -> -x) order matches the image.
    print("\n=== automated calibration checks ===")
    ok = True

    # round-trip pixel -> world -> pixel
    max_rt = 0.0
    for _, r, c, _, _ in PROBES:
        wx, wy, _ = cal.pixel_to_world(r, c)
        rr, cc = cal.world_to_pixel(wx, wy)
        max_rt = max(max_rt, abs(rr - r), abs(cc - c))
    print(f"  round-trip pixel->world->pixel max error: {max_rt:.2e} px "
          f"({'PASS' if max_rt < 1e-6 else 'FAIL'})")
    ok = ok and max_rt < 1e-6

    # dish centre should map to the plate centre
    cwx, cwy, _ = cal.pixel_to_world(S // 2, S // 2)
    centre_err = ((cwx - DISH_CENTER_X_M) ** 2 + (cwy - DISH_CENTER_Y_M) ** 2) ** 0.5
    print(f"  dish-centre pixel -> ({cwx:.4f},{cwy:.4f}), plate centre "
          f"({DISH_CENTER_X_M:.4f},{DISH_CENTER_Y_M:.4f}); err {centre_err*1000:.4f} mm "
          f"({'PASS' if centre_err < 1e-6 else 'FAIL'})")
    ok = ok and centre_err < 1e-6

    # spatial order: col up -> world +y; row up -> world -x
    wp = {label: cal.pixel_to_world(r, c)[:2] for label, r, c, _, _ in PROBES}
    tl, tr = wp["TL-red"], wp["TR-green"]
    bl = wp["BL-blue"]
    cond_col_y = tr[1] > tl[1]          # higher col -> larger y
    cond_row_x = bl[0] < tl[0]          # higher row -> smaller x
    print(f"  col (left->right) maps to +y: {cond_col_y} "
          f"({'PASS' if cond_col_y else 'FAIL'})")
    print(f"  row (top->bottom) maps to -x: {cond_row_x} "
          f"({'PASS' if cond_row_x else 'FAIL'})")
    ok = ok and cond_col_y and cond_row_x

    # all four targets are distinct
    pts = list(wp.values())
    dmin = min(((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
               for i in range(len(pts)) for j in range(i + 1, len(pts)))
    print(f"  min pairwise target separation: {dmin*1000:.1f} mm "
          f"({'PASS' if dmin > 0.01 else 'FAIL'})")
    ok = ok and dmin > 0.01

    print(f"\n  OVERALL: {'PASS - calibration convention is self-consistent' if ok else 'FAIL'}")
    print("  (The 3-D render validates world-space placement; the pixel-exact")
    print("   2-D overlay in the demo is the accuracy view. PyBullet's box")
    print("   cube-unwrap means the 3-D plate texture is illustrative only.)")


if __name__ == "__main__":
    main()
