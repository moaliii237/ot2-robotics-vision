"""Task 12 end-to-end demo: petri-dish photo -> CV -> controller -> GIF.

Pipeline:

    1. U-Net inference -> binary root mask (dish-cropped).
    2. Detect up to 5 plants -> root-tip pixel per plant.
    3. Calibrate each tip pixel into the OT-2 world frame using the sim
       plate geometry (Y2B-2023-OT2_Twin/custom.urdf: 150 mm square, world
       pos (0.1827, 0.1370, 0.087), -pi/2 z-rotation).
    4. Drive the pipette to each tip with the chosen controller (PID-SIMC
       from Task 10 v3, or RL-PPO Path B from Task 11).
    5. Top-down render straight on the dish photo, so the GIF shows accuracy
       to the pixel without a PyBullet camera's perspective / UV ambiguity.
       Each frame shows:
         - green X at every detected root tip
         - yellow ring around the active target
         - live orange circle for the pipette (via inverse calibration)
         - colour-coded permanent dot at every settled dispense location
         - red error line from pipette to active target, labelled in mm

Output:
    * GIF of the inoculation routine
    * JSON log: CV detections + per-target controller stats + the
      calibration used.

Example:
    python task12_13_integration_benchmarking\\demo_cv_to_controller.py `
        --image task12_13_integration_benchmarking\\cv\\samples\\test_image_09.png `
        --controller rl `
        --gif task12_13_integration_benchmarking\\results\\demo_rl_09.gif
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import os

import cv2
import imageio.v2 as imageio
import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cv"))

from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from ot2.control.pid import AxisPID, XYZPID
from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.envs.ot2_env import OT2Env
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette
from root_pipeline import detect_tips_in_image, get_dish_bbox

from calibration import (
    AGAR_SURFACE_Z_M, DEFAULT_DISPENSE_Z_M, DISH_CENTER_X_M, DISH_CENTER_Y_M,
    DISH_SIZE_M, Calibration,
)

DT = 1.0 / 240.0
MAX_STEPS_PER_TARGET = 600
SETTLE_THRESH_MM = 1.0
DWELL_HOLD = 10
DISPLAY_DISH_PX = 640   # rendered dish side in pixels (output frame is square)

try:
    _FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
    _FONT_BIG = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 24)
except OSError:
    _FONT = ImageFont.load_default()
    _FONT_BIG = _FONT

PLANT_PALETTE = [
    (235, 70, 70),    # red
    (60, 200, 130),   # green
    (80, 140, 240),   # blue
    (240, 175, 60),   # orange
    (200, 120, 220),  # magenta
]


# ---------- controllers ----------

def build_pid_simc() -> XYZPID:
    return XYZPID(
        x=AxisPID(kp=72.73408306256486, ki=1323.153703458222, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        y=AxisPID(kp=77.0519465340697,  ki=1485.2571151046516, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
        z=AxisPID(kp=86.29272019286083, ki=1860.0441691493995, kd=0.0,
                  settle_threshold=0.0, min_output=0.0),
    )


# ---------- shared helper: annotate the dish-cropped photo with primary roots ----------

def build_annotated_dish_image(img_orig: np.ndarray,
                               dish_bbox: tuple[int, int, int, int],
                               plants: list,
                               include_paths: bool = True,
                               include_tips: bool = True,
                               line_thickness: int | None = None,
                               tip_radius: int | None = None,
                               ) -> np.ndarray:
    """Dish-cropped photo with each primary root drawn on top, colour-coded
    per plant. RGB.

    Serves as both the 2D top-down background and the 3D PyBullet texture.
    Line/tip sizes scale with the crop resolution: at full res a 6 px line on
    a 2756 px crop shrinks to ~1.3 px once the 2D panel downscales to ~640 px
    and disappears. ~W/110 thickness survives the shrink.
    """
    x1, y1, x2, y2 = dish_bbox
    crop = img_orig[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    W = max(x2 - x1, 1)
    if line_thickness is None:
        line_thickness = max(10, W // 110)     # ~25 px on a 2756 px crop
    if tip_radius is None:
        tip_radius = max(22, W // 70)           # ~39 px on a 2756 px crop
    font_scale = max(1.2, W / 1200.0)
    font_thick = max(3, W // 700)

    for plant in plants:
        color = PLANT_PALETTE[plant.plant_index % len(PLANT_PALETTE)]
        # Treat the array as RGB here; the final conversion handles BGR.
        if include_paths and len(plant.primary_root_path) > 1:
            pts = np.array(
                [[c - x1, r - y1] for (r, c) in plant.primary_root_path],
                dtype=np.int32,
            )
            # White casing under the colour so the trace reads on any agar shade.
            cv2.polylines(rgb, [pts], isClosed=False, color=(255, 255, 255),
                          thickness=line_thickness + max(6, W // 350),
                          lineType=cv2.LINE_AA)
            cv2.polylines(rgb, [pts], isClosed=False, color=color,
                          thickness=line_thickness, lineType=cv2.LINE_AA)
        if include_tips:
            tr, tc = plant.tip
            tc_l = tc - x1
            tr_l = tr - y1
            cv2.circle(rgb, (tc_l, tr_l), tip_radius + max(5, W // 400),
                       (255, 255, 255), max(3, W // 500))
            cv2.circle(rgb, (tc_l, tr_l), tip_radius, color, -1)
            cv2.putText(rgb, f"P{plant.plant_index + 1}",
                        (tc_l + tip_radius + 6, tr_l + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thick, cv2.LINE_AA)
    return rgb


# ---------- 2D top-down dish renderer (pixel-accurate) ----------

class TopDownRenderer:
    """Top-down view of the scene drawn straight on the dish-cropped photo.

    Every overlay (tips, pipette, droplets, error vector) is placed via the
    calibration's exact pixel<->world transform, so the GIF is pixel-accurate.
    No PyBullet camera, no UV mapping, no perspective.
    """

    def __init__(self, img_orig: np.ndarray,
                 dish_bbox: tuple[int, int, int, int],
                 plants: list,
                 calibration: Calibration,
                 display_side_px: int = DISPLAY_DISH_PX,
                 annotated_rgb: np.ndarray | None = None,
                 ):
        x1, y1, x2, y2 = dish_bbox
        if annotated_rgb is None:
            crop = img_orig[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            crop_rgb = annotated_rgb
        Hc, Wc = crop_rgb.shape[:2]
        side = max(Hc, Wc)
        padded = np.full((side, side, 3), 255, dtype=np.uint8)
        oy = (side - Hc) // 2
        ox = (side - Wc) // 2
        padded[oy:oy + Hc, ox:ox + Wc] = crop_rgb
        self._pad_xy = (ox, oy)
        self._dish_orig_size = (Hc, Wc)
        self._scale = display_side_px / side    # display-px per dish-px
        canvas = cv2.resize(padded, (display_side_px, display_side_px),
                            interpolation=cv2.INTER_AREA)
        self.background = canvas
        self.display_side_px = display_side_px
        self.cal = calibration
        self.dish_bbox = dish_bbox
        self.plants = plants

    # ----- coordinate conversions used by the renderer -----

    def raw_pixel_to_display(self, raw_row: float, raw_col: float
                             ) -> tuple[int, int]:
        """Raw-image pixel -> display canvas (row, col)."""
        dx, dy = self._pad_xy
        dish_row = raw_row - self.dish_bbox[1] + dy
        dish_col = raw_col - self.dish_bbox[0] + dx
        return (int(round(dish_row * self._scale)),
                int(round(dish_col * self._scale)))

    def world_to_display(self, world_x: float, world_y: float
                         ) -> tuple[int, int]:
        """OT-2 world (x, y) -> display canvas (row, col)."""
        raw_row, raw_col = self.cal.world_to_pixel(world_x, world_y)
        return self.raw_pixel_to_display(raw_row, raw_col)

    # ----- frame composition -----

    def render(self,
               *,
               active_plant_idx: int,
               pipette_world: np.ndarray,
               target_world: np.ndarray,
               droplets_world: list[tuple[int, np.ndarray]],
               header: str,
               step: int,
               error_mm: float,
               settled_badge: str | None,
               ) -> np.ndarray:
        """Build one annotated frame. droplets_world is a list of
        (plant_index, world_xyz_array)."""
        img = Image.fromarray(self.background.copy())
        draw = ImageDraw.Draw(img, "RGBA")

        # Detected tips: green X if inactive, yellow ring if it's the target.
        for i, plant in enumerate(self.plants):
            tr, tc = self.raw_pixel_to_display(plant.tip[0], plant.tip[1])
            if i == active_plant_idx:
                draw.ellipse([tc - 26, tr - 26, tc + 26, tr + 26],
                             outline=(255, 235, 60), width=4)
                self._cross(draw, tr, tc, size=14, color=(255, 235, 60), width=4)
            else:
                self._cross(draw, tr, tc, size=10,
                            color=(40, 220, 110), width=3)
            draw.text((tc + 16, tr - 10), f"P{i+1}",
                      fill=(255, 255, 255), font=_FONT)

        # Droplet history, colour-coded by plant.
        for plant_idx, w in droplets_world:
            dr, dc = self.world_to_display(float(w[0]), float(w[1]))
            color = PLANT_PALETTE[plant_idx % len(PLANT_PALETTE)]
            draw.ellipse([dc - 11, dr - 11, dc + 11, dr + 11],
                         fill=(*color, 255),
                         outline=(255, 255, 255), width=2)

        # Error vector from pipette to the active target.
        pr, pc = self.world_to_display(float(pipette_world[0]),
                                       float(pipette_world[1]))
        if target_world is not None:
            tr, tc = self.world_to_display(float(target_world[0]),
                                           float(target_world[1]))
            draw.line([(pc, pr), (tc, tr)], fill=(255, 80, 80, 230), width=3)

        # Live pipette (orange).
        draw.ellipse([pc - 14, pr - 14, pc + 14, pr + 14],
                     outline=(255, 140, 0), width=4)
        draw.ellipse([pc - 4, pr - 4, pc + 4, pr + 4],
                     fill=(255, 140, 0, 255))

        # Live numbers panel, top-left.
        pad = 12
        panel_lines = [
            f"step    {step:4d}",
            f"target  ({target_world[0]:+.4f}, {target_world[1]:+.4f}) m"
                if target_world is not None else "target  -",
            f"pipette ({pipette_world[0]:+.4f}, {pipette_world[1]:+.4f}) m",
            f"error   {error_mm:7.3f} mm",
        ]
        panel_w = 440
        panel_h = 40 + len(panel_lines) * 22 + 12
        draw.rectangle([(0, 0), (panel_w, panel_h)], fill=(0, 0, 0, 200))
        draw.text((pad, pad - 2), header,
                  fill=(120, 230, 255), font=_FONT_BIG)
        for i, line in enumerate(panel_lines):
            draw.text((pad, pad + 32 + i * 22), line,
                      fill=(230, 230, 230), font=_FONT)

        # Status badge, top-right, when a target just settled.
        if settled_badge:
            bw, bh = 280, 56
            bx = self.display_side_px - bw - 14
            draw.rectangle([(bx, 14), (bx + bw, 14 + bh)],
                           fill=(20, 140, 80, 235))
            draw.text((bx + 18, 24), settled_badge,
                      fill=(255, 255, 255), font=_FONT_BIG)

        # Legend, bottom-left.
        legend_lines = [
            ("green X = detected tip", (40, 220, 110)),
            ("yellow ring = active target", (255, 235, 60)),
            ("orange = pipette (live)", (255, 140, 0)),
            ("colour dot = settled droplet", (200, 200, 200)),
        ]
        lh = 22
        lp = 12
        lw = 360
        lH = lp * 2 + len(legend_lines) * lh
        ly0 = self.display_side_px - lH - 12
        draw.rectangle([(12, ly0), (12 + lw, ly0 + lH)],
                       fill=(0, 0, 0, 190))
        for i, (txt, col) in enumerate(legend_lines):
            draw.text((lp + 16, ly0 + lp + i * lh), txt,
                      fill=(*col, 255), font=_FONT)

        return np.asarray(img)

    @staticmethod
    def _cross(draw, row: int, col: int, size: int, color, width: int):
        s = size
        draw.line([(col - s, row - s), (col + s, row + s)],
                  fill=color, width=width)
        draw.line([(col - s, row + s), (col + s, row - s)],
                  fill=color, width=width)


# ---------- 3D PyBullet renderer (angled "up-and-side" view of the OT-2) ----------

def _project_world_to_screen(world_pts, view, proj, w, h) -> np.ndarray:
    """Project world (x,y,z) points to screen pixels using the OpenGL
    view/proj matrices PyBullet returns (column-major 16-tuples)."""
    V = np.array(view, dtype=np.float64).reshape(4, 4, order="F")
    P = np.array(proj, dtype=np.float64).reshape(4, 4, order="F")
    out = []
    for (x, y, z) in world_pts:
        clip = P @ (V @ np.array([x, y, z, 1.0]))
        wclip = clip[3] if abs(clip[3]) > 1e-12 else 1e-12
        ndc = clip[:3] / wclip
        sx = (ndc[0] * 0.5 + 0.5) * w
        sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * h
        out.append((sx, sy))
    return np.array(out, dtype=np.float32)


class PyBulletRenderer:
    """Render the live OT-2 sim from a fixed angled camera, overlaying the
    dish photo (with root traces) onto the plate's top face in image space.

    Image-space overlay instead of a PyBullet box texture, because PyBullet
    cube-unwraps a GEOM_BOX texture and won't paint the full photo onto the
    top face (the plate just rendered one flat colour). So we project the
    plate's four top-face corners into the render, warp the annotated photo
    onto that quad once (the camera is static), and composite it each frame
    only where the plate is actually the visible surface (seg mask == specimen
    id). That way the pipette and chassis occlude it correctly and droplets
    stay on top. The corner mapping matches the calibration, so a droplet at a
    calibrated tip world coord lands on that tip in the photo.
    """

    def __init__(self, specimen_id: int, annotated_rgb: np.ndarray,
                 frame_size: tuple[int, int] = (520, 640)):
        self.specimen_id = specimen_id
        self.frame_w, self.frame_h = frame_size

        # Camera: above-and-side, slight tilt down. Looks at the dish centre.
        self._eye = [DISH_CENTER_X_M + 0.32,
                     DISH_CENTER_Y_M - 0.28,
                     AGAR_SURFACE_Z_M + 0.45]
        self._target = [DISH_CENTER_X_M,
                        DISH_CENTER_Y_M,
                        AGAR_SURFACE_Z_M + 0.04]
        self._view = p.computeViewMatrix(cameraEyePosition=self._eye,
                                         cameraTargetPosition=self._target,
                                         cameraUpVector=[0.0, 0.0, 1.0])
        self._proj = p.computeProjectionMatrixFOV(
            fov=38, aspect=self.frame_w / self.frame_h,
            nearVal=0.05, farVal=3.0)

        # Warp the annotated photo onto the plate quad once (static camera).
        # Image corners (cv2 x=col, y=row) -> world plate-top corners, same
        # mapping as calibration.sim_dish:
        #   row 0  -> world_x = cx + size/2 ;  row H -> cx - size/2
        #   col 0  -> world_y = cy - size/2 ;  col W -> cy + size/2
        Hc, Wc = annotated_rgb.shape[:2]
        half = DISH_SIZE_M / 2.0
        cx, cy, z = DISH_CENTER_X_M, DISH_CENTER_Y_M, AGAR_SURFACE_Z_M
        world_corners = [
            (cx + half, cy - half, z),   # img TL (0,0)
            (cx + half, cy + half, z),   # img TR (0,Wc)
            (cx - half, cy + half, z),   # img BR (Hc,Wc)
            (cx - half, cy - half, z),   # img BL (Hc,0)
        ]
        dst = _project_world_to_screen(world_corners, self._view, self._proj,
                                       self.frame_w, self.frame_h)
        src = np.array([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]], dtype=np.float32)
        Hmat = cv2.getPerspectiveTransform(src, dst)
        self._warped_rgb = cv2.warpPerspective(
            annotated_rgb, Hmat, (self.frame_w, self.frame_h),
            flags=cv2.INTER_AREA)
        ones = np.full((Hc, Wc), 255, np.uint8)
        self._warp_mask = cv2.warpPerspective(
            ones, Hmat, (self.frame_w, self.frame_h),
            flags=cv2.INTER_NEAREST) > 0

        self._target_marker_id: int | None = None
        self._droplet_ids: list[int] = []

    # ----- visual markers in the 3D scene -----

    def set_active_target(self, world_target: np.ndarray) -> None:
        if self._target_marker_id is None:
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.005,
                                     rgbaColor=[1.0, 0.85, 0.1, 0.95])
            self._target_marker_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vs,
                basePosition=world_target.tolist(),
            )
        else:
            p.resetBasePositionAndOrientation(
                self._target_marker_id, world_target.tolist(), [0, 0, 0, 1])

    def add_droplet(self, world_pos: np.ndarray, plant_idx: int) -> None:
        rgb = PLANT_PALETTE[plant_idx % len(PLANT_PALETTE)]
        rgba_norm = [c / 255.0 for c in rgb] + [0.95]
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.0055,
                                 rgbaColor=rgba_norm)
        pos = [world_pos[0], world_pos[1], AGAR_SURFACE_Z_M + 0.005]
        body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                                    basePosition=pos)
        self._droplet_ids.append(body_id)

    # ----- frame capture -----

    def render(self) -> np.ndarray:
        _, _, rgba, _, seg = p.getCameraImage(
            width=self.frame_w, height=self.frame_h,
            viewMatrix=self._view, projectionMatrix=self._proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.asarray(rgba, dtype=np.uint8
                         ).reshape(self.frame_h, self.frame_w, 4)[:, :, :3].copy()
        seg = np.asarray(seg, dtype=np.int32).reshape(self.frame_h, self.frame_w)
        # Paint the warped photo only where the plate top face is visible
        # (seg == specimen id) and the warp covers. Droplets, target sphere
        # and pipette carry other seg ids, so they're left alone and occlude
        # the photo for free.
        plate = (seg == self.specimen_id) & self._warp_mask
        rgb[plate] = self._warped_rgb[plate]
        return rgb


# ---------- compose: stack the two views and add header ----------

def compose_frame(sim_rgb: np.ndarray, dish_rgb: np.ndarray,
                  *,
                  header: str, status: str | None,
                  controller: str,
                  plant_idx: int, n_plants: int,
                  step: int, error_mm: float,
                  target: np.ndarray, pipette: np.ndarray,
                  ) -> np.ndarray:
    """Side-by-side composition: [3D sim | 2D accuracy panel]."""
    sim_h, sim_w = sim_rgb.shape[:2]
    dish_h, dish_w = dish_rgb.shape[:2]
    H = max(sim_h, dish_h)

    def pad(img: np.ndarray, height: int) -> np.ndarray:
        if img.shape[0] == height:
            return img
        out = np.full((height, img.shape[1], 3), 30, dtype=img.dtype)
        out[:img.shape[0]] = img
        return out

    sim_p = pad(sim_rgb, H)
    dish_p = pad(dish_rgb, H)
    separator = np.full((H, 6, 3), 30, dtype=np.uint8)
    combined = np.concatenate([sim_p, separator, dish_p], axis=1)

    # Top header bar spanning both panels.
    bar_h = 56
    bar = np.full((bar_h, combined.shape[1], 3), 12, dtype=np.uint8)
    pil_bar = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil_bar, "RGBA")
    draw.text((16, 14), header, fill=(120, 230, 255), font=_FONT_BIG)
    sub = (f"Plant {plant_idx+1}/{n_plants}  step {step:4d}   "
           f"target ({target[0]:+.3f},{target[1]:+.3f})  "
           f"pipette ({pipette[0]:+.3f},{pipette[1]:+.3f})  "
           f"err {error_mm:6.2f} mm")
    draw.text((16, 28 + 4), sub, fill=(220, 220, 220), font=_FONT)
    if status:
        sw_, sh_ = 280, 40
        sx = combined.shape[1] - sw_ - 14
        draw.rectangle([(sx, 8), (sx + sw_, 8 + sh_)],
                       fill=(20, 140, 80, 235))
        draw.text((sx + 14, 14), status, fill=(255, 255, 255),
                  font=_FONT_BIG)
    final = np.concatenate([np.asarray(pil_bar), combined], axis=0)
    return final


# ---------- per-controller drive ----------

def _emit_frame(*,
                td: TopDownRenderer, pb: PyBulletRenderer,
                frames: list, plant_idx: int, n_plants: int,
                ctrl_label: str, step: int, error_mm: float,
                pipette: np.ndarray, target: np.ndarray,
                droplets: list, badge: str | None) -> None:
    """Capture one combined 3D+2D frame and append it to frames."""
    sim_rgb = pb.render()
    dish_rgb = td.render(
        active_plant_idx=plant_idx,
        pipette_world=pipette,
        target_world=target,
        droplets_world=droplets,
        header=f"top-down (pixel-accurate)",
        step=step,
        error_mm=error_mm,
        settled_badge=None,    # status badge goes on the outer header
    )
    frames.append(compose_frame(
        sim_rgb, dish_rgb,
        header=f"OT-2 inoculating ({ctrl_label})",
        status=badge,
        controller=ctrl_label,
        plant_idx=plant_idx, n_plants=n_plants,
        step=step, error_mm=error_mm,
        target=target, pipette=pipette,
    ))


def drive_to_target_pid(sim, pid: XYZPID, target: np.ndarray,
                        max_steps: int, frames: list, plant_idx: int,
                        n_plants: int, ctrl_label: str,
                        frame_every: int,
                        td: TopDownRenderer,
                        pb: PyBulletRenderer,
                        droplets: list,
                        home: np.ndarray,
                        ) -> dict:
    # Re-home (zeroed velocity) at the start of every plant, same as the RL
    # env does on reset(), so both controllers start each plant from the same
    # pose. Settle is checked post-step to match the RL env's distance_m.
    teleport_pipette(sim, home)
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=2)
    pid.reset()
    pb.set_active_target(target)
    settle_step = None
    consecutive = 0
    badge = None
    pos = precise_pipette_position(sim)
    for step in range(max_steps):
        a = pid.step(target, pos, dt=DT)
        sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
        pos = precise_pipette_position(sim)                 # read after the step
        err_m = float(np.linalg.norm(target - pos))
        if err_m < SETTLE_THRESH_MM / 1000.0:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle_step is None:
                settle_step = step
                droplets.append((plant_idx, pos.copy()))
                pb.add_droplet(pos, plant_idx)
                badge = f"INOCULATED P{plant_idx+1}"
        else:
            consecutive = 0
        if step % frame_every == 0 or step == max_steps - 1:
            _emit_frame(td=td, pb=pb, frames=frames,
                        plant_idx=plant_idx, n_plants=n_plants,
                        ctrl_label=ctrl_label, step=step,
                        error_mm=err_m * 1000.0, pipette=pos,
                        target=target, droplets=list(droplets), badge=badge)
    final_pos = precise_pipette_position(sim)
    final_err = float(np.linalg.norm(target - final_pos))
    if settle_step is None and final_err < SETTLE_THRESH_MM / 1000.0:
        droplets.append((plant_idx, final_pos.copy()))
        pb.add_droplet(final_pos, plant_idx)
    return {"final_err_mm": final_err * 1000.0,
            "settle_step": settle_step,
            "settled": settle_step is not None,
            "final_pos_world_m": final_pos.tolist()}


def drive_to_target_rl(env: OT2Env, model: PPO, target: np.ndarray,
                       max_steps: int, frames: list, plant_idx: int,
                       n_plants: int, ctrl_label: str,
                       frame_every: int,
                       td: TopDownRenderer,
                       pb: PyBulletRenderer,
                       droplets: list,
                       ) -> dict:
    obs, info = env.reset(options={"target": target})
    pb.set_active_target(target)
    settle_step = None
    consecutive = 0
    final_err = float("inf")
    final_pos = np.asarray(env._pipette())
    badge = None
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        final_err = info["distance_m"] * 1000.0
        final_pos = np.asarray(info["pipette"])
        if info["distance_m"] < SETTLE_THRESH_MM / 1000.0:
            consecutive += 1
            if consecutive >= DWELL_HOLD and settle_step is None:
                settle_step = step
                droplets.append((plant_idx, final_pos.copy()))
                pb.add_droplet(final_pos, plant_idx)
                badge = f"INOCULATED P{plant_idx+1}"
        else:
            consecutive = 0
        if step % frame_every == 0 or step == max_steps - 1:
            _emit_frame(td=td, pb=pb, frames=frames,
                        plant_idx=plant_idx, n_plants=n_plants,
                        ctrl_label=ctrl_label, step=step,
                        error_mm=final_err, pipette=final_pos,
                        target=target, droplets=list(droplets), badge=badge)
        if term or trunc:
            break
    return {"final_err_mm": final_err,
            "settle_step": settle_step,
            "settled": settle_step is not None,
            "final_pos_world_m": final_pos.tolist()}


# ---------- main ----------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--unet_weights", type=str,
                    default=str(HERE / "cv" / "weights" / "unet_256px.keras"))
    ap.add_argument("--rl_weights", type=str,
                    default=str(HERE.parent / "task11_rl_controller" / "weights" / "latest.zip"))
    ap.add_argument("--controller", choices=["pid", "rl"], default="rl")
    ap.add_argument("--gif", type=str, default=str(HERE / "results" / "demo.gif"))
    ap.add_argument("--out_json", type=str,
                    default=str(HERE / "results" / "demo.json"))
    ap.add_argument("--frame_every", type=int, default=5)
    ap.add_argument("--gif_duration", type=float, default=0.07)
    ap.add_argument("--max_steps_per_target", type=int,
                    default=MAX_STEPS_PER_TARGET)
    ap.add_argument("--dispense_z", type=float, default=DEFAULT_DISPENSE_Z_M,
                    help="OT-2 z-height for dispensing (m above world origin)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve output paths BEFORE make_sim chdirs into the sim folder.
    args.gif = str(Path(args.gif).resolve())
    args.out_json = str(Path(args.out_json).resolve())
    args.image = str(Path(args.image).resolve())
    args.unet_weights = str(Path(args.unet_weights).resolve())
    args.rl_weights = str(Path(args.rl_weights).resolve())
    print(f"=== Task 12 demo: {Path(args.image).name} -> CV -> {args.controller.upper()} ===")

    # 1. CV
    print(f"\n[1/4] U-Net inference on {Path(args.image).name}")
    result, img_orig, _ = detect_tips_in_image(args.unet_weights, args.image)
    print(f"      detected {result.n_detected} plants")

    # 2. Dish bbox + sim-dish calibration.
    # Warn if the requested dispense z is below the envelope clip floor
    # (z_min + 0.005); otherwise the logged calibration z and the z actually
    # commanded could quietly disagree.
    z_floor = float(PRACTICAL_ENVELOPE.low[2]) + 0.005
    if args.dispense_z < z_floor:
        print(f"      [WARN] --dispense_z {args.dispense_z:.4f} is below the "
              f"envelope clip floor {z_floor:.4f}; it will be raised to "
              f"{z_floor:.4f}. Using {z_floor:.4f}.")
        args.dispense_z = z_floor
    dish_bbox = get_dish_bbox(img_orig)
    cal = Calibration.sim_dish(dish_bbox, z_m=args.dispense_z)
    print(f"\n[2/4] calibration:")
    print(f"      dish bbox (px) = {dish_bbox}")
    print(f"      {cal}")
    for plant in result.plants:
        wx, wy, wz = cal.pixel_to_world(plant.tip[0], plant.tip[1])
        ok = cal.contains_in_envelope(plant.tip[0], plant.tip[1])
        flag = "OK" if ok else "OUTSIDE-ENVELOPE"
        print(f"      P{plant.plant_index+1}  pixel ({plant.tip[0]:4d}, "
              f"{plant.tip[1]:4d})  ->  world ({wx:+.4f}, {wy:+.4f}, "
              f"{wz:+.4f}) m  {flag}")

    tips_world = cal.map_tips([plant.tip for plant in result.plants])
    # Clip into the envelope so a near-rim tip doesn't send a controller
    # chasing an unreachable target for a whole episode.
    clipped = []
    for w in tips_world:
        clipped.append(np.clip(w,
                               PRACTICAL_ENVELOPE.low + 0.005,
                               PRACTICAL_ENVELOPE.high - 0.005))

    # 3. Build annotated dish image (used by BOTH 2D and 3D renderers)
    annotated_rgb = build_annotated_dish_image(
        img_orig=img_orig, dish_bbox=dish_bbox, plants=result.plants)
    td = TopDownRenderer(img_orig=img_orig, dish_bbox=dish_bbox,
                         plants=result.plants, calibration=cal,
                         annotated_rgb=annotated_rgb)

    frames: list[np.ndarray] = []
    per_target_log: list[dict] = []
    droplets: list[tuple[int, np.ndarray]] = []   # (plant_idx, world_xyz)

    if args.controller == "pid":
        print("\n[3/4] PID-SIMC controller (Task 10 v3 SIMC gains)")
        sim = make_sim(num_agents=1, render=False)
        sim.run([[0, 0, 0, 0]], num_steps=20)
        spec_id = sim.specimenIds[0]
        pb = PyBulletRenderer(specimen_id=spec_id, annotated_rgb=annotated_rgb)
        pid = build_pid_simc()
        home = PRACTICAL_ENVELOPE.center  # per-plant re-home, like the RL reset
        for i, target in enumerate(clipped):
            print(f"   P{i+1}/{len(clipped)} target={target.tolist()}")
            log = drive_to_target_pid(sim, pid, target,
                                      args.max_steps_per_target, frames,
                                      plant_idx=i, n_plants=len(clipped),
                                      ctrl_label="PID-SIMC",
                                      frame_every=args.frame_every,
                                      td=td, pb=pb, droplets=droplets,
                                      home=home)
            log["target_world_m"] = target.tolist()
            log["tip_pixel"] = result.plants[i].tip
            log["plant_index"] = i
            per_target_log.append(log)
            print(f"        err={log['final_err_mm']:.3f} mm  settle={log['settle_step']}")
    else:
        print("\n[3/4] RL-PPO controller (Path B best model)")
        env = OT2Env(seed=0, max_steps=args.max_steps_per_target,
                     success_threshold=SETTLE_THRESH_MM / 1000.0,
                     dwell_steps=DWELL_HOLD, randomise_start=False)
        env.reset()    # build the sim so specimenIds is populated
        spec_id = env.sim.specimenIds[0]
        pb = PyBulletRenderer(specimen_id=spec_id, annotated_rgb=annotated_rgb)
        model = PPO.load(args.rl_weights, device="cpu")
        for i, target in enumerate(clipped):
            print(f"   P{i+1}/{len(clipped)} target={target.tolist()}")
            log = drive_to_target_rl(env, model, target,
                                     args.max_steps_per_target, frames,
                                     plant_idx=i, n_plants=len(clipped),
                                     ctrl_label="RL-PPO",
                                     frame_every=args.frame_every,
                                     td=td, pb=pb, droplets=droplets)
            log["target_world_m"] = target.tolist()
            log["tip_pixel"] = result.plants[i].tip
            log["plant_index"] = i
            per_target_log.append(log)
            print(f"        err={log['final_err_mm']:.3f} mm  settle={log['settle_step']}")

    # 3b. Hold on the last frame for a moment before tearing down the sim.
    if per_target_log:
        last = per_target_log[-1]
        final_pos = np.asarray(last["final_pos_world_m"])
        target_arr = np.asarray(last["target_world_m"])
        n_ok = sum(1 for t in per_target_log if t["settled"])
        for _ in range(20):
            _emit_frame(td=td, pb=pb, frames=frames,
                        plant_idx=len(per_target_log) - 1,
                        n_plants=len(per_target_log),
                        ctrl_label=args.controller.upper(),
                        step=last.get("settle_step") or 0,
                        error_mm=last["final_err_mm"],
                        pipette=final_pos, target=target_arr,
                        droplets=list(droplets),
                        badge=f"{n_ok}/{len(per_target_log)} INOCULATED")

    # Safe to close now.
    if args.controller == "rl":
        env.close()
    else:
        sim.close()

    # 4. Write outputs
    gif_path = Path(args.gif)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] writing {gif_path} ({len(frames)} frames)")
    # 128-colour palette to keep the GIF small. The photo plus flat overlays
    # compress fine at that depth.
    quantised: list[np.ndarray] = []
    for f in frames:
        im = Image.fromarray(f).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=128)
        quantised.append(np.asarray(im.convert("RGB")))
    imageio.mimsave(str(gif_path), quantised, duration=args.gif_duration,
                    loop=0, subrectangles=True)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "image": args.image,
        "controller": args.controller,
        "calibration": {
            "label": cal.label,
            "origin_x_m": cal.origin_x_m, "origin_y_m": cal.origin_y_m,
            "m_per_px_row": cal.m_per_px_row, "m_per_px_col": cal.m_per_px_col,
            "z_m": cal.z_m, "dish_bbox_px": list(dish_bbox),
            "row_world_axis": cal.row_world_axis, "row_sign": cal.row_sign,
            "col_world_axis": cal.col_world_axis, "col_sign": cal.col_sign,
        },
        "cv_detections": result.to_jsonable(),
        "per_target": per_target_log,
        "summary": {
            "n_targets": len(per_target_log),
            "mean_err_mm": float(np.mean([t["final_err_mm"] for t in per_target_log])),
            "max_err_mm": float(np.max([t["final_err_mm"] for t in per_target_log])),
            "min_err_mm": float(np.min([t["final_err_mm"] for t in per_target_log])),
            "success_rate_lt_1mm": float(np.mean(
                [1.0 if t["settled"] else 0.0 for t in per_target_log])),
        },
    }, indent=2))
    print(f"      wrote {out_json}")

    print("\n=== summary ===")
    for t in per_target_log:
        ok = "OK" if t["settled"] else "MISS"
        print(f"  P{t['plant_index']+1}: err={t['final_err_mm']:6.3f} mm  "
              f"settle_step={t['settle_step']}  [{ok}]")


if __name__ == "__main__":
    main()
