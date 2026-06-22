"""Pixel-to-world calibration for the petri-dish images.

A root tip has no 3-D ground truth, it's just a pixel in a photo. So this
isn't "derived" from the render, it's a defined linear convention: map the
dish photo onto the 150 mm square footprint of the sim's specimen plate,
which sits at a known world pose:

    plate footprint   0.150 m x 0.150 m  (from custom.urdf <box size>)
    plate centre      (0.18270, 0.13700) m in the OT-2 world frame
                      ( = robot base (0,0,0.03) + loadURDF offset
                        (0.18275-0.00005, 0.163-0.026, 0.057), sim_class.py )
    agar surface z    0.087 + 0.0075 = 0.0945 m  (top face of the box)
    dispense z        0.175 m  (z_min on the practical envelope is 0.170, so
                      the pipette hovers here and drops)

Axis routing: the image is cropped to the dish bbox via get_dish_bbox()
first (same as the Task 2 / Task 5 inference path), then tip pixels are in
dish-local (row, col) with the origin at the dish top-left. The convention:

    image column (left->right)  ->  world +y      (col_sign = +1)
    image row    (top->bottom)  ->  world -x      (row_sign = -1)

so:

    world_x = (dish_center_x + 0.075) - dish_row * (0.150 / dish_h_px)
    world_y = (dish_center_y - 0.075) + dish_col * (0.150 / dish_w_px)
    world_z = dispense_z

Which image axis maps to which world axis, and the signs, are a choice. All
8 axis-aligned orientations land the tips on the same footprint; we just
fix one. It is not meant to be the unique orientation implied by the plate's
-pi/2 render rotation. PyBullet cube-unwraps the box texture (it doesn't
paint the whole photo onto the top face), so the render can't pin an
absolute orientation anyway, and nothing here depends on it.

verify_calibration.py checks the convention is self-consistent: round-trip
through world_to_pixel, dish centre -> plate centre, and that the four
corners / detected tips map to distinct world points whose left/right and
near/far order matches the photo. The demo's 2-D overlay also draws with
world_to_pixel, so the pipette appears at the exact pixel its pose maps to.

from_explicit() takes an arbitrary measured affine; sim_dish() bakes in the
constants above for the Y2B sim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ot2.envelope import PRACTICAL_ENVELOPE


# ----- constants read off the sim source -----

# custom.urdf: <box size="0.15 0.15 0.015"/>
DISH_SIZE_M = 0.150

# sim_class.py loadURDF (agent at world origin):
#   offset = [0.18275 - 0.00005, 0.163 - 0.026, 0.057]
#   base z = position[2] + offset[2] = 0.03 + 0.057 = 0.087
DISH_CENTER_X_M = 0.18270
DISH_CENTER_Y_M = 0.13700
DISH_BASE_Z_M = 0.08700                # center of the URDF box
DISH_THICKNESS_M = 0.015
AGAR_SURFACE_Z_M = DISH_BASE_Z_M + DISH_THICKNESS_M / 2   # 0.0945 m

# Hover height for dispensing. Above the envelope z_min (0.170) but close
# enough to the agar that a drop looks plausible. The chassis floor stops
# descent below ~0.170 anyway.
DEFAULT_DISPENSE_Z_M = 0.175


@dataclass(frozen=True)
class Calibration:
    """Affine pixel-to-world transform.

    Two constructors:

    * from_explicit(...) takes a raw affine (origin + scale + optional
      flips), for a directly measured calibration.
    * sim_dish(...) reads the dish geometry from the sim source (URDF +
      sim_class.py loadURDF). Pass raw-image pixel coords and the dish bbox
      from get_dish_bbox. This is the one to use for the Y2B-OT2 sim.
    """
    # World metres at the dish top-left corner (dish_row=0, dish_col=0).
    # sim_dish derives these from DISH_CENTER +/- 0.075 m, with the dish's
    # -90deg z-rotation already folded in.
    origin_x_m: float
    origin_y_m: float
    # Metres per dish-relative pixel, per axis. Two scales because the dish
    # bbox is rarely square (e.g. 2750 x 2700 px).
    m_per_px_row: float
    m_per_px_col: float
    z_m: float
    # Tip pixels come in raw-image coords; subtract (dish_y1, dish_x1) to get
    # dish-relative coords before the transform.
    dish_y1_px: int = 0
    dish_x1_px: int = 0
    # Row axis: which world axis it drives and the sign. sim_dish uses -1
    # (row down = world -x, from the -pi/2 rotation).
    row_world_axis: str = "x"     # "x" or "y"
    row_sign: int = +1            # +1 or -1
    col_world_axis: str = "y"     # the other one
    col_sign: int = +1
    label: str = "unspecified"

    def pixel_to_world(self, raw_row: float, raw_col: float
                       ) -> tuple[float, float, float]:
        """Convert a raw-image pixel (row, col) into (x, y, z) world metres."""
        dish_row = raw_row - self.dish_y1_px
        dish_col = raw_col - self.dish_x1_px
        delta_row = dish_row * self.m_per_px_row * self.row_sign
        delta_col = dish_col * self.m_per_px_col * self.col_sign
        # Route row/col onto world x/y per the *_world_axis fields.
        dx = delta_row if self.row_world_axis == "x" else delta_col
        dy = delta_col if self.col_world_axis == "y" else delta_row
        return (float(self.origin_x_m + dx),
                float(self.origin_y_m + dy),
                float(self.z_m))

    def world_to_pixel(self, world_x: float, world_y: float
                       ) -> tuple[float, float]:
        """Inverse of pixel_to_world: world (x, y) -> (raw_row, raw_col).

        The demo's top-down render uses this to draw the pipette's live world
        position back onto the dish photo at the right pixel.
        """
        dx = world_x - self.origin_x_m
        dy = world_y - self.origin_y_m
        # Undo the axis routing.
        if self.row_world_axis == "x":
            delta_row = dx
            delta_col = dy
        else:
            delta_row = dy
            delta_col = dx
        dish_row = delta_row / (self.m_per_px_row * self.row_sign)
        dish_col = delta_col / (self.m_per_px_col * self.col_sign)
        return (dish_row + self.dish_y1_px,
                dish_col + self.dish_x1_px)

    def map_tips(self, tips: Iterable[tuple[int, int]]) -> np.ndarray:
        out = np.array(
            [self.pixel_to_world(r, c) for r, c in tips],
            dtype=np.float64,
        )
        return out.reshape(-1, 3)

    def contains_in_envelope(self, raw_row: float, raw_col: float,
                             margin: float = 0.0) -> bool:
        x, y, z = self.pixel_to_world(raw_row, raw_col)
        return PRACTICAL_ENVELOPE.contains(np.array([x, y, z]), margin=margin)

    # ----- factories -----

    @classmethod
    def from_explicit(cls, *,
                      origin_x_m: float, origin_y_m: float,
                      m_per_px_row: float, m_per_px_col: float,
                      z_m: float,
                      row_world_axis: str = "x", row_sign: int = +1,
                      col_world_axis: str = "y", col_sign: int = +1,
                      dish_y1_px: int = 0, dish_x1_px: int = 0,
                      label: str = "explicit") -> "Calibration":
        return cls(origin_x_m=origin_x_m, origin_y_m=origin_y_m,
                   m_per_px_row=m_per_px_row, m_per_px_col=m_per_px_col,
                   z_m=z_m, row_world_axis=row_world_axis, row_sign=row_sign,
                   col_world_axis=col_world_axis, col_sign=col_sign,
                   dish_y1_px=dish_y1_px, dish_x1_px=dish_x1_px, label=label)

    @classmethod
    def sim_dish(cls,
                 dish_bbox_px: tuple[int, int, int, int],
                 z_m: float = DEFAULT_DISPENSE_Z_M,
                 dish_center: tuple[float, float] = (DISH_CENTER_X_M, DISH_CENTER_Y_M),
                 dish_size_m: float = DISH_SIZE_M,
                 ) -> "Calibration":
        """Build a calibration for the Y2B-OT2 sim's specimen plate.

        Args:
            dish_bbox_px: (x1, y1, x2, y2) from get_dish_bbox, raw-image px.
            z_m:          dispense height (default 0.175 m).
            dish_center:  (x, y) world metres of the plate centre. Default is
                          from the URDF + loadURDF call.
            dish_size_m:  plate side length (default 0.15 m).

        Under the URDF's -pi/2 z-rotation:
          - image row (top->bottom) -> dish local +y -> world -x
          - image col (left->right) -> dish local +x -> world +y
        So the origin (top-left of the dish) sits at
          world_x = dish_center_x + 0.075
          world_y = dish_center_y - 0.075
        with per-pixel deltas
          d(world_x) / d(row) = -dish_size_m / dish_h_px      (row_sign = -1)
          d(world_y) / d(col) = +dish_size_m / dish_w_px      (col_sign = +1)
        """
        x1, y1, x2, y2 = dish_bbox_px
        dish_w_px = x2 - x1
        dish_h_px = y2 - y1
        if dish_w_px <= 0 or dish_h_px <= 0:
            raise ValueError(f"bad dish bbox: {dish_bbox_px}")
        m_per_px_col = dish_size_m / dish_w_px
        m_per_px_row = dish_size_m / dish_h_px
        return cls(
            origin_x_m=dish_center[0] + dish_size_m / 2,
            origin_y_m=dish_center[1] - dish_size_m / 2,
            m_per_px_row=m_per_px_row,
            m_per_px_col=m_per_px_col,
            z_m=z_m,
            row_world_axis="x", row_sign=-1,
            col_world_axis="y", col_sign=+1,
            dish_y1_px=int(y1), dish_x1_px=int(x1),
            label=f"sim_dish (bbox={dish_bbox_px}, z={z_m:.3f}m)",
        )

    @classmethod
    def placeholder(cls, image_shape: tuple[int, int],
                    z_m: float | None = None) -> "Calibration":
        """Stretch the raw image onto the practical envelope (debug only)."""
        H, W = image_shape
        env = PRACTICAL_ENVELOPE
        z = z_m if z_m is not None else float(env.center[2])
        return cls(
            origin_x_m=float(env.low[0]),
            origin_y_m=float(env.low[1]),
            m_per_px_row=(env.high[1] - env.low[1]) / H,
            m_per_px_col=(env.high[0] - env.low[0]) / W,
            z_m=z,
            row_world_axis="y", row_sign=+1,
            col_world_axis="x", col_sign=+1,
            label="PLACEHOLDER (image stretched to envelope; not physical)",
        )


def default(image_shape: tuple[int, int]) -> Calibration:
    return Calibration.placeholder(image_shape=image_shape)


# ---------- self-test ----------

if __name__ == "__main__":
    # Fake a dish bbox over the centre 2700x2700 of a 4202x3006 image.
    H, W = 3006, 4202
    bbox = (746, 153, 3456, 2853)   # x1, y1, x2, y2
    cal = Calibration.sim_dish(bbox)
    print(cal)
    print(f"  agar surface z = {AGAR_SURFACE_Z_M:.4f} m, dispense z = {cal.z_m:.4f} m")
    print(f"  dish width = {bbox[2]-bbox[0]} px, height = {bbox[3]-bbox[1]} px")
    print(f"  m/px col = {cal.m_per_px_col*1000:.4f} mm/px")
    print(f"  m/px row = {cal.m_per_px_row*1000:.4f} mm/px")
    for label, (r, c) in [
        ("dish top-left   (raw)", (bbox[1],   bbox[0])),
        ("dish top-right  (raw)", (bbox[1],   bbox[2])),
        ("dish bot-left   (raw)", (bbox[3],   bbox[0])),
        ("dish bot-right  (raw)", (bbox[3],   bbox[2])),
        ("dish centre     (raw)", ((bbox[1]+bbox[3])//2, (bbox[0]+bbox[2])//2)),
    ]:
        x, y, z = cal.pixel_to_world(r, c)
        ok = cal.contains_in_envelope(r, c)
        print(f"  {label}  -> world ({x:+.4f}, {y:+.4f}, {z:+.4f})  envelope_ok={ok}")
