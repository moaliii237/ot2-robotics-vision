"""Fast unit tests for the pure-numpy core (no PyBullet / SB3 needed).

These cover the three pieces that are easy to get wrong and have no sim
dependency: the pixel<->world calibration, the PID anti-windup, and the
work-envelope containment check. Run with:

    pytest robotics/tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# calibration.py lives in the task12 folder, not in the ot2 package.
TASK12 = Path(__file__).resolve().parents[1] / "task12_13_integration_benchmarking"
if str(TASK12) not in sys.path:
    sys.path.insert(0, str(TASK12))

from ot2.control.pid import AxisPID
from ot2.envelope import Envelope, PRACTICAL_ENVELOPE

from calibration import Calibration


# ----- Calibration pixel <-> world round-trip -----

def test_calibration_roundtrip():
    bbox = (746, 153, 3456, 2853)  # x1, y1, x2, y2
    cal = Calibration.sim_dish(bbox)
    # Pick a handful of raw pixels inside the dish and round-trip them.
    pixels = [(153, 746), (2853, 3456), (1500, 2100), (500, 1000)]
    for raw_row, raw_col in pixels:
        x, y, _z = cal.pixel_to_world(raw_row, raw_col)
        back_row, back_col = cal.world_to_pixel(x, y)
        assert back_row == pytest.approx(raw_row, abs=1e-6)
        assert back_col == pytest.approx(raw_col, abs=1e-6)


def test_calibration_dish_center_maps_to_plate_center():
    bbox = (746, 153, 3456, 2853)
    cal = Calibration.sim_dish(bbox)
    cx_px = (bbox[0] + bbox[2]) / 2.0
    cy_px = (bbox[1] + bbox[3]) / 2.0
    x, y, _z = cal.pixel_to_world(cy_px, cx_px)  # (row, col) = (y, x)
    # Dish centre pixel should land on the plate centre in world coords.
    assert x == pytest.approx(0.18270, abs=1e-4)
    assert y == pytest.approx(0.13700, abs=1e-4)


# ----- AxisPID anti-windup -----

def test_axispid_antiwindup_stops_integral_when_saturated():
    # Big gains + a large constant error drive the output hard into the
    # +out_max rail. Conditional anti-windup should stop the integral from
    # growing without bound while it stays pinned there.
    pid = AxisPID(kp=1000.0, ki=1000.0, kd=0.0, settle_threshold=0.0, min_output=0.0)
    setpoint, measurement, dt = 1.0, 0.0, 1.0 / 240.0
    for _ in range(50):
        out = pid.step(setpoint, measurement, dt)
        assert out <= pid.out_max  # never exceeds the saturation rail
    saturated_integral = pid._integral
    # A few more steps should not keep growing the integral (it is frozen
    # while pushing further into the same-direction saturation).
    for _ in range(50):
        pid.step(setpoint, measurement, dt)
    assert pid._integral == pytest.approx(saturated_integral)


def test_axispid_settle_zone_outputs_zero():
    pid = AxisPID(kp=50.0, ki=5.0, kd=0.0, settle_threshold=5.0e-4, min_output=0.0)
    # Error of 0.1 mm is inside the 0.5 mm settle band -> output parked at 0.
    out = pid.step(setpoint=0.0001, measurement=0.0, dt=1.0 / 240.0)
    assert out == 0.0


# ----- Envelope.contains -----

def test_envelope_contains_basic():
    env = Envelope(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)
    assert env.contains(np.array([0.5, 0.5, 0.5]))
    assert env.contains(np.array([0.0, 0.0, 0.0]))      # on the low corner
    assert env.contains(np.array([1.0, 1.0, 1.0]))      # on the high corner
    assert not env.contains(np.array([-0.01, 0.5, 0.5]))
    assert not env.contains(np.array([0.5, 0.5, 1.01]))


def test_envelope_contains_margin():
    env = Envelope(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)
    # With a 0.1 margin the corner is now outside, but the centre is still in.
    assert not env.contains(np.array([0.0, 0.0, 0.0]), margin=0.1)
    assert env.contains(np.array([0.5, 0.5, 0.5]), margin=0.1)


def test_practical_envelope_center_is_inside():
    assert PRACTICAL_ENVELOPE.contains(PRACTICAL_ENVELOPE.center)
