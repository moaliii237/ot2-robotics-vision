"""Per-axis PID controller for the OT-2 sim.

A few things worth knowing about this controller (full write-up in
task10_pid_controller/README.md):

The output is in action-space units, [-1, 1], not m/s. The sim turns an action
into a joint velocity command and then a motor force, so working in "metres per
second" at this layer would be pretending to a precision we don't have.

Derivative is taken on the measurement rather than the error, which avoids the
kick you'd otherwise get every time the setpoint jumps (tuning and the demo both
drive through a sequence of targets).

Anti-windup is conditional integration: if the output is saturated and the error
points the same way as the integral, we stop accumulating. Simplest thing that
stays stable without resorting to back-calculation.

`min_output` is a floor that punches through the static-friction dead zone Task 9
turned up for small actions. It only kicks in while |error| > settle_threshold,
so the controller still settles to rest at the target rather than buzzing.

The z axis has a stronger motor (800 N vs 500 N on x/y) and a higher velocity gain
per unit action (~0.56 vs ~0.4 m/s), so its default gains are lower. See
task09_robotics_environment/README.md for where those numbers come from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class AxisPID:
    """PID for a single axis."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0

    # Saturation limits on the controller output. The sim clips actions to
    # [-1, 1] internally too, so these defaults match.
    out_min: float = -1.0
    out_max: float = +1.0

    # Below |error| <= settle_threshold the output is forced to 0 so the
    # carriage isn't fighting static friction at the target.
    settle_threshold: float = 5.0e-4   # 0.5 mm

    # Smallest output magnitude that actually moves the carriage. Anything
    # nonzero below this gets bumped up to it (only while still outside settle).
    min_output: float = 0.12

    # Internal state.
    _integral: float = field(default=0.0, init=False)
    _last_measurement: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._last_measurement = None

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute the next control action.

        Args:
            setpoint: desired axis value in metres.
            measurement: current axis value in metres.
            dt: time step in seconds. Use the sim's `1/240`.

        Returns:
            Action component in [out_min, out_max].
        """
        error = setpoint - measurement

        if abs(error) <= self.settle_threshold:
            # Park: zero output, but keep the integral as it was so we don't
            # lose accumulated trim if the target shifts a bit.
            self._last_measurement = measurement
            return 0.0

        # Proportional.
        p = self.kp * error

        # Derivative on measurement. d(measurement)/dt has opposite sign from
        # d(error)/dt when setpoint is constant, so we negate.
        if self._last_measurement is None:
            d_meas = 0.0
        else:
            d_meas = (measurement - self._last_measurement) / dt
        self._last_measurement = measurement
        d = -self.kd * d_meas

        # Tentative output before integral commit.
        u_unsat = p + self.ki * self._integral + d

        # Conditional anti-windup: only accumulate integral if not pushing
        # further into saturation in the same direction as the error.
        sat_high = u_unsat >= self.out_max
        sat_low = u_unsat <= self.out_min
        pushing_into_sat = (sat_high and error > 0) or (sat_low and error < 0)
        if not pushing_into_sat:
            self._integral += error * dt
            u_unsat = p + self.ki * self._integral + d

        # Saturate.
        u = float(np.clip(u_unsat, self.out_min, self.out_max))

        # Min-output floor: punch through the static-friction dead zone.
        if 0 < abs(u) < self.min_output:
            u = self.min_output * (1.0 if u > 0 else -1.0)

        return u


@dataclass
class XYZPID:
    """Three independent `AxisPID` instances for x, y, z.

    Defaults are the result of `tune_pid.py`'s search (this file's defaults
    will be overwritten when tuning completes, treat them as a sensible start
    not a final answer).
    """

    x: AxisPID = field(default_factory=lambda: AxisPID(kp=35.0, ki=4.0, kd=0.8))
    y: AxisPID = field(default_factory=lambda: AxisPID(kp=35.0, ki=4.0, kd=0.8))
    # z motor is stronger -> smaller gains, otherwise overshoots.
    z: AxisPID = field(default_factory=lambda: AxisPID(kp=25.0, ki=3.0, kd=0.6))

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()
        self.z.reset()

    def step(
        self,
        setpoint: Sequence[float] | np.ndarray,
        measurement: Sequence[float] | np.ndarray,
        dt: float = 1.0 / 240.0,
    ) -> np.ndarray:
        """Return the per-axis action as a numpy array of shape (3,)."""
        sp = np.asarray(setpoint, dtype=np.float64)
        m = np.asarray(measurement, dtype=np.float64)
        if sp.shape != (3,) or m.shape != (3,):
            raise ValueError("setpoint and measurement must be 3-vectors")
        return np.array(
            [
                self.x.step(sp[0], m[0], dt),
                self.y.step(sp[1], m[1], dt),
                self.z.step(sp[2], m[2], dt),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "XYZPID":
        """Build from a dict of {axis: {kp, ki, kd, ...}}."""
        def _build(sub: dict) -> AxisPID:
            return AxisPID(**sub)
        return cls(x=_build(d["x"]), y=_build(d["y"]), z=_build(d["z"]))

    def to_dict(self) -> dict:
        def _dump(a: AxisPID) -> dict:
            return {
                "kp": a.kp, "ki": a.ki, "kd": a.kd,
                "out_min": a.out_min, "out_max": a.out_max,
                "settle_threshold": a.settle_threshold,
                "min_output": a.min_output,
            }
        return {"x": _dump(self.x), "y": _dump(self.y), "z": _dump(self.z)}
