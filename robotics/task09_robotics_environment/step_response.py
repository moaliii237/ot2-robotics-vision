"""Step-response characterisation per axis.

For each axis and each step input magnitude, record the pipette position over
time and extract standard step-response metrics:

- rise time t_r (10% to 90% of final value)
- peak time t_p
- percent overshoot %OS
- settling time t_s (5% band)
- steady-state value (final - reference)

Also emits the raw time-series so `task10_pid_controller/sysid.py` can fit a
plant model from the same data.

Run from project root:
    python task09_robotics_environment\step_response.py

Writes:
    results/step_response.json (per-axis time series + extracted metrics)
    results/step_response.png  (per-axis step response plot)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_pipette


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
DURATION_S = 0.20          # 48 sim steps
STEPS = int(DURATION_S / DT)

# Smaller magnitudes than the smoke test. With duration 200 ms and our
# axis-aware low-end start (below), magnitudes 0.1/0.2/0.3 give clean
# velocity-step responses that finish well before any joint stop.
STEP_MAGNITUDES = [0.1, 0.2, 0.3]
SETTLE_BAND = 0.05         # 5% settling band as fraction of final value


# Start position per axis: at the LOW end of the envelope on the stepped axis
# (so a positive step has maximum free travel), at the centre of the other axes.
def axis_start(axis: int) -> np.ndarray:
    low = PRACTICAL_ENVELOPE.low.copy()
    centre = PRACTICAL_ENVELOPE.center
    start = centre.copy()
    start[axis] = low[axis] + 0.005  # 5 mm in from the floor
    return start


def extract_step_metrics(t: np.ndarray, y: np.ndarray, ref_velocity: float) -> dict:
    """Given a velocity step response y(t) (the *velocity* on the axis, m/s),
    extract standard time-domain metrics.

    For our plant, action -> velocity is essentially first-order, so we
    expect no overshoot and a clean exponential approach to ref_velocity.
    """
    # Steady-state estimate: last 5% of samples averaged
    n = len(y)
    y_ss = float(np.mean(y[int(0.95 * n):]))

    # Steady-state error vs commanded velocity reference
    ss_err = y_ss - ref_velocity

    # Rise time 10% -> 90% of final value (final = y_ss)
    target_lo = 0.10 * y_ss
    target_hi = 0.90 * y_ss
    t_lo = None
    t_hi = None
    if y_ss > 0:
        for i, v in enumerate(y):
            if t_lo is None and v >= target_lo:
                t_lo = t[i]
            if v >= target_hi:
                t_hi = t[i]
                break
    elif y_ss < 0:
        for i, v in enumerate(y):
            if t_lo is None and v <= target_lo:
                t_lo = t[i]
            if v <= target_hi:
                t_hi = t[i]
                break
    rise_time_s = (t_hi - t_lo) if (t_lo is not None and t_hi is not None) else None

    # Peak time + overshoot
    if y_ss >= 0:
        peak_idx = int(np.argmax(y))
        peak_value = float(y[peak_idx])
        overshoot_pct = ((peak_value - y_ss) / y_ss * 100) if y_ss > 1e-9 else 0.0
    else:
        peak_idx = int(np.argmin(y))
        peak_value = float(y[peak_idx])
        overshoot_pct = ((peak_value - y_ss) / y_ss * 100) if abs(y_ss) > 1e-9 else 0.0
    peak_time_s = float(t[peak_idx])

    # Settling time (5% band)
    band_lo = (1.0 - SETTLE_BAND) * y_ss
    band_hi = (1.0 + SETTLE_BAND) * y_ss
    lo = min(band_lo, band_hi)
    hi = max(band_lo, band_hi)
    settling_time_s = None
    # walk from the end; find the last excursion outside the band
    for i in range(len(y) - 1, -1, -1):
        if y[i] < lo or y[i] > hi:
            settling_time_s = float(t[i + 1]) if i + 1 < len(y) else None
            break
    if settling_time_s is None:
        settling_time_s = 0.0  # already inside band from t=0

    return {
        "ref_velocity_m_per_s": ref_velocity,
        "ss_value_m_per_s": y_ss,
        "ss_error_m_per_s": ss_err,
        "rise_time_s_10_90": rise_time_s,
        "peak_time_s": peak_time_s,
        "peak_value_m_per_s": peak_value,
        "overshoot_pct": overshoot_pct,
        "settling_time_s_5pct": settling_time_s,
    }


def probe_axis(sim, axis: int, magnitude: float) -> dict:
    """Reset, teleport to the axis-aware low-end start, command a step on
    `axis`, record y(t)."""
    sim.reset(num_agents=1)
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)
    teleport_pipette(sim, axis_start(axis))
    sim.run([[0.0, 0.0, 0.0, 0]], num_steps=2)

    pos_t = [precise_pipette_position(sim)[axis]]
    t = [0.0]

    action = [0.0, 0.0, 0.0, 0]
    action[axis] = magnitude

    for step in range(STEPS):
        sim.run([action], num_steps=1)
        pos_t.append(precise_pipette_position(sim)[axis])
        t.append((step + 1) * DT)

    pos = np.array(pos_t)
    time = np.array(t)
    # Velocity = numerical derivative
    vel = np.gradient(pos, time)
    return {"axis": axis, "magnitude": magnitude, "time": time, "pos": pos, "vel": vel}


def main():
    sim = make_sim(num_agents=1, render=False)
    sim.run([[0, 0, 0, 0]], num_steps=20)

    # Empirical velocity gains from Task 9 smoke test (for ref_velocity = gain * magnitude)
    # x: 0.97 m/s per unit, y: 0.97, z: 1.12. Use these as the reference.
    gains = {0: 0.97, 1: 0.97, 2: 1.12}

    all_data = {}
    fig, axes = plt.subplots(3, len(STEP_MAGNITUDES), figsize=(5 * len(STEP_MAGNITUDES), 9), sharex=True)
    if len(STEP_MAGNITUDES) == 1:
        axes = axes.reshape(-1, 1)

    for axis in (0, 1, 2):
        all_data[f"axis_{['x','y','z'][axis]}"] = {}
        for ci, mag in enumerate(STEP_MAGNITUDES):
            res = probe_axis(sim, axis, mag)
            ref_v = gains[axis] * mag
            metrics = extract_step_metrics(res["time"], res["vel"], ref_v)
            all_data[f"axis_{['x','y','z'][axis]}"][f"mag_{mag}"] = {
                "metrics": metrics,
                "time": res["time"].tolist(),
                "pos_m": res["pos"].tolist(),
                "vel_m_per_s": res["vel"].tolist(),
            }
            ax = axes[axis, ci]
            ax.plot(res["time"], res["vel"], color="#4C9AFF", label="measured velocity")
            ax.axhline(ref_v, color="green", linestyle="--", label=f"commanded {ref_v:.3f}")
            if metrics["rise_time_s_10_90"]:
                ax.axvspan(0, metrics["rise_time_s_10_90"], alpha=0.1, color="orange",
                           label=f"rise {metrics['rise_time_s_10_90']*1000:.1f} ms")
            ax.set_xlim(0, DURATION_S)
            ax.grid(alpha=0.3)
            ax.set_title(f"axis {['x','y','z'][axis]} step {mag}", fontsize=10)
            ax.set_ylabel("vel (m/s)")
            if axis == 2:
                ax.set_xlabel("t (s)")
            ax.legend(fontsize=7)
            print(f"axis={['x','y','z'][axis]}  mag={mag}  ref_v={ref_v:.3f}  "
                  f"rise={metrics['rise_time_s_10_90']*1000:.1f} ms  "
                  f"overshoot={metrics['overshoot_pct']:.2f}%  "
                  f"settle={metrics['settling_time_s_5pct']*1000:.1f} ms  "
                  f"ss_err={metrics['ss_error_m_per_s']*1000:.3f} mm/s")
    sim.close()

    fig.suptitle("Velocity step response per axis at three step magnitudes")
    fig.tight_layout()
    out_png = RESULTS / "step_response.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_png}")

    out_json = RESULTS / "step_response.json"
    out_json.write_text(json.dumps({
        "settings": {"dt": DT, "duration_s": DURATION_S, "step_magnitudes": STEP_MAGNITUDES,
                     "settle_band_pct": SETTLE_BAND * 100, "gains_used": gains},
        "data": all_data,
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
