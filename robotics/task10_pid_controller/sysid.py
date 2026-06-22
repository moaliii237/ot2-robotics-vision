"""System identification: fit a first-order plant model per axis.

Reads `task09_robotics_environment/results/step_response.json` (from the
Task 9 step-response characterisation) and fits

    v(t) = K * u_step * (1 - exp(-t/tau))

per axis, where K is the steady-state velocity gain (m/s per unit action)
and tau is the time constant (s).

A second-order critically-damped form was tried too and didn't lower the
residual, so first-order it is. From the action input, PyBullet's velocity
loop really does behave first-order.

Writes:
    results/plant_model.json: per-axis K, tau, fit residual, raw + fitted curves
    results/sysid_fit.png:    per-axis measured vs fitted velocity, with the model overlay
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

T9_RESULTS = HERE.parent / "task09_robotics_environment" / "results" / "step_response.json"


def first_order_step(t: np.ndarray, K: float, tau: float, u_step: float) -> np.ndarray:
    """v(t) = K * u_step * (1 - exp(-t/tau))."""
    return K * u_step * (1.0 - np.exp(-t / tau))


def fit_axis(traces: dict) -> dict:
    """Fit one (K, tau) jointly across every step magnitude of the axis.

    All magnitude traces are concatenated and the per-sample magnitude is
    passed into curve_fit so a single K and tau cover them all.
    """
    all_t: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_mag: list[np.ndarray] = []
    for mag_label, data in traces.items():
        mag = float(mag_label.split("_")[1])
        t = np.asarray(data["time"], dtype=np.float64)
        v = np.asarray(data["vel_m_per_s"], dtype=np.float64)
        all_t.append(t)
        all_v.append(v)
        all_mag.append(np.full_like(t, mag))
    T = np.concatenate(all_t)
    V = np.concatenate(all_v)
    M = np.concatenate(all_mag)

    def model(X, K, tau):
        t_arr, u_arr = X
        return K * u_arr * (1.0 - np.exp(-t_arr / np.maximum(tau, 1e-6)))

    popt, pcov = curve_fit(model, (T, M), V, p0=[1.0, 0.005],
                           bounds=([0.0, 1e-4], [5.0, 1.0]))
    K, tau = popt
    # Goodness-of-fit
    V_pred = model((T, M), *popt)
    residuals = V - V_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((V - V.mean())**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return {"K_m_per_s_per_unit": float(K), "tau_s": float(tau),
            "rmse_m_per_s": rmse, "r_squared": float(r_squared),
            "cov": pcov.tolist()}


def main():
    if not T9_RESULTS.exists():
        raise SystemExit(f"missing {T9_RESULTS}; run task09 step_response.py first")
    data = json.loads(T9_RESULTS.read_text())["data"]

    model = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis_idx, label in enumerate(["x", "y", "z"]):
        axis_key = f"axis_{label}"
        traces = data[axis_key]
        fit = fit_axis(traces)
        model[axis_key] = fit
        print(f"axis {label}:  K={fit['K_m_per_s_per_unit']:.4f} m/s/unit  "
              f"tau={fit['tau_s']*1000:.3f} ms  RMSE={fit['rmse_m_per_s']*1000:.3f} mm/s  "
              f"R^2={fit['r_squared']:.5f}")
        ax = axes[axis_idx]
        K, tau = fit["K_m_per_s_per_unit"], fit["tau_s"]
        colors = ["#4C9AFF", "#FF6B6B", "#54B567"]
        for ci, (mag_label, trace) in enumerate(traces.items()):
            mag = float(mag_label.split("_")[1])
            t = np.asarray(trace["time"])
            v = np.asarray(trace["vel_m_per_s"])
            ax.plot(t * 1000, v, "o", markersize=3, color=colors[ci % len(colors)], alpha=0.5,
                    label=f"measured u={mag}")
            t_fit = np.linspace(t.min(), t.max(), 200)
            ax.plot(t_fit * 1000, first_order_step(t_fit, K, tau, mag),
                    "-", color=colors[ci % len(colors)], linewidth=1.5,
                    label=f"fit K={K:.2f},tau={tau*1000:.1f}ms u={mag}")
        ax.set_title(f"axis {label}", fontsize=11)
        ax.set_xlabel("t (ms)")
        if axis_idx == 0:
            ax.set_ylabel("velocity (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("System identification: first-order plant fit per axis\nv(t) = K * u_step * (1 - exp(-t/tau))")
    fig.tight_layout()
    out_png = RESULTS / "sysid_fit.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_png}")

    out_json = RESULTS / "plant_model.json"
    out_json.write_text(json.dumps({
        "model_form": "v(t) = K * u_step * (1 - exp(-t/tau))",
        "axes": model,
        "notes": [
            "Fit jointly over all step magnitudes per axis.",
            "v(t) is the velocity response in m/s, u_step is the action magnitude.",
            "K has units m/s per unit-action; tau in seconds.",
        ],
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
