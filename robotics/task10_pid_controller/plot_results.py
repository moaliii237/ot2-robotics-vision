"""Plots: phase 1 grid, phase 2 grid, robustness, demo error-vs-step.

Run after tune_pid, refine_pid, robustness_sweep, run_pid_demo.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
R = HERE / "results"


def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def _agg(rows):
    errs = [float(r["final_err_mm"]) for r in rows]
    succ = sum(1 for r in rows if r["settled"].lower() == "true") / len(rows)
    return {"median": float(np.median(errs)), "max": float(np.max(errs)),
            "min": float(np.min(errs)), "succ": succ}


def phase1():
    rows = read_csv(R / "tuning_log.csv")
    by = defaultdict(list)
    for r in rows:
        by[(float(r["kp_xy"]), float(r["ki_xy"]), float(r["kd_xy"]))].append(r)
    combos = sorted(by.keys())

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(combos))
    medians = [_agg(by[c])["median"] for c in combos]
    maxes = [_agg(by[c])["max"] for c in combos]
    labels = [f"Kp{c[0]:g}\nKi{c[1]:g}\nKd{c[2]:g}" for c in combos]
    ax.bar(x - 0.2, medians, width=0.4, label="median final err (mm)", color="#4C9AFF")
    ax.bar(x + 0.2, maxes, width=0.4, label="max final err (mm)", color="#FF6B6B")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1 mm spec")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("error (mm, log scale)")
    ax.set_title("Phase 1: 24-combo grid (Kp_xy x Ki_xy x Kd_xy)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = R / "tuning_phase1.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def phase2():
    rows = read_csv(R / "refine_log.csv")
    by = defaultdict(list)
    for r in rows:
        by[(float(r["kp"]), float(r["kff"]))].append(r)
    combos = sorted(by.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(combos))
    medians = [_agg(by[c])["median"] for c in combos]
    maxes = [_agg(by[c])["max"] for c in combos]
    labels = [f"Kp={c[0]:g}\nkff={c[1]:g}" for c in combos]
    ax.bar(x - 0.2, medians, width=0.4, label="median final err (mm)", color="#4C9AFF")
    ax.bar(x + 0.2, maxes, width=0.4, label="max final err (mm)", color="#FF6B6B")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("error (mm)")
    ax.set_title("Phase 2: refinement of Kp and feed-forward (Ki=0, Kd=0)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = R / "tuning_phase2.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def robustness():
    rows = read_csv(R / "robustness_log.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, seed in enumerate(seeds):
        errs = sorted([float(r["final_err_mm"]) for r in rows if int(r["seed"]) == seed])
        ax.plot(np.arange(len(errs)), errs, "o-", label=f"seed={seed}")
    ax.axhline(0.001, color="green", linestyle=":", label="1 um")
    ax.axhline(1.0, color="black", linestyle="--", label="1 mm spec")
    ax.set_yscale("log")
    ax.set_xlabel("target index (sorted by err)")
    ax.set_ylabel("final error (mm)")
    ax.set_title("Robustness: final error across 3 seeds, 10 targets each")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = R / "robustness.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def demo_trace():
    log = json.loads((R / "demo_log.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 5))
    by_t = defaultdict(list)
    for row in log["trajectory"]:
        by_t[row["target"]].append(row)
    for label, rows in by_t.items():
        steps = [r["step"] for r in rows]
        errs = [r["err_mm"] for r in rows]
        ax.plot(steps, errs, label=label)
    ax.axhline(1.0, color="black", linestyle="--", label="1 mm spec")
    ax.axhline(0.001, color="green", linestyle=":", label="1 um")
    ax.set_yscale("log")
    ax.set_xlabel("sim step (240 Hz)")
    ax.set_ylabel("error (mm)")
    ax.set_title("Demo: error vs step for each of 5 targets")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = R / "demo_trace.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    phase1(); phase2(); robustness(); demo_trace()
