"""Annotated debug PNGs from the CV pipeline output.

Per image:
  - left:  original grayscale
  - right: original + red mask overlay + per-plant bbox, tip marker, length

For eyeballing the segmentation before wiring it to the OT-2 controllers.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Add this dir to sys.path so root_pipeline imports when run as a script.
# Can't use `from cv.root_pipeline import ...` because the cv package name
# collides with the OpenCV module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from root_pipeline import detect_tips_in_image


PLANT_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#4361ee", "#f77f00"]


def render_one(weights_path: str, img_path: str, out_path: str) -> None:
    result, img_orig, binary = detect_tips_in_image(weights_path, img_path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title(Path(img_path).name)
    axes[0].axis("off")

    overlay = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
    overlay[binary > 0] = [220, 80, 80]
    axes[1].imshow(overlay)

    for p in result.plants:
        color = PLANT_COLORS[p.plant_index % len(PLANT_COLORS)]
        minr, minc, maxr, maxc = p.bbox
        axes[1].add_patch(mpatches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=2, edgecolor=color, facecolor="none"))
        # shoot base (^, top) and tip (v, bottom)
        axes[1].plot(p.shoot_base[1], p.shoot_base[0], marker="^",
                     color=color, markersize=10, markeredgecolor="black")
        axes[1].plot(p.tip[1], p.tip[0], marker="v",
                     color=color, markersize=12, markeredgecolor="black")
        axes[1].text(
            minc, max(minr - 12, 14),
            f"P{p.plant_index + 1}  tip=({p.tip[0]}, {p.tip[1]})  "
            f"len={p.length_px:.0f}px",
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, pad=2, edgecolor="none"))

    axes[1].set_title(
        f"{result.n_detected} plants detected | "
        f"^ = shoot base | v = root tip (OT-2 dispense target)")
    axes[1].axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str,
                    default=str(here / "weights" / "unet_256px.keras"))
    ap.add_argument("--images", type=str, nargs="+", required=True,
                    help="image path(s) or glob")
    ap.add_argument("--out_dir", type=str,
                    default=str(here.parent / "results" / "cv_viz"))
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.images:
        if any(c in pat for c in "*?["):
            paths.extend(sorted(glob.glob(pat)))
        else:
            paths.append(pat)
    if not paths:
        raise SystemExit("no images matched")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ip in paths:
        out_path = out_dir / (Path(ip).stem + "_viz.png")
        render_one(args.weights, ip, str(out_path))


if __name__ == "__main__":
    main()
