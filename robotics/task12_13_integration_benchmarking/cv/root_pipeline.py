"""Root segmentation + tip extraction pipeline for OT-2 integration.

Adapted from Mohammadali Jaberi's Task 8 notebook (Block B retake, March 2026).
That notebook measured primary-root length; here the OT-2 needs to navigate to
each root tip, so the pipeline gains a tip-coordinate extractor.

Per image:

    1. Read the dish photo (grayscale).
    2. Pad to a multiple of 256, tile into 256x256 patches, run the U-Net on
       all of them in one batch, stitch the masks back together.
    3. Threshold at 0.5 -> binary mask.
    4. Morphological close (5x5 ellipse) to bridge thin root fragments.
    5. Connected components: keep the N_PLANTS (=5) largest blobs with
       area >= min_area, sorted left-to-right by x-centroid.
    6. Per plant: skeletonize, then find the root tip = farthest skeleton
       point from the topmost pixel (the shoot base), plus the path length.

detect_tips_in_image returns, per plant, the tip pixel, mask bbox, and
primary-root length, ready for calibration.py to turn into OT-2 world coords.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TF info logs

import heapq
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


# ---------- module-level constants (match the notebook) ----------

PATCH_SIZE = 256
THRESHOLD = 0.5
N_PLANTS = 5
MIN_AREA = 200          # connected-component noise filter (px)
MORPH_KERNEL = (5, 5)   # closing kernel for fragment bridging
# Drop CCs whose bbox aspect ratio is more extreme than this. A real seedling
# root is columnar but not absurdly so; the dish rim and edge shadows make
# 40-50:1 strips that would otherwise win "keep N largest". 10:1 clears the
# rim noise without dropping real plants on the Kaggle set.
MAX_BBOX_ASPECT = 10.0


# ---------- custom_objects for load_model ----------

def f1(y_true, y_pred):
    """Pixel-level F1 (= Dice for binary masks)."""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = tp / (pos + K.epsilon())
    precision = tp / (pp + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    intersection = K.sum(yt * yp)
    return 1 - (2.0 * intersection + smooth) / (K.sum(yt) + K.sum(yp) + smooth)


def bce_dice_loss(y_true, y_pred):
    weight = 1.0 + 2.0 * y_true
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, axis=-1)
    weighted_bce = tf.reduce_mean(weight[..., 0] * bce)
    return 0.5 * weighted_bce + 0.5 * dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {"f1": f1, "bce_dice_loss": bce_dice_loss}


# ---------- data classes ----------

@dataclass
class PlantDetection:
    """One detected plant in an image."""
    plant_index: int                # 0..N_PLANTS-1, left-to-right
    bbox: tuple[int, int, int, int] # (min_row, min_col, max_row, max_col), pixel
    centroid: tuple[float, float]   # (row, col) of CC centroid
    shoot_base: tuple[int, int]     # topmost skeleton pixel (row, col)
    tip: tuple[int, int]            # root-tip pixel (row, col)
    length_px: float                # primary-root length (BFS-diameter, px)
    mask_area_px: int               # pixel count of the plant mask
    primary_root_path: list[tuple[int, int]] = field(default_factory=list)
    # ^ (row, col) skeleton pixels from shoot_base -> tip: the primary root
    # we navigate to.

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "plant_index": self.plant_index,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "shoot_base": list(self.shoot_base),
            "tip": list(self.tip),
            "length_px": float(self.length_px),
            "mask_area_px": int(self.mask_area_px),
            "primary_root_path": [list(p) for p in self.primary_root_path],
        }


@dataclass
class ImageResult:
    image_path: str
    image_shape: tuple[int, int]                # (H, W) of original
    plants: list[PlantDetection] = field(default_factory=list)
    n_detected: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_shape": list(self.image_shape),
            "n_detected": self.n_detected,
            "plants": [p.to_jsonable() for p in self.plants],
        }


# ---------- model loader ----------

_loaded_model: tf.keras.Model | None = None
_loaded_model_path: str | None = None


def get_model(weights_path: str | Path) -> tf.keras.Model:
    """Lazy-load the U-Net once and cache for reuse across calls."""
    global _loaded_model, _loaded_model_path
    weights_path = str(Path(weights_path).resolve())
    if _loaded_model is None or _loaded_model_path != weights_path:
        _loaded_model = load_model(weights_path, custom_objects=CUSTOM_OBJECTS,
                                   compile=False)
        _loaded_model_path = weights_path
    return _loaded_model


# ---------- petri-dish ROI (Task 2 traditional CV) ----------

def get_dish_bbox(img: np.ndarray,
                  black_threshold: int = 80,
                  padding: int = 20) -> tuple[int, int, int, int]:
    """Find the petri-dish bounding box with traditional CV.

    Port of get_dish_bbox from the Task 5 inference template: threshold the
    dark border, close + erode to clean up, take the largest contour.

    Returns (x1, y1, x2, y2) crop coords in the image's pixel frame.
    """
    binary = (img > black_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contour found, fall back to the whole image.
        H, W = img.shape
        return 0, 0, W, H
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W = img.shape
    return (max(0, x - padding), max(0, y - padding),
            min(W, x + w + padding), min(H, y + h + padding))


# ---------- inference ----------

def predict_full_mask(model: tf.keras.Model, img_path: str | Path,
                      patch_size: int = PATCH_SIZE,
                      threshold: float = THRESHOLD,
                      batch_size: int = 16,
                      use_dish_crop: bool = True,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Tile, predict, stitch.

    With use_dish_crop=True (the Task 5 inference path), crop to the dish
    bbox, run the U-Net on the crop only, then drop the mask back into a
    full-image zero array so it lines up 1:1 with the input pixels.

    With False, predict on the full raw image (faster, noisier near the
    borders).

    Returns (img_orig_uint8_gray, binary_mask_uint8), both (H, W).
    """
    img_orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"could not read {img_path}")
    H, W = img_orig.shape

    if use_dish_crop:
        x1, y1, x2, y2 = get_dish_bbox(img_orig)
        roi = img_orig[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, W, H
        roi = img_orig

    roi_f = roi.astype(np.float32) / 255.0
    Hroi, Wroi = roi_f.shape
    pad_h = (patch_size - Hroi % patch_size) % patch_size
    pad_w = (patch_size - Wroi % patch_size) % patch_size
    roi_pad = np.pad(roi_f, ((0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = roi_pad.shape

    patches, coords = [], []
    for r in range(0, Hp, patch_size):
        for c in range(0, Wp, patch_size):
            patches.append(roi_pad[r:r + patch_size, c:c + patch_size, None])
            coords.append((r, c))
    patches = np.asarray(patches, dtype=np.float32)
    preds = model.predict(patches, batch_size=batch_size, verbose=0)

    pred_pad = np.zeros((Hp, Wp), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        pred_pad[r:r + patch_size, c:c + patch_size] = preds[i, :, :, 0]
    pred_roi = pred_pad[:Hroi, :Wroi]
    binary_roi = (pred_roi > threshold).astype(np.uint8)

    # Back into raw-image coords so detected pixels are in the original frame.
    binary = np.zeros((H, W), dtype=np.uint8)
    binary[y1:y2, x1:x2] = binary_roi
    return img_orig, binary


# ---------- plant detection ----------

def detect_plants(binary_mask: np.ndarray,
                  n_plants: int = N_PLANTS,
                  min_area: int = MIN_AREA,
                  max_aspect_ratio: float = MAX_BBOX_ASPECT,
                  ) -> tuple[list[Any], np.ndarray]:
    """Connected components, filter, keep N largest, sort left-to-right.

    Filters in order:
      1. min_area: drop noise.
      2. max_aspect_ratio: drop dish-rim strips. Without it a thin 40px-wide
         strip spanning the dish height can outvote a real plant in "keep N
         largest".
      3. keep N largest by area.
      4. sort left-to-right by x-centroid for stable per-image labels.

    Returns (list[regionprops], labeled_array). Both may be empty.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    labeled = label(cleaned, connectivity=2)
    candidates = []
    for prop in regionprops(labeled):
        if prop.area < min_area:
            continue
        r1, c1, r2, c2 = prop.bbox
        h, w = max(r2 - r1, 1), max(c2 - c1, 1)
        aspect = max(h, w) / min(h, w)
        if aspect > max_aspect_ratio:
            # Likely dish rim / shadow strip, not a plant.
            continue
        candidates.append(prop)
    if not candidates:
        return [], labeled
    if len(candidates) > n_plants:
        candidates = sorted(candidates, key=lambda p: p.area, reverse=True)[:n_plants]
    candidates.sort(key=lambda p: p.centroid[1])  # x-centroid, left-to-right
    return candidates, labeled


# ---------- root-tip extraction (the new bit) ----------

def _bfs_with_parents(start: tuple[int, int],
                      pixel_set: set[tuple[int, int]]
                      ) -> tuple[tuple[int, int], float,
                                 dict[tuple[int, int], tuple[int, int] | None]]:
    """Dijkstra on the 8-neighbour skeleton graph, with parent pointers.

    Returns (farthest_pixel, geodesic_distance_px, parents_map). The parents
    map rebuilds the shoot->tip primary-root path.

    Edges aren't uniform (orthogonal = 1.0, diagonal = sqrt(2)), so a plain
    FIFO BFS wouldn't give true shortest paths once the skeleton has a cycle:
    it locks a node's cost on first discovery, which can over-estimate. The
    skeletons here are tree-like so BFS would usually agree, but Dijkstra is
    correct for any topology at negligible cost, so tip selection and length_px
    are proper geodesics either way.
    """
    dist: dict[tuple[int, int], float] = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue  # stale priority-queue entry
        r, c = node
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_set:
                    step = 1.4142135623730951 if (dr and dc) else 1.0
                    nd = d + step
                    if nd < dist.get(nb, float("inf")):
                        dist[nb] = nd
                        parents[nb] = node
                        heapq.heappush(pq, (nd, nb))
    farthest = max(dist, key=dist.get)
    return farthest, dist[farthest], parents


def _bfs_farthest(start: tuple[int, int],
                  pixel_set: set[tuple[int, int]]
                  ) -> tuple[tuple[int, int], float]:
    """Thin wrapper that drops the parents map for the original use case."""
    far, dist, _ = _bfs_with_parents(start, pixel_set)
    return far, dist


def extract_root_tip(plant_mask: np.ndarray
                     ) -> tuple[tuple[int, int] | None, tuple[int, int] | None,
                                 float, list[tuple[int, int]]]:
    """Skeletonize a plant mask and find (shoot_base, root_tip, length, path).

    shoot_base = topmost skeleton pixel (smallest row).
    root_tip   = farthest skeleton pixel from the shoot base by 8-connected
                 Dijkstra, in geodesic distance (diagonals = sqrt(2)).
    path       = (row, col) skeleton pixels from shoot_base -> tip, rebuilt
                 from parent pointers. This is the primary root the OT-2
                 navigates to.

    Returns ((sr, sc), (tr, tc), length_px, path), or (None, None, 0.0, [])
    on a degenerate mask. The None signals "no tip" so callers skip the plant.
    A sentinel like (0,0) would be a real top-left pixel and get calibrated
    and driven to as if it were a tip.
    """
    if int(plant_mask.sum()) < 10:
        return None, None, 0.0, []
    skel = skeletonize(plant_mask > 0)
    if not skel.any():
        return None, None, 0.0, []
    rows, cols = np.where(skel)
    pixels = set(zip(rows.tolist(), cols.tolist()))
    # shoot base = topmost (smallest row) skeleton pixel
    shoot_base = min(pixels, key=lambda p: p[0])
    # Path from the shoot base to the farthest skeleton pixel, i.e. the
    # longest chain starting at the shoot. (A double-BFS diameter would find
    # the longest path anywhere in the skeleton, which could run between two
    # side branches and skip the shoot. We specifically want the shoot end.)
    tip, length, parents = _bfs_with_parents(shoot_base, pixels)
    # Reconstruct path tip -> shoot, then reverse.
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = tip
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return shoot_base, tip, float(length), path


# ---------- top-level orchestrator ----------

def detect_tips_in_image(weights_path: str | Path,
                         img_path: str | Path,
                         n_plants: int = N_PLANTS,
                         min_area: int = MIN_AREA,
                         threshold: float = THRESHOLD,
                         ) -> tuple[ImageResult, np.ndarray, np.ndarray]:
    """End-to-end: image path -> [PlantDetection,...].

    Returns (ImageResult, original_grayscale, binary_mask).
    The two arrays are returned so callers can render visualisations
    without re-running the segmentation.
    """
    model = get_model(weights_path)
    img_orig, binary = predict_full_mask(model, img_path, threshold=threshold)
    props, labeled = detect_plants(binary, n_plants=n_plants, min_area=min_area)

    result = ImageResult(image_path=str(img_path),
                         image_shape=img_orig.shape,
                         n_detected=0)
    for prop in props:
        plant_mask = (labeled == prop.label).astype(np.uint8)
        shoot, tip, length, path = extract_root_tip(plant_mask)
        if tip is None or shoot is None:
            # Degenerate mask, no usable tip. Skip rather than emit a (0,0)
            # target that would get driven to and counted as a real dispense.
            continue
        result.plants.append(PlantDetection(
            plant_index=len(result.plants),
            bbox=tuple(int(v) for v in prop.bbox),
            centroid=(float(prop.centroid[0]), float(prop.centroid[1])),
            shoot_base=(int(shoot[0]), int(shoot[1])),
            tip=(int(tip[0]), int(tip[1])),
            length_px=length,
            mask_area_px=int(prop.area),
            primary_root_path=[(int(r), int(c)) for r, c in path],
        ))
    result.n_detected = len(result.plants)
    return result, img_orig, binary


# ---------- CLI ----------

def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str,
                    default=str(Path(__file__).resolve().parent / "weights" / "unet_256px.keras"))
    ap.add_argument("--images", type=str, nargs="+", required=True,
                    help="image path(s) or a glob")
    ap.add_argument("--out_json", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "results" / "cv_detections.json"))
    ap.add_argument("--n_plants", type=int, default=N_PLANTS)
    ap.add_argument("--min_area", type=int, default=MIN_AREA)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    # Expand globs in args.images
    import glob
    paths: list[str] = []
    for pat in args.images:
        if any(ch in pat for ch in "*?["):
            paths.extend(sorted(glob.glob(pat)))
        else:
            paths.append(pat)
    if not paths:
        raise SystemExit("no images matched")

    all_results: list[dict[str, Any]] = []
    for ip in paths:
        print(f"[ cv ] {ip}")
        result, _img, _msk = detect_tips_in_image(
            args.weights, ip,
            n_plants=args.n_plants, min_area=args.min_area, threshold=args.threshold,
        )
        for p in result.plants:
            print(f"   plant {p.plant_index+1}: tip=({p.tip[0]:4d}, {p.tip[1]:4d}) "
                  f"length={p.length_px:7.1f} px  area={p.mask_area_px:6d} px")
        all_results.append(result.to_jsonable())

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "weights": str(Path(args.weights).resolve()),
        "args": vars(args),
        "results": all_results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    _cli()
