import os, re
import numpy as np
import cv2
from sensor.filament_array import extract_filament_array
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def number(name):
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else -1

def parse_pose(name):
    stem = os.path.splitext(name)[0]

    m = re.search(
        r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$",
        stem
    )
    if not m:
        return None

    frame, t, x, y, z = m.groups()
    return {
        "frame": int(frame),
        "t": float(t),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }

def paste_mask(no_sum, seen_sum, mask, cx, cy):
    h, w = mask.shape
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h

    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(no_sum.shape[1], x1), min(no_sum.shape[0], y1)

    if dx0 >= dx1 or dy0 >= dy1:
        return

    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    no_sum[dy0:dy1, dx0:dx1] += mask[sy0:sy1, sx0:sx1]
    seen_sum[dy0:dy1, dx0:dx1] += 1

def merge_no_filament_folder(
    folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\",
    empty_i=2418-149,
    full_i=1946-149,
    radius=25,
    threshold=0.07,
    px_per_mm=10.0,
    vote_threshold=0.5,
    min_votes=1,
    margin_px=100,
    flip_x=False,
    flip_y=True,
    swap_xy=False,
    skip_refs=True,
    out_path="merged_no_filament.png",
):
    folder = os.path.join(folder, "")
    photos = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS],
        key=number
    )

    items = []
    for i, name in enumerate(photos):
        if skip_refs and i in {empty_i, full_i}:
            continue

        pose = parse_pose(name)
        if pose is None:
            continue

        mask = extract_filament_array(
            folder=folder,
            empty_i=empty_i,
            full_i=full_i,
            img_i=i,
            radius=radius,
            threshold=threshold,
        ).astype(np.uint8)

        x, y = pose["x"], pose["y"]
        if swap_xy:
            x, y = y, x
        if flip_x:
            x = -x
        if flip_y:
            y = -y

        items.append((x, y, mask, name))

        if len(items) % 50 == 0:
            print("processed", len(items), "images")

    if not items:
        raise ValueError("No valid images found.")

    xs = np.array([it[0] for it in items])
    ys = np.array([it[1] for it in items])
    max_h = max(it[2].shape[0] for it in items)
    max_w = max(it[2].shape[1] for it in items)

    min_x = int(np.floor(xs.min() * px_per_mm - max_w / 2 - margin_px))
    max_x = int(np.ceil (xs.max() * px_per_mm + max_w / 2 + margin_px))
    min_y = int(np.floor(ys.min() * px_per_mm - max_h / 2 - margin_px))
    max_y = int(np.ceil (ys.max() * px_per_mm + max_h / 2 + margin_px))

    W, H = max_x - min_x, max_y - min_y
    no_sum = np.zeros((H, W), dtype=np.float32)
    seen_sum = np.zeros((H, W), dtype=np.float32)

    for x, y, mask, name in items:
        cx = int(round(x * px_per_mm - min_x))
        cy = int(round(y * px_per_mm - min_y))
        paste_mask(no_sum, seen_sum, mask, cx, cy)

    prob_no_filament = no_sum / np.maximum(seen_sum, 1)
    covered = seen_sum >= min_votes
    merged_mask = covered & (prob_no_filament >= vote_threshold)

    img = np.full((H, W), 127, dtype=np.uint8)
    img[covered & ~merged_mask] = 0
    img[merged_mask] = 255

    cv2.imwrite(out_path, img)

    return {
        "image": img,
        "prob_no_filament": prob_no_filament,
        "seen_count": seen_sum,
        "merged_mask": merged_mask,
        "out_path": out_path,
        "px_per_mm": px_per_mm,
    }
if __name__ == "__main__":
    merge_no_filament_folder()