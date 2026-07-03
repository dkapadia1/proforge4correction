import os, re
import numpy as np
import cv2
from filament_array import extract_filament_array, path_to_grad, undo_preprocess_mask
from tqdm import tqdm
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
def paste_bool(canvas, mask, cx, cy, mode="or"):
    h, w = mask.shape
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h

    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)

    if dx0 >= dx1 or dy0 >= dy1:
        return

    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    if mode == "or":
        canvas[dy0:dy1, dx0:dx1] |= mask[sy0:sy1, sx0:sx1]
    else:
        canvas[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
def resize_mask(mask, sx=1.0, sy=1.0):
    h, w = mask.shape
    nw = max(1, int(round(w * sx)))
    nh = max(1, int(round(h * sy)))
    return cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
def merge_no_filament_folder(
    folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\",
    empty_i=2418-149,
    full_i=1946-149,
    radius=25,
    threshold=0.15,
    px_per_mm=10.0,
    margin_px=200,
    degree=26.2,
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

    empty_path = os.path.join(folder, photos[empty_i])
    full_path = os.path.join(folder, photos[full_i])
    empty_grad = path_to_grad(empty_path, radius=radius, degree=degree)
    full_grad = path_to_grad(full_path, radius=radius, degree=degree)

    frames = []
    for i, name in enumerate(photos):
        if skip_refs and i in {empty_i, full_i}:
            continue

        pose = parse_pose(name)
        if pose is None:
            continue

        x, y = pose["x"], pose["y"]

        if swap_xy:
            x, y = y, x
        if flip_x:
            x = -x
        if flip_y:
            y = -y

        frames.append((i, name, x, y))

    if not frames:
        raise ValueError("No valid frame filenames found.")

    sample_i, sample_name, _, _ = frames[0]
    sample_mask, _ = extract_filament_array(folder = folder,
        img_i=0,
        grad = empty_grad,
        full_grad = full_grad,
        radius=radius,
        threshold=threshold,)

    mask_h, mask_w = sample_mask.shape

    xs = np.array([f[2] for f in frames])
    ys = np.array([f[3] for f in frames])

    min_x = int(np.floor(xs.min() * px_per_mm - mask_w - margin_px))
    max_x = int(np.ceil(xs.max() * px_per_mm + mask_w + margin_px))
    min_y = int(np.floor(ys.min() * px_per_mm - mask_h - margin_px))
    max_y = int(np.ceil(ys.max() * px_per_mm + mask_h + margin_px))

    W, H = max_x - min_x, max_y - min_y

    no_filament_canvas = np.zeros((H, W), dtype=bool)
    covered_canvas = np.zeros((H, W), dtype=bool)

    for i, name, x, y in tqdm(frames, desc="Merging frames"):
        if i == sample_i:
            mask = sample_mask
        else:
            mask, preprocess_pack = extract_filament_array(
                folder = folder,
                img_i=i,
                grad = empty_grad,
                full_grad = full_grad,
                radius=radius,
                threshold=threshold,
                )
            mask = undo_preprocess_mask(mask, preprocess_pack, radius=radius)
        cx = int(round(x * px_per_mm - min_x))
        cy = int(round(y * px_per_mm - min_y))
        debug = np.zeros_like(mask, dtype=bool)
        debug[debug.shape[0]//2, debug.shape[1]//2] = 1
        paste_bool(no_filament_canvas, mask, cx, cy, mode="or")
        paste_bool(covered_canvas, np.ones_like(mask, dtype=bool), cx, cy, mode="or")

    img = np.full((H, W), 127, dtype=np.uint8)

    # covered but not marked no-filament = assume filament
    img[covered_canvas] = 0

    # any positive model result wins
    img[no_filament_canvas] = 255

    cv2.imwrite(out_path, img)

    return {
        "image": img,
        "no_filament_mask": no_filament_canvas,
        "covered_mask": covered_canvas,
        "out_path": out_path,
        "px_per_mm": px_per_mm,
    }
if __name__ == "__main__":
    out = merge_no_filament_folder(radius=35, px_per_mm=26.13)
    
