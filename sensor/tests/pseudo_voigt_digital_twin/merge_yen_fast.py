"""
Fast merge for the offset-tuned Yen filament extractor.

This avoids expanding each cropped laser mask back to a full camera-sized mask.
Instead, it pastes the cropped mask at the correct offset inside the camera
footprint on the merged world canvas.
"""

import os
import re

import cv2
import numpy as np

from filament_yen_fast import (
    DEFAULT_EMPTY_I,
    DEFAULT_FOLDER,
    DEFAULT_FULL_I,
    IMG_EXTS,
    PX_PER_MM,
    extract_filament_array,
    list_photos,
    make_fit_cache,
    undo_preprocess_mask,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


def number(name):
    m = re.search(r"frame_(\d+)", str(name)) or re.search(r"\d+", str(name))
    return int(m.group(1)) if m else -1


def parse_pose(name):
    stem = os.path.splitext(name)[0]
    m = re.search(r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$", stem)
    if not m:
        return None
    frame, t, x, y, z = m.groups()
    return {"frame": int(frame), "t": float(t), "x": float(x), "y": float(y), "z": float(z)}


def _clip_rect(dst_shape, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dst_shape[1], x1), min(dst_shape[0], y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return None
    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    return dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1


def paste_bool(canvas, mask, x0, y0, mode="or"):
    h, w = mask.shape
    r = _clip_rect(canvas.shape, int(x0), int(y0), w, h)
    if r is None:
        return
    dx0, dy0, dx1, dy1, sx0, sy0, sx1, sy1 = r
    src = mask[sy0:sy1, sx0:sx1]
    if mode == "or":
        canvas[dy0:dy1, dx0:dx1] |= src
    else:
        canvas[dy0:dy1, dx0:dx1] = src


def paste_camera_rect(canvas, image_shape, cx, cy):
    h, w = image_shape[:2]
    x0 = int(round(cx)) - w // 2
    y0 = int(round(cy)) - h // 2
    r = _clip_rect(canvas.shape, x0, y0, w, h)
    if r is None:
        return
    dx0, dy0, dx1, dy1, *_ = r
    canvas[dy0:dy1, dx0:dx1] = True


def paste_cropped_camera_mask(canvas, crop_mask, crop_info, cx, cy):
    """Paste a cropped output-image mask without allocating the full camera mask."""
    H, W = crop_info["original_shape"]
    y0, _, x0, _ = crop_info["constant_crop_bounds"]
    dst_x0 = int(round(cx)) - W // 2 + int(x0)
    dst_y0 = int(round(cy)) - H // 2 + int(y0)
    paste_bool(canvas, crop_mask.astype(bool, copy=False), dst_x0, dst_y0, mode="or")


def _frame_list(photos, empty_i, full_i, skip_refs=True, flip_x=False, flip_y=True, swap_xy=False):
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
    return frames


def merge_no_filament_folder(
    folder=DEFAULT_FOLDER,
    empty_i=DEFAULT_EMPTY_I,
    full_i=DEFAULT_FULL_I,
    radius=25,
    threshold="yen",
    px_per_mm=PX_PER_MM,
    margin_px=200,
    degree=26.2,
    flip_x=False,
    flip_y=True,
    swap_xy=False,
    skip_refs=True,
    out_path="merged_no_filament.png",
    fit_cache=None,
    max_frames=None,
    return_cache=False,
    **extract_kwargs,
):
    del degree  # kept for old-call compatibility; geometry is handled by filament_yen_fast.
    folder = os.path.abspath(os.path.join(str(folder), ""))
    fit_cache = make_fit_cache() if fit_cache is None else fit_cache

    photos = list_photos(folder, fit_cache)
    frames = _frame_list(
        photos,
        empty_i=empty_i,
        full_i=full_i,
        skip_refs=skip_refs,
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
    )
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    if not frames:
        raise ValueError("No valid frame filenames found.")

    sample_i, _, _, _ = frames[0]
    sample_mask, sample_pack = extract_filament_array(
        folder=folder,
        photos=photos,
        empty_i=empty_i,
        full_i=full_i,
        img_i=sample_i,
        radius=radius,
        threshold=threshold,
        fit_cache=fit_cache,
        **extract_kwargs,
    )
    img_h, img_w = sample_pack["original_shape"]

    xs = np.array([f[2] for f in frames], dtype=float)
    ys = np.array([f[3] for f in frames], dtype=float)

    min_x = int(np.floor(xs.min() * px_per_mm - img_w - margin_px))
    max_x = int(np.ceil(xs.max() * px_per_mm + img_w + margin_px))
    min_y = int(np.floor(ys.min() * px_per_mm - img_h - margin_px))
    max_y = int(np.ceil(ys.max() * px_per_mm + img_h + margin_px))

    W, H = max_x - min_x, max_y - min_y
    no_filament_canvas = np.zeros((H, W), dtype=bool)
    covered_canvas = np.zeros((H, W), dtype=bool)

    for i, _name, x, y in tqdm(frames, desc="Merging frames"):
        cx = int(round(x * px_per_mm - min_x))
        cy = int(round(y * px_per_mm - min_y))

        if i == sample_i:
            mask, pack = sample_mask, sample_pack
        else:
            mask, pack = extract_filament_array(
                folder=folder,
                photos=photos,
                empty_i=empty_i,
                full_i=full_i,
                img_i=i,
                radius=radius,
                threshold=threshold,
                fit_cache=fit_cache,
                **extract_kwargs,
            )

        paste_cropped_camera_mask(no_filament_canvas, mask, pack, cx, cy)
        paste_camera_rect(covered_canvas, pack["original_shape"], cx, cy)

    img = np.full((H, W), 127, dtype=np.uint8)
    img[covered_canvas] = 0
    img[no_filament_canvas] = 255
    cv2.imwrite(out_path, img)

    out = {
        "image": img,
        "no_filament_mask": no_filament_canvas,
        "covered_mask": covered_canvas,
        "out_path": out_path,
        "px_per_mm": px_per_mm,
        "bounds": (min_x, max_x, min_y, max_y),
        "frame_count": len(frames),
    }
    if return_cache:
        out["fit_cache"] = fit_cache
    return out


if __name__ == "__main__":
    out = merge_no_filament_folder(radius=35, px_per_mm=PX_PER_MM, return_cache=True)
    print("wrote", out["out_path"], "frames", out["frame_count"])
    print("cache buckets", sorted(out["fit_cache"].keys()))
