import os
import re

import cv2
import numpy as np
from tqdm import tqdm

from filament_array import extract_filament_array, undo_preprocess_mask
from gcode_coordinate_mask import make_gcode_coordinate_mask, transform_xy

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def number(name):
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else -1


def parse_pose(name):
    stem = os.path.splitext(name)[0]
    m = re.search(r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$", stem)
    if not m:
        return None
    frame, t, x, y, z = m.groups()
    return {"frame": int(frame), "t": float(t), "x": float(x), "y": float(y), "z": float(z)}


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
    dst = np.s_[dy0:dy1, dx0:dx1]
    src = mask[sy0:sy1, sx0:sx1]

    if mode == "or":
        canvas[dst] |= src
    else:
        canvas[dst] = src


def read_no_filament_mask(folder, img_i, empty_grad, full_grad, radius, threshold):
    mask, preprocess_pack = extract_filament_array(
        folder=folder,
        img_i=img_i,
        grad=empty_grad,
        full_grad=full_grad,
        radius=radius,
        threshold=threshold,
    )
    return undo_preprocess_mask(mask, preprocess_pack, radius=radius)


def merge_no_filament_and_miss_print_coord(
    folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\",
    gcode_file="P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode",
    empty_i=2418 - 149,
    full_i=1946 - 149,
    radius=25,
    threshold=0.15,
    px_per_mm=26.13,
    margin_px=200,
    degree=26.2,
    flip_x=False,
    flip_y=True,
    swap_xy=False,
    skip_refs=True,
    layer_index=0,
    include_previous_layers=False,
    z_tol=0.08,
    gcode_theta_deg=0.0,
    gcode_line_width_mm=0.45,
    merged_out_path="merged_no_filament.png",
    miss_out_path="miss_print.png",
    expected_out_path=None,
):
    """
    Merge reader masks and compare to one coordinate-based expected G-code mask.

    Coordinate mapping used by both arrays:
      transformed_x, transformed_y = transform_xy(raw_x, raw_y, flip_x, flip_y, swap_xy)
      col = round((transformed_x - origin_x_mm) * px_per_mm)
      row = round((transformed_y - origin_y_mm) * px_per_mm)

    miss_print is only:
      expected filament AND camera covered AND reader said NO filament
    """
    folder = os.path.join(folder, "")
    photos = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS],
        key=number,
    )

    # Placeholders kept because extract_filament_array accepts these inputs in your current setup.
    empty_path = os.path.join(folder, photos[empty_i])
    full_path = os.path.join(folder, photos[full_i])
    empty_grad = None  # path_to_grad(empty_path, radius=radius, degree=degree)
    full_grad = None   # path_to_grad(full_path, radius=radius, degree=degree)
    _ = empty_path, full_path, degree

    frames = []
    for i, name in enumerate(photos):
        if skip_refs and i in {empty_i, full_i}:
            continue
        pose = parse_pose(name)
        if pose is None:
            continue
        x, y = transform_xy(pose["x"], pose["y"], flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
        frames.append((i, name, pose, x, y))

    if not frames:
        raise ValueError("No valid frame filenames found.")

    sample_mask = read_no_filament_mask(folder, frames[0][0], empty_grad, full_grad, radius, threshold)
    mask_h, mask_w = sample_mask.shape

    xs = np.array([f[3] for f in frames])
    ys = np.array([f[4] for f in frames])

    min_x_px = int(np.floor(xs.min() * px_per_mm - mask_w - margin_px))
    max_x_px = int(np.ceil(xs.max() * px_per_mm + mask_w + margin_px))
    min_y_px = int(np.floor(ys.min() * px_per_mm - mask_h - margin_px))
    max_y_px = int(np.ceil(ys.max() * px_per_mm + mask_h + margin_px))
    W, H = max_x_px - min_x_px, max_y_px - min_y_px

    # This is the important part: the merge canvas has a real coordinate origin.
    origin_xy_mm = (min_x_px / px_per_mm, min_y_px / px_per_mm)

    no_filament_canvas = np.zeros((H, W), dtype=bool)
    covered_canvas = np.zeros((H, W), dtype=bool)

    for n, (i, name, pose, x, y) in enumerate(tqdm(frames, desc="Merging no-filament masks")):
        mask = sample_mask if n == 0 else read_no_filament_mask(folder, i, empty_grad, full_grad, radius, threshold)
        cx = int(round((x - origin_xy_mm[0]) * px_per_mm))
        cy = int(round((y - origin_xy_mm[1]) * px_per_mm))
        paste_bool(no_filament_canvas, mask, cx, cy, mode="or")
        paste_bool(covered_canvas, np.ones_like(mask, dtype=bool), cx, cy, mode="or")

    merged_img = np.full((H, W), 127, dtype=np.uint8)
    merged_img[covered_canvas] = 255
    merged_img[no_filament_canvas] = 0
    cv2.imwrite(merged_out_path, merged_img)

    expected_coord_mask, expected_info = make_gcode_coordinate_mask(
        gcode_file=gcode_file,
        px_per_mm=px_per_mm,
        layer_index=layer_index,
        origin_xy_mm=origin_xy_mm,
        mask_shape=(H, W),
        theta_deg=gcode_theta_deg,
        line_width_mm=gcode_line_width_mm,
        include_previous_layers=include_previous_layers,
        z_tol=z_tol,
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
        return_info=True,
    )

    expected_canvas = expected_coord_mask.mask
    miss_canvas = expected_canvas & covered_canvas & no_filament_canvas
    cv2.imwrite(miss_out_path, miss_canvas.astype(np.uint8) * 255)

    if expected_out_path is not None:
        cv2.imwrite(expected_out_path, expected_canvas.astype(np.uint8) * 255)

    return {
        "merged_image": merged_img,
        "no_filament_mask": no_filament_canvas,
        "covered_mask": covered_canvas,
        "expected": expected_coord_mask,
        "expected_filament_mask": expected_canvas,
        "miss_print_mask": miss_canvas,
        "merged_out_path": merged_out_path,
        "miss_out_path": miss_out_path,
        "expected_out_path": expected_out_path,
        "px_per_mm": px_per_mm,
        "origin_xy_mm": origin_xy_mm,
        "min_x_px": min_x_px,
        "min_y_px": min_y_px,
        "target_z": expected_info["target_z"],
        "layer_zs": expected_info["layer_zs"],
        "unsupported_gcode_arcs": expected_info["unsupported_arcs"],
    }


if __name__ == "__main__":
    out = merge_no_filament_and_miss_print_coord(
        radius=35,
        px_per_mm=26.13,
        gcode_file="P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode",
        layer_index=0,
        merged_out_path="merged_no_filament.png",
        miss_out_path="miss_print.png",
        expected_out_path="expected_filament.png",
    )
    print({k: out[k] for k in ["merged_out_path", "miss_out_path", "expected_out_path", "origin_xy_mm", "target_z"]})
