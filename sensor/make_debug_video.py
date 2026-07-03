import os
import re
import cv2
import numpy as np
from tqdm import tqdm

from filament_array import (
    extract_filament_array,
    path_to_grad,
    undo_preprocess_mask,
    number,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fit_to_box(img, box_w, box_h):
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.full((box_h, box_w, 3), 40, dtype=np.uint8)
    y0 = (box_h - nh) // 2
    x0 = (box_w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def parse_pose(name):
    stem = os.path.splitext(name)[0]
    m = re.search(r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$", stem)
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


def paste_bool(canvas, mask, cx, cy):
    h, w = mask.shape
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h

    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return

    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    canvas[dy0:dy1, dx0:dx1] |= mask[sy0:sy1, sx0:sx1]


def merged_vis(no_filament_canvas, covered_canvas, cx=None, cy=None):
    img = np.full(no_filament_canvas.shape, 127, dtype=np.uint8)
    img[covered_canvas] = 0
    img[no_filament_canvas] = 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if cx is not None and cy is not None:
        cv2.drawMarker(
            img, (cx, cy), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
        )
    return img


def make_debug_video(
    folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\",
    empty_i=2420 - 149,
    full_i=1948 - 149,
    radius=25,
    threshold=0.25,
    degree=26.2,
    out_path="debug_original_mask_merged.mp4",
    fps=12,
    panel_w=640,
    panel_h=480,
    max_frames=None,
    px_per_mm=10.0,
    margin_px=200,
    flip_x=False,
    flip_y=True,
    swap_xy=False,
    skip_refs=True,
):
    folder = os.path.join(folder, "")
    all_files = sorted(os.listdir(folder), key=number)

    empty_path = os.path.join(folder, all_files[empty_i])
    full_path = os.path.join(folder, all_files[full_i])
    empty_grad = path_to_grad(empty_path, radius=radius, degree=degree)
    full_grad = path_to_grad(full_path, radius=radius, degree=degree)

    valid = []
    for i, name in enumerate(all_files):
        if os.path.splitext(name)[1].lower() not in IMG_EXTS:
            continue
        if skip_refs and i in {empty_i, full_i}:
            continue
        pose = parse_pose(name)
        if pose is None:
            continue
        valid.append((i, name, pose))

    if max_frames is not None:
        valid = valid[:max_frames]
    if not valid:
        raise ValueError("No valid image frames with parseable poses found.")

    sample_i = valid[0][0]
    sample_mask, sample_info = extract_filament_array(
        folder=folder,
        img_i=sample_i,
        grad=empty_grad,
        full_grad=full_grad,
        radius=radius,
        threshold=threshold,
    )
    sample_mask = undo_preprocess_mask(sample_mask, sample_info, radius=radius)
    mask_h, mask_w = sample_mask.shape

    xs, ys = [], []
    for _, _, pose in valid:
        x, y = pose["x"], pose["y"]
        if swap_xy:
            x, y = y, x
        if flip_x:
            x = -x
        if flip_y:
            y = -y
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    min_x = int(np.floor(xs.min() * px_per_mm - mask_w - margin_px))
    max_x = int(np.ceil(xs.max() * px_per_mm + mask_w + margin_px))
    min_y = int(np.floor(ys.min() * px_per_mm - mask_h - margin_px))
    max_y = int(np.ceil(ys.max() * px_per_mm + mask_h + margin_px))
    W, H = max_x - min_x, max_y - min_y

    no_filament_canvas = np.zeros((H, W), dtype=bool)
    covered_canvas = np.zeros((H, W), dtype=bool)

    title_h = 55
    frame_w = panel_w * 3
    frame_h = panel_h + title_h
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    for i, name, pose in tqdm(valid, desc="Writing debug video"):
        path = os.path.join(folder, name)
        orig = cv2.imread(path, cv2.IMREAD_COLOR)
        if orig is None:
            continue

        crop_mask, info = extract_filament_array(
            folder=folder,
            img_i=i,
            grad=empty_grad,
            full_grad=full_grad,
            radius=radius,
            threshold=threshold,
        )
        unrot_mask = undo_preprocess_mask(crop_mask, info, radius=radius)

        x, y = pose["x"], pose["y"]
        if swap_xy:
            x, y = y, x
        if flip_x:
            x = -x
        if flip_y:
            y = -y

        cx = int(round(x * px_per_mm - min_x))
        cy = int(round(y * px_per_mm - min_y))

        paste_bool(no_filament_canvas, unrot_mask, cx, cy)
        paste_bool(covered_canvas, np.ones_like(unrot_mask, dtype=bool), cx, cy)

        mask_img = cv2.cvtColor((unrot_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        merged_img = merged_vis(no_filament_canvas, covered_canvas, cx, cy)

        left = fit_to_box(orig, panel_w, panel_h)
        mid = fit_to_box(mask_img, panel_w, panel_h)
        right = fit_to_box(merged_img, panel_w, panel_h)

        frame = np.full((frame_h, frame_w, 3), 25, dtype=np.uint8)
        frame[title_h:, 0:panel_w] = left
        frame[title_h:, panel_w:2 * panel_w] = mid
        frame[title_h:, 2 * panel_w:3 * panel_w] = right

        cv2.putText(frame, f"{i}: {name}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "original image", (12, title_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "unrotated no-filament mask", (panel_w + 12, title_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "growing merged map", (2 * panel_w + 12, title_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print("saved:", out_path)


if __name__ == "__main__":
    make_debug_video()