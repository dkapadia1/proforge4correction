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
    m = re.search(
        r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$",
        os.path.splitext(name)[0],
    )
    if not m:
        return None
    frame, t, x, y, z = m.groups()
    return dict(frame=int(frame), t=float(t), x=float(x), y=float(y), z=float(z))


def paste_or(canvas, mask, cx, cy):
    h, w = mask.shape
    x0, y0 = cx - w // 2, cy - h // 2
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1 = min(canvas.shape[1], x0 + w)
    dy1 = min(canvas.shape[0], y0 + h)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx0, sy0 = dx0 - x0, dy0 - y0
    canvas[dy0:dy1, dx0:dx1] |= mask[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]


def read_masks(folder, img_i, radius, threshold):
    mask, pack = extract_filament_array(
        folder=folder,
        img_i=img_i,
        radius=radius,
        threshold=threshold,
        grad=None,
        full_grad=None,
    )
    no_filament = undo_preprocess_mask(mask, pack, radius=radius)
    evaluated = undo_preprocess_mask(np.ones_like(mask, bool), pack, radius=radius)
    return no_filament, evaluated


def merge_miss_print_or(
    folder,
    gcode_file,
    radius=35,
    threshold=0.15,
    px_per_mm=26.13,
    margin_px=200,
    flip_x=False,
    flip_y=True,
    swap_xy=False,
    layer_index=0,
    gcode_line_width_mm=0.45,
    max_frame_z=0.20,
    merged_out_path='merged_no_filament_or.png',
    expected_out_path='expected_filament.png',
    miss_out_path='miss_print_or.png',
):
    folder = os.path.join(folder, '')
    photos = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS],
        key=number,
    )

    frames = []
    for i, name in enumerate(photos):
        pose = parse_pose(name)
        if pose is None or (max_frame_z is not None and pose['z'] > max_frame_z):
            continue
        x, y = transform_xy(
            pose['x'], pose['y'],
            flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy,
        )
        frames.append((i, name, pose, x, y))

    if not frames:
        raise ValueError('No valid scan frames found.')

    sample_no, _ = read_masks(folder, frames[0][0], radius, threshold)
    mask_h, mask_w = sample_no.shape
    xs = np.array([f[3] for f in frames])
    ys = np.array([f[4] for f in frames])

    min_x_px = int(np.floor(xs.min() * px_per_mm - mask_w - margin_px))
    max_x_px = int(np.ceil(xs.max() * px_per_mm + mask_w + margin_px))
    min_y_px = int(np.floor(ys.min() * px_per_mm - mask_h - margin_px))
    max_y_px = int(np.ceil(ys.max() * px_per_mm + mask_h + margin_px))
    width, height = max_x_px - min_x_px, max_y_px - min_y_px
    origin_xy_mm = (min_x_px / px_per_mm, min_y_px / px_per_mm)

    no_or = np.zeros((height, width), bool)
    covered_or = np.zeros((height, width), bool)

    for i, _, _, x, y in tqdm(frames, desc='Merging OR no-filament mask'):
        no_mask, evaluated = read_masks(folder, i, radius, threshold)
        cx = int(round((x - origin_xy_mm[0]) * px_per_mm))
        cy = int(round((y - origin_xy_mm[1]) * px_per_mm))
        paste_or(no_or, no_mask, cx, cy)
        paste_or(covered_or, evaluated, cx, cy)

    merged = np.full((height, width), 127, np.uint8)
    merged[covered_or] = 255
    merged[no_or] = 0
    cv2.imwrite(merged_out_path, merged)

    expected, info = make_gcode_coordinate_mask(
        gcode_file=gcode_file,
        px_per_mm=px_per_mm,
        layer_index=layer_index,
        origin_xy_mm=origin_xy_mm,
        mask_shape=(height, width),
        line_width_mm=gcode_line_width_mm,
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
        return_info=True,
    )
    expected_mask = expected.mask
    cv2.imwrite(expected_out_path, expected_mask.astype(np.uint8) * 255)

    miss = expected_mask & no_or
    cv2.imwrite(miss_out_path, miss.astype(np.uint8) * 255)

    return {
        'merged_image': merged,
        'no_filament_mask': no_or,
        'covered_mask': covered_or,
        'expected_filament_mask': expected_mask,
        'miss_print_mask': miss,
        'origin_xy_mm': origin_xy_mm,
        'target_z': info['target_z'],
        'frames_used': len(frames),
    }
