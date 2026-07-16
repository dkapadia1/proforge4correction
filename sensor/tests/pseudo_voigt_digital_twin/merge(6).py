import os, re, importlib.util
import numpy as np
import cv2
from filament_array_offset_yen import extract_filament_array, undo_preprocess_mask
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def number(name):
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else -1


def parse_pose(name):
    stem = os.path.splitext(name)[0]
    m = re.search(
        r"frame_(\d+)_t_([-\d.]+)_x_([-\d.]+)_y_([-\d.]+)_z_([-\d.]+)$",
        stem,
    )
    if not m:
        return None

    frame, t, x, y, z = m.groups()
    return {"frame": int(frame), "t": float(t), "x": float(x), "y": float(y), "z": float(z)}


def _load_gcode_module():
    """Load gcode_expected_print_mask.py, with a fallback for downloaded '(2)' files."""
    try:
        import gcode_expected_print_mask as mod
        return mod
    except ModuleNotFoundError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("gcode_expected_print_mask(3).py"):
        path = os.path.join(here, fname)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("gcode_expected_print_mask", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    raise ModuleNotFoundError(
        "Could not find gcode_expected_print_mask.py next to merge.py. "
        "Pass gcode_file=None to skip expected-print comparison."
    )


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
    slc = np.s_[dy0:dy1, dx0:dx1]
    src = mask[sy0:sy1, sx0:sx1]

    if mode == "or":
        canvas[slc] |= src
    else:
        canvas[slc] = src


def resize_mask(mask, sx=1.0, sy=1.0):
    h, w = mask.shape
    nw = max(1, int(round(w * sx)))
    nh = max(1, int(round(h * sy)))
    return cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)


def _canvas_xy(pose, flip_x=False, flip_y=True, swap_xy=False):
    x, y = pose["x"], pose["y"]
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return x, y


def _read_no_filament_mask(folder, img_i, empty_grad, full_grad, radius, threshold):
    mask, preprocess_pack = extract_filament_array(
        folder=folder,
        img_i=img_i,
        grad=empty_grad,
        full_grad=full_grad,
        radius=radius,
        threshold=threshold,
    )
    return undo_preprocess_mask(mask, preprocess_pack, radius=radius), mask.shape, preprocess_pack


def _expected_mask_for_frame(
    gcode_mod,
    segs,
    pose,
    raw_mask_shape,
    preprocess_pack,
    radius,
    px_per_mm,
    theta_deg,
    view_center_offset_mm,
    line_width_mm,
    include_previous_layers,
    z_tol,
):
    selected, _target_z, _layer_zs = gcode_mod.pick_layer_segments_nearest_z(
        segs,
        pose["z"],
        include_previous_layers=include_previous_layers,
        z_tol=z_tol,
    )
    raw_expected = gcode_mod.expected_print_mask_from_segments(
        selected,
        x=pose["x"],
        y=pose["y"],
        px_to_mm=1.0 / float(px_per_mm),
        mask_shape=raw_mask_shape,
        theta_deg=theta_deg,
        view_center_offset_mm=view_center_offset_mm,
        line_width_mm=line_width_mm,
    )
    return undo_preprocess_mask(raw_expected, preprocess_pack, radius=radius)


def merge_no_filament_folder(
    folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\",
    empty_i=2418 - 149,
    full_i=1946 - 149,
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
    gcode_file=None,
    missed_out_path="missed_print.png",
    expected_out_path=None,
    gcode_theta_deg=0.0,
    gcode_line_width_mm=0.45,
    gcode_view_center_offset_mm=(0.0, 0.0),
    include_previous_layers=False,
    z_tol=0.08,
):
    """
    Merge read no-filament masks, and optionally compare them against G-code.

    no_filament_mask:
        True where the image reader says there is NO filament.

    expected_filament_mask:
        True where the G-code says filament should be present.

    missed_print_mask:
        expected_filament_mask & no_filament_mask & covered_mask.
        This only flags missing expected filament. It does not flag extra printed filament.
    """
    folder = os.path.join(folder, "")
    photos = sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS],
        key=number,
    )

    empty_path = os.path.join(folder, photos[empty_i])
    full_path = os.path.join(folder, photos[full_i])
    empty_grad = None  # path_to_grad(empty_path, radius=radius, degree=degree)
    full_grad = None   # path_to_grad(full_path, radius=radius, degree=degree)

    frames = []
    for i, name in enumerate(photos):
        if skip_refs and i in {empty_i, full_i}:
            continue
        pose = parse_pose(name)
        if pose is None:
            continue
        cx_mm, cy_mm = _canvas_xy(pose, flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
        frames.append((i, name, pose, cx_mm, cy_mm))

    if not frames:
        raise ValueError("No valid frame filenames found.")

    sample_i, _sample_name, _sample_pose, _, _ = frames[0]
    sample_mask, _sample_raw_shape, _sample_pack = _read_no_filament_mask(
        folder, sample_i, empty_grad, full_grad, radius, threshold
    )
    mask_h, mask_w = sample_mask.shape

    xs = np.array([f[3] for f in frames])
    ys = np.array([f[4] for f in frames])
    min_x = int(np.floor(xs.min() * px_per_mm - mask_w - margin_px))
    max_x = int(np.ceil(xs.max() * px_per_mm + mask_w + margin_px))
    min_y = int(np.floor(ys.min() * px_per_mm - mask_h - margin_px))
    max_y = int(np.ceil(ys.max() * px_per_mm + mask_h + margin_px))
    W, H = max_x - min_x, max_y - min_y

    no_filament_canvas = np.zeros((H, W), dtype=bool)
    covered_canvas = np.zeros((H, W), dtype=bool)
    expected_canvas = np.zeros((H, W), dtype=bool) if gcode_file is not None else None

    if gcode_file is not None:
        gcode_mod = _load_gcode_module()
        segs, unsupported = gcode_mod.parse_gcode_extrusion_segments(gcode_file)
    else:
        gcode_mod = segs = unsupported = None

    for i, name, pose, x, y in tqdm(frames, desc="Merging frames"):
        if i == sample_i:
            mask, raw_shape, preprocess_pack = sample_mask, _sample_raw_shape, _sample_pack
        else:
            mask, raw_shape, preprocess_pack = _read_no_filament_mask(
                folder, i, empty_grad, full_grad, radius, threshold
            )

        cx = int(round(x * px_per_mm - min_x))
        cy = int(round(y * px_per_mm - min_y))
        paste_bool(no_filament_canvas, mask, cx, cy, mode="or")
        paste_bool(covered_canvas, np.ones_like(mask, dtype=bool), cx, cy, mode="or")

        if expected_canvas is not None:
            expected = _expected_mask_for_frame(
                gcode_mod,
                segs,
                pose,
                raw_shape,
                preprocess_pack,
                radius,
                px_per_mm,
                gcode_theta_deg,
                gcode_view_center_offset_mm,
                gcode_line_width_mm,
                include_previous_layers,
                z_tol,
            )
            paste_bool(expected_canvas, expected, cx, cy, mode="or")

    img = np.full((H, W), 127, dtype=np.uint8)
    img[covered_canvas] = 255                    # covered and not marked no-filament = assume filament
    img[no_filament_canvas] = 0                  # model says no filament
    cv2.imwrite(out_path, img)

    missed_canvas = None
    missed_img = None
    if expected_canvas is not None:
        missed_canvas = expected_canvas & no_filament_canvas & covered_canvas
        missed_img = np.zeros((H, W), dtype=np.uint8)
        missed_img[missed_canvas] = 255
        cv2.imwrite(missed_out_path, missed_img)

        if expected_out_path is not None:
            cv2.imwrite(expected_out_path, expected_canvas.astype(np.uint8) * 255)

    return {
        "image": img,
        "no_filament_mask": no_filament_canvas,
        "covered_mask": covered_canvas,
        "expected_filament_mask": expected_canvas,
        "missed_print_mask": missed_canvas,
        "missed_print_image": missed_img,
        "out_path": out_path,
        "missed_out_path": missed_out_path if missed_canvas is not None else None,
        "expected_out_path": expected_out_path,
        "unsupported_gcode_arcs": unsupported,
        "px_per_mm": px_per_mm,
        "min_x": min_x,
        "min_y": min_y,
    }


if __name__ == "__main__":
    out = merge_no_filament_folder(
        radius=35,
        px_per_mm=26.13,
        gcode_file="P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode",
        missed_out_path="missed_print.png",
        expected_out_path="expected_print.png",
    )
