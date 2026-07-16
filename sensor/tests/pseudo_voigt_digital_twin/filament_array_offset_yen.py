"""
Drop-in filament mask extractor using the offset-tuned synthetic laser method.

This is based on offset_tuning.ipynb:
  1. build the expected empty/full local laser mask from the frame XYZ,
  2. render a synthetic laser image with the empty-template center offset fixed at -26 rows,
  3. take absdiff(actual, synthetic),
  4. crop to the constant laser ROI,
  5. threshold with Yen and remove small objects.

The returned crop-info object is constant/geometry-based, not detected from the
random image. It is compatible with the undo_preprocess_mask() below, and also
with the older undo_preprocess_mask(mask, info, radius=25) behavior because the
"crop" field is filled accordingly.
"""

import os
import random
import re

import cv2
import numpy as np

from mask_to_laser import (
    LASER_W,
    LASER_L,
    CANVAS_SHAPE,
    RECT_ROW0,
    RECT_COL0,
    read_gray,
    pseudo_voigt_profile,
    fit_template_median_profile,
    rotate_canvas_to_output,
    rotate_local_mask,
    laser_roi_mask,
)
from gcode_expected_print_mask import gcode_expected_print_mask, parse_frame_xyz


DEFAULT_FOLDER = r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\"
PX_PER_MM = 26.13
PX_TO_MM = 1.0 / PX_PER_MM
THETA_DEG = 26.2
EMPTY_CENTER_OFFSET_ROWS = -26.0
DEFAULT_EMPTY_I = 2419 - 149
DEFAULT_FULL_I = 1949 - 149


def number(p):
    match = re.search(r"\d+", str(p))
    return int(match.group()) if match else -1


def _photo_path(folder, photos, i):
    return os.path.join(folder, photos[i])


def fit_with_center_offset(fit_info, offset_rows=0.0):
    out = fit_info.copy()
    params = out["measured_params"].copy()
    params[1] = out["measured_center"] + float(offset_rows)
    fit_profile = pseudo_voigt_profile(out["x_canon"], *params)

    out["aligned_params"] = params
    out["fit_profile"] = fit_profile
    out["fit_rect"] = np.repeat(fit_profile[:, None], LASER_L, axis=1)
    out["target_center"] = float(params[1])
    out["center_shift_applied"] = float(offset_rows)
    return out


def make_laser_image(
    local_mask_wl,
    empty_template_img,
    full_template_img,
    mask_true="empty",
    reducer=np.median,
    background="zeros",
    fit_cache=None,
    return_debug=False,
    empty_row_pad=35,
    full_row_pad=0,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
):
    local_mask_wl = np.asarray(local_mask_wl, dtype=bool)
    if local_mask_wl.shape != (LASER_W, LASER_L):
        raise ValueError(f"local mask must have shape {(LASER_W, LASER_L)}, got {local_mask_wl.shape}")

    empty = read_gray(str(empty_template_img))
    full = read_gray(str(full_template_img))
    if empty.shape != full.shape:
        raise ValueError(f"empty/full shapes differ: {empty.shape} vs {full.shape}")

    fit_cache = {} if fit_cache is None else fit_cache

    def cache_key(name, img, row_pad):
        return (
            name,
            img.shape,
            img.dtype.str,
            round(float(img.mean()), 6),
            round(float(img.std()), 6),
            getattr(reducer, "__name__", str(reducer)),
            int(row_pad),
        )

    def get_base_fit(name, img, row_pad):
        key = cache_key(name, img, row_pad)
        if key not in fit_cache:
            fit_cache[key] = fit_template_median_profile(
                img, reducer=reducer, row_pad=row_pad, target_center=None
            )
        return fit_cache[key]

    full_fit = get_base_fit("full", full, full_row_pad)
    empty_fit = fit_with_center_offset(
        get_base_fit("empty", empty, empty_row_pad), empty_center_offset_rows
    )

    if mask_true == "empty":
        local_out = np.where(local_mask_wl, empty_fit["fit_rect"], full_fit["fit_rect"])
    elif mask_true == "full":
        local_out = np.where(local_mask_wl, full_fit["fit_rect"], empty_fit["fit_rect"])
    else:
        raise ValueError("mask_true must be 'empty' or 'full'")

    canvas = np.zeros(CANVAS_SHAPE, dtype=float)
    canvas[RECT_ROW0 : RECT_ROW0 + LASER_W, RECT_COL0 : RECT_COL0 + LASER_L] = local_out
    fitted_output = rotate_canvas_to_output(canvas, interpolation=cv2.INTER_LINEAR)
    roi = laser_roi_mask()

    if isinstance(background, str) and background == "zeros":
        out = fitted_output
    elif isinstance(background, str) and background == "empty_template":
        out = empty.astype(float).copy()
        out[roi] = fitted_output[roi]
    else:
        out = np.asarray(background, dtype=float).copy()
        if out.shape != empty.shape:
            raise ValueError("background ndarray must match template image shape")
        out[roi] = fitted_output[roi]

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if not return_debug:
        return out_u8

    return out_u8, {
        "empty_fit": empty_fit,
        "full_fit": full_fit,
        "local_out": local_out,
        "fitted_output": fitted_output,
        "laser_roi": roi,
        "rotated_mask": rotate_local_mask(local_mask_wl),
        "fit_cache": fit_cache,
    }


def crop_to_mask(img, mask, pad=25, return_bounds=False):
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("cannot crop: mask is empty")

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad + 1)
    crop = img[y0:y1, x0:x1]
    return (crop, (y0, y1, x0, x1)) if return_bounds else crop


def constant_preprocess_crop_info(img, crop_bounds, radius=25):
    """Return a geometry-based crop pack for merge/undo, without image-driven crop detection."""
    y0, y1, x0, x1 = map(int, crop_bounds)
    H, W = img.shape[:2]
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    return {
        "mode": "constant_output_crop",
        "original_shape": (H, W),
        "rotated_shape": (H, W),
        "M": ident.copy(),
        "Minv": ident.copy(),
        # Older undo_preprocess_mask() inserts at band_top + radius.
        # This makes old code paste the returned mask at exactly y0, x0.
        "crop": (y0 - int(radius), y1 - int(radius), x0, x1),
        "constant_crop_bounds": (y0, y1, x0, x1),
        "roi": img[y0:y1, x0:x1],
        "peak": None,
        "laser_roi": laser_roi_mask(),
    }


def _threshold_yen(diff):
    try:
        from skimage.filters import threshold_yen

        return float(threshold_yen(diff))
    except Exception:
        arr = np.asarray(diff)
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        t, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(t)


def _remove_small_objects(mask, max_size=60, connectivity=10):
    try:
        from skimage.morphology import remove_small_objects

        try:
            return remove_small_objects(mask, max_size=max_size, connectivity=connectivity)
        except TypeError:
            return remove_small_objects(mask, min_size=max_size, connectivity=connectivity)
    except Exception:
        labels, stats_mask = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[1:3]
        out = np.zeros_like(mask, dtype=bool)
        for label in range(1, stats_mask.shape[0]):
            if stats_mask[label, cv2.CC_STAT_AREA] >= max_size:
                out[labels == label] = True
        return out


def diff_to_mask(diff, threshold="yen", small_object_max_size=60, connectivity=10):
    diff = np.abs(diff)
    t = _threshold_yen(diff) if threshold in (None, "yen") else float(threshold)
    mask = diff > t
    return _remove_small_objects(mask, max_size=small_object_max_size, connectivity=connectivity)


def expected_empty_mask_from_gcode(img_path, theta_deg=THETA_DEG, px_to_mm=PX_TO_MM, **gcode_kwargs):
    x, y, z = parse_frame_xyz(str(img_path))
    return ~gcode_expected_print_mask(x, y, z, theta_deg=theta_deg, px_to_mm=px_to_mm, **gcode_kwargs)


def extract_filament_array(
    folder=DEFAULT_FOLDER,
    empty_i=DEFAULT_EMPTY_I,
    full_i=DEFAULT_FULL_I,
    img_i=None,
    radius=25,
    threshold="yen",
    grad=None,
    full_grad=None,
    return_score=False,
    theta_deg=THETA_DEG,
    px_to_mm=PX_TO_MM,
    crop_pad=25,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
    small_object_max_size=60,
    connectivity=10,
    mask_true="empty",
    fit_cache=None,
    **gcode_kwargs,
):
    """
    Drop-in replacement for filament_array(5).extract_filament_array().

    Returns
    -------
    mask, crop_info
        mask is the Yen-thresholded cropped diff mask, equivalent to the notebook's `bin`.
        crop_info is a constant geometry-based pack for undoing the crop.
    mask, crop_info, diff_crop
        Returned when return_score=True.
    """
    del grad, full_grad  # accepted only for old-call compatibility

    photos = sorted(os.listdir(folder), key=number)
    empty_path = _photo_path(folder, photos, empty_i)
    full_path = _photo_path(folder, photos, full_i)
    img_path = _photo_path(folder, photos, random.randrange(len(photos)) if img_i is None else img_i)
    if img_i is None:
        print(img_path)

    actual = read_gray(str(img_path))
    local_mask_wl = expected_empty_mask_from_gcode(
        img_path, theta_deg=theta_deg, px_to_mm=px_to_mm, **gcode_kwargs
    )

    twin, dbg = make_laser_image(
        local_mask_wl,
        empty_path,
        full_path,
        mask_true=mask_true,
        background=actual,
        fit_cache=fit_cache,
        return_debug=True,
        empty_center_offset_rows=empty_center_offset_rows,
    )

    diff = cv2.absdiff(actual, twin)
    diff_crop, bounds = crop_to_mask(diff, dbg["laser_roi"], pad=crop_pad, return_bounds=True)
    mask = diff_to_mask(
        diff_crop,
        threshold=threshold,
        small_object_max_size=small_object_max_size,
        connectivity=connectivity,
    )
    crop_info = constant_preprocess_crop_info(actual, bounds, radius=radius)

    if return_score:
        return mask, crop_info, diff_crop
    return mask, crop_info


def extract_filament_array_offset_mask(*args, **kwargs):
    return extract_filament_array(*args, **kwargs)


def undo_preprocess_mask(mask, info, radius=25):
    """Undo the returned constant crop, with fallback for old filament_array crop packs."""
    if info.get("mode") == "constant_output_crop":
        H, W = info["original_shape"]
        y0, y1, x0, x1 = info["constant_crop_bounds"]
        out = np.zeros((H, W), dtype=bool)
        mh, mw = mask.shape
        out[y0 : y0 + mh, x0 : x0 + mw] = mask.astype(bool)
        return out

    H, W = info["original_shape"]
    band_top, _, left, _ = info["crop"]
    rot_mask = np.zeros((H, W), dtype=np.uint8)
    mh, mw = mask.shape
    y0 = band_top + radius
    x0 = left
    rot_mask[y0 : y0 + mh, x0 : x0 + mw] = mask.astype(np.uint8) * 255
    unrot = cv2.warpAffine(rot_mask, info["Minv"], (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return unrot > 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mask, pack, diff = extract_filament_array(img_i=101, return_score=True)
    print("mask pixels:", int(mask.sum()))
    plt.figure(); plt.title("diff crop"); plt.imshow(diff, cmap="magma")
    plt.figure(); plt.title("mask"); plt.imshow(mask, cmap="gray")
    plt.figure(); plt.title("full-size mask"); plt.imshow(undo_preprocess_mask(mask, pack), cmap="gray")
    plt.show()
