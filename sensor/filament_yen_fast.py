"""
Fast offset-tuned Yen-mask filament extractor.

Public API stays close to filament_array_offset_yen.py:
    mask, crop_info = extract_filament_array(...)
    full_mask = undo_preprocess_mask(mask, crop_info)

Main speedups:
  - shared fit_cache across calls
  - cached sorted photo lists, empty/full image reads, fitted profiles, ROI bounds
  - no full-size actual.copy() per frame
  - diff is computed only inside the constant laser crop
  - OpenCV connected-components cleanup instead of importing skimage morphology per frame
"""

import os
import random
import re
from typing import Any

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


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_FOLDER = r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\"
PX_PER_MM = 26.13
PX_TO_MM = 1.0 / PX_PER_MM
THETA_DEG = 26.2
EMPTY_CENTER_OFFSET_ROWS = -26.0
DEFAULT_EMPTY_I = 2419 - 149
DEFAULT_FULL_I = 1949 - 149


def make_fit_cache() -> dict[str, Any]:
    return {}


def number(name):
    m = re.search(r"frame_(\d+)", str(name)) or re.search(r"\d+", str(name))
    return int(m.group(1)) if m else -1


def _cache(fit_cache=None):
    return {} if fit_cache is None else fit_cache


def _bucket(fit_cache, name):
    if name not in fit_cache:
        fit_cache[name] = {}
    return fit_cache[name]


def _folder_key(folder):
    return os.path.abspath(os.path.join(str(folder), ""))


def list_photos(folder=DEFAULT_FOLDER, fit_cache=None):
    fit_cache = _cache(fit_cache)
    folder = _folder_key(folder)
    photos_cache = _bucket(fit_cache, "photos")
    if folder not in photos_cache:
        photos_cache[folder] = sorted(
            [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS],
            key=number,
        )
    return photos_cache[folder]


def _photo_path(folder, photos, i):
    return os.path.join(_folder_key(folder), photos[int(i)])


def read_gray_cached(path, fit_cache=None, keep=False):
    """Cache references/templates; avoid caching every actual frame unless keep=True."""
    path = os.path.abspath(str(path))
    if not keep:
        return np.ascontiguousarray(read_gray(path))

    fit_cache = _cache(fit_cache)
    gray_cache = _bucket(fit_cache, "gray")
    if path not in gray_cache:
        gray_cache[path] = np.ascontiguousarray(read_gray(path))
    return gray_cache[path]


def fit_with_center_offset(fit_info, offset_rows=0.0):
    out = fit_info.copy()
    params = out["measured_params"].copy()
    params[1] = out["measured_center"] + float(offset_rows)
    fit_profile = pseudo_voigt_profile(out["x_canon"], *params).astype(np.float32, copy=False)

    out["aligned_params"] = params
    out["fit_profile"] = fit_profile
    out["fit_rect"] = np.repeat(fit_profile[:, None], LASER_L, axis=1).astype(np.float32, copy=False)
    out["target_center"] = float(params[1])
    out["center_shift_applied"] = float(offset_rows)
    return out


def _reducer_name(reducer):
    return getattr(reducer, "__name__", str(reducer))


def _template_fits(
    empty_template_img,
    full_template_img,
    reducer=np.median,
    fit_cache=None,
    empty_row_pad=35,
    full_row_pad=0,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
):
    fit_cache = _cache(fit_cache)
    empty_path = os.path.abspath(str(empty_template_img))
    
    full_path = os.path.abspath(str(full_template_img))
    print(empty_path, full_path)
    empty = read_gray_cached(empty_path, fit_cache, keep=True)
    full = read_gray_cached(full_path, fit_cache, keep=True)
    if empty.shape != full.shape:
        raise ValueError(f"empty/full shapes differ: {empty.shape} vs {full.shape}")

    base_fit_cache = _bucket(fit_cache, "base_fit")
    reducer_key = _reducer_name(reducer)

    def base_fit(name, img, path, row_pad):
        key = (name, path, reducer_key, int(row_pad))
        if key not in base_fit_cache:
            base_fit_cache[key] = fit_template_median_profile(
                img, reducer=reducer, row_pad=row_pad, target_center=None
            )
        return base_fit_cache[key]

    full_base = base_fit("full", full, full_path, full_row_pad)
    empty_base = base_fit("empty", empty, empty_path, empty_row_pad)

    offset_cache = _bucket(fit_cache, "offset_fit")
    offset_key = (empty_path, reducer_key, int(empty_row_pad), float(empty_center_offset_rows))
    if offset_key not in offset_cache:
        offset_cache[offset_key] = fit_with_center_offset(empty_base, empty_center_offset_rows)

    return offset_cache[offset_key], full_base, empty, full


def _render_parts(
    empty_template_img,
    full_template_img,
    mask_true="empty",
    reducer=np.median,
    fit_cache=None,
    empty_row_pad=35,
    full_row_pad=0,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
):
    fit_cache = _cache(fit_cache)
    key = (
        os.path.abspath(str(empty_template_img)),
        os.path.abspath(str(full_template_img)),
        mask_true,
        _reducer_name(reducer),
        int(empty_row_pad),
        int(full_row_pad),
        float(empty_center_offset_rows),
    )
    parts_cache = _bucket(fit_cache, "render_parts")
    if key in parts_cache:
        return parts_cache[key]

    empty_fit, full_fit, empty_img, full_img = _template_fits(
        empty_template_img,
        full_template_img,
        reducer=reducer,
        fit_cache=fit_cache,
        empty_row_pad=empty_row_pad,
        full_row_pad=full_row_pad,
        empty_center_offset_rows=empty_center_offset_rows,
    )

    empty_rect = np.asarray(empty_fit["fit_rect"], dtype=np.float32)
    full_rect = np.asarray(full_fit["fit_rect"], dtype=np.float32)
    if mask_true == "empty":
        base_rect, delta_rect = full_rect, empty_rect - full_rect
    elif mask_true == "full":
        base_rect, delta_rect = empty_rect, full_rect - empty_rect
    else:
        raise ValueError("mask_true must be 'empty' or 'full'")

    parts = {
        "empty_fit": empty_fit,
        "full_fit": full_fit,
        "empty_img": empty_img,
        "full_img": full_img,
        "base_rect": np.ascontiguousarray(base_rect, dtype=np.float32),
        "delta_rect": np.ascontiguousarray(delta_rect, dtype=np.float32),
    }
    parts_cache[key] = parts
    return parts


def _work_arrays(fit_cache=None):
    fit_cache = _cache(fit_cache)
    work = _bucket(fit_cache, "work")
    if "local_out" not in work:
        work["local_out"] = np.empty((LASER_W, LASER_L), dtype=np.float32)
    if "canvas" not in work:
        work["canvas"] = np.zeros(CANVAS_SHAPE, dtype=np.float32)
    return work["local_out"], work["canvas"]


def _roi_info(fit_cache=None, crop_pad=25):
    fit_cache = _cache(fit_cache)
    roi_cache = _bucket(fit_cache, "roi")
    key = int(crop_pad)
    if key not in roi_cache:
        roi = laser_roi_mask().astype(bool, copy=False)
        ys, xs = np.where(roi)
        if ys.size == 0:
            raise ValueError("laser_roi_mask() is empty")
        y0 = max(0, int(ys.min()) - key)
        y1 = min(roi.shape[0], int(ys.max()) + key + 1)
        x0 = max(0, int(xs.min()) - key)
        x1 = min(roi.shape[1], int(xs.max()) + key + 1)
        roi_cache[key] = {"roi": roi, "bounds": (y0, y1, x0, x1), "roi_crop": roi[y0:y1, x0:x1]}
    return roi_cache[key]


def _render_fitted_output(
    local_mask_wl,
    empty_template_img,
    full_template_img,
    mask_true="empty",
    reducer=np.median,
    fit_cache=None,
    empty_row_pad=35,
    full_row_pad=0,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
):
    local_mask_wl = np.asarray(local_mask_wl, dtype=bool)
    if local_mask_wl.shape != (LASER_W, LASER_L):
        raise ValueError(f"local mask must have shape {(LASER_W, LASER_L)}, got {local_mask_wl.shape}")

    parts = _render_parts(
        empty_template_img,
        full_template_img,
        mask_true=mask_true,
        reducer=reducer,
        fit_cache=fit_cache,
        empty_row_pad=empty_row_pad,
        full_row_pad=full_row_pad,
        empty_center_offset_rows=empty_center_offset_rows,
    )
    local_out, canvas = _work_arrays(fit_cache)

    np.multiply(local_mask_wl, parts["delta_rect"], out=local_out, casting="unsafe")
    local_out += parts["base_rect"]

    canvas.fill(0.0)
    canvas[RECT_ROW0 : RECT_ROW0 + LASER_W, RECT_COL0 : RECT_COL0 + LASER_L] = local_out
    return rotate_canvas_to_output(canvas, interpolation=cv2.INTER_LINEAR), parts


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
    fitted_output, parts = _render_fitted_output(
        local_mask_wl,
        empty_template_img,
        full_template_img,
        mask_true=mask_true,
        reducer=reducer,
        fit_cache=fit_cache,
        empty_row_pad=empty_row_pad,
        full_row_pad=full_row_pad,
        empty_center_offset_rows=empty_center_offset_rows,
    )
    roi = _roi_info(fit_cache, crop_pad=0)["roi"]

    if isinstance(background, str) and background == "zeros":
        out = fitted_output
    elif isinstance(background, str) and background == "empty_template":
        out = parts["empty_img"].astype(np.float32, copy=True)
        out[roi] = fitted_output[roi]
    else:
        out = np.asarray(background, dtype=np.float32).copy()
        if out.shape != parts["empty_img"].shape:
            raise ValueError("background ndarray must match template image shape")
        out[roi] = fitted_output[roi]

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if not return_debug:
        return out_u8
    return out_u8, {
        "empty_fit": parts["empty_fit"],
        "full_fit": parts["full_fit"],
        "fitted_output": fitted_output,
        "laser_roi": roi,
        "rotated_mask": rotate_local_mask(np.asarray(local_mask_wl, dtype=bool)),
        "fit_cache": fit_cache,
    }


def crop_to_mask(img, mask, pad=25, return_bounds=False):
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("cannot crop: mask is empty")
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(img.shape[0], int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(img.shape[1], int(xs.max()) + int(pad) + 1)
    crop = img[y0:y1, x0:x1]
    return (crop, (y0, y1, x0, x1)) if return_bounds else crop


def constant_preprocess_crop_info(img_or_shape, crop_bounds, radius=25, laser_roi=None):
    y0, y1, x0, x1 = map(int, crop_bounds)
    if isinstance(img_or_shape, tuple):
        H, W = map(int, img_or_shape[:2])
        roi = None
    else:
        H, W = img_or_shape.shape[:2]
        roi = img_or_shape[y0:y1, x0:x1]
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return {
        "mode": "constant_output_crop",
        "original_shape": (H, W),
        "rotated_shape": (H, W),
        "M": ident.copy(),
        "Minv": ident.copy(),
        "crop": (y0 - int(radius), y1 - int(radius), x0, x1),
        "constant_crop_bounds": (y0, y1, x0, x1),
        "roi": roi,
        "peak": None,
        "laser_roi": laser_roi,
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
    min_area = int(max_size)
    if min_area <= 1:
        return mask.astype(bool, copy=False)
    conn = 8 if int(connectivity) > 1 else 4
    labels, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), conn)[1:3]
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def diff_to_mask(diff, threshold="yen", small_object_max_size=60, connectivity=10):
    t = _threshold_yen(diff) if threshold in (None, "yen") else float(threshold)
    return _remove_small_objects(
        np.asarray(diff) > t,
        max_size=small_object_max_size,
        connectivity=connectivity,
    )


def _hashable_kwargs(kwargs):
    return tuple(sorted((k, repr(v)) for k, v in kwargs.items()))


def expected_empty_mask_from_gcode(img_path, theta_deg=THETA_DEG, px_to_mm=PX_TO_MM, fit_cache=None, **gcode_kwargs):
    fit_cache = _cache(fit_cache)
    path = os.path.abspath(str(img_path))
    key = (path, float(theta_deg), float(px_to_mm), _hashable_kwargs(gcode_kwargs))
    mask_cache = _bucket(fit_cache, "expected_empty_mask")
    if key not in mask_cache:
        x, y, z = parse_frame_xyz(path)
        mask_cache[key] = ~gcode_expected_print_mask(x, y, z, theta_deg=theta_deg, px_to_mm=px_to_mm, **gcode_kwargs)
    return mask_cache[key]


def laser_diff_crop(
    actual,
    local_mask_wl,
    empty_template_img,
    full_template_img,
    mask_true="empty",
    reducer=np.median,
    fit_cache=None,
    crop_pad=25,
    empty_row_pad=35,
    full_row_pad=0,
    empty_center_offset_rows=EMPTY_CENTER_OFFSET_ROWS,
    return_debug=False,
):
    fitted_output, parts = _render_fitted_output(
        local_mask_wl,
        empty_template_img,
        full_template_img,
        mask_true=mask_true,
        reducer=reducer,
        fit_cache=fit_cache,
        empty_row_pad=empty_row_pad,
        full_row_pad=full_row_pad,
        empty_center_offset_rows=empty_center_offset_rows,
    )
    roi_info = _roi_info(fit_cache, crop_pad=crop_pad)
    y0, y1, x0, x1 = roi_info["bounds"]
    roi_crop = roi_info["roi_crop"]

    actual_crop = np.asarray(actual[y0:y1, x0:x1])
    fit_crop = np.clip(fitted_output[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    actual_u8 = np.clip(actual_crop, 0, 255).astype(np.uint8, copy=False)

    diff_crop = np.zeros(actual_u8.shape, dtype=np.uint8)
    diff_inside = cv2.absdiff(actual_u8, fit_crop)
    diff_crop[roi_crop] = diff_inside[roi_crop]

    if not return_debug:
        return diff_crop, roi_info["bounds"]
    return diff_crop, roi_info["bounds"], {
        "empty_fit": parts["empty_fit"],
        "full_fit": parts["full_fit"],
        "laser_roi": roi_info["roi"],
        "roi_crop": roi_crop,
        "fitted_output": fitted_output,
        "fit_cache": fit_cache,
    }


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
    photos=None,
    cache_actual=False,
    **gcode_kwargs,
):
    """Return the cropped no-filament mask and constant crop_info pack."""
    del grad, full_grad
    fit_cache = _cache(fit_cache)
    folder = _folder_key(folder)
    photos = list_photos(folder, fit_cache) if photos is None else photos

    if img_i is None:
        img_i = random.randrange(len(photos))
        print(_photo_path(folder, photos, img_i))

    empty_path = _photo_path(folder, photos, empty_i)
    full_path = _photo_path(folder, photos, full_i)
    img_path = _photo_path(folder, photos, img_i)

    actual = read_gray_cached(img_path, fit_cache, keep=cache_actual)
    local_mask_wl = expected_empty_mask_from_gcode(
        img_path,
        theta_deg=theta_deg,
        px_to_mm=px_to_mm,
        fit_cache=fit_cache,
        **gcode_kwargs,
    )

    if return_score:
        diff_crop, bounds, _debug = laser_diff_crop(
            actual,
            local_mask_wl,
            empty_path,
            full_path,
            mask_true=mask_true,
            fit_cache=fit_cache,
            crop_pad=crop_pad,
            empty_center_offset_rows=empty_center_offset_rows,
            return_debug=True,
        )
    else:
        diff_crop, bounds = laser_diff_crop(
            actual,
            local_mask_wl,
            empty_path,
            full_path,
            mask_true=mask_true,
            fit_cache=fit_cache,
            crop_pad=crop_pad,
            empty_center_offset_rows=empty_center_offset_rows,
        )

    mask = diff_to_mask(
        diff_crop,
        threshold=threshold,
        small_object_max_size=small_object_max_size,
        connectivity=connectivity,
    )
    crop_info = constant_preprocess_crop_info(actual.shape, bounds, radius=radius, laser_roi=_roi_info(fit_cache, crop_pad=0)["roi"])
    return (mask, crop_info, diff_crop) if return_score else (mask, crop_info)


def extract_filament_array_offset_mask(*args, **kwargs):
    return extract_filament_array(*args, **kwargs)


def undo_preprocess_mask(mask, info, radius=25):
    """Expand a cropped mask back to the full camera image."""
    if info.get("mode") == "constant_output_crop":
        H, W = info["original_shape"]
        y0, _, x0, _ = info["constant_crop_bounds"]
        out = np.zeros((H, W), dtype=bool)
        mh, mw = mask.shape
        out[y0 : y0 + mh, x0 : x0 + mw] = mask.astype(bool, copy=False)
        return out

    H, W = info["original_shape"]
    band_top, _, left, _ = info["crop"]
    rot_mask = np.zeros((H, W), dtype=np.uint8)
    mh, mw = mask.shape
    y0 = band_top + int(radius)
    x0 = left
    rot_mask[y0 : y0 + mh, x0 : x0 + mw] = mask.astype(np.uint8) * 255
    unrot = cv2.warpAffine(rot_mask, info["Minv"], (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return unrot > 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cache = make_fit_cache()
    mask, pack, diff = extract_filament_array(img_i=101, return_score=True, fit_cache=cache)
    print("mask pixels:", int(mask.sum()))
    print("cache buckets:", sorted(cache.keys()))
    plt.figure(); plt.title("diff crop"); plt.imshow(diff, cmap="magma")
    plt.figure(); plt.title("mask"); plt.imshow(mask, cmap="gray")
    plt.figure(); plt.title("full-size mask"); plt.imshow(undo_preprocess_mask(mask, pack), cmap="gray")
    print(cache["photos"].keys())
    img_path = _photo_path(DEFAULT_FOLDER, cache["photos"]['C:\\Users\\dhruv\\Documents\\dhruv_python\\disc2accurate'], 101)
    plt.figure(); plt.title("actual"); plt.imshow(read_gray_cached(img_path, cache), cmap="gray")
    plt.show()
