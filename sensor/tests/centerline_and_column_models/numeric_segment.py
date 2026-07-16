import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from scipy.signal import savgol_filter
from typing import Tuple, List, Dict, Optional

# ---------- utilities (baseline_subtract, numeric_fwhm) ----------
def baseline_subtract(y: np.ndarray, window: int = 51, polyorder: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    if window >= len(y):
        window = len(y) - (1 - len(y) % 2)
        window = max(window, 3)
    baseline = savgol_filter(y, window_length=window, polyorder=min(polyorder, window-1))
    y_bs = y - baseline
    # estimate baseline noise as robust std of residuals
    noise = float(np.std(y_bs - np.median(y_bs)))
    return y_bs, baseline, noise

def numeric_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    y0 = np.min(y)
    y1 = y - y0
    peak = np.max(y1)
    if peak <= 0:
        return np.nan
    half = peak / 2.0
    above = y1 >= half
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    left_idx = idx[0]
    right_idx = idx[-1]
    def interp_cross(i0, i1):
        x0, y0v = x[i0], y1[i0]
        x1, y1v = x[i1], y1[i1]
        if y1v == y0v:
            return x1
        return x0 + (half - y0v) * (x1 - x0) / (y1v - y0v)
    left_x = x[0] if left_idx == 0 else interp_cross(left_idx - 1, left_idx)
    right_x = x[-1] if right_idx == len(x) - 1 else interp_cross(right_idx, right_idx + 1)
    return float(right_x - left_x)

# ---------- column metrics (returns noise too) ----------
def compute_column_metrics(col: np.ndarray,
                           x_grid: np.ndarray,
                           baseline_window: int = 51,
                           smooth_sigma: float = 1.0) -> Dict[str, float]:
    y = np.asarray(col, dtype=float)
    y_bs, baseline, noise = baseline_subtract(y, window=baseline_window)
    y_s = gaussian_filter1d(y_bs, sigma=smooth_sigma)
    amp = float(np.max(y) - np.min(y))
    fwhm = numeric_fwhm(x_grid, y_s)
    x0 = float(x_grid[np.argmax(y_s)])
    return {'amp': amp, 'fwhm': float(fwhm) if not np.isnan(fwhm) else np.nan, 'x0': x0, 'noise': noise, 'y_s': y_s}

# ---------- confidence calculator ----------
def compute_confidences(amps: np.ndarray,
                        fwhms: np.ndarray,
                        fwhm_smoothed: np.ndarray,
                        noises: np.ndarray,
                        fit_success: Optional[np.ndarray] = None,
                        fit_residuals: Optional[np.ndarray] = None,
                        weights: Optional[Dict[str, float]] = None,
                        neighbor_window: int = 3) -> np.ndarray:
    """
    Return confidence scores in [0,1] for each column.
    weights keys: amp, width, stability, snr, fit
    """
    n = len(amps)
    if weights is None:
        weights = {'amp': 0.35, 'width': 0.25, 'stability': 0.2, 'snr': 0.15, 'fit': 0.05}

    # amplitude score (normalized 0..1)
    amp_min, amp_max = np.nanmin(amps), np.nanmax(amps)
    amp_range = max(1e-12, amp_max - amp_min)
    amp_score = np.clip((amps - amp_min) / amp_range, 0.0, 1.0)

    # width validity score: positive finite fwhm -> scaled by quantile
    fwhm_pos = np.where(np.isfinite(fwhms) & (fwhms > 0), fwhms, 0.0)
    # scale by 90th percentile to avoid outliers dominating
    p90 = max(1e-12, np.nanpercentile(fwhm_pos, 90))
    width_score = np.clip(fwhm_pos / p90, 0.0, 1.0)

    # stability: how close fwhm is to local smoothed value
    # normalized difference -> stability = 1 - clipped relative diff
    diff = np.abs(fwhms - fwhm_smoothed)
    # scale by p90 to get relative measure
    stability_score = 1.0 - np.clip(diff / (p90 + 1e-12), 0.0, 1.0)
    stability_score = np.nan_to_num(stability_score, nan=0.0)

    # snr score: amp / noise, map to [0,1] using logistic-like scaling
    snr = np.zeros(n, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(noises > 0, amps / (noises + 1e-12), 0.0)
    # map SNR to [0,1] using saturating function: s/(s + s50)
    s50 = 5.0  # SNR at which score ~0.5; tuneable
    snr_score = snr / (snr + s50)
    snr_score = np.clip(snr_score, 0.0, 1.0)

    # fit score (optional)
    fit_score = np.zeros(n, dtype=float)
    if fit_success is not None:
        # success gives base 0.5, residuals reduce it
        fit_score = np.where(fit_success, 0.5, 0.0)
        if fit_residuals is not None:
            # map residuals to [0,1] by inverse scaling
            r = np.asarray(fit_residuals, dtype=float)
            r_med = np.nanmedian(r[np.isfinite(r)])
            r_scale = max(1e-12, r_med * 2.0 + 1e-12)
            fit_score = np.where(np.isfinite(r), fit_score * (1.0 / (1.0 + r / r_scale)), fit_score)

    # neighbor agreement: boost stability by local majority agreement
    # compute fraction of neighbors within tolerance
    neighbor_tol = max(1e-8, np.nanmedian(fwhm_smoothed[fwhm_smoothed>0]) * 0.5)
    neighbor_agree = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - neighbor_window)
        hi = min(n, i + neighbor_window + 1)
        window_vals = fwhm_smoothed[lo:hi]
        if np.all(np.isnan(window_vals)) or np.isnan(fwhm_smoothed[i]):
            neighbor_agree[i] = 0.0
        else:
            center = fwhm_smoothed[i]
            within = np.sum(np.abs(window_vals - center) <= neighbor_tol)
            neighbor_agree[i] = within / (hi - lo)
    # combine stability and neighbor agreement
    stability_combined = 0.6 * stability_score + 0.4 * neighbor_agree

    # final weighted sum
    total_weight = weights['amp'] + weights['width'] + weights['stability'] + weights['snr'] + weights['fit']
    conf = (weights['amp'] * amp_score +
            weights['width'] * width_score +
            weights['stability'] * stability_combined +
            weights['snr'] * snr_score +
            weights['fit'] * fit_score) / max(1e-12, total_weight)

    # clip and return
    conf = np.clip(conf, 0.0, 1.0)
    return conf

# ---------- full pipeline with confidences ----------
def segment_filament_with_confidence(img: np.ndarray,
                                     baseline_window: int = 51,
                                     col_smooth_sigma: float = 1.0,
                                     metric_smooth_sigma: float = 2.0,
                                     amp_thresh: Optional[float] = None,
                                     width_thresh: Optional[float] = None,
                                     min_segment_length: int = 3,
                                     refine_with_fit: bool = False,
                                     fit_method: str = 'robust',
                                     fit_f_scale: float = 3.0,
                                     confidence_threshold: float = 0.5,
                                     confidence_weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Returns dict with:
      - mask: boolean mask from thresholding (same as before)
      - confidence_mask: mask where confidence >= confidence_threshold
      - confidences: per-column confidence in [0,1]
      - segments: list of (start,end) from mask
      - segments_confident: list of (start,end) from confidence_mask
      - amps, fwhm_numeric, fwhm_smoothed, noises, x0s
      - optional fit_results if refine_with_fit True
    """
    n_rows, n_cols = img.shape
    x_grid = np.arange(n_rows)

    amps = np.zeros(n_cols, dtype=float)
    fwhm_numeric = np.full(n_cols, np.nan, dtype=float)
    x0s = np.full(n_cols, np.nan, dtype=float)
    noises = np.zeros(n_cols, dtype=float)

    # compute metrics per column
    for c in range(n_cols):
        col = img[:, c]
        metrics = compute_column_metrics(col, x_grid, baseline_window=baseline_window, smooth_sigma=col_smooth_sigma)
        amps[c] = metrics['amp']
        fwhm_numeric[c] = metrics['fwhm']
        x0s[c] = metrics['x0']
        noises[c] = metrics['noise']

    # smooth fwhm series for thresholding and stability
    fwhm_for_smooth = np.nan_to_num(fwhm_numeric, nan=0.0)
    fwhm_smoothed = gaussian_filter1d(fwhm_for_smooth, sigma=metric_smooth_sigma)

    # threshold and segment (same logic as before)
    if amp_thresh is None:
        amp_thresh = 0.05 * np.nanmax(amps) if np.nanmax(amps) > 0 else 0.0
    if width_thresh is None:
        positive = fwhm_smoothed[fwhm_smoothed > 0]
        width_thresh = float(np.nanquantile(positive, 0.6)) if positive.size > 0 else 0.0

    mask = (amps > amp_thresh) & (fwhm_smoothed > width_thresh)
    labeled, ncomp = label(mask)
    final_mask = np.zeros_like(mask, dtype=bool)
    segments = []
    for lab in range(1, ncomp + 1):
        inds = np.where(labeled == lab)[0]
        if inds.size >= min_segment_length:
            final_mask[inds] = True
            segments.append((int(inds[0]), int(inds[-1])))

    # optional refinement with fits
    fit_results = None
    fit_success = None
    fit_residuals = None
    if refine_with_fit:
        fit_results = [None] * n_cols
        fit_success = np.zeros(n_cols, dtype=bool)
        fit_residuals = np.full(n_cols, np.nan, dtype=float)
        for c in np.where(final_mask)[0]:
            col = img[:, c]
            try:
                res = pseudo_voigt_fit(x_grid, col, p0=None, bounds=None, method=fit_method, robust_f_scale=fit_f_scale)
                fit_results[c] = res
                fit_success[c] = bool(res.get('success', False))
                # use residual norm if available
                resid = res.get('residuals', None)
                if resid is not None:
                    fit_residuals[c] = float(np.linalg.norm(resid) / (np.sqrt(len(resid)) + 1e-12))
                # optionally replace numeric fwhm with model numeric fwhm
                if 'fwhm_numeric' in res and not np.isnan(res['fwhm_numeric']):
                    fwhm_numeric[c] = res['fwhm_numeric']
            except Exception:
                fit_results[c] = None
                fit_success[c] = False
                fit_residuals[c] = np.nan

    # compute confidences
    confidences = compute_confidences_scaled(amps, fwhm_numeric, fwhm_smoothed, noises,
                                      fit_success=fit_success, fit_residuals=fit_residuals,
                                      weights=confidence_weights)

    confidence_mask = confidences >= confidence_threshold

    # segments from confidence mask
    labeled_c, ncomp_c = label(confidence_mask)
    segments_confident = []
    final_conf_mask = np.zeros_like(confidence_mask, dtype=bool)
    for lab in range(1, ncomp_c + 1):
        inds = np.where(labeled_c == lab)[0]
        if inds.size >= min_segment_length:
            final_conf_mask[inds] = True
            segments_confident.append((int(inds[0]), int(inds[-1])))

    result = {
        'mask': final_mask,
        'segments': segments,
        'confidences': confidences,
        'confidence_mask': final_conf_mask,
        'segments_confident': segments_confident,
        'amps': amps,
        'fwhm_numeric': fwhm_numeric,
        'fwhm_smoothed': fwhm_smoothed,
        'noises': noises,
        'x0s': x0s,
        'fit_results': fit_results
    }
    return result
def contrast_stretch(arr, low_pct=5.0, high_pct=99.0, eps=1e-12):
    """Map arr so that low_pct -> 0, high_pct -> 1, clamp outside."""
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, low_pct)
    hi = np.nanpercentile(a, high_pct)
    if hi - lo < eps:
        return np.clip((a - lo) / (eps), 0.0, 1.0)
    stretched = (a - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)

def compute_confidences_scaled(amps: np.ndarray,
                               fwhms: np.ndarray,
                               fwhm_smoothed: np.ndarray,
                               noises: np.ndarray,
                               fit_success: Optional[np.ndarray] = None,
                               fit_residuals: Optional[np.ndarray] = None,
                               weights: Optional[Dict[str, float]] = None,
                               neighbor_window: int = 3,
                               amp_gamma: float = 2.0,
                               s50: float = 3.0,
                               stretch_low: float = 2.0,
                               stretch_high: float = 98.0) -> np.ndarray:
    """
    Compute a confidence score in [0,1] with stronger punishment for low intensity
    and contrast stretching so normal values are near 0 and peaks near 1.

    Key knobs:
      - amp_gamma: exponent applied to normalized amplitude (>=1). Larger -> punishes low amp.
      - s50: SNR value that maps to ~0.5 in SNR score. Smaller -> harsher SNR penalty.
      - stretch_low/high: percentiles used to stretch final combined score.
    """
    n = len(amps)
    # default weights if not provided
    if weights is None:
        weights = {'amp': 0.45, 'width': 0.15, 'stability': 0.15, 'snr': 0.2, 'fit': 0.05}

    # --- amplitude score (nonlinear) ---
    amp = np.asarray(amps, dtype=float)
    amp_min, amp_max = np.nanmin(amp), np.nanmax(amp)
    amp_range = max(1e-12, amp_max - amp_min)
    amp_norm = np.clip((amp - amp_min) / amp_range, 0.0, 1.0)
    amp_score = amp_norm ** amp_gamma   # nonlinear: low amps get punished

    # --- width score (positive widths only) ---
    fwhm_pos = np.where(np.isfinite(fwhms) & (fwhms > 0), fwhms, 0.0)
    p90 = max(1e-12, np.nanpercentile(fwhm_pos, 90))
    width_score = np.clip(fwhm_pos / p90, 0.0, 1.0)

    # --- stability score (agreement with smoothed) ---
    diff = np.abs(fwhms - fwhm_smoothed)
    stability_score = 1.0 - np.clip(diff / (p90 + 1e-12), 0.0, 1.0)
    stability_score = np.nan_to_num(stability_score, nan=0.0)

    # --- SNR score (saturating) ---
    noises = np.asarray(noises, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(noises > 0, amp / (noises + 1e-12), 0.0)
    snr_score = snr / (snr + s50)
    snr_score = np.clip(snr_score, 0.0, 1.0)

    # --- fit score (optional) ---
    fit_score = np.zeros(n, dtype=float)
    if fit_success is not None:
        fit_score = np.where(fit_success, 0.6, 0.0)  # success gives a stronger base
        if fit_residuals is not None:
            r = np.asarray(fit_residuals, dtype=float)
            r_med = np.nanmedian(r[np.isfinite(r)]) if np.any(np.isfinite(r)) else 1.0
            r_scale = max(1e-12, r_med * 2.0)
            fit_score = np.where(np.isfinite(r), fit_score * (1.0 / (1.0 + r / r_scale)), fit_score)

    # --- neighbor agreement (local consensus) ---
    neighbor_tol = max(1e-8, np.nanmedian(fwhm_smoothed[fwhm_smoothed>0]) * 0.5)
    neighbor_agree = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - neighbor_window)
        hi = min(n, i + neighbor_window + 1)
        window_vals = fwhm_smoothed[lo:hi]
        if np.all(np.isnan(window_vals)) or np.isnan(fwhm_smoothed[i]):
            neighbor_agree[i] = 0.0
        else:
            center = fwhm_smoothed[i]
            within = np.sum(np.abs(window_vals - center) <= neighbor_tol)
            neighbor_agree[i] = within / (hi - lo)
    stability_combined = 0.6 * stability_score + 0.4 * neighbor_agree

    # --- weighted sum ---
    total_weight = weights['amp'] + weights['width'] + weights['stability'] + weights['snr'] + weights['fit']
    raw = (weights['amp'] * amp_score +
           weights['width'] * width_score +
           weights['stability'] * stability_combined +
           weights['snr'] * snr_score +
           weights['fit'] * fit_score) / max(1e-12, total_weight)

    # --- contrast stretch so normal -> near 0 and peaks -> 1 ---
    conf = contrast_stretch(raw, low_pct=stretch_low, high_pct=stretch_high)
    return conf