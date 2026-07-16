import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares

# ---------- helpers ----------
def contrast_stretch(arr, low_pct=2.0, high_pct=98.0, eps=1e-12):
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, low_pct)
    hi = np.nanpercentile(a, high_pct)
    if hi - lo < eps:
        return np.clip((a - lo) / (eps), 0.0, 1.0)
    stretched = (a - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)

def baseline_subtract(y, window=51, polyorder=3):
    if window >= len(y):
        window = len(y) - (1 - len(y) % 2)
        window = max(window, 3)
    baseline = savgol_filter(y, window_length=window, polyorder=min(polyorder, window-1))
    y_bs = y - baseline
    noise = float(np.std(y_bs - np.median(y_bs)))
    return y_bs, baseline, noise

def numeric_fwhm(x, y):
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

# ---------- per-column width estimator ----------
def compute_width_for_column(col,
                             x_grid,
                             sat_thresh=254,
                             baseline_window=51,
                             col_smooth_sigma=1.0,
                             min_wing_points=6,
                             use_pseudo_voigt_fit=None):
    """
    Returns (consensus_width, base_confidence, sat_frac, method_used, details_dict)
    - use_pseudo_voigt_fit: callable pseudo_voigt_fit(x,y,...) or None
    """
    col = np.asarray(col, dtype=float)
    n = len(col)
    sat_pixels = col >= sat_thresh
    sat_frac = float(np.mean(sat_pixels))

    # baseline subtract + smooth
    y_bs, baseline, noise = baseline_subtract(col, window=baseline_window)
    y_s = gaussian_filter1d(y_bs, sigma=col_smooth_sigma)

    # 1) numeric FWHM (only if no saturation in central region)
    fwhm_raw = np.nan
    if not np.any(sat_pixels):
        fwhm_raw = numeric_fwhm(x_grid, y_s)

    # 2) wing fit (masked fit) if saturation present
    fwhm_wing = np.nan
    if np.any(sat_pixels):
        mask = ~sat_pixels
        if mask.sum() >= min_wing_points:
            try:
                if use_pseudo_voigt_fit is not None:
                    # call user-supplied fit; expect it to return 'fwhm_numeric' or similar
                    res = use_pseudo_voigt_fit(x_grid[mask], col[mask], p0=None, bounds=None, method='robust')
                    fwhm_wing = res.get('fwhm_numeric', np.nan)
                else:
                    # fallback: fit Gaussian to wings using simple curve_fit
                    def gauss(x, A, x0, sigma, B):
                        return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + B
                    xw = x_grid[mask]
                    yw = col[mask]
                    # initial guess
                    A0 = np.max(yw) - np.min(yw)
                    x0_0 = xw[np.argmax(yw)]
                    sigma0 = max(1.0, np.std(xw))
                    p0 = [A0, x0_0, sigma0, np.min(yw)]
                    try:
                        popt, _ = curve_fit(gauss, xw, yw, p0=p0, maxfev=10000)
                        sigma_fit = abs(popt[2])
                        fwhm_wing = 2.3548 * sigma_fit
                    except Exception:
                        fwhm_wing = np.nan
            except Exception:
                fwhm_wing = np.nan

    # 3) edge width (gradient-based)
    try:
        g = np.gradient(y_s)
        peak_idx = int(np.argmax(y_s))
        grad_thresh = 0.25 * np.nanmax(np.abs(g)) if np.nanmax(np.abs(g)) > 0 else 0.0
        left_candidates = np.where((np.arange(n) < peak_idx) & (g > grad_thresh))[0]
        right_candidates = np.where((np.arange(n) > peak_idx) & (g < -grad_thresh))[0]
        if left_candidates.size and right_candidates.size:
            left_edge = left_candidates[-1]
            right_edge = right_candidates[0]
            fwhm_edge = float(x_grid[right_edge] - x_grid[left_edge])
        else:
            fwhm_edge = np.nan
    except Exception:
        fwhm_edge = np.nan

    # 4) second moment
    try:
        ypos = y_s - np.min(y_s)
        if np.sum(ypos) > 0:
            mu = np.sum(x_grid * ypos) / np.sum(ypos)
            var = np.sum(((x_grid - mu)**2) * ypos) / np.sum(ypos)
            fwhm_moment = 2.3548 * np.sqrt(max(var, 0.0))
        else:
            fwhm_moment = np.nan
    except Exception:
        fwhm_moment = np.nan

    # combine into consensus width
    estimates = np.array([fwhm_raw, fwhm_wing, fwhm_edge, fwhm_moment], dtype=float)
    valid = ~np.isnan(estimates)
    if not np.any(valid):
        return np.nan, 0.0, sat_frac, 'none', {'raw':fwhm_raw,'wing':fwhm_wing,'edge':fwhm_edge,'moment':fwhm_moment,'noise':noise}

    # choose weights depending on saturation
    if sat_frac < 0.01:
        weights = np.array([0.55, 0.1, 0.2, 0.15])   # prefer raw numeric FWHM
    else:
        weights = np.array([0.0, 0.45, 0.35, 0.2])   # prefer wing/edge/moment when clipped

    w = weights[valid] / np.sum(weights[valid])
    consensus = float(np.nansum(w * estimates[valid]))

    # base confidence before global scaling: combine SNR, saturation penalty, and method reliability
    amp = float(np.max(col) - np.min(col))
    snr = amp / (noise + 1e-12)
    snr_score = snr / (snr + 3.0)   # s50=3 default
    sat_penalty = 1.0 - sat_frac**1.5
    # method reliability: prefer raw>wing>edge>moment
    method_idx = np.argmax(valid)  # index of first valid estimate in order raw,wing,edge,moment
    method_reliability = [1.0, 0.8, 0.9, 0.85][method_idx]
    base_conf = float(snr_score * sat_penalty * method_reliability)

    # choose method label
    method_labels = ['raw','wing','edge','moment']
    method_used = method_labels[method_idx]

    details = {'raw':fwhm_raw,'wing':fwhm_wing,'edge':fwhm_edge,'moment':fwhm_moment,'noise':noise,'snr':snr}
    return consensus, base_conf, sat_frac, method_used, details

# ---------- pipeline that uses the per-column estimator ----------
def segment_filament_with_saturation(img,
                                     baseline_window=51,
                                     col_smooth_sigma=1.0,
                                     metric_smooth_sigma=2.0,
                                     amp_gamma=2.0,
                                     confidence_threshold=0.5,
                                     min_segment_length=3,
                                     use_pseudo_voigt_fit=None):
    """
    img: H x W (grayscale) or H x W x 3 (RGB). If RGB, will prefer unsaturated channel.
    use_pseudo_voigt_fit: optional callable for wing fits (signature like pseudo_voigt_fit)
    Returns dict with widths, confidences, sat_frac, method_used, mask, segments.
    """
    # if RGB, pick best unsaturated channel per column
    if img.ndim == 3 and img.shape[2] == 3:
        # convert to per-channel arrays
        red = img[...,0].astype(float)
        green = img[...,1].astype(float)
        blue = img[...,2].astype(float)
        # choose channel per column with least saturation
        n_rows, n_cols = red.shape
        widths = np.full(n_cols, np.nan, dtype=float)
        base_conf = np.zeros(n_cols, dtype=float)
        sat_frac = np.zeros(n_cols, dtype=float)
        method_used = np.array(['']*n_cols, dtype=object)
        details_list = [None]*n_cols
        x_grid = np.arange(n_rows)
        for c in range(n_cols):
            # pick channel with lowest sat fraction
            cols = [red[:,c], green[:,c], blue[:,c]]
            sat_fracs = [np.mean(ch >= 254) for ch in cols]
            best_idx = int(np.argmin(sat_fracs))
            col = cols[best_idx]
            w, conf, sfrac, method, details = compute_width_for_column(col, x_grid,
                                                                       sat_thresh=254,
                                                                       baseline_window=baseline_window,
                                                                       col_smooth_sigma=col_smooth_sigma,
                                                                       use_pseudo_voigt_fit=use_pseudo_voigt_fit)
            widths[c] = w
            base_conf[c] = conf
            sat_frac[c] = sfrac
            method_used[c] = method
            details_list[c] = details
    else:
        # grayscale
        n_rows, n_cols = img.shape
        widths = np.full(n_cols, np.nan, dtype=float)
        base_conf = np.zeros(n_cols, dtype=float)
        sat_frac = np.zeros(n_cols, dtype=float)
        method_used = np.array(['']*n_cols, dtype=object)
        details_list = [None]*n_cols
        x_grid = np.arange(n_rows)
        for c in range(n_cols):
            col = img[:, c].astype(float)
            w, conf, sfrac, method, details = compute_width_for_column(col, x_grid,
                                                                       sat_thresh=254,
                                                                       baseline_window=baseline_window,
                                                                       col_smooth_sigma=col_smooth_sigma,
                                                                       use_pseudo_voigt_fit=use_pseudo_voigt_fit)
            widths[c] = w
            base_conf[c] = conf
            sat_frac[c] = sfrac
            method_used[c] = method
            details_list[c] = details

    # smooth widths for stability
    widths_for_smooth = np.nan_to_num(widths, nan=0.0)
    widths_smoothed = gaussian_filter1d(widths_for_smooth, sigma=metric_smooth_sigma)

    # amplitude array for final scoring
    if img.ndim == 3 and img.shape[2] == 3:
        # use max across channels
        amps = np.max(img.astype(float), axis=2).max(axis=0) - np.min(img.astype(float), axis=2).min(axis=0)
    else:
        amps = np.max(img.astype(float), axis=0) - np.min(img.astype(float), axis=0)

    # compute final raw confidence: combine base_conf with amplitude nonlinearity
    amp_min, amp_max = np.nanmin(amps), np.nanmax(amps)
    amp_range = max(1e-12, amp_max - amp_min)
    amp_norm = np.clip((amps - amp_min) / amp_range, 0.0, 1.0)
    amp_score = amp_norm ** amp_gamma
    raw_conf = amp_score * 0.6 + base_conf * 0.4   # tune weights: amplitude dominates

    # reduce confidence further by saturation fraction (stronger penalty)
    raw_conf = raw_conf * (1.0 - sat_frac**1.8)

    # contrast stretch so background ~0 and peaks ~1
    confidences = contrast_stretch(raw_conf, low_pct=2.0, high_pct=98.0)

    # threshold and extract segments
    mask = confidences >= confidence_threshold
    # remove short islands
    from scipy.ndimage import label
    labeled, ncomp = label(mask)
    final_mask = np.zeros_like(mask, dtype=bool)
    segments = []
    for lab in range(1, ncomp+1):
        inds = np.where(labeled == lab)[0]
        if inds.size >= min_segment_length:
            final_mask[inds] = True
            segments.append((int(inds[0]), int(inds[-1])))

    return {
        'widths': widths,
        'widths_smoothed': widths_smoothed,
        'confidences': confidences,
        'base_conf': base_conf,
        'sat_frac': sat_frac,
        'method_used': method_used,
        'details': details_list,
        'mask': final_mask,
        'segments': segments
    }
