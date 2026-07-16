import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def gaussian(i, A, mu, sigma):
    return A * np.exp(-(i - mu)**2 / (2 * sigma**2))
def fit_gaussian_to_intensity(intensities, threshold_factor=0.2, threshold=None):
    if len(intensities) == 0:
        return 0, 0, 0  # Return default values if no intensities are provided
    intensities = np.array(intensities, dtype=float)
    indices = np.arange(len(intensities))
    #only take values above a certain threshold to avoid noise
    if threshold is None:
        threshold = threshold_factor * np.max(intensities)  
    mask = intensities > threshold
    if np.sum(mask) == 0:
        return 0, 0, 0  # Return default values if no intensities are above the threshold
    intensities -= threshold
    intensities *= mask
    indices = indices
    # Weighted mean (center of mass)
    mu = np.sum(indices * intensities) / np.sum(intensities)

    # Weighted variance
    sigma2 = np.sum(intensities * (indices - mu)**2) / np.sum(intensities)
    sigma = np.sqrt(sigma2)

    # Amplitude (peak height)
    A = np.max(intensities)
    return A, mu, sigma


def apply_gaussian_fit_across_columns(rotated, threshold_factor=0.2, threshold=None):
    image_means = []
    As = []
    sigmas = []
    for i in range(rotated.shape[1]):
        A, mui, sigmai = fit_gaussian_to_intensity(rotated[:, i], threshold_factor, threshold=threshold)  
        image_means.append(mui)
        As.append(A)
        sigmas.append(sigmai)
    return np.array(As), np.array(image_means), np.array(sigmas)
def smooth_signal(x, kernel_size=31):
    """
    Simple 1D Gaussian-like smoothing via normalized box convolution.
    kernel_size should be odd and larger for stronger smoothing.
    """
    if kernel_size <= 1:
        return x.copy()
    k = kernel_size // 2
    kernel = np.ones(2*k+1, dtype=float)
    kernel /= kernel.sum()
    # pad reflect to avoid edge artifacts
    xp = np.pad(x, (k, k), mode='reflect')
    sm = np.convolve(xp, kernel, mode='valid')
    return sm
import numpy as np
from scipy.signal import savgol_filter

def detect_filament_global_sigma(
    img,
    A, m, s,
    use_detrend=False,
    detrend_method='poly',   # 'poly' or 'rolling'
    poly_order=2,
    rolling_win=101,
    sigma_frac_thresh=0.1,   # baseline offset; keep 0 to use strict > median
    min_segment_len=8,
    gap_close=3,
    red_ratio_thresh=1.3,
    debug=False
):
    """
    Detect broad regions where sigma is above global baseline.

    Parameters
    ----------
    img : HxW x3 image (RGB)
    A, m, s : 1D arrays length W
    use_detrend : bool remove slow global trend before thresholding
    detrend_method : 'poly' or 'rolling'
    poly_order : polynomial order for detrending
    rolling_win : window length for rolling median detrend (odd)
    sigma_frac_thresh : fractional offset above median to mark interaction (0 means > median)
    min_segment_len : minimum contiguous columns to accept a segment
    gap_close : close gaps of this many columns inside segments
    red_ratio_thresh : R/(G+eps) threshold for filament classification
    debug : print diagnostics

    Returns
    -------
    segments, labels, features, mask, diagnostics
    """

    A = np.asarray(A, dtype=float)
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    W = len(s)
    assert len(A) == W and len(m) == W

    # 1) global baseline
    sigma_global = np.median(s)

    # 2) optional detrend to remove very slow background tilt
    s_proc = s.copy()
    if use_detrend:
        if detrend_method == 'poly':
            x = np.arange(W)
            # fit polynomial to s and subtract
            coeffs = np.polyfit(x, s, poly_order)
            trend = np.polyval(coeffs, x)
            s_proc = s - trend + np.median(trend)  # keep same median level
        elif detrend_method == 'rolling':
            # rolling median detrend
            k = rolling_win if rolling_win % 2 == 1 else rolling_win + 1
            pad = k // 2
            s_pad = np.pad(s, pad, mode='reflect')
            # fast rolling median via sliding window (simple implementation)
            med = np.array([np.median(s_pad[i:i+k]) for i in range(W)])
            s_proc = s - med + np.median(med)
        else:
            raise ValueError("detrend_method must be 'poly' or 'rolling'")

    # 3) fractional deviation from global median (use original global median)
    A_sigma = (s_proc - sigma_global) / (sigma_global + 1e-9)

    # 4) initial mask: strictly greater than median plus optional fraction
    mask = A_sigma > sigma_frac_thresh

    # 5) close small gaps and remove tiny islands
    # close gaps
    if gap_close > 0:
        pad = gap_close
        kernel = np.ones(2*pad+1, dtype=int)
        mask_padded = np.pad(mask.astype(int), pad, mode='constant', constant_values=0)
        conv = np.convolve(mask_padded, kernel, mode='valid')
        mask = conv > 0

    # remove islands shorter than min_segment_len
    segments = []
    in_seg = False
    start = 0
    for i in range(W):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif not mask[i] and in_seg:
            in_seg = False
            end = i - 1
            if (end - start + 1) >= min_segment_len:
                segments.append((start, end))
    if in_seg:
        end = W - 1
        if (end - start + 1) >= min_segment_len:
            segments.append((start, end))

    # 6) refine boundaries using derivative if segments exist
    # compute smoothed derivative of s_proc to find where slope rises/falls
    if len(segments) > 0:
        # smooth s_proc lightly to compute derivative
        try:
            window = 11 if W >= 11 else (W // 2 * 2 + 1)
            s_sg = savgol_filter(s_proc, window_length=window, polyorder=2)
        except Exception:
            s_sg = s_proc
        ds = np.gradient(s_sg)

        refined_segments = []
        for lo, hi in segments:
            # expand left until derivative becomes small or boundary reached
            L = lo
            while L > 0 and ds[L] > 0:
                L -= 1
            R = hi
            while R < W-1 and ds[R] > 0:
                R += 1
            # clamp and ensure min length
            if (R - L + 1) >= min_segment_len:
                refined_segments.append((max(0, L), min(W-1, R)))
        segments = refined_segments

    # 7) classify segments using color at fitted centers
    labels = []
    features = []
    imgf = img.astype(float) if img.dtype != float else img
    H = imgf.shape[0]
    for lo, hi in segments:
        seg_len = hi - lo + 1
        mean_sigma = float(np.mean(s[lo:hi+1]))
        mean_sigma_proc = float(np.mean(s_proc[lo:hi+1]))
        frac_mean = (mean_sigma_proc - sigma_global) / (sigma_global + 1e-9)

        # color sampling at fitted centers
        reds, greens, blues = [], [], []
        for col in range(lo, hi+1):
            row = int(round(m[col]))
            row = max(0, min(H-1, row))
            R, G, B = imgf[row, col, 0], imgf[row, col, 1], imgf[row, col, 2]
            reds.append(R); greens.append(G); blues.append(B)
        if len(reds) > 0:
            Rm = float(np.mean(reds)); Gm = float(np.mean(greens)); Bm = float(np.mean(blues))
            red_ratio = Rm / (Gm + 1e-9)
        else:
            Rm = Gm = Bm = red_ratio = 0.0

        is_filament = (frac_mean > sigma_frac_thresh) and (red_ratio > red_ratio_thresh) and (seg_len >= min_segment_len)

        labels.append("filament" if is_filament else "other")
        features.append({
            "start": int(lo), "end": int(hi), "length": int(seg_len),
            "mean_sigma": mean_sigma, "mean_sigma_proc": mean_sigma_proc,
            "frac_sigma": float(frac_mean),
            "mean_R": Rm, "mean_G": Gm, "mean_B": Bm, "red_ratio": float(red_ratio)
        })

    diagnostics = {
        "sigma_global": float(sigma_global),
        "s_min": float(np.min(s)), "s_max": float(np.max(s)),
        "num_segments": len(segments)
    }

    if debug:
        print("sigma_global:", diagnostics["sigma_global"])
        print("s range:", diagnostics["s_min"], diagnostics["s_max"])
        print("A_sigma min/max:", float(np.min(A_sigma)), float(np.max(A_sigma)))
        print("initial mask sum:", int(np.sum(mask)))
        print("segments:", segments)
        for f, lab in zip(features, labels):
            print(lab, f)

    return segments, labels, features, mask, diagnostics

def detect_filament_segments(
    img,
    A, m, s,
    sigma_frac_thresh = .03,
    min_segment_len=4,
    red_ratio_thresh=1.3,
    expand_margin=2
):
    """
    Detect broad interaction regions where sigma slowly increases.

    Parameters
    ----------
    img : HxW x3 uint8 or float image (RGB)
    A, m, s : 1D arrays length W (amplitude, mean row, sigma)
    smooth_kernel : int smoothing kernel length (odd). Larger -> more emphasis on slow trends.
    sigma_frac_thresh : fractional increase above global baseline to mark interaction (e.g. 0.03 = 3%)
    min_segment_len : minimum contiguous columns to accept a segment
    red_ratio_thresh : mean R/(G+eps) threshold to prefer filament
    expand_margin : expand detected segments by this many columns on each side

    Returns
    -------
    segments : list of (start_col, end_col)
    labels : list of "filament" or "other"
    features : list of dicts with per-segment stats
    masks : boolean array length W marking detected interaction columns
    """

    # ensure numpy arrays
    A = np.asarray(A, dtype=float)
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    W = len(A)
    assert len(m) == W and len(s) == W, "A, m, s must have same length"

    # 1) global baseline and smoothed sigma
    sigma_global = np.median(s)
    # 3) initial interaction mask (broad regions where smoothed sigma exceeds threshold)
    mask = s > np.median(s)

    # 4) expand short gaps and remove tiny islands
    # simple morphological-like operations using convolution
    # expand by margin
    if expand_margin > 0:
        pad = expand_margin
        kernel = np.ones(2*pad+1, dtype=int)
        mask_padded = np.pad(mask.astype(int), pad, mode='constant', constant_values=0)
        conv = np.convolve(mask_padded, kernel, mode='valid')
        mask = conv > 0

    # remove tiny islands shorter than min_segment_len
    segments = []
    in_seg = False
    start = 0
    for i in range(W):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif not mask[i] and in_seg:
            in_seg = False
            end = i - 1
            if (end - start + 1) >= min_segment_len:
                segments.append((start, end))
    if in_seg:
        end = W - 1
        if (end - start + 1) >= min_segment_len:
            segments.append((start, end))

    # 5) classify segments using internal stats and color
    labels = []
    features = []
    imgf = img.astype(float) if img.dtype != float else img
    H = imgf.shape[0]

    for (lo, hi) in segments:
        seg_len = hi - lo + 1
        seg_s = s[lo:hi+1]
        seg_s_smooth = s[lo:hi+1]
        mean_sigma = float(np.mean(seg_s))
        mean_sigma_smooth = float(np.mean(seg_s_smooth))
        frac_mean = (mean_sigma_smooth - sigma_global) / (sigma_global + 1e-9)

        # color sampling at fitted centers (clamp rows)
        reds, greens, blues = [], [], []
        for col in range(lo, hi+1):
            row = int(round(m[col]))
            if row < 0: row = 0
            if row >= H: row = H - 1
            R, G, B = imgf[row, col, 0], imgf[row, col, 1], imgf[row, col, 2]
            reds.append(R); greens.append(G); blues.append(B)

        if len(reds) > 0:
            Rm = float(np.mean(reds))
            Gm = float(np.mean(greens))
            Bm = float(np.mean(blues))
            red_ratio = Rm / (Gm + 1e-9)
        else:
            Rm = Gm = Bm = red_ratio = 0.0

        # classification rule (tunable)
        # require: sufficiently long, fractional sigma increase, and red dominance
        is_filament = (seg_len >= min_segment_len) and (frac_mean > sigma_frac_thresh) and (red_ratio > red_ratio_thresh)

        labels.append("filament" if is_filament else "other")
        features.append({
            "start": int(lo),
            "end": int(hi),
            "length": int(seg_len),
            "mean_sigma": mean_sigma,
            "mean_sigma_smooth": mean_sigma_smooth,
            "frac_sigma_above_global": float(frac_mean),
            "mean_R": Rm,
            "mean_G": Gm,
            "mean_B": Bm,
            "red_ratio": float(red_ratio),
        })

    return segments, labels, features, mask



if __name__ == "__main__":
    img = skimage.io.imread(r"C:\Users\dhruv\Documents\dhruv_python\00015238539.jpg")
    rotated = skimage.transform.rotate(img, 106.89 - .57, resize=True)
    background = skimage.io.imread(r"C:\Users\dhruv\Documents\dhruv_python\00121700172.jpg")
    background_rotated = skimage.transform.rotate(background, 106.89 - .57, resize=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Rotated Image")
    plt.imshow(rotated)
    plt.subplot(1, 3, 2)
    plt.title("Rotated Background")
    plt.imshow(background_rotated)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    diff = cv2.absdiff(rotated, background_rotated)
    plt.imshow(diff)
    plt.show()
    #diff not working, stick with original images for now
    one_column = rotated[:, rotated.shape[1] // 2, 0]
    background_one_column = background_rotated[:, background_rotated.shape[1] // 2, 0]

    image_means = []
    image_dim = 125
    y = rotated.shape[0] // 2
    def exp_expose(x, k=5.0):
        return (1 - np.exp(-k * x)) / (1 - np.exp(-k))
    rotated_masked = exp_expose(rotated)
    _, binary = cv2.threshold(rotated_masked[:, :, 0], .9 * np.max(rotated_masked[:, :, 0]), 1, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    rotated_masked[mask == 0] = 0
    # Example usage
    one_column = rotated_masked[:, rotated_masked.shape[1] // 2, 0]
    background_one_column = background_rotated[:, background_rotated.shape[1] // 2, 0]
    A, mu, sigma = fit_gaussian_to_intensity(one_column)
    print("A =", A)
    print("mu =", mu)
    print("sigma =", sigma)
    backA, backmu, backsigma = fit_gaussian_to_intensity(background_one_column)
    print("Background A =", backA)
    print("Background mu =", backmu)
    print("Background sigma =", backsigma)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Gaussian Fit to ROTATED Image")
    plt.plot(one_column, label="Data")
    plt.plot(gaussian(np.arange(len(one_column)), A, mu, sigma), label="Gaussian Fit")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Gaussian Fit to BACKGROUND ROTATED Image")
    plt.plot(background_one_column, label="Data")
    plt.plot(gaussian(np.arange(len(background_one_column)), backA, backmu, backsigma), label="Gaussian Fit")
    plt.legend()
    plt.show()

    for i in range(rotated.shape[1]):
        Ai, mui, sigmai = fit_gaussian_to_intensity(rotated_masked[:, i, 0], 0)  
        image_means.append(mui)
    image_means = np.array(image_means)
    plt.figure(figsize=(12, 6))
    rotated_image_with_means = rotated_masked
    for i in range(rotated.shape[1]):
        cv2.line(rotated_image_with_means, (i, int(image_means[i])), (i, int(image_means[i])), (0, 1, 0), 1)
    plt.title("Rotated Image with Gaussian Means")
    plt.imshow(rotated_image_with_means)
    plt.show()
    gaussian_plots = np.array([gaussian(np.arange(rotated.shape[0]), A, image_means[i], sigma) for i in range(rotated.shape[1])])
    plt.figure(figsize=(12, 6))
    plt.title("Gaussian Fits Across Columns")
    plt.imshow(gaussian_plots, aspect='auto', extent=[0, rotated.shape[1], rotated.shape[0], 0])
    plt.colorbar(label='Intensity')
    plt.show()
    #template match background column to entire rotated image
    result = cv2.matchTemplate(rotated[:, :, :], background_one_column, cv2.TM_CCOEFF_NORMED)
    #show all matches above a certain threshold
    threshold = 0.5
    loc = np.where(result >= threshold)
    plt.figure(figsize=(12, 6))
    plt.title("Template Matching Result")
    plt.imshow(result, cmap='hot')
    plt.colorbar()
    plt.show()
    #plot locations of matches on original image
    plt.figure(figsize=(12, 6))
    plt.title("Matches on Rotated Image")
    plt.imshow(rotated)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(rotated, pt, (pt[0] + background_one_column.shape[0], pt[1] + background_one_column.shape[0]), (0, 255, 0), 2)
    plt.imshow(rotated)
    plt.show()

