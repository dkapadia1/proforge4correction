#https://pdf.sciencedirectassets.com/271963/1-s2.0-S0141635924X00069/1-s2.0-S014163592400206X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFoaCXVzLWVhc3QtMSJHMEUCIQCVM%2BSgYNQUcNo3Op7lgyuUersyPk7BctcH%2FDDs2%2FePRQIgeomcnzSQmXb1gzJCus7o01L8RrzwGHwz7NQS%2BwD0Ng4qswUIIxAFGgwwNTkwMDM1NDY4NjUiDN%2F7neTvH%2BpDFMng%2BCqQBR7DoCQM0DzF3DQNrShlCyCP2aEOQHYdWt%2Fx%2BIeLbD54TUTEVCNme5%2B6HqqQItSp1AB2EYh1owEMspSDk7heCYHNxdH0R7x%2FKxhyCcNc7TBEUlV8DR0n2aXiAb2cHg1ML4hzd8ry6BwlgJNa01bWf94Rbpc679fHFbJ341guF9RaLkvinV%2FKCtEDQskanD8Q3LPjaeVvJm0FS%2FE%2BjUULTmE45mOQbKi%2BNWTdVEKG8LitNj4nUX7BtgjZtHKjDaV39otkNjaR3Cw1v4BXOKC56P4GDhaF0hWy5VntkJZaEhZK%2BYVLdMnv8Em8zonMZ3S%2BOrOsEqbKon%2Fuh%2Bztit5FurzTrunwBaE%2F5lhkViZnX72i3ycLGApSsJIHQFUuPhwl5QsCa2%2BF0XZEeMHgc4CdT1fIVG85%2B%2BeJoV1EnHNt4gN887Z8Qe52VyeXOrNh3xw9ieKmezjbVWxEkVVmFTccJsLEWs3fyuidhBJUvsaODaYKh8ONzOmLMxjo%2Bekr%2BT7HWAD4K8Y8Sxb6sbPZMBB7iCvyGtM5JMqSwqqOkna%2B2pG0ztF0%2BQ9WXXv3c02sdttt1ICL97A9R3j8ukeX%2B%2F20NJxOYW4WSn%2BXbfG3StLiVemj9S288KUbFDSKQ%2FJ%2BnsEcQjg7jo3%2BGCAwyLYQOE8PODqxc691GCItzTj%2BlR7SUbxj5U%2FzqzRFFnLKzQj4%2BYMhapErU%2FCIWNIzm88RGAr2R1ebIEJDbiLvcCNM9K6fVm3yjqR3R9yel9v9d1jFFLIwna41t9efdHuLO8is78KCQ7PNkYT96%2FVU4hdpSIr%2BTQJmvmwYNfbZsUVRQZJb6l2HPdMedQ9D52i5c0c%2FmxoBmfIVRkZQaONCJRcSFYPOOmUHMNrU%2BtAGOrEBWODDKLDrHcAUWHOh2A%2B7Bdp2HvnLhql9TqdQlCUo5J0cws93BGXc1DG0zA%2B37u%2FAdwsiw0KmbgVnC%2BPpsO%2BjJpDKG%2BxkUAdeaz2QZLRnR3wv%2FrnpPbnoox7DzlJue0MCfQNazpYiNg8wX3kV1sBQaA4acLmACNY04Pt2uwAGazYDTvmha9zn9kSZVr4W%2BKjbAajF3Z5%2Biwo4zZtErAqJL13e9LaulYZvC2WIlh6wobRS&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260602T114709Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYW57MZXLR%2F20260602%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b7b870aa76a3aadc95308b4960e8af81a1033cf95a573d7bd0102f0df10ee6a3&hash=5bf4c52965114c7136bec5051144bbd56fd1d7b119e2bc1511f3bd55fa8d43f8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S014163592400206X&tid=spdf-26cc8765-bbfd-4eb0-aa1a-4a8b562c7a93&sid=7db2ed2f778752484858663997a50ba1aeaagxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0015055256505358000b&rr=a05638bd98330cd3&cc=fr


import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d, find_peaks
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from skimage.morphology import reconstruction
from matplotlib.widgets import Slider
from analyze_five import fit_gaussian_to_intensity, gaussian, apply_gaussian_fit_across_columns, detect_filament_segments, detect_filament_global_sigma
from analyze_sevon import pseudo_voigt_fit, apply_psuedo_voigt_fit_across_columns
from numeric_segment import segment_filament_with_confidence, compute_confidences_scaled
from contrast_segmentation import segment_filament_with_saturation
import time
import os
import re
# -------------------------
# Utility helpers
# -------------------------
def to_gray(img):
    if img.ndim == 3:
        #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img[:,:,2]
    return img.copy()

def visualize_column_scroll(gray, out_x, out_y, gaussian_pack = None, voigt_pack = None):
    """
    Interactive viewer for scrolling through image columns and plotting intensity.

    gray  : 2D grayscale image (H x W)
    out_x : array of x-coordinates of detected points
    out_y : array of y-coordinates of detected points
    """

    # --- initial column ---
    col = gray.shape[1] // 2
    one_column = gray[:, col]
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)

    # main intensity plot
    (line,) = ax.plot(one_column, 'k-', label='column intensity')
    xs = np.arange(len(one_column))
    if gaussian_pack is None:
        A, mui, sigmai = fit_gaussian_to_intensity(one_column, .6)    
    else:
        A, mui, sigmai = gaussian_pack[0][col], gaussian_pack[1][col], gaussian_pack[2][col]
    gaussian_fit = gaussian(xs, A, mui, sigmai)
    (gaussian_plot,) = ax.plot(xs, gaussian_fit, 'b-', label='gaussian fit')
    # centerline point
    mask = (out_x == col)
    if np.any(mask):
        y = out_y[mask][0]
        (point_plot,) = ax.plot(y, one_column[int(y)], 'ro', label='centerline point')
    else:
        point_plot = ax.plot([], [], 'ro')[0]
    if voigt_pack is None:
        res = pseudo_voigt_fit(xs, one_column, method='robust', robust_loss='huber', robust_f_scale=3.0)
        voigt_fit = res['model'](xs)
    else:  
        res = voigt_pack[0][col]
        voigt_fit = voigt_pack[3][col](xs)
    (voigt_plot,) = ax.plot(xs, voigt_fit, 'r-', label="voigt fit")
        
    ax.set_title(f"{sigmai}")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.axhline(.8 * np.max(gray))
    # --- slider ---
    axcol = plt.axes([0.15, 0.1, 0.7, 0.05])
    slider = Slider(axcol, 'Column', 0, gray.shape[1]-1, valinit=col, valstep=1)

    # --- update callback ---
    def update(val):
        c = int(slider.val)
        col_data = gray[:, c]
        line.set_ydata(col_data)
        xs = np.arange(len(col_data))
        if gaussian_pack is None:
            A, mui, sigmai = fit_gaussian_to_intensity(col_data, .6)    
        else:
            A, mui, sigmai = gaussian_pack[0][c], gaussian_pack[1][c], gaussian_pack[2][c]
        gaussian_fit = gaussian(xs, A, mui, sigmai)
        gaussian_plot.set_data(xs, gaussian_fit)
        if voigt_pack is None:
            res = pseudo_voigt_fit(xs, col_data, method='robust', robust_loss='huber', robust_f_scale=3.0)
            voigt_fit = res['model'](xs)
        else:  
            res = voigt_pack[0][c]
            voigt_fit = voigt_pack[3][c](xs)
        voigt_plot.set_data(xs, voigt_fit)
        ax.set_title(f"{sigmai}")
        mask = (out_x == c)
        if np.any(mask):
            y = out_y[mask][0]
            point_plot.set_data([y], [col_data[int(y)]])
        else:
            point_plot.set_data([], [])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def median_and_binary(img_gray, med_ksize=3, bin_thresh=None):
    med = cv2.medianBlur(img_gray, med_ksize)
    if bin_thresh is None:
        bin_thresh = threshold_otsu(med)
    _, bw = cv2.threshold(med, bin_thresh, 255, cv2.THRESH_BINARY)
    bw = bw.astype(bool)
    # remove tiny objects
    bw = remove_small_objects(bw, min_size=8)
    return med, bw.astype(np.uint8) * 255

# -------------------------
# Column-wise linewidth estimation
# -------------------------
def column_linewidths(binary_stripe):
    h, w = binary_stripe.shape
    widths = np.zeros(w, dtype=np.int32)
    for j in range(w):
        col = binary_stripe[:, j] > 0
        if not col.any():
            widths[j] = 0
            continue
        top = np.argmax(col)
        bottom = h - 1 - np.argmax(col[::-1])
        widths[j] = max(0, bottom - top + 1)
    return widths

# -------------------------
# Region separation into A, B, C
# -------------------------
def region_separation(widths):
    # remove zeros and extremes
    nonzero = widths[widths > 0]
    if nonzero.size == 0:
        return np.zeros_like(widths, dtype=np.uint8)  # all background
    # drop lowest and highest 5%
    lo = np.percentile(nonzero, 5)
    hi = np.percentile(nonzero, 95)
    trimmed = nonzero[(nonzero >= lo) & (nonzero <= hi)]
    if trimmed.size == 0:
        wom = np.median(nonzero)
    else:
        wom = threshold_otsu(trimmed) if trimmed.size > 1 else trimmed.mean()
    # initial split X and Y by wom
    X_idx = widths <= wom
    Y_idx = widths > wom
    wpX = widths[X_idx].mean() if X_idx.any() else 0
    wpY = widths[Y_idx].mean() if Y_idx.any() else 0
    sigmaX = widths[X_idx].std() if X_idx.any() else 1.0
    sigmaY = widths[Y_idx].std() if Y_idx.any() else 1.0
    # compute wl1 and wl2 per paper eq (3)
    denom = (sigmaX + sigmaY) if (sigmaX + sigmaY) != 0 else 1.0
    wl1 = (sigmaX / denom) * wom + (sigmaY / denom) * wpX + sigmaX
    wl2 = (sigmaY / denom) * wom + (sigmaX / denom) * wpY + sigmaY
    # assign regions A (small), B (large), C (middle)
    regions = np.full_like(widths, fill_value=2, dtype=np.uint8)  # default C=2
    # region A if width <= wl1 for column and next 3 columns
    for i in range(len(widths)):
        window = widths[i:i+4]
        if window.size == 4 and np.all(window <= wl1):
            regions[i] = 0  # A
    # region B if width > wl2 for column and next 3 columns
    for i in range(len(widths)):
        window = widths[i:i+4]
        if window.size == 4 and np.all(window > wl2):
            regions[i] = 1  # B
    # remaining are C (2)
    return regions

# -------------------------
# MRSAC template creation
# -------------------------
def make_mrsac_template_regionA(widths_region):
    # remove zeros
    arr = widths_region[widths_region > 0]
    if arr.size == 0:
        return np.ones((3,3)) / 9.0
    wvA = int(np.round(arr.mean()))
    wmA = int(arr.max())
    RA = wmA if wmA % 2 == 1 else wmA + 1
    LA = int(np.ceil(1.75 * (wmA - wvA))) if wmA - wvA > 0 else 3
    RA = max(RA, 3)
    LA = max(LA, 3)
    # crucial region size
    RAp = wvA if wvA % 2 == 1 else wvA + 1
    LAp = int(np.ceil(1.25 * (wmA - wvA))) if wmA - wvA > 0 else 1
    # build kernel
    k = 1.0
    lam = 0.4
    kernel = np.full((RA, LA), lam * k, dtype=np.float32)
    # place crucial region centered
    r0 = (RA - RAp) // 2
    c0 = (LA - LAp) // 2
    kernel[r0:r0+RAp, c0:c0+LAp] = k
    kernel /= kernel.sum()
    return kernel
# -------------------------
# FIX 1: Cap kernel lateral size in region B/C
# Large LB is the main culprit for bump smoothing
# -------------------------
def make_mrsac_template_regionB(widths_region, max_lateral=20):  # add max_lateral cap
    arr = widths_region[widths_region > 0]
    if arr.size == 0:
        return np.ones((3,3)) / 9.0
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    low = int(n * 0.125)
    high = int(n * 0.875)
    arr_trim = arr_sorted[low:high] if high > low else arr_sorted
    wnB = int(arr_trim.min())
    wmB = int(arr_trim.max())
    RB = wmB if wmB % 2 == 1 else wmB + 1
    # KEY CHANGE: cap LB so lateral spread doesn't blur small bumps
    LB = max_lateral
    RBp = wnB if wnB % 2 == 1 else wnB + 1
    lam = 0.1
    k = 1.0
    kernel = np.full((RB, LB), lam * k, dtype=np.float32)
    r0 = (RB - RBp) // 2
    kernel[r0:r0+RBp, :] = k
    kernel /= kernel.sum()
    return kernel

# -------------------------
# Convolution and stitching
# -------------------------
def apply_mrsac_and_stitch(img_gray, regions, widths):
    h, w = img_gray.shape
    # build kernels per region using widths subset
    A_mask = regions == 0
    B_mask = regions == 1
    C_mask = regions == 2
    # kernels
    kernelA = make_mrsac_template_regionA(widths[A_mask]) if A_mask.any() else np.ones((3,3))/9.0
    kernelB = make_mrsac_template_regionB(widths[B_mask]) if B_mask.any() else np.ones((3,3))/9.0
    kernelC = make_mrsac_template_regionB(widths[C_mask]) if C_mask.any() else np.ones((3,3))/9.0
    # convolve full image with each kernel
    IA = convolve2d(img_gray.astype(np.float32), kernelA, mode='same', boundary='symm')
    IB = convolve2d(img_gray.astype(np.float32), kernelB, mode='same', boundary='symm')
    IC = convolve2d(img_gray.astype(np.float32), kernelC, mode='same', boundary='symm')
    # stitch: for each column choose region's convolved column
    I_stitched = np.zeros_like(img_gray, dtype=np.float32)
    for j in range(w):
        if regions[j] == 0:
            I_stitched[:, j] = IA[:, j]
        elif regions[j] == 1:
            I_stitched[:, j] = IB[:, j]
        else:
            I_stitched[:, j] = IC[:, j]
    return I_stitched, (kernelA, kernelB, kernelC)
# -------------------------
# Hessian eigen and subpixel center calculation
# -------------------------
def compute_lambda_max_and_normals(Ixx, Iyy, Ixy):
    # compute eigenvalues and eigenvectors for 2x2 Hessian per pixel
    # lambda1, lambda2
    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    # eigenvalues:
    temp = np.sqrt(np.maximum(0.0, (Ixx - Iyy)**2 + 4.0 * Ixy**2))
    lam1 = 0.5 * (tr + temp)
    lam2 = 0.5 * (tr - temp)
    # choose lambda with larger absolute value
    lam_abs1 = np.abs(lam1)
    lam_abs2 = np.abs(lam2)
    lammax = np.where(lam_abs1 >= lam_abs2, lam1, lam2)
    # eigenvector for lammax: solve (Ixx - lam) * nx + Ixy * ny = 0
    # choose nx = Ixy, ny = lam - Ixx (normalized)
    nx = Ixy.copy()
    ny = lammax - Ixx
    norm = np.sqrt(nx*nx + ny*ny) + 1e-12
    nx /= norm
    ny /= norm
    return lammax, nx, ny
def compute_stitched_derivatives(stitched, regions, sigma_a, sigma_b, sigma_c):
    """Compute Hessian derivatives region-wise and stitch, matching the
    convolution stitching approach already used for intensity."""

    def derivs(img, sigma):
        sigma_x = sigma * .7  # less smoothing for derivatives
        sigma_y = sigma * 1.3
        sigma = (sigma_y, sigma_x)  # anisotropic smoothing: more along rows
        Ix  = ndimage.gaussian_filter(img, sigma=sigma, order=[0,1])
        Iy  = ndimage.gaussian_filter(img, sigma=sigma, order=[1,0])
        Ixx = ndimage.gaussian_filter(img, sigma=sigma, order=[0,2])
        Iyy = ndimage.gaussian_filter(img, sigma=sigma, order=[2,0])
        Ixy = ndimage.gaussian_filter(img, sigma=sigma, order=[1,1])
        return Ix, Iy, Ixx, Iyy, Ixy

    dA = derivs(stitched, sigma_a)
    dB = derivs(stitched, sigma_b)
    dC = derivs(stitched, sigma_c)

    h, w = stitched.shape
    # pre-allocate
    Ix  = np.empty((h, w), dtype=np.float32)
    Iy  = np.empty_like(Ix)
    Ixx = np.empty_like(Ix)
    Iyy = np.empty_like(Ix)
    Ixy = np.empty_like(Ix)

    for j in range(w):
        src = dA if regions[j] == 0 else (dB if regions[j] == 1 else dC)
        Ix[:, j]  = src[0][:, j]
        Iy[:, j]  = src[1][:, j]
        Ixx[:, j] = src[2][:, j]
        Iyy[:, j] = src[3][:, j]
        Ixy[:, j] = src[4][:, j]

    return Ix, Iy, Ixx, Iyy, Ixy
def smooth_centerline(xs, ys, method='savgol', window_length=7, polyorder=2):
    """
    method='none'   -> raw subpixel points, no smoothing at all
    method='savgol' -> Savitzky-Golay: preserves peak shape much better than PCHIP
    method='pchip'  -> original behaviour (avoid for bump measurement)
    """
    if xs.size < 2:
        return xs, ys

    # fill any missing columns first (linear gap fill)
    all_x = np.arange(int(xs.min()), int(xs.max()) + 1)
    all_y = np.interp(all_x, xs, ys)   # linear interp for gaps only

    if method == 'none':
        return all_x.astype(float), all_y

    if method == 'savgol':
        # window_length must be odd and < len(all_y)
        wl = min(window_length, len(all_y) if len(all_y) % 2 == 1 else len(all_y) - 1)
        wl = max(wl, polyorder + 2)
        if wl % 2 == 0:
            wl += 1
        smoothed = savgol_filter(all_y, window_length=wl, polyorder=polyorder)
        return all_x.astype(float), smoothed

    # fallback: original PCHIP
    from scipy.interpolate import PchipInterpolator
    uniq_x, idx = np.unique(xs, return_index=True)
    pchip = PchipInterpolator(uniq_x, ys[idx])
    return all_x.astype(float), pchip(all_x)
# -------------------------
# Full pipeline function
# -------------------------
# -------------------------
# Updated pipeline  (drop-in replacement for extract_centerline_from_image)
# -------------------------

def binary_stripe_centers_per_column(bw):
    """
    Compute the midpoint of the binary stripe per column.
    This is derived from the raw thresholded image, not the convolution,
    so it correctly follows physical stripe displacement at bumps.
    Returns a dict: col -> float row center
    """
    h, w = bw.shape
    centers = {}
    for j in range(w):
        col = bw[:, j] > 0
        if not col.any():
            continue
        top    = np.argmax(col)
        bottom = h - 1 - np.argmax(col[::-1])

        centers[j] = (top + bottom) / 2.0
    return centers


def subpixel_centers_guided(img, Ix, Iy, Ixx, Iyy, Ixy, lammax, nx, ny,
                             binary_centers, search_radius=4):
    """
    Drop-in replacement for subpixel_centers.

    Instead of searching rows above the 50th percentile of the stitched
    image (which gets misled by MRSAC smearing), we search a tight window
    around the binary-mask stripe centre. The binary mask tracks physical
    stripe position correctly at bumps.

    search_radius: how many rows above/below the binary centre to search.
    Set to half the maximum expected bump displacement in pixels. If your
    stripe is ~10px wide, 4-6 rows is usually right.
    """
    h, w = img.shape
    centers = []

    for j in range(w):
        if j not in binary_centers:
            continue

        approx_row = binary_centers[j]
        r_lo = max(0,   int(np.floor(approx_row)) - search_radius)
        r_hi = min(h-1, int(np.ceil (approx_row)) + search_radius)

        candidates = []
        for i in range(r_lo, r_hi + 1):
            nx_ = nx[i, j];  ny_ = ny[i, j]
            denom = (nx_*nx_*Ixx[i,j]
                   + 2*nx_*ny_*Ixy[i,j]
                   + ny_*ny_*Iyy[i,j])
            if np.abs(denom) < 1e-8:
                continue
            numer = nx_*Ix[i,j] + ny_*Iy[i,j]
            t = numer / denom
            if -0.5 <= t <= 0.5:
                sub_r = i + t * ny_
                candidates.append((j, sub_r, lammax[i, j], i))

        if candidates:
            # pick strongest ridge response in the window
            best = max(candidates, key=lambda c: abs(c[2]))
            centers.append(best)
        else:
            # Hessian found nothing clean: fall back to binary midpoint
            # (integer precision, but at least it's at the right place)
            centers.append((j, approx_row, 0.0, int(round(approx_row))))

    return centers


def denoise_and_select_guided(centers):
    """
    Simplified selection for the guided case.
    Since subpixel_centers_guided already restricts to the correct stripe
    location, we just pick the max-|lambda| candidate per column with no
    row-proximity gating (the binary guide already enforces proximity).
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for col, rsub, lam, irow in centers:
        grouped[col].append((rsub, lam, irow))

    xs, ys = [], []
    for col in sorted(grouped.keys()):
        items = grouped[col]
        best  = max(items, key=lambda it: abs(it[1]))
        xs.append(col)
        ys.append(best[0])

    return np.array(xs, dtype=float), np.array(ys, dtype=float)
def preprocess_image(IMAGE_PATH, degree = 16.5):
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 270 + degree, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    center = (img.shape[1] // 2, img.shape[0] // 2)
    #rotate 16.5 degrees counterclockwise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
    mag = np.abs(gx)
    vertical_energy = mag.mean(axis=1)
    peak = np.argmax(vertical_energy)
    band_top = max(0, peak - 40)
    band_bottom = min(gray.shape[0], peak + 40)
    width = 250
    roi = gray[band_top:band_bottom, :]
    gx_roi = cv2.Sobel(roi, cv2.CV_32F, dx=0, dy=1, ksize=3)
    mag_roi = np.abs(gx_roi)
    horizontal_energy = mag_roi.mean(axis=0)
    th = np.percentile(horizontal_energy, 70)
    mask = horizontal_energy > th
    cols = np.where(mask)[0]
    left = max(0, cols[0] + 5)
    right = min(gray.shape[1], cols[-1] - 5)
    roi = img[band_top:band_bottom, left:right]
    #crop to region of interest if needed
    #morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    seed = cv2.erode(roi, kernel)
    # result = reconstruction(seed, roi, method='erosion')
    # print(np.sum(result-roi))
    return roi
def laser_template_matching(img, background):
    gray_img = to_gray(img)
    gray_bg = to_gray(background)
    laser = np.where(gray_bg.mean(axis=1) > 100)  # Threshold to find laser pixels
    if(len(laser[0]) == 0):
        print("No laser pixels found. Try adjusting the threshold.")
        return None
    laser_slice = slice(laser[0][0], laser[0][-1])
    # print(f"Laser slice: {laser_slice}")
    distro = gray_bg[laser_slice].mean(axis=1).astype(np.uint8)
    distro = distro[:, np.newaxis]
    result = cv2.matchTemplate(gray_img, distro, cv2.TM_CCOEFF_NORMED)
    line = np.argmax(result, axis=0) + (laser_slice.stop - laser_slice.start) // 2
    return line
def zhao_hessian_method(img_gray, med_ksize, search_radius, smooth_method, savgol_window):
    med, bw = median_and_binary(img_gray.astype(np.uint8), med_ksize=med_ksize)
    widths  = column_linewidths(bw)
    regions = region_separation(widths)
    stitched, _ = apply_mrsac_and_stitch(img_gray, regions, widths)
    # per-region sigmas
    def mean_width(r):
        w = widths[regions == r]
        return int(np.round(w.mean())) if w.size else 3
    wpa, wpb, wpc = mean_width(0), mean_width(1), mean_width(2)
    wpa = wpa + 1 if wpa % 2 == 0 else max(3, wpa)
    wpb = wpb + 1 if wpb % 2 == 0 else max(3, wpb)
    wpc = wpc + 1 if wpc % 2 == 0 else max(3, wpc)
    sigma_a = max(0.8, np.ceil(wpa / 3**0.5))
    sigma_b = max(0.8, np.ceil(wpb / 3**0.5))
    sigma_c = max(0.8, np.ceil(wpc / 3**0.5))
    Ix, Iy, Ixx, Iyy, Ixy = compute_stitched_derivatives(
        stitched, regions, sigma_a, sigma_b, sigma_c)
    lammax, nx, ny = compute_lambda_max_and_normals(Ixx, Iyy, Ixy)
    centers = subpixel_centers_guided(
        stitched, Ix, Iy, Ixx, Iyy, Ixy, lammax, nx, ny,
        binary_stripe_centers_per_column(bw), search_radius=search_radius)
    xs, ys = denoise_and_select_guided(centers)
    out_x, out_y = smooth_centerline(xs, ys, method=smooth_method,
                                     window_length=savgol_window)
    return xs, ys, out_x, out_y
def maxes(gray):
    max_xs = np.arange(gray.shape[1])
    max_ys = np.array([np.argmax(gray[:, j]) for j in max_xs])
    return max_xs, max_ys
def distro_based_filament_extraction(img_bgr):
    gray = to_gray(img_bgr)
    threshold = .8 * np.max(gray) #threshold_otsu(gray.astype(np.uint8)/ 255) * .5
    # print(threshold)
    Amps, gaussian_means, sigmas = apply_gaussian_fit_across_columns(gray.astype(np.uint8), threshold = threshold)
    segments, labels, features, mask, diagnostics = detect_filament_global_sigma(img_bgr[:, :, ::-1], Amps, gaussian_means, sigmas, debug=False)
    pack = zip(segments, labels, features, mask)
    # print(pack
    def z_mean(v):
        v = v.astype(np.float32)
        r =  (v - np.mean(v)) / (np.std(v) + 1e-6)
        return ((r - r.min()) / (r.max() - r.min()) * 255).astype(np.uint8)
    amp_z = z_mean(Amps)
    sigma_z = z_mean(sigmas)
    # print(amp_z, sigma_z)
    def get_mask(v):
        _, mask = cv2.threshold(
            v.reshape(-1, 1),
            0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        return mask.flatten().astype(bool)
    mask_a = Amps > threshold * 100
    mask_s = get_mask(sigma_z)
    mask = mask_a & mask_s
    idx = np.where(mask)[0]
    gaussian_pack = (Amps, gaussian_means, sigmas)
    region_pack = (segments, labels, features, mask, diagnostics)
    # print("myballs", gaussian_pack, pack)
    return gaussian_pack, pack
import numpy as np


def filament_confidence_skew(
    img_bgr: np.ndarray,
    laser_y: int = 22,
    window_radius: int = 20,
    skew_threshold: float = 0.5,
    skew_full: float = 2.0,
    debug=True,
) -> np.ndarray:
    """
    Returns confidence (0-1) per column that filament is present.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    laser_y : int
        Expected laser centerline when no filament is present.
    window_radius : int
        Vertical pixels above/below laser_y to analyze.
    skew_threshold : float
        Skew magnitude below this gives confidence 0.
    skew_full : float
        Skew magnitude above this gives confidence 1.

    Returns
    -------
    confidence : np.ndarray
        Shape (width,), float32 in [0,1]

    """
    # Use red channel since laser is red
    red = img_bgr[:, :, 2].astype(np.float32)
    h, w = red.shape
    y0 = laser_y - window_radius
    y1 = laser_y + window_radius
    ys = np.arange(y0, y1, dtype=np.float32)
    confidence = np.zeros(w, dtype=np.float32)
    skews = np.zeros(w, dtype=np.float32)
    for x in range(w):
        profile = red[y0:y1, x]
        total = profile.sum()
        if total < 1:
            continue
        # intensity-weighted mean
        mu = np.sum(profile * ys) / total
        # intensity-weighted std
        var = np.sum(profile * (ys - mu) ** 2) / total
        if var < 1e-6:
            continue
        sigma = np.sqrt(var)
        # intensity-weighted skewness
        skew = (
            np.sum(profile * (ys - mu) ** 3) / total
        ) / (sigma ** 3)
        skews[x] = skew
        skew_mag = skew
        conf = (
            (skew_mag - skew_threshold)
            / (skew_full - skew_threshold)
        )
        confidence[x] = np.clip(conf, 0.0, 1.0)
    skews = np.array(skews)
    grad = savgol_filter(skews, window_length=20, polyorder=4,  deriv=1)
    confidence_weighted_grad = grad * confidence ** 2
    grad = confidence_weighted_grad
    n = len(grad)
    x_idx = np.arange(n)
    center = (n - 1) / 2
    sigma = 50 / np.sqrt(2 * np.log(2))
    weights = np.exp(
        -0.5 * ((x_idx - center) / sigma) ** 2
    )
    weighted_grad = weights * grad
    left_grad = np.argmax(weighted_grad)
    right_grad = left_grad + np.argmin(grad[left_grad:])
    region = grad[left_grad:right_grad]
    x = np.linspace(-1.0, 1.0, len(region))
    y = region - np.mean(region) 
    y*=128
    A = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3
    ])

    a, a1, a2, a3 = np.linalg.lstsq(A, y, rcond=None)[0]
    fit = a + a1 * x + a3 * x**3 + a2 * x**2
    # split confidence
    split_strength = a1 / (
        abs(a3) + 1e-8
    )
    split_confidence = np.clip(
        split_strength / .5,
        0,
        1
    )
    if(debug):
        print(a1, a3)
    return confidence, skews, left_grad, right_grad, grad, split_confidence, fit, (x + 1) * len(region) / 2 + left_grad
def is_split_skew_model(path):
    img = preprocess_image(path)
    return filament_confidence_skew(img_bgr=img, skew_threshold=0, laser_y=43, skew_full=.2, debug=False)
def extract_centerline_from_image(path, med_ksize=3, search_radius=4,
                                  smooth_method='savgol', savgol_window=20,
                                  visualize=False):
    times = ["preprocess", "hessian","maxing", "gaussian", "template", 'voigt', 'numeric']
    time_dict = {}
    time_dict[times[0]] = time.time()
    img_bgr = preprocess_image(path)
    background = preprocess_image(path)
    gray = to_gray(img_bgr).astype(np.float32)
    time_dict[times[0]] =  time.time() - time_dict[times[0]]
    time_dict[times[1]] = time.time()
    xs, ys, out_x, out_y = zhao_hessian_method(img_gray=gray, med_ksize=med_ksize, search_radius=search_radius,
                                              smooth_method=smooth_method, savgol_window=savgol_window)
    time_dict[times[1]] =  time.time() - time_dict[times[1]]
    time_dict[times[2]] = time.time()
    max_xs, max_ys = maxes(gray)
    time_dict[times[2]] = time.time() - time_dict[times[2]]
    time_dict[times[3]] = time.time()
    gaussian_pack, pack = distro_based_filament_extraction(img_bgr)
    gaussian_means = gaussian_pack[1]
    time_dict[times[3]] = time.time() - time_dict[times[3]]
    time_dict[times[4]] = time.time()
    template_centers = laser_template_matching(img_bgr, background)
    time_dict[times[4]] = time.time() - time_dict[times[4]]
    time_dict[times[5]] = time.time()
    voigt_pack = apply_psuedo_voigt_fit_across_columns(gray)
    voigt_means = voigt_pack[2][:, 1]
    params = voigt_pack[2]
    time_dict[times[5]] = time.time() - time_dict[times[5]]
    time_dict[times[6]] = time.time()
    res = segment_filament_with_confidence(gray)
    time_dict[times[6]] = time.time() - time_dict[times[6]]
    print(time_dict)
    confidence, skews, left_grad, right_grad, grad, split_confidence, fit, x = filament_confidence_skew(img_bgr=img_bgr, skew_threshold=0, laser_y=43, skew_full=.2)

    if visualize:
        # also show other methods of center estimation for comparison
        plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1)
        plt.imshow(img_bgr[:,:,::-1])
        plt.plot(xs,     ys,     'b.', markersize=3, label='subpixel detected')
        plt.plot(out_x,  out_y,  'c-', linewidth=1.2, label=f'centerline ({smooth_method})')
        # plt.plot(bcols,  brows,  'y.', markersize=2, label='binary midpoint (guide)')
        plt.plot(max_xs, max_ys, 'm.', markersize=1, label='max intensity (noisy)')
        plt.plot(gaussian_means, 'g-', linewidth=1.2, label='gaussian means')
        plt.plot(voigt_means, 'b-', linewidth=1.2, label='voigt means')
        
        for (seg, lab, feat, mask) in pack:
            color = 'blue' if lab == 'filament' else 'green'
            plt.axvline(seg[0], color = color)
            plt.axvline(seg[1], color = color)
        plt.axvline(left_grad, color = "yellow")
        plt.axvline(right_grad, color = "orange")
        # plt.plot(template_centers, 'b.', markersize=3, label='template matching')
        #move the legend outside the plot area to avoid covering the image
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(path)
        plt.plot(gaussian_pack[2])
        plt.figure(figsize=(10, 5))
        plt.title(f"{split_confidence}")
        plt.plot(confidence)
        plt.plot(skews)
        plt.plot(grad * 128)
        plt.plot(x, fit)
        # plt.show()
        # print(time_dict)
        visualize_column_scroll(gray, out_x, out_y, gaussian_pack=gaussian_pack, voigt_pack=None)  # interactive viewer for column intensities and centerline points
        # plt.show()
        # plt.figure()
        # plt.plot(sigmas)
        # plt.show()
    return xs, ys, out_x, out_y
# -------------------------
# Save CSV
# -------------------------
def save_centerline_csv(x, y, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y'])
        for xi, yi in zip(x, y):
            writer.writerow([float(xi), float(yi)])
def max_deviation_from_centerline(path, med_ksize=3, search_radius=4,
                                  smooth_method='savgol', savgol_window=7,
                                  visualize=False):
    xs, ys, out_x, out_y = extract_centerline_from_image(path, med_ksize, search_radius, smooth_method, savgol_window, visualize)
    edge_length = 20
    left = out_x[0:edge_length], out_y[0:edge_length]
    right = out_x[-edge_length:], out_y[-edge_length:]
    edge = np.concatenate([left, right], axis=1)
    line_of_best_fit = np.poly1d(np.polyfit(edge[0], edge[1], deg=1))
    deviations = np.abs(out_y - line_of_best_fit(out_x))
    max_dev = np.max(deviations)
    if visualize and False:
        # plt.subplot(1, 3, 3)
        plt.figure(figsize=(12, 6))
        plt.plot(out_x, out_y, 'b-', label='centerline')
        plt.plot(out_x, line_of_best_fit(out_x), 'r--', label='line of best fit')
        plt.title(f'Max deviation from centerline: {max_dev:.2f} pixels')
        plt.show()
    return max_dev

def number(p):
    match = re.search(r'\d+', p)
    return int(match.group()) if match else -1
def plot_1d_heatmap(dists):
    mat = np.stack(dists)   # shape (num_dists, N)
    plt.imshow(mat, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Index")
    plt.ylabel("Distribution #")
    plt.show()
if __name__ == "__main__":
    import random
    folder = r"C:\Users\dhruv\Documents\dhruv_python\frames\\"
    photos = sorted(os.listdir(folder), key = number)
    random.shuffle(photos)
    allsigs = []
    for p in photos[285:330]:
        INPUT_PATH = folder + p
        max_dev = max_deviation_from_centerline(
            INPUT_PATH,
            med_ksize=11,
            search_radius=4,
            smooth_method='savgol',
            savgol_window=7,
            visualize=True
        )
        print(f"Max deviation from centerline: {max_dev:.2f} pixels")

            
