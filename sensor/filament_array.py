import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# from analyze_six import preprocess_image, to_gray
import re
import os
import random
from numpy.lib.stride_tricks import sliding_window_view
import cv2
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img[:,:,2]
    return img.copy()
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
def get_preprocess_crop_info(image_path, degree=26.2):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, 270 + degree, 1.0)
    Minv = cv2.invertAffineTransform(M)

    rot = cv2.warpAffine(img, M, (W, H))

    gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    gy = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=5)

    vertical_energy = np.abs(gy).mean(axis=1)
    peak = np.argmax(vertical_energy)

    band_top = max(0, peak - 40)
    band_bottom = min(H, peak + 40)

    roi_gray = gray[band_top:band_bottom, :]
    gy_roi = cv2.Sobel(roi_gray, cv2.CV_32F, dx=0, dy=1, ksize=3)

    horizontal_energy = np.abs(gy_roi).mean(axis=0)
    th = np.percentile(horizontal_energy, 70)

    cols = np.where(horizontal_energy > th)[0]
    left = max(0, cols[0] + 5)
    right = min(W, cols[-1] - 5)

    return {
        "original_shape": (H, W),
        "rotated_shape": (H, W),
        "M": M,
        "Minv": Minv,
        "crop": (band_top, band_bottom, left, right),
        "roi": rot[band_top:band_bottom, left:right],
        'peak' : peak,
    }
def undo_preprocess_mask(mask, info, radius=25):
    H, W = info["original_shape"]
    band_top, band_bottom, left, right = info["crop"]

    rot_mask = np.zeros((H, W), dtype=np.uint8)

    mh, mw = mask.shape
    roi_h = band_bottom - band_top

    # match_template_1d_image returns height roi_h - 2*radius,
    # so the mask y coords are centered at y=radius...roi_h-radius
    y0 = band_top + radius
    y1 = y0 + mh
    x0 = left
    x1 = left + mw

    rot_mask[y0:y1, x0:x1] = mask.astype(np.uint8) * 255

    unrot = cv2.warpAffine(
        rot_mask,
        info["Minv"],
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

    return unrot > 0
def zscore(v):
    v = v.astype(float)
    return (v - v.mean()) / (v.std() + 1e-9)


def make_general_laser_template(bg_gray, radius=25, use_derivative=False):
    """
    Builds one canonical vertical laser profile from a pure-background image.

    bg_gray: pure background image, grayscale
    radius: half-height of template window
    """
    bg_gray = bg_gray.astype(float)
    h, w = bg_gray.shape
    ys = np.arange(h)
    patches = []
    for x in range(w):
        col = bg_gray[:, x]
        # Estimate center from the background column itself
        p = col - np.percentile(col, 10)
        p = np.maximum(p, 0)
        total = p.sum()
        if total <= 1e-9:
            continue
        cy = int(np.round(np.sum(ys * p) / total))
        y0 = cy - radius
        y1 = cy + radius + 1
        if y0 < 0 or y1 > h:
            continue
        patch = col[y0:y1].astype(float)
        if use_derivative:
            patch = np.gradient(patch)

        patches.append(zscore(patch))
    template = np.median(np.stack(patches, axis=0), axis=0)
    template = zscore(template)

    return template
def match_template_1d_column(col, template, use_derivative=False):
    """
    Slides template vertically through one image column.
    Returns correlation score for each y-position.
    """
    col = col.astype(float)
    L = len(template)
    h = len(col)

    scores = np.full(h - L + 1, -np.inf)

    for y0 in range(h - L + 1):
        patch = col[y0:y0 + L]

        if use_derivative:
            patch = np.gradient(patch)

        patch = zscore(patch)
        # normalized dot product / correlation
        scores[y0] = np.mean(patch - template)

    return scores
def template_feature_distance_image(
    img_gray,
    template,
    use_derivative=True,
    w_corr=1.0,
    w_width=2.0,
    w_swing=0.5,
):
    img = img_gray.astype(float, copy=False)
    t = np.asarray(template, dtype=float)
    L = len(t)
    eps = 1e-9

    win = sliding_window_view(img, L, axis=0)  # (h-L+1, w, L)

    if use_derivative:
        win = np.gradient(win, axis=-1)

    # correlation term: still useful for rough shape alignment
    zwin = (win - win.mean(axis=-1, keepdims=True)) / (win.std(axis=-1, keepdims=True) + eps)
    zt = (t - t.mean()) / (t.std() + eps)
    corr = np.tensordot(zwin, zt, axes=([-1], [0])) / L
    plt.figure()
    plt.title("correlation")
    print(corr.max(), corr.min())
    plt.imshow(corr)

    # width term: distance between positive and negative derivative lobes
    win_argmax = np.argmax(win, axis=-1)
    win_argmin = np.argmin(win, axis=-1)
    win_width = np.abs(win_argmax - win_argmin).astype(float)

    t_width = float(abs(np.argmax(t) - np.argmin(t)))

    # swing term: derivative peak-to-trough amplitude
    win_swing = win.max(axis=-1) - win.min(axis=-1)
    t_swing = float(t.max() - t.min())
    plt.figure()
    plt.title("swing")
    plt.imshow(win_swing)
    width_err = np.abs(np.log((win_width + 1.0) / (t_width + 1.0)))
    swing_err = np.abs(np.log((win_swing + eps) / (t_swing + eps)))

    return w_corr * corr * w_width * width_err * w_swing * swing_err
def match_template_1d_image(img_gray, template, use_derivative=False):
    #return template_feature_distance_image(img_gray, template, use_derivative=use_derivative, w_corr=.25, w_width=.25, w_swing=.5)
    img = img_gray.astype(float, copy=False)
    t = np.asarray(template, dtype=float)
    L = len(t)

    win = sliding_window_view(img, L, axis=0)  # (h-L+1, w, L)

    if use_derivative:
        win = np.gradient(win, axis=-1)

    mu = win.mean(axis=-1)
    sd = win.std(axis=-1) + 1e-9

    # same as: mean(zscore(patch) * template)
    return np.clip((np.tensordot(win, t, axes=([-1], [0])) / L - mu * t.mean()) / sd, 0, 1)
def number(p):
    match = re.search(r'\d+', p)
    return int(match.group()) if match else -1
def path_to_grad(path, radius=25, degree=26.2):
    gray = to_gray(preprocess_image(path, degree=degree))
    grad = make_general_laser_template(gray, use_derivative=True, radius=radius)
    return grad
def extract_filament_array(folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\", 
                           empty_i=2420-149, 
                           full_i=1949-149, 
                           img_i=None, 
                           radius=25, 
                           threshold=.07,
                           grad = None,
                           full_grad = None):
    photos = sorted(os.listdir(folder), key=number)
    empty_path = folder + photos[empty_i]
    full_path = folder + photos[full_i]
    
    random_path = folder + (random.choice(photos) if img_i is None else photos[img_i])
    if(img_i is None):
        print(random_path)
    random_pre = get_preprocess_crop_info(random_path, degree=26.2)
    random_gray = to_gray(random_pre['roi'])
    
    if grad is None:
        empty_gray = to_gray(preprocess_image(empty_path, degree=26.2))
        grad = make_general_laser_template(empty_gray, use_derivative=True, radius=radius)
    if full_grad is None:
        full_gray = to_gray(preprocess_image(full_path, degree=26.2))
        full_grad = make_general_laser_template(full_gray, use_derivative=True, radius=radius)

    corr = match_template_1d_image(random_gray, grad, True)
    full_corr = match_template_1d_image(random_gray, full_grad, True)

    combin = full_corr * (1 - corr)
    mask = np.mean(combin, axis=0, keepdims=True)
    combin = np.broadcast_to(mask, combin.shape)
    return combin < threshold, random_pre
if __name__ == "__main__":
    #change folder
    # folder = r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\"
    # photos = sorted(os.listdir(folder), key = number)
    # empty_path = folder + photos[2418]
    # full_path = folder + photos[1946]
    # random_path = folder + random.choice(photos)
    # print(random_path)
    # #to rotated ndarray
    # empty_gray = (to_gray(preprocess_image(empty_path, degree=26.2)))
    # random_gray = (to_gray(preprocess_image(random_path, degree=26.2)))
    # radius = 30
    # template = make_general_laser_template(empty_gray, radius = radius)
    # grad = make_general_laser_template(empty_gray, use_derivative=True, radius=radius)
    # img = np.zeros_like(random_gray)
    # corr_img = np.zeros(random_gray.shape, dtype=float)
    # full_corr_img = np.zeros(random_gray.shape, dtype=float)
    # for i in range(img.shape[1]):
    #     corr_img[radius:-radius, i] = match_template_1d_column(random_gray[:, i], grad, True)
    # full_gray = to_gray(preprocess_image(full_path, degree=26.2))
    # full_grad = make_general_laser_template(full_gray, use_derivative=True, radius=radius)
    # for i in range(img.shape[1]):
    #     full_corr_img[radius:-radius, i] = match_template_1d_column(random_gray[:, i], full_grad, True)
    import time
    t = time.time()
    mask, pack = extract_filament_array(img_i = 20)
    print(np.sum(mask))
    plt.figure()
    plt.title("mask")
    plt.imshow(mask)
    plt.figure()
    rmask = undo_preprocess_mask(mask, pack, 25)
    plt.imshow(rmask)
    plt.title("unrotated mask")
    plt.show()

    # plt.imshow(
    #     combin,
    #     cmap="seismic",
    #     norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    # )
    # plt.colorbar(label="filament")
    # plt.show()
    # plt.imshow(combin[radius:-radius, :] < 0.07)
    # plt.show()
    # plt.imshow(extract_filament_array())
    # plt.show()