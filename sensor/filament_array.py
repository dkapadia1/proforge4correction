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
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        scores[y0] = np.mean(patch * template)

    return scores
def match_template_1d_image(img_gray, template, use_derivative=False):
    img = img_gray.astype(float, copy=False)
    t = np.asarray(template, dtype=float)
    L = len(t)

    win = sliding_window_view(img, L, axis=0)  # (h-L+1, w, L)

    if use_derivative:
        win = np.gradient(win, axis=-1)

    mu = win.mean(axis=-1)
    sd = win.std(axis=-1) + 1e-9

    # same as: mean(zscore(patch) * template)
    return (np.tensordot(win, t, axes=([-1], [0])) / L - mu * t.mean()) / sd
def number(p):
    match = re.search(r'\d+', p)
    return int(match.group()) if match else -1
def extract_filament_array(folder=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\", 
                           empty_i=2418-149, 
                           full_i=1946-149, 
                           img_i=None, 
                           radius=25, 
                           threshold=.07):
    photos = sorted(os.listdir(folder), key=number)

    empty_path = folder + photos[empty_i]
    full_path = folder + photos[full_i]
    random_path = folder + (random.choice(photos) if img_i is None else photos[img_i])

    empty_gray = to_gray(preprocess_image(empty_path, degree=26.2))
    random_gray = to_gray(preprocess_image(random_path, degree=26.2))
    full_gray = to_gray(preprocess_image(full_path, degree=26.2))

    grad = make_general_laser_template(empty_gray, use_derivative=True, radius=radius)
    full_grad = make_general_laser_template(full_gray, use_derivative=True, radius=radius)

    corr = match_template_1d_image(random_gray, grad, True)
    full_corr = match_template_1d_image(random_gray, full_grad, True)

    combin = full_corr * (1 - corr)
    return combin < threshold
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
    print(np.sum(extract_filament_array()))
    print(time.time() - t)
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