import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from analyze_six import is_split_skew_model, preprocess_image
from combine_methods import detect_filament_split, plot_path
import pickle
FOLDER_PATH = r"C:\Users\dhruv\Documents\dhruv_python\frames"

_COORD_PATTERN = re.compile(
    r"_x_(-?\d+(?:\.\d+)?)_y_(-?\d+(?:\.\d+)?)_z_(-?\d+(?:\.\d+)?)"
)

def is_split_skew_model2(path):
    img_bgr = preprocess_image(path)
    result = detect_filament_split(img_bgr, laser_y=42, window_radius=40, debug=False)
    return result
def parse_coords(filename):
    match = _COORD_PATTERN.search(filename)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    raise ValueError(f"Filename does not contain x/y/z coordinates: {filename}")


def get_split_confidence(value):
    """Extract confidence score (index 6) from is_split_skew_model output."""
    if isinstance(value, (tuple, list)):
        return float(value[5])
    if isinstance(value, dict):
        #return value['split_confidence']
        best = value.get("best_split", None)
        return float(best['split_confidence'])
    return float(value)

def accuracy_binary(xs, ys, confs, labels):
    # assume all x < y points should be 0, all else should be 1
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    confs = np.asarray(confs)
    expected = np.where(labels == 1, 1.0, 0.0)
    residuals = ( expected[:-1] - confs)
    print(np.mean(residuals))
    

    return residuals
def heatmap_from_folder(folder, grid_resolution=400, sigma_scale=0.5):
    """
    Scan `folder` for images, run is_split_skew_model on each, parse (x, y)
    from the filename, then render a smooth confidence heatmap.

    Parameters
    ----------
    folder          : path to image directory
    grid_resolution : pixel resolution of the interpolated grid (default 400)
    sigma_scale     : controls Gaussian blur radius relative to point spacing;
                      increase to make blobs merge more aggressively (default 0.5)
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    correct = 0
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    print(f"Found {len(files)} image file(s) in '{folder}'.")

    xs, ys, confs, labels = [], [], [], []
    with open(FOLDER_PATH + r'/labels.pkl', 'rb') as l:
        truths = pickle.load(l)
        labels = list(truths.values())
        fnames = list(truths.keys())
    i = 0
    for fname in files:
        try:
            x, y, _ = parse_coords(fname)
        except ValueError as e:
            print(f"  [skip] {fname}: {e}")
            continue

        fpath = os.path.join(folder, fname)
        try:
            result = is_split_skew_model2(fpath)
            conf = get_split_confidence(result)
            xs.append(x)
            ys.append(y)
            confs.append(conf)
            print(f"  {fname:50s} x={x:8.2f}  y={y:8.2f}  conf={conf:.4f} label = {labels[i]}")
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            raise e
        i+=1
    if not xs:
        print("No valid data points — nothing to plot.")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    confs = np.array(confs)
    labels = np.array(labels)
    confs = np.abs(accuracy_binary(xs, ys, confs, labels))


    # ------------------------------------------------------------------ #
    #  Build a fine regular grid and interpolate with cubic + fallback    #
    # ------------------------------------------------------------------ #
    # Add small padding around the extent so edge blobs aren't clipped
    pad_x = (xs.max() - xs.min()) * 0.08 or 1.0
    pad_y = (ys.max() - ys.min()) * 0.08 or 1.0

    xi = np.linspace(xs.min() - pad_x, xs.max() + pad_x, grid_resolution)
    yi = np.linspace(ys.min() - pad_y, ys.max() + pad_y, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # Cubic for smooth interior, nearest-neighbour fills edges (no NaN halos)
    Zi_cubic   = griddata((xs, ys), confs, (Xi, Yi), method="cubic")
    Zi_nearest = griddata((xs, ys), confs, (Xi, Yi), method="nearest")
    Zi = np.where(np.isnan(Zi_cubic), Zi_nearest, Zi_cubic)

    # ------------------------------------------------------------------ #
    #  Optional Gaussian smoothing to merge nearby blobs                  #
    # ------------------------------------------------------------------ #
    # Estimate typical pixel spacing from the median inter-point distance
    from scipy.ndimage import gaussian_filter
    if len(xs) > 1:
        spacing_x = np.median(np.diff(np.unique(np.sort(xs))))
        spacing_y = np.median(np.diff(np.unique(np.sort(ys))))
        pixels_per_unit_x = grid_resolution / (xi[-1] - xi[0])
        pixels_per_unit_y = grid_resolution / (yi[-1] - yi[0])
        sigma_px = (
            sigma_scale
            * np.mean([spacing_x * pixels_per_unit_x,
                       spacing_y * pixels_per_unit_y])
        )
        sigma_px = max(sigma_px, 1.0)          # at least 1 px
        Zi = gaussian_filter(Zi, sigma=sigma_px)

    # ------------------------------------------------------------------ #
    #  Plot                                                                #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(11, 8))

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.RdYlGn          # red = low confidence, green = high

    im = ax.imshow(
        Zi,
        extent=[xi[0], xi[-1], yi[0], yi[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",  # second pass of bilinear for display
    )

    # Scatter the raw sample points on top so locations are visible
    sc = ax.scatter(
        xs, ys,
        c=confs,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.6,
        s=60,
        zorder=5,
        label="Sample points",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Split Confidence", fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title("Split Confidence Heatmap (X–Y Plane)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)

    plt.tight_layout()

    out_path = os.path.join(folder, "split_confidence_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved → {out_path}")
    plt.show()
    for i, x in enumerate(np.abs(confs)):
        if(x > .5):
            plot_path(fnames[i], label = f"{labels[i]} {x}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER_PATH
    # Optional: pass grid resolution and sigma scale as extra CLI args
    res   = int(sys.argv[2])   if len(sys.argv) > 2 else 400
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    heatmap_from_folder(folder, grid_resolution=res, sigma_scale=sigma)