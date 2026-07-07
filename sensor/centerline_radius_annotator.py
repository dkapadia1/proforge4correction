#!/usr/bin/env python3
"""
Centerline + radius annotator for filament images.

Workflow
--------
- Opens images from a folder in random order.
- Browse until you find a useful image.
- Annotate by clicking centerline points.
- Adjust radius with keyboard keys, not mouse wheel.
- Saves annotations as JSON.

Controls
--------
Browsing:
    n / Space : next random image
    b         : previous image
    f         : find image by filename substring
    s         : save current annotation
    q / Esc   : quit

Annotation:
    Left click       : add point at cursor with current radius
    Right click      : delete nearest point
    u               : undo last point
    c               : clear current image points
    [ / ]           : decrease / increase current radius by 0.25 px
    - / =           : decrease / increase current radius by 1.0 px
    1 / 2 / 3 / 4   : set radius to 1 / 2 / 3 / 4 px
    g               : toggle generated soft/binary mask preview
    t               : toggle detector mask overlay if enabled in code
    p               : toggle point/spline overlay
    + / _           : zoom in / out
    arrow keys      : pan when zoomed

Output
------
annotations.json:
{
  "image_name.jpg": {
    "points": [{"x": 100.5, "y": 237.2, "r": 3.0}, ...],
    "quality": "annotated"
  }
}
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_images(folder: Path):
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    if not files:
        raise SystemExit(f"No images found in: {folder}")
    return files


def draw_text(img, text, org, scale=0.48, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def catmull_rom(points, samples_per_seg=20):
    pts = np.asarray(points, np.float32)
    if len(pts) < 2:
        return pts
    out = []
    P = np.vstack([pts[0], pts, pts[-1]])
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                              (-p0 + 3*p1 - 3*p2 + p3) * t3))
    out.append(pts[-1])
    return np.asarray(out, np.float32)


def soft_mask_from_points(shape, ann_points, edge_sigma=0.75):
    """Returns float32 soft mask in [0,1] from centerline points with per-point radius."""
    h, w = shape[:2]
    if len(ann_points) == 0:
        return np.zeros((h, w), np.float32)

    pts = [(p["x"], p["y"]) for p in ann_points]
    radii = np.asarray([p["r"] for p in ann_points], np.float32)

    curve = catmull_rom(pts, 16)
    if len(curve) == 0:
        return np.zeros((h, w), np.float32)

    # Interpolate radius along curve by nearest original point index along x/order.
    # For annotation/eval this is fine and fast enough.
    orig = np.asarray(pts, np.float32)
    d2 = ((curve[:, None, :] - orig[None, :, :]) ** 2).sum(axis=2)
    curve_r = radii[np.argmin(d2, axis=1)]

    # Draw a high-res-ish distance field by sampling local windows around curve points.
    dist = np.full((h, w), np.inf, np.float32)
    for (x, y), r in zip(curve, curve_r):
        pad = int(math.ceil(r + 4 * edge_sigma + 2))
        x0, x1 = max(0, int(x) - pad), min(w, int(x) + pad + 1)
        y0, y1 = max(0, int(y) - pad), min(h, int(y) + pad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - r
        dist[y0:y1, x0:x1] = np.minimum(dist[y0:y1, x0:x1], d)

    return (1.0 / (1.0 + np.exp(dist / edge_sigma))).astype(np.float32)


def detector_mask_ycrcb(img_bgr, den_min=12, den_max=80, thresh=2.8):
    """Optional overlay of the cheap clipped-ratio detector."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    score = y.astype(np.float32) / np.clip(cr.astype(np.float32) - cb.astype(np.float32), den_min, den_max)
    score = cv2.blur(score, (3, 3))
    return score > thresh


class Annotator:
    def __init__(self, folder, out_json, seed=0):
        self.folder = Path(folder)
        self.out_json = Path(out_json)
        self.files = load_images(self.folder)
        self.rng = random.Random(seed)
        self.order = list(range(len(self.files)))
        self.rng.shuffle(self.order)
        self.pos = 0

        self.ann = self.load_annotations()
        self.img = None
        self.path = None
        self.name = None
        self.radius = 3.0
        self.show_mask = False
        self.show_detector = False
        self.show_points = True
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.win = "centerline_radius_annotator"

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.load_current()

    def load_annotations(self):
        if self.out_json.exists():
            with open(self.out_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_json.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.ann, f, indent=2)
        tmp.replace(self.out_json)
        print(f"Saved {self.out_json}")

    def current_points(self):
        self.ann.setdefault(self.name, {"points": [], "quality": "unlabeled"})
        return self.ann[self.name]["points"]

    def load_current(self):
        idx = self.order[self.pos]
        self.path = self.files[idx]
        self.name = str(self.path.relative_to(self.folder)).replace("\\", "/")
        self.img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self.img is None:
            print(f"Could not read {self.path}; skipping")
            self.next()
            return
        self.ann.setdefault(self.name, {"points": [], "quality": "unlabeled"})
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

    def next(self):
        self.pos = (self.pos + 1) % len(self.order)
        self.load_current()

    def prev(self):
        self.pos = (self.pos - 1) % len(self.order)
        self.load_current()

    def find(self):
        query = input("filename substring: ").strip().lower()
        if not query:
            return
        matches = [i for i, p in enumerate(self.files) if query in str(p).lower()]
        if not matches:
            print("No match.")
            return
        target = matches[0]
        if target not in self.order:
            self.order.insert(self.pos + 1, target)
            self.pos += 1
        else:
            self.pos = self.order.index(target)
        self.load_current()

    def screen_to_img(self, x, y):
        if self.zoom <= 1.0:
            return float(x), float(y)
        return float(x / self.zoom + self.pan_x), float(y / self.zoom + self.pan_y)

    def nearest_point_i(self, x, y, max_dist=12):
        pts = self.current_points()
        if not pts:
            return None
        d = [(p["x"] - x) ** 2 + (p["y"] - y) ** 2 for p in pts]
        i = int(np.argmin(d))
        return i if d[i] <= max_dist * max_dist else None

    def on_mouse(self, event, x, y, flags, param):
        ix, iy = self.screen_to_img(x, y)
        h, w = self.img.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points().append({"x": round(ix, 3), "y": round(iy, 3), "r": round(self.radius, 3)})
            self.ann[self.name]["quality"] = "annotated"

        elif event == cv2.EVENT_RBUTTONDOWN:
            i = self.nearest_point_i(ix, iy)
            if i is not None:
                self.current_points().pop(i)
                if not self.current_points():
                    self.ann[self.name]["quality"] = "unlabeled"

    def view_image(self):
        view = self.img.copy()

        if self.show_detector:
            m = detector_mask_ycrcb(self.img)
            overlay = np.zeros_like(view)
            overlay[m] = (0, 180, 255)
            view = cv2.addWeighted(view, 1.0, overlay, 0.35, 0)

        if self.show_mask and self.current_points():
            sm = soft_mask_from_points(view.shape, self.current_points())
            overlay = np.zeros_like(view)
            overlay[sm > 0.5] = (0, 255, 0)
            view = cv2.addWeighted(view, 1.0, overlay, 0.35, 0)

        if self.show_points and self.current_points():
            pts = self.current_points()
            xy = [(p["x"], p["y"]) for p in pts]
            curve = catmull_rom(xy, 20)
            if len(curve) >= 2:
                for a, b in zip(curve[:-1], curve[1:]):
                    cv2.line(view, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (255, 255, 0), 1, cv2.LINE_AA)
            for p in pts:
                c = (int(round(p["x"])), int(round(p["y"])))
                cv2.circle(view, c, max(1, int(round(p["r"]))), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(view, c, 2, (0, 0, 255), -1, cv2.LINE_AA)

        h, w = view.shape[:2]
        text1 = f"{self.pos+1}/{len(self.order)}  {self.name}"
        text2 = f"points={len(self.current_points())}  radius={self.radius:.2f}  quality={self.ann[self.name]['quality']}"
        text3 = "n/space next | b prev | f find | click add | right-click delete | [] radius | s save | q quit"
        draw_text(view, text1, (8, 20))
        draw_text(view, text2, (8, 42))
        draw_text(view, text3, (8, h - 10), scale=0.43)

        if self.zoom > 1.0:
            zh, zw = int(h / self.zoom), int(w / self.zoom)
            self.pan_x = int(np.clip(self.pan_x, 0, max(0, w - zw)))
            self.pan_y = int(np.clip(self.pan_y, 0, max(0, h - zh)))
            crop = view[self.pan_y:self.pan_y + zh, self.pan_x:self.pan_x + zw]
            view = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

        return view

    def run(self):
        while True:
            cv2.imshow(self.win, self.view_image())
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), 27):
                self.save()
                break
            elif key in (ord("n"), ord(" ")):
                self.next()
            elif key == ord("b"):
                self.prev()
            elif key == ord("f"):
                self.find()
            elif key == ord("s"):
                self.save()
            elif key == ord("u"):
                if self.current_points():
                    self.current_points().pop()
            elif key == ord("c"):
                self.ann[self.name] = {"points": [], "quality": "unlabeled"}
            elif key == ord("["):
                self.radius = max(0.25, self.radius - 0.25)
            elif key == ord("]"):
                self.radius += 0.25
            elif key == ord("-"):
                self.radius = max(0.25, self.radius - 1.0)
            elif key in (ord("="), ord("+")):
                self.radius += 1.0
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                self.radius = float(chr(key))
            elif key == ord("g"):
                self.show_mask = not self.show_mask
            elif key == ord("t"):
                self.show_detector = not self.show_detector
            elif key == ord("p"):
                self.show_points = not self.show_points
            elif key == ord("+"):
                self.zoom = min(8.0, self.zoom * 1.25)
            elif key == ord("_"):
                self.zoom = max(1.0, self.zoom / 1.25)
            # arrow key codes vary by platform; these handle common OpenCV values.
            elif key in (81, 2424832 & 0xFF):  # left
                self.pan_x -= 20
            elif key in (83, 2555904 & 0xFF):  # right
                self.pan_x += 20
            elif key in (82, 2490368 & 0xFF):  # up
                self.pan_y -= 20
            elif key in (84, 2621440 & 0xFF):  # down
                self.pan_y += 20

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing images")
    ap.add_argument("--out", default="annotations.json", help="output annotation JSON")
    ap.add_argument("--seed", type=int, default=0, help="random image order seed")
    args = ap.parse_args()
    Annotator(args.folder, args.out, args.seed).run()


if __name__ == "__main__":
    main()
