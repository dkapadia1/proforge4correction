import argparse, csv, re
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def parse_xy_from_name(path):
    """
    Expected filename example:
    frame_149_t_5876.328144_x_227.282317_y_140.818221_z_0.120763.jpg

    Returns:
        x, y
    Ignores:
        frame index, t, z
    """
    m = re.search(
        r"_x_([-+]?\d*\.?\d+)_y_([-+]?\d*\.?\d+)",
        path.stem
    )
    if not m:
        return np.nan, np.nan

    return float(m.group(1)), float(m.group(2))


class Picker:
    def __init__(self, paths, out_csv, max_width=1800):
        self.paths = paths
        self.out_csv = Path(out_csv)
        self.max_width = max_width
        self.i = 0
        self.pending = None
        self.rows = []
        self.scale = 1.0
        self.left_w = 0

    def load_pair(self):
        a = cv2.imread(str(self.paths[self.i]))
        b = cv2.imread(str(self.paths[self.i + 1]))
        if a is None or b is None:
            raise RuntimeError("Could not read an image.")

        h = max(a.shape[0], b.shape[0])
        canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
        canvas[:a.shape[0], :a.shape[1]] = a
        canvas[:b.shape[0], a.shape[1]:a.shape[1] + b.shape[1]] = b

        self.left_w = a.shape[1]
        self.scale = min(1.0, self.max_width / canvas.shape[1])
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)

        self.canvas = canvas
        self.redraw()

    def to_original_xy(self, x, y):
        x, y = x / self.scale, y / self.scale
        if x < self.left_w:
            return "a", x, y
        return "b", x - self.left_w, y

    def redraw(self):
        img = self.canvas.copy()
        title = f"{self.paths[self.i].name}  ->  {self.paths[self.i + 1].name}"
        cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        divider_x = int(self.left_w * self.scale)
        cv2.line(img, (divider_x, 0), (divider_x, img.shape[0]), (255, 255, 255), 1)

        info = "Click point in LEFT image, then matching point in RIGHT image. n=next, u=undo, s=save, q=quit"
        cv2.putText(img, info, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        for r in self.rows:
            if r["img_a"] != self.paths[self.i].name:
                continue

            ax = int(r["px_ax"] * self.scale)
            ay = int(r["px_ay"] * self.scale)
            bx = int((r["px_bx"] + self.left_w) * self.scale)
            by = int(r["px_by"] * self.scale)

            cv2.circle(img, (ax, ay), 5, (0, 255, 255), -1)
            cv2.circle(img, (bx, by), 5, (0, 255, 255), -1)
            cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 1)

        if self.pending is not None:
            _, x, y = self.pending
            cv2.circle(img, (int(x * self.scale), int(y * self.scale)), 6, (0, 0, 255), -1)

        cv2.imshow("pixel matcher", img)

    def click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        side, ox, oy = self.to_original_xy(x, y)

        if self.pending is None:
            if side != "a":
                print("First click must be in the LEFT image.")
                return
            self.pending = (side, ox, oy)
            self.redraw()
            return

        if side != "b":
            print("Second click must be in the RIGHT image.")
            return

        _, ax, ay = self.pending
        bx, by = ox, oy
        ca_x, ca_y = parse_xy_from_name(self.paths[self.i])
        cb_x, cb_y = parse_xy_from_name(self.paths[self.i + 1])

        self.rows.append({
            "img_a": self.paths[self.i].name,
            "img_b": self.paths[self.i + 1].name,
            "coord_ax": ca_x,
            "coord_ay": ca_y,
            "coord_bx": cb_x,
            "coord_by": cb_y,
            "px_ax": ax,
            "px_ay": ay,
            "px_bx": bx,
            "px_by": by,
            "dpx_x": ax - bx,
            "dpx_y": ay - by,
            "dcoord_x": cb_x - ca_x,
            "dcoord_y": cb_y - ca_y,
        })

        self.pending = None
        print(f"Added match #{len(self.rows)}")
        self.redraw()

    def save(self):
        if not self.rows:
            print("No rows to save.")
            return

        with self.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"Saved {len(self.rows)} matches to {self.out_csv}")
        self.fit_transform()

    def fit_transform(self):
        rows = [r for r in self.rows if not np.isnan(r["dcoord_x"]) and not np.isnan(r["dcoord_y"])]
        if not rows:
            print("Need coordinate-labeled matches to fit px -> coord scale.")
            return

        scales = []

        for r in rows:
            dpx = np.hypot(r["dpx_x"], r["dpx_y"])
            dc = np.hypot(r["dcoord_x"], r["dcoord_y"])

            if dpx < 1e-6:
                continue

            scales.append(dc / dpx)

        if not scales:
            print("No usable matches: pixel movement was too small.")
            return

        scales = np.array(scales)
        scale = np.median(scales)

        print("\nSingle px -> coord conversion:")
        print(f"coord_per_pixel = {scale:.9f}")
        print(f"pixels_per_coord = {1 / scale:.9f}")
        print(f"mean coord_per_pixel = {scales.mean():.9f}")
        print(f"std  coord_per_pixel = {scales.std():.9f}")
        print(f"num matches = {len(scales)}\n")

    def run(self):
        cv2.namedWindow("pixel matcher", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("pixel matcher", self.click)
        self.load_pair()

        while True:
            k = cv2.waitKey(20) & 0xFF

            if k == ord("q"):
                break

            if k == ord("s"):
                self.save()

            if k == ord("u"):
                if self.pending is not None:
                    self.pending = None
                elif self.rows:
                    self.rows.pop()
                self.redraw()

            if k == ord("n"):
                self.pending = None
                self.i = min(self.i + 1, len(self.paths) - 2)
                self.load_pair()

        self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=r"C:\Users\dhruv\Documents\dhruv_python\disc2accurate\\")
    ap.add_argument("--out", default="pixel_matches.csv")
    ap.add_argument("--max-width", type=int, default=1800)
    args = ap.parse_args()

    paths = sorted(
        [p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS],
        key=natural_key,
    )

    if len(paths) < 2:
        raise SystemExit("Need at least 2 images in the folder.")

    Picker(paths, args.out, args.max_width).run()


if __name__ == "__main__":
    main()