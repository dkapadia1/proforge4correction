from dataclasses import dataclass
from pathlib import Path
import math
import re

import cv2
import numpy as np

DEFAULT_GCODE_PATH = Path("P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode")
PX_PER_MM = 26.13
LINE_WIDTH_MM = 0.45

_WORD_RE = re.compile(r"([A-Za-z])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _words(code):
    return {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}


def _set_xyz(pos, vals, absolute_xyz):
    for k in "XYZ":
        if k in vals:
            pos[k] = vals[k] if absolute_xyz else pos[k] + vals[k]


def _set_e(pos, vals, absolute_e):
    old_e = pos["E"]
    if "E" in vals:
        pos["E"] = vals["E"] if absolute_e else pos["E"] + vals["E"]
    return pos["E"] - old_e


def _arc_points(start, end, vals, clockwise, step_mm=0.05):
    """Approximate a G2/G3 XY arc using I/J center offsets."""
    p0 = np.array([start["X"], start["Y"]], float)
    p1 = np.array([end["X"], end["Y"]], float)

    if "I" not in vals and "J" not in vals:
        return None

    center = p0 + np.array([vals.get("I", 0.0), vals.get("J", 0.0)], float)
    radius = float(np.linalg.norm(p0 - center))
    if radius <= 1e-12:
        return None

    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])

    if np.allclose(p0, p1):
        sweep = -2 * math.pi if clockwise else 2 * math.pi
    elif clockwise:
        if a1 >= a0:
            a1 -= 2 * math.pi
        sweep = a1 - a0
    else:
        if a1 <= a0:
            a1 += 2 * math.pi
        sweep = a1 - a0

    n = max(2, int(math.ceil(abs(sweep) * radius / step_mm)))
    ang = a0 + np.linspace(0.0, sweep, n + 1)
    return np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])


def parse_gcode_extrusion_segments(gcode_file=DEFAULT_GCODE_PATH, arc_step_mm=0.05):
    """Return extrusion segments as [x0, y0, z0, x1, y1, z1, de, line_no]."""
    pos = dict(X=0.0, Y=0.0, Z=np.nan, E=0.0)
    absolute_xyz, absolute_e = True, True
    segs, unsupported = [], []

    for line_no, raw in enumerate(Path(gcode_file).read_text(errors="ignore").splitlines(), 1):
        code = raw.split(";", 1)[0].strip()
        if not code:
            continue

        cmd = code.split()[0].upper()
        vals = _words(code)

        if cmd == "G90":
            absolute_xyz = True
        elif cmd == "G91":
            absolute_xyz = False
        elif cmd == "M82":
            absolute_e = True
        elif cmd == "M83":
            absolute_e = False
        elif cmd == "G92":
            for k in "XYZE":
                if k in vals:
                    pos[k] = vals[k]
        elif cmd in {"G0", "G1"}:
            old = pos.copy()
            _set_xyz(pos, vals, absolute_xyz)
            de = _set_e(pos, vals, absolute_e)
            dxy = math.hypot(pos["X"] - old["X"], pos["Y"] - old["Y"])
            if cmd == "G1" and de > 1e-9 and dxy > 1e-9:
                z0 = old["Z"] if not np.isnan(old["Z"]) else pos["Z"]
                z1 = pos["Z"] if not np.isnan(pos["Z"]) else z0
                segs.append((old["X"], old["Y"], z0, pos["X"], pos["Y"], z1, de, line_no))
        elif cmd in {"G2", "G3"}:
            old = pos.copy()
            _set_xyz(pos, vals, absolute_xyz)
            de = _set_e(pos, vals, absolute_e)
            pts = _arc_points(old, pos, vals, clockwise=(cmd == "G2"), step_mm=arc_step_mm)
            if de <= 1e-9:
                continue
            if pts is None:
                unsupported.append((line_no, raw))
                continue

            z0 = old["Z"] if not np.isnan(old["Z"]) else pos["Z"]
            z1 = pos["Z"] if not np.isnan(pos["Z"]) else z0
            zs = np.linspace(z0, z1, len(pts))
            de_piece = de / max(1, len(pts) - 1)
            for j in range(len(pts) - 1):
                if np.linalg.norm(pts[j + 1] - pts[j]) > 1e-12:
                    segs.append((pts[j, 0], pts[j, 1], zs[j], pts[j + 1, 0], pts[j + 1, 1], zs[j + 1], de_piece, line_no))

    arr = np.array(segs, dtype=float) if segs else np.zeros((0, 8), dtype=float)
    return arr, unsupported


def layer_z_values(segs, decimals=5):
    if len(segs) == 0:
        return np.array([], dtype=float)
    mid_z = (segs[:, 2] + segs[:, 5]) / 2
    mid_z = mid_z[~np.isnan(mid_z)]
    return np.unique(np.round(mid_z, decimals))


def pick_layer_segments_by_index(segs, layer_index=0, include_previous_layers=False, z_tol=0.08):
    """Pick a layer by sorted Z index. If the G-code has no usable Z values, return all segments."""
    zs = layer_z_values(segs)
    if len(segs) == 0 or len(zs) == 0:
        return segs, None, zs

    target_z = zs[layer_index]
    mid_z = (segs[:, 2] + segs[:, 5]) / 2
    keep = mid_z <= target_z + z_tol if include_previous_layers else np.abs(mid_z - target_z) <= z_tol
    return segs[keep], target_z, zs


def transform_xy(x, y, flip_x=False, flip_y=False, swap_xy=False):
    """Transform printer/file XY into the same coordinate convention as the merge canvas."""
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return float(x), float(y)


def transform_segments_xy(segs, flip_x=False, flip_y=False, swap_xy=False):
    out = np.array(segs, dtype=float, copy=True)
    if len(out) == 0:
        return out

    for a, b in [(0, 1), (3, 4)]:
        xy = np.array([transform_xy(x, y, flip_x, flip_y, swap_xy) for x, y in out[:, [a, b]]])
        out[:, [a, b]] = xy
    return out


@dataclass
class FilamentCoordMask:
    """
    Bool filament mask with explicit printer-space coordinates.

    origin_xy_mm is the transformed printer coordinate of pixel [row=0, col=0].
    theta_deg=0 means columns are +X and rows are +Y in transformed printer coordinates.
    """
    mask: np.ndarray
    origin_xy_mm: tuple
    px_per_mm: float
    theta_deg: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False

    @property
    def shape(self):
        return self.mask.shape

    @property
    def height(self):
        return self.mask.shape[0]

    @property
    def width(self):
        return self.mask.shape[1]

    @property
    def origin_x_mm(self):
        return float(self.origin_xy_mm[0])

    @property
    def origin_y_mm(self):
        return float(self.origin_xy_mm[1])

    def _basis(self):
        th = np.deg2rad(self.theta_deg)
        along = np.array([np.cos(th), np.sin(th)])
        across = np.array([-np.sin(th), np.cos(th)])
        return along, across

    def xy_to_pixel(self, x, y, already_transformed=False, rounded=True):
        """Convert raw printer/camera XY to (row, col) in this mask."""
        if not already_transformed:
            x, y = transform_xy(x, y, self.flip_x, self.flip_y, self.swap_xy)
        q = np.array([x - self.origin_x_mm, y - self.origin_y_mm], float)
        along, across = self._basis()
        col = float(np.dot(q, along) * self.px_per_mm)
        row = float(np.dot(q, across) * self.px_per_mm)
        return (int(round(row)), int(round(col))) if rounded else (row, col)

    def pixel_to_xy(self, row, col, undo_transform=False):
        """Convert (row, col) back to transformed printer XY, or raw printer XY if undo_transform=True."""
        along, across = self._basis()
        xy = np.array(self.origin_xy_mm, float) + (col / self.px_per_mm) * along + (row / self.px_per_mm) * across
        x, y = float(xy[0]), float(xy[1])
        if undo_transform:
            if self.flip_x:
                x = -x
            if self.flip_y:
                y = -y
            if self.swap_xy:
                x, y = y, x
        return x, y

    def inside_xy(self, x, y, already_transformed=False):
        row, col = self.xy_to_pixel(x, y, already_transformed=already_transformed)
        return 0 <= row < self.height and 0 <= col < self.width

    def has_filament_at_xy(self, x, y, already_transformed=False):
        row, col = self.xy_to_pixel(x, y, already_transformed=already_transformed)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return bool(self.mask[row, col])

    def crop_centered_at_xy(self, x, y, crop_shape, already_transformed=False, fill=False):
        """Return a crop centered at raw printer/camera XY. Areas outside the mask are filled with fill."""
        row, col = self.xy_to_pixel(x, y, already_transformed=already_transformed)
        h, w = map(int, crop_shape)
        out = np.full((h, w), bool(fill), dtype=bool)

        r0, c0 = row - h // 2, col - w // 2
        r1, c1 = r0 + h, c0 + w
        dr0, dc0 = max(0, r0), max(0, c0)
        dr1, dc1 = min(self.height, r1), min(self.width, c1)
        if dr0 >= dr1 or dc0 >= dc1:
            return out

        sr0, sc0 = dr0 - r0, dc0 - c0
        sr1, sc1 = sr0 + (dr1 - dr0), sc0 + (dc1 - dc0)
        out[sr0:sr1, sc0:sc1] = self.mask[dr0:dr1, dc0:dc1]
        return out


def _segment_bounds_in_pixel_basis(segs, theta_deg):
    pts = np.vstack([segs[:, [0, 1]], segs[:, [3, 4]]])
    th = np.deg2rad(theta_deg)
    along = np.array([np.cos(th), np.sin(th)])
    across = np.array([-np.sin(th), np.cos(th)])
    u = pts @ along
    v = pts @ across
    return u.min(), u.max(), v.min(), v.max(), along, across


def rasterize_segments_to_coord_mask(
    segs,
    origin_xy_mm,
    mask_shape,
    px_per_mm=PX_PER_MM,
    theta_deg=0.0,
    line_width_mm=LINE_WIDTH_MM,
):
    """Rasterize already-transformed segments into an explicitly positioned coordinate mask."""
    rows, cols = map(int, mask_shape)
    mask = np.zeros((rows, cols), np.uint8)
    if len(segs) == 0:
        return mask.astype(bool)

    th = np.deg2rad(theta_deg)
    along = np.array([np.cos(th), np.sin(th)])
    across = np.array([-np.sin(th), np.cos(th)])
    origin = np.array(origin_xy_mm, float)
    thickness = max(1, int(round(line_width_mm * float(px_per_mm))))

    for x0, y0, _z0, x1, y1, _z1, _de, _line in segs:
        p0 = np.array([x0, y0], float) - origin
        p1 = np.array([x1, y1], float) - origin
        c0 = int(round(np.dot(p0, along) * px_per_mm))
        r0 = int(round(np.dot(p0, across) * px_per_mm))
        c1 = int(round(np.dot(p1, along) * px_per_mm))
        r1 = int(round(np.dot(p1, across) * px_per_mm))
        cv2.line(mask, (c0, r0), (c1, r1), 255, thickness, cv2.LINE_AA)

    return mask > 0


def make_gcode_coordinate_mask(
    gcode_file=DEFAULT_GCODE_PATH,
    px_per_mm=PX_PER_MM,
    layer_index=0,
    origin_xy_mm=None,
    mask_shape=None,
    margin_mm=5.0,
    theta_deg=0.0,
    line_width_mm=LINE_WIDTH_MM,
    include_previous_layers=False,
    z_tol=0.08,
    flip_x=False,
    flip_y=False,
    swap_xy=False,
    arc_step_mm=0.05,
    return_info=False,
):
    """
    Build a full expected-filament mask with a coordinate system attached.

    If origin_xy_mm and mask_shape are omitted, the mask is tightly fit around the selected G-code.
    For comparison with a merge canvas, pass the merge canvas origin and shape explicitly.
    """
    segs, unsupported = parse_gcode_extrusion_segments(gcode_file, arc_step_mm=arc_step_mm)
    selected, target_z, layer_zs = pick_layer_segments_by_index(
        segs, layer_index=layer_index, include_previous_layers=include_previous_layers, z_tol=z_tol
    )
    selected = transform_segments_xy(selected, flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)

    if len(selected) == 0:
        if origin_xy_mm is None:
            origin_xy_mm = (0.0, 0.0)
        if mask_shape is None:
            mask_shape = (1, 1)
    elif origin_xy_mm is None or mask_shape is None:
        u0, u1, v0, v1, along, across = _segment_bounds_in_pixel_basis(selected, theta_deg)
        pad = float(margin_mm) + 0.5 * float(line_width_mm)
        u0, u1, v0, v1 = u0 - pad, u1 + pad, v0 - pad, v1 + pad

        if origin_xy_mm is None:
            # Pixel [0, 0] represents this world coordinate.
            origin_xy_mm = tuple(u0 * along + v0 * across)
        if mask_shape is None:
            width = int(math.ceil((u1 - u0) * px_per_mm)) + 1
            height = int(math.ceil((v1 - v0) * px_per_mm)) + 1
            mask_shape = (max(1, height), max(1, width))

    mask = rasterize_segments_to_coord_mask(
        selected,
        origin_xy_mm=origin_xy_mm,
        mask_shape=mask_shape,
        px_per_mm=px_per_mm,
        theta_deg=theta_deg,
        line_width_mm=line_width_mm,
    )
    coord_mask = FilamentCoordMask(
        mask=mask,
        origin_xy_mm=tuple(map(float, origin_xy_mm)),
        px_per_mm=float(px_per_mm),
        theta_deg=float(theta_deg),
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
    )

    if not return_info:
        return coord_mask
    return coord_mask, {
        "segments": selected,
        "target_z": target_z,
        "layer_zs": layer_zs,
        "unsupported_arcs": unsupported,
        "origin_xy_mm": coord_mask.origin_xy_mm,
        "mask_shape": coord_mask.shape,
        "px_per_mm": coord_mask.px_per_mm,
    }


if __name__ == "__main__":
    coord_mask, info = make_gcode_coordinate_mask(return_info=True)
    cv2.imwrite("expected_filament_coord.png", coord_mask.mask.astype(np.uint8) * 255)
    print({k: info[k] for k in ["origin_xy_mm", "mask_shape", "target_z", "px_per_mm"]})
