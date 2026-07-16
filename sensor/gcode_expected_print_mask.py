from pathlib import Path
import math
import re

import cv2
import numpy as np
from mask_to_laser import LASER_W, LASER_L
DEFAULT_GCODE_PATH = Path("P4_one_layer_annular_disc_60OD_12ID_0p20H.gcode")
MASK_SHAPE = (LASER_W, LASER_L)  # rows/across laser width, cols/along laser length
THETA_DEG = 0.0         # 0 => mask columns point along printer +X
LINE_WIDTH_MM = 0.45
VIEW_CENTER_OFFSET_MM = (0.0, 0.0)

_WORD_RE = re.compile(r"([A-Za-z])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

XYZ_RE = re.compile(r'(?:^|[_-])x_([-+]?\d*\.?\d+)_y_([-+]?\d*\.?\d+)_z_([-+]?\d*\.?\d+)')

def parse_frame_xyz(path):
    m = XYZ_RE.search(Path(path).stem)
    if not m:
        raise ValueError(f'Could not parse x/y/z from {path}')
    return tuple(map(float, m.groups()))
def parse_gcode_extrusion_segments(gcode_file=DEFAULT_GCODE_PATH):
    """Return extrusion line segments as [x0,y0,z0,x1,y1,z1,de,line_no]."""
    pos = dict(X=0.0, Y=0.0, Z=np.nan, E=0.0)
    absolute_xyz, absolute_e = True, True
    segs, unsupported = [], []

    for line_no, raw in enumerate(Path(gcode_file).read_text(errors="ignore").splitlines(), 1):
        code = raw.split(";", 1)[0].strip()
        if not code:
            continue
        cmd = code.split()[0].upper()

        if cmd == "G90":
            absolute_xyz = True
        elif cmd == "G91":
            absolute_xyz = False
        elif cmd == "M82":
            absolute_e = True
        elif cmd == "M83":
            absolute_e = False
        elif cmd == "G92":
            vals = {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}
            for k in "XYZE":
                if k in vals:
                    pos[k] = vals[k]
        elif cmd in {"G0", "G1"}:
            vals = {k.upper(): float(v) for k, v in _WORD_RE.findall(code)}
            old = pos.copy()

            for k in "XYZ":
                if k in vals:
                    pos[k] = vals[k] if absolute_xyz else pos[k] + vals[k]

            e0 = pos["E"]
            if "E" in vals:
                pos["E"] = vals["E"] if absolute_e else pos["E"] + vals["E"]

            de = pos["E"] - e0
            dxy = math.hypot(pos["X"] - old["X"], pos["Y"] - old["Y"])
            if cmd == "G1" and de > 1e-9 and dxy > 1e-9:
                z0 = old["Z"] if not np.isnan(old["Z"]) else pos["Z"]
                z1 = pos["Z"] if not np.isnan(pos["Z"]) else z0
                segs.append((old["X"], old["Y"], z0, pos["X"], pos["Y"], z1, de, line_no))
        elif cmd in {"G2", "G3"} and "E" in code.upper():
            unsupported.append((line_no, raw))

    return np.array(segs, dtype=float) if segs else np.zeros((0, 8), dtype=float), unsupported


def layer_z_values(segs, decimals=5):
    if len(segs) == 0:
        return np.array([], dtype=float)
    return np.unique(np.round((segs[:, 2] + segs[:, 5]) / 2, decimals))


def pick_layer_segments_nearest_z(segs, z, include_previous_layers=False, z_tol=0.08):
    """Select the extrusion layer nearest to measured z."""
    zs = layer_z_values(segs)
    if len(zs) == 0:
        return segs, None, zs
    target_z = zs[np.argmin(np.abs(zs - z))]
    mid_z = (segs[:, 2] + segs[:, 5]) / 2
    keep = mid_z <= target_z + z_tol if include_previous_layers else np.abs(mid_z - target_z) <= z_tol
    return segs[keep], target_z, zs


def expected_print_mask_from_segments(
    segs,
    x,
    y,
    px_to_mm,
    mask_shape=MASK_SHAPE,
    theta_deg=THETA_DEG,
    view_center_offset_mm=VIEW_CENTER_OFFSET_MM,
    line_width_mm=LINE_WIDTH_MM,
):
    """Project printer-XY extrusion segments into local laser-mask pixels."""
    rows, cols = map(int, mask_shape)
    mask = np.zeros((rows, cols), np.uint8)

    px_per_mm = 1.0 / float(px_to_mm)
    cam = np.array([x, y], float) + np.array(view_center_offset_mm, float)
    th = np.deg2rad(theta_deg)
    along = np.array([np.cos(th), np.sin(th)])
    across = np.array([-np.sin(th), np.cos(th)])
    half = np.array([cols / 2, rows / 2], float)
    thickness = max(1, int(round(line_width_mm * px_per_mm)))

    for x0, y0, _z0, x1, y1, _z1, _de, _line in segs:
        p0, p1 = np.array([x0, y0]) - cam, np.array([x1, y1]) - cam
        c0 = int(round(np.dot(p0, along) * px_per_mm + half[0]))
        r0 = int(round(np.dot(p0, across) * px_per_mm + half[1]))
        c1 = int(round(np.dot(p1, along) * px_per_mm + half[0]))
        r1 = int(round(np.dot(p1, across) * px_per_mm + half[1]))
        cv2.line(mask, (c0, r0), (c1, r1), 255, thickness, cv2.LINE_AA)

    return mask > 0


def gcode_expected_print_mask(
    x,
    y,
    z,
    px_to_mm=(1 / 26.13),
    gcode_file=DEFAULT_GCODE_PATH,
    mask_shape=MASK_SHAPE,
    theta_deg=THETA_DEG,
    view_center_offset_mm=VIEW_CENTER_OFFSET_MM,
    line_width_mm=LINE_WIDTH_MM,
    include_previous_layers=False,
    z_tol=0.08,
):
    """
    Return a bool mask of the G-code extrusion expected under the camera/laser view.

    Parameters
    ----------
    x, y, z : float
        Absolute camera/toolhead position in printer mm.
    px_to_mm : float
        Millimeters per pixel. If you have pixels/mm, pass 1 / pixels_per_mm.
    gcode_file : path-like
        G-code file to parse. Defaults to DEFAULT_GCODE_PATH.
    """
    segs, unsupported = parse_gcode_extrusion_segments(gcode_file)
    selected, _target_z, _layer_zs = pick_layer_segments_nearest_z(
        segs, z, include_previous_layers=include_previous_layers, z_tol=z_tol
    )
    return expected_print_mask_from_segments(
        selected,
        x=x,
        y=y,
        px_to_mm=px_to_mm,
        mask_shape=mask_shape,
        theta_deg=theta_deg,
        view_center_offset_mm=view_center_offset_mm,
        line_width_mm=line_width_mm,
    )


# Short alias.
expected_print_mask = gcode_expected_print_mask
