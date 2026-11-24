"""Co-registration of a DSM to a reference elevation model (Nuth & Kaab 2011), shared by anchor.py and check.py.

On land that is not built-up, with slopes of 5-45 deg, the height difference dh = DSM - reference divided by
tan(slope), against the aspect of the slope, is a cosine whose amplitude and phase are the horizontal shift of the DSM
(a shifted surface is too high on slopes facing one way and too low on the others). The shift is found in a few
iterations; the vertical offset is then the median dh on flat open ground, where a surface model and a terrain
model see the same thing. Slope and aspect come from the reference.
"""
import cv2
import numpy as np
from scipy import ndimage

OPEN = [20, 30, 40, 60, 90]          # WorldCover: shrubs, grass, crops, bare, wetland


def slope_aspect(ref, res, blur=1.5):
    """slope (deg) and aspect (deg clockwise from north, the way the slope faces) of a height grid, row 0 north"""
    gs = cv2.GaussianBlur(np.nan_to_num(ref, nan=0).astype(np.float32), (0, 0), blur)
    gy, gx = np.gradient(gs, res)                                 # gy: d/drow (south), gx: d/dcol (east)
    return np.degrees(np.arctan(np.hypot(gx, gy))), np.degrees(np.arctan2(-gx, gy)) % 360


def shifted(a, sx, sy, res):
    """a resampled so that the value at a cell is what a had at (x + sx, y + sy), metres east and north"""
    i, j = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
    return ndimage.map_coordinates(a, [i - sy / res, j + sx / res], order=1, mode='constant', cval=np.nan)


def nuth_kaab(H, ref, sel, slope, aspect, res, iterations=6):
    """horizontal shift (east, north, metres) of H against ref, fitted on the cells of sel"""
    sx = sy = 0.0
    it = 0
    for it in range(iterations):
        d = (shifted(H, sx, sy, res) if it else H) - ref
        m = sel & np.isfinite(d)
        med = np.median(d[m]); nm = 1.4826 * np.median(np.abs(d[m] - med))
        m &= np.abs(d - med) < 3 * nm
        y = d[m] / np.tan(np.radians(slope[m])); ps = np.radians(aspect[m])
        keep = np.abs(y - np.median(y)) < 5 * 1.4826 * np.median(np.abs(y - np.median(y)))
        A = np.c_[np.cos(ps[keep]), np.sin(ps[keep]), np.ones(keep.sum())]
        a_, b_, _ = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        sx, sy = sx + b_, sy + a_
        if np.hypot(a_, b_) < 0.05:
            break
    return float(sx), float(sy), it + 1


def coregister(H, ref, land, wc, res, open_margin_m=0.0):
    """shift and offset of H against ref. open_margin_m: flat open ground used for the offset only where it is that far
    from anything else (a coarse reference mixes the roofs next to a small field into the field's height)."""
    slope, aspect = slope_aspect(ref, res)
    sel = land & (wc != 50) & (slope > 5) & (slope < 45) & np.isfinite(ref)
    sx, sy, it = nuth_kaab(H, ref, sel, slope, aspect, res)
    flat_open = land & np.isin(wc, OPEN) & (slope < 10)
    if open_margin_m > 0:
        flat_open &= ndimage.distance_transform_edt(np.isin(wc, OPEN)) * res >= open_margin_m
    d = shifted(H, sx, sy, res) - ref
    m = flat_open & np.isfinite(d)
    bias = float(np.median(d[m]))
    return dict(shift_east_m=sx, shift_north_m=sy, shift_m=float(np.hypot(sx, sy)), vertical_offset_m=bias,
                iterations=it, cells_slopes=int(sel.sum()), cells_offset=int(m.sum()),
                offset_nmad_m=float(1.4826 * np.median(np.abs(d[m] - bias))))
