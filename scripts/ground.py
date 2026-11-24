#!/usr/bin/env python3
"""Step 7. The ground under roofs and canopy, from the DSM itself.

    python3 -I scripts/ground.py data/hiroshima/fnb_global.yaml

A flood spreads over the ground between the buildings, and the high ground people can run to is ground, not a roof.
A stereo DSM sees roofs and treetops; at 2.5 m it sees few of the streets between them. Two estimates of the ground:

1. A robust opening, everywhere: over a window wider than most buildings (ground.window_m), the low percentile of
   the anchored DSM (ground.percentiles[0]) is the ground between them, and the high percentile of that over the same
   window (ground.percentiles[1]) puts back the shape of hills wider than the window. Percentiles rather than min and
   max, so that single noisy cells do not pull the surface down. Water takes no part: it counts as higher than any
   land in the first pass and lower in the second, so the shore is not dragged down to the sea.
2. Under a dense downtown the DSM sees no ground at all (Hiroshima: round Hatchobori even the lowest 5% of the DSM
   within 600 m stand 2-4 m above the laser DEM's streets), and the opening stays on the roofs. There the ground
   comes from the ground the DSM does see nearby: open land (parks, river banks, fields, plazas: WorldCover grass,
   bare, crops, shrubs, wetland, mostly open 10 m cells that are not raised above their surroundings), spread over
   the built-up land by normalized convolution (a Gaussian of ground.interp_sigma_m). It is used where the land
   within ground.built_up_window_m is mostly built-up (ground.built_up_min) and flat (ground.flat_max_deg), blended
   in over a short range; hills keep the opening (they are forest, or slope).
The ground is never above the DSM. Computed at ground.res (10 m), back on the DSM grid by bilinear interpolation.
The settings were chosen on Hiroshima against the GSI laser DEM (floods.py, a few were tried): on flat built-up land
the ground model is +0.3 m off GSI in the median (the opening alone: +1.2 m), and the downtown is no longer a hill.

Writes product/ground.tif (heights above the geoid, like dsm_anchored.tif) and product/ground.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load

p = load(sys.argv[1])
g = Grid(p)
H = g.read(p.product / 'dsm_anchored.tif')
water = np.nan_to_num(g.read(p.product / 'water.tif')) > 0
wc = np.nan_to_num(g.read(p.vref / 'worldcover.tif')).astype(np.uint8)
BIG = 1e5
lo_p, hi_p = p.ground_percentiles
f = int(round(p.ground_res / g.res))
res = f * g.res


def down(a, how=np.nanmean):
    """per block of f x f cells (NaN ignored), padded up to whole blocks"""
    hh, ww = -(-g.height // f) * f, -(-g.width // f) * f
    b = np.full((hh, ww), np.nan, np.float32); b[:g.height, :g.width] = a
    with np.errstate(all='ignore'):
        return how(b.reshape(hh // f, f, ww // f, f), axis=(1, 3))


def up(a):
    """back on the DSM grid: bilinear between block centres, gaps from the nearest"""
    ok = np.isfinite(a)
    a = a[tuple(ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True))]
    r = (np.arange(g.height) + 0.5) / f - 0.5
    c = (np.arange(g.width) + 0.5) / f - 0.5
    R, C = np.meshgrid(r, c, indexing='ij')
    return ndimage.map_coordinates(a, [R, C], order=1, mode='nearest').astype(np.float32)


# holes in the DSM take no part either (a hole filled from a tall neighbour would stand as a hill): like water
w10 = down(water.astype(np.float32)) > 0.5
h10 = down(np.where(water, np.nan, H))
hole10 = np.isnan(h10) & ~w10
h10 = np.where(np.isnan(h10), 0.0, h10)

# 1. the robust opening
size = int(round(p.ground_window_m / res)) | 1
low = ndimage.percentile_filter(np.where(w10 | hole10, BIG, h10), lo_p, size=size)
lo2 = np.where(w10 | hole10 | (low >= BIG / 2), -BIG, low)
op = ndimage.percentile_filter(lo2, hi_p, size=size)
op = np.where((op <= -BIG / 2) | w10, np.nan, op)
G = up(op)
info = dict(window_m=p.ground_window_m, res_m=res, percentiles=[lo_p, hi_p])

# 2. under dense built-up land: the ground seen nearby, spread by normalized convolution
if p.ground_interp_sigma_m > 0:
    hmed = down(np.where(water, np.nan, H), np.nanmedian)
    obs = down(np.isin(wc, [20, 30, 40, 60, 90]).astype(np.float32)) > 0.75
    obs = ndimage.binary_erosion(obs, iterations=1) & np.isfinite(hmed) & ~w10 & (hmed < low + 2.0)
    s = p.ground_interp_sigma_m / res
    num = ndimage.gaussian_filter(np.where(obs, hmed, 0.0), s)
    den = ndimage.gaussian_filter(obs.astype(np.float32), s)
    with np.errstate(all='ignore'):
        I = np.where(den > 1e-7, num / den, np.nan)              # far observations count too, just less
    bu = ndimage.uniform_filter(np.nan_to_num(down((wc == 50).astype(np.float32))), int(round(p.ground_built_up_window_m / res)) | 1)
    sl = []
    for surf in (I, op):
        gy, gx = np.gradient(ndimage.gaussian_filter(np.nan_to_num(surf, nan=0.0), 3), res)
        sl.append(np.degrees(np.arctan(np.hypot(gx, gy))))
    # flat in the spread ground, and in the opening too unless the land is all buildings: a hill foot keeps the
    # opening, a downtown whose roofs make a plateau in the opening does not
    slope = np.maximum(sl[0], sl[1] * np.clip((0.9 - bu) / 0.1, 0, 1))
    wgt = np.clip((bu - p.ground_built_up_min) / 0.15, 0, 1) * np.clip((p.ground_flat_max_deg - slope) / 1.5, 0, 1)
    wgt = np.where(np.isfinite(I), wgt, 0.0)
    W, Iu = up(wgt), up(I)
    G = W * np.minimum(Iu, G) + (1 - W) * G
    info.update(interp_sigma_m=p.ground_interp_sigma_m, built_up_window_m=p.ground_built_up_window_m,
                built_up_min=p.ground_built_up_min, flat_max_deg=p.ground_flat_max_deg,
                observations_share=float(obs.mean()), interpolated_share_of_land=float((W[~water] > 0.5).mean()))
G = np.where(np.isfinite(H), np.minimum(G, H), G).astype(np.float32)
G[water] = np.nan
g.write(p.product / 'ground.tif', G)
d = (H - G)[~water & np.isfinite(H)]
info.update(dsm_minus_ground_median_m=float(np.median(d)), dsm_minus_ground_p90_m=float(np.percentile(d, 90)),
            land_more_than_3m_above_ground=float((d > 3).mean()))
json.dump(info, open(p.product / 'ground.json', 'w'), indent=1)
print(f'ground: opening of {p.ground_window_m:g} m (percentiles {lo_p:g}/{hi_p:g})' +
      (f'; under flat built-up land ({info["interpolated_share_of_land"]:.0%} of the land) the open ground seen nearby, '
       f'sigma {p.ground_interp_sigma_m:g} m' if p.ground_interp_sigma_m > 0 else '') +
      f'; the DSM stands {info["dsm_minus_ground_median_m"]:.1f} m above it in the median, more than 3 m on {info["land_more_than_3m_above_ground"]:.0%} of the land')
