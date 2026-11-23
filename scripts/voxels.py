#!/usr/bin/env python3
"""Step 7. The voxel model: the DSM as columns of cubes of voxels.size metres, the sea as a layer of water at 0 m.

    python3 -I scripts/voxels.py data/hiroshima/fb.yaml
    python3 scripts/voxview.py data/hiroshima/fb.yaml       # look at it (a window); render.py makes the pictures

1. The DSM in cells of voxels.size: the median of the measured 2.5 m cells. The anchored DSM (check.py:
   product/dsm_anchored.tif, tied to the GSI DEM by a shift and an offset) when it exists and voxels.anchored is not
   false, else product/dsm.tif as produced; heights above sea level either way, but only the anchored one has the
   right level (the RPCs alone leave it several metres off).
2. Holes of the DSM on land no wider than voxels.fill_m are filled from their rims (source 1). Wider holes stay open:
   no column. A filled hole beside a tall building is most likely ground the slanted views could not see.
3. A cell is a column from voxels.base up to its height, rounded to whole voxels: the top face of the column is the
   height to +-size/2. Only the voxels that can be seen are stored: the top one and, down the sides, those above the
   lowest neighbouring column (an open hole or the edge of the model counts as nothing: the side shows to the base).
4. Water cells (product/water.tif, more than half of the cell) are one voxel whose top is sea level, 0 m (source 2).
Colour: the orthophoto (nadir view, panchromatic) tinted by ESA WorldCover (trees green, fields and bare ground
warmer, water blue); side voxels take their column's colour.

Writes product/voxels.npz: ijk (int16; i east, j north, k up), rgb (uint8), views (uint8: pairs that measured the cell),
source (uint8: 0 measured, 1 filled, 2 water, 3 side), cover (WorldCover class), origin (x, y, z of the outer corner of voxel 0,0,0: grid CRS,
z above sea level), size, dims, unit, crs; and product/voxels.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load

p = load(sys.argv[1])
g = Grid(p)
s = p.voxel_size
f = int(round(s / g.res))
if abs(f * g.res - s) > 1e-6:
    raise SystemExit(f'voxels.size must be a multiple of dsm.res ({g.res} m)')
anchored = p.product / 'dsm_anchored.tif'
src_dsm = anchored if anchored.exists() and p.cfg.get('voxels', {}).get('anchored', True) else p.product / 'dsm.tif'
H = g.read(src_dsm)
water = np.nan_to_num(g.read(p.product / 'water.tif')) > 0
cnt = np.nan_to_num(g.read(p.product / 'dsm_count.tif'))
ortho = g.read(p.product / 'ortho.tif')
wc = np.nan_to_num(g.read(p.vref / 'worldcover.tif')).astype(np.uint8)
hh, ww = g.height // f * f, g.width // f * f


def blocks(a):
    return a[:hh, :ww].reshape(hh // f, f, ww // f, f).transpose(0, 2, 1, 3).reshape(hh // f, ww // f, f * f)


with np.errstate(all='ignore'):
    Hb = np.nanmedian(blocks(H), axis=2).astype(np.float32)
    Ob = np.nanmean(blocks(np.where(ortho > 0, ortho, np.nan)), axis=2)
Wb = blocks(water.astype(np.float32)).mean(axis=2) > 0.5
Cb = blocks(cnt).max(axis=2).astype(np.uint8)
wcb = blocks(wc)                                                      # the most common WorldCover class of the cell
classes = np.unique(wc)
WCb = classes[np.argmax(np.stack([(wcb == k).sum(axis=2) for k in classes]), axis=0)].astype(np.uint8)
Hb[Wb] = np.nan
rows, cols = Hb.shape

# ---- holes on land: fill the narrow ones from their rims
hole = np.isnan(Hb) & ~Wb
lab, n = ndimage.label(hole)
depth = ndimage.distance_transform_edt(hole) * s                      # metres to the nearest non-hole cell
widest = ndimage.maximum(depth, lab, index=np.arange(1, n + 1)) if n else np.array([])
fill = np.isin(lab, np.flatnonzero(2 * widest <= p.voxel_fill_m) + 1)
src = np.zeros((rows, cols), np.uint8)
if fill.any():
    known = np.isfinite(Hb)
    lo_, hi_ = np.nanmin(Hb), np.nanmax(Hb)
    z8 = np.where(known, (Hb - lo_) / (hi_ - lo_), 0).astype(np.float32)
    inp = cv2.inpaint(z8, fill.astype(np.uint8), 3, cv2.INPAINT_NS) * (hi_ - lo_) + lo_
    Hb[fill] = inp[fill]
    src[fill] = 1
src[Wb] = 2
Hb[Wb] = 0.0
Ob = np.where(np.isfinite(Ob), Ob, np.nan)
if np.isnan(Ob).any():                                                # colour for cells without an ortho value
    idx = ndimage.distance_transform_edt(np.isnan(Ob), return_distances=False, return_indices=True)
    Ob = Ob[tuple(idx)]

# ---- columns: top voxel index; only what can be seen
top = np.full((rows, cols), -1, np.int32)
ok = np.isfinite(Hb)
top[ok] = np.maximum(1, np.round((Hb[ok] - p.voxel_base) / s)).astype(np.int32) - 1
pad = np.pad(top, 1, constant_values=-1)
nbr_min = np.minimum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
klo = np.clip(nbr_min + 1, 0, None)
klo = np.minimum(klo, top)
r_, c_ = np.nonzero(top >= 0)
kt, kl = top[r_, c_], klo[r_, c_]
n_per = kt - kl + 1
R_ = np.repeat(r_, n_per); C_ = np.repeat(c_, n_per)
start = np.repeat(np.cumsum(n_per) - n_per, n_per)
K = np.repeat(kl, n_per) + (np.arange(n_per.sum()) - start)
is_top = K == np.repeat(kt, n_per)
i = C_.astype(np.int16)                                               # east
j = (rows - 1 - R_).astype(np.int16)                                  # north (row 0 is the north edge)
ijk = np.c_[i, j, K.astype(np.int16)]
gray = np.clip(Ob[R_, C_], 0, 255)
cover = WCb[R_, C_].astype(np.uint8)
TINT = {10: (0.80, 1.00, 0.72), 20: (0.92, 1.00, 0.80), 30: (0.90, 1.00, 0.78), 40: (1.00, 0.97, 0.78), 60: (1.00, 0.95, 0.85)}
rgb = np.repeat(gray[:, None], 3, axis=1)
for k, t in TINT.items():                                             # the panchromatic grey, tinted by WorldCover
    rgb[cover == k] *= np.array(t)
wat = src[R_, C_] == 2
rgb[wat] = 0.35 * rgb[wat] + 0.65 * np.array([40, 95, 165])
rgb = np.clip(rgb, 0, 255).astype(np.uint8)
source = np.where(is_top, src[R_, C_], 3).astype(np.uint8)
views = Cb[R_, C_]
x0, y0, x1, y1 = g.bounds
origin = np.array([x0, y1 - rows * s, p.voxel_base])
dims = np.array([cols, rows, int(K.max()) + 1])
np.savez_compressed(p.product / 'voxels.npz', ijk=ijk, rgb=rgb, views=views, source=source, cover=cover, origin=origin, size=s,
                    dims=dims, unit='m', crs=str(g.crs))
info = dict(dsm=src_dsm.name, size=s, base=p.voxel_base, columns=int(ok.sum()), voxels=int(len(ijk)), dims=dims.tolist(),
            land_columns=int((ok & ~Wb).sum()), water_columns=int(Wb.sum()), filled_columns=int(fill.sum()),
            open_hole_cells=int((hole & ~fill).sum()), open_hole_share_of_land=float((hole & ~fill).sum() / max((~Wb).sum(), 1)),
            top_m=float(p.voxel_base + (K.max() + 1) * s))
json.dump(info, open(p.product / 'voxels.json', 'w'), indent=1)
print(f'{info["voxels"]:,} voxels of {s:g} m ({dims[0]} x {dims[1]} x {dims[2]}), {info["land_columns"]:,} land columns '
      f'({info["filled_columns"]:,} filled), {info["water_columns"]:,} water; open holes {info["open_hole_share_of_land"]:.1%} of the land')
