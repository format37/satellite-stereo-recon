#!/usr/bin/env python3
"""Step 5. The DSM: the pairs fused, heights above sea level, the holes explained; an orthophoto.

    python3 -I scripts/dsm.py data/hiroshima/fb.yaml

1. Each pair's DSM (stereo/<pair>/run-DEM.tif, heights above the WGS84 ellipsoid) onto the run's grid. Cells over
   water (product/water.tif) are dropped: stereo on water returns noise, and the water level is known anyway.
2. More than one pair: per cell one height from the pairs that measured it (dsm.fuse): median; mean; weighted = the
   inverse-variance mean, weight (B/H)^2 (forward-backward 1, nadir pairs 0.25), after dropping a pair more than
   5 m off the median of three. The count of pairs per cell is kept.
3. Heights above sea level: H = h - N, N = the geoid model dsm.geoid above the ellipsoid. gsigeo2011 gives heights
   above T.P. (Tokyo Bay mean sea level, the Japanese datum, the one the GSI DEM and Japanese tide tables use);
   egm2008 the global geoid. Both N are kept in dsm.json; here they differ by 0.2 m.
4. The holes left on land: 2 cloud (a view of the cell's pairs has a cloud over it), 3 no match (occlusion behind
   buildings and steep slopes, shadow, little texture), next to 1 water.
5. Orthophoto: the nadir view through its RPC onto the DSM (holes filled from their edges, the sea at sea level).

Writes product/dsm.tif (m above sea level, nan = no measurement), dsm_ellipsoid.tif, dsm_count.tif (pairs per cell),
holes.tif (0 measured, 1 water, 2 cloud, 3 no match), ortho.tif (8-bit, nadir), dsm_<pair>.tif (each pair, above
sea level), dsm.json.
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load, pair_name
import rpc as R

p = load(sys.argv[1])
g = Grid(p)
p.product.mkdir(parents=True, exist_ok=True)
if p.reuse:                                          # the products of the earlier steps come with the reused variant
    import shutil
    for f in ('views.json', 'masks.json', 'water.tif'):
        shutil.copy2(p.project / 'out' / p.reuse / 'product' / f, p.product / f)
ref = p.vref
water = g.read(p.product / 'water.tif')
water = np.nan_to_num(water) > 0
N = g.read(ref / f'geoid_{p.geoid}.tif')
N_egm = g.read(ref / 'geoid_egm2008.tif')
info = {'geoid': p.geoid, 'N_min': float(np.nanmin(N)), 'N_max': float(np.nanmax(N)),
        'N_minus_egm2008': float(np.nanmean(N - N_egm)), 'pairs': {}}

stack = []
for pr in p.pairs:
    f = p.stereo / pair_name(pr) / 'run-DEM.tif'
    with rasterio.open(f) as d:
        same = d.transform.almost_equals(g.transform) and (d.width, d.height) == (g.width, g.height)
    h = g.read(f) if same else g.warp(str(f), 'bilinear')
    h[h < -1000] = np.nan
    h[water] = np.nan
    stack.append(h)
    land = ~water
    e = g.read(p.stereo / pair_name(pr) / 'run-IntersectionErr.tif') if (p.stereo / pair_name(pr) / 'run-IntersectionErr.tif').exists() else None
    info['pairs'][pair_name(pr)] = dict(grid_aligned=bool(same), land_coverage=float(np.isfinite(h[land]).mean()),
                                        intersection_error_median=float(np.nanmedian(e[np.isfinite(h)])) if e is not None else None)
    g.write(p.product / f'dsm_{pair_name(pr)}.tif', h - N)
    print(f'pair {pair_name(pr)}: {np.isfinite(h[land]).mean():.1%} of the land measured'
          f'{"" if same else " (resampled onto the grid)"}')
S = np.stack(stack)
cnt = np.isfinite(S).sum(0).astype(np.uint8)
with np.errstate(all='ignore'), warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    if p.fuse == 'weighted':
        # inverse variance: a pair's height error goes with 1 / (B/H), so its weight with (B/H)^2. Where three pairs
        # measured a cell, one that is more than 5 m off their median is dropped first.
        vj = json.load(open(p.product / 'views.json'))['pairs']
        w = np.array([vj[pair_name(pr)]['base_to_height'] ** 2 for pr in p.pairs], np.float32)[:, None, None]
        med = np.nanmedian(S, axis=0)
        keep = np.isfinite(S) & ~((np.abs(S - med) > 5) & (cnt >= 3))
        h = np.sum(np.where(keep, S, 0) * w, axis=0) / np.sum(keep * w, axis=0)
        info['weights'] = {pair_name(pr): float(x) for pr, x in zip(p.pairs, w.ravel())}
    elif p.fuse == 'median':
        h = np.nanmedian(S, axis=0)
    else:
        h = np.nanmean(S, axis=0)
h = h.astype(np.float32)
H = (h - N).astype(np.float32)
g.write(p.product / 'dsm_ellipsoid.tif', h)
g.write(p.product / 'dsm.tif', H)
g.write(p.product / 'dsm_count.tif', cnt, nodata=255, dtype='uint8')

# ---- the holes on land: clouds of the views of the pairs, else no match
holes = np.zeros(g.shape, np.uint8)
holes[water] = 1
cloudy = np.zeros(g.shape, bool)
gsi = g.read(ref / 'gsi.tif')
terr = np.where(np.isfinite(h), h, np.where(np.isfinite(gsi), gsi + N, N))         # where to look from, ellipsoidal
st = 4                                                                              # 10 m is enough for this
lon, lat = g.lonlat(st)
ts = terr[::st, ::st]
for v in p.views:
    m = cv2.imread(str(p.prep / f'{v}_mask.png'), cv2.IMREAD_GRAYSCALE)
    rc = R.read_tif(p.prep / f'{v}_m.tif')
    r, c = R.project(rc, lon, lat, ts)
    r, c = np.round(r).astype(int), np.round(c).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < m.shape[0]) & (c < m.shape[1])
    cl = np.zeros(r.shape, bool); cl[ok] = m[r[ok], c[ok]] == 2
    cl = cv2.resize(cl.astype(np.uint8), (g.width, g.height), interpolation=cv2.INTER_NEAREST).astype(bool)
    cloudy |= cl
land = ~water
holes[land & np.isnan(H) & cloudy] = 2
holes[land & np.isnan(H) & ~cloudy] = 3
g.write(p.product / 'holes.tif', holes, nodata=255, dtype='uint8')
info.update(land_share=float(land.mean()), measured_of_land=float(np.isfinite(H[land]).mean()),
            cloud_holes_of_land=float((holes[land] == 2).mean()), nomatch_holes_of_land=float((holes[land] == 3).mean()),
            count_of_land={int(k): float((cnt[land] == k).mean()) for k in range(len(p.pairs) + 1)},
            height_p01=float(np.nanpercentile(H, 1)), height_p99=float(np.nanpercentile(H, 99)),
            height_min=float(np.nanmin(H)), height_max=float(np.nanmax(H)))
print(f'fused ({p.fuse} of {len(p.pairs)} pair{"s" * (len(p.pairs) > 1)}): {info["measured_of_land"]:.1%} of the land measured; '
      f'holes: cloud {info["cloud_holes_of_land"]:.2%}, no match {info["nomatch_holes_of_land"]:.1%} of the land')
print(f'heights above sea level ({p.geoid}): 1% {info["height_p01"]:.1f} m, 99% {info["height_p99"]:.1f} m '
      f'(min {info["height_min"]:.1f}, max {info["height_max"]:.1f}); N = {info["N_min"]:.2f}..{info["N_max"]:.2f} m')

# ---- orthophoto from the nadir view (else the first view): holes filled from their nearest measured cell
ov = 'N' if (p.prep / 'N.tif').exists() else p.views[0]
idx = ndimage.distance_transform_edt(np.isnan(h), return_distances=False, return_indices=True)
hf = h[tuple(idx)] if np.isfinite(h).any() else np.zeros_like(h)
hf = np.where(water, N, hf).astype(np.float32)
hf = np.where(np.isfinite(h), h, cv2.GaussianBlur(hf, (0, 0), 3))
with rasterio.open(p.prep / f'{ov}.tif') as d:
    img = d.read(1).astype(np.float32); rc = R.from_rasterio(d.rpcs)
ortho = np.zeros(g.shape, np.uint8)
lon, lat = g.lonlat(1)
for i0 in range(0, g.height, 512):
    sl = slice(i0, i0 + 512)
    r, c = R.project(rc, lon[sl], lat[sl], hf[sl])
    ortho[sl] = np.clip(cv2.remap(img, c.astype(np.float32), r.astype(np.float32), cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 255).astype(np.uint8)
g.write(p.product / 'ortho.tif', ortho, nodata=0, dtype='uint8')
info['ortho_view'] = ov
json.dump(info, open(p.product / 'dsm.json', 'w'), indent=1)
