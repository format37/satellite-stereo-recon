#!/usr/bin/env python3
"""Step 3. Masks: the sea and the clouds in every view's crop; the stereo inputs with both blanked out.

    python3 -I scripts/masks.py data/hiroshima/fb.yaml

Water. ESA WorldCover class 80 on the grid (ref/worldcover.tif), shrunk by masks.water_shrink_m so that quays and
sea walls stay in the images. Water joined to the edge of the box through other water (bridges up to 60 m long
do not cut it) is the sea, with the tidal rivers; the rest is inland water. In a view, a pixel is water when the
ground it sees at sea level (h = the geoid N, about 33 m above the ellipsoid here) falls into the water mask.

Clouds. A cloud floats a kilometre or more above the ground, so each view sees it over a different spot of ground:
a slanted view puts it 0.5 km away from where the nadir view does for every km of height. A bright roof sits over
the same spot in every view. So a bright blob of one view (masks.cloud_dn, area >= masks.cloud_min_m2) is a cloud
when, at the ground behind it (GSI terrain + geoid), the other views are not bright; within 40 px (100 m) counts as
"the same spot", which tolerates buildings up to about 100 m between the slanted views. Clouds are grown by
masks.cloud_grow_m. The test needs two views; masks.cloud_views lists them (the nadir view helps even when it is in
no pair). The ground under a cloud of one view can still be matched in another pair: that is what the nadir buys.

Writes prep/<view>_m.tif (the crop with water and clouds set to 0, the input of stereo and bundle adjustment),
prep/<view>_mask.png (0 kept, 1 water, 2 cloud, 3 no data in the scene), prep/<view>_mask.jpg (a look: water blue,
cloud red), product/water.tif (on the grid: 0 land, 1 sea and tidal rivers, 2 inland water), product/masks.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load
import rpc as R

p = load(sys.argv[1])
if p.reuse:
    raise SystemExit(f'{p.name} reuses {p.reuse}: nothing to do here, run from dsm on')
g = Grid(p)
ref = p.vref
info = {'views': {}}

N = g.read(ref / f'geoid_{p.geoid}.tif')
n_sea = float(np.nanmean(N))                                                             # sea level above the ellipsoid
gsi = g.read(ref / 'gsi.tif')
terr = np.where(np.isfinite(gsi), gsi, 0.0) + np.where(np.isfinite(N), N, n_sea)          # ellipsoidal terrain, sea = geoid
terr = cv2.GaussianBlur(terr.astype(np.float32), (0, 0), 4)                              # 10 m: a surface to project onto
terrain = lambda lon, lat: g.sample(terr, lon, lat)

views = list(dict.fromkeys(p.views + (p.cloud_views if p.clouds else [])))
img, rpcs, prof = {}, {}, {}
for v in views:
    with rasterio.open(p.prep / f'{v}.tif') as ds:
        img[v] = ds.read(1); prof[v] = ds.profile; rpcs[v] = R.from_rasterio(ds.rpcs)

# ---- water, on a 10 m grid over everything the crops see (they reach beyond the box), and on the run's grid
ll = [R.localize(rpcs[v], [0, 0, a.shape[0] - 1, a.shape[0] - 1], [0, a.shape[1] - 1, 0, a.shape[1] - 1], n_sea)
      for v, a in img.items()]
wide = [min(x[0].min() for x in ll), min(x[1].min() for x in ll), max(x[0].max() for x in ll), max(x[1].max() for x in ll)]
gw = Grid(p, res=10, aoi=wide)
wfile = ref / 'worldcover_wide.tif'
if wfile.exists() and gw.read(wfile).shape == gw.shape:
    wc = np.nan_to_num(gw.read(wfile))
else:
    from reference import worldcover
    wc = worldcover(gw, wide)
    gw.write(wfile, wc.astype(np.uint8), nodata=0)
water = (wc == 80) & (p.water != 'none')
sh = int(round(p.water_shrink_m / gw.res))
if sh > 0:
    water = ndimage.binary_erosion(water, structure=ndimage.generate_binary_structure(2, 1), iterations=sh)
joined, _ = ndimage.label(ndimage.binary_dilation(water, iterations=3) | water)        # bridges up to 60 m do not cut it
edge = np.setdiff1d(np.unique(np.r_[joined[0], joined[-1], joined[:, 0], joined[:, -1]]), [0])
sea = water & np.isin(joined, edge)
Wwide = np.where(sea, 1, np.where(water, 2, 0)).astype(np.uint8)
W = np.nan_to_num(g.warp_array(Wwide.astype(np.float32), gw.transform, gw.crs, 'nearest', nodata=255)).astype(np.uint8)
g.write(p.product / 'water.tif', W, nodata=255, dtype='uint8')
info['water_source'] = p.water
info['grid_sea'] = float((W == 1).mean()); info['grid_inland_water'] = float((W == 2).mean())
print(f'water on the grid: sea and tidal rivers {(W == 1).mean():.1%}, inland water {(W == 2).mean():.2%} '
      f'(WorldCover read over {gw.width * 10 / 1e3:.1f} x {gw.height * 10 / 1e3:.1f} km, all the crops see)')


def ground_map(v, h, grid, step=16):
    """(row, col) on `grid` of the ground each pixel of view v sees at height h: coarse inverse, interpolated"""
    H, Wd = img[v].shape
    rr = np.unique(np.r_[np.arange(0, H, step), H - 1]); cc = np.unique(np.r_[np.arange(0, Wd, step), Wd - 1])
    Rg, Cg = np.meshgrid(rr, cc, indexing='ij')
    lon, lat = R.localize(rpcs[v], Rg, Cg, h)
    gi, gj = (x.astype(np.float32) for x in grid.lonlat_to_ij(lon, lat))
    my = np.interp(np.arange(H), rr, np.arange(len(rr))).astype(np.float32)       # full-res pixel -> coarse sample index
    mx = np.interp(np.arange(Wd), cc, np.arange(len(cc))).astype(np.float32)
    MX, MY = np.meshgrid(mx, my)
    fi = cv2.remap(gi, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fj = cv2.remap(gj, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return fi, fj


# ---- clouds: bright pixels that no other view sees bright at the same ground, for any height up to 100 m above it.
# Works at half resolution (2 x 2 means, 5 m). For a bright pixel of view v: the point where its ray meets the
# terrain, then the ray from there up to 100 m; that stretch of ray projected into another view w is a short piece
# of its epipolar line (40 px of a slanted view per 100 m from the other slanted one, 20 px from the nadir). A roof,
# a white tank, a sports ground up to 100 m tall is bright somewhere on that piece; a cloud 0.5 km up or more is not.
def half(a):
    H2, W2 = a.shape[0] // 2, a.shape[1] // 2
    return a[:2 * H2, :2 * W2].reshape(H2, 2, W2, 2).mean(axis=(1, 3))


DZ = np.array([0, 10, 20, 30, 45, 60, 80, 100.0])
TOL = int(p.cfg.get('masks', {}).get('cloud_tolerance_px', 4))           # full-res px: what the views' RPCs disagree by
cloud = {v: np.zeros(img[v].shape, bool) for v in views}
lo, cand, tol = {}, {}, {}
for v in views:
    a = img[v]
    thr = 200.0 if p.cloud_dn == 'auto' else float(p.cloud_dn)
    lo[v] = half(a)
    cand[v] = lo[v] >= thr
    k = 2 * max(1, TOL // 2) + 1
    tol[v] = cv2.dilate((lo[v] >= thr - 30).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
min_half = p.cloud_min_m2 / 25.0
for v in (views if p.clouds and len(views) > 1 else []):
    rows, cols = np.nonzero(cand[v])
    fr, fc = rows * 2 + 0.5, cols * 2 + 0.5                                  # the centre of the 2 x 2 block, full res
    lon0, lat0, h0 = R.localize_dem(rpcs[v], fr, fc, terrain, h0=n_sea)
    lon1, lat1 = R.localize(rpcs[v], fr, fc, h0 + 100)
    seen = np.zeros(rows.size, bool); valid = np.zeros(rows.size, bool)
    for w in views:
        if w == v:
            continue
        Hw, Ww = lo[w].shape
        for dz in DZ:
            t = dz / 100
            rw, cw = R.project(rpcs[w], lon0 + (lon1 - lon0) * t, lat0 + (lat1 - lat0) * t, h0 + dz)
            iw, jw = np.floor((rw + 0.5) / 2).astype(int), np.floor((cw + 0.5) / 2).astype(int)
            inside = (iw >= 0) & (jw >= 0) & (iw < Hw) & (jw < Ww)
            ok = np.zeros(rows.size, bool); ok[inside] = lo[w][iw[inside], jw[inside]] > 0
            valid |= ok
            seen[ok] |= tol[w][iw[ok], jw[ok]]
    m = np.zeros(lo[v].shape, np.uint8)
    m[rows[valid & ~seen], cols[valid & ~seen]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))        # speckle: glints, odd pixels
    lab, n = ndimage.label(m)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    m = np.isin(lab, np.flatnonzero(sizes >= min_half) + 1).astype(np.uint8)
    # the dimmer body of a cloud: pixels >= 140 joined to a core, no farther than 200 m from it
    near_core = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) <= 40 if m.any() else m.astype(bool)
    lab, _ = ndimage.label((lo[v] >= 140) & near_core)
    m = (np.isin(lab, np.setdiff1d(np.unique(lab[m > 0]), [0])) | (m > 0)).astype(np.uint8)
    m = ndimage.binary_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))).astype(np.uint8)
    core = int(m.sum())
    m = (cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) <= p.cloud_grow_m / 5.0).astype(np.uint8) if core else m
    full = np.zeros(img[v].shape, np.uint8)
    full[:2 * m.shape[0], :2 * m.shape[1]] = cv2.resize(m, (2 * m.shape[1], 2 * m.shape[0]), interpolation=cv2.INTER_NEAREST)
    cloud[v] = full.astype(bool)
    n_cl = ndimage.label(m)[1]
    info['views'][v] = dict(cloud_dn=thr, candidates=int(rows.size), unexplained=int((valid & ~seen).sum()),
                            clouds=int(n_cl), cloud_core_ha=core * 25 / 1e4, cloud_share=float(cloud[v].mean()), tolerance_px=TOL)
    print(f'{v}: {rows.size} half-res pixels >= {thr:.0f}, {(valid & ~seen).sum()} seen bright by no other view; '
          f'{n_cl} clouds, {core * 25 / 1e4:.1f} ha, {cloud[v].mean():.2%} of the crop after growing {p.cloud_grow_m:.0f} m')

# ---- water in the views, the stereo inputs
for v in views:
    a = img[v]
    gi, gj = ground_map(v, n_sea, gw)
    wv = cv2.remap(Wwide, gj, gi, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = np.zeros(a.shape, np.uint8)
    m[wv > 0] = 1
    m[cloud[v]] = 2
    m[a == 0] = 3
    cv2.imwrite(str(p.prep / f'{v}_mask.png'), m)
    am = np.where(m > 0, 0, a).astype(np.uint8)
    with rasterio.open(p.prep / f'{v}_m.tif', 'w', **{k: x for k, x in prof[v].items() if k not in ('nodata', 'transform', 'crs')},
                       nodata=0, rpcs=R.to_rasterio(rpcs[v])) as d:
        d.write(am, 1)
    q = cv2.cvtColor(a[::4, ::4], cv2.COLOR_GRAY2BGR).astype(np.float32)
    mq = m[::4, ::4]
    q[mq == 1] = 0.5 * q[mq == 1] + [110, 40, 0]
    q[mq == 2] = 0.4 * q[mq == 2] + [0, 0, 150]
    cv2.imwrite(str(p.prep / f'{v}_mask.jpg'), np.clip(q, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    d = info['views'].setdefault(v, {})
    d.update(water_share=float((m == 1).mean()), cloud_share=float((m == 2).mean()), kept_share=float((m == 0).mean()))
    print(f'{v}: water {(m == 1).mean():.1%}, cloud {(m == 2).mean():.2%}, kept {(m == 0).mean():.1%} -> {v}_m.tif')
json.dump(info, open(p.product / 'masks.json', 'w'), indent=1)
