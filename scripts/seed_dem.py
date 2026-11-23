#!/usr/bin/env python3
"""The DEM the views are mapprojected onto before stereo (stereo.mapproject): a smooth version of an earlier run.

    python3 -I scripts/seed_dem.py data/hiroshima/fb_map.yaml

stereo.mapproject names another variant of the same project (e.g. fb). Its fused DSM above the ellipsoid
(product/dsm_ellipsoid.tif) has its holes filled from their edges, the water set to sea level (the geoid), and is
smoothed over ~40 m and written at 10 m: smooth enough that buildings do not fold the mapprojected images (ASP
wants a smooth seed; stereo then measures what the seed lacks). Nothing outside the scenes goes into it.
Writes prep/seed_dem.tif, 200 m wider than the grid on every side.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load

p = load(sys.argv[1])
seed = p.project / 'out' / str(p.cfg['stereo']['mapproject']) / 'product' / 'dsm_ellipsoid.tif'
if not seed.exists():
    raise SystemExit(f'{seed} not found: run that variant first')
g = Grid(p)
h = g.read(seed)
N = g.read(p.vref / f'geoid_{p.geoid}.tif')
water = np.nan_to_num(g.read(p.product / 'water.tif')) > 0
h[water] = N[water]
idx = ndimage.distance_transform_edt(np.isnan(h), return_distances=False, return_indices=True)
h = h[tuple(idx)]
h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 16 / 1.0)                 # 16 cells of 2.5 m: sigma 40 m
lon0, lat0, lon1, lat1 = p.aoi
pad = 0.003
gw = Grid(p, res=10, aoi=[lon0 - pad * 1.2, lat0 - pad, lon1 + pad * 1.2, lat1 + pad])
out = gw.warp_array(h, g.transform, g.crs, 'bilinear')
idx = ndimage.distance_transform_edt(np.isnan(out), return_distances=False, return_indices=True)
out = out[tuple(idx)]
gw.write(p.prep / 'seed_dem.tif', out.astype(np.float32), nodata=-9999.0)
print(f'seed DEM from {seed.parent.parent.name}: {gw.width} x {gw.height} cells of 10 m, {np.min(out):.1f} .. {np.max(out):.1f} m above the ellipsoid')
