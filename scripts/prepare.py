#!/usr/bin/env python3
"""Step 1. Crop every view to the area of interest, each crop with its own RPC; the geometry of the views and pairs.

    python3 -I scripts/prepare.py data/hiroshima/fb.yaml

For every view of the YAML's pairs (and the views masks.cloud_views needs): the box `aoi` (lon/lat) at the heights
`heights` (above the ellipsoid) is projected into the image with its RPC, and the crop is that window plus
`crop_margin` pixels on every side.

Writes
  prep/<view>.tif     the pixels as delivered, the RPC in the TIFF (GDAL/ASP convention, see rpc.py), no georeferencing:
                      ASP treats it as a raw image
  prep/<view>.jpg     a look at it (1/4 size) with the box drawn at 40 m above the ellipsoid (about sea level)
  product/views.json  per view: the scene, the window, the time, the sun, where the camera looks from (azimuth and
                      off-nadir angle at the centre of the box) and how far a point moves in the image per 100 m of height;
                      per pair: convergence angle, base-to-height ratio, the height one pixel of disparity is worth
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project import load, pair_name
import rpc as R

p = load(sys.argv[1])
views = list(dict.fromkeys(p.views + (p.cloud_views if p.clouds else [])))
p.prep.mkdir(parents=True, exist_ok=True); p.product.mkdir(parents=True, exist_ok=True)
lon0, lat0, lon1, lat1 = p.aoi
lonc, latc = (lon0 + lon1) / 2, (lat0 + lat1) / 2
t = np.linspace(0, 1, 65)
box_lon = np.r_[lon0 + (lon1 - lon0) * t, np.full(65, lon1), lon1 - (lon1 - lon0) * t, np.full(65, lon0)]
box_lat = np.r_[np.full(65, lat1), lat1 - (lat1 - lat0) * t, np.full(65, lat0), lat0 + (lat1 - lat0) * t]
M_LAT = 111132.954 - 559.822 * np.cos(2 * np.radians(latc))               # metres per degree here
M_LON = 111412.84 * np.cos(np.radians(latc)) - 93.5 * np.cos(3 * np.radians(latc))


def scene_files(d):
    img, rpcf, hdrf = (sorted(Path(d).glob(g)) for g in ('IMG-*.tif', 'RPC-*.txt', 'HDR-*.txt'))
    if not (len(img) == len(rpcf) == len(hdrf) == 1):
        raise SystemExit(f'{d}: need exactly one IMG-*.tif, RPC-*.txt and HDR-*.txt (JAXA 1B2R + RPC product)')
    return img[0], rpcf[0], hdrf[0]


info = {'aoi': p.aoi, 'heights': p.heights, 'views': {}, 'pairs': {}}
for v in views:
    img, rpcf, hdrf = scene_files(p.scenes[v])
    hdr = dict(re.findall(r'(\w+)="([^"]*)"', open(hdrf).read()))
    r = R.read_jaxa(rpcf)
    rows, cols = [], []
    for h in np.linspace(*p.heights, 5):
        a, b = R.project(r, box_lon, box_lat, h); rows.append(a); cols.append(b)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    with rasterio.open(img) as src:
        H, W = src.height, src.width
        r0 = int(np.floor(rows.min())) - p.crop_margin; r1 = int(np.ceil(rows.max())) + p.crop_margin
        c0 = int(np.floor(cols.min())) - p.crop_margin; c1 = int(np.ceil(cols.max())) + p.crop_margin
        clipped = r0 < 0 or c0 < 0 or r1 >= H or c1 >= W
        r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, H - 1), min(c1, W - 1)
        win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        a = src.read(1, window=win)
    rc = R.shifted(r, r0, c0)
    out = p.prep / f'{v}.tif'
    prof = dict(driver='GTiff', width=a.shape[1], height=a.shape[0], count=1, dtype='uint8', tiled=True,
                blockxsize=512, blockysize=512, compress='deflate')
    with rasterio.open(out, 'w', **prof, rpcs=R.to_rasterio(rc)) as dst:
        dst.write(a, 1)
    if R.read_tif(out)['LINE_OFF'] != rc['LINE_OFF']:
        raise SystemExit(f'{out}: the RPC did not survive the write')
    # where the camera looks from, at the centre of the box: the ray through the ground point at 0 m and 1000 m
    rr, cc = R.project(rc, lonc, latc, 0.0)
    lo2, la2 = R.localize(rc, rr, cc, 1000.0)
    dE, dN = (lo2 - lonc) * M_LON, (la2 - latc) * M_LAT                        # ground shift of the ray per 1000 m up
    los = np.array([dE, dN, 1000.0]); los /= np.linalg.norm(los)
    d100 = np.subtract(R.project(rc, lonc, latc, 100.0), R.project(rc, lonc, latc, 0.0))
    look = rc.copy()
    q = a[::4, ::4]
    jpg = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    br, bc = R.project(rc, box_lon, box_lat, 40.0)
    cv2.polylines(jpg, [np.c_[bc / 4, br / 4].astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(p.prep / f'{v}.jpg'), jpg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    info['views'][v] = dict(
        scene=hdr['SceneID'], product=hdr['ProductID'], file=img.name, time_utc=hdr['SceneCenterTime'],
        incidence_header=hdr.get('IncidentAngle'), sun_elevation=float(hdr['SunAngleElevation']),
        sun_azimuth=float(hdr['SunAngleAzimuth']), gsd=float(hdr['PixelSize']),
        window=dict(col=c0, row=r0, width=a.shape[1], height=a.shape[0]), clipped=bool(clipped),
        image_size=[W, H], zero_pixels=float((a == 0).mean()),
        look_azimuth=float(np.degrees(np.arctan2(dE, dN)) % 360), off_nadir=float(np.degrees(np.arctan2(np.hypot(dE, dN), 1000))),
        shift_per_100m_px=[float(d100[0]), float(d100[1])], ground_shift_per_1000m=[float(dE), float(dN)], los=los.tolist())
    print(f'{v}: {img.name} window col {c0} row {r0} {a.shape[1]} x {a.shape[0]} -> {out.name}'
          f'{"  CLIPPED by the image edge" if clipped else ""}; looks from azimuth {info["views"][v]["look_azimuth"]:.1f} deg, '
          f'{info["views"][v]["off_nadir"]:.1f} deg off nadir; +100 m of height moves a point by {d100[0]:+.1f} rows {d100[1]:+.1f} cols')

for pr in p.pairs + ([tuple(x) for x in [('F', 'B'), ('N', 'F'), ('N', 'B')] if all(y in info['views'] for y in x)]):
    a, b = (info['views'][v] for v in pr)
    conv = np.degrees(np.arccos(np.clip(np.dot(a['los'], b['los']), -1, 1)))
    bh = np.hypot(*np.subtract(a['ground_shift_per_1000m'], b['ground_shift_per_1000m'])) / 1000
    info['pairs'][pair_name(pr)] = dict(convergence_deg=float(conv), base_to_height=float(bh), metres_per_px_disparity=float(a['gsd'] / bh))
for k, v in info['pairs'].items():
    print(f'pair {k}: convergence {v["convergence_deg"]:.1f} deg, B/H {v["base_to_height"]:.2f}, '
          f'one pixel of disparity = {v["metres_per_px_disparity"]:.2f} m of height')
json.dump(info, open(p.product / 'views.json', 'w'), indent=1)
