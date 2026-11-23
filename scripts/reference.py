#!/usr/bin/env python3
"""Step 2. Independent data on the run's grid: the GSI elevation model, land cover, geoid models, Copernicus DEM.

    python3 -I scripts/reference.py data/hiroshima/fb.yaml

Nothing here comes from the satellite scenes. Downloads are cached in out/ref/ (shared by the variants); what is
written goes to out/<variant>/ref/, on the grid of the DSM (geo.py):
  gsi.tif, gsi_source.tif  GSI elevation tiles (地理院タイル 標高タイル), heights above T.P. (Tokyo Bay mean sea level,
                           the Japanese height datum), for each cell the first layer of check.reference that has it:
                           1 dem5a = DEM5A, airborne laser, 5 m, the ground without buildings and trees (accuracy ~0.3 m)
                           2 dem5b = DEM5B, photogrammetry, 5 m; 3 dem5c = DEM5C; 4 dem10b = DEM10B, 10 m (from contours)
                           Source: 国土地理院 (GSI Japan), https://maps.gsi.go.jp/development/ichiran.html
  geoid_<model>.tif        geoid height N above the ellipsoid (H = h - N): gsigeo2011 (GSI's geoid, fitted to the
                           levelled T.P. heights that the GSI DEM uses), jpgeo2024 (its successor for the 2025 height
                           datum), egm2008 (global). Grids from https://cdn.proj.org (PROJ-data)
  worldcover.tif           ESA WorldCover 2021 v200, 10 m land cover (10 trees, 20 shrubs, 30 grass, 40 crops,
                           50 built-up, 60 bare, 80 water, 90 wetland), https://esa-worldcover.org (CC BY 4.0)
  copernicus.tif           Copernicus GLO-30 DSM (TanDEM-X, 2011-15), heights above EGM2008: a surface model like ours,
                           30 m. © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under
                           COPERNICUS by the European Union and ESA
  ref.json                 what came from where, coverage, the spread between the geoid models
"""
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load

UA = {'User-Agent': 'satellite-stereo-recon/0.1 (research script)'}
GSI = {'dem5a': ('dem5a_png', 15), 'dem5b': ('dem5b_png', 15), 'dem5c': ('dem5c_png', 15), 'dem10b': ('dem_png', 14)}
GEOIDS = {'gsigeo2011': 'jp_gsi_gsigeo2011.tif', 'jpgeo2024': 'jp_gsi_jpgeo2024.tif', 'egm2008': 'us_nga_egm08_25.tif'}
WORLDCOVER = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif'
COPERNICUS = ('https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif')
R = 6378137.0


def worldcover(g, aoi):
    """ESA WorldCover classes on a grid (0 = no data), read from the 3 x 3 degree tiles the box touches"""
    lon0, lat0, lon1, lat1 = aoi
    tiles = {(3 * math.floor(la / 3), 3 * math.floor(lo / 3)) for la in (lat0, lat1) for lo in (lon0, lon1)}
    wc = np.zeros(g.shape, np.float32)
    for la, lo in sorted(tiles):
        t = f'{"N" if la >= 0 else "S"}{abs(la):02d}{"E" if lo >= 0 else "W"}{abs(lo):03d}'
        w = g.warp('/vsicurl/' + WORLDCOVER.format(t=t), 'nearest', src_nodata=0)
        wc = np.where(np.isfinite(w) & (wc == 0), w, wc)
    return wc


def fetch(url, path):
    """download once; returns False for a missing tile (404)"""
    path = Path(path)
    if path.exists():
        return True
    if path.with_suffix('.none').exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + '.part'); tmp.write_bytes(data); tmp.rename(path)
            time.sleep(0.05)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                path.with_suffix('.none').touch()
                return False
            err = e
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f'{url}: {err}')


def tile_range(aoi, z, pad=0.004):
    lon0, lat0, lon1, lat1 = aoi
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return range(tx(lon0 - pad), tx(lon1 + pad) + 1), range(ty(lat1 + pad), ty(lat0 - pad) + 1)


def gsi_layer(name, aoi, cache):
    """mosaic of one GSI elevation tile layer in Web Mercator: array (nan = no data), transform"""
    layer, z = GSI[name]
    xs, ys = tile_range(aoi, z)
    a = np.full((len(ys) * 256, len(xs) * 256), np.nan, np.float32)
    got = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            f = cache / layer / str(z) / str(x) / f'{y}.png'
            if not fetch(f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png', f):
                continue
            rgb = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.int64)
            v = (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2]        # GSI: x = R*2^16 + G*2^8 + B, 0.01 m
            h = np.where(v < 2 ** 23, v, v - 2 ** 24) * 0.01
            h = np.where(v == 2 ** 23, np.nan, h).astype(np.float32)          # (128, 0, 0) = no data
            a[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256] = h
            got += 1
    n = 2 ** z
    px = 2 * math.pi * R / n / 256
    tr = from_origin(xs[0] / n * 2 * math.pi * R - math.pi * R, math.pi * R - ys[0] / n * 2 * math.pi * R, px, px)
    return a, tr, got, len(xs) * len(ys)


if __name__ == '__main__':
    p = load(sys.argv[1])
    g = Grid(p)
    out = p.run / 'ref'; out.mkdir(parents=True, exist_ok=True); cache = p.ref; cache.mkdir(parents=True, exist_ok=True)
    info = {'grid': dict(crs=str(g.crs), res=g.res, width=g.width, height=g.height, bounds=g.bounds)}
    print(f'grid {g.width} x {g.height} cells of {g.res} m in {g.crs}')

    # ---- GSI elevation tiles, in the order of preference
    H = np.full(g.shape, np.nan, np.float32); S = np.zeros(g.shape, np.uint8); info['gsi'] = {}
    for k, name in enumerate(p.reference, 1):
        a, tr, got, total = gsi_layer(name, p.aoi, cache / 'gsi')
        w = g.warp_array(a, tr, 'EPSG:3857', 'bilinear')
        new = np.isnan(H) & np.isfinite(w)
        H[new] = w[new]; S[new] = k
        info['gsi'][name] = dict(tiles=got, of=total, share_of_grid=float(new.mean()))
        print(f'GSI {name}: {got}/{total} tiles, fills {new.mean():.1%} of the grid')
    g.write(out / 'gsi.tif', H); g.write(out / 'gsi_source.tif', S, nodata=0)
    info['gsi_range'] = [float(np.nanmin(H)), float(np.nanmax(H))]
    print(f'GSI heights {np.nanmin(H):.1f} .. {np.nanmax(H):.1f} m (T.P.); no data on {np.isnan(H).mean():.1%} (the sea)')

    # ---- geoid models
    info['geoid'] = {}
    N = {}
    for name, fn in GEOIDS.items():
        f = cache / 'geoid' / fn
        fetch(f'https://cdn.proj.org/{fn}', f)
        N[name] = g.warp(str(f), 'bilinear')
        g.write(out / f'geoid_{name}.tif', N[name])
        info['geoid'][name] = dict(file=fn, min=float(np.nanmin(N[name])), max=float(np.nanmax(N[name])), mean=float(np.nanmean(N[name])))
        print(f'geoid {name}: N = {np.nanmin(N[name]):.3f} .. {np.nanmax(N[name]):.3f} m above the ellipsoid')
    for a, b in (('gsigeo2011', 'egm2008'), ('jpgeo2024', 'gsigeo2011')):
        d = N[a] - N[b]
        info['geoid'][f'{a}-{b}'] = dict(min=float(np.nanmin(d)), max=float(np.nanmax(d)), mean=float(np.nanmean(d)))
        print(f'  {a} - {b}: {np.nanmean(d):+.3f} m ({np.nanmin(d):+.3f} .. {np.nanmax(d):+.3f})')

    # ---- land cover and the Copernicus DSM: read straight from their cloud-optimised GeoTIFFs (only the blocks needed)
    lon0, lat0, lon1, lat1 = p.aoi
    wc = worldcover(g, p.aoi)
    g.write(out / 'worldcover.tif', wc.astype(np.uint8), nodata=0)
    cls = {10: 'trees', 20: 'shrubs', 30: 'grass', 40: 'crops', 50: 'built-up', 60: 'bare', 80: 'water', 90: 'wetland'}
    info['worldcover'] = {cls.get(int(c), str(int(c))): float((wc == c).mean()) for c in np.unique(wc)}
    print('WorldCover:', ', '.join(f'{k} {v:.1%}' for k, v in info['worldcover'].items()))
    cop = np.full(g.shape, np.nan, np.float32)
    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            t = f'N{la:02d}_00_E{lo:03d}_00'
            w = g.warp('/vsicurl/' + COPERNICUS.format(t=t), 'bilinear')
            cop = np.where(np.isnan(cop), w, cop)
    g.write(out / 'copernicus.tif', cop)
    info['copernicus_range'] = [float(np.nanmin(cop)), float(np.nanmax(cop))]
    print(f'Copernicus GLO-30: {np.nanmin(cop):.1f} .. {np.nanmax(cop):.1f} m (EGM2008)')
    json.dump(info, open(out / 'ref.json', 'w'), indent=1)
