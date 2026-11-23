#!/usr/bin/env python3
"""RPC camera models: read JAXA's fixed-width RPC text, evaluate it, invert it, write it into a crop.

    from rpc import read_jaxa, project, localize
    r = read_jaxa('.../RPC-ALPSMN110862910_O1B2R_UN.txt')
    row, col = project(r, lon, lat, h)          # ground -> image
    lon, lat = localize(r, row, col, h)         # image -> ground at a height
    python3 -I scripts/rpc.py RPC-*.txt HDR-*.txt   # the check below: the scene corners at 0 m

Conventions
- JAXA/RESTEC count image coordinates from the centre of the top-left pixel = (1, 1) (Document_rpc_g_e.pdf in every
  product). Checked on the Hiroshima triplet: the four scene corners of the HDR file, at 0 m above the ellipsoid,
  project to (0.5, 0.5) and (lines + 0.5, columns + 0.5) within 0.01 px, i.e. to the outer corners of the image.
  GDAL and ASP put the centre of the top-left pixel at (0, 0), and GDAL reads the RPC-*.txt next to the image as it
  is: one pixel off in rows and columns (3.5 m on the ground). Every RPC this repo writes is in the GDAL/ASP
  convention: LINE_OFF and SAMP_OFF of the JAXA file minus 1, minus the corner of the crop.
- Heights are above the ellipsoid (GRS80 in the JAXA file, WGS84 for ASP: 0.1 mm apart), never above sea level.
- row, col: image coordinates, 0 = centre of the first pixel. lon, lat in degrees.
"""
import sys

import numpy as np

FIELDS = [('LINE_OFF', 6), ('SAMP_OFF', 5), ('LAT_OFF', 8), ('LONG_OFF', 9), ('HEIGHT_OFF', 5),
          ('LINE_SCALE', 6), ('SAMP_SCALE', 5), ('LAT_SCALE', 8), ('LONG_SCALE', 9), ('HEIGHT_SCALE', 5)]
COEFS = ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']
CHUNK = 1_000_000


def read_jaxa(path):
    """JAXA fixed-width RPC text (NITF RPC00B order) -> dict in the GDAL/ASP convention (offsets minus 1)."""
    s = open(path).read().strip()
    r, o = {}, 0
    for k, n in FIELDS:
        r[k] = float(s[o:o + n]); o += n
    if len(s) != o + 80 * 12:
        raise ValueError(f'{path}: {len(s)} characters, a JAXA RPC has {o + 960}')
    c = np.array([float(s[o + 12 * i:o + 12 * (i + 1)]) for i in range(80)])
    for i, k in enumerate(COEFS):
        r[k] = c[20 * i:20 * (i + 1)]
    r['LINE_OFF'] -= 1
    r['SAMP_OFF'] -= 1
    return r


def _terms(P, L, H):                                  # RPC00B: 1 L P H LP LH PH L2 P2 H2 PLH L3 LP2 LH2 L2P P3 PH2 L2H P2H H3
    return np.stack([np.ones_like(P), L, P, H, L * P, L * H, P * H, L * L, P * P, H * H, P * L * H, L ** 3, L * P * P,
                     L * H * H, L * L * P, P ** 3, P * H * H, L * L * H, P * P * H, H ** 3])


def project(r, lon, lat, h):
    """ground -> image (row, col), any broadcastable arrays"""
    lon, lat, h = np.broadcast_arrays(*(np.asarray(x, float) for x in (lon, lat, h)))
    shape = lon.shape
    lon, lat, h = lon.ravel(), lat.ravel(), h.ravel()
    row, col = np.empty(lon.size), np.empty(lon.size)
    for s in range(0, lon.size, CHUNK):
        e = slice(s, s + CHUNK)
        T = _terms((lat[e] - r['LAT_OFF']) / r['LAT_SCALE'], (lon[e] - r['LONG_OFF']) / r['LONG_SCALE'],
                   (h[e] - r['HEIGHT_OFF']) / r['HEIGHT_SCALE'])
        row[e] = (r['LINE_NUM_COEFF'] @ T) / (r['LINE_DEN_COEFF'] @ T) * r['LINE_SCALE'] + r['LINE_OFF']
        col[e] = (r['SAMP_NUM_COEFF'] @ T) / (r['SAMP_DEN_COEFF'] @ T) * r['SAMP_SCALE'] + r['SAMP_OFF']
    return row.reshape(shape), col.reshape(shape)


def _affine_guess(r, h):
    """least-squares affine map (row, col, 1) -> (lon, lat) over the model's domain at height h"""
    g = np.linspace(-0.8, 0.8, 9)
    lat, lon = np.meshgrid(r['LAT_OFF'] + g * r['LAT_SCALE'], r['LONG_OFF'] + g * r['LONG_SCALE'], indexing='ij')
    row, col = project(r, lon, lat, h)
    A = np.c_[row.ravel(), col.ravel(), np.ones(row.size)]
    return np.linalg.lstsq(A, np.c_[lon.ravel(), lat.ravel()], rcond=None)[0]       # 3 x 2


def localize(r, row, col, h, iters=10, tol=1e-4):
    """image -> ground (lon, lat) at the given height(s) above the ellipsoid: Newton steps from an affine guess"""
    row, col, h = np.broadcast_arrays(*(np.asarray(x, float) for x in (row, col, h)))
    shape = row.shape
    row, col, h = row.ravel(), col.ravel(), h.ravel()
    M = _affine_guess(r, float(np.mean(h)) if h.size else 0.0)
    lon = M[0, 0] * row + M[1, 0] * col + M[2, 0]
    lat = M[0, 1] * row + M[1, 1] * col + M[2, 1]
    e = 1e-6                                          # degrees: 0.1 m, 0.04 px
    for _ in range(iters):
        r0, c0 = project(r, lon, lat, h)
        dr, dc = row - r0, col - c0
        if dr.size == 0 or max(np.abs(dr).max(), np.abs(dc).max()) < tol:
            break
        r1, c1 = project(r, lon + e, lat, h)
        r2, c2 = project(r, lon, lat + e, h)
        a, b, c, d = (r1 - r0) / e, (r2 - r0) / e, (c1 - c0) / e, (c2 - c0) / e
        det = a * d - b * c
        lon = lon + (d * dr - b * dc) / det
        lat = lat + (a * dc - c * dr) / det
    return lon.reshape(shape), lat.reshape(shape)


def localize_dem(r, row, col, dem, h0=0.0, iters=30, tol=0.05):
    """image -> ground on a surface: dem(lon, lat) gives the height above the ellipsoid (nan = unknown, keeps the
    last height). Fixed-point iteration on the height; it converges where the slope is below 1/tan(off-nadir)."""
    h = np.full(np.shape(row), float(h0))
    for _ in range(iters):
        lon, lat = localize(r, row, col, h)
        hn = dem(lon, lat)
        hn = np.where(np.isfinite(hn), hn, h)
        done = np.abs(hn - h).max() < tol if h.size else True
        h = hn
        if done:
            break
    return lon, lat, h


def shifted(r, row0, col0):
    """the same model for a crop whose first pixel is (row0, col0) of the image"""
    s = dict(r)
    s['LINE_OFF'] = r['LINE_OFF'] - row0
    s['SAMP_OFF'] = r['SAMP_OFF'] - col0
    return s


def to_rasterio(r):
    from rasterio.rpc import RPC
    return RPC(height_off=r['HEIGHT_OFF'], height_scale=r['HEIGHT_SCALE'], lat_off=r['LAT_OFF'], lat_scale=r['LAT_SCALE'],
               line_den_coeff=[float(x) for x in r['LINE_DEN_COEFF']], line_num_coeff=[float(x) for x in r['LINE_NUM_COEFF']],
               line_off=r['LINE_OFF'], line_scale=r['LINE_SCALE'], long_off=r['LONG_OFF'], long_scale=r['LONG_SCALE'],
               samp_den_coeff=[float(x) for x in r['SAMP_DEN_COEFF']], samp_num_coeff=[float(x) for x in r['SAMP_NUM_COEFF']],
               samp_off=r['SAMP_OFF'], samp_scale=r['SAMP_SCALE'])


def from_rasterio(m):
    r = {'LINE_OFF': m.line_off, 'SAMP_OFF': m.samp_off, 'LAT_OFF': m.lat_off, 'LONG_OFF': m.long_off,
         'HEIGHT_OFF': m.height_off, 'LINE_SCALE': m.line_scale, 'SAMP_SCALE': m.samp_scale, 'LAT_SCALE': m.lat_scale,
         'LONG_SCALE': m.long_scale, 'HEIGHT_SCALE': m.height_scale}
    for k, a in zip(COEFS, (m.line_num_coeff, m.line_den_coeff, m.samp_num_coeff, m.samp_den_coeff)):
        r[k] = np.array(a, float)
    return r


def read_tif(path):
    """the RPC stored in one of our crops"""
    import rasterio
    with rasterio.open(path) as ds:
        if ds.rpcs is None:
            raise ValueError(f'{path}: no RPC')
        return from_rasterio(ds.rpcs)


if __name__ == '__main__':                            # python3 -I scripts/rpc.py RPC-*.txt HDR-*.txt
    import re
    r = read_jaxa(sys.argv[1])
    hdr = dict(re.findall(r'(\w+)="([^"]*)"', open(sys.argv[2]).read()))
    nl, nc = int(hdr['Lines']), int(hdr['Columns'])
    want = {'LeftTop': (-0.5, -0.5), 'RightTop': (-0.5, nc - 0.5), 'LeftBottom': (nl - 0.5, -0.5),
            'RightBottom': (nl - 0.5, nc - 0.5), 'Center': ((nl - 1) / 2, (nc - 1) / 2)}
    for k, (wr, wc) in want.items():
        row, col = project(r, float(hdr[f'Scene{k}Longitude']), float(hdr[f'Scene{k}Latitude']), 0.0)
        lon, lat = localize(r, row, col, 0.0)
        print(f'{k:12s} at 0 m: row {row:10.3f} col {col:10.3f}  (outer corner / centre: {wr:.1f}, {wc:.1f})  '
              f'back to ground: {abs(lon - float(hdr[f"Scene{k}Longitude"])) * 1e6:.2f}, '
              f'{abs(lat - float(hdr[f"Scene{k}Latitude"])) * 1e6:.2f} microdegrees')
