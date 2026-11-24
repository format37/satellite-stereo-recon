#!/usr/bin/env python3
"""Step 11. The data of the interactive demo: the model as columns, the ground under it, the colours, the numbers.

    python3 -I scripts/demo.py data/hiroshima/fnb_global.yaml [OUTDIR]

Everything on a grid of demo.cell (5 m), heights in steps of demo.zstep (0.5 m) above today's mean sea level
(demo.msl_m above the geoid): the grid of cells.py, which floods.py checks against the reference. The demo (a web page)
floods the ground from the sea as the level rises and finds the nearest high ground for a point; it needs nothing else.

  surface.png   the anchored DSM, the median of the cell: what is drawn (roofs, canopy, hills). 16 bit:
                R = high byte, G = low byte of (height - z0) / zstep; B = flags: 1 sea or tidal river (where the sea
                comes from), 2 inland water, 4 measured by stereo (else filled from around), 8 outside the scenes
  ground.png    the ground model (ground.py), the same coding; B = the WorldCover class
  colour.jpg    the nadir view (panchromatic), tinted by WorldCover, water darkened
  meta.json     grid, georeference (lon/lat of a cell: a quadratic in column and row, < 1 cm off over the box),
                the level's reference and accuracy, the sun, the local marks for the slider, the demo's settings,
                the sources to credit
Writes product/demo/, and copies it to OUTDIR when given (the blog's public/media/<name>/: never in git).
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import Cells
from geo import Grid
from project import load

p = load(sys.argv[1])
if not p.demo:
    print(f'{p.name}: no demo section: nothing to do')
    sys.exit(0)
D = p.demo
g = Grid(p)
C = Cells(p, g)                                             # the grid of the demo: cells.py (floods.py checks the same)
cell, zstep, msl, rows, cols = C.cell, C.zstep, C.msl, C.rows, C.cols
S, Gm, WC, O, z0, SEA_FLOOR = C.S, C.G, C.WC, C.O, C.z0, C.sea_floor
water, sea, inland, measured, outside = C.water, C.sea, C.inland, C.measured, C.outside
out = p.product / 'demo'
out.mkdir(parents=True, exist_ok=True)
flags = (sea * 1 | inland * 2 | measured * 4 | outside * 8).astype(np.uint8)


def png16(path, a, b):
    v = C.quantized(a)
    cv2.imwrite(str(path), np.dstack([b, (v & 255).astype(np.uint8), (v >> 8).astype(np.uint8)]), [cv2.IMWRITE_PNG_COMPRESSION, 9])


png16(out / 'surface.png', S, flags)                                    # cv2 writes BGR: B, G = low byte, R = high byte
png16(out / 'ground.png', Gm, WC)

# colours: the panchromatic grey tinted by land cover (as the voxel model), water darker and bluer
TINT = {10: (0.80, 1.00, 0.72), 20: (0.92, 1.00, 0.80), 30: (0.90, 1.00, 0.78), 40: (1.00, 0.97, 0.78), 60: (1.00, 0.95, 0.85)}
rgb = np.repeat(np.clip(O, 0, 255)[..., None], 3, axis=2).astype(np.float32)
for k, t in TINT.items():
    rgb[WC == k] *= np.array(t, np.float32)
rgb[water] = 0.35 * rgb[water] + 0.65 * np.array([40, 95, 125], np.float32)
cv2.imwrite(str(out / 'colour.jpg'), cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 88])

# georeference: lon/lat of a cell centre as a quadratic in (column, row)
x0, y0, x1, y1 = g.bounds
cs, rs = np.meshgrid(np.linspace(0, cols, 21), np.linspace(0, rows, 21))
lon, lat = g.to_ll.transform(x0 + (cs + 0.5) * cell, y1 - (rs + 0.5) * cell)
A = np.c_[np.ones(cs.size), cs.ravel(), rs.ravel(), cs.ravel() ** 2, cs.ravel() * rs.ravel(), rs.ravel() ** 2]
coef = {k: np.linalg.lstsq(A, v.ravel(), rcond=None)[0].tolist() for k, v in (('lon', lon), ('lat', lat))}

J = lambda f_: json.load(open(p.product / f_)) if (p.product / f_).exists() else {}
views, anchor, check, dsm_info = J('views.json'), J('anchor.json'), J('check.json'), J('dsm.json')
nv = views.get('views', {}).get('N') or next(iter(views.get('views', {}).values()), {})
acc = {}
if check.get('anchored'):
    fo = check['anchored']['flat open ground']
    acc = dict(against='GSI laser DEM (independent)', nmad=fo['nmad'], le95=fo['le95'], median=fo['median'], rms=fo['rms'])
elif anchor.get('offset_nmad_m'):
    acc = dict(against=f'{anchor["reference"]} (the anchor itself)', nmad=anchor['offset_nmad_m'])
land = ~water
fl = J('floods.json').get('curve')
reference = dict(name='GSI laser DEM', levels=fl['levels'], km2=fl['GSI DEM (ground)']) if fl else None
meta = dict(
    name=D.get('name', p.project.name), w=cols, h=rows, cell=cell, zstep=zstep, z0=z0, sea_floor=SEA_FLOOR,
    crs=str(g.crs), e0=x0, n1=y1, lonlat=coef,
    datum=dict(geoid=p.geoid, msl_above_geoid=msl,
               note='heights in m above today\'s mean sea level'),
    level=dict(anchor=anchor.get('reference', 'none'), offset_removed_m=anchor.get('vertical_offset_m'),
               shift_m=anchor.get('shift_m')),
    accuracy=acc, measured_share=float(measured[land].mean()),
    top=float(np.nanmax(S)), land_km2=float(land.sum() * cell * cell / 1e6),
    sun=dict(azimuth=nv.get('sun_azimuth'), elevation=nv.get('sun_elevation')),
    scene=dict(sensor='ALOS PRISM', views=p.views, pairs=[''.join(x) for x in p.pairs], date=(nv.get('time_utc') or '')[:8]),
    marks=list(D.get('marks') or []), margin_m=float(D.get('margin_m', 5)), speed_kmh=float(D.get('speed_kmh', 8)),
    refuge_min_ha=float(D.get('refuge_min_ha', 1)),
    reference=reference,
    credits=list(D.get('credits') or ['Includes material (c) JAXA 2008, distributed by RESTEC (ALOS PRISM)',
                                       'Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA',
                                       'ESA WorldCover 2021 (c) ESA WorldCover project, contains modified Copernicus Sentinel data (2021)']),
)
json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
sizes = {fn.name: fn.stat().st_size for fn in out.iterdir()}
print(f'demo: {cols} x {rows} cells of {cell:g} m, heights {z0:.1f}..{meta["top"]:.1f} m above mean sea level in {zstep:g} m steps; '
      + ', '.join(f'{k} {v / 1e6:.1f} MB' for k, v in sizes.items()))
if len(sys.argv) > 2:
    dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
    for fn in out.iterdir():
        shutil.copy2(fn, dst / fn.name)
    print(f'copied to {dst}')
