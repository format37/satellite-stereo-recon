#!/usr/bin/env python3
"""Step 6. The level: tie the DSM to a reference elevation model by a horizontal shift and a vertical offset.

    python3 -I scripts/anchor.py data/hiroshima/fnb_global.yaml

The RPCs alone leave the level of a stereo DSM metres off (Hiroshima: -2.7 m with forward-backward, +6.1 m with the
three views) and its position about a pixel off. Nothing in the images can tell; one outside height model can.
anchor.reference names it:
  copernicus  Copernicus GLO-30 (TanDEM-X radar, 2011-15, global, free; heights above EGM2008). The default: it
              exists everywhere, so the method works for any place with stereo images
  gsi         the GSI elevation tiles of check.reference (Japan only; airborne laser where DEM5A exists)
  none        no anchoring: product/dsm_anchored.tif is the DSM as produced
Both models are compared on the run's datum (dsm.geoid). Shift: Nuth & Kaab on non-built-up slopes of 5-45 deg;
offset: the median difference on flat open ground (WorldCover grass, crops, bare, shrubs, wetland, slope < 10 deg).
Copernicus has 30 m cells, so a small field among houses carries some of their roofs: only open ground at least
anchor.open_margin_m (30 m) from anything else sets its offset.

Writes product/dsm_anchored.tif (heights above the geoid, like dsm.tif) and product/anchor.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coreg import coregister, shifted
from geo import Grid
from project import load

p = load(sys.argv[1])
g = Grid(p)
ref = p.vref
H = g.read(p.product / 'dsm.tif')
wc = np.nan_to_num(g.read(ref / 'worldcover.tif')).astype(int)
water = np.nan_to_num(g.read(p.product / 'water.tif')) > 0
land = ~water & (wc != 80) & (wc > 0)
N_run = g.read(ref / f'geoid_{p.geoid}.tif')

if p.anchor == 'none':
    info = dict(reference='none', shift_east_m=0.0, shift_north_m=0.0, shift_m=0.0, vertical_offset_m=0.0)
    Ha = H
else:
    if p.anchor == 'copernicus':
        R = g.read(ref / 'copernicus.tif') + g.read(ref / 'geoid_egm2008.tif') - N_run     # EGM2008 -> the run's datum
        margin = p.anchor_open_margin_m
    elif p.anchor == 'gsi':
        R = g.read(ref / 'gsi.tif') + g.read(ref / 'geoid_gsigeo2011.tif') - N_run         # T.P. -> the run's datum
        margin = 0.0
    else:
        raise SystemExit(f'anchor.reference must be copernicus, gsi or none, not {p.anchor}')
    info = dict(reference=p.anchor, **coregister(H, R, land, wc, g.res, margin), open_margin_m=margin)
    Ha = (shifted(H, info['shift_east_m'], info['shift_north_m'], g.res) - info['vertical_offset_m']).astype(np.float32)
info['geoid'] = p.geoid
g.write(p.product / 'dsm_anchored.tif', Ha)
json.dump(info, open(p.product / 'anchor.json', 'w'), indent=1)
print(f'anchored to {info["reference"]}: shift {info["shift_east_m"]:+.2f} m east, {info["shift_north_m"]:+.2f} m north, '
      f'offset {info["vertical_offset_m"]:+.2f} m removed' +
      (f' (from {info["cells_offset"]:,} cells of flat open ground, NMAD {info["offset_nmad_m"]:.2f} m)' if 'cells_offset' in info else ''))
