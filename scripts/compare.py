#!/usr/bin/env python3
"""Variants of one project side by side: coverage, accuracy, and how their DSMs differ.

    python3 -I scripts/compare.py data/hiroshima/fb.yaml data/hiroshima/fnb.yaml

The first YAML is the baseline. Writes data/<project>/out/compare.md and compare_<a>_<b>.png (the second DSM minus the
first, land only; the anchored DSMs when check.py has made them, so that the levels, which only ground control
fixes, do not hide the shapes). Both variants must share the grid (same aoi, crs, dsm.res).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid, stats
from project import load

ps = [load(y) for y in sys.argv[1:]]
g = Grid(ps[0])
out = ps[0].project / 'out'
rows = []
for p in ps:
    d = json.load(open(p.product / 'dsm.json')); c = json.load(open(p.product / 'check.json'))
    b, a = c['before'], c['after']
    rows.append((p.variant, ' + '.join(''.join(x) for x in p.pairs), d['measured_of_land'], d['nomatch_holes_of_land'], d['cloud_holes_of_land'],
                 b['flat open ground'], a['flat open ground'], c['coregistration'], a['built-up'], a['trees'], c['copernicus_30m']['all land'],
                 b['all land']))
L = [f'# {ps[0].project.name}: variants side by side', '',
     '| variant | pairs | land measured | no match | cloud | flat open, as produced: median / RMS / NMAD | shift, m | offset, m | '
     'flat open, after: RMS / NMAD / 95% | built-up median | trees median | all land, as produced: median / RMS | Copernicus 30 m: RMS / NMAD |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for v, pr, ml, nm, cl, fb, fa, co, bu, tr, cop, al in rows:
    L.append(f'| {v} | {pr} | {ml:.1%} | {nm:.1%} | {cl:.2%} | {fb["median"]:+.2f} / {fb["rms"]:.2f} / {fb["nmad"]:.2f} | '
             f'{co["shift_east_m"]:+.1f} E {co["shift_north_m"]:+.1f} N | {co["vertical_offset_m"]:+.2f} | '
             f'{fa["rms"]:.2f} / {fa["nmad"]:.2f} / {fa["le95"]:.2f} | {bu["median"]:+.1f} | {tr["median"]:+.1f} | '
             f'{al["median"]:+.2f} / {al["rms"]:.2f} | {cop["rms"]:.2f} / {cop["nmad"]:.2f} |')
water = np.nan_to_num(g.read(ps[0].product / 'water.tif')) > 0
wc = np.nan_to_num(g.read(ps[0].vref / 'worldcover.tif')).astype(int)
pick = lambda p: p.product / ('dsm_anchored.tif' if (p.product / 'dsm_anchored.tif').exists() else 'dsm.tif')
H0 = g.read(pick(ps[0]))
for p in ps[1:]:
    H1 = g.read(pick(p))
    d = H1 - H0
    land = ~water
    both = land & np.isfinite(d)
    filled = land & np.isnan(H0) & np.isfinite(H1)
    lost = land & np.isfinite(H0) & np.isnan(H1)
    L += ['', f'## {p.variant} - {ps[0].variant} ({pick(p).stem} - {pick(ps[0]).stem}: the shapes, the levels taken out)', '',
          f'- where both measured ({both.mean() / land.mean():.1%} of the land): ' +
          (lambda s: f'median {s["median"]:+.2f}, RMS {s["rms"]:.2f}, NMAD {s["nmad"]:.2f}, 95% of |d| {s["le95"]:.2f} m')(stats(d[both])),
          f'- built-up: ' + (lambda s: f'median {s["median"]:+.2f}, NMAD {s["nmad"]:.2f} m')(stats(d[both & (wc == 50)])) +
          '; trees: ' + (lambda s: f'median {s["median"]:+.2f}, NMAD {s["nmad"]:.2f} m')(stats(d[both & (wc == 10)])),
          f'- {p.variant} fills {filled.sum() / land.sum():.1%} of the land that {ps[0].variant} left open, and leaves open '
          f'{lost.sum() / land.sum():.1%} that {ps[0].variant} measured']
    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    im = ax.imshow(np.where(land, d, np.nan), cmap='RdBu_r', vmin=-10, vmax=10)
    ax.imshow(np.where(filled, 1.0, np.nan), cmap='Greens', vmin=0, vmax=1.5, alpha=0.9)
    ax.set_title(f'{p.variant} - {ps[0].variant}, m (green: measured only by {p.variant})', fontsize=10); ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.savefig(out / f'compare_{ps[0].variant}_{p.variant}.png', bbox_inches='tight'); plt.close(fig)
open(out / 'compare.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
