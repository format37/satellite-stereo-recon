#!/usr/bin/env python3
"""Step 9. The two questions of the demo, asked of the models and of the check reference: where does the sea go as
it rises, and where is the nearest high ground.

    python3 -I scripts/floods.py data/hiroshima/fnb_global.yaml

At the demo's cell (demo.cell, 5 m), for the GSI DEM (bare ground, the reference), the anchored DSM (surface: roofs
and canopy), the ground model of ground.py, and the DSM at the level the RPCs alone give:
- the sea at a level (floods.levels, m above today's mean sea level, demo.msl_m above the geoid): land reached from
  the sea through cells lower than the level (a bathtub: water that stays, no waves), km² and overlap with GSI's;
- high ground: land at least floods.high_ground_m above mean sea level, in pieces of at least demo.refuge_min_ha.
  For every cell of land below the map level by GSI, the running time to the nearest piece, as the demo finds it
  (demo.speed_kmh on the flat, Tobler's slope factor, 8 neighbours, round the water), on each model and on GSI;
  whether the piece a model's route ends on is really high (within 50 m of GSI's high ground) or a roof.
Writes product/floods.json, floods.md and floods_map.png (the sea at floods.map_level, high ground, wrong high ground).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load

p = load(sys.argv[1])
if not p.reference or not p.floods:
    print(f'{p.name}: floods needs check.reference (GSI) and a floods section: nothing to do')
    sys.exit(0)
g = Grid(p)
ref = p.vref
cell = float(p.demo.get('cell', 5)); f = int(round(cell / g.res))
msl = float(p.demo.get('msl_m', 0.0))
levels = [float(x) for x in p.floods.get('levels', [1, 2, 3, 5, 10, 20, 50])]
RH = float(p.floods.get('high_ground_m', 10))
LM = float(p.floods.get('map_level', 3))
A_MIN = int(round(float(p.demo.get('refuge_min_ha', 1)) * 1e4 / cell ** 2))
hh, ww = g.height // f * f, g.width // f * f
rows, cols = hh // f, ww // f


def blocks(a):
    return a[:hh, :ww].reshape(rows, f, cols, f).transpose(0, 2, 1, 3).reshape(rows, cols, f * f)


def med(a):
    with np.errstate(all='ignore'):
        return np.nanmedian(blocks(a), axis=2).astype(np.float32) - msl          # m above today's mean sea level


def fillnan(a):
    m = np.isnan(a)
    if m.any():
        a = a[tuple(ndimage.distance_transform_edt(m, return_distances=False, return_indices=True))]
    return a


N_run = g.read(ref / f'geoid_{p.geoid}.tif')
wat = np.nan_to_num(g.read(p.product / 'water.tif'))
sea = blocks((wat == 1).astype(np.float32)).mean(axis=2) > 0.5
inland = (blocks((wat == 2).astype(np.float32)).mean(axis=2) > 0.5) & ~sea
G = med(g.read(ref / 'gsi.tif') + g.read(ref / 'geoid_gsigeo2011.tif') - N_run)
land = ~sea & ~inland & np.isfinite(G)
anchor = json.load(open(p.product / 'anchor.json')) if (p.product / 'anchor.json').exists() else {'reference': '?'}
models = {'GSI DEM (ground)': G,
          'PRISM surface, anchored': fillnan(med(g.read(p.product / 'dsm_anchored.tif'))),
          'PRISM ground model': fillnan(med(g.read(p.product / 'ground.tif'))),
          'PRISM surface, level from the RPCs': fillnan(med(g.read(p.product / 'dsm.tif')))}
for k in models:
    models[k] = np.where(sea, np.nan, models[k])
G = models['GSI DEM (ground)']
REF = 'GSI DEM (ground)'
km2 = cell * cell / 1e6


def flooded(H, L):
    passable = sea | (np.nan_to_num(H, nan=1e9) < L)
    lab, _ = ndimage.label(passable)
    keep = np.zeros(lab.max() + 1, bool); keep[np.unique(lab[sea])] = True; keep[0] = False
    return keep[lab] & land


def high_ground(H, R):
    hi = land & (np.nan_to_num(H, nan=-1e9) >= R)
    lab, _ = ndimage.label(hi, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel()); size[0] = 0
    return (size >= A_MIN)[lab]


def run_times(H, targets):
    """seconds to the nearest target cell, running as in the demo: demo.speed_kmh on the flat, Tobler's slope factor,
    8 neighbours, over land; and the target cell each cell runs to. One search backwards from all the targets."""
    walk = ~sea & ~inland & np.isfinite(H)
    Hn, ids = np.nan_to_num(H), np.arange(rows * cols).reshape(rows, cols)
    v0, flat = float(p.demo.get('speed_kmh', 8)) / 3.6, np.exp(-3.5 * 0.05)
    r_, c_, d_ = [], [], []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        a = (slice(max(0, -dr), rows - max(0, dr)), slice(max(0, -dc), cols - max(0, dc)))   # from here
        b = (slice(max(0, dr), rows - max(0, -dr)), slice(max(0, dc), cols - max(0, -dc)))   # to the neighbour
        ok = walk[a] & walk[b]
        step = cell * (np.sqrt(2) if dr and dc else 1.0)
        v = v0 * np.exp(-3.5 * np.abs((Hn[b] - Hn[a]) / step + 0.05)) / flat
        r_.append(ids[b][ok]); c_.append(ids[a][ok]); d_.append(step / v[ok])   # backwards: b -> a costs the run a -> b
    A = csr_matrix((np.concatenate(d_), (np.concatenate(r_), np.concatenate(c_))), shape=(rows * cols,) * 2)
    t, _, src = dijkstra(A, indices=np.flatnonzero(targets & walk), min_only=True, limit=3600, return_predecessors=True)
    return t.reshape(rows, cols), src.reshape(rows, cols)


res = dict(cell_m=cell, msl_above_geoid_m=msl, geoid=p.geoid, anchor=anchor['reference'], land_km2=float(land.sum() * km2),
           flood={}, high_ground={})
for L in levels:
    fG = flooded(G, L)
    row = {}
    for name, H in models.items():
        fM = fG if name == REF else flooded(H, L)
        both = (fM & fG).sum()
        row[name] = dict(km2=float(fM.sum() * km2), iou=float(both / max((fM | fG).sum(), 1)), of_gsi=float(both / max(fG.sum(), 1)))
    res['flood'][f'{L:g}'] = row
rG = high_ground(G, RH)
dG = ndimage.distance_transform_edt(~rG) * cell
tG, _ = run_times(G, rG)
people = land & (G < LM) & np.isfinite(tG)                   # low land from which GSI's high ground is within the hour
res['people_km2'] = float(people.sum() * km2)
maps = {}
for name, H in models.items():
    if name == REF:
        res['high_ground'][name] = dict(km2=float(rG.sum() * km2), minutes_median=float(np.median(tG[people]) / 60),
                                        minutes_p90=float(np.percentile(tG[people], 90) / 60))
        maps[name] = rG
        continue
    rM = high_ground(H, RH)
    tM, src = run_times(H, rM)
    a, b, s_ = tM[people], tG[people], src[people]
    real = np.zeros(a.size, bool); real[s_ >= 0] = dG.ravel()[s_[s_ >= 0]] <= 50
    res['high_ground'][name] = dict(km2=float(rM.sum() * km2), wrong_km2=float((rM & (dG > 50)).sum() * km2),
                                    minutes_median=float(np.median(np.where(np.isfinite(a), a, 1e9)) / 60),
                                    within_2min=float(np.mean(np.abs(a - b) <= 120)), longer_5min=float(np.mean(a - b > 300)),
                                    none=float(np.mean(~np.isfinite(a))), route_to_real=float(np.mean(real)))
    maps[name] = rM
# the whole curve: land the sea reaches by level, for the reference and the ground model (demo.py puts the reference's
# beside the model's number in the demo)
lv = np.r_[np.arange(0, 20, 0.5), np.arange(20, 60, 1.0), np.arange(60, np.nanmax(G) + 5, 5.0)]
res['curve'] = dict(levels=[float(x) for x in lv], **{k: [float(flooded(models[k], L).sum() * km2) for L in lv]
                                                     for k in ('GSI DEM (ground)', 'PRISM ground model')})
json.dump(res, open(p.product / 'floods.json', 'w'), indent=1)

names = list(models)
L_ = [f'# {p.name}: where the sea goes, and the nearest high ground', '',
      f'Cells of {cell:g} m; levels in m above today\'s mean sea level ({msl:+.2f} m above {p.geoid}); the DSM anchored to '
      f'{anchor["reference"]}; the GSI DEM is the reference. Land: {res["land_km2"]:.1f} km².', '',
      '## Land the sea reaches (km², overlap with GSI as IoU)', '', '| level | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 1) + '|']
for L, row in res['flood'].items():
    L_.append(f'| {L} m | ' + ' | '.join(f'{v["km2"]:.1f}' + ('' if k == names[0] else f' ({v["iou"]:.2f})') for k, v in row.items()) + ' |')
L_ += ['', f'## High ground: at least {RH:g} m, pieces of {A_MIN * cell ** 2 / 1e4:g} ha or more; running to it from the {res["people_km2"]:.1f} km² '
       f'of land below {LM:g} m (GSI)', '', f'Running as in the demo: {float(p.demo.get("speed_kmh", 8)):g} km/h on the flat, slower uphill (Tobler), '
       'round the water, to the nearest piece by time.', '',
       '| model | km² | wrong (not high in GSI) | running time, median | within 2 min of GSI\'s | 5 min longer or more | the route ends on real high ground |',
       '|---|---|---|---|---|---|---|']
for k, v in res['high_ground'].items():
    if k == names[0]:
        L_.append(f'| {k} | {v["km2"]:.1f} | | {v["minutes_median"]:.1f} min (90%: {v["minutes_p90"]:.1f}) | | | |')
    else:
        L_.append(f'| {k} | {v["km2"]:.1f} | {v["wrong_km2"]:.1f} | {v["minutes_median"]:.1f} min | {v["within_2min"]:.0%} | '
                  f'{v["longer_5min"]:.0%} | {v["route_to_real"]:.0%} |')
open(p.product / 'floods.md', 'w').write('\n'.join(L_) + '\n')
print('\n'.join(L_))

# ---- the map: the sea at the map level, high ground right and wrong, for GSI, the surface and the ground model
SEA, FLOOD, HIGH, WRONG = '#c9d6e3', '#2a78d6', '#1baf7a', '#eb6834'
near_real = dG <= 50
fig, axs = plt.subplots(1, 3, figsize=(19, 7.6), dpi=110)
fig.patch.set_facecolor('#fcfcfb')
for ax, name in zip(axs, names[:3]):
    H = models[name]
    Hn = np.nan_to_num(H, nan=0.0)
    gy, gx = np.gradient(Hn, cell)
    sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    shade = np.clip(np.sin(np.radians(45)) * np.cos(sl) + np.cos(np.radians(45)) * np.sin(sl) * np.cos(np.radians(315) - asp), 0, 1)
    rgb = np.dstack([0.55 + 0.4 * shade] * 3)
    rgb[sea | inland] = matplotlib.colors.to_rgb(SEA)
    rgb[flooded(H, LM)] = matplotlib.colors.to_rgb(FLOOD)
    rgb[maps[name] & near_real] = matplotlib.colors.to_rgb(HIGH)
    rgb[maps[name] & ~near_real] = matplotlib.colors.to_rgb(WRONG)
    ax.imshow(rgb, interpolation='antialiased'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#d9d8d3')
    fk = res['flood'][f'{LM:g}'][name]['km2'] if f'{LM:g}' in res['flood'] else flooded(H, LM).sum() * km2
    hg = res['high_ground'][name]
    sub = (f'sea reaches {fk:.1f} km² · high ground: a {hg["minutes_median"]:.1f} min run in the median' if name == names[0] else
           f'sea reaches {fk:.1f} km² · {hg["within_2min"]:.0%} of the low city run within 2 min of GSI\'s time')
    ax.set_title(name, fontsize=13, color='#0b0b0b', loc='left', pad=22)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=10.5, color='#52514e', va='bottom')
fig.legend(handles=[Patch(color=SEA, label='sea, rivers, ponds (WorldCover)'),
                    Patch(color=FLOOD, label=f'land the sea reaches at {LM:g} m above mean sea level'),
                    Patch(color=HIGH, label=f'ground ≥{RH:g} m, pieces ≥{A_MIN * cell ** 2 / 1e4:g} ha (within 50 m of the GSI one)'),
                    Patch(color=WRONG, label='"high ground" that is not: roofs, blocks of buildings')],
           loc='lower center', ncol=2, frameon=False, fontsize=11, labelcolor='#0b0b0b')
fig.text(0.01, 0.985, f'{p.name}: where the sea goes at {LM:g} m, and where the high ground is', fontsize=15, color='#0b0b0b', va='top')
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.03)
fig.savefig(p.product / 'floods_map.png', facecolor=fig.get_facecolor())
