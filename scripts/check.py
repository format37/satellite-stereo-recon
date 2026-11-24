#!/usr/bin/env python3
"""Step 6. The DSM against independent elevation models: the numbers, and where it fails.

    python3 -I scripts/check.py data/hiroshima/fb.yaml

dh = our DSM - reference, heights above T.P. (sea level), on land only (not water, product/water.tif and WorldCover).

Reference 1: GSI elevation tiles (ref/gsi.tif), DEM5A from airborne laser where it exists, else DEM10B. A terrain
model: buildings and trees are removed from it, so over a roof dh is the building's height, over a forest the
canopy's. Hence the numbers by WorldCover class:
  open      grass, crops, bare, shrubs, wetland: the like-for-like comparison of ground with ground
  built-up  roofs among streets: dh is mostly building height
  trees     canopy height plus the slopes the forest is on
and 'flat open ground' (open, slope < 10 deg) as the headline accuracy. Slope and aspect come from the reference.
Co-registration (Nuth & Kaab 2011): on land that is not built-up, with slopes of 5-45 deg, dh / tan(slope) against
aspect gives the horizontal shift of the DSM against the reference; the vertical offset is then the median dh on
flat open ground. Numbers are given before (absolute accuracy: the RPCs alone, no ground control) and after removing
shift and offset (what is left: noise and the DSM's own shape errors). The offset is what a flood model would have
to correct for if the DSM were used without control points.
Reference 2: Copernicus GLO-30 (ref/copernicus.tif, a surface model like ours, 2011-15), both averaged to 30 m.
Where it fails: dh by slope, by distance to water, by number of pairs; forest on steep slopes by aspect (is it the
sun?) and by curvature (is it the shape: ridges cut, valleys filled?); the largest connected areas of
|dh - median of their class| > 10 m.
Low-lying land: the area below 1, 2, 3, 5 m T.P. by the GSI DEM and by the DSM, and their overlap: what a flood
model built on the DSM would get right.

The DSM tied to a reference (product/dsm_anchored.tif) is anchor.py's: anchored to Copernicus GLO-30 (global), the
check against the GSI DEM is independent of it; anchored to the GSI DEM itself, it is the co-registration above.
Its numbers are given apart ('anchored'), with the reference that set its level, by class and by slope.
The ground model (product/ground.tif, ground.py) is checked the same way, ground against ground: by class, on flat
built-up land (where the DSM stands on roofs), and by slope.

Writes product/check.json, check.md (the tables), check_dh.png (map of dh over the hillshade), check_hist.png.
Runs only where check.reference names GSI layers (Japan); elsewhere anchor.json has the agreement with Copernicus.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coreg import nuth_kaab
from geo import Grid, stats
from project import load

p = load(sys.argv[1])
if not p.reference:
    print(f'{p.name}: no check reference (check.reference is empty): nothing to check against')
    sys.exit(0)
g = Grid(p)
ref = p.vref
H = g.read(p.product / 'dsm.tif')
gsi = g.read(ref / 'gsi.tif')
src = np.nan_to_num(g.read(ref / 'gsi_source.tif')).astype(int)
wc = np.nan_to_num(g.read(ref / 'worldcover.tif')).astype(int)
water = np.nan_to_num(g.read(p.product / 'water.tif')) > 0
cnt = np.nan_to_num(g.read(p.product / 'dsm_count.tif')).astype(int)
N_gsi, N_egm = g.read(ref / 'geoid_gsigeo2011.tif'), g.read(ref / 'geoid_egm2008.tif')
N_run = g.read(ref / f'geoid_{p.geoid}.tif')
H_tp = H + N_run - N_gsi                                   # our DSM in T.P. heights whatever dsm.geoid was
cop = g.read(ref / 'copernicus.tif') + N_egm - N_gsi        # Copernicus (EGM2008) in T.P. heights

land = ~water & (wc != 80) & (wc > 0)
CLS = {'open': np.isin(wc, [20, 30, 40, 60, 90]), 'built-up': wc == 50, 'trees': wc == 10}
gs = cv2.GaussianBlur(np.nan_to_num(gsi, nan=0).astype(np.float32), (0, 0), 1.5)
gy, gx = np.gradient(gs, g.res)                             # gy: d/drow (south), gx: d/dcol (east)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = np.degrees(np.arctan2(-gx, gy)) % 360               # direction of steepest descent, clockwise from north
dist_w = ndimage.distance_transform_edt(~water) * g.res
flat_open = CLS['open'] & (slope < 10)


def table(dh, sel=None):
    base = land & np.isfinite(dh) & np.isfinite(gsi)
    if sel is not None:
        base &= sel
    return stats(dh[base])


def shifted(a, sx, sy):
    """a resampled so that the value at a cell is what a had at (x + sx, y + sy)"""
    i, j = np.mgrid[0:g.height, 0:g.width].astype(np.float32)
    return ndimage.map_coordinates(a, [i - sy / g.res, j + sx / g.res], order=1, mode='constant', cval=np.nan)


res = {'reference': 'GSI elevation tiles: ' + ', '.join(p.reference), 'gsi_source_share_of_land': {}}
for k, name in enumerate(p.reference, 1):
    res['gsi_source_share_of_land'][name] = float((src[land] == k).mean())
dh = H_tp - gsi
res['before'] = {'all land': table(dh), 'flat open ground': table(dh, flat_open)}
res['before'].update({c: table(dh, m) for c, m in CLS.items()})

# ---- Nuth & Kaab: horizontal shift on slopes, then the vertical offset on flat open ground (coreg.py)
sel0 = land & ~CLS['built-up'] & (slope > 5) & (slope < 45) & np.isfinite(gsi)
sx, sy, n_it = nuth_kaab(H_tp, gsi, sel0, slope, aspect, g.res)
it = n_it - 1
d = shifted(H_tp, sx, sy) - gsi
bias = float(np.median(d[land & flat_open & np.isfinite(d)]))
res['coregistration'] = dict(method='Nuth & Kaab 2011 on non-built-up land, slope 5-45 deg; offset = median dh on flat open ground',
                             shift_east_m=float(sx), shift_north_m=float(sy), shift_m=float(np.hypot(sx, sy)),
                             vertical_offset_m=bias, iterations=it + 1, cells=int(sel0.sum()))
dh2 = d - bias
res['after'] = {'all land': table(dh2), 'flat open ground': table(dh2, flat_open)}
res['after'].update({c: table(dh2, m) for c, m in CLS.items()})
# the DSM as anchor.py tied it to its reference (heights above T.P. here). Without ground control a PRISM DSM's level
# is only as good as the RPCs: the three delivered for Hiroshima disagree by up to 2.4 px along track.
af = p.product / 'anchor.json'
if af.exists():
    anch = json.load(open(af))
    Hc_anch = g.read(p.product / 'dsm_anchored.tif') + N_run - N_gsi
else:                                                      # older runs: anchored here, to the GSI DEM
    anch = dict(reference='gsi (check.py)', shift_east_m=float(sx), shift_north_m=float(sy), vertical_offset_m=bias)
    Hc_anch = (shifted(H_tp, sx, sy) - bias).astype(np.float32)
    g.write(p.product / 'dsm_anchored.tif', (Hc_anch - N_run + N_gsi).astype(np.float32))
dha = Hc_anch - gsi
res['anchored'] = {'reference': anch['reference'], 'shift_east_m': anch['shift_east_m'], 'shift_north_m': anch['shift_north_m'],
                   'vertical_offset_m': anch['vertical_offset_m'], 'all land': table(dha), 'flat open ground': table(dha, flat_open)}
res['anchored'].update({c: table(dha, m) for c, m in CLS.items()})
SLOPES = ((0, 5), (5, 15), (15, 30), (30, 90))
flat_built = CLS['built-up'] & (slope < 5)
res['anchored']['built-up, slope under 5 deg'] = table(dha, flat_built)
res['anchored_by_slope'] = {f'{a}-{b} deg': table(dha, (slope >= a) & (slope < b) & ~CLS['built-up']) for a, b in SLOPES}
gf = p.product / 'ground.tif'
if gf.exists():                                            # the ground model, on the level of the anchored DSM
    dg = g.read(gf) + N_run - N_gsi - gsi
    res['ground'] = {'all land': table(dg), 'flat open ground': table(dg, flat_open),
                     'built-up, slope under 5 deg': table(dg, flat_built)}
    res['ground'].update({c: table(dg, m) for c, m in CLS.items()})
    res['ground_by_slope'] = {f'{a}-{b} deg': table(dg, (slope >= a) & (slope < b) & ~CLS['built-up']) for a, b in SLOPES}
res['pairs'] = {}                                          # each pair on its own, shifted and offset like the fusion
for pr in p.pairs:
    f = p.product / f'dsm_{"".join(pr)}.tif'
    if f.exists() and len(p.pairs) > 1:
        Hp = g.read(f) + N_run - N_gsi
        dp = Hp - gsi
        dpc = shifted(Hp, sx, sy) - gsi - bias
        res['pairs'][''.join(pr)] = {'flat open ground, as produced': table(dp, flat_open), 'flat open ground, after': table(dpc, flat_open),
                                     'built-up, after': table(dpc, CLS['built-up']), 'trees, after': table(dpc, CLS['trees']),
                                     'land measured': float(np.isfinite(Hp[land]).mean())}
res['after_by_gsi_source'] = {name: table(dh2, flat_open & (src == k)) for k, name in enumerate(p.reference, 1)}
res['after_by_slope'] = {f'{a}-{b} deg': table(dh2, (slope >= a) & (slope < b) & ~CLS['built-up'])
                         for a, b in ((0, 5), (5, 15), (15, 30), (30, 90))}
res['after_by_distance_to_water'] = {f'{a}-{b} m': table(dh2, (dist_w >= a) & (dist_w < b) & ~CLS['built-up'] & ~CLS['trees'])
                                     for a, b in ((0, 25), (25, 100), (100, 99999))}
if len(p.pairs) > 1:
    res['after_by_pairs'] = {f'{k} pairs': table(dh2, (cnt == k) & flat_open) for k in range(1, len(p.pairs) + 1)}

# ---- Copernicus GLO-30, both averaged to 30 m blocks
B = int(round(30 / g.res))
hh, ww = g.height // B * B, g.width // B * B
def blocks(a, frac=0.5):
    x = a[:hh, :ww].reshape(hh // B, B, ww // B, B)
    ok = np.isfinite(x).mean(axis=(1, 3)) >= frac
    with np.errstate(all='ignore'):
        mean = np.nanmean(x, axis=(1, 3))
    return np.where(ok, mean, np.nan)
d30 = blocks(H_tp) - blocks(cop, 0.99)
land30 = blocks(np.where(land, 1.0, np.nan), 0.9) == 1
wc30 = {c: blocks(np.where(m, 1.0, np.nan), 0.6) == 1 for c, m in CLS.items()}
res['copernicus_30m'] = {'all land': stats(d30[land30])}
res['copernicus_30m'].update({c: stats(d30[land30 & m]) for c, m in wc30.items()})
d30a = blocks(Hc_anch) - blocks(cop, 0.99)                 # the anchored DSM: surface against surface, the shape
res['copernicus_30m_anchored'] = {'all land': stats(d30a[land30])}
res['copernicus_30m_anchored'].update({c: stats(d30a[land30 & m]) for c, m in wc30.items()})

# ---- forest on slopes: is the error about the sun (aspect) or about the shape of the ground (curvature)?
lap = cv2.Laplacian(cv2.GaussianBlur(np.nan_to_num(gsi, nan=0).astype(np.float32), (0, 0), 4), cv2.CV_32F) / g.res ** 2
steep = land & CLS['trees'] & (slope > 15) & np.isfinite(dh2)
res['trees_steep_by_aspect'] = {f'faces {a} deg': table(dh2, steep & (((aspect - a + 22.5) % 360) < 45)) for a in range(0, 360, 45)}
q = np.nanpercentile(lap[steep], [0, 10, 30, 70, 90, 100]) if steep.any() else np.zeros(6)
names_c = ['ridges (most convex 10%)', 'convex 10-30%', 'middle 30-70%', 'concave 70-90%', 'valleys (most concave 10%)']
res['trees_steep_by_curvature'] = {nm: table(dh2, steep & (lap >= a) & (lap <= b)) for nm, a, b in zip(names_c, q[:-1], q[1:])}

# ---- low-lying land: what a flood model would see. Area of land below a height, by the GSI DEM (bare ground) and by
# the DSM (roofs and canopy included) as produced and after co-registration; overlap of the DSM's area with GSI's.
Hc = shifted(H_tp, sx, sy) - bias
both = land & np.isfinite(gsi) & np.isfinite(Hc) & np.isfinite(H_tp) & np.isfinite(Hc_anch)
res['low_land'] = {}
for t in (1, 2, 3, 5):
    a, b, c2, c3 = both & (gsi < t), both & (H_tp < t), both & (Hc < t), both & (Hc_anch < t)
    res['low_land'][f'below {t} m'] = dict(
        gsi_km2=float(a.sum() * g.res ** 2 / 1e6), dsm_km2=float(b.sum() * g.res ** 2 / 1e6), dsm_coreg_km2=float(c2.sum() * g.res ** 2 / 1e6),
        dsm_anchored_km2=float(c3.sum() * g.res ** 2 / 1e6),
        iou=float((a & b).sum() / max((a | b).sum(), 1)), iou_coreg=float((a & c2).sum() / max((a | c2).sum(), 1)),
        iou_anchored=float((a & c3).sum() / max((a | c3).sum(), 1)),
        gsi_low_also_low_in_dsm_coreg=float((a & c2).sum() / max(a.sum(), 1)))

# ---- where it fails: the largest areas that differ from their class by more than 10 m
cls_med = np.zeros(g.shape, np.float32)
for c, m in CLS.items():
    cls_med[m] = res['after'][c].get('median', 0.0)
bad = land & np.isfinite(dh2) & (np.abs(dh2 - cls_med) > 10)
bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
lab, n = ndimage.label(bad)
area = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * g.res ** 2
top = np.argsort(area)[::-1][:12] + 1
hot = []
names = {10: 'trees', 20: 'shrubs', 30: 'grass', 40: 'crops', 50: 'built-up', 60: 'bare', 90: 'wetland'}
for k in top:
    m = lab == k
    ii, jj = np.nonzero(m)
    x, y = g.xy()
    lon, lat = g.to_ll.transform(x[int(jj.mean())], y[int(ii.mean())])
    dom = np.bincount(wc[m], minlength=101).argmax()
    hot.append(dict(lon=round(float(lon), 5), lat=round(float(lat), 5), area_ha=round(float(m.sum() * g.res ** 2 / 1e4), 2),
                    dh_median=round(float(np.nanmedian(dh2[m])), 1), cover=names.get(int(dom), str(dom)),
                    slope_median=round(float(np.median(slope[m])), 1), dist_to_water_m=round(float(np.median(dist_w[m])), 0)))
res['hotspots'] = hot
res['bad_share_of_land'] = float(bad[land].mean())
json.dump(res, open(p.product / 'check.json', 'w'), indent=1)

# ---- tables
def row(name, s, cell=None):
    if not s.get('n'):
        return f'| {name} | 0 | | | | | | |'
    return (f'| {name} | {s["n"] * (cell or g.res ** 2) / 1e6:.2f} | {s["mean"]:+.2f} | {s["median"]:+.2f} | {s["rms"]:.2f} | '
            f'{s["nmad"]:.2f} | {s["le68"]:.2f} | {s["le95"]:.2f} |')
hdr = '| cells | km² | mean | median | RMS | NMAD | 68% of abs | 95% of abs |\n|---|---|---|---|---|---|---|---|'
L = [f'# {p.name}: DSM against GSI (heights above T.P., metres; dh = DSM - GSI)', '',
     f'GSI layers on the land: ' + ', '.join(f'{k} {v:.1%}' for k, v in res['gsi_source_share_of_land'].items()), '',
     '## As produced (RPCs only, no ground control)', '', hdr]
L += [row(k, v) for k, v in res['before'].items()]
c = res['coregistration']
L += ['', f'## After co-registration: shift {c["shift_east_m"]:+.2f} m east, {c["shift_north_m"]:+.2f} m north '
          f'({c["shift_m"]:.2f} m), offset {c["vertical_offset_m"]:+.2f} m removed', '', hdr]
L += [row(k, v) for k, v in res['after'].items()]
a_ = res['anchored']
L += ['', f'## Anchored by anchor.py to {a_["reference"]}: shift {a_["shift_east_m"]:+.2f} m east, {a_["shift_north_m"]:+.2f} m north, '
          f'offset {a_["vertical_offset_m"]:+.2f} m removed (product/dsm_anchored.tif, the one the voxels and the demo use)', '', hdr]
L += [row(k, v) for k, v in a_.items() if isinstance(v, dict)]
L += ['', '### anchored, by slope (not built-up)', '', hdr] + [row(k, v) for k, v in res['anchored_by_slope'].items()]
if 'ground' in res:
    L += ['', '## The ground model (ground.py) against GSI: ground against ground, on the level of the anchored DSM', '', hdr]
    L += [row(k, v) for k, v in res['ground'].items()]
    L += ['', '### the ground model, by slope (not built-up)', '', hdr] + [row(k, v) for k, v in res['ground_by_slope'].items()]
    L += ['', '## After co-registration to GSI (the tables below: the shape, shift and offset against GSI removed)']
for sec, title in (('after_by_slope', 'by slope (not built-up)'), ('after_by_distance_to_water', 'by distance to water (open ground)'),
                   ('after_by_gsi_source', 'by GSI layer (flat open ground)'), ('after_by_pairs', 'by number of pairs (flat open ground)')):
    if sec in res:
        L += ['', f'### {title}', '', hdr] + [row(k, v) for k, v in res[sec].items()]
if res['pairs']:
    L += ['', '### each pair on its own (shift and offset of the fusion)', '', hdr]
    for k, v in res['pairs'].items():
        L += [row(f'{k}: {kk}', vv) for kk, vv in v.items() if kk != 'land measured']
for sec, title in (('trees_steep_by_aspect', 'forest on slopes over 15 deg, by the way the slope faces (sun from 152 deg)'),
                   ('trees_steep_by_curvature', 'forest on slopes over 15 deg, by the curvature of the ground (GSI)')):
    L += ['', f'### {title}', '', hdr] + [row(k, v) for k, v in res[sec].items()]
L += ['', '## Low-lying land (on the cells both models have): km² below a height, and how well the areas overlap (IoU)', '',
      '| height, T.P. | GSI (ground) | DSM as produced | DSM co-registered | IoU as produced | IoU co-registered | GSI low land the DSM also has low |',
      '|---|---|---|---|---|---|---|']
L += [f'| {k} | {v["gsi_km2"]:.2f} | {v["dsm_km2"]:.2f} | {v["dsm_coreg_km2"]:.2f} | {v["iou"]:.2f} | {v["iou_coreg"]:.2f} | '
      f'{v["gsi_low_also_low_in_dsm_coreg"]:.0%} |' for k, v in res['low_land'].items()]
L += ['', '## Copernicus GLO-30 (surface model), both at 30 m, no co-registration', '', hdr]
L += [row(k, v, (B * g.res) ** 2) for k, v in res['copernicus_30m'].items()]
L += ['', f'## Copernicus GLO-30, both at 30 m, our DSM anchored (to {a_["reference"]})', '', hdr]
L += [row(k, v, (B * g.res) ** 2) for k, v in res['copernicus_30m_anchored'].items()]
L += ['', f'## Largest areas off their class by more than 10 m ({res["bad_share_of_land"]:.1%} of the land)', '',
      '| lat | lon | ha | median dh | cover | slope | to water, m |', '|---|---|---|---|---|---|---|']
L += [f'| {h["lat"]} | {h["lon"]} | {h["area_ha"]} | {h["dh_median"]:+.1f} | {h["cover"]} | {h["slope_median"]} | {h["dist_to_water_m"]:.0f} |' for h in hot]
open(p.product / 'check.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ---- pictures: dh over the hillshade, histograms
hs = cv2.GaussianBlur(np.nan_to_num(gsi, nan=0).astype(np.float32), (0, 0), 1)
gy2, gx2 = np.gradient(hs, g.res)
az, alt = np.radians(315), np.radians(45)
shade = np.clip(np.sin(alt) * np.cos(np.arctan(np.hypot(gx2, gy2))) +
                np.cos(alt) * np.sin(np.arctan(np.hypot(gx2, gy2))) * np.cos(az - np.arctan2(-gx2, gy2)), 0, 1)
fig, ax = plt.subplots(figsize=(12, 10.6), dpi=110)
ax.imshow(shade, cmap='gray', vmin=0, vmax=1.2)
show = np.where(land, dh2, np.nan)
im = ax.imshow(show, cmap='RdBu_r', vmin=-20, vmax=20, alpha=0.85)
ax.imshow(np.where(water, 1.0, np.nan), cmap='Blues', vmin=0, vmax=2, alpha=0.5)
ax.set_title(f'{p.name}: DSM - GSI DEM after co-registration (m); roofs and canopy show red, holes grey', fontsize=10)
ax.set_axis_off(); fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
fig.savefig(p.product / 'check_dh.png', bbox_inches='tight'); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
for c, m in [('flat open ground', flat_open)] + list(CLS.items()):
    v = dh2[land & m & np.isfinite(dh2)]
    ax.hist(np.clip(v, -30, 50), bins=160, range=(-30, 50), histtype='step', density=True, label=f'{c} ({v.size * g.res ** 2 / 1e6:.1f} km²)')
ax.set_xlabel('DSM - GSI DEM, m (after co-registration)'); ax.set_ylabel('density'); ax.legend(); ax.set_title(p.name)
fig.savefig(p.product / 'check_hist.png', bbox_inches='tight'); plt.close(fig)
