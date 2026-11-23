#!/usr/bin/env python3
"""Every number of a run on one page: out/<variant>/summary.md.

    python3 -I scripts/summary.py data/hiroshima/fb.yaml

Reads what the steps wrote (product/*.json, ba/*stats.txt, the ASP logs) and measures nothing itself.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project import load, pair_name

p = load(sys.argv[1])
P = p.product
J = lambda f: json.load(open(P / f)) if (P / f).exists() else {}
views, masks, dsm, chk, vox = J('views.json'), J('masks.json'), J('dsm.json'), J('check.json'), J('voxels.json')
ref = json.load(open(p.vref / 'ref.json')) if (p.vref / 'ref.json').exists() else {}


def ba_stats(f):
    out = {}
    if f.exists():
        for line in open(f):
            if not line.startswith('#') and ',' in line:
                name, mean, med, n = [x.strip() for x in line.split(',')]
                out[Path(name).stem.replace('_m', '')] = (float(mean), float(med), int(n))
    return out


def wall_minutes(pair):
    """wall time of parallel_stereo for a pair, from the time stamps run_stereo.sh prints"""
    f = p.logs / 'run_stereo.out'
    if not f.exists():
        return None
    t0 = t1 = None
    for line in open(f):
        m = re.match(r'(\d\d):(\d\d):(\d\d) (\w+)(?: (\w) (\w))?', line)
        if not m:
            continue
        sec = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        if m[4] == 'parallel_stereo' and (m[5] or '') + (m[6] or '') == pair:
            t0 = sec
        elif m[4] == 'point2dem' and t0 is not None and line.split()[2] == pair:
            t1 = sec
            break
    return None if t0 is None or t1 is None else ((t1 - t0) % 86400) / 60


L = [f'# {p.name}', '', f'`{p.yaml.relative_to(p.project.parent.parent)}`: views {", ".join(p.views)}, pairs '
     f'{", ".join(pair_name(x) for x in p.pairs)}; {p.algorithm}, subpixel mode {p.subpixel}; DSM {p.res:g} m; geoid {p.geoid}', '']
L += ['## Views', '', '| view | scene | time UTC | off nadir | looks from | crop px | water | cloud |', '|---|---|---|---|---|---|---|---|']
for v, d in views.get('views', {}).items():
    mv = masks.get('views', {}).get(v, {})
    L.append(f'| {v} | {d["scene"]} | {d["time_utc"][9:17]} | {d["off_nadir"]:.1f} deg | {d["look_azimuth"]:.0f} deg | '
             f'{d["window"]["width"]} x {d["window"]["height"]} | {mv.get("water_share", 0):.1%} | {mv.get("cloud_share", 0):.2%} |')
L += ['', '| pair | convergence | B/H | height per px of disparity |', '|---|---|---|---|']
for k, d in views.get('pairs', {}).items():
    L.append(f'| {k} | {d["convergence_deg"]:.1f} deg | {d["base_to_height"]:.2f} | {d["metres_per_px_disparity"]:.2f} m |')

b0, b1 = ba_stats(p.ba / 'run-initial_residuals_stats.txt'), ba_stats(p.ba / 'run-final_residuals_stats.txt')
if b1:
    L += ['', f'## Bundle adjustment ({p.ba_fixed} fixed)', '', '| view | matches | median px before | after | mean before | after |',
          '|---|---|---|---|---|---|']
    for v in b1:
        L.append(f'| {v} | {b1[v][2]} | {b0.get(v, (0, 0, 0))[1]:.3f} | {b1[v][1]:.3f} | {b0.get(v, (0, 0, 0))[0]:.3f} | {b1[v][0]:.3f} |')

L += ['', '## Stereo and DSM', '', '| pair | land measured | ray gap, median m | parallel_stereo wall time |', '|---|---|---|---|']
for k, d in dsm.get('pairs', {}).items():
    t = wall_minutes(k)
    gap = d.get('intersection_error_median')
    L.append(f'| {k} | {d["land_coverage"]:.1%} | {"" if gap is None else f"{gap:.2f}"} | {"" if t is None else f"{t:.1f} min"} |')
if dsm:
    L += ['', f'Fused ({p.fuse}): {dsm["measured_of_land"]:.1%} of the land measured; holes on land: no match '
              f'{dsm["nomatch_holes_of_land"]:.1%}, cloud {dsm["cloud_holes_of_land"]:.2%}. Heights above sea level ({dsm["geoid"]}, '
              f'N = {dsm["N_min"]:.2f}..{dsm["N_max"]:.2f} m; {dsm["N_minus_egm2008"]:+.2f} m against EGM2008): 1% {dsm["height_p01"]:.1f} m, '
              f'99% {dsm["height_p99"]:.1f} m.']
if chk:
    def s(d, k='after'):
        x = chk[k].get(d, {})
        return (f'median {x["median"]:+.2f}, RMS {x["rms"]:.2f}, NMAD {x["nmad"]:.2f}, 95% {x["le95"]:.2f} m ({x["n"] * p.res ** 2 / 1e6:.2f} km²)'
                if x.get('n') else 'no cells')
    c = chk['coregistration']
    L += ['', '## Against the GSI DEM (laser, bare ground; dh = DSM - GSI)', '',
          f'- flat open ground, as produced: {s("flat open ground", "before")}',
          f'- all land, as produced: {s("all land", "before")}',
          f'- co-registration: shift {c["shift_east_m"]:+.2f} m east, {c["shift_north_m"]:+.2f} m north, offset {c["vertical_offset_m"]:+.2f} m',
          f'- flat open ground, after: {s("flat open ground")}',
          f'- built-up, after: {s("built-up")} (roofs: building height)',
          f'- trees, after: {s("trees")} (canopy)',
          f'- Copernicus GLO-30, 30 m, all land: ' + (lambda x: f'median {x["median"]:+.2f}, RMS {x["rms"]:.2f}, NMAD {x["nmad"]:.2f} m')(chk['copernicus_30m']['all land']),
          '', f'Tables: `product/check.md`. Largest failures: ' + '; '.join(
              f'{h["area_ha"]} ha of {h["cover"]} at {h["lat"]}, {h["lon"]} ({h["dh_median"]:+.0f} m)' for h in chk['hotspots'][:5])]
if vox:
    L += ['', '## Voxels', '', f'{vox["voxels"]:,} voxels of {vox["size"]:g} m ({vox["dims"][0]} x {vox["dims"][1]} x {vox["dims"][2]}), '
              f'{vox["land_columns"]:,} land columns of which {vox["filled_columns"]:,} filled, {vox["water_columns"]:,} water at 0 m; '
              f'open holes {vox["open_hole_share_of_land"]:.1%} of the land. `python3 scripts/voxview.py {p.yaml.relative_to(p.project.parent.parent)}`']
if ref:
    L += ['', '## Reference data on the grid', '',
          'GSI: ' + ', '.join(f'{k} {v["share_of_grid"]:.1%} of the grid' for k, v in ref.get('gsi', {}).items()) +
          f'; WorldCover: ' + ', '.join(f'{k} {v:.1%}' for k, v in ref.get('worldcover', {}).items())]
L += ['', 'Pictures: `product/renders/` (dsm.png, dsm_city.png, holes.png, voxels_*.png), `product/check_dh.png`, `check_hist.png`.']
open(p.run / 'summary.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
