#!/usr/bin/env python3
"""Step 8. Pictures of the DSM and the voxel model, headless (matplotlib, Open3D's offscreen renderer through EGL).

    python3 -I scripts/render.py data/hiroshima/fb.yaml

Writes product/renders/:
  dsm.png           the DSM above sea level: colour by height over a hillshade, water blue, holes grey
  dsm_city.png      the same at full resolution over the centre of the delta (3 x 3 km around the castle)
  ortho_city.png    the orthophoto of that square, to compare
  holes.png         what is measured and why the rest is not (water, cloud, no match)
  fails.png         the largest disagreements with the GSI DEM (check.py), 600 m squares: ortho, DSM, GSI, difference
  voxels_*.png      the voxel model, lit by the sun of the acquisition (azimuth 152 deg, elevation 41.5 deg):
                    overview (cubes of 2 x voxels.size), the delta from the south-east, the castle and the station
                    up close; *_height.png coloured by height above sea level
These are pictures of a product made from licensed scenes: they stay with the run (docs/data.md).
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import Grid
from project import load
from voxmesh import columns, mesh, rebin

p = load(sys.argv[1])
g = Grid(p)
out = p.product / 'renders'; out.mkdir(parents=True, exist_ok=True)
H = g.read(p.product / 'dsm.tif')
holes = np.nan_to_num(g.read(p.product / 'holes.tif'), nan=255).astype(int)
water = holes == 1
ortho = g.read(p.product / 'ortho.tif')
views = json.load(open(p.product / 'views.json'))
sun_az, sun_el = views['views'][p.views[0]]['sun_azimuth'], views['views'][p.views[0]]['sun_elevation']
x0, y0, x1, y1 = g.bounds
CITY = (132.4594, 34.3963)                                   # Hiroshima Castle / Peace Park, the centre of the delta


def relief(Hm, res, vmax=None):
    vmax = vmax or float(np.nanpercentile(Hm, 99.5))
    ls = LightSource(azdeg=315, altdeg=45)
    z = np.where(np.isfinite(Hm), Hm, np.nanmedian(Hm))
    cmap = plt.get_cmap('terrain')
    rgb = ls.shade(np.clip(z, -5, vmax), cmap=cmap, blend_mode='soft', vert_exag=2, dx=res, dy=res, vmin=-30, vmax=vmax)[..., :3]
    return rgb, vmax


def save_map(rgb, extent, title, path, vmax=None, cbar=True):
    fig, ax = plt.subplots(figsize=(11, 10), dpi=120)
    ax.imshow(rgb, extent=extent)
    ax.set_title(title, fontsize=10); ax.set_xlabel('UTM 53N east, km'); ax.set_ylabel('north, km')
    if cbar and vmax:
        sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(-30, vmax)); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label='m above sea level (T.P.)')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


# ---- DSM, whole box
rgb, vmax = relief(H, g.res)
rgb[water] = [0.55, 0.70, 0.88]
rgb[(holes == 2)] = [0.85, 0.35, 0.85]
rgb[(holes == 3)] = [0.6, 0.6, 0.6]
ext = [x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3]
save_map(rgb, ext, f'{p.name}: DSM, {g.res:g} m, heights above sea level ({p.geoid}); water blue, no match grey, cloud magenta',
         out / 'dsm.png', vmax)

# ---- the centre of the delta at full resolution, with the orthophoto
cx, cy = g.from_ll.transform(*CITY)
j0, i0 = int((cx - 1500 - x0) / g.res), int((y1 - (cy + 1500)) / g.res)
n = int(3000 / g.res)
sub = (slice(max(i0, 0), i0 + n), slice(max(j0, 0), j0 + n))
rgb_c, _ = relief(H[sub], g.res, vmax=80)
rgb_c[water[sub]] = [0.55, 0.70, 0.88]
rgb_c[holes[sub] >= 2] = [0.6, 0.6, 0.6]
ext_c = [(cx - 1500) / 1e3, (cx + 1500) / 1e3, (cy - 1500) / 1e3, (cy + 1500) / 1e3]
save_map(rgb_c, ext_c, f'{p.name}: DSM over the centre of the delta (castle, Peace Park), 3 x 3 km, colour 0-80 m', out / 'dsm_city.png', 80)
fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
ax.imshow(ortho[sub], cmap='gray', vmin=0, vmax=255, extent=ext_c); ax.set_title(f'{p.name}: orthophoto (nadir view on the DSM), same square', fontsize=10)
fig.savefig(out / 'ortho_city.png', bbox_inches='tight'); plt.close(fig)

# ---- holes
hc = np.zeros(g.shape + (3,), np.float32) + 0.92
hc[holes == 0] = [0.35, 0.65, 0.35]; hc[water] = [0.55, 0.70, 0.88]; hc[holes == 2] = [0.85, 0.35, 0.85]; hc[holes == 3] = [0.15, 0.15, 0.15]
dj = json.load(open(p.product / 'dsm.json'))
save_map(hc, ext, f'{p.name}: measured (green) {dj["measured_of_land"]:.1%} of the land; no match (black) {dj["nomatch_holes_of_land"]:.1%}, '
         f'cloud (magenta) {dj["cloud_holes_of_land"]:.2%}; water blue', out / 'holes.png', cbar=False)
print('DSM pictures ->', out)

# ---- where it fails: the largest disagreements of check.py, side by side with the orthophoto and the GSI DEM
ck = json.load(open(p.product / 'check.json')) if (p.product / 'check.json').exists() else None
if ck:
    gsi = g.read(p.run / 'ref' / 'gsi.tif')
    co = ck['coregistration']
    seen, picks = set(), []
    for h_ in ck['hotspots']:                                  # one per kind: tall buildings, ridges, valleys...
        kind = (h_['cover'], h_['dh_median'] > 0)
        if kind not in seen:
            seen.add(kind); picks.append(h_)
    picks = picks[:4]
    half = int(300 / g.res)
    fig, axs = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)), dpi=100)
    for r_, h_ in enumerate(np.atleast_1d(picks)):
        px, py = g.from_ll.transform(h_['lon'], h_['lat'])
        jc, ic = int((px - x0) / g.res), int((y1 - py) / g.res)
        sl = (slice(max(ic - half, 0), ic + half), slice(max(jc - half, 0), jc + half))
        dsm_c = H[sl] - co['vertical_offset_m']
        lo_, hi_ = np.nanpercentile(np.r_[gsi[sl][np.isfinite(gsi[sl])], dsm_c[np.isfinite(dsm_c)]], [2, 98])
        for c_, (a_, cm, t_) in enumerate(((ortho[sl], 'gray', 'orthophoto (nadir)'), (dsm_c, 'terrain', 'DSM, offset removed'),
                                           (gsi[sl], 'terrain', 'GSI DEM5A (laser, ground)'), (dsm_c - gsi[sl], 'RdBu_r', 'DSM - GSI'))):
            ax = axs[r_, c_] if len(picks) > 1 else axs[c_]
            kw = dict(vmin=0, vmax=255) if c_ == 0 else (dict(vmin=-20, vmax=20) if c_ == 3 else dict(vmin=lo_, vmax=hi_))
            im_ = ax.imshow(a_, cmap=cm, **kw); ax.set_xticks([]); ax.set_yticks([])
            if c_ in (1, 3):
                fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(t_ if r_ == 0 else '', fontsize=9)
        (axs[r_, 0] if len(picks) > 1 else axs[0]).set_ylabel(f"{h_['lat']:.4f}, {h_['lon']:.4f}\n{h_['cover']}, {h_['dh_median']:+.0f} m", fontsize=8)
    fig.suptitle(f'{p.name}: where the DSM and the GSI DEM differ most (600 x 600 m each)', fontsize=10)
    fig.savefig(out / 'fails.png', bbox_inches='tight'); plt.close(fig)

# ---- voxels, offscreen
import open3d as o3d
R = o3d.visualization.rendering


def render(m, center, az, el, dist, path, size=(1800, 1100), fov=30):
    r = R.OffscreenRenderer(*size)
    mat = R.MaterialRecord(); mat.shader = 'defaultLit'
    r.scene.add_geometry('m', m, mat)
    r.scene.set_background([0.93, 0.95, 0.98, 1.0])
    sd = np.array([np.sin(np.radians(sun_az)) * np.cos(np.radians(sun_el)), np.cos(np.radians(sun_az)) * np.cos(np.radians(sun_el)),
                   np.sin(np.radians(sun_el))])
    r.scene.scene.set_sun_light(-sd, [1.0, 0.97, 0.92], 60000)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_indirect_light_intensity(12000)
    eye = np.asarray(center) + dist * np.array([np.sin(np.radians(az)) * np.cos(np.radians(el)),
                                               np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
    r.setup_camera(fov, list(map(float, center)), list(map(float, eye)), [0.0, 0.0, 1.0])
    img = np.asarray(r.render_to_image())
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    del r


vox = p.product / 'voxels.npz'
c_all = columns(vox)
W_m, H_m = c_all['z'].shape[1] * c_all['size'], c_all['z'].shape[0] * c_all['size']
lx, ly = cx - c_all['origin'][0], cy - c_all['origin'][1]                     # the castle in the model's metres
shots = []
c2 = rebin(c_all, 2)
for mode in ('rgb', 'height'):
    m = mesh(c2, color=mode)
    tag = '' if mode == 'rgb' else '_height'
    render(m, (W_m / 2, H_m / 2 + 800, 0), 170, 34, 15500, out / f'voxels_overview{tag}.png')
    shots.append(f'voxels_overview{tag}.png ({len(m.triangles) / 1e6:.1f}M triangles, {c2["size"]:g} m cubes)')
    del m
box = (lx - 2500, ly - 2600, lx + 2500, ly + 1800)
c_city = columns(vox, box=box)
for mode in ('rgb', 'height'):
    m = mesh(c_city, color=mode, top=60)
    tag = '' if mode == 'rgb' else '_height'
    ccx, ccy = lx - c_city['origin'][0] + c_all['origin'][0] - c_all['origin'][0], ly - (c_city['origin'][1] - c_all['origin'][1])
    ccx = lx - (c_city['origin'][0] - c_all['origin'][0])
    render(m, (ccx, ccy - 400, 10), 150, 32, 5200, out / f'voxels_delta{tag}.png')
    render(m, (ccx + 600, ccy + 300, 15), 200, 40, 1700, out / f'voxels_castle{tag}.png')
    shots.append(f'voxels_delta{tag}.png, voxels_castle{tag}.png ({len(m.triangles) / 1e6:.1f}M triangles, {c_city["size"]:g} m cubes)')
    del m
print('voxel renders ->', out, '|', '; '.join(shots))
