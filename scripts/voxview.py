#!/usr/bin/env python3
"""voxview - the voxel model in a window (Open3D). After voxview.py of theta-360-recon, for a height field.

    python3 scripts/voxview.py data/hiroshima/fb.yaml                 # the run's product/voxels.npz
    python3 scripts/voxview.py path/to/voxels.npz --times 2           # start with cubes twice as large
    python3 scripts/voxview.py data/hiroshima/fb.yaml --box 2000 2000 6000 5000   # a part: x0 y0 x1 y1, m from the SW corner

The model is in metres: x east, y north from its south-west corner, z above sea level (T.P.). The whole box at
5 m is ~15M triangles; start with --times 2 on a small GPU.

Keys:  [ / ]  finer / coarser cubes (whole multiples of the model's size)
       C      colour: orthophoto -> height above sea level -> source (grey measured, orange filled, blue water)
              -> pairs that measured it (red 1 .. green 3)
       W / S  raise / lower a flat water level by 1 m (every top below it turns blue; no flood fill: a preview)
       B      background dark/light     L  lighting on/off     R  reset view     H  help     Q/Esc  quit
Mouse: left-drag rotate, right/ctrl-drag pan, wheel zoom.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxmesh import columns, mesh, rebin

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('input', help='variant YAML or a voxels.npz')
ap.add_argument('--times', type=int, default=1, help='start with cubes this many times the model\'s size')
ap.add_argument('--box', type=float, nargs=4, help='x0 y0 x1 y1: only this part, metres from the south-west corner')
ap.add_argument('--color', choices=['rgb', 'height', 'source', 'views'], default='rgb')
ap.add_argument('--size', default='1600x1000')
a = ap.parse_args()
if a.input.endswith(('.yaml', '.yml')):
    from project import load
    src = load(a.input).product / 'voxels.npz'
else:
    src = Path(a.input)
if not src.exists():
    raise SystemExit(f'{src} not found: run scripts/voxels.py first')
base = columns(src, box=a.box)
print(f'{src}: {np.isfinite(base["z"]).sum():,} columns of {base["size"]:g} m, '
      f'{base["z"].shape[1] * base["size"] / 1e3:.1f} x {base["z"].shape[0] * base["size"] / 1e3:.1f} km, '
      f'tops {np.nanmin(base["z"]):.0f} .. {np.nanmax(base["z"]):.0f} m above sea level')
modes = ['rgb', 'height', 'source', 'views']
st = dict(times=a.times, mode=modes.index(a.color), dark=False, light=True, level=None)


def build():
    c = rebin(base, st['times'])
    m = mesh(c, color=modes[st['mode']])
    if st['level'] is not None:
        col = np.asarray(m.vertex_colors); v = np.asarray(m.vertices)
        under = v[:, 2] <= st['level'] + 1e-3
        col[under] = 0.3 * col[under] + 0.7 * np.array([0.24, 0.47, 0.78])
        m.vertex_colors = o3d.utility.Vector3dVector(col)
    print(f'cubes of {c["size"]:g} m, {len(m.triangles) / 1e6:.1f}M triangles, colour: {modes[st["mode"]]}'
          + (f', water level {st["level"]:+.0f} m' if st['level'] is not None else ''))
    return m


W_, H_ = map(int, a.size.split('x'))
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window('voxview', W_, H_)
geom = build()
vis.add_geometry(geom)


def style():
    o = vis.get_render_option()
    o.background_color = np.array([0.08, 0.08, 0.1]) if st['dark'] else np.array([0.93, 0.95, 0.98])
    o.light_on = st['light']
    o.mesh_show_back_face = True


def home(v=vis):
    ctr = v.get_view_control()
    ctr.set_front([-0.25, -0.75, 0.6]); ctr.set_up([0, 0, 1])
    z = base['z']; ctr.set_lookat([z.shape[1] * base['size'] / 2, z.shape[0] * base['size'] / 2, 0]); ctr.set_zoom(0.55)
    return True


def rebuild(v):
    global geom
    cam = v.get_view_control().convert_to_pinhole_camera_parameters()
    v.remove_geometry(geom, reset_bounding_box=False)
    geom = build()
    v.add_geometry(geom, reset_bounding_box=False)
    v.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    return True


def finer(v): st['times'] = max(1, st['times'] - 1); return rebuild(v)
def coarser(v): st['times'] += 1; return rebuild(v)
def cmode(v): st['mode'] = (st['mode'] + 1) % len(modes); return rebuild(v)
def up(v): st['level'] = (0 if st['level'] is None else st['level'] + 1); return rebuild(v)
def down(v): st['level'] = (None if st['level'] in (None, 0) else st['level'] - 1); return rebuild(v)
def bg(v): st['dark'] = not st['dark']; style(); return True
def light(v): st['light'] = not st['light']; style(); return True
def helpk(v): print(__doc__.split('Keys:')[1]); return False


for k, fn in ((ord('['), finer), (ord(']'), coarser), (ord('C'), cmode), (ord('W'), up), (ord('S'), down), (ord('B'), bg),
              (ord('L'), light), (ord('R'), home), (ord('H'), helpk)):
    vis.register_key_callback(k, fn)
style(); home()
vis.run()
vis.destroy_window()
