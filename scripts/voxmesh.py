#!/usr/bin/env python3
"""The voxel model as a height field and as a mesh of the faces that can be seen (for the viewer and the renders).

    from voxmesh import columns, rebin, mesh
    c = columns('data/hiroshima/out/fb/product/voxels.npz')     # per cell: top (m above sea level), colour, source
    c = rebin(c, 2)                                             # cubes twice as large (towers kept: max height)
    m = mesh(c, color='rgb')                                    # open3d TriangleMesh, local metres (x east, y north)

A cube model of a city is mostly hidden faces: of a column only the top and the parts of its sides above the
neighbours can be seen. The mesh has one quad per top and one per visible side of a column (as tall as the step),
so a 2400 x 2200 model at 5 m is ~15M triangles instead of the 12 per cube of a voxel grid.
"""
import numpy as np

SOURCE_RGB = {0: (175, 175, 175), 1: (235, 150, 60), 2: (60, 120, 200), 3: (120, 120, 120)}


def columns(path, box=None):
    """voxels.npz -> dict: z (top face, m above sea level; nan = no column), rgb, source, views; size, origin.
    box = (x0, y0, x1, y1) in local metres from the model's corner: keep that part only."""
    Z = np.load(path)
    ijk = Z['ijk'].astype(np.int64)
    W, Hh = int(Z['dims'][0]), int(Z['dims'][1])
    s = float(Z['size']); o = np.asarray(Z['origin'], float)
    key = ijk[:, 1] * W + ijk[:, 0]
    order = np.lexsort((ijk[:, 2], key))
    last = np.r_[key[order][1:] != key[order][:-1], True]
    sel = order[last]                                                   # the top voxel of every column
    z = np.full(Hh * W, np.nan, np.float32); z[key[sel]] = o[2] + (ijk[sel, 2] + 1) * s
    rgb = np.zeros((Hh * W, 3), np.uint8); rgb[key[sel]] = Z['rgb'][sel]
    src = np.full(Hh * W, 255, np.uint8); src[key[sel]] = Z['source'][sel]
    vw = np.zeros(Hh * W, np.uint8); vw[key[sel]] = Z['views'][sel]
    c = dict(z=z.reshape(Hh, W), rgb=rgb.reshape(Hh, W, 3), source=src.reshape(Hh, W), views=vw.reshape(Hh, W),
             size=s, base=float(o[2]), origin=o[:2].copy(), crs=str(Z['crs']) if 'crs' in Z else '')
    if box is not None:
        x0, y0, x1, y1 = box
        i0, i1 = max(0, int(x0 // s)), min(W, int(np.ceil(x1 / s)))
        j0, j1 = max(0, int(y0 // s)), min(Hh, int(np.ceil(y1 / s)))
        for k in ('z', 'rgb', 'source', 'views'):
            c[k] = c[k][j0:j1, i0:i1]
        c['origin'] = c['origin'] + [i0 * s, j0 * s]
    return c


def rebin(c, t):
    """cubes t times larger: the highest top of the t x t cells (towers stay), their mean colour"""
    if t <= 1:
        return c
    z = c['z']; Hh, W = z.shape[0] // t * t, z.shape[1] // t * t
    b = lambda a: a[:Hh, :W].reshape(Hh // t, t, W // t, t, *a.shape[2:]).swapaxes(1, 2).reshape(Hh // t, W // t, t * t, *a.shape[2:])
    zb = b(z)
    with np.errstate(all='ignore'):
        zt = np.nanmax(zb, axis=2)
        rgb = np.nanmean(np.where(np.isfinite(zb)[..., None], b(c['rgb']).astype(np.float32), np.nan), axis=2)
    src = b(c['source']).min(axis=2)
    s2 = c['size'] * t
    base2 = np.floor(c['base'] / s2) * s2                              # sea level stays on a face of the grid
    zt = base2 + np.maximum(1, np.round((zt - base2) / s2)) * s2
    return dict(z=zt.astype(np.float32), rgb=np.nan_to_num(rgb).astype(np.uint8), source=src, views=b(c['views']).max(axis=2),
                size=s2, base=base2, origin=c['origin'].copy(), crs=c['crs'])


def colours(c, mode='rgb', top=None):
    z = c['z']
    if mode == 'rgb':
        return c['rgb'].astype(np.float32) / 255
    if mode == 'source':
        out = np.zeros(z.shape + (3,), np.float32)
        for k, v in SOURCE_RGB.items():
            out[c['source'] == k] = np.array(v) / 255
        return out
    if mode == 'views':
        t = np.clip(c['views'] / 3.0, 0, 1)[..., None]
        return (1 - t) * np.array([0.9, 0.3, 0.2]) + t * np.array([0.2, 0.7, 0.3])
    # height: a terrain ramp from sea level, the sea blue
    lo, hi = 0.0, float(top or np.nanpercentile(z, 99.5))
    t = np.clip((np.nan_to_num(z) - lo) / max(hi - lo, 1), 0, 1)
    stops = np.array([[0.20, 0.55, 0.35], [0.55, 0.75, 0.40], [0.90, 0.85, 0.55], [0.75, 0.55, 0.35], [0.95, 0.95, 0.95]])
    x = t * (len(stops) - 1); k = np.minimum(x.astype(int), len(stops) - 2); u = (x - k)[..., None]
    out = (1 - u) * stops[k] + u * stops[k + 1]
    out[c['source'] == 2] = np.array(SOURCE_RGB[2]) / 255
    return out.astype(np.float32)


def mesh(c, color='rgb', side_shade=0.8, top=None):
    """open3d TriangleMesh of the visible faces; vertices in metres from the model's corner (x east, y north, z up)"""
    import open3d as o3d
    z = c['z']; s = c['size']; Hh, W = z.shape
    col = colours(c, color, top)
    jj, ii = np.nonzero(np.isfinite(z))
    zt = z[jj, ii]
    # the model's rows run north to south in the arrays of columns()? no: j = 0 is the south edge there
    V, F, C, Nn = [], [], [], []
    def quads(p0, p1, p2, p3, rgb, normal):
        n = len(p0)
        if n == 0:
            return
        base = sum(len(v) for v in V)
        V.append(np.stack([p0, p1, p2, p3], 1).reshape(-1, 3))
        idx = base + np.arange(n)[:, None] * 4
        F.append(np.concatenate([idx + [0, 1, 2], idx + [0, 2, 3]]))
        C.append(np.repeat(rgb, 4, axis=0))
        Nn.append(np.repeat(np.tile(normal, (n, 1)), 4, axis=0))
    x0, x1, y0, y1 = ii * s, (ii + 1) * s, jj * s, (jj + 1) * s
    quads(np.c_[x0, y0, zt], np.c_[x1, y0, zt], np.c_[x1, y1, zt], np.c_[x0, y1, zt], col[jj, ii], [0, 0, 1.0])
    pad = np.pad(np.nan_to_num(z, nan=-1e9), 1, constant_values=-1e9)
    floor = c['base']
    for dj, di, nrm in ((0, 1, [1, 0, 0]), (0, -1, [-1, 0, 0]), (1, 0, [0, 1, 0]), (-1, 0, [0, -1, 0])):
        zn = np.maximum(pad[1 + jj + dj, 1 + ii + di], floor)
        m = zt > zn + 1e-3
        a, b_, lo, hi = ii[m], jj[m], zn[m], zt[m]
        rgb = col[b_, a] * side_shade
        if di == 1:
            X = (a + 1) * s; quads(np.c_[X, b_ * s, lo], np.c_[X, (b_ + 1) * s, lo], np.c_[X, (b_ + 1) * s, hi], np.c_[X, b_ * s, hi], rgb, nrm)
        elif di == -1:
            X = a * s; quads(np.c_[X, (b_ + 1) * s, lo], np.c_[X, b_ * s, lo], np.c_[X, b_ * s, hi], np.c_[X, (b_ + 1) * s, hi], rgb, nrm)
        elif dj == 1:
            Y = (b_ + 1) * s; quads(np.c_[(a + 1) * s, Y, lo], np.c_[a * s, Y, lo], np.c_[a * s, Y, hi], np.c_[(a + 1) * s, Y, hi], rgb, nrm)
        else:
            Y = b_ * s; quads(np.c_[a * s, Y, lo], np.c_[(a + 1) * s, Y, lo], np.c_[(a + 1) * s, Y, hi], np.c_[a * s, Y, hi], rgb, nrm)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.concatenate(V).astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.concatenate(F).astype(np.int32))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(C), 0, 1).astype(np.float64))
    m.vertex_normals = o3d.utility.Vector3dVector(np.concatenate(Nn).astype(np.float64))
    return m
