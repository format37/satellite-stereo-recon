#!/usr/bin/env python3
"""Read a project YAML: settings and the paths of the run.

    from project import load;  p = load('data/hiroshima/fb.yaml')
    eval "$(python3 scripts/project.py data/hiroshima/fb.yaml)"     # the same as shell variables

A project is a folder data/<project>/ with the satellite products and one or more YAML files. Every YAML is one
variant of the run and writes into its own folder:
    data/<project>/scenes/...            the products as delivered (here JAXA 1B2R + RPC: IMG-*.tif, RPC-*.txt, HDR-*.txt)
    data/<project>/<variant>.yaml
    data/<project>/out/<variant>/        prep/ ba/ stereo/ product/ logs/ summary.md
    data/<project>/out/ref/              downloads shared by the variants: GSI elevation tiles, land cover, geoid grids
A YAML may start from another one: `base: fb.yaml` takes every key of fb.yaml and the YAML changes what it lists
(sections are merged key by key). data/hiroshima/fb.yaml has every key commented. `stereo: {reuse: <variant>}`
takes prep/, ref/, ba/ and stereo/ of that variant (same box and views): only dsm.py and the steps after it run,
e.g. to try another fusion of the same pairs.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


def merge(a, b):
    """b over a, sections (dicts) key by key"""
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(open(path)) or {}
    if cfg.get('base'):
        cfg = merge(read(path.parent / cfg['base']), {k: v for k, v in cfg.items() if k != 'base'})
    return cfg


def load(path):
    path = Path(path).resolve()
    cfg = read(path)
    proj = path.parent
    run = proj / 'out' / path.stem
    g = lambda k: cfg.get(k) or {}
    masks, stereo, dsm, check, vox = (g(k) for k in ('masks', 'stereo', 'dsm', 'check', 'voxels'))
    anchor, ground, demo, floods, figures = (g(k) for k in ('anchor', 'ground', 'demo', 'floods', 'figures'))
    pairs = [tuple(pr) for pr in cfg['pairs']]
    src = proj / 'out' / stereo['reuse'] if stereo.get('reuse') else run      # stereo.reuse: another variant's prep, ref, ba, stereo
    views = list(dict.fromkeys(v for pr in pairs for v in pr))          # in order of first use: the first is the reference
    p = SimpleNamespace(
        yaml=path, project=proj, name=f'{proj.name}/{path.stem}', variant=path.stem, run=run, cfg=cfg,
        scenes={k: proj / v for k, v in cfg['scenes'].items()}, pairs=pairs, views=views,
        all_views=list(cfg['scenes']),
        prep=src / 'prep', ba=src / 'ba', stereo=src / 'stereo', product=run / 'product', logs=run / 'logs',
        ref=proj / 'out' / 'ref', vref=src / 'ref', reuse=stereo.get('reuse') or '',
        aoi=[float(x) for x in cfg['aoi']], crs=cfg.get('crs', 'EPSG:32653'),
        heights=[float(x) for x in cfg.get('heights', [-50, 1000])], crop_margin=int(cfg.get('crop_margin', 64)),
        # masks: water from a land-cover map, clouds from the views themselves
        water=masks.get('water', 'worldcover'), water_shrink_m=float(masks.get('water_shrink_m', 10)),
        clouds=bool(masks.get('clouds', True)), cloud_dn=masks.get('cloud_dn', 'auto'),
        cloud_min_m2=float(masks.get('cloud_min_m2', 2000)), cloud_grow_m=float(masks.get('cloud_grow_m', 60)),
        cloud_views=list(masks.get('cloud_views', list(cfg['scenes']))),
        terrain=masks.get('terrain', 'copernicus'),   # the ground the cloud test and the hole map look from: copernicus | gsi
        # stereo (NASA Ames Stereo Pipeline)
        bundle_adjust=bool(stereo.get('bundle_adjust', True)), algorithm=stereo.get('algorithm', 'asp_mgm'),
        subpixel=int(stereo.get('subpixel', 9)), stereo_options=list(stereo.get('options') or []),
        ba_options=list(stereo.get('ba_options') or []),
        # the view whose RPC stays as delivered in bundle adjustment (the frame of the result): N when it is used
        ba_fixed=stereo.get('ba_fixed') or ('N' if 'N' in views else views[0]),
        mapproject=stereo.get('mapproject') or '',    # another variant: mapproject the views onto its smoothed DSM first
        # DSM
        res=float(dsm.get('res', 2.5)), geoid=dsm.get('geoid', 'egm2008'), fuse=dsm.get('fuse', 'median'),
        max_tri_error=float(dsm.get('max_triangulation_error', 0)), point2dem_options=list(dsm.get('point2dem_options') or []),
        # the level: tie the DSM to an outside height model (anchor.py)
        anchor=anchor.get('reference', 'copernicus'), anchor_open_margin_m=float(anchor.get('open_margin_m', 30)),
        # the ground under roofs and canopy (ground.py)
        ground_window_m=float(ground.get('window_m', 210)), ground_res=float(ground.get('res', 10)),
        ground_percentiles=[float(x) for x in ground.get('percentiles', [10, 90])],
        ground_interp_sigma_m=float(ground.get('interp_sigma_m', 300)), ground_built_up_window_m=float(ground.get('built_up_window_m', 620)),
        ground_built_up_min=float(ground.get('built_up_min', 0.5)), ground_flat_max_deg=float(ground.get('flat_max_deg', 5)),
        ground_hill_relief_m=float(ground.get('hill_relief_m', 15)), ground_canopy_m=float(ground.get('canopy_m', 0)),
        # check against an independent elevation model (GSI layers: Japan only; empty = no check)
        reference=list(check.get('reference', [])),
        dsm_reference=check.get('dsm_reference', 'copernicus'),
        floods=floods,                                # floods.py: flood extents and high ground against the check reference
        demo=demo,                                    # demo.py: the data of the interactive demo
        figures=figures,                              # figures.py: the pictures of the write-up
        # voxels
        voxel_size=float(vox.get('size', 5)), voxel_base=float(vox.get('base', -5)), voxel_fill_m=float(vox.get('fill_m', 50)),
    )
    return p


def pair_name(pr):
    return ''.join(pr)


if __name__ == '__main__':
    p = load(sys.argv[1])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from geo import Grid
    g = Grid(p)
    b = g.bounds
    print(f'GRID_BOUNDS="{" ".join(f"{x:.3f}" for x in b)}"; GRID_BOUNDS_PAD="{b[0] - 200:.3f} {b[1] - 200:.3f} {b[2] + 200:.3f} {b[3] + 200:.3f}"; '
          f'FIXED={p.views.index(p.ba_fixed)}; MAPPROJECT="{p.mapproject}"; ', end='')
    print(f'RUN="{p.run}"; PREP="{p.prep}"; BA="{p.ba}"; STEREO="{p.stereo}"; PRODUCT="{p.product}"; LOGS="{p.logs}"; '
          f'REF="{p.ref}"; VIEWS="{" ".join(p.views)}"; PAIRS="{" ".join(pair_name(pr) for pr in p.pairs)}"; '
          f'ALGORITHM={p.algorithm}; SUBPIXEL={p.subpixel}; BUNDLE_ADJUST={int(p.bundle_adjust)}; RES={p.res}; CRS="{p.crs}"; '
          f'STEREO_OPTIONS="{" ".join(p.stereo_options)}"; BA_OPTIONS="{" ".join(p.ba_options)}"; '
          f'POINT2DEM_OPTIONS="{" ".join(p.point2dem_options)}"; MAX_TRI_ERROR={p.max_tri_error}')
