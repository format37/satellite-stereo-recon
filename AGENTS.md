# Instructions for AI agents

You are helping someone turn satellite stereo scenes with RPC camera models into a DSM above sea level, check it, and
build a voxel model from it, with the tools in this repo. Read README.md first. Rules that prevent the expensive
mistakes:

## Data

1. **Never commit a scene or anything made from it.** The scenes come with a licence (docs/data.md). `.gitignore`
   keeps `data/*/*` out except the YAML files and `summary.md`, and every `*.tif`, `*.npz`, `*.ply` anywhere. Pictures
   of the products (renders, maps) stay in `data/<project>/out/` or `reports/`, both ignored. Check `git status`
   before every commit.
2. **One YAML per variant**, in `data/<project>/`; every script takes it. Results go to `data/<project>/out/<variant>/`.
   A variant can start from another with `base: other.yaml`. Downloads shared by the variants go to
   `data/<project>/out/ref/`. `out/` may be a link to a big disk.

## Geometry

- **JAXA RPCs count pixels from 1.** `scripts/rpc.py` reads them and subtracts 1 (checked on the scene corners:
  `python3 -I scripts/rpc.py RPC-*.txt HDR-*.txt`). GDAL reads the `RPC-*.txt` next to an ALOS image without that
  correction: never run ASP on the delivered `IMG-*.tif`, run it on `prep/<view>_m.tif`, which carries the corrected
  RPC in the TIFF and no georeferencing.
- **Heights.** Everything ASP gives is above the WGS84 ellipsoid. `product/dsm.tif` is above sea level (T.P. through
  GSIGEO2011, `dsm.geoid`); `product/dsm_ellipsoid.tif` is not. The geoid is 33 m above the ellipsoid at Hiroshima:
  mixing the two breaks any flood logic by 33 m. Say which one a number is in. docs/datum.md has the tide levels.
- **Bundle adjustment drifts without an anchor.** Unconstrained, it moved the camera centres by 25-53 m and the points
  of one view by 17 m. `run_stereo.sh` keeps one view fixed (`stereo.ba_fixed`, default N, else the first view).
- **The RPCs alone do not give the level.** The three delivered RPCs of Hiroshima disagree along track by 1.4 px (N
  against F) and 2.4 px (N against B): each pair alone is off by -2.8 m (F-B), +7.1 m (N-F), -12.2 m (N-B) on flat
  ground. Two views cannot see this (it becomes height); three views can, and bundle adjustment makes the pairs agree,
  but at the level of the fixed view (N: +6 m; F: -2.7 m). Only an outside height model sets the level: `anchor.py`
  writes `product/dsm_anchored.tif`, tied to `anchor.reference` (copernicus by default: global, so the method works
  anywhere; gsi for the older Hiroshima variants), and everything after it (ground, voxels, demo) uses it. Report "as
  produced" and "anchored" apart, and say which reference anchored it: anchored to Copernicus, Hiroshima's level is
  0.76 m below the GSI laser DEM; anchored to GSI, the check against GSI is not independent.
- **A DSM is not the ground.** Roofs and canopy stand in it. `ground.py` makes a ground model from the DSM itself
  (a robust opening; under dense built-up land, the open ground seen nearby spread by normalized convolution). Floods
  and high ground use it, never the DSM. Sea walls and river embankments are narrower than the pixels: below about
  5 m (the 95% error) a bathtub on it floods land that the walls protect; `floods.py` shows by how much.
- **Clouds float.** A cloud is over different ground in every view; `masks.py` finds them so. A cloud in one view
  leaves a hole in that view's pairs only: another pair may fill it.

## Running

- ASP on PATH **after** the Python of requirements.txt: ASP ships its own `python3`, which has no `yaml`. The shell
  scripts use `$VIRTUAL_ENV/bin/python` or `$PY` when set.
- Long jobs run detached, log to a file, and are watched with `scripts/progress.sh`. A finished ASP step is not run
  again: delete `ba/` or `stereo/<pair>/` to redo it (after changing masks or stereo options, you must).
- Never edit a shell script while it is running: bash reads it as it goes (it happened here: the run finished its
  loop, then hit a syntax error in the new text). Write a new file and `mv` it over: the running bash keeps the old one.

## Verification

- **Report the check against the GSI DEM, never "it looks right".** `check.py` prints the numbers. The GSI DEM is
  bare ground: over roofs and forest dh is building and canopy height, not error. The accuracy of the DSM is the
  number on **flat open ground**; quote "all land" too, but say what it contains.
- **Before and after co-registration are different claims.** Before: what the RPCs alone give (absolute). After: the
  shape of the DSM once a shift and an offset are removed, which needs ground control (here: the GSI DEM itself).
- The Copernicus check is surface against surface at 30 m: it says whether roofs and canopy are at the right height
  on average, not whether single buildings are.
- AW3D30 is made from PRISM: it is not an independent check of a PRISM DSM.
- The demo's data (`demo.py`) are rasters made from the scenes: they go to `product/demo/` and, for the blog, to its
  `public/media/<name>/`, which the blog's `.gitignore` keeps out of git. Never into a post folder.
- Low bundle-adjustment residuals (0.2 px) say the views agree with each other, not that the heights are right.
