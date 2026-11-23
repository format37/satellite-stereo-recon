# satellite-stereo-recon

Satellite stereo scenes with RPC camera models turned into a surface model (DSM) in metres above sea level, checked
against independent elevation models, and into a voxel model. Built on the [NASA Ames Stereo
Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline) (ASP), which reads RPC models directly. First project:
Hiroshima from an ALOS PRISM tri-stereo set of 23 Feb 2008 (2.5 m, forward / nadir / backward), the groundwork for a
model of how a district of the city floods as the sea rises.

- The layout (one YAML per variant, results next to it, `run.sh` with `FROM=`, `progress.sh`) follows
  [fpv-photogrammetry](https://github.com/format37/fpv-photogrammetry) and
  [theta-360-recon](https://github.com/format37/theta-360-recon).
- The scenes are not in the repository and never will be: they come with a licence. [docs/data.md](docs/data.md)
  says where they come from and what the licence allows; nothing made from them (rasters, pictures) is committed either.
- Heights: [docs/datum.md](docs/datum.md). The satellite gives heights above the ellipsoid; the sea at Hiroshima is
  33 m above it.

```
scenes (1B2R + RPC) ─► crops with their own RPC ─► masks: sea (WorldCover), clouds (they float: each view sees
                         (pixel origin fixed)          them over other ground) ─► bundle adjustment (one view fixed)
  ─► stereo per pair (ASP, MGM) ─► DSM per pair ─► fused ─► heights above T.P. (GSI geoid) ─► check against the
     GSI laser DEM and Copernicus GLO-30 ─► voxels (columns of cubes, the sea at 0 m) ─► pictures, summary
```

## What goes where

```
data/<project>/scenes/...            the products as delivered (a link is fine): IMG-*.tif, RPC-*.txt, HDR-*.txt
data/<project>/<variant>.yaml        one variant of the run; `base: other.yaml` starts from another one
data/<project>/out/<variant>/        prep/ ref/ ba/ stereo/ product/ logs/ summary.md
data/<project>/out/ref/              downloads shared by the variants (GSI tiles, geoid grids, land cover)
data/<project>/summary.md            the five lines worth knowing
```

`data/hiroshima/fb.yaml` has every key commented: the scenes, the stereo pairs, the box (`aoi`, lon/lat), the masks,
the ASP settings, the DSM grid and geoid, the references of the check, the voxel size. `fnb.yaml` is the same with
all three views.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && source .venv/bin/activate
# ASP: a release tarball from https://github.com/NeoGeographyToolkit/StereoPipeline/releases (3.7.0 was used), unpacked
export PATH=$PATH:/path/to/StereoPipeline-3.7.0-2026-06-08-x86_64-Linux/bin     # after the venv: ASP brings its own python3
```

## Run

```bash
ln -s /path/to/ALOS_PRISM_RPCs_Hiroshima_Stereo_Triplet data/hiroshima/scenes      # the three unpacked archives
scripts/run.sh data/hiroshima/fb.yaml 2>&1 | tee data/hiroshima/out/fb.log          # everything, ~25 min on 8 cores
FROM=dsm scripts/run.sh data/hiroshima/fb.yaml      # from a later step: prepare reference masks stereo dsm check voxels report
scripts/progress.sh data/hiroshima/fb.yaml          # in a second terminal
python3 -I scripts/compare.py data/hiroshima/fb.yaml data/hiroshima/fnb.yaml        # variants side by side
python3 scripts/voxview.py data/hiroshima/fb.yaml --times 2                         # the voxel model in a window
```

## The steps one by one

Each takes the same YAML and can be run on its own.

| step | script | writes |
|---|---|---|
| crops of every view round the box, each with its RPC (JAXA's pixel origin fixed); view and pair geometry | `prepare.py` | `prep/<view>.tif`, `prep/<view>.jpg`, `product/views.json` |
| GSI DEM tiles, geoid grids, WorldCover, Copernicus DEM on the run's grid | `reference.py` | `ref/*.tif`, `ref/ref.json` |
| water (WorldCover) and clouds (multi-view test) blanked out of the crops | `masks.py` | `prep/<view>_m.tif`, `prep/<view>_mask.jpg`, `product/water.tif` |
| bundle adjustment, stereo per pair, a DSM per pair (ASP) | `run_stereo.sh` | `ba/`, `stereo/<pair>/run-DEM.tif` |
| pairs fused, heights above sea level, holes explained, orthophoto | `dsm.py` | `product/dsm.tif`, `dsm_ellipsoid.tif`, `holes.tif`, `ortho.tif` |
| against the GSI DEM (by land cover, slope, distance to water; co-registration) and Copernicus | `check.py` | `product/check.md`, `check.json`, `check_dh.png` |
| columns of cubes, the sea at 0 m | `voxels.py` (look: `voxview.py`) | `product/voxels.npz` |
| pictures (headless), every number on one page | `render.py`, `summary.py` | `product/renders/`, `summary.md` |

Tools that take paths: `rpc.py RPC-*.txt HDR-*.txt` checks a JAXA RPC against its scene corners; `voxmesh.py` is the
mesh of visible cube faces behind the renders and the viewer.

## Hiroshima in numbers

(filled in from `data/hiroshima/out/*/summary.md` and `compare.md`)

## Requirements

- Linux; 8 cores and 16 GB were plenty for a 12 x 11 km box (ASP took ~2.5 GB per process, 4 processes).
- NASA Ames Stereo Pipeline 3.x on PATH (3.7.0 was used); Python 3.10+ with `requirements.txt`.
- Network on first use: GSI tiles, PROJ geoid grids, WorldCover and Copernicus (read straight from their cloud files).
- The renders use Open3D's offscreen renderer (EGL): no display needed. `voxview.py` opens a window.

## License

MIT for the code. The data has its own terms: [docs/data.md](docs/data.md).
