# satellite-stereo-recon

Satellite stereo scenes with RPC camera models turned into a surface model (DSM) in metres above sea level, checked
against independent elevation models, and into a voxel model. Built on the [NASA Ames Stereo
Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline) (ASP), which reads RPC models directly. First project:
Hiroshima from an ALOS PRISM tri-stereo set of 23 Feb 2008 (2.5 m, forward / nadir / backward), and an interactive
demo built on it: the sea rising round the city, and the nearest high ground for any point. Every input besides the
images is global (WorldCover, Copernicus GLO-30, EGM2008), so the same run works for any place with RPC stereo; the
GSI laser DEM of Japan only checks the result.

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
  ─► stereo per pair (ASP, MGM) ─► DSM per pair ─► fused ─► heights above the geoid ─► the level from Copernicus
     GLO-30 (anchor) ─► the ground under roofs and canopy ─► check against the GSI laser DEM: heights, floods, high
     ground ─► voxels ─► the demo's data (columns of 5 m in 0.5 m steps, the ground, colours) ─► pictures, summary
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
the ASP settings, the DSM grid and geoid, the references of the check, the voxel size. The other variants start from
it: `fnb.yaml` all three views (three pairs, median), `fnb_w.yaml` the same pairs weighted by (B/H)^2 (it reuses
fnb's stereo: `stereo: {reuse: fnb}`), `fb_s3.yaml` subpixel mode 3, `fb_map.yaml` stereo on views mapprojected onto
fb's smoothed DSM (`stereo: {mapproject: fb}`). `fnb_w` came out best (the numbers below). `fnb_global.yaml` is
`fnb_w` with only global inputs (EGM2008 heights, the level from Copernicus GLO-30, the cloud test over Copernicus):
the variant of the demo. The older variants keep the Japanese geoid and the GSI DEM as their anchor.

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
scripts/run.sh data/hiroshima/fnb.yaml 2>&1 | tee data/hiroshima/out/fnb.log        # three pairs: 30 min of stereo
FROM=dsm scripts/run.sh data/hiroshima/fnb_w.yaml   # from a later step: prepare reference masks stereo dsm anchor ground check floods voxels demo report
TO=stereo scripts/run.sh data/hiroshima/fb.yaml     # stop after a step
python3 -I scripts/demo.py data/hiroshima/fnb_global.yaml ../blog-site/public/media/hiroshima-sea   # the demo's data, copied for the blog
scripts/progress.sh data/hiroshima/fnb.yaml         # in a second terminal
python3 -I scripts/compare.py data/hiroshima/fb.yaml data/hiroshima/fnb_w.yaml      # variants side by side
python3 scripts/voxview.py data/hiroshima/fnb_w.yaml --times 2                      # the voxel model in a window
```

## The steps one by one

Each takes the same YAML and can be run on its own.

| step | script | writes |
|---|---|---|
| crops of every view round the box, each with its RPC (JAXA's pixel origin fixed); view and pair geometry | `prepare.py` | `prep/<view>.tif`, `prep/<view>.jpg`, `product/views.json` |
| GSI DEM tiles, geoid grids, WorldCover, Copernicus DEM on the run's grid | `reference.py` | `ref/*.tif`, `ref/ref.json` |
| water (WorldCover) and clouds (multi-view test) blanked out of the crops | `masks.py` | `prep/<view>_m.tif`, `prep/<view>_mask.jpg`, `product/water.tif` |
| bundle adjustment (one view fixed), stereo per pair, a DSM per pair (ASP); optionally on views mapprojected onto another variant's smoothed DSM | `run_stereo.sh` (`seed_dem.py`) | `ba/`, `stereo/<pair>/run-DEM.tif` |
| pairs fused, heights above sea level, holes explained, orthophoto | `dsm.py` | `product/dsm.tif`, `dsm_ellipsoid.tif`, `holes.tif`, `ortho.tif` |
| the level and position from a reference height model (`anchor.reference`: copernicus, gsi, none), Nuth & Kaab | `anchor.py` (`coreg.py`) | `product/dsm_anchored.tif`, `anchor.json` |
| the ground under roofs and canopy: a robust opening; under dense built-up land, the open ground seen nearby spread by normalized convolution | `ground.py` | `product/ground.tif`, `ground.json` |
| against the GSI DEM (by land cover, slope, curvature, distance to water, pair; co-registration; the anchored DSM; low-lying land) and Copernicus (Japan only: needs `check.reference`) | `check.py` | `product/check.md`, `check.json`, `check_dh.png` |
| the demo's two questions against the GSI DEM: where the sea goes by level (a bathtub), the nearest high ground | `floods.py` | `product/floods.md`, `floods.json`, `floods_map.png` |
| columns of cubes from the anchored DSM, the sea at 0 m | `voxels.py` (look: `voxview.py`) | `product/voxels.npz` |
| the demo's data: surface and ground in 5 m cells and 0.5 m steps, colours, georeference, accuracy, the reference's flood curve | `demo.py` | `product/demo/` (copied where asked) |
| pictures (headless), every number on one page | `render.py`, `summary.py` | `product/renders/`, `summary.md` |

Tools that take paths: `rpc.py RPC-*.txt HDR-*.txt` checks a JAXA RPC against its scene corners; `voxmesh.py` is the
mesh of visible cube faces behind the renders and the viewer.

## Hiroshima in numbers

`data/hiroshima/summary.md` has the five lines. The box: 12.2 x 10.8 km over the Ota delta, the port and the hills
either side; 31% of it is sea and tidal rivers. Every number is against the GSI DEM5A (airborne laser, bare ground,
5 m), on land, heights above T.P.; "flat open ground" is grass, fields and bare land with slopes under 10 deg, where
ground meets ground. "Anchored" means after a shift and an offset fitted to the GSI DEM (Nuth & Kaab): the shape of
the DSM, not its level.

| variant | what | land measured | flat open ground, as produced: median / RMS / NMAD | anchored: RMS / NMAD / 95% | all land, as produced: median / RMS |
|---|---|---|---|---|---|
| `fb` | forward + backward, MGM | 89.4% | -2.70 / 4.01 / 1.42 m | 2.89 / 1.41 / 4.78 m | -0.67 / 6.96 m |
| `fnb` | the three pairs, median | 99.2% | +6.11 / 6.78 / 1.50 m | 2.92 / 1.50 / 5.18 m | +8.38 / 11.11 m |
| `fnb_w` | the three pairs, weight (B/H)^2 | 99.2% | +6.11 / 6.77 / 1.46 m | 2.91 / 1.46 / 5.11 m | +8.35 / 11.16 m |
| `fb_s3` | as fb, subpixel mode 3 (3x slower) | 92.5% | -2.61 / 4.51 / 1.77 m | 3.61 / 1.77 / 6.52 m | +0.02 / 7.13 m |
| `fb_map` | as fb, on the views mapprojected onto fb's own DSM, smoothed (ASP's advice for slopes) | 93.8% | -2.89 / 4.24 / 1.50 m | 3.08 / 1.49 / 5.11 m | -0.70 / 8.12 m |
| `fnb_global` | as fnb_w, only global inputs; anchored to Copernicus GLO-30, not GSI | 99.2% | +6.45 / 7.04 / 1.46 m | 3.02 / 1.46 / 5.15 m, level -0.76 m | +8.59 / 11.36 m |

The demo's questions (`fnb_global`, `floods.md`; 5 m cells, levels above today's mean sea level, a bathtub from the
sea; the GSI laser DEM as the reference; its ground model sends 99% of the people on land below 3.44 m to real
ground of 10 m or more, the nearest piece 48 m off GSI's in the median):

| sea level | GSI DEM (ground) | DSM, anchored (roofs) | ground model (IoU with GSI) |
|---|---|---|---|
| 1 m | 0.2 km² | 2.0 km² | 18.8 km² (0.00) |
| 2 m | 10.1 km² | 8.2 km² | 35.2 km² (0.24) |
| 3.44 m (Nankai Trough tsunami, Hiroshima City's maximum) | 42.1 km² | 21.6 km² | 48.9 km² (0.80) |
| 5 m | 49.4 km² | 32.7 km² | 52.2 km² (0.90) |
| 10 m | 54.2 km² | 48.0 km² | 56.7 km² (0.94) |
| 20 m | 57.8 km² | 56.0 km² | 60.0 km² (0.95) |

What the runs showed:

| question | answer |
|---|---|
| Does the vertical datum matter? | By 33 m: the geoid (GSIGEO2011) is 32.9-33.5 m above the ellipsoid over the box. EGM2008 is 0.19 m below it, JPGEO2024 (the 2025 datum) 0.16 m above. |
| Do the RPCs give the level? | No. Each pair with the delivered RPCs: F-B -2.75 m, N-F +7.05 m, N-B -12.22 m on flat ground: the views disagree along track by 1.4 px (N-F) and 2.4 px (N-B). Bundle adjustment of three views makes the pairs agree (to 0.6 m) at the level of the view kept fixed: +6.1 m with N, -2.7 m with F alone. Ground control is needed; `dsm_anchored.tif` uses the GSI DEM. |
| What does the nadir view add? | Completeness: 89.4% -> 99.2% of the land, the cloud hole of the forward view filled, the ground behind tall buildings. Not precision: the nadir pairs have half the base (NMAD 2.3-2.6 m alone); weighted fusion keeps the F-B precision. |
| Where does it fail? | Forested hills: NMAD 1.7 m on slopes under 5 deg, 4.0 at 5-15, 7.0 at 15-30, 8.5 above 30. Not the sun (all aspects alike) but the shape: ridges come out 5 m low, valleys 14 m high (canopy and MGM smoothing); a narrow ridge (Motoujina) 30 m low. |
| Can it run without Japanese data? | Yes: `fnb_global` uses only global inputs. Anchored to Copernicus GLO-30, its level is 0.76 m below the GSI laser DEM on flat open ground; the spread is the same (NMAD 1.46 m, 95% within 5.2 m). The images give the shape, one outside model gives the level. |
| And the ground under the roofs? | `ground.py`: on flat built-up land +0.3 m off GSI in the median (anchored to GSI). From 5 m up the flood matches GSI's with IoU 0.90-0.95; at 3.44 m 0.80. Below about 3 m it floods far more than GSI (1 m: 19 km² against 0.2): sea walls and river embankments are narrower than 2.5 m pixels, and the level is 0.76 m low. The demo shows GSI's number beside its own. |
| Is it a ground model? | No: roofs and canopy. GSI has 10.7 km² of the box's land below 1 m T.P. and 21 km² below 2 m (spring high water is +1.9 m: the inner delta lies behind sea walls); the anchored DSM has 1.9 and 5.6 km² there (IoU 0.09 and 0.18). |
| And as a surface model? | Against Copernicus GLO-30 (TanDEM-X, 30 m), anchored: built-up median +1.05 m, RMS 3.5; open +0.58 m, RMS 2.9; trees -4.3 m, RMS 10.6. |
| Do the usual quality settings help? | No. ASP's subpixel mode 3 instead of MGM's own: NMAD 1.77 m on flat ground instead of 1.41, worse slopes, 3x the time. Stereo on views mapprojected onto fb's own smoothed DSM: 4% more land, but NMAD 1.49 m and slopes worse (7.9 m at 15-30 deg): MGM leans on the smooth seed where the forest has little texture. |
| Clouds? | Five small cumulus in the box's north: masked in each view by the multi-view test (173 / 64 / 104 ha in F / N / B); F-B lost 1.1% of the land to them, the three pairs 0.16%. |

Pitfalls found on the way: GDAL reads JAXA's `RPC-*.txt` without the 1-pixel origin shift; bundle adjustment
without a fixed view drifts (camera centres 25-53 m, the points of one view 17 m); ASP's `python3` shadows the venv
when ASP comes first on PATH; ASP's subpixel mode 3 was worse than MGM's own here.

## Requirements

- Linux; 8 cores and 16 GB were plenty for a 12 x 11 km box (ASP took ~2.5 GB per process, 4 processes).
- NASA Ames Stereo Pipeline 3.x on PATH (3.7.0 was used); Python 3.10+ with `requirements.txt`.
- Network on first use: GSI tiles, PROJ geoid grids, WorldCover and Copernicus (read straight from their cloud files).
- The renders use Open3D's offscreen renderer (EGL): no display needed. `voxview.py` opens a window.

## License

MIT for the code. The data has its own terms: [docs/data.md](docs/data.md).
