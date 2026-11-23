#!/usr/bin/env bash
# The whole way from satellite scenes to a DSM above sea level, its check against independent elevation models and a
# voxel model, for one variant YAML.
#
#   scripts/run.sh data/hiroshima/fb.yaml 2>&1 | tee data/hiroshima/out/fb.log
#   FROM=dsm scripts/run.sh data/hiroshima/fb.yaml       # start at a later step: prepare reference masks stereo dsm check voxels report
#   scripts/progress.sh data/hiroshima/fb.yaml          # in a second terminal
#
# Every line below is a script that can also be run on its own with the same YAML. Needs the Python of
# requirements.txt (an active venv, or PY=/path/to/python) and the NASA Ames Stereo Pipeline on PATH.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=${1:?usage: run.sh VARIANT.yaml}
PY=${PY:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}; PY=${PY:-python3}; export PY
STEPS=(prepare reference masks stereo dsm check voxels report); FROM=${FROM:-prepare}; go=0
on() { [ "$1" = "$FROM" ] && go=1; [ $go = 1 ]; }
[[ " ${STEPS[*]} " == *" $FROM "* ]] || { echo "FROM must be one of: ${STEPS[*]}"; exit 1; }
t0=$SECONDS; step() { echo "== $(date +%T) $1 ($(( (SECONDS - t0) / 60 )) min so far)"; }

if on prepare;   then step prepare;   "$PY" -I "$T/prepare.py" "$Y"; fi    # 1. crops with their own RPC, view geometry -> prep/, product/views.json
if on reference; then step reference; "$PY" -I "$T/reference.py" "$Y"; fi  # 2. GSI DEM, geoids, land cover, Copernicus on the grid -> ref/
if on masks;     then step masks;     "$PY" -I "$T/masks.py" "$Y"; fi      # 3. sea and clouds -> prep/<view>_m.tif, product/water.tif
if on stereo;    then step stereo;    "$T/run_stereo.sh" "$Y"; fi          # 4. bundle adjustment, stereo, a DSM per pair (ASP) -> ba/, stereo/
if on dsm;       then step dsm;       "$PY" -I "$T/dsm.py" "$Y"; fi        # 5. fused, above sea level, holes, orthophoto -> product/dsm.tif
if on check;     then step check;     "$PY" -I "$T/check.py" "$Y"; fi      # 6. against the GSI DEM and Copernicus -> product/check.md
if on voxels;    then step voxels;    "$PY" -I "$T/voxels.py" "$Y"; fi     # 7. the voxel model -> product/voxels.npz (look: scripts/voxview.py)
if on report;    then step report                                          # 8. pictures, every number of the run on one page
  "$PY" -I "$T/render.py" "$Y"; "$PY" -I "$T/summary.py" "$Y"; fi
step done
