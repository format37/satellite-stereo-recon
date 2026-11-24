#!/usr/bin/env bash
# The whole way from satellite scenes to a DSM above sea level tied to a reference height model, the ground under it,
# its check against independent elevation models, a voxel model and the data of the interactive demo, for one
# variant YAML.
#
#   scripts/run.sh data/hiroshima/fnb_global.yaml 2>&1 | tee data/hiroshima/out/fnb_global.log
#   FROM=dsm scripts/run.sh data/hiroshima/fb.yaml      # start at a later step (the list is STEPS below)
#   TO=stereo scripts/run.sh data/hiroshima/fb.yaml     # stop after a step
#   scripts/progress.sh data/hiroshima/fb.yaml          # in a second terminal
#
# Every line below is a script that can also be run on its own with the same YAML. Needs the Python of
# requirements.txt (an active venv, or PY=/path/to/python) and the NASA Ames Stereo Pipeline on PATH. Steps whose
# section is missing from the YAML (check.reference, floods, demo) say so and do nothing.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=${1:?usage: run.sh VARIANT.yaml}
PY=${PY:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}; PY=${PY:-python3}; export PY
STEPS=(prepare reference masks stereo dsm anchor ground check floods voxels demo report)
FROM=${FROM:-prepare}; TO=${TO:-report}; go=0; stop=0
on() { [ $stop = 1 ] && return 1; [ "$1" = "$FROM" ] && go=1; [ "$1" = "$TO" ] && stop=1; [ $go = 1 ]; }
for s in "$FROM" "$TO"; do [[ " ${STEPS[*]} " == *" $s "* ]] || { echo "FROM and TO must be among: ${STEPS[*]}"; exit 1; }; done
t0=$SECONDS; step() { echo "== $(date +%T) $1 ($(( (SECONDS - t0) / 60 )) min so far)"; }

if on prepare;   then step prepare;   "$PY" -I "$T/prepare.py" "$Y"; fi    # 1. crops with their own RPC, view geometry -> prep/, product/views.json
if on reference; then step reference; "$PY" -I "$T/reference.py" "$Y"; fi  # 2. geoids, land cover, Copernicus (and the GSI DEM) on the grid -> ref/
if on masks;     then step masks;     "$PY" -I "$T/masks.py" "$Y"; fi      # 3. sea and clouds -> prep/<view>_m.tif, product/water.tif
if on stereo;    then step stereo;    "$T/run_stereo.sh" "$Y"; fi          # 4. bundle adjustment, stereo, a DSM per pair (ASP) -> ba/, stereo/
if on dsm;       then step dsm;       "$PY" -I "$T/dsm.py" "$Y"; fi        # 5. fused, above the geoid, holes, orthophoto -> product/dsm.tif
if on anchor;    then step anchor;    "$PY" -I "$T/anchor.py" "$Y"; fi     # 6. the level from a reference model -> product/dsm_anchored.tif
if on ground;    then step ground;    "$PY" -I "$T/ground.py" "$Y"; fi     # 7. the ground under roofs and canopy -> product/ground.tif
if on check;     then step check;     "$PY" -I "$T/check.py" "$Y"; fi      # 8. against the GSI DEM and Copernicus -> product/check.md
if on floods;    then step floods;    "$PY" -I "$T/floods.py" "$Y"; fi     # 9. flood extents, high ground against the GSI DEM -> product/floods.md
if on voxels;    then step voxels;    "$PY" -I "$T/voxels.py" "$Y"; fi     # 10. the voxel model -> product/voxels.npz (look: scripts/voxview.py)
if on demo;      then step demo;      "$PY" -I "$T/demo.py" "$Y"; fi       # 11. the data of the interactive demo -> product/demo/
if on report;    then step report                                          # 12. pictures, every number of the run on one page
  "$PY" -I "$T/render.py" "$Y"; "$PY" -I "$T/summary.py" "$Y"; fi
step done
