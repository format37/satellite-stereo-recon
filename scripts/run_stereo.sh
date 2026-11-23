#!/usr/bin/env bash
# Step 4. Stereo with the NASA Ames Stereo Pipeline (ASP): bundle adjustment of the views, then a point cloud and a
# DSM (heights above the ellipsoid) per pair.
#
#   scripts/run_stereo.sh data/hiroshima/fb.yaml
#
# Needs ASP 3.x on PATH (bundle_adjust, parallel_stereo, point2dem); developed with 3.7.0. Reads prep/<view>_m.tif
# (masks.py: the crops with their RPC, water and clouds set to 0 = no data).
# Writes ba/run-*             the adjustments, one rotation + translation per view; the view stereo.ba_fixed (N when it
#                             is used, else the first) stays as delivered and gives the frame
#        stereo/<pair>/run-*  ASP's files: run-PC.tif the point cloud, run-DEM.tif the pair's DSM on the run's grid
#                             (ellipsoidal heights), run-IntersectionErr.tif the gap between the two rays per cell
# stereo.mapproject: <variant> first mapprojects the views onto a smooth DSM of that variant (seed_dem.py) and runs
# stereo on those (ASP's advice for steep terrain); the result is still measured by the RPCs of the views alone.
# A finished step is not run again: delete ba/ or stereo/<pair>/ to redo it.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=${1:?usage: run_stereo.sh VARIANT.yaml}
PY=${PY:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}; PY=${PY:-python3}   # the Python with this repo's requirements (ASP brings its own python3)
eval "$("$PY" -I "$T/project.py" "$Y")"
command -v parallel_stereo >/dev/null || { echo "ASP is not on PATH: export PATH=/path/to/StereoPipeline/bin:\$PATH"; exit 1; }
mkdir -p "$BA" "$STEREO" "$LOGS"
imgs=(); for v in $VIEWS; do imgs+=("$PREP/${v}_m.tif"); done
BAP=()
if [ "$BUNDLE_ADJUST" = 1 ]; then
  if [ ! -f "$BA/run-final_residuals_stats.txt" ]; then
    echo "$(date +%T) bundle_adjust $VIEWS (fixed: index $FIXED)"
    bundle_adjust -t rpc "${imgs[@]}" -o "$BA/run" --fixed-camera-indices "$FIXED" $BA_OPTIONS > "$LOGS/bundle_adjust.log" 2>&1
  fi
  BAP=(--bundle-adjust-prefix "$BA/run")
fi
read -r X0 Y0 X1 Y1 <<< "$GRID_BOUNDS"
SEED=(); SUF=_m
if [ -n "$MAPPROJECT" ]; then                     # stereo.mapproject: the views onto a smooth DSM of another variant first
  [ -f "$PREP/seed_dem.tif" ] || "$PY" -I "$T/seed_dem.py" "$Y"
  read -r P0 Q0 P1 Q1 <<< "$GRID_BOUNDS_PAD"
  for v in $VIEWS; do
    if [ ! -f "$PREP/${v}_map.tif" ]; then
      echo "$(date +%T) mapproject $v onto the DSM of $MAPPROJECT"
      mapproject -t rpc --tr "$RES" --t_srs "$CRS" --t_projwin "$P0" "$Q0" "$P1" "$Q1" "${BAP[@]}" \
        "$PREP/seed_dem.tif" "$PREP/${v}_m.tif" "$PREP/${v}_map.tif" > "$LOGS/mapproject_$v.log" 2>&1
    fi
  done
  SEED=("$PREP/seed_dem.tif"); SUF=_map
fi
for pr in $PAIRS; do
  L=${pr:0:1}; R=${pr:1:1}; S="$STEREO/$pr"
  if [ ! -f "$S/run-PC.tif" ]; then
    echo "$(date +%T) parallel_stereo $L $R ($ALGORITHM, subpixel mode $SUBPIXEL)"
    parallel_stereo -t rpc --stereo-algorithm "$ALGORITHM" --subpixel-mode "$SUBPIXEL" --nodata-value 0 \
      --processes 4 --threads-multiprocess 2 "${BAP[@]}" $STEREO_OPTIONS \
      "$PREP/${L}${SUF}.tif" "$PREP/${R}${SUF}.tif" "$S/run" "${SEED[@]}" > "$LOGS/stereo_$pr.log" 2>&1
  fi
  echo "$(date +%T) point2dem $pr"
  extra=(); [ "$MAX_TRI_ERROR" != "0.0" ] && extra=(--max-valid-triangulation-error "$MAX_TRI_ERROR")
  point2dem --t_srs "$CRS" --tr "$RES" --t_projwin "$X0" "$Y0" "$X1" "$Y1" --gdal-tap --errorimage \
    --nodata-value -9999 "${extra[@]}" $POINT2DEM_OPTIONS "$S/run-PC.tif" -o "$S/run" > "$LOGS/point2dem_$pr.log" 2>&1
done
echo "$(date +%T) stereo done: $PAIRS"
