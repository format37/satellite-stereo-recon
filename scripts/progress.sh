#!/usr/bin/env bash
# Progress of a run. In a second terminal: scripts/progress.sh data/hiroshima/fb.yaml   (Ctrl+C to stop)
# It only reads the folders and logs of the run, so it cannot disturb it.
PY=${PY:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}; PY=${PY:-python3}
eval "$("$PY" -I "$(dirname "$0")/project.py" "${1:?usage: progress.sh VARIANT.yaml}")"
n() { ls -d $1 2>/dev/null | wc -l; }
while true; do
  line="$(date +%T)  crops $(n "$PREP/?_m.tif")  ba $([ -f "$BA/run-final_residuals_stats.txt" ] && echo done || echo -)"
  for pr in $PAIRS; do
    S="$STEREO/$pr"; tiles=$(n "$S/run-*_*_*_*/")
    line="$line | $pr tiles $tiles: corr $(n "$S/run-*_*_*_*/*-D.tif") rfne $(n "$S/run-*_*_*_*/*-RD.tif") tri $(n "$S/run-*_*_*_*/*-PC.tif")"
    [ -f "$S/run-PC.tif" ] && line="$line cloud"; [ -f "$S/run-DEM.tif" ] && line="$line DEM"
  done
  for f in dsm.tif check.json voxels.npz renders/voxels_overview.png; do [ -e "$PRODUCT/$f" ] && line="$line | $f"; done
  printf '\r%s   ' "$line"
  sleep 10
done
