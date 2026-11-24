# Where the data comes from

Nothing in this file is in the repository: the satellite scenes have their own licence, and every raster made from
them (crops, masks, DSMs, orthophotos, voxel models, pictures) stays on the machine that processed them. The
`.gitignore` keeps `data/*/*` out except the YAML files and the short `summary.md` numbers.

## The satellite scenes (the input)

An ALOS PRISM tri-stereo set over Hiroshima Bay, as three ZIP archives (one per camera), found as
`ALOS_PRISM_RPCs_Hiroshima_Stereo_Triplet/`:

| archive | bytes | scene | product | incidence | scene centre time (UTC) |
|---|---|---|---|---|---|
| `GEOTIFF_RPC_PSM_Froward.zip` (sic) | 188,565,714 | ALPSMF110862855 | O1B2R_UF | 26.4 deg, looking ahead | 2008-02-23 02:00:07 |
| `GEOTIFF_RPC_PSM_Nadir.zip` | 165,739,932 | ALPSMN110862910 | O1B2R_UN | 1.2 deg | 2008-02-23 02:00:52 |
| `GEOTIFF_RPC_PSM_Backward.zip` | 188,246,530 | ALPSMB110862965 | O1B2R_UB | 26.8 deg, looking back | 2008-02-23 02:01:37 |

What the files say about themselves (`HDR-*.txt`, `summary.txt`, `README.txt`):

- **Satellite and sensor**: ALOS ("Daichi", JAXA, 2006-2011), PRISM (Panchromatic Remote-sensing Instrument for
  Stereo Mapping): three pushbroom cameras, forward, nadir and backward (+-23.8 deg along track), 2.5 m, 8 bit.
  `PSM` is JAXA's sensor code for PRISM. Orbit 11086, path 81, frames 2855 / 2910 / 2965, descending, sun at 41.5 deg
  elevation, azimuth 152 deg.
- **Level 1B2R + RPC**: radiometrically and geometrically corrected, georeferenced on the ellipsoid (UTM zone 53N,
  ITRF97/GRS80, the image grid turned 14 deg to follow the orbit, cubic convolution), no terrain correction: relief
  displaces the forward and backward images by +-20 px per 100 m of height, which is what stereo measures.
  16000 lines x 14852-16488 columns (40 x 41 km). Orbit "Precision", attitude "OnSitePrecision".
- **RPC**: `RPC-*.txt`, JAXA's fixed-width text (field order of NITF RPC00B, `Document_rpc_g_e.pdf`), fitted to the
  rigorous model on 500 control points, residual sigma under 0.01 px. Heights above the GRS80 ellipsoid. Image
  coordinates count from (1, 1) at the centre of the first pixel: see `scripts/rpc.py`; GDAL reads the file as it is
  and is one pixel off.
- **Processing**: L1B1 2009-01-07, 1B2R + RPC by RESTEC-PD on 2009-12-09 (processing version 001-003).
- **Distributor and licence**: RESTEC (Remote Sensing Technology Center of Japan), ALOS End User License Agreement
  (`enduser_license_e.txt` in every archive): use for the end user's own internal purposes; "Extracted Images" (views
  of the product, at most 1024 x 768 px on a web site) with the notice `(c)JAXA 2008 DISTRIBUTED BY RESTEC`;
  value-added products (a DSM no longer has the original pixel structure) carry `INCLUDES MATERIAL (c)JAXA 2008`, and
  selling or transferring them needs a "Category B" purchase. Read the licence itself before publishing anything made
  from these scenes.
- **Where it was downloaded from**: Apollo Mapping, <https://apollomapping.com/download-poster-thank-you> (the
  archives are dated 2018-02-07). That page refuses automated requests, so whatever it says about the terms of the
  download was not read here; the licence inside the archives is the one above.

How to use them: unpack the three archives anywhere, and point `data/<project>/scenes` at the folder (a link is fine):

```bash
ln -s /path/to/ALOS_PRISM_RPCs_Hiroshima_Stereo_Triplet data/hiroshima/scenes
```

## Independent data (downloaded by `scripts/reference.py`, cached in `data/<project>/out/ref/`)

| what | used for | source and licence |
|---|---|---|
| GSI elevation tiles DEM5A / DEM5B / DEM5C (5 m) and DEM10B (10 m), heights above T.P. | the accuracy check | 国土地理院 (Geospatial Information Authority of Japan), 地理院タイル 標高タイル, <https://maps.gsi.go.jp/development/ichiran.html>; GSI content terms (attribution, compatible with CC BY 4.0): 出典：国土地理院 |
| GSIGEO2011 (`jp_gsi_gsigeo2011.tif`), JPGEO2024 (`jp_gsi_jpgeo2024.tif`) | heights above T.P. | GSI's geoid models via PROJ-data, <https://cdn.proj.org> |
| EGM2008 2.5' (`us_nga_egm08_25.tif`) | the global geoid, for comparison | NGA, via PROJ-data |
| ESA WorldCover 2021 v200, 10 m | water mask; land-cover classes of the check, the anchor and the ground model | <https://esa-worldcover.org>, CC BY 4.0, (c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium |
| Copernicus DEM GLO-30 | the level of the DSM (`anchor.py`, the default anchor: global); a second check, surface against surface | (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. Read from the AWS open data bucket `copernicus-dem-30m` |

None of them comes from the PRISM scenes. Note that JAXA's own AW3D30 is made from PRISM, so it would not be an
independent check of a PRISM DSM in the same way; the GSI DEM (airborne laser) is.
