# Heights: ellipsoid, geoid, sea level

Every height a satellite gives is a height above the **ellipsoid** (WGS84 / GRS80): the RPCs of PRISM are written so,
and so is every point ASP triangulates. The ellipsoid is a geometric surface, not the sea. Water settles on the
**geoid**, which around Hiroshima is 33 m above the ellipsoid (32.9 to 33.5 m over the box). A DSM used as it comes out
of stereo would put the sea at +33 m: a flood model built on it would drown nothing until the water rose 33 m.

`dsm.py` writes both: `product/dsm_ellipsoid.tif` (h) and `product/dsm.tif` (H = h - N, metres above sea level),
with N from the model in `dsm.geoid`:

| model | N here, m | used for |
|---|---|---|
| `gsigeo2011` (default) | 32.89 .. 33.52 | heights above **T.P.** (Tokyo Peil, the mean sea level of Tokyo Bay, the zero of Japanese heights). GSI fitted this geoid to levelled heights, so H agrees with the GSI DEM, Japanese maps and Japanese tide tables, which are all in T.P. |
| `egm2008` | 32.71 .. 33.29 | the global geoid; 0.19 m below GSIGEO2011 here (the T.P. zero is not exactly on the global geoid) |
| `jpgeo2024` | 33.06 .. 33.66 | GSI's successor for the height datum Japan moved to in April 2025; 0.16 m above GSIGEO2011 here. The GSI DEM tiles used for the check were made in the old system, so the check uses GSIGEO2011 |

The three agree within 0.35 m, well under what PRISM stereo resolves (one pixel of disparity is 2.5 m of height for the
forward-backward pair). The choice that matters is ellipsoid versus geoid: 33 m.

## Sea level is not one number

T.P. 0 m is the reference, not the water. At Hiroshima port (Ujina) the Hiroshima prefecture tables give, above the
chart datum (CDL): mean sea level 2.00 m, T.P. 1.84 m, mean high water at spring tides (HWL) 3.76 m, mean low water at
spring tides (LWL) 0.03 m. In T.P.:

| level | m above T.P. |
|---|---|
| mean sea level at Hiroshima | +0.16 |
| mean high water, spring tides | +1.92 |
| mean low water, spring tides | -1.81 |

Sources: Hiroshima prefecture port tables (<https://www.pref.hiroshima.lg.jp/uploaded/life/1138509_9761068_misc.pdf>,
<https://www.pref.hiroshima.lg.jp/soshiki/221/outline-a02.html>); JMA's tide table for Hiroshima puts its datum 2.00 m
below mean sea level (<https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/pdf_hourly/2025/Q8.pdf>). Check the
current values before using them.

So a flood model should say which water it starts from: mean sea level (+0.16 m), a spring high tide (+1.9 m), or a
storm surge on top of that. A rise of the sea is added to that level. The reclaimed land of the port and the delta
lie within a few metres of these numbers, so the 0.2 m between geoid models and the 0.16 m between T.P. and the local
mean sea level are small next to the tide, but not zero.

## The water in the scenes

The scenes were taken on 2008-02-23 at 11:00 JST at some state of the tide (not looked up). The DSM does not use
the water surface: water is masked (WorldCover) and is placed at a level of our choosing in the voxel model (0 m).
