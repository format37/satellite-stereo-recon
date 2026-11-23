#!/usr/bin/env python3
"""The grid every product of a run lives on, and small raster helpers.

    from geo import Grid;  g = Grid(p)         # p = project.load(yaml)
    g.transform, g.width, g.height, g.crs      # the box `aoi` in `crs`, snapped to `dsm.res`; row 0 = north edge
    g.write('x.tif', array, nodata=np.nan)     # GeoTIFF on the grid
    g.read('x.tif')                            # back, as float32 with nan for nodata
    g.warp('any.tif', resampling='bilinear')   # any raster onto the grid (nan where it has no data)
    g.lonlat_to_ij(lon, lat), g.sample(a, lon, lat)
"""
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage


class Grid:
    def __init__(self, p, res=None, aoi=None):
        self.res = float(res or p.res)
        self.crs = rasterio.crs.CRS.from_string(p.crs)
        x0, y0, x1, y1 = transform_bounds('EPSG:4326', self.crs, *(aoi or p.aoi), densify_pts=64)
        x0, y0 = np.floor(x0 / self.res) * self.res, np.floor(y0 / self.res) * self.res
        x1, y1 = np.ceil(x1 / self.res) * self.res, np.ceil(y1 / self.res) * self.res
        self.bounds = (x0, y0, x1, y1)
        self.width, self.height = int(round((x1 - x0) / self.res)), int(round((y1 - y0) / self.res))
        self.transform = from_origin(x0, y1, self.res, self.res)
        self.to_ll = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)
        self.from_ll = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)

    @property
    def shape(self):
        return self.height, self.width

    def xy(self):
        """map coordinates of the cell centres: x (width,), y (height,)"""
        x0, y0, x1, y1 = self.bounds
        return x0 + (np.arange(self.width) + 0.5) * self.res, y1 - (np.arange(self.height) + 0.5) * self.res

    def lonlat(self, step=1):
        x, y = self.xy()
        X, Y = np.meshgrid(x[::step], y[::step])
        return self.to_ll.transform(X, Y)

    def lonlat_to_ij(self, lon, lat):
        """fractional (row, col) of the grid, 0 = centre of the first cell"""
        x, y = self.from_ll.transform(lon, lat)
        x0, y0, x1, y1 = self.bounds
        return (y1 - np.asarray(y)) / self.res - 0.5, (np.asarray(x) - x0) / self.res - 0.5

    def sample(self, a, lon, lat, order=1):
        """values of a grid array at lon/lat (nan outside)"""
        i, j = self.lonlat_to_ij(lon, lat)
        return ndimage.map_coordinates(np.asarray(a, np.float32), [np.ravel(i), np.ravel(j)], order=order, mode='constant',
                                       cval=np.nan).reshape(np.shape(i))

    def profile(self, dtype='float32', nodata=None):
        return dict(driver='GTiff', width=self.width, height=self.height, count=1, dtype=dtype, crs=self.crs,
                    transform=self.transform, nodata=nodata, tiled=True, blockxsize=512, blockysize=512, compress='deflate',
                    predictor=3 if dtype.startswith('float') else 2)

    def write(self, path, a, nodata=None, dtype=None):
        a = np.asarray(a)
        dtype = dtype or str(a.dtype)
        if dtype == 'bool':
            a, dtype = a.astype('uint8'), 'uint8'
        if dtype.startswith('float') and nodata is None:
            nodata = np.nan
        with rasterio.open(path, 'w', **self.profile(dtype, nodata)) as d:
            d.write(a.astype(dtype), 1)

    def read(self, path):
        with rasterio.open(path) as d:
            a = d.read(1).astype(np.float32)
            if d.nodata is not None and not np.isnan(d.nodata):
                a[a == d.nodata] = np.nan
        return a

    def warp(self, src, resampling='bilinear', band=1, src_nodata=None):
        """any raster (path or open dataset) onto the grid; nan where it has no data"""
        out = np.full(self.shape, np.nan, np.float32)
        ds = rasterio.open(src) if isinstance(src, str) else src
        try:
            reproject(rasterio.band(ds, band), out, src_nodata=src_nodata if src_nodata is not None else ds.nodata,
                      dst_transform=self.transform, dst_crs=self.crs, dst_nodata=np.nan,
                      resampling=getattr(Resampling, resampling))
        finally:
            if isinstance(src, str):
                ds.close()
        return out

    def warp_array(self, a, transform, crs, resampling='bilinear', nodata=np.nan):
        out = np.full(self.shape, np.nan, np.float32)
        reproject(np.asarray(a, np.float32), out, src_transform=transform, src_crs=crs, src_nodata=nodata,
                  dst_transform=self.transform, dst_crs=self.crs, dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
        return out


def stats(d):
    """robust numbers of a height difference (any shape, nan ignored)"""
    d = np.asarray(d, np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0)
    med = float(np.median(d))
    return dict(n=int(d.size), mean=float(d.mean()), median=med, std=float(d.std()), rms=float(np.sqrt(np.mean(d ** 2))),
                nmad=float(1.4826 * np.median(np.abs(d - med))), le68=float(np.percentile(np.abs(d), 68.27)),
                le90=float(np.percentile(np.abs(d), 90)), le95=float(np.percentile(np.abs(d), 95)),
                p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)))
