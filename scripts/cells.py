#!/usr/bin/env python3
"""The grid of the demo: the products of a run on cells of demo.cell, exactly as the web page gets them.

    from cells import Cells;  c = Cells(p, g)       # p = project.load(yaml), g = geo.Grid(p)
    c.rows, c.cols, c.cell, c.zstep, c.msl
    c.sea, c.inland, c.water       today's water per cell: the sea with its tidal rivers, ponds, both
    c.outside, c.measured, c.WC    no scene behind the cell; a height from stereo (else filled); WorldCover class
    c.S, c.G                       surface and ground, m above today's mean sea level, in steps of demo.zstep
    c.O                            the orthophoto, mean of the cell
    c.z0, c.sea_floor              the lowest value, and the floor the sea is given (there is no bathymetry)
    c.blocks(a), c.median(a)       any raster of the run's grid: its values per cell, their median

demo.py writes these out and floods.py asks its questions of the same numbers, so that what the check reports is
what the page shows. Three decisions live here:
- Today's water. The stereo mask (product/water.tif) was shrunk by masks.water_shrink_m to keep the quays in the
  images; here all of WorldCover's water counts, the sea being the part joined to the sea of water.tif.
- Land is not below today's mean sea level. The water mask says a cell is land, so the sea at today's level does
  not stand on it. A model that is a metre low near the shore would otherwise open with land under water (Hiroshima:
  1.8 km²). Only level 0 changes: one step up, the same cells flood either way.
- Heights are rounded to demo.zstep, as the page stores them.
"""
import cv2
import numpy as np
from scipy import ndimage

SEA_FLOOR = -10.0                                           # m: a floor for the water to stand on


class Cells:
    def __init__(self, p, g):
        D = p.demo
        self.cell, self.zstep, self.msl = float(D.get('cell', 5)), float(D.get('zstep', 0.5)), float(D.get('msl_m', 0.0))
        f = self.f = int(round(self.cell / g.res))
        self.hh, self.ww = g.height // f * f, g.width // f * f
        self.rows, self.cols = self.hh // f, self.ww // f
        wat = np.nan_to_num(g.read(p.product / 'water.tif'))
        wcb = self.blocks(np.nan_to_num(g.read(p.vref / 'worldcover.tif')).astype(np.uint8))
        classes = np.unique(wcb)
        self.WC = classes[np.argmax(np.stack([(wcb == k).sum(axis=2) for k in classes]), axis=0)].astype(np.uint8)
        sea0 = self.blocks((wat == 1).astype(np.float32)).mean(axis=2) > 0.5
        water = sea0 | (self.blocks((wat == 2).astype(np.float32)).mean(axis=2) > 0.5) | (self.WC == 80)
        lab, _ = ndimage.label(water)
        self.sea = np.isin(lab, np.setdiff1d(np.unique(lab[sea0]), [0]))
        self.inland, self.water = water & ~self.sea, water
        S = self.median(g.read(p.product / 'dsm_anchored.tif')) - self.msl
        G = self.median(g.read(p.product / 'ground.tif')) - self.msl
        ortho = g.read(p.product / 'ortho.tif')
        with np.errstate(all='ignore'):
            O = np.nanmean(self.blocks(np.where(ortho > 0, ortho, np.nan)), axis=2)
        self.measured = np.isfinite(S) & ~water
        self.outside = np.isnan(O)                           # no scene behind the cell
        self.O = np.where(np.isnan(O), 0, O)
        S, G = np.where(water, np.nan, S), np.where(water, np.nan, G)
        S = self.inpaint(S, np.isnan(S) & ~water)
        G = self.inpaint(G, np.isnan(G) & ~water)
        S, G = np.maximum(S, 0.0), np.maximum(G, 0.0)        # land is not below today's mean sea level
        G = np.minimum(G, S)                                 # and the ground is never above the surface
        # inland water (ponds): the level of its rim, so that it floods when the sea gets there; the sea: its floor
        rim = tuple(ndimage.distance_transform_edt(water, return_distances=False, return_indices=True))   # the nearest land
        S, G = np.where(self.inland, S[rim], S), np.where(self.inland, G[rim], G)
        S[self.sea] = SEA_FLOOR; G[self.sea] = SEA_FLOOR
        self.sea_floor = SEA_FLOOR
        self.z0 = float(np.floor(min(np.nanmin(S), np.nanmin(G)) / self.zstep) * self.zstep)
        self.S, self.G = self.step(S), self.step(G)

    def blocks(self, a):
        f = self.f
        return a[:self.hh, :self.ww].reshape(self.rows, f, self.cols, f).transpose(0, 2, 1, 3).reshape(self.rows, self.cols, f * f)

    def median(self, a):
        with np.errstate(all='ignore'):
            return np.nanmedian(self.blocks(a), axis=2).astype(np.float32)

    def quantized(self, a):
        """the 16-bit number the page stores for a height"""
        return np.clip(np.round((a - self.z0) / self.zstep), 0, 65535).astype(np.uint16)

    def step(self, a):
        """a height as the page reads it back"""
        return (self.z0 + self.quantized(a) * self.zstep).astype(np.float32)

    @staticmethod
    def inpaint(a, holes):
        """fill holes from their rims (Navier-Stokes), on a 0..1 scale"""
        if not holes.any():
            return a
        lo, hi = np.nanmin(a), np.nanmax(a)
        z = np.where(np.isfinite(a), (a - lo) / (hi - lo), 0).astype(np.float32)
        return np.where(holes, cv2.inpaint(z, holes.astype(np.uint8), 3, cv2.INPAINT_NS) * (hi - lo) + lo, a)
