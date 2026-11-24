#!/usr/bin/env python3
"""The pictures of the write-up, from the products of a run (optional; after floods).

    python3 -I scripts/figures.py data/hiroshima/fnb_global.yaml [OUTDIR]

  fig-views.jpg    the same ground in every view, each drawn on the map grid at mean sea level: what lies at sea level
                   falls on the same pixel in all of them, what stands above it slides away from the camera, one way
                   in the forward view and the other in the backward one. That slide is what stereo measures. A cross
                   marks the highest point of the DSM in the frame on the map, a ring where each view shows it.
  fig-error.jpg    the anchored DSM minus the check reference (GSI): roofs and canopy, and where the shape is wrong
  fig-ground.jpg   relief of the anchored DSM, of the ground model and of the reference, side by side
  fig-flood.jpg    the land the sea reaches at two levels, on the ground model and on the reference (floods.py)
Settings: the `figures` section of the YAML (where to look, which levels, figures.languages). Every picture is drawn
once per language: fig-views.jpg in English, fig-views.ru.jpg, fig-views.ja.jpg. The credit notices stay as the licence
words them. No picture is larger than 1024 x 768 px, the size the licence of the ALOS scenes names for images on a web
site (docs/data.md). Pictures 2-4 need the check reference (Japan); the first needs only the scenes.
Lettering: FIGURES_FONT and FIGURES_FONT_BOLD name font files (any FreeType reads; several separated by ':', a face of
a collection as file#index), used like a browser's font stack: each character from the first file that has it. DejaVu
Sans Mono is the last resort (no Japanese).
Writes product/figures/ and copies it to OUTDIR when given (the blog's public/media/<name>/: never into git).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rpc as R
from geo import Grid
from project import load

p = load(sys.argv[1])
F = p.figures
g = Grid(p)
out = p.product / 'figures'
out.mkdir(parents=True, exist_ok=True)
msl = float(p.demo.get('msl_m', 0.0))
MAXW, MAXH = 1024, 768
NOTICE_IMG = F.get('notice_image', '(c)JAXA 2008 DISTRIBUTED BY RESTEC')
NOTICE_VAP = F.get('notice_product', 'INCLUDES MATERIAL (c)JAXA 2008')
SEA = np.array([38, 50, 62], np.float32)                   # today's water: a dark slate, not a data colour
INK, BOX = (241, 241, 243), (8, 10, 14, 175)
T1, T2, T3 = 19, 16, 15                                    # lettering: titles, legends, credits (the pictures are shown
                                                           # at about half their width in a column of text)
LOG = {}

# the words in the pictures, per language (place names and levels come from the YAML)
TEXT = {
    'en': dict(views={'F': 'forward', 'N': 'nadir', 'B': 'backward'}, m='m', error_title='surface model minus laser, m',
               lower='lower', higher='higher', surface='surface model', ground='ground model', laser='laser survey',
               sea='sea at +{L} m', areas='model {a} km² · laser {b} km²', both='under water in both',
               only_model='only in the model', only_laser='only in the laser'),
    'ru': dict(views={'F': 'вперёд', 'N': 'надир', 'B': 'назад'}, m='м', error_title='модель поверхности минус лазер, м',
               lower='ниже', higher='выше', surface='модель поверхности', ground='модель рельефа', laser='лазерная съёмка',
               sea='море на +{L} м', areas='модель {a} км² · лазер {b} км²', both='под водой в обеих',
               only_model='только в модели', only_laser='только в лазерной'),
    'ja': dict(views={'F': '前方視', 'N': '直下視', 'B': '後方視'}, m='m', error_title='表層モデル − レーザー測量（m）',
               lower='低い', higher='高い', surface='表層モデル', ground='地形モデル', laser='レーザー測量',
               sea='海面 +{L} m', areas='モデル {a} km² · レーザー {b} km²', both='両方で水没',
               only_model='モデルだけ', only_laser='レーザーだけ'),
}


def num(x, fmt, lang):
    """a number as the language writes it (a decimal comma in Russian)"""
    s = format(x, fmt)
    return s.replace('.', ',') if lang == 'ru' else s


def one(v, lang):
    """a text of the YAML: one string, or one per language"""
    return v if v is None or isinstance(v, str) else v.get(lang, v.get('en'))


# ---- lettering: a font stack, as a browser has it
def _files(bold):
    env = [os.environ.get('FIGURES_FONT_BOLD', '')] if bold else []
    env.append(os.environ.get('FIGURES_FONT', ''))
    files = []
    for part in os.pathsep.join(env).split(os.pathsep):
        path, _, idx = part.partition('#')
        if path and Path(path).exists():
            files.append((path, int(idx or 0)))
    import matplotlib
    files.append((str(Path(matplotlib.get_data_path()) / 'fonts/ttf' / ('DejaVuSansMono-Bold.ttf' if bold else 'DejaVuSansMono.ttf')), 0))
    return files


class Face:
    """one size of a font stack: each character from the first font that has a glyph for it"""
    _made = {}
    _missing = set()

    def __init__(self, size, bold):
        self.fonts = [ImageFont.truetype(f, size, index=i) for f, i in _files(bold)]
        self.notdef = [(f.getmask('\U000F0000').getbbox(), f.getlength('\U000F0000')) for f in self.fonts]
        self.pick = {}

    def which(self, ch):
        if ch not in self.pick:
            k = next((k for k, f in enumerate(self.fonts) if (f.getmask(ch).getbbox(), f.getlength(ch)) != self.notdef[k]), None)
            if k is None:
                if ch not in Face._missing:
                    print(f'  no font has {ch!r}: set FIGURES_FONT')
                    Face._missing.add(ch)
                k = 0
            self.pick[ch] = k
        return self.pick[ch]

    def runs(self, text):
        out_ = []
        for ch in text:
            k = (out_[-1][0] if out_ else 0) if ch.isspace() else self.which(ch)
            if out_ and out_[-1][0] == k:
                out_[-1][1] += ch
            else:
                out_.append([k, ch])
        return [(self.fonts[k], s) for k, s in out_]

    def length(self, text):
        return sum(f.getlength(s) for f, s in self.runs(text))

    def draw(self, d, xy, text, fill=INK, anchor='la'):
        """anchor as Pillow's: l/m/r, then a (ascender)/m (middle)/s (baseline)/d (descender)"""
        x, y = xy
        asc, desc = self.fonts[0].getmetrics()
        x -= {'l': 0, 'm': 0.5, 'r': 1}[anchor[0]] * self.length(text)
        base = {'a': y + asc, 't': y + asc, 'm': y + (asc - desc) / 2, 's': y, 'd': y - desc, 'b': y - desc}[anchor[1]]
        for f, s in self.runs(text):
            d.text((x, base), s, font=f, fill=fill, anchor='ls')
            x += f.getlength(s)


def font(size, bold=False):
    if (size, bold) not in Face._made:
        Face._made[size, bold] = Face(size, bold)
    return Face._made[size, bold]


def label(im, xy, text, size=T1, anchor='lt', bold=False, box=BOX):
    """text in a translucent box; xy is the corner named by anchor (l/r, t/b). Returns the box."""
    d = ImageDraw.Draw(im, 'RGBA')
    ft = font(size, bold)
    lines = text.split('\n')
    lh = int(round(size * 1.45))
    w, h = max(ft.length(s) for s in lines), lh * len(lines)
    px, py = int(size * 0.55), int(size * 0.3)
    x, y = xy
    if 'r' in anchor:
        x -= w + 2 * px
    if 'b' in anchor:
        y -= h + 2 * py
    d.rounded_rectangle([x, y, x + w + 2 * px, y + h + 2 * py], 2, fill=box)
    for k, s in enumerate(lines):
        ft.draw(d, (x + px, y + py + k * lh + (lh - size) / 2 - 1), s)
    return x, y, x + w + 2 * px, y + h + 2 * py


def hillshade(z, res, az=315.0, alt=45.0, ve=1.0):
    z = np.where(np.isfinite(z), z, np.nanmedian(z)).astype(np.float32)
    gy, gx = np.gradient(z * ve, res)
    sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    return np.clip(np.sin(b) * np.cos(sl) + np.cos(b) * np.sin(sl) * np.cos(a - asp), 0, 1)


def shrink(a, size, valid=None):
    """area mean of a (nan ignored) on a smaller grid of size (w, h); nan where less than 30% of a cell has data"""
    m = np.isfinite(a) if valid is None else valid & np.isfinite(a)
    num_ = cv2.resize(np.where(m, a, 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    with np.errstate(all='ignore'):
        return np.where(den > 0.3, num_ / den, np.nan)


def fit(w, h, maxw=MAXW, maxh=MAXH):
    s = min(maxw / w, maxh / h)
    return int(round(w * s)), int(round(h * s))


def scale_bar(im, x, y, metres, m_per_px, text):
    d = ImageDraw.Draw(im, 'RGBA')
    L = metres / m_per_px
    d.rounded_rectangle([x - 8, y - T2 - 14, x + L + 8, y + 9], 2, fill=BOX)
    d.line([x, y, x + L, y], fill=INK, width=2)
    for xx in (x, x + L):
        d.line([xx, y - 5, xx, y + 1], fill=INK, width=2)
    font(T2).draw(d, (x + L / 2, y - 8), text, anchor='ms')


def save(im, name, quality=90):
    assert im.width <= MAXW and im.height <= MAXH, (name, im.size)
    im.convert('RGB').save(out / name, quality=quality, optimize=True, progressive=True, subsampling=0)
    print(f'{name}: {im.width} x {im.height} px, {(out / name).stat().st_size / 1e3:.0f} kB')


# ---- OKLab, for a diverging scale whose two arms have the same lightness steps
def to_lab(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = np.cbrt(0.4122214708 * c[..., 0] + 0.5363325363 * c[..., 1] + 0.0514459929 * c[..., 2])
    m = np.cbrt(0.2119034982 * c[..., 0] + 0.6806995451 * c[..., 1] + 0.1073969566 * c[..., 2])
    s = np.cbrt(0.0883024619 * c[..., 0] + 0.2817188376 * c[..., 1] + 0.6299787005 * c[..., 2])
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)


def to_rgb(lab):
    l = (lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]) ** 3
    m = (lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]) ** 3
    s = (lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]) ** 3
    c = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                  -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1).clip(0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def hexrgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


MID = '#f0efec'                                            # the neutral midpoint: nothing
BLUE = [MID, '#b7d3f6', '#5598e7', '#1c5cab', '#0d366b']   # one hue, light to dark: lower than the reference
_b = to_lab(np.array([hexrgb(h) for h in BLUE]))
_hue = np.arctan2(*to_lab(np.array(hexrgb('#e34948')))[[2, 1]])      # the red of the palette: higher than the reference
_c = np.hypot(_b[:, 1], _b[:, 2])
_r = np.stack([_b[:, 0], _c * np.cos(_hue), _c * np.sin(_hue)], -1); _r[0] = _b[0]


def diverging(v, vmax):
    """v in metres -> RGB: blue below zero, red above, the same lightness at the same distance from zero"""
    t = np.clip(np.nan_to_num(v) / vmax, -1, 1)
    x = np.abs(t) * (len(BLUE) - 1)
    i = np.minimum(x.astype(int), len(BLUE) - 2); f = (x - i)[..., None]
    arm = np.where((t < 0)[..., None], _b[i] * (1 - f) + _b[i + 1] * f, _r[i] * (1 - f) + _r[i + 1] * f)
    return to_rgb(arm)


def colourbar(im, x, y, w, vmax, left, right, title):
    d = ImageDraw.Draw(im, 'RGBA')
    d.rounded_rectangle([x - 16, y - 2 * T2 - 30, x + w + 16, y + T2 + 26], 2, fill=BOX)
    font(T2, True).draw(d, (x, y - T2 - 18), title, anchor='ls')
    bar = diverging(np.linspace(-vmax, vmax, w)[None, :].repeat(12, 0), vmax).astype(np.uint8)
    im.paste(Image.fromarray(bar), (x, y))
    ft = font(T2)
    for v in (-vmax, -vmax / 2, 0, vmax / 2, vmax):
        xx = x + (v + vmax) / (2 * vmax) * (w - 1)
        d.line([xx, y + 12, xx, y + 16], fill=INK, width=1)
        ft.draw(d, (xx, y + 17), f'{v:+.0f}' if v else '0', anchor='ma')
    ft.draw(d, (x, y - 7), left, anchor='ls')
    ft.draw(d, (x + w, y - 7), right, anchor='rs')


# ---------------------------------------------------------------- 1. the views on one grid
def fig_views(lang, sfx):
    t = TEXT[lang]
    lon0, lat0 = F.get('views_at') or g.to_ll.transform(*[(g.bounds[0] + g.bounds[2]) / 2, (g.bounds[1] + g.bounds[3]) / 2])
    s = float(F.get('views_m_per_px', 5.0))
    views = [v for v in F.get('views', p.all_views) if (p.prep / f'{v}.tif').exists()]
    gap = 6
    pw = (MAXW - gap * (len(views) - 1)) // len(views)
    ph = min(int(F.get('views_height_px', 430)), MAXH)
    xc, yc = g.from_ll.transform(lon0, lat0)
    N = g.read(p.vref / f'geoid_{p.geoid}.tif')
    H = g.read(p.product / ('dsm_anchored.tif' if (p.product / 'dsm_anchored.tif').exists() else 'dsm.tif'))
    i0, j0 = (int(round(float(v))) for v in g.lonlat_to_ij(lon0, lat0))
    h0 = float(N[i0, j0]) + msl                             # mean sea level, above the ellipsoid
    # the highest point of the DSM in the picture: the thing to watch
    rr = int(min(pw, ph) * s / 2 / g.res * 0.8)
    win = H[i0 - rr:i0 + rr, j0 - rr:j0 + rr]
    k = np.unravel_index(np.nanargmax(ndimage.median_filter(np.nan_to_num(win, nan=-1e9), 5)), win.shape)
    it, jt = i0 - rr + k[0], j0 - rr + k[1]
    xt, yt = g.xy()[0][jt], g.xy()[1][it]
    lont, latt = g.to_ll.transform(xt, yt)
    ht = float(H[it, jt])                                   # above the geoid
    info = json.load(open(p.product / 'views.json'))['views']
    im = Image.new('RGB', (pw * len(views) + gap * (len(views) - 1), ph), (16, 20, 24))
    LOG['views'] = dict(at=[lon0, lat0], m_per_px=s, sea_level_above_ellipsoid_m=h0, top=dict(lon=lont, lat=latt, above_msl_m=ht - msl), slide={})
    for n, v in enumerate(views):
        with __import__('rasterio').open(p.prep / f'{v}.tif') as d:
            img = d.read(1).astype(np.float32); rc = R.from_rasterio(d.rpcs)
        f2 = 2                                              # drawn twice as fine, then averaged down
        xs = xc + (np.arange(pw * f2) + 0.5) * s / f2 - pw * s / 2
        ys = yc - (np.arange(ph * f2) + 0.5) * s / f2 + ph * s / 2
        X, Y = np.meshgrid(xs, ys)
        lo, la = g.to_ll.transform(X, Y)
        r, c = R.project(rc, lo, la, h0)
        a = cv2.remap(img, c.astype(np.float32), r.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
        lo_, hi_ = np.percentile(a[a > 0], [1, 99.7])
        a = np.clip((a - lo_) / (hi_ - lo_), 0, 1) ** 0.72 * 255
        im.paste(Image.fromarray(a.astype(np.uint8)).convert('RGB'), (n * (pw + gap), 0))
        # where the top is drawn: its pixel, seen as if it lay at sea level
        rt, ct = R.project(rc, lont, latt, ht + float(N[it, jt]))
        lo2, la2 = R.localize(rc, rt, ct, h0)
        x2, y2 = g.from_ll.transform(float(lo2), float(la2))
        dE, dN = x2 - xt, y2 - yt
        LOG['views']['slide'][v] = dict(east_m=dE, north_m=dN, m=float(np.hypot(dE, dN)), azimuth=float(np.degrees(np.arctan2(dE, dN)) % 360))
        d = ImageDraw.Draw(im, 'RGBA')
        ox = n * (pw + gap)
        cx, cy = ox + (xt - xc) / s + pw / 2, (yc - yt) / s + ph / 2
        ax_, ay_ = cx + dE / s, cy - dN / s                    # where this view shows the top
        for col, wd in (((16, 20, 24, 255), 6), ((255, 255, 255, 255), 3)):   # a cross: the top's place on the map, the
            d.line([cx - 10, cy, cx + 10, cy], fill=col, width=wd)            # same pixel in every panel
            d.line([cx, cy - 10, cx, cy + 10], fill=col, width=wd)
        for col, wd in (((16, 20, 24, 255), 5), ((255, 255, 255, 255), 3)):   # a ring: where this view shows it
            d.ellipse([ax_ - 13, ay_ - 13, ax_ + 13, ay_ + 13], outline=col, width=wd)
        label(im, (ox + 8, 8), f'{v} · {t["views"].get(v, v)} · {num(info[v]["off_nadir"], ".1f", lang)}°', bold=True)
        if n == 0:
            scale_bar(im, ox + 18, ph - 20, 500, s, f'500 {t["m"]}')
    label(im, (im.width - 8, ph - 8), NOTICE_IMG, T3, 'rb')
    save(im, f'fig-views{sfx}.jpg', 88)
    if not sfx:
        sl = LOG['views']['slide']
        print('  the cross: the highest point of the DSM here, {:.0f} m above mean sea level at {:.5f}, {:.5f}; drawn at sea level it lands '.format(ht - msl, latt, lont) +
              ', '.join(f'{v}: {sl[v]["m"]:.0f} m towards {sl[v]["azimuth"]:.0f} deg' for v in views))


# ---------------------------------------------------------------- reference-based pictures
def tp(name):
    """a product of the run, in heights above today's mean sea level"""
    return g.read(p.product / name) - msl


def fig_error(gsi, water, land, lang, sfx):
    t = TEXT[lang]
    vmax = float(F.get('error_range_m', 20))
    dh = np.where(land, tp('dsm_anchored.tif') - gsi, np.nan)
    w, h = fit(g.width, g.height)
    d = shrink(dh, (w, h))
    wat = cv2.resize(water.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
    sh = cv2.resize(hillshade(gsi, g.res, ve=1.5), (w, h), interpolation=cv2.INTER_AREA)
    rgb = diverging(d, vmax) * (0.62 + 0.38 * np.clip(sh / 0.72, 0, 1.25))[..., None]
    rgb[np.isnan(d)] = [137, 135, 129]                      # land with no height, or no reference
    rgb[wat] = SEA
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    for k, m in enumerate(F.get('error_marks', []), 1):
        i, j = g.lonlat_to_ij(m[0], m[1])
        x, y = float(j) * w / g.width, float(i) * h / g.height
        dr = ImageDraw.Draw(im, 'RGBA')
        dr.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(16, 20, 24, 235), outline=(255, 255, 255, 255), width=2)
        font(T2, True).draw(dr, (x, y), str(k), anchor='mm')
    cx, cy = F.get('error_bar_at', [30, h - 62])
    colourbar(im, int(cx), int(cy), 330, vmax, t['lower'], t['higher'], one(F.get('error_title'), lang) or t['error_title'])
    label(im, (w - 8, h - 8), NOTICE_VAP, T3, 'rb')
    save(im, f'fig-error{sfx}.jpg', 90)
    LOG['error'] = dict(range_m=vmax, px=[w, h], m_per_px=g.width * g.res / w)


def relief(z, res, wat):
    """grey relief, lit from the north-west, heights pulled up 2.5 times so that roofs show; flat land a mid grey"""
    sh = hillshade(z, res, ve=2.5)
    tone = 0.88 + 0.12 * np.clip(np.nan_to_num(z) / 60.0, 0, 1)
    v = np.clip((0.60 + 1.05 * (sh - np.sin(np.radians(45)))) * tone, 0.06, 0.97) * 255
    rgb = np.repeat(v[..., None], 3, 2) * np.array([1.0, 0.985, 0.955])
    rgb[wat] = SEA
    return rgb


def fig_ground(gsi, water, lang, sfx):
    t = TEXT[lang]
    lon0, lat0 = F.get('ground_at') or g.to_ll.transform((g.bounds[0] + g.bounds[2]) / 2, (g.bounds[1] + g.bounds[3]) / 2)
    f = int(round(float(F.get('ground_m_per_px', 5.0)) / g.res))
    gap = 6
    pw = (MAXW - 2 * gap) // 3
    ph = min(int(F.get('ground_height_px', 470)), MAXH)
    i0, j0 = (int(round(float(v))) for v in g.lonlat_to_ij(lon0, lat0))
    sl = (slice(i0 - ph * f // 2, i0 - ph * f // 2 + ph * f), slice(j0 - pw * f // 2, j0 - pw * f // 2 + pw * f))
    wat = cv2.resize(water[sl].astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) > 0.5
    im = Image.new('RGB', (3 * pw + 2 * gap, ph), (16, 20, 24))
    panels = ((t['surface'], tp('dsm_anchored.tif')), (t['ground'], tp('ground.tif')), (t['laser'], gsi))
    for n, (name, z) in enumerate(panels):
        zz = z[sl]
        zz = np.where(np.isfinite(zz), zz, ndimage.median_filter(np.nan_to_num(zz), 5))
        z5 = cv2.resize(zz.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
        im.paste(Image.fromarray(np.clip(relief(z5, g.res * f, wat), 0, 255).astype(np.uint8)), (n * (pw + gap), 0))
        label(im, (n * (pw + gap) + 8, 8), name, bold=True)
    scale_bar(im, 18, ph - 20, 500, g.res * f, f'500 {t["m"]}')
    label(im, (pw + gap + pw - 8, ph - 8), NOTICE_VAP, T3, 'rb')        # under the two panels made from the scenes
    save(im, f'fig-ground{sfx}.jpg', 90)
    LOG['ground'] = dict(at=[lon0, lat0], m_per_px=g.res * f, px=[pw, ph])


def fig_flood(gsi5_shade, lang, sfx):
    t = TEXT[lang]
    fm = p.product / 'floods_maps.npz'
    if not fm.exists():
        print('fig-flood: no product/floods_maps.npz (floods.py writes it): skipped')
        return
    z = np.load(fm)
    levels = [float(x) for x in z['levels']]
    sea, land = z['sea'] | z['inland'], z['land']
    rows, cols = sea.shape
    gap = 12
    pw = (MAXW - gap * (len(levels) - 1)) // len(levels)
    pw, ph = fit(cols, rows, pw, MAXH)
    BOTH, MODEL, REF = (np.array(hexrgb(c), np.float32) for c in ('#2a78d6', '#eb6834', '#1baf7a'))
    names = {float(k): v for k, v in (F.get('flood_names') or {}).items()}
    im = Image.new('RGB', (pw * len(levels) + gap * (len(levels) - 1), ph), (16, 20, 24))
    LOG['flood'] = {}
    for n, L in enumerate(levels):
        m, r = z['model'][n], z['reference'][n]
        dry = 150 + 95 * np.clip(gsi5_shade / 0.72, 0.4, 1.1)
        rgb = np.repeat(dry[..., None], 3, 2)
        for mask, col in ((m & r, BOTH), (m & ~r, MODEL), (r & ~m, REF)):
            rgb[mask] = col
        rgb[~land & ~sea] = [137, 135, 129]
        num_ = np.stack([cv2.resize((rgb[..., k] * ~sea).astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) for k in range(3)], -1)
        den = cv2.resize((~sea).astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
        small = np.where(den[..., None] > 0.5, num_ / np.maximum(den[..., None], 1e-6), SEA)
        ox = n * (pw + gap)
        im.paste(Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)), (ox, 0))
        km2 = float(z['cell']) ** 2 / 1e6
        a, b, c = float(m.sum() * km2), float(r.sum() * km2), float((m & r).sum() / max((m | r).sum(), 1))
        LOG['flood'][f'{L:g}'] = dict(model_km2=a, reference_km2=b, iou=c)
        name = one(names.get(L), lang)
        label(im, (ox + 8, 8), t['sea'].format(L=num(L, 'g', lang)) + (f' · {name}' if name else '') + '\n' +
              t['areas'].format(a=num(a, '.1f', lang), b=num(b, '.1f', lang)), T2, bold=True)
    d = ImageDraw.Draw(im, 'RGBA')
    keys = ((BOTH, t['both']), (MODEL, t['only_model']), (REF, t['only_laser']))
    ft = font(T2)
    bw = 40 + max(ft.length(s) for _, s in keys) + 14
    x, y = F.get('flood_legend_at', [10, ph - 96])
    d.rounded_rectangle([x, y, x + bw, y + 86], 2, fill=BOX)
    for k, (col, s) in enumerate(keys):
        yy = y + 10 + k * 24
        d.rectangle([x + 11, yy, x + 29, yy + 16], fill=tuple(int(v) for v in col) + (255,))
        ft.draw(d, (x + 40, yy + 8), s, anchor='lm')
    label(im, (im.width - 8, ph - 8), NOTICE_VAP, T3, 'rb')
    save(im, f'fig-flood{sfx}.jpg', 90)


LANGS = [l for l in F.get('languages', ['en']) if l in TEXT]
REFERENCE = p.reference and (p.vref / 'gsi.tif').exists() and (p.product / 'ground.tif').exists()
if REFERENCE:
    N_run, N_gsi = g.read(p.vref / f'geoid_{p.geoid}.tif'), g.read(p.vref / 'geoid_gsigeo2011.tif')
    gsi = g.read(p.vref / 'gsi.tif') + N_gsi - N_run - msl   # the reference above today's mean sea level, like the products
    wc = np.nan_to_num(g.read(p.vref / 'worldcover.tif')).astype(np.uint8)
    water = (np.nan_to_num(g.read(p.product / 'water.tif')) > 0) | (wc == 80)
    land = ~water & (wc > 0) & np.isfinite(gsi)
    f5 = int(round(float(p.demo.get('cell', 5)) / g.res))
    hh, ww = g.height // f5 * f5, g.width // f5 * f5
    with np.errstate(all='ignore'):
        g5 = np.nanmean(gsi[:hh, :ww].reshape(hh // f5, f5, ww // f5, f5), axis=(1, 3))
    shade5 = hillshade(g5, g.res * f5, ve=1.5)
for lang in LANGS:
    sfx = '' if lang == 'en' else f'.{lang}'
    fig_views(lang, sfx)
    if REFERENCE:
        fig_error(gsi, water, land, lang, sfx)
        fig_ground(gsi, water, lang, sfx)
        fig_flood(shade5, lang, sfx)
if not REFERENCE:
    print('no check reference or no ground model: only fig-views')
json.dump(LOG, open(out / 'figures.json', 'w'), indent=1)
if len(sys.argv) > 2:
    dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
    for f_ in out.glob('fig-*.jpg'):
        shutil.copy(f_, dst / f_.name)
    print(f'copied to {dst}')
