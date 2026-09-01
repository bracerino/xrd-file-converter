"""Digitize curves from a plot image (or a photo of one) into .xy files.

Many diffraction patterns are only available as a picture: a figure in a paper,
a screenshot, or a phone photo of a poster or a slide. This tool turns such an
image back into numbers. The user marks the four corners of the plot area and
types in the axis values that belong to them; every pixel is then mapped to
data coordinates through a projective transform, so photos taken at an angle
(rotated, skewed, keystoned) are handled as well as flat screenshots. Curves are
picked out by their colour, cleaned of stray specks, and averaged column by
column into a regular x/y series that can be downloaded as .xy.

Exposed as ``run_image_digitizer_section()`` for the Streamlit app.
"""

import base64
import re
import zipfile
from io import BytesIO

import numpy as np
import plotly.graph_objs as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           find_objects, label, uniform_filter)
from scipy.signal import find_peaks, savgol_filter

from download_log import log_download
from ui_style import apply_button_style


# Long side the uploaded image is scaled down to before anything is measured.
# Everything (corner picking, colour search, extraction) happens in this space,
# which keeps a 12-megapixel phone photo as responsive as a small screenshot
# while still leaving far more pixels than a digitized curve needs.
MAX_SIDE = 1600

# Corner order used everywhere: it matches the unit square below, so index 0 is
# the corner where both axes have their starting value.
CORNER_LABELS = ["Bottom-left", "Bottom-right", "Top-right", "Top-left"]
UNIT_SQUARE = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

FONT_SIZE = 20
AXIS_FONT = 20
TICK_FONT = 16

TOOL_NAME = "Image/photo digitizer"

SHAPE_STRAIGHT = "▭ Straight (screenshot / figure)"
SHAPE_TILTED = "⬟ Tilted or photographed"


# --------------------------------------------------------------------------- #
#  Image loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _prepare_image(data: bytes):
    """Decode an uploaded image into RGB and HSV arrays of workable size.

    The EXIF orientation tag is applied first so portrait phone photos are not
    left lying on their side. Returns ``(rgb, hsv)``, both ``(H, W, 3)`` uint8;
    the HSV copy is what the colour search works on.
    """
    img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    return (np.asarray(img, dtype=np.uint8),
            np.asarray(img.convert("HSV"), dtype=np.uint8))


@st.cache_data(show_spinner=False)
def _image_data_uri(rgb: np.ndarray) -> str:
    """JPEG data URI of the image, used as a plotly background layer.

    Sending the raw array to the browser (``go.Image``) would mean megabytes of
    JSON per rerun; a compressed background image is a fraction of that.
    """
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
#  Projective mapping between pixels and data coordinates
# --------------------------------------------------------------------------- #
def _homography(corners: np.ndarray) -> np.ndarray:
    """3x3 matrix mapping the four pixel corners onto the unit square.

    ``corners`` is ordered bottom-left, bottom-right, top-right, top-left. The
    result sends them to (0,0), (1,0), (1,1), (0,1), i.e. to fractions of the
    plot's x and y range, with v growing upwards. A projective (not affine) map
    is used so that a photo taken from an angle straightens out correctly.
    """
    A = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(corners, UNIT_SQUARE)):
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v]
        b[2 * i], b[2 * i + 1] = u, v
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(h, 1.0).reshape(3, 3)

    # Points on the far side of the horizon come back with a negative scale
    # factor and would masquerade as valid coordinates. Fixing the sign at the
    # centre of the quad lets a simple "scale > 0" test reject them later.
    cx, cy = corners.mean(axis=0)
    if H[2, 0] * cx + H[2, 1] * cy + H[2, 2] < 0:
        H = -H
    return H


def _to_uv(H, px, py):
    """Map pixel coordinates to unit-square coordinates (u, v, scale)."""
    d = H[2, 0] * px + H[2, 1] * py + H[2, 2]
    safe = np.where(np.abs(d) < 1e-12, np.nan, d)
    return ((H[0, 0] * px + H[0, 1] * py + H[0, 2]) / safe,
            (H[1, 0] * px + H[1, 1] * py + H[1, 2]) / safe,
            d)


def _from_uv(Hinv, u, v):
    """Map unit-square coordinates back to pixel coordinates."""
    d = Hinv[2, 0] * u + Hinv[2, 1] * v + Hinv[2, 2]
    return ((Hinv[0, 0] * u + Hinv[0, 1] * v + Hinv[0, 2]) / d,
            (Hinv[1, 0] * u + Hinv[1, 1] * v + Hinv[1, 2]) / d)


@st.cache_data(show_spinner=False)
def _uv_maps(shape, corners_key):
    """Per-pixel u/v maps and the in-plot mask for one set of corners.

    Cached because the maps only change when the user moves a corner, while the
    colour controls around them are re-run on every interaction.
    """
    h, w = shape
    corners = np.asarray(corners_key, dtype=float).reshape(4, 2)
    H = _homography(corners)
    px = np.arange(w, dtype=np.float32)[None, :]
    py = np.arange(h, dtype=np.float32)[:, None]
    u, v, d = _to_uv(H, px, py)
    inside = (d > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return u.astype(np.float32), v.astype(np.float32), inside


def _detect_frame(rgb: np.ndarray):
    """Guess the axis frame of a clean plot as ``(left, right, top, bottom)``.

    Looks for full-width dark rows and full-height dark columns, which is what
    the box around a printed figure looks like. Photos rarely qualify (the
    frame is tilted and the lighting uneven), in which case ``None`` is
    returned and the caller falls back to a generic starting rectangle.
    """
    gray = rgb.mean(axis=2)
    dark = gray < 0.6 * np.percentile(gray, 90)
    h, w = dark.shape
    rows = np.where(dark.mean(axis=1) > 0.5)[0]
    cols = np.where(dark.mean(axis=0) > 0.5)[0]
    if rows.size < 2 or cols.size < 2:
        return None
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    if (bottom - top) < 0.3 * h or (right - left) < 0.3 * w:
        return None
    return float(left), float(right), float(top), float(bottom)


# --------------------------------------------------------------------------- #
#  Automatic plot-area detection
# --------------------------------------------------------------------------- #
def _ink_mask(rgb):
    """Pixels clearly darker than their surroundings, i.e. the printed lines.

    One global threshold cannot serve a photo, where the lit corner of a page
    is brighter than the ink in the shaded one. Comparing every pixel with the
    average of its own neighbourhood makes the frame stand out under any
    lighting, which is what the line search below needs.
    """
    gray = rgb.mean(axis=2)
    return gray < uniform_filter(gray, size=51) - 18


def _line_coverage(line, ink, shape):
    """Fraction of a line's path through the image that runs along ink.

    A printed rule is inked over most of its length; a row of text lies at one
    angle too, but covers only a short stretch of the page. Scoring coverage
    this way is what lets the box search below trust its lines — and unlike the
    longest unbroken run, it is not thrown off by a curve crossing the frame.
    """
    h, w = shape
    angle, rho, _ = line
    ca, sa = np.cos(angle), np.sin(angle)
    reach = int(np.hypot(h, w) / 2) + 1
    t = np.arange(-reach, reach + 1)
    x = np.rint(rho * ca - t * sa + w / 2.0).astype(int)
    y = np.rint(rho * sa + t * ca + h / 2.0).astype(int)
    on_image = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if on_image.sum() < 50:
        return 0.0
    return float(ink[y[on_image], x[on_image]].mean())


def _refine_line(line, ys, xs, shape, tol=4.0):
    """Refit a line found by the Hough vote to the ink it actually sits on.

    The vote is quantized to a quarter of a degree and a whole pixel, which is
    enough to find a rule but not to place a corner on it. Fitting the ink
    within a few pixels of the line (total least squares, so a near-vertical
    rule is no harder than a horizontal one) pins it down properly.
    """
    h, w = shape
    angle, rho, coverage = line
    x0, y0 = xs - w / 2.0, ys - h / 2.0
    for _ in range(2):
        near = np.abs(x0 * np.cos(angle) + y0 * np.sin(angle) - rho) < tol
        if near.sum() < 50:
            break
        px, py = x0[near], y0[near]
        mx, my = px.mean(), py.mean()
        _, vecs = np.linalg.eigh(np.cov(np.vstack([px - mx, py - my])))
        nx, ny = vecs[:, 0]                      # normal = least-spread axis
        new_angle = np.arctan2(ny, nx)
        new_rho = mx * nx + my * ny
        if np.cos(new_angle - angle) < 0:        # same line, opposite normal
            new_angle -= np.pi * np.sign(new_angle)
            new_rho = -new_rho
        if abs(new_angle - angle) > np.deg2rad(5):
            break
        angle, rho = float(new_angle), float(new_rho)
    return angle, rho, coverage


def _hough_family(ys, xs, angles, shape):
    """Strongest straight lines through the ink, around one direction.

    Returns ``(angle, rho, coverage)`` triples with rho measured from the image
    centre, best first, with repeats of the same physical line removed.
    """
    h, w = shape
    x0 = xs - w / 2.0
    y0 = ys - h / 2.0
    span = int(np.hypot(h, w) / 2) + 2
    acc = np.zeros((angles.size, 2 * span + 1), dtype=np.int32)
    for i, a in enumerate(angles):
        rho = np.rint(x0 * np.cos(a) + y0 * np.sin(a)).astype(int) + span
        acc[i] = np.bincount(np.clip(rho, 0, 2 * span), minlength=2 * span + 1)

    peak = acc.max()
    if peak < 40:
        return []
    lines, work = [], acc.copy()
    for _ in range(8):
        i, j = np.unravel_index(work.argmax(), work.shape)
        if work[i, j] < max(40, 0.25 * peak):
            break
        lines.append((float(angles[i]), float(j - span), int(work[i, j])))
        # One physical line answers at several neighbouring angles; clearing a
        # band of rho at every angle retires it in one go.
        work[:, max(0, j - 25):j + 26] = 0
    return lines


def _order_corners(points):
    """Sort four intersections into bottom-left, bottom-right, top-right, top-left."""
    pts = sorted(points, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array([bottom[0], bottom[1], top[1], top[0]], dtype=float)


def _quad_area(pts):
    """Shoelace area of a quadrilateral."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_convex(pts):
    """True when the four corners are in order and do not cross over."""
    signs = []
    for i in range(4):
        a = pts[(i + 1) % 4] - pts[i]
        b = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        signs.append(np.sign(np.cross(a, b)))
    return len(set(signs)) == 1 and 0.0 not in signs


def _intersect(l1, l2, shape):
    """Where two (angle, rho) lines meet, in pixel coordinates."""
    h, w = shape
    A = np.array([[np.cos(l1[0]), np.sin(l1[0])],
                  [np.cos(l2[0]), np.sin(l2[0])]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, np.array([l1[1], l2[1]]))
    return x + w / 2.0, y + h / 2.0


def _coloured_pixels(hsv):
    """Mask of pixels colourful enough to belong to a drawn curve."""
    return (hsv[..., 1] >= 70) & (hsv[..., 2] >= 70) & (hsv[..., 2] <= 250)


def _looks_like_a_curve(mask):
    """Score how much a mask behaves like a graph of y against x.

    A plotted curve has one short vertical stroke per column; a box outline has
    two (its top and its bottom edge) plus tall sides, and a filled shape has
    one very long one. Judging candidate colours this way is what keeps the
    search on the diffraction curve and off the coloured frames and banners
    around it.
    """
    columns = mask.sum(axis=0)
    used = columns > 0
    if used.sum() < 20:
        return 0.0, 0
    starts = mask[:1].astype(np.int8).sum(axis=0) + \
        (np.diff(mask.astype(np.int8), axis=0) == 1).sum(axis=0)
    single = (starts == 1) & (columns <= 15) & used
    return float(single.sum() / used.sum()), int(used.sum())


def _long_pieces(mask, min_width):
    """Drop the parts of a mask that do not run far enough across the picture.

    A plotted curve crosses its axes; a legend key, a marker or a stray speck
    of the same colour does not. This is what keeps the legend of a figure from
    dragging the detected plot area out over it.
    """
    lab, n = label(binary_dilation(mask, np.ones((3, 3), bool)))
    if n == 0:
        return mask
    keep = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is not None and (sl[1].stop - sl[1].start) >= min_width:
            keep.append(i)
    if not keep:
        return np.zeros_like(mask)
    return mask & np.isin(lab, keep)


def _is_straight(mask, tolerance=3.0):
    """True when a mask is essentially a straight line rather than a curve.

    A ruled border or a leader line drawn in the same colour as a curve would
    otherwise be taken for one; a plotted curve wanders away from its own best
    straight line, and a printed rule does not.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return True
    fit = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(fit, ys.astype(float), rcond=None)
    return float((ys - fit @ coef).std()) < tolerance


def _curve_anchor(hsv, coloured, limit=4000):
    """A sample of the curve pixels that the plot area has to contain.

    Whatever else the picture holds, the curves the user is after are inside
    the plot area — so the frame is the smallest box that still contains these
    points, which is what lets the search ignore everything else on a poster,
    a slide or a page. Filled shapes are dropped first (a curve is thin), then
    each colour is judged on whether it is drawn like a curve at all, and every
    colour that passes is used: on a figure with a whole series of curves, one
    of them alone rarely reaches into every corner of the plot.
    """
    if coloured.sum() < 200:
        return None
    core = binary_dilation(binary_erosion(coloured, np.ones((25, 25), bool)),
                           np.ones((31, 31), bool))
    thin = coloured & ~core
    if thin.sum() < 200:
        thin = coloured

    hue = hsv[..., 0]
    width = hsv.shape[1]
    counts = np.bincount(hue[thin] // 8, minlength=32)
    floor = max(100, int(0.02 * thin.sum()))

    union = np.zeros(thin.shape, bool)
    for top in np.where(counts >= floor)[0]:
        near = np.isin(hue // 8, [(top - 1) % 32, top, (top + 1) % 32]) & thin
        near = _long_pieces(near, 0.12 * width)
        score, span = _looks_like_a_curve(near)
        if score >= 0.4 and span >= 0.15 * width and not _is_straight(near):
            union |= near

    ys, xs = np.nonzero(union)
    if xs.size < 100:
        return None
    step = max(1, xs.size // limit)
    return xs[::step].astype(float), ys[::step].astype(float)


def _detect_plot_quad(rgb, hsv):
    """Find the four corners of the plot frame, tilted or photographed or not.

    Long straight rules are pulled out of the ink with a Hough transform, then
    the two horizontal and two vertical ones are chosen that make the smallest
    box still holding the coloured curve. Returns the corners in the usual
    order, or ``None`` when nothing convincing turns up.
    """
    shape = rgb.shape[:2]
    h, w = shape
    coloured = _coloured_pixels(hsv)
    ink = _ink_mask(rgb) & ~binary_dilation(coloured, np.ones((3, 3), bool))
    ys, xs = np.nonzero(ink)
    if xs.size < 500:
        return None
    step = max(1, xs.size // 40000)
    xs, ys = xs[::step].astype(float), ys[::step].astype(float)

    thick = binary_dilation(ink, np.ones((3, 3), bool))
    families = []
    for lo, hi in [(70.0, 110.0), (-20.0, 20.0)]:
        raw = _hough_family(ys, xs, np.deg2rad(np.arange(lo, hi + 0.01, 0.25)), shape)
        fitted = [_refine_line(l, ys, xs, shape) for l in raw]
        scored = [(a, r, _line_coverage((a, r, c), thick, shape)) for a, r, c in fitted]
        good = sorted((l for l in scored if l[2] >= 0.4), key=lambda l: -l[2])[:6]
        families.append(good)
    horiz, vert = families
    if len(horiz) < 2 or len(vert) < 2:
        return None

    anchor = _curve_anchor(hsv, coloured)
    centre = (np.array([w / 2.0]), np.array([h / 2.0]))
    points = anchor if anchor is not None else centre
    px, py = points

    best, best_key = None, None
    for i in range(len(horiz)):
        for j in range(i + 1, len(horiz)):
            for k in range(len(vert)):
                for m in range(k + 1, len(vert)):
                    corners = []
                    for hl in (horiz[i], horiz[j]):
                        for vl in (vert[k], vert[m]):
                            hit = _intersect(hl, vl, shape)
                            if hit is not None:
                                corners.append(hit)
                    if len(corners) != 4:
                        continue
                    quad = _order_corners(corners)
                    if not _is_convex(quad):
                        continue
                    area = _quad_area(quad)
                    if area < 0.06 * h * w:
                        continue
                    u, v, d = _to_uv(_homography(quad), px, py)
                    inside = np.mean((d > 0) & (u >= -0.01) & (u <= 1.01)
                                     & (v >= -0.01) & (v <= 1.01))
                    if inside < 0.98:
                        continue
                    strength = sum(l[2] for l in (horiz[i], horiz[j],
                                                  vert[k], vert[m]))
                    # With a curve to enclose, the tightest box wins; without
                    # one, the clearest lines do.
                    key = (area, -strength) if anchor is not None else (-strength, area)
                    if best_key is None or key < best_key:
                        best, best_key = quad, key
    return best


# --------------------------------------------------------------------------- #
#  Tick marks along an axis
# --------------------------------------------------------------------------- #
def _edge_profile(gray, p0, p1, inward, lo, hi):
    """Mean brightness in a band running parallel to one edge of the plot area.

    The band follows the edge whatever angle the picture was taken at, so tick
    marks stay a dip in a one-dimensional profile instead of a shape that has
    to be found in two.
    """
    edge = p1 - p0
    length = float(np.hypot(*edge))
    if length < 20:
        return None, None
    normal = inward / max(np.hypot(*inward), 1e-9)
    steps = int(round(length))
    t = (np.arange(steps) + 0.5) / steps
    base = p0[None, :] + t[:, None] * edge[None, :]
    h, w = gray.shape
    total = np.zeros(steps)
    offsets = np.arange(lo, hi + 1)
    for off in offsets:
        pts = base + normal[None, :] * off
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        total += gray[yi, xi]
    return t, total / offsets.size


def _dark_runs(t, profile, min_len, max_len, drop=35, merge=0):
    """Centres of the dark stretches in an edge profile, as fractions of the edge.

    ``merge`` closes gaps shorter than that many pixels first, which is what
    turns the separate strokes of "100" into one label rather than three.
    """
    if t is None:
        return []
    dark = profile < np.median(profile) - drop
    if dark.mean() > 0.5:          # the band is not sitting on clean paper
        return []
    if merge > 0:
        dark = binary_closing(dark, np.ones(2 * merge + 1, bool))
    runs, start = [], None
    for i, is_dark in enumerate(np.append(dark, False)):
        if is_dark and start is None:
            start = i
        elif not is_dark and start is not None:
            if min_len <= i - start <= max_len:
                runs.append(0.5 * (t[start] + t[i - 1]))
            start = None
    return runs


def _regularly_spaced(marks, tol=0.15):
    """True when marks are spaced the way the ticks of an axis are.

    Real ticks sit on a regular pitch, with the occasional one missing where
    something is drawn over it. Anything else — a curve crossing the axis, say
    — lands at arbitrary places, and must not be mistaken for a tick.
    """
    if len(marks) < 3:
        return True
    steps = np.diff(sorted(marks))
    base = float(np.median(steps))
    if base <= 0:
        return False
    ratios = steps / base
    return bool(np.all(np.abs(ratios - np.rint(ratios)) < tol) and
                np.all(ratios > 0.8))


def _axis_marks(rgb, hsv, corners, axis):
    """Tick marks and printed labels along the X or Y edge of the plot area.

    Ticks are looked for on both sides of the edge, since a plot may draw them
    pointing into the frame or out of it, and the labels a little further out.
    The curves themselves are painted out first: one of them crossing the axis
    is dark too, and would otherwise be counted as a tick.
    """
    gray = np.where(_coloured_pixels(hsv), 255.0, rgb.mean(axis=2))
    p0, p1 = (corners[0], corners[1]) if axis == "x" else (corners[0], corners[3])
    inward = corners.mean(axis=0) - 0.5 * (p0 + p1)
    ticks = _dark_runs(*_edge_profile(gray, p0, p1, inward, 3, 12), 2, 25)
    if len(ticks) < 2:
        ticks = _dark_runs(*_edge_profile(gray, p0, p1, -inward, 3, 12), 2, 25)
    labels = _dark_runs(*_edge_profile(gray, p0, p1, -inward, 16, 70),
                        8, 220, merge=12)
    return ticks, labels


def _tick_anchors(rgb, hsv, corners, axis):
    """Where to park the two reference marks: on the outermost labelled ticks.

    A tick with a number under it is one the user can read off the picture and
    type in, so those are the ones worth marking. Their value cannot be read
    from the image — that is the one thing the tool asks for — but their
    position can, which is the part that is easy to get wrong by hand.
    """
    ticks, labels = _axis_marks(rgb, hsv, corners, axis)
    if not _regularly_spaced(ticks):
        ticks = []
    if not _regularly_spaced(labels):
        labels = []              # rotated axis text, not a row of numbers
    p0, p1 = (corners[0], corners[1]) if axis == "x" else (corners[0], corners[3])
    edge = float(np.hypot(*(p1 - p0)))

    if len(labels) >= 2:
        picks = [labels[0], labels[-1]]
        if len(ticks) >= 2:
            # A label sits on its own tick, so snap to the nearest one — but
            # only a short way, or a sparse tick list drags it across the plot.
            reach = min(0.5 * float(np.median(np.diff(sorted(ticks)))),
                        25.0 / max(edge, 1.0))
            picks = [min(ticks, key=lambda c: abs(c - p))
                     if min(abs(c - p) for c in ticks) < reach else p
                     for p in picks]
    elif len(ticks) >= 2:
        picks = [ticks[0], ticks[-1]]
    else:
        return None
    if abs(picks[1] - picks[0]) < 0.2:
        return None
    return [p0 + f * (p1 - p0) for f in picks]


# --------------------------------------------------------------------------- #
#  Reading the printed tick numbers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _reading_image(data: bytes, max_side=4000):
    """A high-resolution grey copy of the picture, for reading its lettering.

    Everything else works on a picture scaled down to :data:`MAX_SIDE`, which is
    ample for tracing a curve but throws away the detail that small printed
    numbers are made of. Curves are painted out, so one crossing the axis is
    never mistaken for a digit.
    """
    img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    gray = np.where(_coloured_pixels(hsv), 255, arr.mean(axis=2)).astype(np.uint8)
    return gray


@st.cache_data(show_spinner=False)
def _digit_templates(height=32):
    """Bitmaps of 0-9, all scaled by one factor so their widths stay comparable.

    Rendered from the font Pillow ships with, so no font on the machine and no
    reader of any kind has to be installed for this to work.
    """
    font = ImageFont.load_default(size=height * 3)
    shapes = {}
    for ch in "0123456789":
        canvas = Image.new("L", (height * 6, height * 6), 255)
        ImageDraw.Draw(canvas).text((height, height), ch, font=font, fill=0)
        ink = np.asarray(canvas) < 128
        ys, xs = np.nonzero(ink)
        shapes[ch] = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    tallest = max(s.shape[0] for s in shapes.values())
    out = {}
    for ch, shape in shapes.items():
        factor = height / tallest
        w = max(3, int(round(shape.shape[1] * factor)))
        h = min(height, max(3, int(round(shape.shape[0] * factor))))
        small = np.asarray(Image.fromarray(shape.astype(np.uint8) * 255)
                           .resize((w, h), Image.BILINEAR)) > 100
        box = np.zeros((height, w), bool)
        box[:h] = small
        out[ch] = box
    return out


def _match_digit(glyph, templates, height=32):
    """The digit a glyph looks most like, and how far ahead of the runner-up."""
    h, w = glyph.shape
    factor = height / h
    scaled = np.asarray(Image.fromarray(glyph.astype(np.uint8) * 255)
                        .resize((max(3, int(round(w * factor))), height),
                                Image.BILINEAR)) > 100
    scores = []
    for ch, template in templates.items():
        wide = max(scaled.shape[1], template.shape[1])
        a = np.zeros((height, wide), bool)
        a[:, :scaled.shape[1]] = scaled
        b = np.zeros((height, wide), bool)
        b[:, :template.shape[1]] = template
        best = max((a & np.roll(b, k, axis=1)).sum()
                   / max((a | np.roll(b, k, axis=1)).sum(), 1) for k in (-2, -1, 0, 1, 2))
        scores.append((best, ch))
    scores.sort(reverse=True)
    return scores[0][1], scores[0][0] - scores[1][0]


def _edge_band(gray, p0, p1, outward, lo, hi):
    """The strip of picture just outside one edge, straightened out.

    Sampling along the edge and its normal means the numbers come out upright
    even when the photo is tilted, and the rotated lettering beside a Y axis
    comes out reading left to right.
    """
    edge = p1 - p0
    length = float(np.hypot(*edge))
    if length < 20:
        return None
    normal = outward / max(np.hypot(*outward), 1e-9)
    along = (np.arange(int(round(length))) + 0.5) / length
    base = p0[None, :] + along[:, None] * edge[None, :]
    across = np.arange(lo, hi + 1)
    pts = base[None, :, :] + across[:, None, None] * normal[None, None, :]
    h, w = gray.shape
    xi = np.clip(np.rint(pts[..., 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[..., 1]).astype(int), 0, h - 1)
    return gray[yi, xi]


def _label_row(ink):
    """The rows of a band that carry the most ink — the row of numbers.

    Tick marks and the axis title share the band with them; the numbers are
    what covers most of it.
    """
    rows = ink.sum(axis=1)
    lit = rows > 0
    if not lit.any():
        return None
    blocks, start, gap = [], None, 0
    for i, on in enumerate(np.append(lit, False)):
        if on:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                blocks.append((start, i - gap + 1))
                start = None
    if start is not None:
        blocks.append((start, len(lit)))
    if not blocks:
        return None
    return max(blocks, key=lambda b: rows[b[0]:b[1]].sum())


def _read_numbers(band, templates, height=32):
    """Read the numbers printed along a straightened band.

    Digits are told apart by matching; a dot and a minus are told apart from
    them by being too short to be a digit, and from each other by where they
    sit, which no amount of matching would do reliably at this size.
    """
    if band is None:
        return []
    ink = band < (np.median(band) - 45)
    row = _label_row(ink)
    if row is None:
        return []
    ink = ink[row[0]:row[1]]
    lab, n = label(ink)
    if n == 0:
        return []

    glyphs = []
    for i, sl in enumerate(find_objects(lab), start=1):
        piece = lab[sl] == i
        if piece.sum() < 5:
            continue
        glyphs.append({"box": piece, "x0": sl[1].start, "x1": sl[1].stop,
                       "y1": sl[0].stop, "h": sl[0].stop - sl[0].start,
                       "w": sl[1].stop - sl[1].start})
    if not glyphs:
        return []
    glyphs.sort(key=lambda g: g["x0"])
    tall = max(g["h"] for g in glyphs)
    floor_y = max(g["y1"] for g in glyphs)
    for g in glyphs:
        if g["h"] < 0.45 * tall:
            g["ch"] = "." if g["y1"] > floor_y - 0.25 * tall else "-"
            g["conf"] = 1.0
        else:
            g["ch"], g["conf"] = _match_digit(g["box"], templates, height)

    typical = float(np.median([g["w"] for g in glyphs if g["h"] > 0.45 * tall]
                              or [1.0]))
    words, current = [], [glyphs[0]]
    for g in glyphs[1:]:
        if g["x0"] - current[-1]["x1"] > 0.9 * typical:
            words.append(current)
            current = [g]
        else:
            current.append(g)
    words.append(current)

    out = []
    for word in words:
        text = "".join(g["ch"] for g in word)
        try:
            value = float(text)
        except ValueError:
            continue
        out.append({"value": value,
                    "pos": float(np.mean([(g["x0"] + g["x1"]) / 2 for g in word])),
                    "text": text})
    return out


def _fit_reading(readings):
    """The straight line through (place, value) that most readings agree on.

    Places are measured in the straightened plot, not in the picture, so the
    line stays straight however the photo was angled. A picture's axis is
    linear there, so a reading only counts if it falls on the same line as the
    others — which is what makes reading the numbers safe: a misread digit
    almost never lands on the line its neighbours define, and a line only two
    readings agree on is not used at all.
    """
    if len(readings) < 3:
        return None
    points = [(r["frac"], r["value"]) for r in readings]
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            (x0, v0), (x1, v1) = points[i], points[j]
            if abs(x1 - x0) < 0.05 or v0 == v1:
                continue
            slope = (v1 - v0) / (x1 - x0)
            offset = v0 - slope * x0
            inliers = [k for k, (x, v) in enumerate(points)
                       if abs(v - (slope * x + offset)) < 0.01 * abs(slope)]
            if best is None or len(inliers) > len(best):
                best = inliers
    if best is None or len(best) < 3:
        return None
    xs = np.array([points[k][0] for k in best])
    vs = np.array([points[k][1] for k in best])
    if xs.max() - xs.min() < 0.3:          # all crowded into one corner
        return None
    slope, offset = np.linalg.lstsq(np.vstack([xs, np.ones_like(xs)]).T,
                                    vs, rcond=None)[0]
    if np.abs(vs - (slope * xs + offset)).max() > 0.01 * abs(slope):
        return None
    steps = np.diff(np.unique(vs))
    step = float(np.median(steps)) if steps.size else 0.0
    return float(slope), float(offset), sorted(vs.tolist()), step


def _read_tick_values(gray, corners, axis, fractions, scale):
    """Read what the axis says, and give the values at the two marked ticks.

    ``fractions`` are where those two marks sit along the straightened axis.
    Returns ``(value_one, value_two, numbers_read)`` or ``None`` when the
    numbers cannot be read or do not agree with each other.
    """
    p0, p1 = (corners[0], corners[1]) if axis == "x" else (corners[0], corners[3])
    big0, big1 = p0 * scale, p1 * scale
    outward = 0.5 * (big0 + big1) - corners.mean(axis=0) * scale
    length = float(np.hypot(*(big1 - big0)))
    band = _edge_band(gray, big0, big1, outward,
                      int(round(6 * scale)), int(round(95 * scale)))
    readings = _read_numbers(band, _digit_templates())

    # Put every reading where it belongs on the straightened axis, so that a
    # photo taken at an angle does not bend the line they are fitted to.
    matrix = _homography(corners)
    for r in readings:
        along = r["pos"] / length
        point = p0 + along * (p1 - p0)
        u, v, _ = _to_uv(matrix, np.array([point[0]]), np.array([point[1]]))
        r["frac"] = float(u[0] if axis == "x" else v[0])
    fit = _fit_reading(readings)
    if fit is None:
        return None
    slope, offset, values, step = fit
    grid = step / 10.0 if step > 0 else 0.0
    out = []
    for f in fractions:
        value = slope * f + offset
        out.append(round(value / grid) * grid if grid else value)
    if out[0] == out[1]:
        return None
    return out[0], out[1], values


# --------------------------------------------------------------------------- #
#  Colour search
# --------------------------------------------------------------------------- #
def _hue_distance(hue, target):
    """Shortest distance between hues on the 0-255 colour wheel."""
    d = np.abs(hue.astype(np.int16) - int(target))
    return np.minimum(d, 256 - d)


@st.cache_data(show_spinner=False)
def _colour_candidates(rgb, hsv, inside, min_sat, min_val, max_colours=12):
    """Find the distinctly coloured curves drawn inside the plot area.

    Coloured pixels (saturated enough not to be paper, ink or grid) are
    histogrammed finely by hue, and every peak of that histogram is one curve.
    Peaks rather than bands: a figure that steps through a colour scale puts
    two blues next to each other, and merging neighbouring hues would hand back
    one curve where the reader sees two. Black text and axes stay out of the
    list by construction, which is why they never have to be masked by hand.
    """
    h = hsv[..., 0][inside]
    s = hsv[..., 1][inside]
    v = hsv[..., 2][inside]
    if h.size == 0:
        return []
    drawn = (s >= min_sat) & (v >= min_val) & (v <= 250)
    if drawn.sum() < 30:
        return []

    hue = h[drawn]
    n_bins = 128                                  # 2.8 degrees per bin
    bins = hue // 2
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(np.r_[counts[-4:], counts, counts[:4]],
                         kernel, mode="same")[4:-4]

    floor = max(50.0, 0.004 * drawn.sum())
    pad = 20                                      # so a red curve at the seam
    extended = np.r_[smooth[-pad:], smooth, smooth[:pad]]   # is not cut in two
    found, _ = find_peaks(extended, height=floor,
                          prominence=0.35 * floor, distance=3)
    peaks = sorted({int((i - pad) % n_bins) for i in found
                    if pad <= i < pad + n_bins})
    if not peaks:
        return []

    # Every hue belongs to the nearest peak, unless it is far from all of them.
    ids = np.arange(n_bins)
    gap = np.abs(ids[:, None] - np.array(peaks)[None, :])
    gap = np.minimum(gap, n_bins - gap)
    owner = gap.argmin(axis=1)
    owner[gap.min(axis=1) > 8] = -1

    flat_rgb = rgb[inside][drawn]
    out = []
    for k in range(len(peaks)):
        member = owner[bins] == k
        if member.sum() < floor:
            continue
        # Circular mean, so a red spanning 250->5 averages to red, not to cyan.
        ang = hue[member].astype(float) * (2 * np.pi / 256)
        mean_hue = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                    % (2 * np.pi)) * (256 / (2 * np.pi))
        out.append({
            "hue": float(mean_hue),
            "count": int(member.sum()),
            "rgb": tuple(int(c) for c in np.median(flat_rgb[member], axis=0)),
        })
    out.sort(key=lambda c: -c["count"])
    return out[:max_colours]


def _otsu(values, lo=0, hi=255):
    """Otsu's split of a channel into its dark and bright halves.

    Used on saturation inside the plot area: paper, ink and printed text sit on
    one side of the split and the drawn curves on the other, which gives a
    colour threshold taken from the picture instead of a guess.
    """
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return lo
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return lo
    weighted = hist * levels
    m0 = np.cumsum(weighted) / np.maximum(w0, 1)
    m1 = (weighted.sum() - np.cumsum(weighted)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -1
    return int(np.clip(levels[int(between.argmax())], lo, hi))


def _colour_mask(hsv, inside, hue, hue_tol, min_sat, min_val):
    """Pixels inside the plot whose hue is close enough to a target colour."""
    return _curve_masks(hsv, inside, [hue], hue_tol, min_sat, min_val)[0]


def _curve_masks(hsv, inside, hues, hue_tol, min_sat, min_val):
    """One mask per curve, with every pixel going to the colour it is nearest.

    Tracing a whole series at once means neighbouring hues — the blue and the
    slightly greener blue of a colour scale — would otherwise both claim the
    same pixels under any tolerance wide enough to catch anti-aliased edges.
    Letting the closest colour win keeps each curve to its own ink.
    """
    tol = hue_tol * 256 / 360.0
    base = (inside & (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val))
    gaps = [_hue_distance(hsv[..., 0], round(h)) for h in hues]
    nearest = gaps[0] if len(gaps) == 1 else np.min(gaps, axis=0)
    return [base & (gap <= tol) & (gap <= nearest) for gap in gaps]


def _dark_mask(hsv, inside, max_val, max_sat):
    """Pixels inside the plot dark enough to be a black or grey curve."""
    return inside & (hsv[..., 2] <= max_val) & (hsv[..., 1] <= max_sat)


def _clean_mask(mask, min_blob, keep_largest, close_gaps):
    """Drop specks and, optionally, everything but the largest curve fragment.

    Anti-aliasing and JPEG noise scatter a few matching pixels over labels and
    tick marks; removing small connected blobs takes those out without touching
    the curve, which is one long component.
    """
    if close_gaps:
        mask = binary_closing(mask, np.ones((3, 3), bool))
    if min_blob <= 1 and not keep_largest:
        return mask
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = ([int(sizes.argmax())] if keep_largest
            else np.where(sizes >= min_blob)[0])
    return np.isin(lab, keep) & (lab > 0)


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
MODE_PEAKS = "Peak-preserving (recommended)"
MODE_CENTRE = "Centre of the stroke"
MODE_MEAN = "Mean of the stroke"
MODE_TOP = "Top edge of the stroke"
MODE_BOTTOM = "Bottom edge of the stroke"
EXTRACTION_MODES = [MODE_PEAKS, MODE_CENTRE, MODE_MEAN, MODE_TOP, MODE_BOTTOM]


def _bin_stats(uu, vv, n_bins):
    """Per-x-step count, middle, bottom and top of the ink stroke."""
    counts = np.zeros(n_bins, dtype=int)
    middle = np.full(n_bins, np.nan)
    mean = np.full(n_bins, np.nan)
    low = np.full(n_bins, np.nan)
    high = np.full(n_bins, np.nan)
    if uu.size == 0:
        return counts, middle, mean, low, high

    idx = np.clip((uu * n_bins).astype(int), 0, n_bins - 1)
    order = np.argsort(idx, kind="stable")
    idx_s, v_s = idx[order], vv[order]
    edges = np.arange(n_bins + 1)
    starts = np.searchsorted(idx_s, edges[:-1], side="left")
    ends = np.searchsorted(idx_s, edges[1:], side="left")
    for i in range(n_bins):
        seg = v_s[starts[i]:ends[i]]
        counts[i] = seg.size
        if seg.size:
            middle[i] = np.median(seg)
            mean[i] = seg.mean()
            low[i] = seg.min()
            high[i] = seg.max()
    return counts, middle, mean, low, high


def _reduce_stroke(mode, counts, middle, mean, low, high):
    """Turn the ink in each x step into the one value the curve had there.

    Where the curve is flat, the ink is just the drawn line and its middle is
    the answer. Where the curve is steep, one x step of ink stretches from the
    value at one edge of the step to the value at the other, so the middle of
    that stretch is not a point on the curve at all — it is halfway down the
    flank. That is what flattens a tall, narrow peak: at its tip a single step
    of ink runs from near the baseline to the apex, and its middle sits well
    below the top.

    The peak-preserving mode therefore takes the top of the ink, less half a
    line width, whenever a step is thicker than the drawn line, and the middle
    everywhere else. The two agree exactly where the ink is one line thick, so
    the switch introduces no step of its own; on a steep flank taking the top
    is the value at the step's leading edge, which is a shift of half a pixel
    in x rather than an error in intensity.
    """
    if mode == MODE_CENTRE:
        return middle
    if mode == MODE_MEAN:
        return mean
    if mode == MODE_TOP:
        return high
    if mode == MODE_BOTTOM:
        return low

    drawn = counts > 0
    if not drawn.any():
        return middle
    # The typical thickness is the width of the line itself: flat stretches
    # outnumber steep ones on any spectrum, so the median measures the pen.
    half_line = 0.5 * float(np.median((high - low)[drawn]))
    return np.fmax(middle, high - half_line)


def _extract_curve(mask, u, v, n_points, mode, n_fine=None):
    """Reduce a curve mask to one y value per x sample.

    Binning happens in data space (u), not in image columns, so a tilted photo
    still yields evenly spaced x values. The stroke is always read at the
    picture's own resolution — one step per pixel across the plot area — and
    the requested number of points is only the grid it is reported on. That
    keeps a coarse output from smearing a sharp peak before it has been
    measured, and a fine one from turning single pixels into gaps.

    Returns the values on the requested grid, and the share of the x range that
    actually carried ink at the picture's resolution.
    """
    uu = u[mask].astype(np.float64)
    vv = v[mask].astype(np.float64)
    ok = np.isfinite(uu) & np.isfinite(vv)
    uu, vv = uu[ok], vv[ok]

    fine = int(min(max(n_fine or n_points, 100), 8000))
    counts, middle, mean, low, high = _bin_stats(uu, vv, fine)
    values = _reduce_stroke(mode, counts, middle, mean, low, high)
    drawn = counts > 0
    coverage = float(drawn.mean())
    if n_points == fine:
        return values, coverage

    u_fine = (np.arange(fine) + 0.5) / fine
    u_out = (np.arange(n_points) + 0.5) / n_points
    out = np.full(n_points, np.nan)

    if n_points < fine:
        # Fewer points than pixels: average the pixel steps that fall in each.
        owner = np.minimum(u_fine * n_points, n_points - 1).astype(int)
        for j in range(n_points):
            block = values[owner == j]
            good = block[np.isfinite(block)]
            if good.size:
                out[j] = good.mean()
        return out, coverage

    # More points than pixels: interpolate between what was measured, but only
    # inside the stretches that carry ink, so real gaps stay gaps.
    good = np.isfinite(values)
    if good.sum() >= 2:
        out = np.interp(u_out, u_fine[good], values[good])
        out[~drawn[np.clip((u_out * fine).astype(int), 0, fine - 1)]] = np.nan
    return out, coverage


def _fill_gaps(u_centres, values):
    """Interpolate across bins the curve missed, without inventing tails."""
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return values, 0
    first, last = np.argmax(valid), len(valid) - 1 - np.argmax(valid[::-1])
    span = slice(first, last + 1)
    holes = int((~valid[span]).sum())
    filled = values.copy()
    filled[span] = np.interp(u_centres[span], u_centres[valid], values[valid])
    return filled, holes


def _to_axis(frac, lo, hi, log):
    """Turn a 0-1 fraction of an axis into the value printed on that axis."""
    if log:
        return 10.0 ** (np.log10(lo) + frac * (np.log10(hi) - np.log10(lo)))
    return lo + frac * (hi - lo)


def _normalize(y, mode):
    """Rescale intensities; picture axes are usually in arbitrary units."""
    if mode == "None (use axis values)" or y.size == 0:
        return y
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y
    if mode == "Maximum = 100":
        peak = np.nanmax(np.abs(finite))
        return y * (100.0 / peak) if peak else y
    if mode == "Maximum = 1":
        peak = np.nanmax(np.abs(finite))
        return y / peak if peak else y
    # Minimum -> 0, maximum -> 100
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    return (y - lo) * (100.0 / (hi - lo)) if hi > lo else y - lo


def _xy_text(x, y, header):
    """Two-column .xy text with a single comment line, as elsewhere in the app."""
    body = "\n".join(f"{xi:.6f}  {yi:.4f}" for xi, yi in zip(x, y))
    return f"# {header}\n{body}\n"


def _safe_name(name: str) -> str:
    """File-name stem of the uploaded image, stripped of awkward characters."""
    stem = name.rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:60] or "image"


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _image_figure(rgb, height=620, dragmode="select"):
    """Empty plotly canvas with the uploaded image as its background."""
    h, w = rgb.shape[:2]
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=_image_data_uri(rgb), xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h, sizing="stretch", layer="below",
    ))
    fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False,
                     visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1,
                     constrain="domain")
    fig.update_layout(
        height=height, dragmode=dragmode, showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=FONT_SIZE),
        newselection=dict(line=dict(color="#ff2d95", width=2)),
        # Placing a point reruns the script; without this the view
        # would snap back to the whole image after every drag.
        uirevision="digitizer",
    )
    return fig


def _add_quad(fig, corners, active=None):
    """Draw the calibration quadrilateral and its numbered corners."""
    loop = np.vstack([corners, corners[:1]])
    fig.add_trace(go.Scatter(
        x=loop[:, 0], y=loop[:, 1], mode="lines",
        line=dict(color="#ff2d95", width=3), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=corners[:, 0], y=corners[:, 1], mode="markers+text",
        text=[label.split("-")[0][0].upper() + label.split("-")[1][0].upper()
              for label in CORNER_LABELS],
        textposition="top center",
        textfont=dict(size=16, color="#ff2d95"),
        marker=dict(size=14, color="#ff2d95", symbol="x-thin",
                    line=dict(width=3, color="#ff2d95")),
        hovertext=CORNER_LABELS, hoverinfo="text",
    ))
    if active is not None:
        fig.add_trace(go.Scatter(
            x=[corners[active, 0]], y=[corners[active, 1]], mode="markers",
            marker=dict(size=26, color="rgba(255,45,149,0.25)",
                        line=dict(width=2, color="#ff2d95")),
            hoverinfo="skip",
        ))


def _add_reference_grid(fig, corners):
    """Overlay the axis grid implied by the calibration.

    If the lines fall on the ticks that are actually printed in the picture,
    the calibration is right; if they drift, a corner needs nudging. This is
    the quickest way for the user to check their own numbers.
    """
    Hinv = np.linalg.inv(_homography(corners))
    xs, ys = [], []
    for frac_u in np.linspace(0, 1, 5):
        v = np.linspace(0, 1, 25)
        px, py = _from_uv(Hinv, np.full_like(v, frac_u), v)
        xs.extend(list(px) + [None])
        ys.extend(list(py) + [None])
    for frac_v in np.linspace(0, 1, 5):
        u = np.linspace(0, 1, 25)
        px, py = _from_uv(Hinv, u, np.full_like(u, frac_v))
        xs.extend(list(px) + [None])
        ys.extend(list(py) + [None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", hoverinfo="skip",
        line=dict(color="rgba(0,140,255,0.55)", width=1, dash="dot"),
    ))


def _style_data_figure(fig, x_title, y_title, height=560):
    """Shared styling for the extracted-curve plot."""
    fig.update_layout(
        height=height,
        font=dict(size=FONT_SIZE),
        hoverlabel=dict(font_size=18),
        margin=dict(l=80, r=20, t=30, b=70),
        xaxis=dict(title=dict(text=x_title, font=dict(size=AXIS_FONT)),
                   tickfont=dict(size=TICK_FONT)),
        yaxis=dict(title=dict(text=y_title, font=dict(size=AXIS_FONT)),
                   tickfont=dict(size=TICK_FONT)),
        legend=dict(font=dict(size=TICK_FONT)),
    )
    return fig


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Pick points: the corners of the plot area and the reference ticks
# --------------------------------------------------------------------------- #
AXIS_FROM_TICKS = "From two ticks in the picture"
AXIS_FROM_EDGES = "Values at the edges of the area"
TICK_MARKS = ["①", "②"]


def _corner_keys(i):
    return f"digi_c{i}x", f"digi_c{i}y"


def _tick_keys(axis, i):
    return f"digi_t{axis}{i}x", f"digi_t{axis}{i}y"


def _pick_targets(rect_mode, x_ticks, y_ticks):
    """Every point the user can place, in the order the picker walks through."""
    targets = []
    for i in ([0, 2] if rect_mode else [0, 1, 2, 3]):
        kx, ky = _corner_keys(i)
        targets.append({"label": CORNER_LABELS[i], "kx": kx, "ky": ky,
                        "kind": "corner", "index": i})
    for axis, wanted in (("x", x_ticks), ("y", y_ticks)):
        if not wanted:
            continue
        for i in (0, 1):
            kx, ky = _tick_keys(axis, i)
            targets.append({"label": f"{axis.upper()} tick {TICK_MARKS[i]}",
                            "kx": kx, "ky": ky, "kind": axis, "index": i})
    return targets


# Every pick point and axis value, whether or not it is on screen right now.
_KEPT_KEYS = ([f"digi_c{i}{a}" for i in range(4) for a in "xy"]
              + [f"digi_t{ax}{i}{a}" for ax in "xy" for i in (0, 1) for a in "xy"]
              + [f"digi_t{ax}v{i}" for ax in "xy" for i in (0, 1)]
              + ["digi_xlo", "digi_xhi", "digi_ylo", "digi_yhi"])


# Starting values for the controls the automatic setup also writes to. They
# are seeded into session state rather than passed to the widgets: Streamlit
# warns, in the app itself, about any widget that carries a default *and* has
# its value set programmatically.
_WIDGET_DEFAULTS = {
    "digi_minsat": 60, "digi_minval": 70, "digi_huetol": 18,
    "digi_minblob": 25, "digi_npoints": 1000,
    "digi_close": True, "digi_fill": True,
    "digi_txv0": 30.0, "digi_txv1": 100.0,
    "digi_tyv0": 0.0, "digi_tyv1": 100.0,
    "digi_xlo": 20.0, "digi_xhi": 80.0,
    "digi_ylo": 0.0, "digi_yhi": 100.0,
}


def _seed_defaults():
    """Give every such control its starting value, once."""
    for key, value in _WIDGET_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _remember_picks():
    """Carry the value of every pick point across the runs that hide it.

    Streamlit forgets a widget as soon as the script stops drawing it, so
    switching to two-corner mode and back, or reading an axis from its edges
    for a moment, would otherwise wipe the points that were off screen. Keeping
    a shadow copy makes those switches free.
    """
    for key in _KEPT_KEYS:
        if key in st.session_state:
            st.session_state[f"_kept_{key}"] = st.session_state[key]
        elif f"_kept_{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"_kept_{key}"]


def _read_corners(rect_mode):
    """Current corners as a (4, 2) array.

    In rectangle mode only two opposite corners are editable and the other two
    follow from them, which is all a straight screenshot needs.
    """
    pts = np.array([[st.session_state[_corner_keys(i)[0]],
                     st.session_state[_corner_keys(i)[1]]] for i in range(4)],
                   dtype=float)
    if rect_mode:
        left, bottom = pts[0]
        right, top = pts[2]
        pts = np.array([[left, bottom], [right, bottom],
                        [right, top], [left, top]], dtype=float)
    return pts


def _set_corner(i, x, y, rect_mode):
    """Move one corner, keeping the box consistent in rectangle mode."""
    kx, ky = _corner_keys(i)
    st.session_state[kx] = float(round(x, 1))
    st.session_state[ky] = float(round(y, 1))
    if rect_mode:
        pts = _read_corners(True)
        for j in range(4):
            jx, jy = _corner_keys(j)
            st.session_state[jx] = float(round(pts[j, 0], 1))
            st.session_state[jy] = float(round(pts[j, 1], 1))


def _place_default_ticks(corners):
    """Park the reference ticks a fifth of the way in along their own axis."""
    bl, br, tr, tl = corners
    for i, f in enumerate((0.2, 0.8)):
        kx, ky = _tick_keys("x", i)
        st.session_state[kx] = float(round(bl[0] + f * (br[0] - bl[0]), 1))
        st.session_state[ky] = float(round(bl[1] + f * (br[1] - bl[1]), 1))
        kx, ky = _tick_keys("y", i)
        st.session_state[kx] = float(round(bl[0] + f * (tl[0] - bl[0]), 1))
        st.session_state[ky] = float(round(bl[1] + f * (tl[1] - bl[1]), 1))


def _write_corners(quad, rect_mode=False):
    """Store a whole quadrilateral, and move the reference ticks with it."""
    for i in range(4):
        kx, ky = _corner_keys(i)
        st.session_state[kx] = float(round(quad[i][0], 1))
        st.session_state[ky] = float(round(quad[i][1], 1))
    st.session_state["digi_shape"] = SHAPE_STRAIGHT if rect_mode else SHAPE_TILTED
    st.session_state["digi_active"] = CORNER_LABELS[0]
    st.session_state["digi_ticks_found"] = []
    _place_default_ticks(np.asarray(quad, dtype=float))


def _reset_corners(rgb):
    """Fall back to a plain rectangle covering the middle of the picture."""
    h, w = rgb.shape[:2]
    frame = _detect_frame(rgb)
    left, right, top, bottom = frame or (0.12 * w, 0.88 * w, 0.12 * h, 0.88 * h)
    _write_corners([(left, bottom), (right, bottom), (right, top), (left, top)],
                   rect_mode=True)


def _selection_state():
    """The picking chart's current selection, if it has one."""
    return st.session_state.get("digi_pick_chart")


def _apply_pending_pick(width, height):
    """Move the active pick point onto the box the user drew on the image.

    This has to run before any calibration widget is created: Streamlit refuses
    to change a widget's value once that widget exists in the current run, so
    handling the selection next to the chart itself would only raise. The
    selection also lingers in the widget after it has been used, hence the
    stamp — a given box is acted on exactly once.
    """
    boxes = ((_selection_state() or {}).get("selection") or {}).get("box") or []
    if not boxes:
        return
    bx = boxes[0].get("x") or []
    by = boxes[0].get("y") or []
    if len(bx) < 2 or len(by) < 2:
        return
    stamp = tuple(round(float(c), 2) for c in (bx[0], bx[1], by[0], by[1]))
    if st.session_state.get("digi_last_box") == stamp:
        return
    st.session_state["digi_last_box"] = stamp

    rect_mode = st.session_state.get("digi_shape", SHAPE_STRAIGHT) == SHAPE_STRAIGHT
    targets = _pick_targets(
        rect_mode,
        st.session_state.get("digi_xmode", AXIS_FROM_TICKS) == AXIS_FROM_TICKS,
        st.session_state.get("digi_ymode", AXIS_FROM_EDGES) == AXIS_FROM_TICKS)
    labels = [t["label"] for t in targets]
    active = st.session_state.get("digi_active")
    idx = labels.index(active) if active in labels else 0

    x = float(np.clip(np.mean(bx), 0, width))
    y = float(np.clip(np.mean(by), 0, height))
    target = targets[idx]
    if target["kind"] == "corner":
        _set_corner(target["index"], x, y, rect_mode)
    else:
        st.session_state[target["kx"]] = round(x, 1)
        st.session_state[target["ky"]] = round(y, 1)
        found = set(st.session_state.get("digi_ticks_found", []))
        found.add(target["kind"].upper())
        st.session_state["digi_ticks_found"] = sorted(found)
    st.session_state["digi_active"] = labels[(idx + 1) % len(labels)]


# --------------------------------------------------------------------------- #
#  Axis calibration from two reference ticks
# --------------------------------------------------------------------------- #
def _range_from_ticks(H, axis, values):
    """Turn two marked ticks into the values at the two edges of the area.

    The user knows what the ticks say, not what the invisible edge of the plot
    frame would read. Both ticks are projected into the straightened plot, and
    the value along that axis is linear there, so two of them fix the whole
    range.
    """
    fracs = []
    for i in (0, 1):
        kx, ky = _tick_keys(axis, i)
        u, v, d = _to_uv(H, np.array([st.session_state[kx]]),
                         np.array([st.session_state[ky]]))
        fracs.append(float(u[0] if axis == "x" else v[0]))
    (f0, f1), (v0, v1) = fracs, values
    if abs(f1 - f0) < 0.05:
        return None
    span = (v1 - v0) / (f1 - f0)
    return v0 - f0 * span, v0 + (1 - f0) * span


# --------------------------------------------------------------------------- #
#  Automatic setup
# --------------------------------------------------------------------------- #
def _round_to(value, step, lo, hi):
    """Snap a computed setting onto a slider's own grid."""
    return int(min(max(round(value / step) * step, lo), hi))


def _auto_configure(rgb, hsv, data=None):
    """Set the plot area and every extraction setting from the image itself.

    Runs as a button callback (and once for each new upload), which is the only
    place where widget values may still be assigned: callbacks execute before
    the script rebuilds its widgets.
    """
    h, w = rgb.shape[:2]
    quad = _detect_plot_quad(rgb, hsv)
    if quad is None:
        st.session_state["digi_auto_msg"] = (
            "warning", "No plot frame found — place the corners of the plot area "
                       "yourself by dragging a box on each.")
        return

    # Square within a pixel or so? Then the simpler two-corner mode is enough.
    rect = (abs(quad[0][1] - quad[1][1]) < 1.5 and abs(quad[2][1] - quad[3][1]) < 1.5
            and abs(quad[0][0] - quad[3][0]) < 1.5 and abs(quad[1][0] - quad[2][0]) < 1.5)
    _write_corners(quad, rect_mode=rect)

    # Park the two reference marks on real ticks, and read what those ticks say
    # where the printing allows it. A mark left at a guessed spot, or a value
    # taken on trust, silently stretches the whole axis — so the reading is only
    # used when the numbers agree with each other, and is shown for checking.
    quad = np.asarray(quad, dtype=float)
    gray = _reading_image(data) if data else None
    scale = gray.shape[1] / w if gray is not None else 1.0
    marked, read = [], []
    for axis in ("x", "y"):
        anchors = _tick_anchors(rgb, hsv, quad, axis)
        if not anchors:
            continue
        for i, point in enumerate(anchors):
            kx, ky = _tick_keys(axis, i)
            st.session_state[kx] = float(round(point[0], 1))
            st.session_state[ky] = float(round(point[1], 1))
        marked.append(axis.upper())
        if gray is None:
            continue
        matrix = _homography(quad)
        u, v, _ = _to_uv(matrix, np.array([p[0] for p in anchors]),
                         np.array([p[1] for p in anchors]))
        fractions = list(u if axis == "x" else v)
        try:
            values = _read_tick_values(gray, quad, axis, fractions, scale)
        except Exception:                  # reading is a convenience, never a blocker
            values = None
        if values:
            st.session_state[f"digi_t{axis}v0"] = float(values[0])
            st.session_state[f"digi_t{axis}v1"] = float(values[1])
            read.append(f"{axis.upper()} ticks read as **{values[0]:g}** and "
                        f"**{values[1]:g}**")
    st.session_state["digi_ticks_found"] = marked

    u, v, inside = _uv_maps((h, w), tuple(np.asarray(quad).round(2).ravel().tolist()))
    notes = []

    candidates = _colour_candidates(rgb, hsv, inside, 55, 65)
    if candidates:
        top = candidates[0]
        mask = _colour_mask(hsv, inside, top["hue"], 25, 55, 65)
        val = hsv[..., 2][mask]
        spread = _hue_distance(hsv[..., 0][mask], round(top["hue"]))
        # Saturation separates the drawn curves from paper, axes and text, and
        # where that line falls is a property of this picture, not a guess.
        st.session_state["digi_minsat"] = _round_to(
            _otsu(hsv[..., 1][inside]), 5, 25, 120)
        # Brightness keeps black lettering out; the curve's own mid-tone is the
        # yardstick.
        st.session_state["digi_minval"] = _round_to(
            0.7 * np.median(val), 5, 30, 130)
        st.session_state["digi_huetol"] = _round_to(
            np.percentile(spread, 97) * 360 / 256 + 3, 1, 8, 45)
        # Blob threshold from the curve itself: a fraction of its own length,
        # which drops legend keys and speckles but keeps dashes.
        lab, n = label(binary_dilation(mask, np.ones((3, 3), bool)))
        largest = np.bincount(lab.ravel())[1:].max() if n else 0
        st.session_state["digi_minblob"] = _round_to(largest / 200, 5, 10, 400)
        for i in range(12):
            st.session_state[f"digi_use_{i}"] = True
        notes.append(f"{len(candidates)} curve"
                     f"{'s' if len(candidates) > 1 else ''} found")
    else:
        notes.append("no coloured curve found — try the black/grey option")

    width_px = float(np.hypot(*(np.asarray(quad[1]) - np.asarray(quad[0]))))
    st.session_state["digi_npoints"] = _round_to(width_px, 50, 100, 4000)
    st.session_state["digi_reducer"] = MODE_PEAKS
    st.session_state["digi_norm"] = "None (use axis values)"
    st.session_state["digi_close"] = True
    st.session_state["digi_fill"] = True
    st.session_state["digi_largest"] = False
    st.session_state["digi_smooth"] = False
    st.session_state["digi_dark"] = not candidates

    if read:
        axis_note = ", ".join(read) + " — check them against the picture."
    elif "X" in marked:
        axis_note = "Ticks ① and ② are marked — type what they say."
    else:
        axis_note = "**X ticks not found** — drag a box onto two you can read."
    st.session_state["digi_auto_msg"] = (
        "success",
        "Plot area" + ("" if rect else " (tilted)") + " and " + ", ".join(notes)
        + ". " + axis_note)


# --------------------------------------------------------------------------- #
#  Overlay of the pick points
# --------------------------------------------------------------------------- #
def _add_picks(fig, targets, active_label):
    """Draw the reference ticks on top of the calibration quadrilateral."""
    xs, ys, texts = [], [], []
    for t in targets:
        if t["kind"] == "corner":
            continue
        xs.append(st.session_state[t["kx"]])
        ys.append(st.session_state[t["ky"]])
        texts.append(t["label"].replace(" tick", ""))
    if not xs:
        return
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="bottom center",
        textfont=dict(size=15, color="#00a06a"),
        marker=dict(size=15, color="rgba(0,160,106,0.35)", symbol="circle",
                    line=dict(width=2, color="#00a06a")),
        hovertext=[t["label"] for t in targets if t["kind"] != "corner"],
        hoverinfo="text",
    ))
    for t in targets:
        if t["label"] == active_label and t["kind"] != "corner":
            fig.add_trace(go.Scatter(
                x=[st.session_state[t["kx"]]], y=[st.session_state[t["ky"]]],
                mode="markers", hoverinfo="skip",
                marker=dict(size=28, color="rgba(0,160,106,0.2)",
                            line=dict(width=2, color="#00a06a"))))


# --------------------------------------------------------------------------- #
#  Main section
# --------------------------------------------------------------------------- #
def _pick_option(key, options):
    """Keep a radio's remembered choice among the options it is offered."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _seed_curve_toggles(count):
    """Every curve that was found is traced unless the user turns it off."""
    for i in range(count):
        if f"digi_use_{i}" not in st.session_state:
            st.session_state[f"digi_use_{i}"] = True


def _curve_toggles(candidates, drawn):
    """A colour chip per curve found, below the plots, to switch each on or off.

    Always drawn, even when every curve is off: the switches are the only way
    back, so they must never be what disappears.
    """
    if not candidates:
        return
    traced = {r["index"]: r for r in drawn if r["index"] is not None}
    per_row = min(len(candidates), 6)
    for start in range(0, len(candidates), per_row):
        cols = st.columns(per_row)
        for j, cand in enumerate(candidates[start:start + per_row]):
            i = start + j
            colour = "#%02x%02x%02x" % cand["rgb"]
            on = bool(st.session_state.get(f"digi_use_{i}", True))
            got = traced.get(i)
            note = (f"{got['coverage'] * 100:.0f}% traced" if got
                    else ("off" if not on else "—"))
            cols[j].markdown(
                f"<div style='height:28px;border-radius:6px;background:{colour};"
                f"border:1px solid rgba(0,0,0,.25);"
                f"opacity:{'1' if on else '0.2'}'></div>"
                f"<div style='font-size:0.75rem;color:#666;text-align:center;"
                f"margin-bottom:-8px'>{note}</div>",
                unsafe_allow_html=True)
            cols[j].checkbox(f"Curve {i + 1}", key=f"digi_use_{i}")


def run_image_digitizer_section():
    """Streamlit UI: image or photo of a plot in, .xy files out."""
    apply_button_style(["digi_remove"])
    st.title("🖼️ Image / Photo → .xy")
    st.caption("Trace the curves in a picture of a plot — a figure from a paper, a "
               "screenshot, or a photo taken at an angle — and save them as .xy files.")

    if "digi_uploader_key" not in st.session_state:
        st.session_state.digi_uploader_key = 0

    def _clear_image():
        st.session_state.digi_uploader_key += 1
        st.session_state.pop("digi_signature", None)

    uploaded = st.file_uploader(
        "Image of a plot (PNG, JPG, BMP, TIFF, WEBP — max 10 MB)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
        key=f"digi_uploader_{st.session_state.digi_uploader_key}",
    )
    if uploaded is None:
        st.info("The plot area, the curves and the axis ticks are found for you. "
                "All you type is what two of the ticks say.")
        return

    data = uploaded.getvalue()
    if not data:
        st.error("The uploaded file is empty.")
        return
    try:
        rgb, hsv = _prepare_image(data)
    except Exception as exc:                       # unreadable / corrupt image
        st.error(f"Could not read this image: {exc}")
        return
    height, width = rgb.shape[:2]

    # All three change widget values, so they run before those widgets exist.
    signature = f"{uploaded.name}:{len(data)}"
    if st.session_state.get("digi_signature") != signature:
        st.session_state["digi_signature"] = signature
        st.session_state.pop("digi_last_box", None)
        _reset_corners(rgb)
        with st.spinner("Reading the picture…"):
            try:
                _auto_configure(rgb, hsv, data)
            except Exception:                      # never block on a guess
                st.session_state["digi_auto_msg"] = (
                    "warning", "Automatic setup did not succeed — place the "
                               "corners of the plot area yourself.")
    _remember_picks()
    _seed_defaults()
    _apply_pending_pick(width, height)

    b1, b2, b3 = st.columns(3)
    b1.button("✨ Auto-detect", key="digi_auto_btn", width="stretch",
              type="primary", on_click=_auto_configure,
              args=(rgb, hsv, data))
    b2.button("↺ Reset area", key="digi_reset_btn", width="stretch",
              type="primary", on_click=_reset_corners, args=(rgb,))
    b3.button("🗑️ Remove image", key="digi_remove", width="stretch",
              type="primary", on_click=_clear_image)

    kind, message = st.session_state.get("digi_auto_msg", (None, None))
    if message:
        (st.success if kind == "success" else st.warning)(message)

    # ── Plot area and axis values ────────────────────────────────────────
    _pick_option("digi_shape", [SHAPE_STRAIGHT, SHAPE_TILTED])
    _pick_option("digi_xmode", [AXIS_FROM_TICKS, AXIS_FROM_EDGES])
    _pick_option("digi_ymode", [AXIS_FROM_EDGES, AXIS_FROM_TICKS])
    rect_mode = st.session_state["digi_shape"] == SHAPE_STRAIGHT
    x_ticks = st.session_state["digi_xmode"] == AXIS_FROM_TICKS
    y_ticks = st.session_state["digi_ymode"] == AXIS_FROM_TICKS
    targets = _pick_targets(rect_mode, x_ticks, y_ticks)
    labels = [t["label"] for t in targets]
    _pick_option("digi_active", labels)

    image_col, axis_col = st.columns([1.35, 1])

    with axis_col:
        st.markdown("##### Axis values")
        x_mode = st.radio("X", [AXIS_FROM_TICKS, AXIS_FROM_EDGES],
                          key="digi_xmode", horizontal=True)
        c1, c2 = st.columns(2)
        if x_mode == AXIS_FROM_TICKS:
            x_values = (c1.number_input(f"X at {TICK_MARKS[0]}", step=1.0,
                                        format="%.4f", key="digi_txv0"),
                        c2.number_input(f"X at {TICK_MARKS[1]}", step=1.0,
                                        format="%.4f", key="digi_txv1"))
        else:
            x_edges = (c1.number_input("X left", step=1.0, format="%.4f",
                                       key="digi_xlo"),
                       c2.number_input("X right", step=1.0, format="%.4f",
                                       key="digi_xhi"))
        y_mode = st.radio("Y", [AXIS_FROM_EDGES, AXIS_FROM_TICKS],
                          key="digi_ymode", horizontal=True)
        c3, c4 = st.columns(2)
        if y_mode == AXIS_FROM_TICKS:
            y_values = (c3.number_input(f"Y at {TICK_MARKS[0]}", step=1.0,
                                        format="%.4f", key="digi_tyv0"),
                        c4.number_input(f"Y at {TICK_MARKS[1]}", step=1.0,
                                        format="%.4f", key="digi_tyv1"))
        else:
            y_edges = (c3.number_input("Y bottom", step=1.0, format="%.4f",
                                       key="digi_ylo"),
                       c4.number_input("Y top", step=1.0, format="%.4f",
                                       key="digi_yhi"))

        with st.expander("Plot area and axis options"):
            st.radio("Shape of the plot area", [SHAPE_STRAIGHT, SHAPE_TILTED],
                     key="digi_shape")
            l1, l2 = st.columns(2)
            log_x = l1.checkbox("Log X", key="digi_logx")
            log_y = l2.checkbox("Log Y", key="digi_logy")
            n1, n2 = st.columns(2)
            x_label = n1.text_input("X name", value="2Theta (deg)", key="digi_xname")
            y_label = n2.text_input("Y name", value="Intensity", key="digi_yname")
            st.radio("Point to place next", labels, key="digi_active")
            st.caption("Drag a box on the image to move the selected point.")
            for t in targets:
                p1, p2, p3 = st.columns([1.2, 1, 1])
                p1.markdown(f"<div style='padding-top:0.5rem;font-size:0.85rem'>"
                            f"{t['label']}</div>", unsafe_allow_html=True)
                p2.number_input("x", min_value=0.0, max_value=float(width),
                                step=1.0, key=t["kx"], label_visibility="collapsed")
                p3.number_input("y", min_value=0.0, max_value=float(height),
                                step=1.0, key=t["ky"], label_visibility="collapsed")

    corners = _read_corners(rect_mode)
    if rect_mode:
        # Keep the two corners this mode hides in step with the box on screen.
        for j in (1, 3):
            jx, jy = _corner_keys(j)
            st.session_state[jx] = float(round(corners[j, 0], 1))
            st.session_state[jy] = float(round(corners[j, 1], 1))

    with image_col:
        if not _is_convex(corners) or _quad_area(corners) < 500:
            st.error("The corners do not form a box — they must go bottom-left, "
                     "bottom-right, top-right, top-left. Use **↺ Reset area**.")
            return
        fig = _image_figure(rgb)
        _add_reference_grid(fig, corners)
        _add_quad(fig, corners, CORNER_LABELS.index(st.session_state["digi_active"])
                  if st.session_state["digi_active"] in CORNER_LABELS else None)
        _add_picks(fig, targets, st.session_state["digi_active"])
        st.plotly_chart(fig, width="stretch", key="digi_pick_chart",
                        on_select="rerun", selection_mode=("box",),
                        config={"displaylogo": False, "scrollZoom": True,
                                "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"]})

    found = st.session_state.get("digi_ticks_found", [])
    for axis, uses_ticks in (("X", x_ticks), ("Y", y_ticks)):
        if uses_ticks and axis not in found:
            st.warning(f"The {axis} ticks were not found — ① and ② are at guessed "
                       f"positions, so drag a box onto the two {axis} ticks you are "
                       "typing values for.")

    H = _homography(corners)
    if x_ticks:
        span = _range_from_ticks(H, "x", x_values)
        if span is None:
            st.error("The two X ticks are too close together.")
            return
        x_lo, x_hi = span
    else:
        x_lo, x_hi = x_edges
    if y_ticks:
        span = _range_from_ticks(H, "y", y_values)
        if span is None:
            st.error("The two Y ticks are too close together.")
            return
        y_lo, y_hi = span
    else:
        y_lo, y_hi = y_edges

    axis_col.caption(f"X **{x_lo:.4g} → {x_hi:.4g}**, Y **{y_lo:.4g} → {y_hi:.4g}** "
                     "across the marked area.")

    if log_x and (x_lo <= 0 or x_hi <= 0):
        st.error("A logarithmic X axis needs both X values to be positive.")
        return
    if log_y and (y_lo <= 0 or y_hi <= 0):
        st.error("A logarithmic Y axis needs both Y values to be positive.")
        return
    if x_lo == x_hi or y_lo == y_hi:
        st.error("The two X values (and the two Y values) must differ.")
        return

    u_map, v_map, inside = _uv_maps((height, width),
                                    tuple(corners.round(2).ravel().tolist()))
    if inside.sum() < 500:
        st.error("The marked plot area is too small.")
        return

    # ── Curves ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Colour and extraction settings"):
        s1, s2, s3 = st.columns(3)
        min_sat = s1.slider("Minimum saturation", min_value=10, max_value=200,
                            step=5, key="digi_minsat",
                            help="How colourful a pixel must be to count as curve.")
        min_val = s2.slider("Minimum brightness", min_value=10, max_value=200,
                            step=5, key="digi_minval",
                            help="Keeps black axes and lettering out.")
        hue_tol = s3.slider("Colour tolerance (°)", min_value=3, max_value=60,
                            step=1, key="digi_huetol")
        e1, e2, e3 = st.columns(3)
        n_points = e1.slider("Points per curve", min_value=100, max_value=4000,
                             step=50, key="digi_npoints")
        _pick_option("digi_reducer", EXTRACTION_MODES)
        mode = e2.selectbox("Value per x step", EXTRACTION_MODES, key="digi_reducer",
                            help="At a sharp peak one step of ink runs from the "
                                 "baseline to the tip; peak-preserving reads the "
                                 "top there and the middle elsewhere.")
        norm_mode = e3.selectbox("Normalisation",
                                 ["None (use axis values)", "Maximum = 100",
                                  "Maximum = 1", "Minimum → 0, maximum → 100"],
                                 key="digi_norm")
        c1, c2, c3, c4 = st.columns(4)
        min_blob = c1.slider("Ignore blobs under (px)", min_value=0, max_value=400,
                             step=5, key="digi_minblob")
        keep_largest = c2.checkbox("Largest piece only", key="digi_largest")
        close_gaps = c3.checkbox("Close small gaps", key="digi_close")
        fill_gaps = c4.checkbox("Interpolate gaps", key="digi_fill")
        d1, d2, d3 = st.columns(3)
        use_dark = d1.checkbox("Trace a black / grey curve", key="digi_dark")
        dark_val = d2.slider("Darkness threshold", 20, 200, 110, 5,
                             key="digi_darkval", disabled=not use_dark)
        dark_sat = d3.slider("Max saturation for grey", 10, 200, 90, 5,
                             key="digi_darksat", disabled=not use_dark)
        s4, s5 = st.columns([1, 2])
        smooth = s4.checkbox("Smooth", key="digi_smooth")
        smooth_win = s5.slider("Smoothing window", 5, 201, 21, 2,
                               key="digi_smoothwin", disabled=not smooth)

    candidates = _colour_candidates(rgb, hsv, inside, min_sat, min_val)
    _seed_curve_toggles(len(candidates))
    wanted = [i for i in range(len(candidates))
              if st.session_state.get(f"digi_use_{i}", True)]

    specs = [{"name": f"curve{i + 1}", "hex": "#%02x%02x%02x" % candidates[i]["rgb"],
              "hue": candidates[i]["hue"], "kind": "hue", "index": i}
             for i in wanted]
    if use_dark:
        specs.append({"name": "dark", "hex": "#333333", "kind": "dark",
                      "index": None})

    if not candidates and not use_dark:
        st.warning("No coloured curve was found inside the marked area. Lower the "
                   "saturation threshold, or switch on the black/grey option.")
        return

    native = int(round(max(np.hypot(*(corners[1] - corners[0])),
                           np.hypot(*(corners[2] - corners[3])))))
    u_centres = (np.arange(n_points) + 0.5) / n_points
    hue_masks = _curve_masks(hsv, inside, [c["hue"] for c in candidates],
                             hue_tol, min_sat, min_val)

    results = []
    for spec in specs:
        raw = (_dark_mask(hsv, inside, dark_val, dark_sat) if spec["kind"] == "dark"
               else hue_masks[spec["index"]])
        mask = _clean_mask(raw, min_blob, keep_largest, close_gaps)
        values, coverage = _extract_curve(mask, u_map, v_map, n_points, mode,
                                          n_fine=native)
        holes = 0
        if fill_gaps:
            values, holes = _fill_gaps(u_centres, values)
        good = np.isfinite(values)
        if good.sum() < 2:
            results.append({**spec, "empty": True})
            continue
        y_vals = _to_axis(values[good], y_lo, y_hi, log_y)
        if smooth and y_vals.size > smooth_win:
            win = smooth_win if smooth_win % 2 else smooth_win + 1
            y_vals = savgol_filter(y_vals, win, 3)
        results.append({**spec, "empty": False,
                        "x": _to_axis(u_centres[good], x_lo, x_hi, log_x),
                        "y": _normalize(y_vals, norm_mode),
                        "u": u_centres[good], "v": values[good],
                        "coverage": coverage, "holes": holes})

    drawn = [r for r in results if not r["empty"]]

    curve_col, check_col = st.columns(2)
    with curve_col:
        data_fig = go.Figure()
        for r in drawn:
            data_fig.add_trace(go.Scatter(x=r["x"], y=r["y"], mode="lines",
                                          name=r["name"],
                                          line=dict(color=r["hex"], width=2)))
        y_title = y_label if norm_mode.startswith("None") else \
            f"{y_label} ({norm_mode.lower()})"
        st.plotly_chart(_style_data_figure(data_fig, x_label, y_title, height=460),
                        width="stretch", key="digi_data_chart",
                        config={"displaylogo": False, "scrollZoom": True})
        if not drawn:
            st.info("Every curve is switched off — turn one back on below.")
    with check_col:
        check_fig = _image_figure(rgb, height=460, dragmode="pan")
        Hinv = np.linalg.inv(H)
        for r in drawn:
            px, py = _from_uv(Hinv, r["u"], r["v"])
            check_fig.add_trace(go.Scatter(x=px, y=py, mode="lines", opacity=0.35,
                                           line=dict(color="#000000", width=4),
                                           hoverinfo="skip", name=r["name"]))
            check_fig.add_trace(go.Scatter(x=px, y=py, mode="lines",
                                           line=dict(color="#ffffff", width=1.5),
                                           hoverinfo="skip", name=r["name"]))
        _add_quad(check_fig, corners)
        st.plotly_chart(check_fig, width="stretch", key="digi_check_chart",
                        config={"displaylogo": False, "scrollZoom": True})

    _curve_toggles(candidates, drawn)
    if not drawn:
        return

    # ── Download ─────────────────────────────────────────────────────────
    base = _safe_name(uploaded.name)
    files = {}
    for r in drawn:
        header = f"{x_label}  {y_label} — digitized from {uploaded.name} ({r['hex']})"
        files[f"{base}_{r['name']}.xy"] = _xy_text(r["x"], r["y"], header)

    if len(files) > 1:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, text in files.items():
                zf.writestr(name, text)
        st.download_button(f"📦 All {len(files)} curves (.zip)", data=buf.getvalue(),
                           file_name=f"{base}_digitized.zip", mime="application/zip",
                           key="digi_dl_zip", width="stretch", type="primary",
                           on_click=log_download, args=(f"{TOOL_NAME} (zip)",))

    per_row = min(len(files), 4)
    cols = st.columns(per_row)
    for i, (name, text) in enumerate(files.items()):
        cols[i % per_row].download_button(
            f"💾 {drawn[i]['name']}", data=text, file_name=name, mime="text/plain",
            key=f"digi_dl_{i}", width="stretch",
            on_click=log_download, args=(TOOL_NAME,))
