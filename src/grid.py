"""Grid reconstruction v2 for the TSA pipeline.

Turns Doc-UFCN odd/even band polygons into a complete tiling of row bands and
column strips, robust to fragmented and missing band predictions. Replaces the
legacy absolute-size gate (w>2000 or h>2000) and single-band gap fill in
TableSegment. Pure geometry: no model or I/O dependencies.
"""
from statistics import median
from typing import Dict, List, Tuple

import cv2
import numpy as np

Band = Tuple[float, float]


def polygons_to_intervals(polygons: List[dict], axis: str) -> List[Band]:
    """Project each polygon's bbox onto the band axis ('y' rows, 'x' cols)."""
    i = 1 if axis == "y" else 0
    out = []
    for p in polygons:
        coords = [pt[i] for pt in p["polygon"]]
        out.append((float(min(coords)), float(max(coords))))
    return sorted(out)


def _intervals_with_orth(polygons: List[dict], axis: str):
    """[(band interval, orthogonal extent interval)] per polygon bbox."""
    i, j = (1, 0) if axis == "y" else (0, 1)
    out = []
    for p in polygons:
        a = [pt[i] for pt in p["polygon"]]
        o = [pt[j] for pt in p["polygon"]]
        out.append(((float(min(a)), float(max(a))),
                    (float(min(o)), float(max(o)))))
    return sorted(out)


def _merge_with_orth(items, merge_gap: float):
    """merge_intervals on (band, orth) pairs; orth ranges union on merge."""
    if not items:
        return []
    items = sorted(items)
    out = [items[0]]
    for (s, e), (o0, o1) in items[1:]:
        (ps, pe), (po0, po1) = out[-1]
        if s - pe <= merge_gap:
            out[-1] = ((ps, max(pe, e)), (min(po0, o0), max(po1, o1)))
        else:
            out.append(((s, e), (o0, o1)))
    return out


def merge_intervals(intervals: List[Band], merge_gap: float) -> List[Band]:
    """Merge intervals that overlap or sit closer than merge_gap (same band)."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s - pe <= merge_gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def filter_small(intervals: List[Band], min_frac: float) -> List[Band]:
    """Drop bands smaller than min_frac * median band size; keep at least 2."""
    if len(intervals) <= 2:
        return list(intervals)
    m = median(e - s for s, e in intervals)
    kept = [(s, e) for s, e in intervals if (e - s) >= min_frac * m]
    return kept if len(kept) >= 2 else list(intervals)


def resolve_overlaps(intervals: List[Band]) -> List[Band]:
    """Cut overlapping neighbours at the midpoint of their overlap."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s < pe:
            mid = (s + pe) / 2.0
            out[-1] = (ps, mid)
            s = mid
        if e > s:
            out.append((s, e))
    return out


def fill_gaps(intervals: List[Band], fill_trigger: float) -> List[Band]:
    """Make bands tile their extent: big gaps get round(gap/median) bands,
    small gaps are closed by extending both neighbours to the gap midpoint."""
    if len(intervals) < 2:
        return list(intervals)
    intervals = sorted(intervals)
    m = median(e - s for s, e in intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        gap = s - pe
        if gap <= 0:
            pass
        elif gap > fill_trigger * m:
            n = max(1, int(round(gap / m)))
            step = gap / n
            for k in range(n):
                out.append((pe + k * step, pe + (k + 1) * step))
        else:
            mid = pe + gap / 2.0
            out[-1] = (ps, mid)
            s = mid
        out.append((s, e))
    return out


def regularise(polygon_groups: List[List[dict]], axis: str, *,
               merge_gap_frac: float = 0.05,
               min_size_frac: float = 0.60,
               fill_trigger_frac: float = 0.40,
               min_orth_frac: float = 0.50) -> List[Band]:
    """polygon_groups: one list of polygons per parity class (odd, even).
    Fragments are merged WITHIN each group (fragments of one band overlap in
    projection; distinct same-parity bands are ~a full band apart), then the
    groups are pooled for overlap resolution and gap filling. Pooling before
    merging would fuse adjacent odd/even bands, which genuinely touch.

    min_orth_frac guards on the ORTHOGONAL extent: a real band spans most of
    the table's other axis, stray blob detections do not, and a single stray
    far outside the table makes fill_gaps back-fill the whole false extent
    with phantom bands. Bands whose pooled orthogonal span is below
    min_orth_frac * (largest span) are dropped. Replaces the legacy absolute
    w/h > 2000 gate; applied after within-parity merging so fragments of one
    band pool their extents first."""
    per_group = []
    for polys in polygon_groups:
        items = _intervals_with_orth(polys, axis)
        if not items:
            continue
        m = median(e - s for (s, e), _ in items)
        per_group += _merge_with_orth(items, merge_gap=merge_gap_frac * m)
    if not per_group:
        return []
    max_orth = max(o1 - o0 for _, (o0, o1) in per_group)
    ivs = sorted(iv for iv, (o0, o1) in per_group
                 if (o1 - o0) >= min_orth_frac * max_orth)
    ivs = filter_small(ivs, min_frac=min_size_frac)
    ivs = resolve_overlaps(ivs)
    ivs = fill_gaps(ivs, fill_trigger=fill_trigger_frac)
    return ivs


def make_grid(row_bands: List[Band], col_bands: List[Band]) -> Dict[str, tuple]:
    """Cells named col{c}_row{r} (1-indexed), value (x, y, w, h)."""
    cells = {}
    for r, (y0, y1) in enumerate(sorted(row_bands), start=1):
        for c, (x0, x1) in enumerate(sorted(col_bands), start=1):
            cells[f"col{c}_row{r}"] = (x0, y0, x1 - x0, y1 - y0)
    return cells


def _band_index(bands: List[Band], v: float) -> int:
    """Index (1-based) of the band containing v, clamped to the extent."""
    bands = sorted(bands)
    v = min(max(v, bands[0][0]), bands[-1][1] - 1e-9)
    for i, (s, e) in enumerate(bands, start=1):
        if s <= v < e or (i == len(bands) and v <= e):
            return i
    return len(bands)


def assign_cell(row_bands: List[Band], col_bands: List[Band],
                cx: float, cy: float) -> str:
    """Centre-point cell lookup; points outside the table clamp to the edge."""
    return f"col{_band_index(col_bands, cx)}_row{_band_index(row_bands, cy)}"


def find_seam(binary: np.ndarray, boundary: int, window: int, axis: str,
              dilate_px: int) -> int:
    """Lowest-ink seam within +/-window of boundary. axis='x': vertical cut
    (dilate horizontally, column profile); axis='y': horizontal cut."""
    if dilate_px > 1:
        kernel = np.ones((1, dilate_px), np.uint8) if axis == "x" else \
                 np.ones((dilate_px, 1), np.uint8)
        binary = cv2.dilate(binary, kernel)
    profile = binary.sum(axis=0) if axis == "x" else binary.sum(axis=1)
    n = len(profile)
    lo = max(0, int(boundary) - window)
    hi = min(n, int(boundary) + window + 1)
    if hi <= lo:
        return int(np.clip(boundary, 0, max(0, n - 1)))
    seg = profile[lo:hi].astype(np.int64)
    best = seg.min()
    # among minima, take the one closest to the geometric boundary
    idxs = np.flatnonzero(seg == best) + lo
    return int(idxs[np.argmin(np.abs(idxs - boundary))])


def _cut_positions(start: float, size: float, bands: List[Band],
                   span_frac: float, min_piece_px: float, *, frac_of_band: bool) -> List[float]:
    """Internal band edges crossing [start, start+size] worth cutting at."""
    bands = sorted(bands)
    end = start + size
    cuts = []
    for s, e in bands[:-1]:
        edge = e  # internal boundary between this band and the next
        if start + min_piece_px < edge < end - min_piece_px:
            over = end - edge  # extension beyond the boundary
            ref_band = bands[_band_index(bands, edge + 1e-6) - 1]
            ref = (ref_band[1] - ref_band[0]) if frac_of_band else size
            threshold = span_frac * ref
            before = edge - start
            if min(before, over) > max(threshold, min_piece_px):
                cuts.append(edge)
    return cuts


def _drop_sliver_cuts(positions: List[float], min_piece_px: float) -> List[float]:
    """positions sorted, includes both box edges; drop interior cuts that
    would create a piece narrower than min_piece_px."""
    if len(positions) < 2:
        return list(positions)
    kept = [positions[0]]
    for p in positions[1:-1]:
        if p - kept[-1] >= min_piece_px and positions[-1] - p >= min_piece_px:
            kept.append(p)
    kept.append(positions[-1])
    return kept


def split_line(line_box, row_bands: List[Band], col_bands: List[Band],
               binary_crop: np.ndarray, *,
               col_span_frac: float = 0.10, row_span_frac: float = 0.40,
               window_px: int = 40, dilate_px: int = 7,
               min_piece_px: int = 20):
    """Split a text-line bbox at column/row boundaries it meaningfully crosses.
    Returns [((x, y, w, h), cell_name), ...] in page coordinates. The pieces
    always partition the line bbox: no overlaps, full coverage."""
    x, y, w, h = line_box
    if not row_bands or not col_bands:
        return [((x, y, w, h), None)]
    xcuts = _cut_positions(x, w, col_bands, col_span_frac, min_piece_px, frac_of_band=False)
    ycuts = _cut_positions(y, h, row_bands, row_span_frac, min_piece_px, frac_of_band=True)

    def seam(boundary, axis):
        rel = boundary - (x if axis == "x" else y)
        cut = find_seam(binary_crop, int(round(rel)), window_px, axis, dilate_px)
        return cut + (x if axis == "x" else y)

    xs = [x] + [seam(c, "x") for c in xcuts] + [x + w]
    ys = [y] + [seam(c, "y") for c in ycuts] + [y + h]
    # dropping a cut merges the would-be sliver into its neighbour piece and
    # keeps the piece set a grid-aligned partition of the line bbox
    xs = _drop_sliver_cuts(sorted(set(xs)), min_piece_px)
    ys = _drop_sliver_cuts(sorted(set(ys)), min_piece_px)

    pieces = []
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            cell = assign_cell(row_bands, col_bands, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
            pieces.append(((x0, y0, x1 - x0, y1 - y0), cell))
    if not pieces:
        pieces = [((x, y, w, h),
                   assign_cell(row_bands, col_bands, x + w / 2.0, y + h / 2.0))]
    return pieces
