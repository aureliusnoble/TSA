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


def regularise(polygons: List[dict], axis: str, *,
               merge_gap_frac: float = 0.10,
               min_size_frac: float = 0.40,
               fill_trigger_frac: float = 0.60) -> List[Band]:
    """polygons (odd+even combined) -> clean, tiling band list."""
    ivs = polygons_to_intervals(polygons, axis)
    if not ivs:
        return []
    m = median(e - s for s, e in ivs)
    ivs = merge_intervals(ivs, merge_gap=merge_gap_frac * m)
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
