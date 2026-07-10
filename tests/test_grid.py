import numpy as np
import pytest
from src import grid


def poly(x0, y0, x1, y1):
    return {"confidence": 1.0, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_polygons_to_intervals_projects_bbox():
    ivs = grid.polygons_to_intervals([poly(10, 100, 3800, 180)], axis="y")
    assert ivs == [(100.0, 180.0)]
    ivs = grid.polygons_to_intervals([poly(10, 100, 3800, 180)], axis="x")
    assert ivs == [(10.0, 3800.0)]


def test_merge_intervals_joins_fragments():
    # A row band predicted as two pieces with a small gap becomes one band
    ivs = [(100, 180), (185, 260), (400, 480)]
    merged = grid.merge_intervals(ivs, merge_gap=10)
    assert merged == [(100, 260), (400, 480)]


def test_filter_small_is_relative_and_keeps_two():
    ivs = [(0, 80), (100, 180), (200, 205)]  # median size 80, sliver 5
    kept = grid.filter_small(ivs, min_frac=0.40)
    assert kept == [(0, 80), (100, 180)]
    # never filter below 2 bands
    assert grid.filter_small([(0, 80), (100, 103)], min_frac=0.40) == [(0, 80), (100, 103)]


def test_resolve_overlaps_cuts_at_midpoint():
    out = grid.resolve_overlaps([(0, 110), (90, 200)])
    assert out == [(0, 100), (100, 200)]


def test_fill_gaps_inserts_by_median():
    # bands of ~100; a 300-gap should get 3 synthetic bands
    ivs = [(0, 100), (400, 500), (500, 600)]
    out = grid.fill_gaps(ivs, fill_trigger=0.60)
    assert len(out) == 6
    starts = [s for s, _ in out]
    assert starts == sorted(starts)
    # exact tiling: each band starts where previous ends
    for (s0, e0), (s1, e1) in zip(out, out[1:]):
        assert e0 == pytest.approx(s1)


def test_fill_gaps_extends_neighbours_for_small_gap():
    ivs = [(0, 100), (120, 220)]  # gap 20 < 0.6*100
    out = grid.fill_gaps(ivs, fill_trigger=0.60)
    assert out == [(0, 110), (110, 220)]


def test_regularise_end_to_end_tiles():
    polys = [poly(0, 0, 3800, 90), poly(0, 100, 1800, 190), poly(1900, 105, 3800, 190),
             poly(0, 500, 3800, 600)]  # second band fragmented, big gap after
    bands = grid.regularise(polys, axis="y")
    for (s0, e0), (s1, e1) in zip(bands, bands[1:]):
        assert e0 == pytest.approx(s1)
    assert bands[0][0] == 0 and bands[-1][1] == 600
    # gap 190..500 (~310) with median ~95 -> 3 inserted bands: total 2+3+1
    assert len(bands) == 6


def test_make_grid_and_assign_cell():
    rows = [(0.0, 100.0), (100.0, 200.0)]
    cols = [(0.0, 50.0), (50.0, 150.0)]
    g = grid.make_grid(rows, cols)
    assert set(g) == {"col1_row1", "col2_row1", "col1_row2", "col2_row2"}
    assert g["col2_row1"] == (50.0, 0.0, 100.0, 100.0)
    assert grid.assign_cell(rows, cols, cx=75, cy=150) == "col2_row2"
    # outside points clamp instead of falling back to col1_row1
    assert grid.assign_cell(rows, cols, cx=9999, cy=-5) == "col2_row1"
