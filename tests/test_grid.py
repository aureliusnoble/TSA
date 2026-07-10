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
    bands = grid.regularise([polys], axis="y")
    for (s0, e0), (s1, e1) in zip(bands, bands[1:]):
        assert e0 == pytest.approx(s1)
    assert bands[0][0] == 0 and bands[-1][1] == 600
    # gap 190..500 (~310) with median ~95 -> 3 inserted bands: total 2+3+1
    assert len(bands) == 6


def test_regularise_does_not_fuse_touching_alternating_bands():
    # odd rows and even rows alternate and TOUCH; they must remain distinct
    odd = [poly(0, 0, 3800, 100), poly(0, 200, 3800, 300)]
    even = [poly(0, 100, 3800, 200), poly(0, 300, 3800, 400)]
    bands = grid.regularise([odd, even], axis="y")
    assert len(bands) == 4
    assert bands == [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 400.0)]


def test_regularise_merges_fragments_within_parity():
    # one odd row predicted as two horizontal fragments (overlapping y-intervals)
    odd = [poly(0, 0, 1800, 100), poly(1900, 5, 3800, 95), poly(0, 200, 3800, 300)]
    even = [poly(0, 100, 3800, 200), poly(0, 300, 3800, 400)]
    bands = grid.regularise([odd, even], axis="y")
    assert len(bands) == 4


def test_make_grid_and_assign_cell():
    rows = [(0.0, 100.0), (100.0, 200.0)]
    cols = [(0.0, 50.0), (50.0, 150.0)]
    g = grid.make_grid(rows, cols)
    assert set(g) == {"col1_row1", "col2_row1", "col1_row2", "col2_row2"}
    assert g["col2_row1"] == (50.0, 0.0, 100.0, 100.0)
    assert grid.assign_cell(rows, cols, cx=75, cy=150) == "col2_row2"
    # outside points clamp instead of falling back to col1_row1
    assert grid.assign_cell(rows, cols, cx=9999, cy=-5) == "col2_row1"


def synth_line(w=400, h=60, words=((10, 120), (160, 260), (300, 390))):
    """White canvas with black 'words' (ink=255 in returned binary mask)."""
    img = np.zeros((h, w), np.uint8)
    for x0, x1 in words:
        img[15:45, x0:x1] = 255
    return img


def test_find_seam_prefers_ink_valley():
    binary = synth_line()
    # boundary at 150 sits in the gap between words 1 and 2 (120..160)
    cut = grid.find_seam(binary, boundary=140, window=40, axis="x", dilate_px=3)
    assert 120 <= cut <= 160


def test_find_seam_falls_back_near_boundary():
    binary = np.full((60, 400), 255, np.uint8)  # solid ink, no valley
    cut = grid.find_seam(binary, boundary=200, window=40, axis="x", dilate_px=3)
    assert 160 <= cut <= 240


def test_split_line_cuts_column_spanner():
    rows = [(0.0, 100.0)]
    cols = [(0.0, 200.0), (200.0, 400.0)]
    binary = synth_line()  # 400 wide; word gap 120..160 near-ish boundary 200? gap 260..300 nearer
    pieces = grid.split_line((0, 20, 400, 60), rows, cols, binary)
    assert len(pieces) == 2
    (b1, c1), (b2, c2) = pieces
    assert c1 == "col1_row1" and c2 == "col2_row1"
    assert b1[0] == 0 and b2[0] + b2[2] == 400  # pieces cover the line
    assert b1[0] + b1[2] == b2[0]               # and abut at the cut


def test_split_line_keeps_minor_overhang_whole():
    rows = [(0.0, 100.0)]
    cols = [(0.0, 380.0), (380.0, 800.0)]  # only 20/400 = 5% overhang
    pieces = grid.split_line((0, 20, 400, 60), rows, cols, synth_line())
    assert len(pieces) == 1 and pieces[0][1] == "col1_row1"


def test_split_line_cuts_stacked_rows():
    rows = [(0.0, 60.0), (60.0, 120.0)]
    cols = [(0.0, 400.0)]
    binary = np.zeros((120, 400), np.uint8)
    binary[10:50, 20:380] = 255   # line 1
    binary[70:110, 20:380] = 255  # line 2 (valley 50..70 around boundary 60)
    pieces = grid.split_line((0, 0, 400, 120), rows, cols, binary)
    assert len(pieces) == 2
    assert {c for _, c in pieces} == {"col1_row1", "col1_row2"}


def test_split_line_partition_property():
    # cross-row + column split: pieces must tile the line bbox exactly
    rows = [(0.0, 60.0), (60.0, 120.0)]
    cols = [(0.0, 200.0), (200.0, 400.0)]
    binary = np.zeros((120, 400), np.uint8)
    binary[10:50, 20:380] = 255
    binary[70:110, 20:380] = 255
    pieces = grid.split_line((0, 0, 400, 120), rows, cols, binary)
    area = sum(w * h for (x, y, w, h), _ in pieces)
    assert area == 400 * 120  # full coverage, no overlap (pieces are grid-aligned)
    # every piece is in exactly one cell and at least min_piece_px wide/tall
    for (x, y, w, h), cell in pieces:
        assert w >= 20 and h >= 20 and cell is not None


def test_split_line_first_piece_sliver_is_absorbed():
    # a column boundary right at the box's left edge region must not emit a sliver
    rows = [(0.0, 100.0)]
    cols = [(0.0, 10.0), (10.0, 400.0)]  # boundary at 10 < min_piece_px from edge
    pieces = grid.split_line((0, 20, 400, 60), rows, cols, synth_line())
    assert all(w >= 20 for (x, y, w, h), _ in pieces)
    # and a legitimate boundary whose SEAM lands near the edge must have its
    # cut dropped (exercises _drop_sliver_cuts), not emit a sliver piece
    cols = [(0.0, 30.0), (30.0, 400.0)]
    binary = np.full((60, 400), 255, np.uint8)  # solid ink...
    binary[:, 5:15] = 0                         # ...except a valley at x~10
    pieces = grid.split_line((0, 20, 400, 60), rows, cols, binary,
                             col_span_frac=0.05, dilate_px=3)
    assert all(w >= 20 for (x, y, w, h), _ in pieces)
    assert sum(w for (x, y, w, h), _ in pieces) == 400  # still a partition


def test_split_line_empty_bands_returns_whole():
    assert grid.split_line((0, 0, 100, 50), [], [(0.0, 100.0)],
                           np.zeros((50, 100), np.uint8)) == [((0, 0, 100, 50), None)]


def test_find_seam_axis_y_direct():
    binary = np.zeros((120, 60), np.uint8)
    binary[10:50, :] = 255
    binary[70:110, :] = 255
    cut = grid.find_seam(binary, boundary=60, window=30, axis="y", dilate_px=3)
    assert 50 <= cut <= 70
