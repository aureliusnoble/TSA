from types import SimpleNamespace

import numpy as np
import pytest

import src.tsa_utils as tsa
from inference import Config, Pipeline


BASE_CFG = dict(
    models=dict(transcription="x", line_extraction="x", row_extraction="x",
                column_extraction="x"),
    directories=dict(input="i", output="o", page_classification="p", table_guide="t"),
)


def test_config_defaults_are_legacy():
    cfg = Config(**BASE_CFG)
    assert cfg.grid_method == "legacy"
    assert cfg.line_split == "off"


def test_config_accepts_new_flags_and_rejects_junk():
    cfg = Config(**BASE_CFG, grid_method="regularised", line_split="cells")
    assert cfg.grid_method == "regularised"
    with pytest.raises(Exception):
        Config(**BASE_CFG, grid_method="bogus")


def band_poly(x0, y0, x1, y1):
    return {"confidence": 1.0, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def fake_polys(axis):
    # 3 clean bands in classes 2/3 (odd/even), 4-class doc-ufcn dict.
    # Adjacent odd/even bands EXACTLY TOUCH, as in real tables: per-parity
    # fragment merging must keep them distinct (pooled merging would fuse them).
    if axis == "y":
        mk = lambda a, b: band_poly(0, a, 3800, b)
    else:
        mk = lambda a, b: band_poly(a, 0, b, 2800)
    return {1: [], 2: [mk(0, 400), mk(800, 1200)], 3: [mk(400, 800)]}


def _bare_pipeline(grid_method):
    self = SimpleNamespace(ts=tsa.TableSegment(),
                           config=SimpleNamespace(grid_method=grid_method),
                           _row_bands=None, _col_bands=None)
    return self


def test_process_grid_cells_regularised_branch():
    self = _bare_pipeline("regularised")
    cells = Pipeline._process_grid_cells(self, fake_polys("x"), fake_polys("y"), 3840, 2900)
    assert len(cells) == 9  # 3 rows x 3 cols
    assert self._row_bands and self._col_bands
    assert all(len(v) == 4 for v in cells.values())


def test_extract_line_images_parses_multidigit_cell_names():
    # Cell names are "colN_rowM" (2 underscore-separated parts); the old parser
    # expected >= 4 parts so every line fell back to column='1', row='1'.
    self = SimpleNamespace(
        config=SimpleNamespace(line_split="off"),
        _row_bands=None, _col_bands=None,
    )
    self._find_cell_assignment = lambda *a: Pipeline._find_cell_assignment(self, *a)

    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    polygons = {1: [{"confidence": 1.0,
                     "polygon": [[60, 110], [140, 110], [140, 130], [60, 130]]}]}
    grid_cells = {"col3_row5": (50, 100, 100, 50)}

    assigned = Pipeline._extract_line_images_memory(
        self, image, polygons, header_y2=0, grid_cells=grid_cells, filename="f.jpg")

    assert len(assigned) == 1
    _, metadata = assigned[0]
    assert metadata["cell_name"] == "col3_row5"
    assert metadata["column"] == "3"
    assert metadata["row"] == "5"


def test_extract_line_images_cells_split_skips_degenerate_polygon():
    # line_split="cells": a wide line crossing a column boundary is split into
    # per-cell pieces; a degenerate polygon whose clamped bbox is empty (all
    # points at the image edge, so w clamps to 0) must not reach cv2.cvtColor,
    # which asserts on an empty crop and would drop the whole page.
    self = SimpleNamespace(
        config=SimpleNamespace(line_split="cells"),
        _row_bands=[(0.0, 200.0)],
        _col_bands=[(0.0, 200.0), (200.0, 400.0)],
    )

    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    image[85:115, 60:180] = 0    # ink left of the column boundary at x=200
    image[85:115, 220:340] = 0   # ink right of it; boundary itself ink-free

    polygons = {1: [
        {"confidence": 1.0,  # spans both column bands
         "polygon": [[50, 80], [350, 80], [350, 120], [50, 120]]},
        {"confidence": 1.0,  # degenerate: bbox x=400 on a 400-wide image
         "polygon": [[400, 10], [400, 10], [400, 10]]},
    ]}

    assigned = Pipeline._extract_line_images_memory(
        self, image, polygons, header_y2=0, grid_cells={}, filename="f.jpg")

    # normal line split into one piece per column band; degenerate skipped
    assert len(assigned) == 2
    assert {m["pre_cell"] for _, m in assigned} == {"col1_row1", "col2_row1"}
    assert all(m["w"] > 0 and m["h"] > 0 for _, m in assigned)
    assert all(img.size > 0 for img, _ in assigned)


def test_line_input_size_default_matches_training():
    cfg = Config(**BASE_CFG)
    assert cfg.line_input_size == 1500
    assert Config(**BASE_CFG, line_input_size=768).line_input_size == 768


def test_recommended_config_parses():
    import yaml
    from pathlib import Path
    data = yaml.safe_load(Path("configs/recommended_grid_v2.yaml").read_text())
    cfg = Config(**data)
    assert cfg.grid_method == "regularised" and cfg.line_split == "cells"
    assert cfg.line_input_size == 1500
    assert cfg.row_norm_mean == [190, 188, 182] and cfg.row_norm_std == [53, 52, 51]
    assert cfg.col_norm_mean == [189, 187, 183] and cfg.col_norm_std == [57, 56, 55]
    assert cfg.models.row_extraction.endswith("rows_v7/model.pth")
