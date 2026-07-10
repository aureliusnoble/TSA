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
