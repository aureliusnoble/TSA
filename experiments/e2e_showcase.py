"""End-to-end processing showcase on five hand-picked v7 test pages.

Runs the best pipeline configuration (regularised grid + cell-aware line splitting,
rows_v7 + cols_b, textlines at 1500) and writes, per page:
  inputs/<stem>.jpg            copy of the raw input scan
  <stem>_1_textlines.png       textline detections (red text, blue header)
  <stem>_2_grid.png            regularised grid cells (tinted, counts in title)
  <stem>_3_assignment.png      split pieces coloured by their assigned cell
  <stem>_table.csv             final reconstructed row-by-column table (TrOCR output)
Usage: conda run -n TSA python -m experiments.e2e_showcase [--scale 0.4]
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import src.grid as grid_v2
from experiments import common
from experiments.eval_e2e import build_pipeline, reconstruct_table
from experiments.overlay_assignment_panels import (cell_colour, draw_box, line_boxes,
                                                   tint_cells, title_bar)
from experiments.overlay_textline_panels import predict_1500

PAGES = [
    "Seine-Saint-Denis_Saint-Denis_1768-1793_33",
    "Territoire de Belfort_Belfort_1814-1822_76",
    "Val-d'Oise_Marines_1838-1844_7",
    "Var_Ollières_1857-1866_20",
    "Yvelines_Rambouillet_1767-1791_12",
]
OUT = common.OUT / "e2e_showcase"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.4)
    args = ap.parse_args()
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline("BC", "new")   # regularised + cells, rows_v7 + cols_b
    for stem in PAGES:
        src = common.find_source(stem)
        assert src is not None, f"source not found for {stem}"
        shutil.copy2(src, OUT / "inputs" / src.name)
        page = dict(stem=stem, source=src)

        # Final table via the real pipeline (also exercises TrOCR).
        df = pipe.process_image(Path(src))
        reconstruct_table(df).to_csv(OUT / f"{stem}_table.csv")

        # Stage visualisations from the same models/settings.
        img = common.load_page(src)
        tl = predict_1500(page)
        rp = common.predict_cached("rows", "v7", page)
        cp = common.predict_cached("cols", "new", page)
        rb = grid_v2.regularise([rp["polygons"].get("2", []), rp["polygons"].get("3", [])], "y")
        cb = grid_v2.regularise([cp["polygons"].get("2", []), cp["polygons"].get("3", [])], "x")
        cells = grid_v2.make_grid(rb, cb)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1: textline detections
        p1 = img.copy()
        for cls, colour in (("1", (60, 60, 230)), ("2", (230, 120, 40))):
            for p in tl["polygons"].get(cls, []):
                pts = np.array(p["polygon"]).reshape(-1, 2)
                cv2.rectangle(p1, (int(pts[:, 0].min()), int(pts[:, 1].min())),
                              (int(pts[:, 0].max()), int(pts[:, 1].max())), colour, 4)
        n_tl = len(tl["polygons"].get("1", []))
        p1 = title_bar(p1, f"1. Textline detection (input 1500): {n_tl} lines")

        # 2: regularised grid
        p2 = title_bar(tint_cells(img, cells),
                       f"2. Regularised grid: {len(rb)} rows x {len(cb)} cols = {len(cells)} cells")

        # 3: assignment of split pieces
        p3 = tint_cells(img, cells)
        n_pieces = 0
        for box in line_boxes(tl):
            x, y, bw, bh = [int(v) for v in box]
            if bw <= 0 or bh <= 0:
                continue
            _, cbin = cv2.threshold(gray[y:y + bh, x:x + bw], 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            pieces = grid_v2.split_line((x, y, bw, bh), rb, cb, cbin)
            n_pieces += len(pieces)
            for pbox, cell in pieces:
                draw_box(p3, pbox, cell_colour(cell) if cell else (0, 0, 0))
        p3 = title_bar(p3, f"3. Cell assignment: {n_tl} lines -> {n_pieces} pieces")

        for suffix, panel in (("1_textlines", p1), ("2_grid", p2), ("3_assignment", p3)):
            if args.scale != 1.0:
                panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale,
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / f"{stem}_{suffix}.png"), panel)
        print(f"{stem}: {n_tl} lines, {len(cells)} cells, {n_pieces} pieces, "
              f"table {len(reconstruct_table(df))} rows")
    print(f"\nshowcase -> {OUT}")


if __name__ == "__main__":
    main()
