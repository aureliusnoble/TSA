"""Contact sheets for the low-confidence scan: a 10x10 thumbnail grid of the 100
lowest-confidence pages per model (rows, textlines), each tile labelled with the
mean confidence and department. Reads lowest100_{rows,textlines}.csv.
Usage: conda run -n TSA python -m experiments.lowconf_sheets
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments import common

ROOT = Path("/home/aurelius/Downloads/Random Raw 5000")
OUT = common.OUT / "lowconf_scan"
TILE_W, TILE_H, LABEL_H = 360, 260, 34
COLS, ROWS_ = 10, 10


def tile(page_rel, label):
    img = cv2.imread(str(ROOT / page_rel), cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((TILE_H, TILE_W, 3), 128, np.uint8)
    else:
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    bar = np.full((LABEL_H, TILE_W, 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1,
                cv2.LINE_AA)
    return np.vstack([img, bar])


def sheet(csv_name, mean_col, out_name, title):
    df = pd.read_csv(OUT / csv_name)
    tiles = [tile(r["page"], f"{r[mean_col]:.2f} {r['department'][:18]}")
             for _, r in df.head(COLS * ROWS_).iterrows()]
    while len(tiles) < COLS * ROWS_:
        tiles.append(np.full((TILE_H + LABEL_H, TILE_W, 3), 255, np.uint8))
    rows_img = [cv2.hconcat(tiles[i * COLS:(i + 1) * COLS]) for i in range(ROWS_)]
    grid = cv2.vconcat(rows_img)
    head = np.full((60, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.imwrite(str(OUT / out_name), cv2.vconcat([head, grid]))
    print("wrote", OUT / out_name)


if __name__ == "__main__":
    sheet("lowest100_rows.csv", "rows_mean", "sheet_lowest100_rows.jpg",
          "100 lowest-confidence pages: ROWS model (rows_v7), ranked by mean confidence")
    sheet("lowest100_textlines.csv", "lines_mean", "sheet_lowest100_textlines.jpg",
          "100 lowest-confidence pages: TEXTLINE model (1500), ranked by mean confidence")
