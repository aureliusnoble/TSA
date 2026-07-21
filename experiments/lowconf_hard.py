"""Hard-pages-with-substantial-content lists from the low-confidence scan.

The naive mean-confidence ranking surfaces empty/sparse pages (few detections,
weak signal). Here we require substantial content (rows_n >= 15 row bands /
lines_n >= 80 text lines, roughly the 10th/5th percentile of the sample) and rank
by the 10th-percentile object confidence (p10): pages whose weakest detections
are weak, despite plenty of content, are genuinely hard pages.

Outputs under OUT/lowconf_scan/: hard100_rows.csv, hard100_textlines.csv,
sheet_hard100_{rows,textlines}.jpg, and page copies in hard100_{rows,textlines}_pages/
(rank- and score-prefixed filenames).
Usage: conda run -n TSA python -m experiments.lowconf_hard
"""
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments import common
from experiments.lowconf_sheets import ROOT, TILE_H, TILE_W, tile

OUT = common.OUT / "lowconf_scan"
MIN_ROWS_N = 15
MIN_LINES_N = 80
COLS, ROWS_ = 10, 10

SPECS = [
    dict(model="rows", n_col="rows_n", n_min=MIN_ROWS_N, rank_col="rows_p10",
         mean_col="rows_mean"),
    dict(model="textlines", n_col="lines_n", n_min=MIN_LINES_N, rank_col="lines_p10",
         mean_col="lines_mean"),
]


def sheet(df, rank_col, out_name, title):
    tiles = [tile(r["page"], f"{r[rank_col]:.2f} n={r['rows_n' if 'rows' in rank_col else 'lines_n']} {r['department'][:14]}")
             for _, r in df.head(COLS * ROWS_).iterrows()]
    while len(tiles) < COLS * ROWS_:
        tiles.append(np.full((TILE_H + 34, TILE_W, 3), 255, np.uint8))
    grid = cv2.vconcat([cv2.hconcat(tiles[i * COLS:(i + 1) * COLS]) for i in range(ROWS_)])
    head = np.full((60, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.imwrite(str(OUT / out_name), cv2.vconcat([head, grid]))


def main():
    df = pd.read_csv(OUT / "scores.csv").drop_duplicates("page", keep="last")
    for s in SPECS:
        sub = df[df[s["n_col"]] >= s["n_min"]].sort_values(s["rank_col"]).head(100)
        sub.to_csv(OUT / f"hard100_{s['model']}.csv", index=False)
        sheet(sub, s["rank_col"], f"sheet_hard100_{s['model']}.jpg",
              f"100 hardest content-rich pages: {s['model'].upper()} "
              f"({s['n_col']}>={s['n_min']}, ranked by p10 confidence)")
        dest = OUT / f"hard100_{s['model']}_pages"
        dest.mkdir(exist_ok=True)
        n = 0
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            src = ROOT / r["page"]
            if not src.is_file():
                print("MISSING:", repr(r["page"][:70]))
                continue
            base = re.sub(r"\s+", " ", re.sub(r"[\n\r]+", " ", Path(r["page"]).name)).strip()
            name = f"{rank:03d}_p10-{r[s['rank_col']]:.3f}_{r['department']}_{base}"
            if Path(name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                name += ".jpg"
            shutil.copy2(src, dest / name)
            n += 1
        depts = sub["department"].value_counts().head(6).to_dict()
        print(f"{s['model']}: {n} pages copied -> {dest.name}; "
              f"p10 range {sub[s['rank_col']].min():.3f}-{sub[s['rank_col']].max():.3f}; "
              f"top depts {depts}")


if __name__ == "__main__":
    main()
