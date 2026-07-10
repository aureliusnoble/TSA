"""Experiment B: to what extent does each grid method uncover the GT grid?
Sweeps regularise() parameters; legacy grid replicated via TableSegment at
production scale. Usage:
  conda run -n TSA python -m experiments.eval_grid          # sweep + summary
  conda run -n TSA python -m experiments.eval_grid --overlays merge=0.15,drop=0.40,fill=0.60
"""
import argparse
import itertools
import json

import cv2
import numpy as np
import pandas as pd

import src.grid as grid_v2
import src.tsa_utils as tsa
from experiments import common, metrics

MERGE = [0.05, 0.10, 0.15, 0.25]
DROP = [0.20, 0.40, 0.60]
FILL = [0.40, 0.60, 0.80]


def gt_grid(page, work_w, work_h):
    rb = common.load_gt_bands(page["rows_gt"], "y", work_w, work_h)
    cb = common.load_gt_bands(page["cols_gt"], "x", work_w, work_h)
    return grid_v2.make_grid(rb, cb)


def legacy_grid(rows_pred, cols_pred, work_w, work_h):
    ts = tsa.TableSegment()
    pc = {int(k): v for k, v in cols_pred["polygons"].items()}
    pr = {int(k): v for k, v in rows_pred["polygons"].items()}
    col_u = ts.combine_polygons(pc, [2, 3])
    row_u = ts.combine_polygons(pr, [2, 3])
    bc = ts.get_bounding_boxes(col_u)
    br = ts.get_bounding_boxes(row_u)
    br = ts.adjust_bounding_boxes(br, "row")
    bc = ts.adjust_bounding_boxes(bc, "column")
    br = [(0, y, w, h) for x, y, w, h in br]
    bc = [(x, 0, w, h) for x, y, w, h in bc]
    br = ts.add_rows(br, work_w)
    return ts.find_grid_cells(br, bc)


def regularised_grid(rows_pred, cols_pred, merge, drop, fill):
    # regularise takes one polygon group PER PARITY CLASS: fragments merge
    # within a class; pooling first would fuse adjacent odd/even bands.
    kw = dict(merge_gap_frac=merge, min_size_frac=drop, fill_trigger_frac=fill)
    rp = [rows_pred["polygons"].get("2", []), rows_pred["polygons"].get("3", [])]
    cp = [cols_pred["polygons"].get("2", []), cols_pred["polygons"].get("3", [])]
    rb = grid_v2.regularise(rp, "y", **kw)
    cb = grid_v2.regularise(cp, "x", **kw)
    return grid_v2.make_grid(rb, cb)


def draw_overlays(pages, m, d, f):
    od = common.OUT / "grid_overlays"
    od.mkdir(exist_ok=True)
    for gen in ("old", "new"):
        for page in pages:
            rp = common.predict_cached("rows", gen, page)
            cp = common.predict_cached("cols", gen, page)
            w, h = rp["work_w"], rp["work_h"]
            img = common.load_page(page["source"])
            for cells, colour in ((gt_grid(page, w, h), (0, 255, 0)),
                                  (regularised_grid(rp, cp, m, d, f), (0, 0, 255))):
                for x, y, cw, ch in cells.values():
                    cv2.rectangle(img, (int(x), int(y)),
                                  (int(x + cw), int(y + ch)), colour, 3)
            cv2.imwrite(str(od / f"{gen}_{page['stem']}.png"), img)
    print(f"overlays -> {od}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", default=None,
                    help="merge=0.15,drop=0.40,fill=0.60 -> draw overlays only (no sweep)")
    args = ap.parse_args()
    pages = common.val_pages()
    if args.overlays:
        kv = dict(item.split("=") for item in args.overlays.split(","))
        draw_overlays(pages, float(kv["merge"]), float(kv["drop"]), float(kv["fill"]))
        return
    rows_out = []
    for gen in ("old", "new"):
        preds = {p["stem"]: (common.predict_cached("rows", gen, p),
                             common.predict_cached("cols", gen, p)) for p in pages}
        for i, page in enumerate(pages, 1):
            print(f"[{gen} {i}/{len(pages)}] {page['stem']}", flush=True)
            rp, cp = preds[page["stem"]]
            w, h = rp["work_w"], rp["work_h"]
            gt = gt_grid(page, w, h)
            rep = metrics.cell_report(legacy_grid(rp, cp, w, h), gt)
            rows_out.append(dict(gen=gen, method="legacy", merge=None, drop=None,
                                 fill=None, page=page["stem"], **rep))
            for m, d, f in itertools.product(MERGE, DROP, FILL):
                rep = metrics.cell_report(regularised_grid(rp, cp, m, d, f), gt)
                rows_out.append(dict(gen=gen, method="regularised", merge=m,
                                     drop=d, fill=f, page=page["stem"], **rep))
    df = pd.DataFrame(rows_out)
    common.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(common.OUT / "grid_sweep.csv", index=False)
    summary = (df.groupby(["gen", "method", "merge", "drop", "fill"], dropna=False)
                 .agg(f1_50=("f1_50", "mean"), mf1=("mf1_50_95", "mean"),
                      abs_count_err=("count_err", lambda s: s.abs().mean()),
                      row_err=("row_count_err", lambda s: s.abs().mean()),
                      col_err=("col_count_err", lambda s: s.abs().mean()),
                      iou=("mean_iou_50", "mean"))
                 .reset_index().sort_values("mf1", ascending=False))
    summary.to_csv(common.OUT / "grid_summary.csv", index=False)
    print(summary.head(15).to_string(index=False))
    print("\nlegacy baselines:")
    print(summary[summary.method == "legacy"].to_string(index=False))


if __name__ == "__main__":
    main()
