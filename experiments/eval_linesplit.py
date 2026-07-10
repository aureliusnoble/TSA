"""Experiment C: with the grid held at GT, do split crops map 1:1 to cells
better than whole text lines? Sweeps split_line parameters.
Usage: conda run -n TSA python -m experiments.eval_linesplit \
           [--overlays col=0.15,row=0.60,win=40,dil=7] [--overlays-only]"""
import argparse
import itertools
import time

import cv2
import numpy as np
import pandas as pd

import src.grid as grid_v2
from experiments import common

COL_SPAN = [0.10, 0.15, 0.25]
ROW_SPAN = [0.40, 0.60, 0.80]
WINDOW = [20, 40, 60]
DILATE = [3, 7, 11]

# Boundary slop in work-scale pixels (~0.26% of page width at 3840). Must stay
# well below typical line height (~100-150px): a page-relative tolerance
# (0.02 * max(w, h) ~ 77px) exceeded the height of most text-line crops, so
# they registered zero cells and silently inflated frac_single_cell.
TOL_PX = 10


def line_boxes(tl_pred):
    out = []
    for p in tl_pred["polygons"].get("1", []):
        xs = [q[0] for q in p["polygon"]]
        ys = [q[1] for q in p["polygon"]]
        box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if box[2] > 0 and box[3] > 0:
            out.append(box)
    return out


def crop_binary(img, box):
    x, y, w, h = [int(v) for v in box]
    gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def containment(box, rb, cb):
    """Cells whose area overlaps box by more than TOL_PX on each axis."""
    x, y, w, h = box
    hit = set()
    for r, (ry0, ry1) in enumerate(sorted(rb), 1):
        oy = min(y + h, ry1) - max(y, ry0)
        if oy <= TOL_PX:
            continue
        for c, (cx0, cx1) in enumerate(sorted(cb), 1):
            ox = min(x + w, cx1) - max(x, cx0)
            if ox > TOL_PX:
                hit.add((c, r))
    return hit


def page_stats(boxes_or_pieces, rb, cb):
    single = 0
    multi = 0
    none = 0
    per_cell = {}
    for box in boxes_or_pieces:
        cells = containment(box, rb, cb)
        if len(cells) == 0:
            none += 1
        elif len(cells) == 1:
            single += 1
            for cell in cells:
                per_cell[cell] = per_cell.get(cell, 0) + 1
        else:
            multi += 1
    nonempty = len(per_cell)
    total = max(1, len(boxes_or_pieces))
    return dict(frac_single_cell=single / total,
                frac_no_cell=none / total,
                multi_cell_crops=multi,
                crops_per_nonempty_cell=(sum(per_cell.values()) / nonempty)
                if nonempty else 0.0)


def page_setup(page):
    """GT bands and (box, binary crop) pairs for one page.
    The binary crop depends only on the box, never on sweep parameters, so it
    is computed once per line and reused across every config."""
    tl = common.predict_cached("textlines", "old", page)
    w, h = tl["work_w"], tl["work_h"]
    rb = common.load_gt_bands(page["rows_gt"], "y", w, h)
    cb = common.load_gt_bands(page["cols_gt"], "x", w, h)
    img = common.load_page(page["source"])
    lines = [(box, crop_binary(img, box)) for box in line_boxes(tl)]
    return lines, rb, cb


def sweep(pages):
    out_rows = []
    for page in pages:
        t0 = time.time()
        lines, rb, cb = page_setup(page)
        base = page_stats([b for b, _ in lines], rb, cb)
        out_rows.append(dict(variant="whole", col=None, row=None, win=None,
                             dil=None, page=page["stem"], **base))
        for cs, rs, wi, di in itertools.product(COL_SPAN, ROW_SPAN, WINDOW, DILATE):
            pieces = []
            for box, bcrop in lines:
                pieces += [b for b, _ in grid_v2.split_line(
                    box, rb, cb, bcrop, col_span_frac=cs, row_span_frac=rs,
                    window_px=wi, dilate_px=di)]
            rep = page_stats(pieces, rb, cb)
            out_rows.append(dict(variant="split", col=cs, row=rs, win=wi,
                                 dil=di, page=page["stem"], **rep))
        print(f"done {page['stem']} ({len(lines)} lines, "
              f"{time.time() - t0:.1f}s)", flush=True)
    return pd.DataFrame(out_rows)


def draw_overlays(pages, kv):
    od = common.OUT / "linesplit_overlays"
    od.mkdir(exist_ok=True)
    for page in pages[:8]:
        tl = common.predict_cached("textlines", "old", page)
        w, h = tl["work_w"], tl["work_h"]
        rb = common.load_gt_bands(page["rows_gt"], "y", w, h)
        cb = common.load_gt_bands(page["cols_gt"], "x", w, h)
        img = common.load_page(page["source"])
        canvas = img.copy()  # draw here; binarise from the untouched img
        for box in line_boxes(tl):
            x, y, bw, bh = [int(v) for v in box]
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 200, 0), 2)
            pieces = grid_v2.split_line(
                box, rb, cb, crop_binary(img, box),
                col_span_frac=float(kv["col"]), row_span_frac=float(kv["row"]),
                window_px=int(kv["win"]), dilate_px=int(kv["dil"]))
            if len(pieces) > 1:
                for (px, py, pw, ph), _ in pieces:
                    cv2.rectangle(canvas, (int(px), int(py)),
                                  (int(px + pw), int(py + ph)), (0, 0, 255), 3)
        cv2.imwrite(str(od / f"{page['stem']}.png"), canvas)
    print(f"overlays -> {od}")


def main():
    cv2.setNumThreads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", default=None,
                    help="col=0.15,row=0.60,win=40,dil=7 -> draw cut overlays")
    ap.add_argument("--overlays-only", action="store_true",
                    help="skip the sweep and only draw overlays")
    args = ap.parse_args()
    pages = common.val_pages()

    if not args.overlays_only:
        df = sweep(pages)
        df.to_csv(common.OUT / "linesplit_sweep.csv", index=False)
        summary = (df.groupby(["variant", "col", "row", "win", "dil"], dropna=False)
                     .agg(frac_single=("frac_single_cell", "mean"),
                          frac_none=("frac_no_cell", "mean"),
                          multi=("multi_cell_crops", "mean"),
                          per_cell=("crops_per_nonempty_cell", "mean"))
                     .reset_index().sort_values("frac_single", ascending=False))
        summary.to_csv(common.OUT / "linesplit_summary.csv", index=False)
        print(summary.head(15).to_string(index=False))
        print(summary[summary.variant == "whole"].to_string(index=False))

    if args.overlays:
        kv = dict(item.split("=") for item in args.overlays.split(","))
        draw_overlays(pages, kv)


if __name__ == "__main__":
    main()
