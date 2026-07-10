"""Detection-style cell metrics: greedy IoU matching, F1 at thresholds,
count accuracy. Primary metrics per design doc 09: F1@0.5, mean F1 over
[.50:.95], cell/row/col count error."""
import re

import numpy as np


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def match_prf(pred, gt, iou_t):
    """Greedy one-to-one matching by IoU (highest first)."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred)
                    for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if v < iou_t:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); ious.append(v)
    tp = len(ious)
    fp, fn = len(pred) - tp, len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                mean_iou=float(np.mean(ious)) if ious else 0.0)


def _dims(cells):
    rows = {int(re.search(r"row(\d+)", k).group(1)) for k in cells}
    cols = {int(re.search(r"col(\d+)", k).group(1)) for k in cells}
    return len(rows), len(cols)


def cell_report(pred_cells, gt_cells):
    pred, gt = list(pred_cells.values()), list(gt_cells.values())
    at50 = match_prf(pred, gt, 0.50)
    f1s = [match_prf(pred, gt, t)["f1"] for t in np.arange(0.50, 0.951, 0.05)]
    pr, pc = _dims(pred_cells)
    gr, gc = _dims(gt_cells)
    return dict(f1_50=at50["f1"], mf1_50_95=float(np.mean(f1s)),
                mean_iou_50=at50["mean_iou"],
                count_err=len(pred) - len(gt),
                row_count_err=pr - gr, col_count_err=pc - gc,
                precision_50=at50["precision"], recall_50=at50["recall"])
