"""Scan a cross-department page sample with the rows and textline models and
score each page's detection confidence, to surface the lowest-confidence pages.

Models (recommended configuration): rows_v7 at input 768 (its training mean/std),
textlines_full at input 1500 (min_cc=1). Production preprocessing: resize to width
3840; rows on the raw colour page, textlines on the Otsu-binarized page.

Per page and model we record: object count, mean/median/p10/min of the per-object
Doc-UFCN confidences (mean class probability over each detected region). Pages with
zero detections score 0 (catastrophic failure belongs at the top of the low list).

Writes incrementally to OUT/lowconf_scan/scores.csv (resume-safe: already-scored
pages are skipped). Afterwards: lowest100_rows.csv / lowest100_textlines.csv +
per-department summary.
Usage: conda run -n TSA python -m experiments.lowconf_scan [--root DIR] [--limit N]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from doc_ufcn.main import DocUFCN

from experiments import common

ROOT = Path("/home/aurelius/Downloads/Random Raw 5000")
OUT = common.OUT / "lowconf_scan"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FIELDS = ["page", "department", "rows_n", "rows_mean", "rows_median", "rows_p10",
          "rows_min", "lines_n", "lines_mean", "lines_median", "lines_p10", "lines_min"]


def find_pages(root):
    pages = []
    for p in sorted(root.rglob("*")):
        if (p.is_file() and p.suffix.lower() in EXTS
                and not p.name.startswith("._")):
            pages.append(p)
    return pages


def conf_stats(polys, classes):
    confs = [item["confidence"] for c in classes for item in polys.get(c, [])]
    if not confs:
        return 0, 0.0, 0.0, 0.0, 0.0
    a = np.array(confs, dtype=float)
    return len(a), float(a.mean()), float(np.median(a)), \
        float(np.percentile(a, 10)), float(a.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = Path(args.root)
    OUT.mkdir(parents=True, exist_ok=True)
    scores_path = OUT / "scores.csv"

    done = set()
    if scores_path.exists():
        with open(scores_path) as fh:
            done = {r["page"] for r in csv.DictReader(fh)}

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows_spec = common.MODELS[("rows", "v7")]
    rows_model = DocUFCN(4, 768, dev)
    rows_model.load(rows_spec["path"], rows_spec["mean"], rows_spec["std"])
    tl_spec = common.MODELS[("textlines", "old")]
    tl_model = DocUFCN(3, 1500, dev)
    tl_model.load(tl_spec["path"], tl_spec["mean"], tl_spec["std"])

    pages = find_pages(root)
    if args.limit:
        pages = pages[:args.limit]
    todo = [p for p in pages if str(p.relative_to(root)) not in done]
    print(f"{len(pages)} pages, {len(done)} already scored, {len(todo)} to do")

    new_file = not scores_path.exists()
    with open(scores_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, p in enumerate(todo):
            rel = str(p.relative_to(root))
            dept = rel.split("/")[0]
            try:
                img = common.load_page(p)
            except Exception as e:
                print(f"SKIP unreadable {rel}: {e}")
                continue
            polys_r, _, _, _ = rows_model.predict(
                img, raw_output=True, mask_output=True, overlap_output=False)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
            polys_l, _, _, _ = tl_model.predict(
                binary, min_cc=1, raw_output=True, mask_output=True,
                overlap_output=False)
            rn, rm, rmed, rp10, rmin = conf_stats(polys_r, [1, 2, 3])
            ln, lm, lmed, lp10, lmin = conf_stats(polys_l, [1, 2])
            writer.writerow(dict(page=rel, department=dept, rows_n=rn,
                                 rows_mean=round(rm, 4), rows_median=round(rmed, 4),
                                 rows_p10=round(rp10, 4), rows_min=round(rmin, 4),
                                 lines_n=ln, lines_mean=round(lm, 4),
                                 lines_median=round(lmed, 4),
                                 lines_p10=round(lp10, 4), lines_min=round(lmin, 4)))
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(todo)}] {rel[:70]}")

    # Ranked outputs
    import pandas as pd
    df = pd.read_csv(scores_path)
    df = df.drop_duplicates("page", keep="last")
    for model, mean_col in (("rows", "rows_mean"), ("textlines", "lines_mean")):
        low = df.sort_values(mean_col).head(100)
        low.to_csv(OUT / f"lowest100_{model}.csv", index=False)
    summary = df.groupby("department").agg(
        pages=("page", "count"), rows_mean=("rows_mean", "mean"),
        lines_mean=("lines_mean", "mean")).sort_values("lines_mean")
    summary.to_csv(OUT / "department_summary.csv")
    print(summary.to_string())
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
