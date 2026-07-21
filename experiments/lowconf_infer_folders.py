"""Run raw model inference on the (hand-pruned) hard-page folders and write
visualisations of the model output next to them.

For each image in hard100_rows_pages/ the rows model (rows_v7 at 768, its training
norm) runs on the production-preprocessed page; for hard100_textlines_pages/ the
textline model (textlines_full at 1500, min_cc=1) runs on the Otsu-binarized page.
The RAW predicted polygons (contours of the argmax mask, no grid/splitting
post-processing) are drawn over the page: filled at low alpha + outlined, one output
image per input, same filename, into sibling folders hard100_{rows,textlines}_inference/.

Colours: rows header green, odd line red, even line blue; textlines text_line red,
header_line blue.
Usage: conda run -n TSA python -m experiments.lowconf_infer_folders
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from doc_ufcn.main import DocUFCN

from experiments import common

BASE = common.OUT / "lowconf_scan"
JOBS = [
    dict(folder="hard100_rows_pages", out="hard100_rows_inference", kind="rows"),
    dict(folder="hard100_textlines_pages", out="hard100_textlines_inference",
         kind="textlines"),
]
COLOURS = {
    "rows": {1: (40, 200, 40), 2: (40, 40, 230), 3: (230, 80, 40)},      # BGR: header green, odd red, even blue
    "textlines": {1: (40, 40, 230), 2: (230, 80, 40)},                    # text red, header blue
}


def load_model(kind, dev):
    if kind == "rows":
        spec = common.MODELS[("rows", "v7")]
        model = DocUFCN(4, 768, dev)
    else:
        spec = common.MODELS[("textlines", "old")]
        model = DocUFCN(3, 1500, dev)
    model.load(spec["path"], spec["mean"], spec["std"])
    return model


def draw_raw(img, polys, colours, alpha=0.30):
    overlay = img.copy()
    for cls, colour in colours.items():
        for item in polys.get(cls, []):
            pts = np.array(item["polygon"], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], colour)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for cls, colour in colours.items():
        for item in polys.get(cls, []):
            pts = np.array(item["polygon"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, colour, 3)
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for job in JOBS:
        src_dir = BASE / job["folder"]
        out_dir = BASE / job["out"]
        out_dir.mkdir(exist_ok=True)
        model = load_model(job["kind"], dev)
        files = sorted(p for p in src_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        for i, p in enumerate(files):
            img = common.load_page(p)
            if job["kind"] == "rows":
                net_in = img
                kwargs = {}
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, b = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                net_in = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
                kwargs = dict(min_cc=1)
            polys, _, _, _ = model.predict(net_in, raw_output=True, mask_output=True,
                                           overlap_output=False, **kwargs)
            out = draw_raw(img, polys, COLOURS[job["kind"]])
            cv2.imwrite(str(out_dir / p.name), out)
            if (i + 1) % 20 == 0:
                print(f"{job['out']}: {i + 1}/{len(files)}")
        del model
        print(f"{job['out']}: done ({len(files)} pages)")


if __name__ == "__main__":
    main()
