"""Round 2 textline panels: previous production model vs the retrained lines_v6,
both at input 1500 with production preprocessing, on the 22 val pages.
  LEFT : textlines_full @1500 (production until today)
  RIGHT: textlines_v6  @1500 (retrained with 19 hard pages; deployed)
Outputs to OUT/textline_panels_v2/. The v6 predictions are cached under
pred_cache/textlines_v6_1500/.
Usage: conda run -n TSA python -m experiments.overlay_textline_panels_v2 [--scale 0.35]
"""
import argparse
import json
from pathlib import Path

import cv2
import torch

from experiments import common
from experiments.overlay_textline_panels import draw, predict_1500, title_bar

CACHE_V6 = common.OUT / "pred_cache" / "textlines_v6_1500"
V6_DIR = common.LIB / "models" / "textlines_v6"


def predict_v6(page):
    CACHE_V6.mkdir(parents=True, exist_ok=True)
    f = CACHE_V6 / f"{page['stem']}.json"
    if f.exists():
        return json.loads(f.read_text())
    from doc_ufcn.main import DocUFCN
    model = predict_v6.model
    if model is None:
        mean, std = common._mean_std(V6_DIR)
        model = DocUFCN(3, 1500, torch.device("cpu"))
        model.load(V6_DIR / "model.pth", mean, std)
        predict_v6.model = model
    img = common.load_page(page["source"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    polys, _, _, _ = model.predict(img, min_cc=1, raw_output=True,
                                   mask_output=True, overlap_output=False)
    out = {"work_w": img.shape[1], "work_h": img.shape[0],
           "polygons": {str(k): v for k, v in polys.items() if k != 0}}
    f.write_text(json.dumps(out, default=common._json_default))
    return out


predict_v6.model = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.35)
    args = ap.parse_args()
    out_dir = common.OUT / "textline_panels_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in common.val_pages():
        old = predict_1500(page)     # textlines_full @1500 (cached)
        new = predict_v6(page)       # textlines_v6 @1500
        img = common.load_page(page["source"])
        n_old = len(old["polygons"].get("1", []))
        n_new = len(new["polygons"].get("1", []))
        left = title_bar(draw(img, old), f"textlines_full @1500: {n_old} lines")
        right = title_bar(draw(img, new), f"textlines_v6 @1500 (NEW): {n_new} lines")
        combo = cv2.hconcat([left, right])
        if args.scale != 1.0:
            combo = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{page['stem']}.png"), combo)
        print(f"{page['stem'][:50]:52} full: {n_old:4}   v6: {n_new:4}")
    print(f"\npanels -> {out_dir}")


if __name__ == "__main__":
    main()
