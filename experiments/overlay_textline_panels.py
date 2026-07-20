"""Before/after textline detection panels for every val page:
  LEFT : textline model at input size 768 (production before the fix)
  RIGHT: textline model at input size 1500 (trained size; production after the fix)
Both run on the production-preprocessed page (resize 3840, global Otsu). Detected
text_line polygons drawn as red boxes, header_line as blue; counts in the titles.
768 detections come from the existing prediction cache; 1500 detections are computed
here on CPU and cached under pred_cache/textlines_old_1500/.
Usage: conda run -n TSA python -m experiments.overlay_textline_panels [--scale 0.35]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments import common

CACHE_1500 = common.OUT / "pred_cache" / "textlines_old_1500"


def predict_1500(page):
    CACHE_1500.mkdir(parents=True, exist_ok=True)
    f = CACHE_1500 / f"{page['stem']}.json"
    if f.exists():
        return json.loads(f.read_text())
    import torch
    from doc_ufcn.main import DocUFCN
    spec = common.MODELS[("textlines", "old")]
    model = predict_1500.model
    if model is None:
        model = DocUFCN(spec["classes"], 1500, torch.device("cpu"))
        model.load(spec["path"], spec["mean"], spec["std"])
        predict_1500.model = model
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


predict_1500.model = None


def draw(img, pred, thickness=4):
    out = img.copy()
    for cls, colour in (("1", (60, 60, 230)), ("2", (230, 120, 40))):  # text red, header blue
        for p in pred["polygons"].get(cls, []):
            pts = np.array(p["polygon"]).reshape(-1, 2)
            cv2.rectangle(out, (int(pts[:, 0].min()), int(pts[:, 1].min())),
                          (int(pts[:, 0].max()), int(pts[:, 1].max())), colour, thickness)
    return out


def title_bar(img, text):
    bar = np.full((90, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.35)
    args = ap.parse_args()
    out_dir = common.OUT / "textline_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in common.val_pages():
        old = common.predict_cached("textlines", "old", page)   # 768, existing cache
        new = predict_1500(page)
        img = common.load_page(page["source"])
        n_old = len(old["polygons"].get("1", []))
        n_new = len(new["polygons"].get("1", []))
        left = title_bar(draw(img, old), f"Input 768 (before fix): {n_old} text lines")
        right = title_bar(draw(img, new), f"Input 1500 (fixed): {n_new} text lines")
        combo = cv2.hconcat([left, right])
        if args.scale != 1.0:
            combo = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{page['stem']}.png"), combo)
        print(f"{page['stem'][:50]:52} 768: {n_old:4} lines   1500: {n_new:4} lines")
    print(f"\npanels -> {out_dir}")


if __name__ == "__main__":
    main()
