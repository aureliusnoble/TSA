"""Fill the Doc-UFCN prediction cache for all val pages (CPU).
Usage: conda run -n TSA python -m experiments.cache_predictions \
           --kinds rows cols textlines --gens old new"""
import argparse
import json
import os

import cv2
import torch
from doc_ufcn.main import DocUFCN

from experiments import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["rows", "cols", "textlines"])
    ap.add_argument("--gens", nargs="+", default=["old", "new"])
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 4))
    pages = common.val_pages()
    print(f"{len(pages)} val pages")
    for kind in args.kinds:
        for gen in args.gens:
            if (kind, gen) not in common.MODELS:
                continue
            spec = common.MODELS[(kind, gen)]
            model = DocUFCN(spec["classes"], 768, torch.device("cpu"))
            model.load(spec["path"], spec["mean"], spec["std"])
            cache = common.OUT / "pred_cache" / f"{kind}_{gen}"
            cache.mkdir(parents=True, exist_ok=True)
            for page in pages:
                f = cache / f"{page['stem']}.json"
                if f.exists():
                    continue
                img = common.load_page(page["source"])
                if kind == "textlines":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, b = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
                polys, _, _, _ = model.predict(img, raw_output=True,
                                               mask_output=True, overlap_output=False)
                f.write_text(json.dumps(
                    {"work_w": img.shape[1], "work_h": img.shape[0],
                     "polygons": {str(k): v for k, v in polys.items() if k != 0}},
                    default=common._json_default))
                print(f"cached {kind}/{gen}/{page['stem']}")


if __name__ == "__main__":
    main()
