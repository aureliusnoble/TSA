"""Shared assets for the grid/line-split experiments (design doc 09)."""
import json
from pathlib import Path

import cv2
import numpy as np

LIB = Path(__file__).resolve().parents[1]
TRAIN = Path("/home/aurelius/Dropbox/Work/PhD/Projects/(TSA)Tables_des_Successions_et_Absences"
             "/Core/Code/Document_Analysis/line_segmentation/training")
DATA = Path("/home/aurelius/Dropbox/Work/PhD/Projects/(TSA)Tables_des_Successions_et_Absences"
            "/Core/Data")
OUT = LIB / "experiments" / "outputs"

SOURCE_DIRS = [
    DATA / "Full Sample" / "Training Data_Full Sample_Third Iteration",
    DATA / "Full Sample" / "Training Data_Full Sample_Second Iteration",
    DATA / "Full Sample" / "Training Data_Full Sample",
]
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".PNG")


def _json_default(o):
    """Serialise numpy scalars (doc_ufcn polygon coords are np.int64)."""
    return o.item()


def _mean_std(run_dir):
    mean = [int(float(v)) for v in (run_dir / "mean").read_text().split()]
    std = [int(float(v)) for v in (run_dir / "std").read_text().split()]
    return mean, std


def _models():
    m = {
        ("rows", "old"): dict(path=LIB / "models/rows_full/model.pth",
                              mean=[228] * 3, std=[71] * 3, classes=4),
        ("cols", "old"): dict(path=LIB / "models/columns_full/model.pth",
                              mean=[229] * 3, std=[71] * 3, classes=4),
        ("textlines", "old"): dict(path=LIB / "models/textlines_full/model.pth",
                                   mean=[221] * 3, std=[80] * 3, classes=3),
    }
    for kind, run in (("rows", "rows_b"), ("cols", "cols_b")):
        rd = TRAIN / "runs" / run
        mean, std = _mean_std(rd)
        m[(kind, "new")] = dict(path=rd / "model.pth", mean=mean, std=std, classes=4)
    # newest rows model (156-page v7 retrain); columns unchanged from "new"
    rd = TRAIN / "runs" / "rows_v7"
    if (rd / "model.pth").exists():
        mean, std = _mean_std(rd)
        m[("rows", "v7")] = dict(path=rd / "model.pth", mean=mean, std=std, classes=4)
    return m


MODELS = _models()


def find_source(stem):
    for d in SOURCE_DIRS:
        for ext in EXTS:
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def val_pages():
    rows_dir = TRAIN / "tsa_rows_fullsample_v6_formatted/val/labels_json"
    cols_dir = TRAIN / "tsa_columns_fullsample_v6_formatted/val/labels_json"
    rows = {p.stem.removeprefix("val_"): p for p in rows_dir.glob("val_*.json")}
    cols = {p.stem.removeprefix("val_"): p for p in cols_dir.glob("val_*.json")}
    pages = []
    for stem in sorted(set(rows) & set(cols)):
        src = find_source(stem)
        if src is None:
            continue
        pages.append(dict(stem=stem, rows_gt=rows[stem], cols_gt=cols[stem], source=src))
    return pages


def load_page(source_path, target_width=3840):
    img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if w != target_width:
        s = target_width / w
        img = cv2.resize(img, (target_width, int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def _gt_scale(d, work_w, work_h):
    gh, gw = d["img_size"]
    return work_w / gw, work_h / gh


def load_gt_boxes(labels_json_path, classes, work_w, work_h):
    d = json.loads(Path(labels_json_path).read_text())
    sx, sy = _gt_scale(d, work_w, work_h)
    boxes = []
    for cls in classes:
        for item in d.get(cls, []):
            xs = [p[0] for p in item["polygon"]]
            ys = [p[1] for p in item["polygon"]]
            boxes.append((min(xs) * sx, min(ys) * sy,
                          (max(xs) - min(xs)) * sx, (max(ys) - min(ys)) * sy))
    return boxes


def load_gt_bands(labels_json_path, axis, work_w, work_h):
    d = json.loads(Path(labels_json_path).read_text())
    odd = "odd line" if "odd line" in d else "odd column"
    even = "even line" if "even line" in d else "even column"
    boxes = load_gt_boxes(labels_json_path, [odd, even], work_w, work_h)
    if axis == "y":
        return sorted((y, y + h) for x, y, w, h in boxes)
    return sorted((x, x + w) for x, y, w, h in boxes)


def predict_cached(kind, gen, page):
    """Doc-UFCN polygons for one page at work scale, cached to JSON."""
    cache = OUT / "pred_cache" / f"{kind}_{gen}"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{page['stem']}.json"
    if f.exists():
        return json.loads(f.read_text())
    from doc_ufcn.main import DocUFCN
    import torch
    spec = MODELS[(kind, gen)]
    img = load_page(page["source"])
    if kind == "textlines":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    model = DocUFCN(spec["classes"], 768, torch.device("cpu"))
    model.load(spec["path"], spec["mean"], spec["std"])
    polys, _, _, _ = model.predict(img, raw_output=True, mask_output=True,
                                   overlap_output=False)
    out = {"work_w": img.shape[1], "work_h": img.shape[0],
           "polygons": {str(k): v for k, v in polys.items() if k != 0}}
    f.write_text(json.dumps(out, default=_json_default))
    return out
