"""Three-panel grid visualisation for every val page:
  GT cells | legacy grid cells | regularised grid cells (chosen defaults).
Uses the cached Doc-UFCN predictions and the same grid builders as eval_grid,
so the panels show exactly what Experiment B measured.
Usage: conda run -n TSA python -m experiments.overlay_grid_panels [--gen new|old] [--scale 0.30]
"""
import argparse

import cv2
import numpy as np

from experiments import common
from experiments.eval_grid import gt_grid, legacy_grid, regularised_grid

# Chosen Experiment B defaults (also the src/grid.py regularise defaults).
BEST = dict(merge=0.05, drop=0.60, fill=0.40)

PANELS = [
    ("Ground truth", (40, 160, 40)),          # BGR green
    ("Legacy (current method)", (0, 140, 255)),   # orange
    ("Regularised (new method)", (60, 60, 230)),  # red
]


def draw_cells(img, cells, colour, thickness=4):
    out = img.copy()
    for x, y, w, h in cells.values():
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), colour, thickness)
    return out


def title_bar(img, text, colour):
    bar = np.full((90, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour, 5, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="new", choices=["new", "old"])
    ap.add_argument("--scale", type=float, default=0.30,
                    help="output downscale factor (panels are 3x3840 px wide raw)")
    args = ap.parse_args()

    out_dir = common.OUT / "grid_panels" / args.gen
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = common.val_pages()
    for page in pages:
        rp = common.predict_cached("rows", args.gen, page)
        cp = common.predict_cached("cols", args.gen, page)
        w, h = rp["work_w"], rp["work_h"]
        img = common.load_page(page["source"])
        grids = [gt_grid(page, w, h),
                 legacy_grid(rp, cp, w, h),
                 regularised_grid(rp, cp, BEST["merge"], BEST["drop"], BEST["fill"])]
        panels = [title_bar(draw_cells(img, g, colour), f"{label}  ({len(g)} cells)", colour)
                  for g, (label, colour) in zip(grids, PANELS)]
        combo = cv2.hconcat(panels)
        if args.scale != 1.0:
            combo = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
        out = out_dir / f"{page['stem']}.png"
        cv2.imwrite(str(out), combo)
        print(f"wrote {out.name}  (GT {len(grids[0])} / legacy {len(grids[1])} / reg {len(grids[2])} cells)")
    print(f"\n{len(pages)} panels -> {out_dir}")


if __name__ == "__main__":
    main()
