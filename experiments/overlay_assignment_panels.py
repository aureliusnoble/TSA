"""Two-panel text-line-to-cell assignment visualisation per val page:
  LEFT : legacy grid + whole-line centre/overlap assignment (production today)
  RIGHT: regularised grid + cell-aware line splitting (new method)

Legibility scheme: every grid cell gets a deterministic colour from a
qualitative palette such that neighbouring cells always differ. Cells are
tinted lightly; each text line (or split piece) is drawn saturated in the
colour of the cell it is ASSIGNED to. A box whose colour does not match the
tint underneath is a visible misassignment. Legacy's col1_row1 fallback (line
matched no cell) is drawn black. Uses the newest rows model (v7) + cols_b for
the grid; textlines from the cached textline model.
Usage: conda run -n TSA python -m experiments.overlay_assignment_panels [--scale 0.35]
"""
import argparse
import re

import cv2
import numpy as np

import src.grid as grid_v2
import src.tsa_utils as tsa
from experiments import common
from experiments.eval_grid import legacy_grid

# Qualitative palette (BGR). Index = (2*row + 5*col) % 10 keeps every
# horizontally or vertically adjacent cell pair on different colours.
PALETTE = [
    (200, 90, 30),    # blue-ish
    (40, 160, 40),    # green
    (30, 60, 220),    # red
    (170, 40, 170),   # purple
    (10, 140, 230),   # orange
    (140, 160, 0),    # teal
    (60, 40, 120),    # brown
    (190, 120, 220),  # pink
    (0, 190, 190),    # yellow-dark
    (120, 190, 90),   # light green
]
FALLBACK_COLOUR = (0, 0, 0)  # black: legacy line that matched no cell


def cell_colour(name):
    m = re.match(r"col(\d+)_row(\d+)", name)
    c, r = int(m.group(1)), int(m.group(2))
    return PALETTE[(2 * r + 5 * c) % len(PALETTE)]


def tint_cells(img, cells, alpha=0.16):
    overlay = img.copy()
    for name, (x, y, w, h) in cells.items():
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)),
                      cell_colour(name), -1)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for x, y, w, h in cells.values():  # thin borders on top
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                      (90, 90, 90), 1)
    return out


def draw_box(img, box, colour, thickness=5):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, thickness)


def legacy_assign(box, cells):
    """Replicates inference._find_cell_assignment: centre in cell + max overlap."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    best, best_overlap = None, 0
    for name, (gx, gy, gw, gh) in cells.items():
        if gx <= cx <= gx + gw and gy <= cy <= gy + gh:
            ox = min(x + w, gx + gw) - max(x, gx)
            oy = min(y + h, gy + gh) - max(y, gy)
            if ox > 0 and oy > 0 and ox * oy > best_overlap:
                best_overlap, best = ox * oy, name
    return best  # None -> production falls back to col1_row1


def line_boxes(tl_pred):
    out = []
    for p in tl_pred["polygons"].get("1", []):
        xs = [q[0] for q in p["polygon"]]
        ys = [q[1] for q in p["polygon"]]
        out.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
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
    out_dir = common.OUT / "assignment_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    from experiments.overlay_textline_panels import predict_1500

    rows_gen = "v7" if ("rows", "v7") in common.MODELS else "new"
    for page in common.val_pages():
        rp = common.predict_cached("rows", rows_gen, page)
        cp = common.predict_cached("cols", "new", page)
        tl = predict_1500(page)  # textlines at trained input size 1500 (post-fix)
        img = common.load_page(page["source"])
        w, h = rp["work_w"], rp["work_h"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = line_boxes(tl)

        # LEFT: legacy grid, whole-line assignment
        lg = legacy_grid(rp, cp, w, h)
        left = tint_cells(img, lg)
        n_fallback = 0
        for box in boxes:
            cell = legacy_assign(box, lg)
            if cell is None:
                n_fallback += 1
                draw_box(left, box, FALLBACK_COLOUR)
            else:
                draw_box(left, box, cell_colour(cell))

        # RIGHT: regularised grid (committed defaults), split pieces
        rb = grid_v2.regularise([rp["polygons"].get("2", []), rp["polygons"].get("3", [])], "y")
        cb = grid_v2.regularise([cp["polygons"].get("2", []), cp["polygons"].get("3", [])], "x")
        rg = grid_v2.make_grid(rb, cb)
        right = tint_cells(img, rg)
        n_pieces = 0
        for box in boxes:
            x, y, bw, bh = [int(v) for v in box]
            if bw <= 0 or bh <= 0:
                continue
            crop = gray[y:y + bh, x:x + bw]
            _, cbin = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            pieces = grid_v2.split_line((x, y, bw, bh), rb, cb, cbin)
            n_pieces += len(pieces)
            for pbox, cell in pieces:
                draw_box(right, pbox, cell_colour(cell) if cell else FALLBACK_COLOUR)

        left = title_bar(left, f"Legacy: whole lines on legacy grid "
                               f"({len(boxes)} lines, {n_fallback} unmatched=black)")
        right = title_bar(right, f"New: split pieces on regularised grid "
                                 f"({len(boxes)} lines -> {n_pieces} pieces)")
        combo = cv2.hconcat([left, right])
        if args.scale != 1.0:
            combo = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
        out = out_dir / f"{page['stem']}.png"
        cv2.imwrite(str(out), combo)
        print(f"wrote {out.name} ({len(boxes)} lines -> {n_pieces} pieces, "
              f"{n_fallback} legacy-unmatched)")
    print(f"\npanels -> {out_dir} (rows model gen: {rows_gen})")


if __name__ == "__main__":
    main()
