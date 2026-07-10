"""E2e: 25 GT pages x {legacy, B, BC} x {old, new}. Requires a free GPU.

Scores with the metric functions imported from the Noah_Paper analysis module
(loose word F1, loose char F1, TED accuracy - reused verbatim, not
reimplemented). Per-page reconstructed tables go to OUT/e2e_tables/, metric
rows append to OUT/e2e_results.csv; already-scored (page, variant, gen)
combinations are skipped on rerun.

Usage:
    conda run -n TSA python -m experiments.eval_e2e --variants legacy
    conda run -n TSA python -m experiments.eval_e2e --variants B BC
    conda run -n TSA python -m experiments.eval_e2e            # all variants
    conda run -n TSA python -m experiments.eval_e2e --rescore  # no GPU: rescore
        every reconstructed table already in OUT/e2e_tables/ and rewrite
        OUT/e2e_results.csv from scratch

TED column alignment: the pipeline pivot has numeric column indices while the
GT tables carry names, so both sides are mapped into the table-guide name
space for the page's layout type. Digi-Texx GT columns (original French
printed headers) are in table order and map positionally; Nievre GT columns
(harmonised English names) are matched to guide names by normalised string
comparison plus a small alias table. Word/char F1 are bag-of-token metrics
and independent of any alignment.
"""
import argparse
import importlib.util
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import yaml

from experiments import common, e2e_gt

NOAH_EVAL = Path("/home/aurelius/Documents/Noah_Paper/analysis/nievre_e2e_eval.py")

VARIANTS = {"legacy": ("legacy", "off"), "B": ("regularised", "off"),
            "BC": ("regularised", "cells")}

RESULT_COLS = ["page", "source", "variant", "gen",
               "loose_word_f1", "loose_char_f1", "ted_acc"]

# Nievre GT names that were harmonised away from the guide vocabulary.
ALIASES = {
    "order number": "article number",
    "revenue of real estate": "declared assets income from buildings",
    "situation of real estate": "declared assets situation of buildings",
}


def load_metrics_module():
    """Import Noah's nievre_e2e_eval.py; it is main-guarded, so exec is safe."""
    spec = importlib.util.spec_from_file_location("nievre_e2e_eval", NOAH_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_seg_models():
    """Gen "new" segmentation weights. rows_v7 supersedes the rows_b entry in
    common.MODELS for the e2e runs; columns stay on cols_b."""
    rows_dir = common.TRAIN / "runs" / "rows_v7"
    rows_mean, rows_std = common._mean_std(rows_dir)
    cols = common.MODELS[("cols", "new")]
    return {"row_model": (rows_dir / "model.pth", rows_mean, rows_std),
            "col_model": (cols["path"], cols["mean"], cols["std"])}


def build_pipeline(variant, gen):
    from inference import Pipeline
    gm, ls = VARIANTS[variant]
    base = yaml.safe_load((Path(__file__).parent / "configs/e2e_base.yaml").read_text())
    base["grid_method"], base["line_split"] = gm, ls
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(base, fh)
        cfg = fh.name
    try:
        pipe = Pipeline(cfg)
    finally:
        Path(cfg).unlink(missing_ok=True)
    if gen == "new":
        # Pipeline._initialize_models hardcodes the deployed means/stds, so the
        # new-generation weights are applied by reloading post-construction.
        for attr, (path, mean, std) in new_seg_models().items():
            getattr(pipe, attr).load(path, mean, std)
    return pipe


def reconstruct_table(df):
    """Pivot the line-level pipeline output into a row x column table.

    Header lines are dropped (GT tables hold data rows only); fragments that
    share a cell are joined in reading order with the "<nl>" separator the GT
    transcriptions use for multi-line cells.
    """
    if df.empty:
        return pd.DataFrame()
    df = df[~df["header"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame()
    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["column"] = pd.to_numeric(df["column"], errors="coerce")
    df = df.dropna(subset=["row", "column"])
    df[["row", "column"]] = df[["row", "column"]].astype(int)
    df["transcription"] = df["transcription"].fillna("").astype(str)
    df = df.sort_values(["row", "column", "y", "x"])
    table = (df.groupby(["row", "column"])["transcription"]
               .agg(lambda v: "<nl>".join(s for s in v if s))
               .unstack("column"))
    # Keep skipped row numbers as empty rows so row positions stay aligned
    # with the detected grid when TED later resets to a positional index.
    return table.reindex(range(1, int(table.index.max()) + 1)).sort_index()


def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def name_pred_columns(table, table_type, guide):
    """Rename the pivot's numeric columns to table-guide names."""
    layout = guide.get(table_type, {})
    table = table.copy()
    table.columns = [layout.get(int(c), f"col{int(c)}") for c in table.columns]
    return table.loc[:, ~table.columns.duplicated()]


def _match_guide_name(col, layout):
    n = _norm_name(col)
    n = ALIASES.get(n, n)
    by_norm = {_norm_name(v): v for v in layout.values()}
    if n in by_norm:
        return by_norm[n]
    for num in sorted(layout):
        g = _norm_name(layout[num])
        if g.startswith(n) or n.startswith(g):
            return layout[num]
    return str(col)


def normalise_gt(gt_df, source, table_type, guide):
    """Bring a GT table into the guide-name column space.

    Nievre GT: drop metadata columns (as nievre_e2e_eval.load_paired_data
    does), then match harmonised English names to guide names. Digi-Texx GT:
    French printed headers in table order, mapped positionally.
    """
    meta = {"filename", "row", "bureau"}
    gt_df = gt_df.drop(columns=[c for c in gt_df.columns if str(c) in meta])
    layout = guide.get(table_type, {})
    if source == "digitexx":
        gt_df.columns = [layout.get(i + 1, f"col{i + 1}")
                         for i in range(len(gt_df.columns))]
    else:
        gt_df.columns = [_match_guide_name(c, layout) for c in gt_df.columns]
    return gt_df.loc[:, ~gt_df.columns.duplicated()]


def score_page(nev, gt_df, pred_df):
    word = nev.loose_word_classification_report(gt_df, pred_df)
    char = nev.char_classification_report_loose(gt_df, pred_df)
    ted = nev.tree_edit_distance_accuracy(gt_df.reset_index(drop=True),
                                          pred_df.reset_index(drop=True))
    return dict(loose_word_f1=word["word_f1"], loose_char_f1=char["char_f1"],
                ted_acc=ted["ted_accuracy"])


def append_result(path, row):
    """Append `row`, first dropping any existing row for the same
    (page, variant, gen) so --force reruns supersede rather than duplicate."""
    new = pd.DataFrame([row])[RESULT_COLS]
    if path.exists():
        prev = pd.read_csv(path)
        keep = ~((prev["page"] == row["page"])
                 & (prev["variant"] == row["variant"])
                 & (prev["gen"] == row["gen"]))
        new = pd.concat([prev[keep], new], ignore_index=True)[RESULT_COLS]
    new.to_csv(path, index=False)


def rescore(nev, guide, pages, results_path, tables_dir):
    """Recompute all three metrics for every reconstructed table already on
    disk (no GPU needed) and rewrite e2e_results.csv from scratch.

    The saved tables hold exactly what score_page saw on the original run:
    string cells with NaN for empty ones, so they are read back with
    dtype=str (empty fields -> NaN, as in the in-memory pivot).
    """
    rows = []
    for variant in VARIANTS:
        for gen in ("old", "new"):
            for p in pages:
                tpath = tables_dir / f"{p['page']}_{variant}_{gen}.csv"
                if not tpath.is_file():
                    print(f"  missing table, skipped: {tpath.name}")
                    continue
                pred = pd.read_csv(tpath, index_col="row", dtype=str)
                gt_df = normalise_gt(nev.safe_read_csv(p["gt"]), p["source"],
                                     p["table_type"], guide)
                met = score_page(nev, gt_df, pred)
                rows.append(dict(page=p["page"], source=p["source"],
                                 variant=variant, gen=gen, **met))
    pd.DataFrame(rows)[RESULT_COLS].to_csv(results_path, index=False)
    print(f"Rescored {len(rows)} tables -> {results_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--gens", nargs="+", default=["old", "new"],
                    choices=["old", "new"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N pages (sanity runs)")
    ap.add_argument("--force", action="store_true",
                    help="rerun combinations already present in e2e_results.csv")
    ap.add_argument("--rescore", action="store_true",
                    help="no GPU: rescore every table in e2e_tables/ against "
                         "its GT and rewrite e2e_results.csv from scratch "
                         "(ignores --variants/--gens/--limit/--force)")
    args = ap.parse_args(argv)

    nev = load_metrics_module()
    guide = e2e_gt.table_guide()
    pages = e2e_gt.pages()
    if args.limit and not args.rescore:
        pages = pages[:args.limit]

    results_path = common.OUT / "e2e_results.csv"
    tables_dir = common.OUT / "e2e_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        rescore(nev, guide, pages, results_path, tables_dir)
        res = pd.read_csv(results_path)
        print("\nMeans by source/variant/gen:")
        print(res.groupby(["source", "variant", "gen"])
              [["loose_word_f1", "loose_char_f1", "ted_acc"]]
              .mean().round(4).to_string())
        return
    done = set()
    if results_path.exists() and not args.force:
        prev = pd.read_csv(results_path)
        done = set(zip(prev["page"], prev["variant"], prev["gen"]))

    for variant in args.variants:
        for gen in args.gens:
            todo = [p for p in pages if (p["page"], variant, gen) not in done]
            if not todo:
                print(f"{variant}/{gen}: nothing to do")
                continue
            print(f"=== {variant}/{gen}: {len(todo)} pages ===", flush=True)
            t_var = time.time()
            pipe = build_pipeline(variant, gen)
            for p in todo:
                t0 = time.time()
                df = pipe.process_image(p["image"])
                table = reconstruct_table(df)
                named = name_pred_columns(table, p["table_type"], guide)
                named.to_csv(tables_dir / f"{p['page']}_{variant}_{gen}.csv",
                             index_label="row")
                gt_df = normalise_gt(nev.safe_read_csv(p["gt"]), p["source"],
                                     p["table_type"], guide)
                met = score_page(nev, gt_df, named)
                append_result(results_path, dict(
                    page=p["page"], source=p["source"], variant=variant,
                    gen=gen, **met))
                print(f"  {p['page'][:58]}: wf1={met['loose_word_f1']:.3f} "
                      f"cf1={met['loose_char_f1']:.3f} ted={met['ted_acc']:.3f} "
                      f"({len(df)} lines, {time.time() - t0:.0f}s)", flush=True)
            del pipe
            import torch
            torch.cuda.empty_cache()
            print(f"=== {variant}/{gen} done in {time.time() - t_var:.0f}s ===",
                  flush=True)

    if results_path.exists():
        res = pd.read_csv(results_path)
        print("\nMeans by source/variant/gen:")
        print(res.groupby(["source", "variant", "gen"])
              [["loose_word_f1", "loose_char_f1", "ted_acc"]]
              .mean().round(4).to_string())


if __name__ == "__main__":
    main()
