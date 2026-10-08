#!/usr/bin/env python3
"""Write a per-departement inference config derived from the production base config.

The base config (configs/recommended_grid_v2.yaml) carries the validated model
paths, grid/line-split settings and normalisation stats. This script copies it
verbatim and overrides ONLY the three directory fields that change per
departement: directories.input, directories.output and directories.temp.

Model paths and the page_classification / table_guide CSV paths are left exactly
as they appear in the base config. In the production config these are
repo-relative (for example "models/rows_v7/model.pth"); inference.py resolves
them against the current working directory, so inference MUST be run from the
repo root. The HPC worker (run_department.sh) always does this.

Usage:
    python scripts/hpc/make_config.py \
        --department Paris \
        --input-dir  /scratch/tsa/Paris/images \
        --output-dir /scratch/tsa/Paris/output \
        --out        configs/generated/Paris.yaml

If --temp-dir is omitted it defaults to <output-dir>/tmp.
If --out is omitted it defaults to configs/generated/<department>.yaml
(relative to the repo root).
"""

import argparse
import sys
from pathlib import Path

import yaml

# Repo root = two levels up from this file (scripts/hpc/make_config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "configs" / "recommended_grid_v2.yaml"

# Optional BERT column classifier (models/column_classification/, fetched from S3
# by setup_env.sh). Paths are kept repo-relative, like the other model paths.
BERT_MODEL_REL = "models/column_classification/bert_columns.pt"
BERT_LABELMAP_REL = "models/column_classification/label_mapping.csv"
# The tokenizer is a HuggingFace name, fetched by name at inference time. On an
# offline compute node it must already be in the HF cache (see README_HPC.md); if
# it cannot be fetched, inference disables BERT and continues without it.
DEFAULT_BERT_TOKENIZER = "camembert-base"


def build_config(base_config: Path, input_dir: Path, output_dir: Path,
                 temp_dir: Path, with_bert: bool = False,
                 bert_tokenizer: str = DEFAULT_BERT_TOKENIZER) -> dict:
    """Load the base config and override only the three directory fields.

    If with_bert is True, add the optional BERT classifier keys to models.
    """
    with base_config.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Base config did not parse to a mapping: {base_config}")
    if "directories" not in cfg or not isinstance(cfg["directories"], dict):
        raise ValueError(f"Base config has no 'directories' section: {base_config}")

    # Store directories as absolute paths so they are unambiguous regardless of
    # the working directory. Model / layout paths are intentionally left as-is
    # (repo-relative in the production config).
    cfg["directories"]["input"] = str(input_dir)
    cfg["directories"]["output"] = str(output_dir)
    cfg["directories"]["temp"] = str(temp_dir)

    if with_bert:
        cfg.setdefault("models", {})
        cfg["models"]["bert_classifier_model"] = BERT_MODEL_REL
        cfg["models"]["bert_classifier_tokenizer"] = bert_tokenizer
        cfg["models"]["bert_classifier_label_map"] = BERT_LABELMAP_REL
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--department", required=True,
                        help="Departement name (used for default --out and a header comment)")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing the extracted page images for this departement")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where the pipeline writes its CSV tables (under output/pages/...)")
    parser.add_argument("--temp-dir", default=None,
                        help="Temp directory (default: <output-dir>/tmp)")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE),
                        help=f"Base config to derive from (default: {DEFAULT_BASE})")
    parser.add_argument("--out", default=None,
                        help="Where to write the generated YAML "
                             "(default: configs/generated/<department>.yaml)")
    bert = parser.add_mutually_exclusive_group()
    bert.add_argument("--with-bert", dest="with_bert", action="store_true",
                      default=None,
                      help="Force-include the BERT column classifier block "
                           "(default: auto - on iff the weights dir exists)")
    bert.add_argument("--no-bert", dest="with_bert", action="store_false",
                      help="Omit the BERT classifier (use on offline nodes with "
                           "no HuggingFace cache for the tokenizer)")
    parser.add_argument("--bert-tokenizer", default=DEFAULT_BERT_TOKENIZER,
                        help=f"HuggingFace tokenizer name for BERT "
                             f"(default: {DEFAULT_BERT_TOKENIZER})")
    args = parser.parse_args()

    base_config = Path(args.base_config).expanduser().resolve()
    if not base_config.is_file():
        print(f"ERROR: base config not found: {base_config}", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    temp_dir = (Path(args.temp_dir).expanduser().resolve()
                if args.temp_dir else output_dir / "tmp")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = REPO_ROOT / "configs" / "generated" / f"{args.department}.yaml"

    # BERT: explicit --with-bert / --no-bert wins; otherwise auto-detect by the
    # presence of the classifier weights (fetched by setup_env.sh).
    bert_model_path = REPO_ROOT / BERT_MODEL_REL
    if args.with_bert is None:
        with_bert = bert_model_path.is_file()
    else:
        with_bert = args.with_bert
        if with_bert and not bert_model_path.is_file():
            print(f"WARNING: --with-bert set but {BERT_MODEL_REL} not found; "
                  f"inference will disable BERT at runtime.", file=sys.stderr)

    cfg = build_config(base_config, input_dir, output_dir, temp_dir,
                       with_bert=with_bert, bert_tokenizer=args.bert_tokenizer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Auto-generated by scripts/hpc/make_config.py for departement: "
              f"{args.department}\n"
              f"# Derived from: {base_config.name} (only directories.* overridden)\n"
              f"# BERT column classifier: {'enabled' if with_bert else 'disabled'}\n")
    with out_path.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
