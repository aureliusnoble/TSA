"""GT adapters for the 25 e2e pages + Digi-Texx hash reversal.

Two ground-truth sources:
- 5 Nievre pages: images in NIEVRE/"pages images" (raw archive names with
  embedded newlines), GT tables in NIEVRE/"ground truth" (sanitised stems,
  harmonised English column names, one row per table row).
- 20 Digi-Texx pages: images in E2E/Raw (md5-hashed names produced by
  tsa_utils.FileProcessor.copy_files_to_sets), GT tables in
  E2E/Transcription_corrected (same stems, original French printed headers
  in table order, "<nl>" cell-internal separators).
"""
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

from experiments import common

E2E = common.DATA / "End_to_End"
NIEVRE = Path("/home/aurelius/Dropbox/Work/PhD/Projects/"
              "(TSA)Tables_des_Successions_et_Absences/Core/Results/Check End-to-End")
LAYOUT = common.LIB / "layout_classification"


def _hash_variants(name):
    """Plausible strings hash_file_name may have been fed for one page.

    The raw scan filenames contain a literal newline before the trailing
    "_pageN.jpg"; the classification CSV strips it, so it is re-inserted.
    Both NFC and NFD unicode forms are tried (accented bureau names).
    """
    with_nl = re.sub(r"_page(\d+)\.jpg$", "\n_page\\1.jpg", name)
    return {unicodedata.normalize(nf, b)
            for b in (name, Path(name).stem, with_nl, Path(with_nl).stem)
            for nf in ("NFC", "NFD")}


def digitexx_departments():
    """Map hashed e2e filenames back to original page names by md5 matching.
    hash_file_name (tsa_utils) hashes the raw filename string.

    A filesystem scan of Core/Data + Core/Results (670k files, name and stem,
    NFC and NFD) matched nothing: the 2023 source tree the sample was hashed
    from is no longer present. The Nievre page classification listing does
    cover it: all 20 pages resolve to Nievre bureaus (departement Nievre,
    archive series 3 Q).
    """
    hashes = {p.stem.split("_", 1)[1]: p.name for p in (E2E / "Raw").glob("*.jpg")}
    listing = pd.read_csv(LAYOUT / "page_classifications_nievre.csv")
    rows = []
    for name in listing["File"].astype(str):
        for v in _hash_variants(name):
            h = hashlib.md5(v.encode()).hexdigest()
            if h in hashes:
                rows.append(dict(hashed=hashes[h], original=name,
                                 bureau=name.split("_")[0],
                                 department="Nievre"))
    df = pd.DataFrame(rows).drop_duplicates("hashed").sort_values("hashed")
    common.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(common.OUT / "e2e_digitexx_departments.csv", index=False)
    return df


def sanitize(name):
    """Mirror Pipeline._sanitize_filename (spaces/newlines -> single '_')."""
    s = name.replace(" ", "_").replace("\n", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def table_guide():
    """{table_type: {col_num: col_name}} from the deployed table guide."""
    df = pd.read_csv(LAYOUT / "table_layouts.csv")
    guide = {}
    for _, r in df.iterrows():
        guide.setdefault(r["table_type"], {})[int(r["col_num"])] = r["col_name"]
    return guide


def _nfc(s):
    return unicodedata.normalize("NFC", str(s))


def _nievre_types():
    """{NFC raw filename (newline stripped): table type} for Nievre pages."""
    df = pd.read_csv(LAYOUT / "page_classifications_nievre.csv")
    return {_nfc(f).replace("\n", ""): t
            for f, t in zip(df["File"].astype(str), df["Type"])}


def _digitexx_types():
    """{hashed filename: table type} from the Digi-Texx delivery metadata."""
    df = pd.read_excel(E2E / "Type_classification.xlsx")
    return dict(zip(df["Filename"], df["Type"]))


def nievre_pages():
    types = _nievre_types()
    gt = {p.stem: p for p in (NIEVRE / "ground truth").glob("*.csv")}
    pages = []
    for img in sorted((NIEVRE / "pages images").glob("*.jpg")):
        stem = sanitize(img.stem)
        if stem not in gt:
            raise FileNotFoundError(f"No Nievre GT csv for image {img.name!r}")
        pages.append(dict(page=stem, image=img, gt=gt[stem], source="nievre",
                          table_type=types.get(_nfc(img.name).replace("\n", ""))))
    return pages


def digitexx_pages():
    types = _digitexx_types()
    pages = []
    for img in sorted((E2E / "Raw").glob("*.jpg")):
        gt = E2E / "Transcription_corrected" / f"{img.stem}.csv"
        if not gt.is_file():
            raise FileNotFoundError(f"No Digi-Texx GT csv for image {img.name!r}")
        pages.append(dict(page=img.stem, image=img, gt=gt, source="digitexx",
                          table_type=types.get(img.name)))
    return pages


def pages():
    """All 25 e2e GT pages (5 Nievre + 20 Digi-Texx)."""
    return nievre_pages() + digitexx_pages()
