#!/usr/bin/env bash
# Process ONE departement end to end with automatic storage rotation.
#
#   bash scripts/hpc/run_department.sh <Departement>
#   e.g. bash scripts/hpc/run_department.sh Paris
#        bash scripts/hpc/run_department.sh Nievre     # accents optional
#
# Steps: skip-if-done -> download zip -> unzip -> make config -> run inference
#        -> tar.gz the CSV tables (+ manifest) -> optional S3 upload
#        -> delete the raw zip and extracted images (keep only the archive).
#
# Scratch is assumed LIMITED: the images for a departement are deleted as soon
# as its results archive exists, so only one departement's images sit on disk
# at a time.
#
# Environment variables (all optional):
#   TSA_SCRATCH      scratch root for downloads/images   (default: <repo>/scratch)
#   TSA_RESULTS_DIR  where result archives are written    (default: <repo>/results)
#   TSA_RESULTS_S3   s3://bucket/prefix to also upload the archive to (default: unset=skip)
#   TSA_AWS_REGION   AWS region for S3 access             (default: eu-west-2)
#   TSA_ENV_NAME     conda env name                       (default: TSA)
#   TSA_KEEP_IMAGES  set to 1 to NOT delete images/zip at the end (debugging)
#
# AWS credentials are read from the standard places (env vars AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials). This script NEVER embeds keys.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
URLS_CSV="${SCRIPT_DIR}/department_urls.csv"
DEPTS_CSV="${SCRIPT_DIR}/departments.csv"

ENV_NAME="${TSA_ENV_NAME:-TSA}"
SCRATCH_ROOT="${TSA_SCRATCH:-${REPO_ROOT}/scratch}"
RESULTS_DIR="${TSA_RESULTS_DIR:-${REPO_ROOT}/results}"
AWS_REGION="${TSA_AWS_REGION:-eu-west-2}"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <Departement>" >&2
  exit 2
fi
DEPT_ARG="$1"

# Inference resolves repo-relative model/layout paths and writes logs/inference.log
# against the CWD, so EVERYTHING runs from the repo root.
cd "${REPO_ROOT}"
mkdir -p logs "${SCRATCH_ROOT}" "${RESULTS_DIR}"

# --- helpers -----------------------------------------------------------------
# Accent-fold + lowercase + strip separators, for tolerant name matching.
fold() {
  printf '%s' "$1" | iconv -f UTF-8 -t ASCII//TRANSLIT 2>/dev/null \
    | tr '[:upper:]' '[:lower:]' | tr -d ' _-'
}

TARGET="$(fold "${DEPT_ARG}")"

# --- resolve the S3 URL from department_urls.csv -----------------------------
URL=""
CANON=""
while IFS=',' read -r dcol ucol; do
  [ "${dcol}" = "department" ] && continue
  [ -z "${dcol}" ] && continue
  if [ "$(fold "${dcol}")" = "${TARGET}" ]; then
    URL="${ucol}"; CANON="${dcol}"; break
  fi
done < "${URLS_CSV}"
URL="${URL%$'\r'}"   # strip a trailing CR if the CSV has CRLF line endings

if [ -z "${URL}" ]; then
  echo "ERROR: '${DEPT_ARG}' not found in ${URLS_CSV}." >&2
  echo "Available departements:" >&2
  tail -n +2 "${URLS_CSV}" | cut -d',' -f1 | sed 's/^/  - /' >&2
  exit 1
fi

# Derive an ASCII-safe slug from the zip basename (e.g. Nievre, Deux-Sevres).
ZIP_NAME="${URL##*/}"            # Nievre.zip
SLUG="${ZIP_NAME%.zip}"         # Nievre
BUCKET_HOST="${URL#https://}"
BUCKET="${BUCKET_HOST%%.s3.*}"  # tsatransferaurelius
S3_KEY="${URL##*amazonaws.com/}"
S3_URI="s3://${BUCKET}/${S3_KEY}"

DEPT_SCRATCH="${SCRATCH_ROOT}/${SLUG}"
ZIP_PATH="${DEPT_SCRATCH}/${ZIP_NAME}"
IMG_DIR="${DEPT_SCRATCH}/images"
WORK_OUT="${DEPT_SCRATCH}/output"
CFG_PATH="${REPO_ROOT}/configs/generated/${SLUG}.yaml"
ARCHIVE="${RESULTS_DIR}/${SLUG}.tar.gz"

echo "=============================================================="
echo " Departement : ${CANON}  (slug: ${SLUG})"
echo " S3 URI      : ${S3_URI}"
echo " Scratch     : ${DEPT_SCRATCH}"
echo " Archive out : ${ARCHIVE}"
echo "=============================================================="

# --- expected page count (from departments.csv) ------------------------------
EXPECTED_PAGES="unknown"
while IFS=',' read -r d _pri _tier _sz pages _rest; do
  [ "${d}" = "department" ] && continue
  if [ "$(fold "${d}")" = "${TARGET}" ]; then EXPECTED_PAGES="${pages}"; break; fi
done < "${DEPTS_CSV}"
EXPECTED_PAGES="${EXPECTED_PAGES%$'\r'}"

# --- STEP 1: skip if already done --------------------------------------------
if [ -f "${ARCHIVE}" ]; then
  echo "==> Results archive already exists: ${ARCHIVE}"
  echo "==> Nothing to do (delete the archive to force a re-run). Exiting."
  exit 0
fi

# --- activate conda env ------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Run scripts/hpc/setup_env.sh first." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# conda's activate.d hooks (e.g. MKL) reference unbound vars; relax -u around them.
set +u
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

mkdir -p "${DEPT_SCRATCH}"

# --- STEP 2+3: download + unzip (resumable, idempotent via marker files) -----
if [ -f "${DEPT_SCRATCH}/.extracted" ]; then
  echo "==> Images already extracted (${DEPT_SCRATCH}/.extracted present); skipping download/unzip."
else
  if [ -f "${DEPT_SCRATCH}/.downloaded" ] && [ -f "${ZIP_PATH}" ]; then
    echo "==> Zip already downloaded; skipping download."
  else
    echo "==> Downloading ${ZIP_NAME} ..."
    if command -v aws >/dev/null 2>&1; then
      # aws s3 cp streams to a temp file and renames on success (atomic).
      aws s3 cp --region "${AWS_REGION}" "${S3_URI}" "${ZIP_PATH}"
    else
      echo "    (aws CLI not found; falling back to curl with resume)"
      # -C - resumes a partial file; -f fails on HTTP errors; -L follows redirects.
      curl -fL -C - -o "${ZIP_PATH}" "${URL}"
    fi
    touch "${DEPT_SCRATCH}/.downloaded"
  fi

  echo "==> Unzipping into ${IMG_DIR} ..."
  mkdir -p "${IMG_DIR}"
  unzip -o -q "${ZIP_PATH}" -d "${IMG_DIR}"
  touch "${DEPT_SCRATCH}/.extracted"
fi

# --- STEP 4: per-departement config ------------------------------------------
echo "==> Writing config ${CFG_PATH} ..."
python scripts/hpc/make_config.py \
  --department "${SLUG}" \
  --input-dir  "${IMG_DIR}" \
  --output-dir "${WORK_OUT}" \
  --out        "${CFG_PATH}"

# --- STEP 5: inference (per-image resume: re-running continues where it left) -
echo "==> Running inference (resumes automatically for already-processed pages) ..."
python inference.py --config "${CFG_PATH}"

# --- STEP 6: archive the CSV tables + manifest -------------------------------
PAGES_DIR="${WORK_OUT}/pages"
if [ ! -d "${PAGES_DIR}" ]; then
  echo "ERROR: expected output directory not found: ${PAGES_DIR}" >&2
  echo "       inference produced no CSV tables; refusing to archive/delete." >&2
  exit 1
fi
PRODUCED=$(find "${PAGES_DIR}" -type f -name '*.csv' | wc -l | tr -d ' ')

MANIFEST="${WORK_OUT}/manifest.txt"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
{
  echo "departement        : ${CANON}"
  echo "slug               : ${SLUG}"
  echo "generated          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_commit        : ${GIT_COMMIT}"
  echo "base_config        : configs/recommended_grid_v2.yaml"
  echo "source_zip         : ${ZIP_NAME}"
  echo "pages_expected     : ${EXPECTED_PAGES}   (Pages ready, from departments.csv)"
  echo "csv_tables_produced: ${PRODUCED}         (one CSV per processed page)"
} > "${MANIFEST}"
echo "----- manifest -----"
cat "${MANIFEST}"
echo "--------------------"
if [ "${EXPECTED_PAGES}" != "unknown" ] && [ "${PRODUCED}" != "${EXPECTED_PAGES}" ]; then
  echo "NOTE: produced (${PRODUCED}) != expected (${EXPECTED_PAGES}). Some pages may have"
  echo "      failed (see logs/inference.log) or the zip may be incomplete for this dept."
fi

echo "==> Creating archive ${ARCHIVE} ..."
# Write to a temp file then move into place so a partial tar never looks complete.
TMP_ARCHIVE="${ARCHIVE}.partial"
tar -czf "${TMP_ARCHIVE}" -C "${WORK_OUT}" pages manifest.txt
mv -f "${TMP_ARCHIVE}" "${ARCHIVE}"
echo "    archive size: $(du -h "${ARCHIVE}" | cut -f1)"

# --- STEP 7: optional upload to S3 results prefix ----------------------------
if [ -n "${TSA_RESULTS_S3:-}" ]; then
  if command -v aws >/dev/null 2>&1; then
    DEST="${TSA_RESULTS_S3%/}/${SLUG}.tar.gz"
    echo "==> Uploading archive to ${DEST} ..."
    aws s3 cp --region "${AWS_REGION}" "${ARCHIVE}" "${DEST}"
  else
    echo "WARNING: TSA_RESULTS_S3 is set but aws CLI not found; skipping upload." >&2
  fi
fi

# --- STEP 8: reclaim scratch (delete zip + extracted images) -----------------
if [ "${TSA_KEEP_IMAGES:-0}" = "1" ]; then
  echo "==> TSA_KEEP_IMAGES=1: leaving zip and images in ${DEPT_SCRATCH}."
else
  echo "==> Reclaiming scratch: deleting zip and extracted images ..."
  rm -rf "${IMG_DIR}"
  rm -f  "${ZIP_PATH}"
  rm -f  "${DEPT_SCRATCH}/.downloaded" "${DEPT_SCRATCH}/.extracted"
fi

echo "==> DONE: ${CANON}. Archive: ${ARCHIVE}"
