#!/usr/bin/env bash
# Create (if needed) and verify the TSA conda environment for inference.
#
#   bash scripts/hpc/setup_env.sh
#
# Idempotent: if the env already exists it is left untouched and only verified.
# Override the env name with TSA_ENV_NAME=... if "TSA" is taken on your cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${TSA_ENV_NAME:-TSA}"
ENV_FILE="${SCRIPT_DIR}/environment.yaml"

cd "${REPO_ROOT}"

echo "==> Repo root : ${REPO_ROOT}"
echo "==> Env name  : ${ENV_NAME}"
echo "==> Env file  : ${ENV_FILE}"

# --- locate conda ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found on PATH." >&2
  echo "Install Miniconda/Miniforge first, or 'module load anaconda' on your cluster," >&2
  echo "then re-run this script." >&2
  exit 1
fi
# Make 'conda activate' usable inside this non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- create env if absent ----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda env '${ENV_NAME}' already exists; skipping creation."
else
  echo "==> Creating conda env '${ENV_NAME}' from ${ENV_FILE} ..."
  # environment.yaml hard-codes name: TSA; -n overrides it if the user changed it.
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

# --- verification: imports ---------------------------------------------------
# NB: `conda run python - <<HEREDOC` does NOT forward stdin, so the script would
# run empty. Pass the code as a -c argument via command substitution instead.
echo "==> Verifying Python imports in '${ENV_NAME}' ..."
conda run -n "${ENV_NAME}" python -c "$(cat <<'PY'
import importlib
mods = ["torch", "torchvision", "cv2", "pydantic", "rich", "tqdm",
        "pandas", "yaml", "PIL", "transformers", "numpy", "requests",
        "shapely", "doc_ufcn"]
for m in mods:
    importlib.import_module(m)
import torch, cv2, transformers
print("VERIFY OK: torch", torch.__version__,
      "| cuda_build", torch.version.cuda,
      "| cv2", cv2.__version__,
      "| transformers", transformers.__version__)
print("VERIFY OK: torch.cuda.is_available() =", torch.cuda.is_available(),
      "(False is expected on a login node with no GPU)")
PY
)"

# --- verification: model weights present -------------------------------------
# A fresh clone ships rows_v7 + cols_b only. The textline and transcription
# weights are large and may not be in git; flag anything the production config
# points at that is missing, so the user fixes it before submitting GPU jobs.
echo "==> Checking model weights referenced by configs/recommended_grid_v2.yaml ..."
conda run -n "${ENV_NAME}" python -c "$(cat <<'PY'
import sys, yaml
from pathlib import Path
root = Path(sys.argv[1])
cfg = yaml.safe_load((root / "configs" / "recommended_grid_v2.yaml").open())
m = cfg["models"]
targets = {
    "transcription": m["transcription"]["path"] if isinstance(m["transcription"], dict) else m["transcription"],
    "line_extraction": m["line_extraction"],
    "row_extraction": m["row_extraction"],
    "column_extraction": m["column_extraction"],
}
missing = []
for name, rel in targets.items():
    p = (root / rel)
    ok = p.exists()
    print(f"    [{'OK ' if ok else 'MISSING'}] {name}: {rel}")
    if not ok:
        missing.append((name, rel))
if missing:
    print()
    print("WARNING: the following model weights are NOT present on disk:")
    for name, rel in missing:
        print(f"    - {name}: {rel}")
    print("Inference will fail until these exist. See README_HPC.md -> 'Model weights'.")
    sys.exit(0)  # warn, do not hard-fail setup
print("All four model paths present.")
PY
)" "${REPO_ROOT}"

echo "==> Done. Activate with:  conda activate ${ENV_NAME}"
