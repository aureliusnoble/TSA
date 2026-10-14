#!/usr/bin/env bash
# SLURM wrapper around run_department.sh for ONE departement.
#
#   sbatch scripts/hpc/run_department.sbatch Paris
#
# =============================================================================
#  Adapted for TU Darmstadt Lichtenberg II (HRZ), 2026-10-14.
#  - No --partition: the site's job_submit plugin picks acc_short/acc/acc_long
#    from --time and --gres (any -p is overwritten).
#  - -n, --mem-per-cpu and -t are mandatory there; --mem is rejected.
#  - Every job must be charged to the active project: -A l0003541.
#  - GPU type: same price per GPU-hour for all types. The env (torch 2.1/cu118)
#    runs on v100/a100/h100 but NOT on rtx6000 (Blackwell). Override the type at
#    submission, e.g.  sbatch --gres=gpu:h100:1 scripts/hpc/run_department.sbatch X
#  - Long departements: submit as a serial array chain; each task resumes and the
#    finishing task cancels the remaining (pending) ones:
#      sbatch -J tsa-<Dept> --array=1-4%1 scripts/hpc/run_department.sbatch <Dept>
# =============================================================================
#SBATCH --job-name=tsa
#SBATCH -A l0003541
#SBATCH -n 1
#SBATCH -c 4                            # cores count as core-h (budget); pipeline is one process
#SBATCH --mem-per-cpu=8000              # MB per core -> 32 GB host RAM in total
#SBATCH -t 24:00:00                     # <= 24 h runs in 'acc' (most GPU nodes); resume via array chain
#SBATCH --gres=gpu:a100:1               # override at submission for h100 / v100
#SBATCH --signal=B:USR1@180             # USR1 to this script 3 min before the time limit
#SBATCH --output=logs/slurm-%x-%j.out   # %x=job name, %j=job id
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch $0 <Departement>" >&2
  exit 2
fi
DEPT="$1"

# Resolve the repo root from this script's location (works under sbatch, which
# copies the script but preserves the submit dir as SLURM_SUBMIT_DIR).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
mkdir -p logs

# --- Lichtenberg II environment ------------------------------------------------
# Miniforge lives on scratch (no site conda); ~/.tsa-hpc/bin holds an `aws`
# wrapper that adds --no-progress to `aws s3 cp` (keeps Slurm logs small).
export PATH="/work/scratch/${USER}/miniforge3/condabin:${HOME}/.tsa-hpc/bin:${PATH}"
export PYTHONNOUSERSITE=1          # ~/.local/lib/python3.10 would shadow cv2 & co. in the env
export MPLBACKEND=Agg
export HF_HOME="${HOME}/.cache/huggingface"   # camembert-base pre-cached on the login node
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AWS_PROFILE=tsa AWS_DEFAULT_REGION=eu-west-2   # keys live in ~/.aws/credentials, not here
export TSA_SCRATCH="/work/scratch/${USER}/tsa"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TSA_INSTANCES="${TSA_INSTANCES:-1}"   # >1: several inference instances share the GPU

echo "SLURM job ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-}) on $(hostname) starting departement: ${DEPT}"
echo "start: $(date -Is)"
nvidia-smi || echo "(nvidia-smi unavailable -- check that a GPU was allocated)"

# Departement slug, resolved exactly like run_department.sh (accent/case/separator-
# insensitive match in department_urls.csv); used only to scope the output dir.
fold() { printf '%s' "$1" | iconv -f UTF-8 -t ASCII//TRANSLIT 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d ' _-'; }
SLUG=""
while IFS=',' read -r dcol ucol; do
  [ "${dcol}" = "department" ] && continue
  if [ "$(fold "${dcol}")" = "$(fold "${DEPT}")" ]; then ucol="${ucol%$'\r'}"; SLUG="${ucol##*/}"; SLUG="${SLUG%.zip}"; break; fi
done < scripts/hpc/department_urls.csv
PAGES_DIR="${TSA_SCRATCH}/${SLUG:-__unknown__}/output/pages"
count_csv() { [ -d "${PAGES_DIR}" ] && find "${PAGES_DIR}" -type f -name '*.csv' | wc -l || echo 0; }
CSV_BEFORE=$(count_csv)
T0=$(date +%s)

# GPU utilisation trace (one line per minute) for throughput analysis.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  > "logs/gpu-${SLURM_JOB_ID:-local}.csv" 2>/dev/null &
SMI_PID=$!

# Near the time limit: stop inference, then delete CSVs written in the last few
# minutes for THIS departement (a page killed mid-write could leave a truncated
# CSV that resume would treat as done); they are recomputed by the next segment.
INFER_PATTERN="inference\.py --config \S*/${SLUG:-__none__}(\.s[0-9]+)?\.yaml"
on_usr1() {
  echo "USR1 at $(date -Is): time limit approaching, stopping inference."
  pkill -TERM -P "${CHILD}" 2>/dev/null || true          # run_department.sh's children
  pkill -TERM -f "${INFER_PATTERN}" 2>/dev/null || true  # every instance of this departement
  sleep 20
  pkill -KILL -f "${INFER_PATTERN}" 2>/dev/null || true
  pkill -KILL -P "${CHILD}" 2>/dev/null || true
  kill -TERM "${CHILD}" 2>/dev/null || true
  if [ -n "${SLUG}" ] && [ -d "${PAGES_DIR}" ]; then
    n=$(find "${PAGES_DIR}" -type f -name '*.csv' -mmin -3 | wc -l)
    find "${PAGES_DIR}" -type f -name '*.csv' -mmin -3 -delete
    echo "removed ${n} CSV(s) written in the last 3 min (recomputed next segment)"
  fi
  echo "segment: $(( $(count_csv) - CSV_BEFORE )) pages in $(( $(date +%s) - T0 )) s"
  kill "${SMI_PID}" 2>/dev/null || true
  exit 0    # not a failure: the next array task resumes
}
trap on_usr1 USR1

# Run in the background and wait, so the USR1 trap can fire during inference.
bash scripts/hpc/run_department.sh "${DEPT}" &
CHILD=$!
set +e
wait "${CHILD}"
RC=$?
set -e
kill "${SMI_PID}" 2>/dev/null || true

echo "end: $(date -Is)  exit=${RC}  pages this segment: $(( $(count_csv) - CSV_BEFORE ))  wall: $(( $(date +%s) - T0 )) s"

# Finished (archive written): drop the remaining segments of an array chain.
if [ "${RC}" -eq 0 ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  scancel -t PENDING "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true
fi
exit "${RC}"
