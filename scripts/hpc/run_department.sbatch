#!/usr/bin/env bash
# SLURM wrapper around run_department.sh for ONE departement.
#
#   sbatch scripts/hpc/run_department.sbatch Paris
#
# =============================================================================
#  !!! CLUSTER-SPECIFIC PLACEHOLDERS -- YOU MUST ADJUST THESE !!!
#  The target cluster's partition names, GPU resource syntax, memory and time
#  limits are UNKNOWN. The values below are PLACEHOLDERS that will NOT be right
#  for your site. Check `sinfo`, your cluster docs, or ask an admin, then edit:
#    --partition : the name of a GPU partition on THIS cluster (guessed: gpu)
#    --gres      : GPU request syntax varies (gpu:1, gpu:a100:1, ...)
#    --mem       : the 1500px line model needs a lot of RAM; 32G is a guess
#    --time      : large depts (Saone-et-Loire ~47k pages) can run for many
#                  hours; 24:00:00 is a placeholder, not a measured figure
#    --cpus-per-task : image decode / unzip benefit from a few cores
# =============================================================================
#SBATCH --job-name=tsa
#SBATCH --partition=gpu                 # <-- ADJUST: GPU partition name
#SBATCH --gres=gpu:1                    # <-- ADJUST: 1 GPU (syntax is site-specific)
#SBATCH --cpus-per-task=4               # <-- ADJUST
#SBATCH --mem=32G                       # <-- ADJUST (1500px line model is memory-hungry)
#SBATCH --time=24:00:00                 # <-- ADJUST (placeholder; depends on page count)
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

echo "SLURM job ${SLURM_JOB_ID:-local} on $(hostname) starting departement: ${DEPT}"
nvidia-smi || echo "(nvidia-smi unavailable -- check that a GPU was allocated)"

bash scripts/hpc/run_department.sh "${DEPT}"
