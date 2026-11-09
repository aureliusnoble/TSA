#!/usr/bin/env bash
# Submit every departement (priority order of departments.csv) to SLURM on
# Lichtenberg II as chains of 24 h jobs (see shape_for).
#
#   bash scripts/hpc/submit_all_lichtenberg.sh            # submit
#   bash scripts/hpc/submit_all_lichtenberg.sh --dry-run  # sbatch --test-only
#   options: --only REGEX (departement names), --after JOBID[:JOBID] (afterok dependency)
#
# - GPU type is flexible (any V100/A100/H100): --gres=gpu:nvd:N with the
#   Blackwell rtx6000 nodes (gboe) excluded, as torch 2.1/cu118 can't use them.
# - Jobs get 1-3 GPUs by size, one inference instance per GPU
#   (TSA_INSTANCES); see shape_for() below.
# - One core and 12 GB host RAM per GPU (each instance is one busy process).
# - Safe to re-run: skips departements that already have a results archive or
#   a queued/running job (job name tsa-<slug>), so it never duplicates work.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SBATCH=/opt/slurm/current/bin/sbatch      # real sbatch: clean --parsable output
RESULTS_DIR="${TSA_RESULTS_DIR:-${REPO_ROOT}/results}"
DRY=""
ONLY=""         # --only REGEX: submit only departements whose name matches
DEP=""          # --after JOBID[:JOBID...]: start only after these jobs succeed
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--test-only" ;;
    --only)    ONLY="$2"; shift ;;
    --after)   DEP="--dependency=afterok:$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p logs

# Page counts in departments.csv are stale for two departements (the zips
# were re-uploaded with more volumes); use the counts from the zips.
declare -A ZIP_PAGES=([Pas-de-Calais]=8612 [Bouches-du-Rhone]=19745
  [Nievre_v2]=1597)   # Nievre_v2: only the re-scraped pages are processed; the rest are seeded

# Job shape by size: every job is a chain of 24 h segments (partition acc).
# One-day jobs can use every GPU node (acc_long excludes several) and are much
# easier for the backfill scheduler to place, which matters once our fair-share
# priority has dropped. 7-day jobs keep their GPUs longer but start later.
# Small departements get 2 GPUs to finish within about a day; mid-size ones 1
# GPU (single GPUs are found fastest); the largest 2.
# Segments: expected days at ~230 pages/h per GPU, plus two spare.
shape_for() {  # pages -> "gpus time segments"
  local p=$1 g
  if   [ "$p" -lt  5000 ]; then g=1
  elif [ "$p" -lt 11000 ]; then g=2
  elif [ "$p" -lt 20000 ]; then g=1
  else                          g=2; fi
  echo "${g} 1-00:00:00 $(( (p + g*230*23 - 1) / (g*230*23) + 2 ))"
}

active=$(squeue -h -u "${USER}" -o '%j' | sort -u)
while IFS=',' read -r dept _pri _tier _sz pages _pm _v zipn url _im inlinks; do
  [ "${dept}" = "department" ] && continue
  [ -z "${url}" ] && continue
  slug="${zipn%.zip}"
  name="tsa-${slug}"
  if [ -n "${ONLY}" ] && ! [[ "${slug}" =~ ${ONLY} ]]; then continue; fi
  if [ -f "${RESULTS_DIR}/${slug}.tar.gz" ]; then
    echo "skip ${dept}: archive exists"; continue
  fi
  if grep -qx "${name}" <<<"${active}"; then
    echo "skip ${dept}: job ${name} already queued/running"; continue
  fi
  pages="${ZIP_PAGES[${slug}]:-${pages}}"
  read -r g time segs <<<"$(shape_for "${pages}")"
  out=$(${SBATCH} ${DRY} ${DEP} --parsable -J "${name}" --array="1-${segs}%1" -t "${time}" \
          -c "${g}" --mem-per-cpu=12000 --gres="gpu:nvd:${g}" -x 'gboe[0001-0008]' \
          --export="ALL,TSA_INSTANCES=${g}" \
          scripts/hpc/run_department.sbatch "${dept}" 2>&1)
  echo "${dept}: ${pages} pages, ${g} GPU(s), ${time} x up to ${segs} -> $(tail -n1 <<<"${out}")"
done < scripts/hpc/departments.csv
