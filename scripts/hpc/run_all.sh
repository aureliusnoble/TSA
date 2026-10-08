#!/usr/bin/env bash
# Process EVERY downloadable departement, in digitisation-priority order,
# one after another, on the current machine (no SLURM).
#
#   bash scripts/hpc/run_all.sh
#
# Rotation means only one departement's images live on scratch at a time:
# run_department.sh deletes a departement's images as soon as its archive is
# written, before the next departement is downloaded.
#
# departments.csv is pre-sorted by priority (Paris first, then the fully-intact
# departements largest-first, then the incomplete ones). Rows with no S3 URL
# (e.g. Seine-Saint-Denis, which has no zip) are skipped automatically.
#
# A departement that fails does NOT stop the batch; failures are listed at the
# end. Already-completed departements (archive present) are skipped instantly.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPTS_CSV="${SCRIPT_DIR}/departments.csv"
cd "${REPO_ROOT}"

ok=(); failed=(); skipped=()

# Column order: department,priority,tier,size_gb,pages,pages_missing,volumes,zip_name,s3_url,in_metadata,in_s3_links
while IFS=',' read -r dept priority tier size_gb pages pmiss vols zipn url inmeta inlinks; do
  [ "${dept}" = "department" ] && continue
  [ -z "${dept}" ] && continue
  inlinks="${inlinks%$'\r'}"   # tolerate CRLF-edited CSVs
  if [ -z "${url}" ] || [ "${inlinks}" = "no" ]; then
    echo ">>> SKIP ${dept}: no S3 zip available."
    skipped+=("${dept}")
    continue
  fi
  echo ""
  echo "############################################################"
  echo "# ${dept}  (priority ${priority}, ~${size_gb} GB, ${pages} pages)"
  echo "############################################################"
  if bash scripts/hpc/run_department.sh "${dept}"; then
    ok+=("${dept}")
  else
    echo ">>> FAILED: ${dept} (continuing with the next departement)"
    failed+=("${dept}")
  fi
done < "${DEPTS_CSV}"

echo ""
echo "================= run_all summary ================="
echo "completed (${#ok[@]}): ${ok[*]:-none}"
echo "skipped   (${#skipped[@]}): ${skipped[*]:-none}"
echo "failed    (${#failed[@]}): ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]

# =============================================================================
# SLURM alternative: submit one job per departement instead of looping here.
#
# Simple fan-out (all independent; the cluster schedules them as GPUs free up).
# Because each job does its own rotation, running several in parallel needs
# ~N x the per-departement scratch high-water mark -- see README_HPC.md.
#
#   while IFS=',' read -r dept _pri _tier _sz _pg _pm _v _z url _im inlinks; do
#     [ "$dept" = "department" ] && continue
#     [ -z "$url" ] && continue
#     sbatch scripts/hpc/run_department.sbatch "$dept"
#   done < scripts/hpc/departments.csv
#
# Strict rotation (one at a time, to cap scratch at a single departement):
# chain jobs with a dependency so each starts only after the previous finishes.
#
#   prev=""
#   while IFS=',' read -r dept _pri _tier _sz _pg _pm _v _z url _im inlinks; do
#     [ "$dept" = "department" ] && continue
#     [ -z "$url" ] && continue
#     if [ -n "$prev" ]; then dep="--dependency=afterany:$prev"; else dep=""; fi
#     jid=$(sbatch $dep --parsable scripts/hpc/run_department.sbatch "$dept")
#     echo "submitted $dept as job $jid"
#     prev="$jid"
#   done < scripts/hpc/departments.csv
# =============================================================================
