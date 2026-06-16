#!/usr/bin/env bash
#
# 01_segment_bone.sh
# -------------------
# Batch-extract binary skull-bone masks from CT volumes using the BioImage Suite
# Web command-line tools (biswebnode). This replicates, in a reproducible/batched
# form, the interactive Dual Viewer workflow:
#
#   1) segmentimage  : 3-class histogram (k-means-style) segmentation of the CT.
#                      Output labels are 0 = air, 1 = soft tissue, 2 = bone.
#   2) thresholdimage: keep only the bone class (label 2) -> binary mask {0,1}.
#
# This matches the original pipeline (see *_segm3d / *_bone3d result logs):
# bone occupies ~3.9% of each CT volume.
#
# Usage:
#   bash scripts/01_segment_bone.sh                 # process all patients
#   bash scripts/01_segment_bone.sh 1BA001 1BA005   # process a subset
#
# Idempotent/resumable: patients whose final bone mask already exists are skipped.

set -euo pipefail

# --- Paths ---------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/mr-ct-data"
SEG_DIR="${ROOT_DIR}/derivatives/segm3d"   # intermediate 3-class label maps
BONE_DIR="${ROOT_DIR}/derivatives/bone"    # final binary bone masks
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${SEG_DIR}" "${BONE_DIR}" "${LOG_DIR}"

# --- Segmentation parameters (match original results logs) ---------------------
NUMCLASSES=3        # air / soft tissue / bone
MAXSIGMARATIO=0.2
NUMBINS=256
SMOOTHHISTO=true
BONE_LABEL=2        # highest-intensity class == bone (labels are 0-indexed)

BISWEB="$(command -v biswebnode || true)"
if [[ -z "${BISWEB}" ]]; then
  echo "ERROR: biswebnode not found on PATH. Install with: npm install -g biswebnode" >&2
  exit 1
fi

# --- Patient list --------------------------------------------------------------
if [[ "$#" -gt 0 ]]; then
  PATIENTS=("$@")
else
  PATIENTS=()
  while IFS= read -r d; do PATIENTS+=("$(basename "$d")"); done \
    < <(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name '1B*' | sort)
fi

SUMMARY="${LOG_DIR}/segment_summary.csv"
echo "patient,status,bone_voxels,seconds" > "${SUMMARY}"

echo "Found ${#PATIENTS[@]} patient(s) to consider."
n_done=0; n_skip=0; n_fail=0

for pid in "${PATIENTS[@]}"; do
  ct="${DATA_DIR}/${pid}/ct.nii.gz"
  seg="${SEG_DIR}/${pid}_segm3d.nii.gz"
  bone="${BONE_DIR}/${pid}_bone.nii.gz"
  plog="${LOG_DIR}/${pid}.log"

  if [[ ! -f "${ct}" ]]; then
    echo "[SKIP] ${pid}: no ct.nii.gz"; echo "${pid},missing_ct,,0" >> "${SUMMARY}"
    n_skip=$((n_skip+1)); continue
  fi
  if [[ -f "${bone}" ]]; then
    echo "[SKIP] ${pid}: bone mask exists"; echo "${pid},exists,,0" >> "${SUMMARY}"
    n_skip=$((n_skip+1)); continue
  fi

  echo "[RUN ] ${pid} ..."
  t0=$SECONDS
  {
    "${BISWEB}" segmentimage -i "${ct}" -o "${seg}" \
      --numclasses "${NUMCLASSES}" --maxsigmaratio "${MAXSIGMARATIO}" \
      --numbins "${NUMBINS}" --smoothhisto "${SMOOTHHISTO}"

    "${BISWEB}" thresholdimage -i "${seg}" -o "${bone}" \
      --low "${BONE_LABEL}" --high $((BONE_LABEL+1)) \
      --replacein true --replaceout true --inval 1 --outval 0 --outtype UChar
  } > "${plog}" 2>&1 || {
      echo "[FAIL] ${pid} (see ${plog})"
      echo "${pid},failed,,$((SECONDS-t0))" >> "${SUMMARY}"
      n_fail=$((n_fail+1)); continue
  }

  dt=$((SECONDS-t0))
  echo "[ OK ] ${pid} (${dt}s)"
  echo "${pid},ok,,${dt}" >> "${SUMMARY}"
  n_done=$((n_done+1))
done

echo "----------------------------------------"
echo "Done. ok=${n_done} skipped=${n_skip} failed=${n_fail}"
echo "Bone masks : ${BONE_DIR}"
echo "Summary    : ${SUMMARY}"
