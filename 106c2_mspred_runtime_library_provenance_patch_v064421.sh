#!/usr/bin/env bash
set -euo pipefail

VERSION="v0.6.4.4.2.1"

BASE_IMAGE="${MSPRED_BASE_IMAGE:-naa-ms-pred:cd616f6-cpu-amd64}"
PATCHED_IMAGE="${MSPRED_PATCHED_IMAGE:-naa-ms-pred:cd616f6-cpu-amd64-r2}"
EXPECTED_COMMIT="cd616f629a82040b950fbf8f0a983cc70ce1cf3b"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUT="$PROJECT_ROOT/data/processed/prediction_v064421_runtime_patch"
QC="$PROJECT_ROOT/qc"

mkdir -p "$OUT" "$QC"

PATCH_DOCKERFILE="$OUT/Dockerfile.runtime_patch"
BUILD_LOG="$OUT/MSPRED_RuntimePatch_Build.log"
VERIFY_LOG="$OUT/MSPRED_RuntimePatch_Verification.txt"
GLACIER_HELP="$OUT/GLACIER_help_after_runtime_patch.txt"
PROV="$OUT/MSPRED_RuntimeLibrary_Provenance_v064421.tsv"
NOTES="$OUT/MSPRED_RuntimeLibrary_Provenance_Notes.txt"
QCFILE="$QC/106c2_mspred_runtime_library_provenance_patch_v064421.tsv"

echo
echo "ms-pred runtime-library provenance patch ${VERSION}"
echo "===================================================="
echo "Base image:    $BASE_IMAGE"
echo "Patched image: $PATCHED_IMAGE"
echo

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker not found." >&2
  exit 1
}

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: base Docker image not found: $BASE_IMAGE" >&2
  exit 1
}

BASE_IMAGE_ID="$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')"
BASE_CREATED="$(docker image inspect "$BASE_IMAGE" --format '{{.Created}}')"
BASE_OS="$(docker image inspect "$BASE_IMAGE" --format '{{.Os}}')"
BASE_ARCH="$(docker image inspect "$BASE_IMAGE" --format '{{.Architecture}}')"

cat > "$PATCH_DOCKERFILE" <<EOF
FROM ${BASE_IMAGE}

USER root

RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       libxrender1 \\
       libxext6 \\
       libsm6 \\
    && rm -rf /var/lib/apt/lists/*

RUN /opt/ms-pred/.venv/bin/python -c "from rdkit.Chem import Draw; from rdkit.Chem.Draw import rdMolDraw2D; print('RDKit Draw import: OK')"

WORKDIR /opt/ms-pred

CMD ["bash"]
EOF

echo "Building patched runtime image..."
docker build \
  --platform linux/amd64 \
  -t "$PATCHED_IMAGE" \
  -f "$PATCH_DOCKERFILE" \
  "$PROJECT_ROOT" 2>&1 | tee "$BUILD_LOG"

PATCHED_IMAGE_ID="$(docker image inspect "$PATCHED_IMAGE" --format '{{.Id}}')"
PATCHED_CREATED="$(docker image inspect "$PATCHED_IMAGE" --format '{{.Created}}')"
PATCHED_OS="$(docker image inspect "$PATCHED_IMAGE" --format '{{.Os}}')"
PATCHED_ARCH="$(docker image inspect "$PATCHED_IMAGE" --format '{{.Architecture}}')"

echo
echo "Verifying patched runtime..."

docker run --rm --platform linux/amd64 "$PATCHED_IMAGE" bash -c '
set -euo pipefail
cd /opt/ms-pred

echo "machine=$(uname -m)"
echo "python=$(/opt/ms-pred/.venv/bin/python --version 2>&1)"
echo "git_commit=$(git rev-parse HEAD)"

echo "git_status_begin"
git status --short
echo "git_status_end"

echo "uv_lock_sha256=$(sha256sum uv.lock | awk "{print \$1}")"

echo "runtime_library_check_begin"
ldconfig -p | grep -E "libXrender\.so\.1|libXext\.so\.6|libSM\.so\.6" || true
echo "runtime_library_check_end"

/opt/ms-pred/.venv/bin/python - <<PY
import platform
import torch
import dgl
import rdkit
import ms_pred
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

print("platform_machine=" + platform.machine())
print("torch_version=" + torch.__version__)
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("dgl_version=" + dgl.__version__)
print("rdkit_version=" + rdkit.__version__)
print("rdkit_draw_import=PASS")
print("ms_pred_import=PASS")
PY
' | tee "$VERIFY_LOG"

echo
echo "Testing exact GLACIER entry-point import/help..."

set +e
docker run --rm --platform linux/amd64 "$PATCHED_IMAGE" bash -c '
cd /opt/ms-pred
/opt/ms-pred/.venv/bin/python src/ms_pred/glacier/predict_smis_joint.py --help
' > "$GLACIER_HELP" 2>&1
GLACIER_HELP_RC=$?
set -e

cat "$GLACIER_HELP"

getv() {
  local key="$1"
  grep "^${key}=" "$VERIFY_LOG" | tail -n1 | cut -d= -f2-
}

MACHINE="$(getv machine)"
PYTHON_VERSION="$(getv python)"
FROZEN_COMMIT="$(getv git_commit)"
UV_LOCK_SHA="$(getv uv_lock_sha256)"
PLATFORM_MACHINE="$(getv platform_machine)"
TORCH_VERSION="$(getv torch_version)"
TORCH_CUDA="$(getv torch_cuda_available)"
DGL_VERSION="$(getv dgl_version)"
RDKIT_VERSION="$(getv rdkit_version)"
RDKIT_DRAW="$(getv rdkit_draw_import)"
MSPRED_IMPORT="$(getv ms_pred_import)"

GIT_STATUS="$(
  awk '
    /^git_status_begin$/ {flag=1; next}
    /^git_status_end$/   {flag=0}
    flag {print}
  ' "$VERIFY_LOG"
)"

EXPECTED_GIT_STATUS="?? uv.lock"

LIBXRENDER_PRESENT="$(
  grep -q "libXrender.so.1" "$VERIFY_LOG" && echo YES || echo NO
)"
LIBXEXT_PRESENT="$(
  grep -q "libXext.so.6" "$VERIFY_LOG" && echo YES || echo NO
)"
LIBSM_PRESENT="$(
  grep -q "libSM.so.6" "$VERIFY_LOG" && echo YES || echo NO
)"

STATUS="PASS"

[[ "$BASE_OS" == "linux" ]] || STATUS="FAIL"
[[ "$BASE_ARCH" == "amd64" ]] || STATUS="FAIL"
[[ "$PATCHED_OS" == "linux" ]] || STATUS="FAIL"
[[ "$PATCHED_ARCH" == "amd64" ]] || STATUS="FAIL"
[[ "$MACHINE" == "x86_64" ]] || STATUS="FAIL"
[[ "$PLATFORM_MACHINE" == "x86_64" ]] || STATUS="FAIL"
[[ "$FROZEN_COMMIT" == "$EXPECTED_COMMIT" ]] || STATUS="FAIL"
[[ "$GIT_STATUS" == "$EXPECTED_GIT_STATUS" ]] || STATUS="FAIL"
[[ "$LIBXRENDER_PRESENT" == "YES" ]] || STATUS="FAIL"
[[ "$LIBXEXT_PRESENT" == "YES" ]] || STATUS="FAIL"
[[ "$LIBSM_PRESENT" == "YES" ]] || STATUS="FAIL"
[[ "$RDKIT_DRAW" == "PASS" ]] || STATUS="FAIL"
[[ "$MSPRED_IMPORT" == "PASS" ]] || STATUS="FAIL"
[[ "$GLACIER_HELP_RC" == "0" ]] || STATUS="FAIL"

printf "Field\tValue\n" > "$PROV"
printf "AuditVersion\t%s\n" "$VERSION" >> "$PROV"
printf "BaseImageTag\t%s\n" "$BASE_IMAGE" >> "$PROV"
printf "BaseImageID\t%s\n" "$BASE_IMAGE_ID" >> "$PROV"
printf "BaseImageCreated\t%s\n" "$BASE_CREATED" >> "$PROV"
printf "PatchedImageTag\t%s\n" "$PATCHED_IMAGE" >> "$PROV"
printf "PatchedImageID\t%s\n" "$PATCHED_IMAGE_ID" >> "$PROV"
printf "PatchedImageCreated\t%s\n" "$PATCHED_CREATED" >> "$PROV"
printf "ImageOS\t%s\n" "$PATCHED_OS" >> "$PROV"
printf "ImageArchitecture\t%s\n" "$PATCHED_ARCH" >> "$PROV"
printf "RuntimeMachine\t%s\n" "$MACHINE" >> "$PROV"
printf "Python\t%s\n" "$PYTHON_VERSION" >> "$PROV"
printf "MSPredCommit\t%s\n" "$FROZEN_COMMIT" >> "$PROV"
printf "GitStatus\t%s\n" "$GIT_STATUS" >> "$PROV"
printf "UVLockSHA256\t%s\n" "$UV_LOCK_SHA" >> "$PROV"
printf "AddedRuntimeLibrary\tlibxrender1\n" >> "$PROV"
printf "AddedRuntimeLibrary\tlibxext6\n" >> "$PROV"
printf "AddedRuntimeLibrary\tlibsm6\n" >> "$PROV"
printf "LibXrenderPresent\t%s\n" "$LIBXRENDER_PRESENT" >> "$PROV"
printf "LibXextPresent\t%s\n" "$LIBXEXT_PRESENT" >> "$PROV"
printf "LibSMPresent\t%s\n" "$LIBSM_PRESENT" >> "$PROV"
printf "TorchVersion\t%s\n" "$TORCH_VERSION" >> "$PROV"
printf "TorchCUDAAvailable\t%s\n" "$TORCH_CUDA" >> "$PROV"
printf "DGLVersion\t%s\n" "$DGL_VERSION" >> "$PROV"
printf "RDKitVersion\t%s\n" "$RDKIT_VERSION" >> "$PROV"
printf "RDKitDrawImport\t%s\n" "$RDKIT_DRAW" >> "$PROV"
printf "MSPredImport\t%s\n" "$MSPRED_IMPORT" >> "$PROV"
printf "GLACIERHelpReturnCode\t%s\n" "$GLACIER_HELP_RC" >> "$PROV"
printf "VerificationStatus\t%s\n" "$STATUS" >> "$PROV"

printf "Check\tExpected\tObserved\tStatus\n" > "$QCFILE"

add_qc() {
  local check="$1"
  local expected="$2"
  local observed="$3"
  local qstatus="FAIL"
  [[ "$expected" == "$observed" ]] && qstatus="PASS"
  printf "%s\t%s\t%s\t%s\n" "$check" "$expected" "$observed" "$qstatus" >> "$QCFILE"
}

add_qc "Patched image OS" "linux" "$PATCHED_OS"
add_qc "Patched image architecture" "amd64" "$PATCHED_ARCH"
add_qc "Runtime machine" "x86_64" "$MACHINE"
add_qc "ms-pred commit unchanged" "$EXPECTED_COMMIT" "$FROZEN_COMMIT"
add_qc "Git status unchanged except generated lockfile" "$EXPECTED_GIT_STATUS" "$GIT_STATUS"
add_qc "libXrender.so.1 present" "YES" "$LIBXRENDER_PRESENT"
add_qc "libXext.so.6 present" "YES" "$LIBXEXT_PRESENT"
add_qc "libSM.so.6 present" "YES" "$LIBSM_PRESENT"
add_qc "RDKit Draw import" "PASS" "$RDKIT_DRAW"
add_qc "ms_pred import" "PASS" "$MSPRED_IMPORT"
add_qc "GLACIER --help return code" "0" "$GLACIER_HELP_RC"

cat > "$NOTES" <<EOF
ms-pred runtime-library provenance patch ${VERSION}

PURPOSE
=======
Create a derived runtime image that preserves the already frozen ms-pred
software environment while adding only the Linux shared libraries required
for RDKit drawing imports used by GLACIER.

BASE IMAGE
==========
${BASE_IMAGE}
${BASE_IMAGE_ID}

PATCHED IMAGE
=============
${PATCHED_IMAGE}
${PATCHED_IMAGE_ID}

FROZEN SOURCE COMMIT
====================
${FROZEN_COMMIT}

ADDED RUNTIME PACKAGES
======================
libxrender1
libxext6
libsm6

WHY
===
GLACIER imports ms_pred.common -> plot_utils -> rdkit.Chem.Draw ->
rdMolDraw2D. The original software image lacked libXrender.so.1, causing
GLACIER to fail before argparse/--help.

This patch does NOT alter:
    ms-pred source commit
    uv.lock dependency resolution
    PyTorch version
    DGL version
    RDKit Python package version
    GLACIER checkpoint
    ICEBERG checkpoints
    benchmark inputs

The original image remains preserved and is not overwritten.

VERIFICATION
============
Runtime architecture:
${MACHINE}

RDKit Draw import:
${RDKIT_DRAW}

GLACIER --help return code:
${GLACIER_HELP_RC}

STATUS
======
${STATUS}

NEXT
====
If PASS, rerun the Tier-1 smoke test using:

    MSPRED_IMAGE=${PATCHED_IMAGE} \\
    python3 scripts/prediction/107_run_mspred_tier1_smoke_test_v0645.py

No checkpoint reacquisition is required.
EOF

echo
echo "Provenance"
echo "----------"
cat "$PROV"

echo
echo "QC"
echo "--"
cat "$QCFILE"

echo
echo "Saved"
echo "-----"
echo "$PATCH_DOCKERFILE"
echo "$BUILD_LOG"
echo "$VERIFY_LOG"
echo "$GLACIER_HELP"
echo "$PROV"
echo "$NOTES"
echo "$QCFILE"

echo
if [[ "$STATUS" == "PASS" ]]; then
  echo "MS-PRED RUNTIME-LIBRARY PROVENANCE PATCH ${VERSION} PASSED"
else
  echo "MS-PRED RUNTIME-LIBRARY PROVENANCE PATCH ${VERSION} FAILED" >&2
  exit 1
fi
