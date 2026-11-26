#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Populates Source/ThirdParty/IREE/Linux from a local IREE runtime build.
#
# Usage:
#   ./build_iree_linux.sh <iree_install_root> [iree_source_root] [iree_build_root]
#
# Does not download anything. See README.md for how to build the IREE runtime.
#
# IREE's install step leaves out two things Schola needs:
#   - C++ headers such as iree/vm/ref_cc.h, which iree/vm/ref.h includes from C++. Copied from
#     <iree_source_root>/runtime/src without overwriting installed (or generated) headers.
#   - libflatcc_runtime.a / libflatcc_parsing.a, which the bytecode module verifier links against.
#     Copied from <iree_build_root>/build_tools/third_party/flatcc.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <iree_install_root> [iree_source_root] [iree_build_root]" >&2
  exit 1
fi

IREE_ROOT="$1"
IREE_SOURCE_ROOT="${2:-}"
IREE_BUILD_ROOT="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_THIRD_PARTY_DIR="${SCRIPT_DIR}/../../../Source/ThirdParty/IREE"
DEST_ROOT="${IREE_THIRD_PARTY_DIR}/Linux"
DEST_INCLUDE="${DEST_ROOT}/include"
DEST_LIB="${DEST_ROOT}/lib"
FLATCC_LIBS=(libflatcc_runtime.a libflatcc_parsing.a)
# The unified library is a single archive of the whole runtime (see iree_cc_unified_library in
# runtime/src/iree/runtime/CMakeLists.txt); the per-component libs duplicate it, so only it is copied.
UNIFIED_LIB=libiree_runtime_unified.a
SCHOLA_IREE_SOURCE="${SCRIPT_DIR}/../../../Source/ScholaIREE"

find_include() {
  for candidate in "${IREE_ROOT}/include" "${IREE_ROOT}/iree-install/include" "${IREE_ROOT}"; do
  if [[ -f "${candidate}/iree/runtime/api.h" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

find_lib() {
  for candidate in "${IREE_ROOT}/lib" "${IREE_ROOT}/iree-install/lib" "${IREE_ROOT}"; do
    if compgen -G "${candidate}/*.a" > /dev/null; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

INCLUDE_ROOT="$(find_include)" || { echo "Could not find iree/runtime/api.h under ${IREE_ROOT}" >&2; exit 1; }
LIB_ROOT="$(find_lib)" || { echo "Could not find *.a under ${IREE_ROOT}" >&2; exit 1; }

mkdir -p "${DEST_INCLUDE}" "${DEST_LIB}"

# Start clean so stale headers/libs from earlier fetches do not accumulate (or get linked twice).
find "${DEST_INCLUDE}" "${DEST_LIB}" -mindepth 1 ! -name .gitkeep -delete 2>/dev/null || true

echo "Copying headers from ${INCLUDE_ROOT} -> ${DEST_INCLUDE}"
cp -a "${INCLUDE_ROOT}/." "${DEST_INCLUDE}/"

if [[ -n "${IREE_SOURCE_ROOT}" ]]; then
  RUNTIME_SRC="${IREE_SOURCE_ROOT}/runtime/src"
  [[ -d "${RUNTIME_SRC}" ]] || { echo "Could not find runtime/src under ${IREE_SOURCE_ROOT}" >&2; exit 1; }
  echo "Adding headers missing from the install from ${RUNTIME_SRC}"
  (cd "${RUNTIME_SRC}" && find . -name '*.h' -print0 | while IFS= read -r -d '' header; do
    if [[ ! -e "${DEST_INCLUDE}/${header}" ]]; then
      mkdir -p "${DEST_INCLUDE}/$(dirname "${header}")"
      cp -a "${header}" "${DEST_INCLUDE}/${header}"
    fi
  done)
fi

echo "Copying ${UNIFIED_LIB} from ${LIB_ROOT} -> ${DEST_LIB}"
[[ -f "${LIB_ROOT}/${UNIFIED_LIB}" ]] || { echo "Could not find ${LIB_ROOT}/${UNIFIED_LIB}" >&2; exit 1; }
cp -a "${LIB_ROOT}/${UNIFIED_LIB}" "${DEST_LIB}/"
# Windows' unified archive omits the MASM ELF-call thunks, so the .ps1 also copies the elf_arch lib. On Linux
# those are inline C in the unified lib already; copy the arch lib too if the install has it (harmless).
[[ -f "${LIB_ROOT}/libiree_hal_local_elf_arch.a" ]] && cp -a "${LIB_ROOT}/libiree_hal_local_elf_arch.a" "${DEST_LIB}/" || true

if [[ -n "${IREE_BUILD_ROOT}" ]]; then
  FLATCC_DIR="${IREE_BUILD_ROOT}/build_tools/third_party/flatcc"
  for lib in "${FLATCC_LIBS[@]}"; do
    [[ -f "${FLATCC_DIR}/${lib}" ]] || { echo "Could not find ${FLATCC_DIR}/${lib}" >&2; exit 1; }
    cp -a "${FLATCC_DIR}/${lib}" "${DEST_LIB}/"
  done
fi

echo "Pruning headers to those used by ScholaIREE"
python3 "${SCRIPT_DIR}/prune_iree.py" "${DEST_INCLUDE}" "${SCHOLA_IREE_SOURCE}"

missing=()
[[ -f "${DEST_LIB}/${UNIFIED_LIB}" ]] || missing+=("${UNIFIED_LIB}")
[[ -f "${DEST_INCLUDE}/iree/runtime/api.h" ]] || missing+=("iree/runtime/api.h")
[[ -f "${DEST_INCLUDE}/iree/vm/ref_cc.h" ]] || missing+=("iree/vm/ref_cc.h (pass iree_source_root)")
for lib in "${FLATCC_LIBS[@]}"; do
  [[ -f "${DEST_LIB}/${lib}" ]] || missing+=("${lib} (pass iree_build_root)")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "IREE layout at ${DEST_ROOT} is incomplete. Missing: ${missing[*]}" >&2
  exit 1
fi

touch "${IREE_THIRD_PARTY_DIR}/IREE.Build.cs"
echo "IREE Linux third-party layout ready at ${DEST_ROOT}"
