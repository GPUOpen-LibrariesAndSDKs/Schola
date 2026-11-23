#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Prepare a Lyra Starter Game project (Unreal Engine 5.8.4, macOS arm64) to
# build with this Schola checkout. See Docs/Lyra-Integration.md.
#
# What it does, in order:
#   1. Validates the host, the Unreal Engine root and the Lyra project.
#   2. Checks that this checkout contains the Schola UObject pointer fixes
#      (PR #56) that a Lyra + UE 5.8 build needs.
#   3. Checks the macOS gRPC/protobuf archives that Schola links against.
#   4. Checks the state of <Lyra>/Plugins/Schola.
#   5. Checks whether Tools/Lyra/patches/commonuser-ue58-mac.patch applies to
#      Lyra's CommonUser plugin.
# Steps 1-5 only read. If any of them reports an error, nothing is changed.
# Otherwise the script then:
#   6. Creates the <Lyra>/Plugins/Schola symlink to this checkout if it is
#      absent.
#   7. Applies the CommonUser patch to the files that are still unpatched.
#
# Safety rules:
#   * Existing files and directories at Plugins/Schola are never touched. An
#     existing symlink to somewhere else is only replaced with
#     --replace-schola-link, and only the link itself is removed.
#   * The patch is applied only if every file is either unpatched or already
#     patched. Unknown or locally modified sources are reported and left
#     alone. Patching is all-or-nothing and re-running it is a no-op.
#   * gRPC archives are only rebuilt with --build-deps, and only when the
#     existing ones are missing or unusable.
#
# Usage:
#   Tools/Lyra/setup_mac.sh --lyra-project /path/to/LyraStarterGame \
#                           --ue-root /path/to/UE_5.8 [options]
#
# Arguments (a flag wins over its environment variable):
#   --lyra-project PATH   Lyra project directory (or its .uproject file).
#                         Env: LYRA_PROJECT_DIR
#   --ue-root PATH        Unreal Engine root, the directory that contains
#                         Engine/Build. Env: UE_ENGINE_ROOT
#
# Options:
#   --check-only          Report everything, change nothing.
#   --verify              Like --check-only, but fail unless the project is
#                         already fully set up (Plugins/Schola linked, patch
#                         applied, gRPC archives present). Used by build_mac.sh.
#   --no-patch            Do not apply the CommonUser patch.
#   --revert-patch        Reverse the CommonUser patch (only if cleanly applied).
#   --replace-schola-link Replace an existing Plugins/Schola symlink that points
#                         somewhere else. Never removes a real directory.
#   --build-deps          Run Resources/Build/mac_dependencies.sh if the Mac
#                         gRPC archives are missing or unusable.
#   --allow-unfixed-schola
#                         Continue even though the Schola UObject pointer fixes
#                         (PR #56) are missing from this checkout. The Lyra
#                         build is expected to fail without them.
#   -h, --help            Show this help.
#
# Exit status: 0 when the setup is complete (or, with --check-only, would be
# complete without further manual steps); 1 otherwise.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
schola_root="$(cd "${script_dir}/../.." && pwd -P)"
patch_file="${script_dir}/patches/commonuser-ue58-mac.patch"
deps_script="${schola_root}/Resources/Build/mac_dependencies.sh"
grpc_mac_dir="${schola_root}/Source/ThirdParty/gRPC/Mac"

lyra_arg="${LYRA_PROJECT_DIR:-}"
ue_arg="${UE_ENGINE_ROOT:-}"
check_only=0
verify=0
apply_patch=1
revert_patch=0
replace_link=0
build_deps=0
allow_unfixed_schola=0

errors=0
warnings=0

info() { printf '    %s\n' "$*"; }
ok()   { printf '[ok] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; warnings=$((warnings + 1)); }
fail() { printf '[error] %s\n' "$*" >&2; errors=$((errors + 1)); }
die()  { printf '[error] %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lyra-project)  [[ $# -ge 2 ]] || die "--lyra-project needs a value."; lyra_arg="$2"; shift 2 ;;
        --lyra-project=*) lyra_arg="${1#*=}"; shift ;;
        --ue-root)       [[ $# -ge 2 ]] || die "--ue-root needs a value."; ue_arg="$2"; shift 2 ;;
        --ue-root=*)     ue_arg="${1#*=}"; shift ;;
        --check-only)    check_only=1; shift ;;
        --verify)        verify=1; check_only=1; shift ;;
        --no-patch)      apply_patch=0; shift ;;
        --revert-patch)  revert_patch=1; shift ;;
        --replace-schola-link) replace_link=1; shift ;;
        --build-deps)    build_deps=1; shift ;;
        --allow-unfixed-schola) allow_unfixed_schola=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "Unknown argument: $1 (see --help)." ;;
    esac
done

if [[ "${revert_patch}" == 1 && "${apply_patch}" == 0 ]]; then
    die "--revert-patch and --no-patch cannot be combined."
fi
[[ -n "${lyra_arg}" ]] || die "Lyra project path is required: pass --lyra-project or set LYRA_PROJECT_DIR."
[[ -n "${ue_arg}" ]]   || die "Unreal Engine root is required: pass --ue-root or set UE_ENGINE_ROOT."

# ---------------------------------------------------------------- host ----
echo "== Host"
if [[ "$(uname)" != "Darwin" ]]; then
    die "This script is for macOS. This machine is $(uname)."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    die "This script targets Apple Silicon (arm64). This shell reports $(uname -m); if it runs under Rosetta, start a native arm64 shell."
fi
command -v git >/dev/null 2>&1 || die "git is required (it is also used to apply the patch)."
if ! xcode-select -p >/dev/null 2>&1; then
    fail "Xcode command-line tools are not installed. Install Xcode, then run: xcode-select --install"
else
    ok "macOS arm64, Xcode developer dir: $(xcode-select -p)"
fi
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    warn "The Metal toolchain is not available. Run once: xcodebuild -downloadComponent MetalToolchain"
fi

# ----------------------------------------------------------- Unreal root ----
echo "== Unreal Engine"
ue_root=""
if [[ -d "${ue_arg}/Engine/Build" ]]; then
    ue_root="$(cd -P "${ue_arg}" && pwd -P)"
elif [[ "$(basename "${ue_arg}")" == "Engine" && -d "${ue_arg}/Build" ]]; then
    ue_root="$(cd -P "${ue_arg}/.." && pwd -P)"
    warn "--ue-root pointed at the Engine directory; using its parent ${ue_root}."
else
    fail "Not an Unreal Engine root (no Engine/Build): ${ue_arg}"
fi
if [[ -n "${ue_root}" ]]; then
    if [[ ! -x "${ue_root}/Engine/Build/BatchFiles/Mac/Build.sh" ]]; then
        fail "Missing ${ue_root}/Engine/Build/BatchFiles/Mac/Build.sh. Is this a Mac engine build?"
    fi
    version_file="${ue_root}/Engine/Build/Build.version"
    if [[ -f "${version_file}" ]]; then
        ue_major="$(sed -n 's/.*"MajorVersion"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "${version_file}" | head -n 1)"
        ue_minor="$(sed -n 's/.*"MinorVersion"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "${version_file}" | head -n 1)"
        ue_patch="$(sed -n 's/.*"PatchVersion"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "${version_file}" | head -n 1)"
        if [[ "${ue_major}.${ue_minor}" != "5.8" ]]; then
            fail "Unreal Engine ${ue_major}.${ue_minor}.${ue_patch} found; this setup is for 5.8.x (tested on 5.8.4)."
        elif [[ "${ue_patch}" != "4" ]]; then
            warn "Unreal Engine 5.8.${ue_patch} found; the setup was written against 5.8.4."
            ok "Unreal Engine root: ${ue_root}"
        else
            ok "Unreal Engine 5.8.4 at ${ue_root}"
        fi
    else
        warn "Cannot read ${version_file}; the engine version was not verified."
    fi
fi

# ------------------------------------------------------------ Lyra project ----
echo "== Lyra project"
lyra_dir=""
lyra_candidate="${lyra_arg}"
if [[ -f "${lyra_candidate}" && "${lyra_candidate}" == *.uproject ]]; then
    lyra_candidate="$(dirname "${lyra_candidate}")"
fi
if [[ -d "${lyra_candidate}" ]]; then
    lyra_dir="$(cd -P "${lyra_candidate}" && pwd -P)"
    if [[ ! -f "${lyra_dir}/LyraStarterGame.uproject" ]]; then
        fail "No LyraStarterGame.uproject in ${lyra_dir}."
        lyra_dir=""
    fi
else
    fail "Lyra project directory does not exist: ${lyra_arg}"
fi
if [[ -n "${lyra_dir}" ]]; then
    case "${lyra_dir}/" in
        "${schola_root}/"*)
            fail "The Lyra project (${lyra_dir}) is inside this Schola checkout. Linking Plugins/Schola would recurse; use a Lyra project outside the repository."
            lyra_dir=""
            ;;
    esac
fi
if [[ -n "${lyra_dir}" ]]; then
    ok "Lyra project: ${lyra_dir}"
fi

# --------------------------------------------- Schola source (PR #56 fixes) ----
echo "== Schola source"
if [[ ! -f "${schola_root}/Schola.uplugin" ]]; then
    fail "Schola.uplugin not found at ${schola_root}; the script must stay inside the repository."
fi
missing_fixes=0
check_object_ptr() {
    local header="$1" type_name="$2"
    if [[ ! -f "${schola_root}/${header}" ]]; then
        fail "Missing ${header}."
        missing_fixes=1
    elif ! grep -Eq "TObjectPtr<${type_name}>[[:space:]]+Connector" "${schola_root}/${header}"; then
        info "${header}: Connector is a raw ${type_name}* (needs TObjectPtr<${type_name}>)"
        missing_fixes=1
    fi
}
check_object_ptr "Source/ScholaTraining/Public/TrainingUtils/GymConnectorManager.h" "UAbstractGymConnector"
check_object_ptr "Source/ScholaImitation/Public/ImitationUtils/ImitationConnectorManager.h" "UAbstractImitationConnector"
if [[ "${missing_fixes}" == 0 ]]; then
    ok "UObject pointer fixes (PR #56) are present."
elif [[ "${allow_unfixed_schola}" == 1 ]]; then
    warn "PR #56 UObject pointer fixes are missing from ${schola_root}. Continuing because of --allow-unfixed-schola; the Lyra build is expected to fail until they are present."
else
    fail "PR #56 (Schola UObject pointer fixes) is not in this checkout. Merge PR #56 into your branch (or check out a branch that contains it) and run again. See Docs/Lyra-Integration.md."
fi

# ------------------------------------------------------------- gRPC archives ----
echo "== macOS gRPC archives"
grpc_problem=""
check_grpc() {
    grpc_problem=""
    local lib_dir="${grpc_mac_dir}/lib" inc_dir="${grpc_mac_dir}/include" name
    if [[ ! -f "${inc_dir}/grpcpp/grpcpp.h" ]]; then
        grpc_problem="missing headers (${inc_dir}/grpcpp/grpcpp.h)"; return 1
    fi
    if [[ ! -f "${inc_dir}/google/protobuf/message.h" ]]; then
        grpc_problem="missing protobuf headers (${inc_dir}/google/protobuf/message.h)"; return 1
    fi
    for name in libgrpc.a libgrpc++.a libgpr.a libprotobuf.a libz.a libcares.a libre2.a \
                libaddress_sorting.a libutf8_validity.a libupb_base_lib.a libabsl_base.a; do
        if [[ ! -f "${lib_dir}/${name}" ]]; then
            grpc_problem="missing ${lib_dir}/${name}"; return 1
        fi
    done
    if command -v lipo >/dev/null 2>&1; then
        for name in libgrpc.a libprotobuf.a libz.a; do
            if ! lipo -archs "${lib_dir}/${name}" 2>/dev/null | grep -qw arm64; then
                grpc_problem="${name} is not an arm64 archive"; return 1
            fi
        done
    fi
    if command -v otool >/dev/null 2>&1; then
        # mac_dependencies.sh pins MACOSX_DEPLOYMENT_TARGET=14.0 to match the engine.
        local minos
        minos="$(otool -l "${lib_dir}/libz.a" 2>/dev/null | awk '/minos/{print $2; exit}')"
        if [[ "${minos}" != "14.0" ]]; then
            grpc_problem="libz.a was built for macOS ${minos:-unknown}, expected 14.0 (rebuild with Resources/Build/mac_dependencies.sh)"; return 1
        fi
    fi
    return 0
}
grpc_needs_build=0
if check_grpc; then
    ok "Mac gRPC archives present in ${grpc_mac_dir} (arm64, macOS 14.0); not rebuilding."
elif [[ "${build_deps}" == 1 ]]; then
    grpc_needs_build=1
    info "${grpc_problem}"
    info "Will run Resources/Build/mac_dependencies.sh (--build-deps)."
else
    info "${grpc_problem}"
    fail "Mac gRPC archives are missing or unusable. Run Resources/Build/mac_dependencies.sh (or re-run this script with --build-deps). See Docs/macOS-Setup.md."
fi

# -------------------------------------------------------------- Plugins/Schola ----
echo "== Plugins/Schola"
plugin_path=""
link_action="none"   # none | create | replace
if [[ -n "${lyra_dir}" ]]; then
    plugin_path="${lyra_dir}/Plugins/Schola"
    if [[ ! -d "${lyra_dir}/Plugins" ]]; then
        fail "${lyra_dir}/Plugins does not exist."
    elif [[ -L "${plugin_path}" ]]; then
        link_target="$(readlink "${plugin_path}")"
        resolved="$(cd -P "${plugin_path}" 2>/dev/null && pwd -P || true)"
        if [[ "${resolved}" == "${schola_root}" ]]; then
            ok "Plugins/Schola already links to this checkout (${link_target})."
        elif [[ "${replace_link}" == 1 ]]; then
            info "Plugins/Schola currently links to ${link_target}${resolved:+ (${resolved})}; it will be replaced."
            link_action="replace"
        else
            fail "Plugins/Schola is a symlink to ${link_target}${resolved:+ (${resolved})}, not to this checkout (${schola_root}). Leaving it alone; pass --replace-schola-link to repoint the link."
        fi
    elif [[ -e "${plugin_path}" ]]; then
        resolved="$(cd -P "${plugin_path}" 2>/dev/null && pwd -P || true)"
        if [[ -n "${resolved}" && "${resolved}" == "${schola_root}" ]]; then
            ok "Plugins/Schola is this checkout."
        else
            fail "${plugin_path} already exists and is not a symlink. It will not be modified. Move or remove it yourself, then run this script again."
        fi
    else
        info "Plugins/Schola is absent; will link it to ${schola_root}."
        link_action="create"
    fi
fi

# ------------------------------------------------------------------- patch ----
echo "== CommonUser patch"
[[ -f "${patch_file}" ]] || die "Patch not found: ${patch_file}"
patch_files=()
while IFS= read -r line; do
    patch_files+=("${line}")
done < <(sed -n 's#^+++ b/##p' "${patch_file}")
[[ "${#patch_files[@]}" -gt 0 ]] || die "No files found in ${patch_file}."

lyra_git_apply() {
    # Run git apply from the Lyra root. GIT_CEILING_DIRECTORIES stops git from
    # finding a repository above the project: inside an enclosing repository
    # git apply silently skips every path that is not relative to the
    # repository root.
    (cd "${lyra_dir}" && GIT_CEILING_DIRECTORIES="$(dirname "${lyra_dir}")" \
        git apply -v -p1 --whitespace=nowarn "$@" "${patch_file}") 2>&1
}

# Sets PATCH_STATE (pristine | applied | missing | incompatible) and
# PATCH_DETAIL for one patched file. Matching is strict: no fuzz and no
# whitespace tolerance, so a modified or different Lyra version never matches.
PATCH_STATE=""; PATCH_DETAIL=""
patch_try() { # [-R]; relies on $rel
    local out rc
    out="$(lyra_git_apply --check "$@" "--include=${rel}")" && rc=0 || rc=$?
    if [[ "${rc}" == 0 && "${out}" == *"Skipped patch '${rel}'"* ]]; then
        PATCH_DETAIL="git skipped ${rel}"; return 1
    fi
    PATCH_DETAIL="$(printf '%s\n' "${out}" | grep -m 1 'patch failed' || true)"
    return "${rc}"
}
patch_state() { # rel
    rel="$1"; PATCH_DETAIL=""
    if [[ ! -f "${lyra_dir}/${rel}" ]]; then
        PATCH_STATE="missing"; return
    fi
    local fwd_detail
    if patch_try; then PATCH_STATE="pristine"; return; fi
    fwd_detail="${PATCH_DETAIL}"
    if patch_try -R; then PATCH_STATE="applied"; return; fi
    PATCH_STATE="incompatible"; PATCH_DETAIL="${fwd_detail}"
}

patch_report=()   # "state|rel"
n_pristine=0; n_applied=0; n_bad=0
if [[ -n "${lyra_dir}" ]]; then
    for rel in "${patch_files[@]}"; do
        patch_state "${rel}"
        patch_report+=("${PATCH_STATE}|${rel}")
        case "${PATCH_STATE}" in
            pristine) n_pristine=$((n_pristine + 1)); info "${rel}: not patched yet" ;;
            applied)  n_applied=$((n_applied + 1));   info "${rel}: already patched" ;;
            missing)  n_bad=$((n_bad + 1)); fail "${rel} does not exist in this Lyra project." ;;
            *)
                n_bad=$((n_bad + 1))
                fail "${rel} does not match the expected UE 5.8.4 Lyra source (modified, or another Lyra version). Not touching it."
                [[ -z "${PATCH_DETAIL}" ]] || info "git: ${PATCH_DETAIL}" >&2
                ;;
        esac
    done
    if [[ "${revert_patch}" == 1 ]]; then
        if [[ "${n_applied}" != "${#patch_files[@]}" ]]; then
            fail "--revert-patch needs every patched file to be in the already-patched state."
        fi
    elif [[ "${n_bad}" == 0 && "${n_applied}" == "${#patch_files[@]}" ]]; then
        ok "CommonUser patch is already applied."
    elif [[ "${n_bad}" == 0 ]]; then
        if [[ "${apply_patch}" == 1 ]]; then
            ok "CommonUser patch applies cleanly (${n_pristine} file(s) to patch)."
        else
            warn "CommonUser patch is not applied and --no-patch was given; the Lyra build will fail on clang until it is."
        fi
    fi
fi

# ------------------------------------------------------------------ verify ----
if [[ "${verify}" == 1 && -n "${lyra_dir}" ]]; then
    echo "== Verify"
    if [[ "${link_action}" != "none" ]]; then
        fail "Plugins/Schola is not linked to this checkout yet. Run Tools/Lyra/setup_mac.sh first."
    fi
    if [[ "${grpc_needs_build}" == 1 ]]; then
        fail "Mac gRPC archives are not built yet. Run Tools/Lyra/setup_mac.sh --build-deps first."
    fi
    if [[ "${n_pristine}" -gt 0 ]]; then
        fail "The CommonUser patch is not applied yet. Run Tools/Lyra/setup_mac.sh first."
    fi
fi

# --------------------------------------------------------- stop on problems ----
echo
if [[ "${errors}" -gt 0 ]]; then
    printf '[error] %d problem(s) found. Nothing was changed.\n' "${errors}" >&2
    exit 1
fi
if [[ "${check_only}" == 1 ]]; then
    if [[ "${verify}" == 1 ]]; then
        echo "[ok] Verified: the project is set up and ready to build."
        exit 0
    fi
    echo "[ok] Check passed. --check-only: nothing was changed."
    [[ "${link_action}" == "none" ]] || info "Would ${link_action} Plugins/Schola."
    [[ "${grpc_needs_build}" == 0 ]] || info "Would run Resources/Build/mac_dependencies.sh."
    [[ "${revert_patch}" == 1 ]] && info "Would revert the CommonUser patch." || {
        [[ "${apply_patch}" == 0 || "${n_pristine}" == 0 ]] || info "Would apply the CommonUser patch to ${n_pristine} file(s)."
    }
    exit 0
fi

# ------------------------------------------------------------------ actions ----
if [[ "${grpc_needs_build}" == 1 ]]; then
    echo "== Building macOS gRPC archives"
    "${deps_script}"
    if ! check_grpc; then
        die "gRPC archives are still unusable after the rebuild: ${grpc_problem}"
    fi
    ok "gRPC archives rebuilt."
fi

if [[ "${link_action}" == "replace" ]]; then
    # rm on a symlink path (no trailing slash) removes the link, not its target.
    rm "${plugin_path}"
fi
if [[ "${link_action}" != "none" ]]; then
    ln -s "${schola_root}" "${plugin_path}"
    ok "Linked ${plugin_path} -> ${schola_root}"
fi

backup_dir=""
keep_backup=0
cleanup() { [[ -z "${backup_dir}" || "${keep_backup}" == 1 ]] || rm -rf "${backup_dir}"; }
trap cleanup EXIT

restore_backup() {
    local rel
    for rel in "${patch_files[@]}"; do
        if [[ -f "${backup_dir}/${rel}" ]]; then
            cp -p "${backup_dir}/${rel}" "${lyra_dir}/${rel}" \
                || { keep_backup=1; printf '[error] Could not restore %s; the original is kept in %s\n' "${rel}" "${backup_dir}" >&2; }
        fi
    done
}

run_patch() { # direction: apply|revert
    local direction="$1" want_before want_after entry state rel reverse_flag=""
    if [[ "${direction}" == "revert" ]]; then
        want_before="applied"; want_after="pristine"; reverse_flag="-R"
    else
        want_before="pristine"; want_after="applied"
    fi
    backup_dir="$(mktemp -d "${TMPDIR:-/tmp}/schola-lyra-backup.XXXXXX")"
    for entry in "${patch_report[@]}"; do
        state="${entry%%|*}"; rel="${entry##*|}"
        [[ "${state}" == "${want_before}" ]] || continue
        mkdir -p "${backup_dir}/$(dirname "${rel}")"
        cp -p "${lyra_dir}/${rel}" "${backup_dir}/${rel}"
    done
    for entry in "${patch_report[@]}"; do
        state="${entry%%|*}"; rel="${entry##*|}"
        [[ "${state}" == "${want_before}" ]] || continue
        if ! out="$(lyra_git_apply ${reverse_flag} "--include=${rel}")" \
           || [[ "${out}" != *"Applied patch '${rel}'"* && "${out}" != *"Applied patch ${rel}"* ]]; then
            restore_backup
            printf '%s\n' "${out}" >&2
            if [[ "${keep_backup}" == 1 ]]; then
                die "Failed to ${direction} the patch on ${rel}, and some files could not be restored (see above)."
            fi
            die "Failed to ${direction} the patch on ${rel}. All files were restored; nothing was left half-patched."
        fi
        info "${direction} ${rel}"
    done
    for entry in "${patch_report[@]}"; do
        rel="${entry##*|}"
        patch_state "${rel}"
        if [[ "${PATCH_STATE}" != "${want_after}" ]]; then
            restore_backup
            die "Verification failed for ${rel} (state: ${PATCH_STATE}). All files were restored."
        fi
    done
}

if [[ "${revert_patch}" == 1 ]]; then
    echo "== Reverting CommonUser patch"
    run_patch revert
    ok "CommonUser patch reverted."
elif [[ "${apply_patch}" == 1 && "${n_pristine}" -gt 0 ]]; then
    echo "== Applying CommonUser patch"
    run_patch apply
    ok "CommonUser patch applied and verified."
fi

echo
if [[ "${warnings}" -gt 0 ]]; then
    printf '[ok] Setup finished with %d warning(s).\n' "${warnings}"
else
    echo "[ok] Setup finished."
fi
echo "Next: Tools/Lyra/build_mac.sh --lyra-project \"${lyra_dir}\" --ue-root \"${ue_root}\""
