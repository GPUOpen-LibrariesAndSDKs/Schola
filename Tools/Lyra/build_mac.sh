#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build the LyraEditor target (Mac, Development) of a Lyra project that has
# been prepared with Tools/Lyra/setup_mac.sh.
#
# Before building, the script runs `setup_mac.sh --verify` so a missing
# Plugins/Schola link, an unpatched CommonUser plugin or missing gRPC archives
# fail in seconds instead of after a long compile. It does not modify anything
# except through UnrealBuildTool itself.
#
# The build runs UnrealBuildTool with:
#   -NoHotReloadFromIDE   no live-coding/hot-reload interaction with an open editor
#   -NoUBA                Unreal Build Accelerator off (as in Docs/macOS-Setup.md)
#   -MaxParallelActions=N compile actions in flight; default 2 so that
#                         memory-constrained Macs do not run out of RAM
#
# Usage:
#   Tools/Lyra/build_mac.sh --lyra-project /path/to/LyraStarterGame \
#                           --ue-root /path/to/UE_5.8 [options] [-- extra UBT args]
#
# Arguments (a flag wins over its environment variable):
#   --lyra-project PATH        Lyra project directory (or its .uproject file).
#                              Env: LYRA_PROJECT_DIR
#   --ue-root PATH             Unreal Engine root, the directory that contains
#                              Engine/Build. Env: UE_ENGINE_ROOT
#   --max-parallel-actions N   Parallel compile actions (positive integer).
#                              Env: LYRA_MAX_PARALLEL_ACTIONS. Default: 2
#
# Options:
#   --dry-run                  Run the checks and print the build command
#                              without running it.
#   --skip-check               Do not run setup_mac.sh --verify first.
#   -h, --help                 Show this help.
#
# Everything after a literal `--` is passed to UnrealBuildTool unchanged.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

lyra_arg="${LYRA_PROJECT_DIR:-}"
ue_arg="${UE_ENGINE_ROOT:-}"
max_actions="${LYRA_MAX_PARALLEL_ACTIONS:-2}"
dry_run=0
skip_check=0
extra_args=()

die() { printf '[error] %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lyra-project)  [[ $# -ge 2 ]] || die "--lyra-project needs a value."; lyra_arg="$2"; shift 2 ;;
        --lyra-project=*) lyra_arg="${1#*=}"; shift ;;
        --ue-root)       [[ $# -ge 2 ]] || die "--ue-root needs a value."; ue_arg="$2"; shift 2 ;;
        --ue-root=*)     ue_arg="${1#*=}"; shift ;;
        --max-parallel-actions) [[ $# -ge 2 ]] || die "--max-parallel-actions needs a value."; max_actions="$2"; shift 2 ;;
        --max-parallel-actions=*) max_actions="${1#*=}"; shift ;;
        --dry-run)       dry_run=1; shift ;;
        --skip-check)    skip_check=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; extra_args=("$@"); break ;;
        *)               die "Unknown argument: $1 (see --help)." ;;
    esac
done

[[ -n "${lyra_arg}" ]] || die "Lyra project path is required: pass --lyra-project or set LYRA_PROJECT_DIR."
[[ -n "${ue_arg}" ]]   || die "Unreal Engine root is required: pass --ue-root or set UE_ENGINE_ROOT."
if [[ ! "${max_actions}" =~ ^[1-9][0-9]*$ ]]; then
    die "--max-parallel-actions must be a positive integer (got '${max_actions}')."
fi
[[ "$(uname)" == "Darwin" ]] || die "This script is for macOS."

# Resolve the project directory and engine root the same way setup_mac.sh does.
lyra_candidate="${lyra_arg}"
if [[ -f "${lyra_candidate}" && "${lyra_candidate}" == *.uproject ]]; then
    lyra_candidate="$(dirname "${lyra_candidate}")"
fi
[[ -d "${lyra_candidate}" ]] || die "Lyra project directory does not exist: ${lyra_arg}"
lyra_dir="$(cd -P "${lyra_candidate}" && pwd -P)"
uproject="${lyra_dir}/LyraStarterGame.uproject"
[[ -f "${uproject}" ]] || die "No LyraStarterGame.uproject in ${lyra_dir}."

if [[ -d "${ue_arg}/Engine/Build" ]]; then
    ue_root="$(cd -P "${ue_arg}" && pwd -P)"
elif [[ "$(basename "${ue_arg}")" == "Engine" && -d "${ue_arg}/Build" ]]; then
    ue_root="$(cd -P "${ue_arg}/.." && pwd -P)"
else
    die "Not an Unreal Engine root (no Engine/Build): ${ue_arg}"
fi
build_sh="${ue_root}/Engine/Build/BatchFiles/Mac/Build.sh"
[[ -x "${build_sh}" ]] || die "Missing ${build_sh}. Is this a Mac engine build?"

if [[ "${skip_check}" == 0 ]]; then
    echo "== Checking setup (setup_mac.sh --verify)"
    if ! "${script_dir}/setup_mac.sh" --verify \
            --lyra-project "${lyra_dir}" --ue-root "${ue_root}"; then
        die "Setup check failed. Fix the problems above (see Docs/Lyra-Integration.md) or pass --skip-check."
    fi
    echo
fi

build_cmd=("${build_sh}" LyraEditor Mac Development
    "-Project=${uproject}"
    -NoHotReloadFromIDE
    -NoUBA
    "-MaxParallelActions=${max_actions}")
if [[ "${#extra_args[@]}" -gt 0 ]]; then
    build_cmd+=("${extra_args[@]}")
fi

echo "== Build command (max parallel actions: ${max_actions})"
printf ' '
printf ' %q' "${build_cmd[@]}"
printf '\n'

if [[ "${dry_run}" == 1 ]]; then
    echo "[ok] --dry-run: not building."
    exit 0
fi

echo
echo "== Building LyraEditor Mac Development"
exec "${build_cmd[@]}"
