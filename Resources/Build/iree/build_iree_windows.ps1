# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Populates Source/ThirdParty/IREE/Win64 from a local IREE runtime build.
#
# Usage examples:
#   .\build_iree_windows.ps1 -IreeInstallRoot C:\iree-install -IreeSourceRoot C:\iree-src -IreeBuildRoot C:\iree-build
#   .\build_iree_windows.ps1 -IreeInstallRoot C:\iree-install   # if the install already has everything
#
# Does not download anything. See README.md for how to build the IREE runtime.
#
# IREE's install step leaves out two things Schola needs:
#   - C++ headers such as iree/vm/ref_cc.h, which iree/vm/ref.h includes from C++. Copied from
#     <IreeSourceRoot>/runtime/src without overwriting installed (or generated) headers.
#   - flatcc_runtime.lib / flatcc_parsing.lib, which the bytecode module verifier links against.
#     Copied from <IreeBuildRoot>/build_tools/third_party/flatcc/<Config>.

param(
    [Parameter(Mandatory = $true)]
    [string]$IreeInstallRoot,

    [string]$IreeSourceRoot,

    [string]$IreeBuildRoot,

    [string]$Config = "Release",

    [string]$Platform = "Win64"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$IreeThirdPartyDir = Join-Path $PluginRoot "Source\ThirdParty\IREE"
$DestRoot = Join-Path $IreeThirdPartyDir $Platform
$DestInclude = Join-Path $DestRoot "include"
$DestLib = Join-Path $DestRoot "lib"
$FlatccLibs = @("flatcc_runtime.lib", "flatcc_parsing.lib")
# The unified library is a single archive of the whole runtime (see iree_cc_unified_library in
# runtime/src/iree/runtime/CMakeLists.txt); the ~230 per-component libs duplicate it, so only it is copied.
$UnifiedLib = "iree_runtime_unified.lib"
# The unified archive omits the MASM object (elf/arch/x86_64_msvc.obj) that defines iree_elf_call_* and
# iree_elf_thunk_*, which the embedded-ELF loader needs. It only exists in this per-component lib.
$ExtraLibs = @("iree_hal_local_elf_arch.lib")
$PruneScript = Join-Path $ScriptDir "prune_iree.py"
$ScholaIreeSource = Join-Path $PluginRoot "Source\ScholaIREE"

function Find-IreeIncludeRoot {
    param([string]$Root)
    $candidates = @(
        (Join-Path $Root "include"),
        (Join-Path $Root "iree-install\include"),
        $Root
    )
    foreach ($candidate in $candidates) {
        $api = Join-Path $candidate "iree\runtime\api.h"
        if (Test-Path $api) { return $candidate }
    }
    throw "Could not find iree/runtime/api.h under $Root"
}

function Find-IreeLibRoot {
    param([string]$Root)
    $candidates = @(
        (Join-Path $Root "lib"),
        (Join-Path $Root "iree-install\lib"),
        $Root
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $libs = Get-ChildItem -Path $candidate -Filter "*.lib" -ErrorAction SilentlyContinue
            if ($libs.Count -gt 0) { return $candidate }
        }
    }
    throw "Could not find *.lib under $Root"
}

function Invoke-Robocopy {
    param([string[]]$Arguments)
    robocopy @Arguments /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
}

$includeRoot = Find-IreeIncludeRoot -Root $IreeInstallRoot
$libRoot = Find-IreeLibRoot -Root $IreeInstallRoot

New-Item -ItemType Directory -Force -Path $DestInclude, $DestLib | Out-Null

# Start clean so stale headers/libs from earlier fetches do not accumulate (or get linked twice).
Get-ChildItem $DestInclude -Force | Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Recurse -Force
Get-ChildItem $DestLib -Force | Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Recurse -Force

Write-Host "Copying headers from $includeRoot -> $DestInclude"
Invoke-Robocopy @($includeRoot, $DestInclude, "/E")

if ($IreeSourceRoot) {
    $runtimeSrc = Join-Path $IreeSourceRoot "runtime\src"
    if (-not (Test-Path $runtimeSrc)) { throw "Could not find runtime\src under $IreeSourceRoot" }
    Write-Host "Adding headers missing from the install from $runtimeSrc"
    Invoke-Robocopy @($runtimeSrc, $DestInclude, "*.h", "/S", "/XC", "/XN", "/XO")
}

Write-Host "Copying $UnifiedLib from $libRoot -> $DestLib"
$unifiedPath = Join-Path $libRoot $UnifiedLib
if (-not (Test-Path $unifiedPath)) { throw "Could not find $unifiedPath (install the IREEDevLibraries-Runtime component)" }
Copy-Item $unifiedPath $DestLib -Force
foreach ($lib in $ExtraLibs) {
    $path = Join-Path $libRoot $lib
    if (-not (Test-Path $path)) { throw "Could not find $path" }
    Copy-Item $path $DestLib -Force
}

if ($IreeBuildRoot) {
    $flatccDir = Join-Path $IreeBuildRoot "build_tools\third_party\flatcc\$Config"
    foreach ($lib in $FlatccLibs) {
        $path = Join-Path $flatccDir $lib
        if (-not (Test-Path $path)) { throw "Could not find $path (build IREE with -DCMAKE_BUILD_TYPE/--config $Config first)" }
        Copy-Item $path $DestLib -Force
    }
}

Write-Host "Pruning headers to those used by ScholaIREE"
python3 $PruneScript $DestInclude $ScholaIreeSource
if ($LASTEXITCODE -ne 0) { throw "prune_iree.py failed with exit code $LASTEXITCODE" }

$missing = @()
if (-not (Test-Path (Join-Path $DestLib $UnifiedLib))) { $missing += $UnifiedLib }
if (-not (Test-Path (Join-Path $DestInclude "iree\runtime\api.h"))) { $missing += "iree/runtime/api.h" }
if (-not (Test-Path (Join-Path $DestInclude "iree\vm\ref_cc.h"))) { $missing += "iree/vm/ref_cc.h (pass -IreeSourceRoot)" }
foreach ($lib in $FlatccLibs) {
    if (-not (Test-Path (Join-Path $DestLib $lib))) { $missing += "$lib (pass -IreeBuildRoot)" }
}
if ($missing.Count -gt 0) {
    throw "IREE layout at $DestRoot is incomplete. Missing: $($missing -join ', ')"
}

$buildCs = Join-Path $IreeThirdPartyDir "IREE.Build.cs"
(Get-Item $buildCs).LastWriteTime = Get-Date
Write-Host "IREE $Platform third-party layout ready at $DestRoot"
