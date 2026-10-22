# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
function Resolve-UnrealEngine {
    param([string]$EnginePath)
    $engineVersion = '5.7'
    if ($EnginePath) {
        $candidates = @($EnginePath)
    } elseif ($env:UE_ENGINE_ROOT) {
        $candidates = @($env:UE_ENGINE_ROOT)
    } else {
        $candidates = @()
        foreach ($key in @("HKLM:\SOFTWARE\EpicGames\Unreal Engine\$engineVersion", "HKLM:\SOFTWARE\WOW6432Node\EpicGames\Unreal Engine\$engineVersion")) {
            $installation = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
            if ($installation.InstalledDirectory) { $candidates += $installation.InstalledDirectory }
        }
        $manifestPath = Join-Path $env:ProgramData 'Epic/UnrealEngineLauncher/LauncherInstalled.dat'
        if (Test-Path -LiteralPath $manifestPath) {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $candidates += @($manifest.InstallationList | Where-Object { $_.AppName -eq "UE_$engineVersion" } | ForEach-Object { $_.InstallLocation })
        }
        $candidates += Join-Path $env:ProgramFiles "Epic Games/UE_$engineVersion"
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath (Join-Path $candidate 'Engine/Binaries/Win64/UnrealEditor.exe'))) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Unreal Engine $engineVersion was not found. Install it with the C++ build tools, then supply -EnginePath 'C:\path\to\UE_$engineVersion' or set UE_ENGINE_ROOT."
}
