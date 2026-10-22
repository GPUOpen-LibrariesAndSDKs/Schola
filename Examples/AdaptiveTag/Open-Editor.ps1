param([string]$EnginePath)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
$taskEngine = Resolve-UnrealEngine -EnginePath $EnginePath
$taskEditor = Join-Path $taskEngine 'Engine/Binaries/Win64/UnrealEditor.exe'
$taskProject = Join-Path $PSScriptRoot 'CSC301Host.uproject'
Start-Process -FilePath $taskEditor -ArgumentList ('"' + $taskProject + '"') -WindowStyle Hidden
