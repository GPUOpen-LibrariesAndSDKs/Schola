param([switch]$Packaged, [switch]$Paused, [switch]$Build, [string]$EnginePath)
$ErrorActionPreference = 'Stop'
$pauseArg = if ($Paused) { ' -TagStartPaused' } else { '' }
if ($Packaged) {
    $game = Join-Path $PSScriptRoot 'Saved/TagLearning/Package/Windows/CSC301Host.exe'
    if (!(Test-Path -LiteralPath $game)) { throw 'Run Package-LearningDemo.ps1 first.' }
    Start-Process -FilePath $game -ArgumentList ('-windowed -ResX=1600 -ResY=1000 -nosplash -NoVSync' + $pauseArg) -WindowStyle Normal
    return
}
. (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
$taskEngine = Resolve-UnrealEngine -EnginePath $EnginePath
if ($Build -or !(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'Binaries/Win64/UnrealEditor-CSC301Host.dll'))) {
    & (Join-Path $PSScriptRoot 'Build.ps1') -EnginePath $taskEngine
}
$taskEditor = Join-Path $taskEngine 'Engine/Binaries/Win64/UnrealEditor.exe'
$taskProject = Join-Path $PSScriptRoot 'CSC301Host.uproject'
$taskArgs = '"' + $taskProject + '" /Game/TagArena/Maps/TagArena -game -windowed -ResX=1600 -ResY=1000 -nosplash -NoVSync' + $pauseArg
Start-Process -FilePath $taskEditor -ArgumentList $taskArgs -WindowStyle Normal
