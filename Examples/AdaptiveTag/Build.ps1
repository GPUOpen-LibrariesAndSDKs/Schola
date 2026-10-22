param([switch]$GenerateProjectFiles, [string]$EnginePath)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
$taskEngine = Resolve-UnrealEngine -EnginePath $EnginePath
$taskProject = Join-Path $PSScriptRoot 'CSC301Host.uproject'
& (Join-Path $PSScriptRoot 'Prepare-LearningDemo.ps1')
if ($GenerateProjectFiles) {
    & "$taskEngine\Engine\Build\BatchFiles\Build.bat" -projectfiles "-project=$taskProject" -game -rocket -progress
} else {
    & "$taskEngine\Engine\Build\BatchFiles\Build.bat" CSC301HostEditor Win64 Development $taskProject -WaitMutex -NoHotReloadFromIDE
}
if ($LASTEXITCODE -ne 0) { throw 'Adaptive Tag build failed. Check the Unreal build output above.' }
