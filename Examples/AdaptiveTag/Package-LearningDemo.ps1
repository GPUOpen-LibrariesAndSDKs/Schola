param([string]$EnginePath)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
$engine = Resolve-UnrealEngine -EnginePath $EnginePath
& (Join-Path $PSScriptRoot 'Build.ps1') -EnginePath $engine
$project = Join-Path $PSScriptRoot 'CSC301Host.uproject'
$archive = Join-Path $PSScriptRoot 'Saved/TagLearning/Package'
& (Join-Path $engine 'Engine/Build/BatchFiles/RunUAT.bat') BuildCookRun "-project=$project" -nop4 -unattended -utf8output -platform=Win64 -clientconfig=Development -build -cook -stage -pak -archive -prereqs "-archivedirectory=$archive" -map=/Game/TagArena/Maps/TagArena -skipbuildeditor -nocompileeditor
if ($LASTEXITCODE -ne 0) { throw 'Native tag demo packaging failed.' }
Write-Output "Packaged game: $archive/Windows/CSC301Host.exe"
