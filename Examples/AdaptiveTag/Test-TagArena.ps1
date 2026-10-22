param([string]$EnginePath)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
$taskEngine = Resolve-UnrealEngine -EnginePath $EnginePath
$taskEditor = Join-Path $taskEngine 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
$taskProject = Join-Path $PSScriptRoot 'CSC301Host.uproject'
$savedPath = Join-Path $PSScriptRoot 'Saved/Automation'
New-Item -ItemType Directory -Force -Path $savedPath | Out-Null
$reportPath = Join-Path $savedPath 'TagArenaSmoke.json'
$taskLog = Join-Path $savedPath 'TagArenaSmoke.log'
$arguments = @($taskProject, '/Game/TagArena/Maps/TagArena', '-game', '-TagArenaSmokeTest', '-NullRHI', '-nosound', '-unattended', '-nop4', '-nosplash', "-abslog=$taskLog")
$startedAt = [DateTime]::UtcNow
$process = Start-Process -FilePath $taskEditor -ArgumentList (($arguments | ForEach-Object { '"'+$_+'"' }) -join ' ') -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Arena smoke test failed ($($process.ExitCode)). See $taskLog" }
if (!(Test-Path -LiteralPath $reportPath) -or (Get-Item -LiteralPath $reportPath).LastWriteTimeUtc -lt $startedAt) { throw 'No fresh arena smoke report.' }
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if (!$report.passed) { throw "Arena smoke assertions failed. See $taskLog" }
$report | ConvertTo-Json -Compress
