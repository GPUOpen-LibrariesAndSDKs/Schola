param([ValidateSet('Native','Arena','Packaged')][string]$Mode='Native', [string]$Executable, [string]$EnginePath)
$ErrorActionPreference='Stop'
$reports = Join-Path $PSScriptRoot 'Saved/TagLearning'
New-Item -ItemType Directory -Force $reports | Out-Null
if ($Mode -eq 'Native') {
    cmake -S (Join-Path $PSScriptRoot 'Tools/LearningTests') -B (Join-Path $reports 'NativeBuild') -G 'Visual Studio 17 2022' -A x64
    if ($LASTEXITCODE -ne 0) { throw 'Native configuration failed.' }
    cmake --build (Join-Path $reports 'NativeBuild') --config Release
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed.' }
    & (Join-Path $reports 'NativeBuild/Release/tag_learning_tests.exe') (Join-Path $reports 'learning-curve.csv')
    if ($LASTEXITCODE -ne 0) { throw 'Native adaptation/lifetime tests failed.' }
    return
}
if ($Mode -eq 'Arena') {
    . (Join-Path $PSScriptRoot 'Tools/Unreal.ps1')
    $taskEngine = Resolve-UnrealEngine -EnginePath $EnginePath
    $Executable=Join-Path $taskEngine 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
    $arguments=@((Join-Path $PSScriptRoot 'CSC301Host.uproject'),'/Game/TagArena/Maps/TagArena','-game')
} else {
    if (!(Test-Path -LiteralPath $Executable)) { throw 'Supply -Executable pointing to the packaged game executable.' }
    $arguments=@()
}
$reportPath=Join-Path $reports 'ArenaLearningTest.json'
$arguments+=@('-TagLearningTest',"-TagLearningReport=$reportPath",'-NullRHI','-nosound','-unattended','-nop4','-nosplash',"-abslog=$reports/ArenaLearningTest.log")
$startedAt=[DateTime]::UtcNow
$process=Start-Process -FilePath $Executable -ArgumentList (($arguments | ForEach-Object { '"'+$_+'"' }) -join ' ') -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Arena learning test failed ($($process.ExitCode)). See Saved/TagLearning/ArenaLearningTest.log" }
if (!(Test-Path -LiteralPath $reportPath) -or (Get-Item -LiteralPath $reportPath).LastWriteTimeUtc -lt $startedAt) { throw 'The current test run did not produce a fresh report.' }
$report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if (!$report.passed) { throw 'Arena learning assertions failed.' }
$report | ConvertTo-Json -Compress
