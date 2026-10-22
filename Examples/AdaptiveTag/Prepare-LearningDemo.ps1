$ErrorActionPreference = 'Stop'
$revision = 'b32d9985c65a5e098a6bbf190fd994962d288b99'
$destination = Join-Path $PSScriptRoot 'ThirdParty/RLtools'
if (Test-Path (Join-Path $destination 'REVISION')) {
    if ((Get-Content (Join-Path $destination 'REVISION') -Raw).Trim() -ne $revision) { throw 'RLtools revision mismatch.' }
    if (!(Test-Path (Join-Path $destination 'include/rl_tools/operations/cpu_mux.h'))) { throw 'Pinned RLtools headers are missing. Restore ThirdParty/RLtools from Git.' }
    Write-Output "RLtools ready: $revision"
    return
}
$cache = Join-Path $PSScriptRoot 'Saved/TagLearning/Download'
New-Item -ItemType Directory -Force $cache | Out-Null
$archive = Join-Path $cache 'rl-tools.zip'
Invoke-WebRequest "https://codeload.github.com/rl-tools/rl-tools/zip/$revision" -OutFile $archive
Expand-Archive -LiteralPath $archive -DestinationPath $cache -Force
$upstream = Join-Path $cache "rl-tools-$revision"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -LiteralPath (Join-Path $upstream 'include') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $upstream 'LICENSE') -Destination $destination -Force
Set-Content -LiteralPath (Join-Path $destination 'REVISION') -Value $revision
Write-Output 'Native dependency prepared. No Python runtime or service is needed.'
