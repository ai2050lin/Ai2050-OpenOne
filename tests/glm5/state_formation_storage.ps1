param([switch]$FinalizeVerifiedCopy)
$ErrorActionPreference = 'Stop'
$sfWorkspace = [IO.Path]::GetFullPath('D:\AI2050\Ai2050-OpenOne')
$sfAllowed = [IO.Path]::GetFullPath('C:\Users\Admin\.codex\visualizations\2026\11\08\01a1bbc6-0c76-7430-aec3-194dcd9f7933')
$sfStore = Join-Path $sfAllowed 'research-storage'
$sfResults = Join-Path $sfWorkspace 'tests\glm5\result'
$sfSource = Join-Path $sfResults 'mlp_puzzle_v1'
$sfDest = Join-Path $sfStore 'mlp_puzzle_v1'
$sfBackup = Join-Path $sfResults 'mlp_puzzle_v1_verified_migration_backup'
$sfNew = Join-Path $sfStore 'state_formation_v1'
$sfLogical = Join-Path $sfResults 'state_formation_v1'
$sfAudit = Join-Path $sfWorkspace 'tests\glm5_temp\state-formation-storage.json'
foreach ($sfPath in @($sfSource,$sfBackup,$sfLogical)) {
    if (-not [IO.Path]::GetFullPath($sfPath).StartsWith($sfResults+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Workspace boundary violation' }
}
foreach ($sfPath in @($sfDest,$sfNew)) {
    if (-not [IO.Path]::GetFullPath($sfPath).StartsWith($sfAllowed+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Authorized storage boundary violation' }
}
if ((Get-PSDrive C).Free -lt 25GB) { throw 'Need 25 GiB reserve before migration and new collection' }
if ((Get-Item -LiteralPath $sfSource).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source already linked; inspect existing migration record before rerunning' }
if (((Test-Path -LiteralPath $sfDest) -and -not $FinalizeVerifiedCopy) -or (Test-Path -LiteralPath $sfBackup) -or (Test-Path -LiteralPath $sfLogical)) { throw 'A destination exists; do not overwrite or delete it' }
if ($FinalizeVerifiedCopy -and (-not (Test-Path -LiteralPath $sfAudit))) { throw 'No verified copy record to resume' }
New-Item -ItemType Directory -Path $sfStore -Force | Out-Null
$sfFiles = @(Get-ChildItem -LiteralPath $sfSource -File -Recurse)
$sfDirs = @(Get-ChildItem -LiteralPath $sfSource -Directory -Recurse)
if (@($sfDirs | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Nested reparse point not allowed' }
$sfBefore = @{}; foreach ($sfFile in $sfFiles) { $sfBefore[$sfFile.FullName] = "$($sfFile.Length):$($sfFile.LastWriteTimeUtc.Ticks)" }
$sfBytes = ($sfFiles | Measure-Object Length -Sum).Sum
$sfTimer = [Diagnostics.Stopwatch]::StartNew()
Write-Output (@{stage='copy';files=$sfFiles.Count;bytes=$sfBytes;source=$sfSource;destination=$sfDest} | ConvertTo-Json -Compress)
if (-not $FinalizeVerifiedCopy) { Copy-Item -LiteralPath $sfSource -Destination $sfDest -Recurse }
$sfManifest = [Collections.Generic.List[object]]::new(); $sfCount=0
foreach ($sfFile in $sfFiles) {
    $sfRelative = $sfFile.FullName.Substring($sfSource.Length+1)
    $sfCopied = Join-Path $sfDest $sfRelative
    $sfOriginalHash = (Get-FileHash -LiteralPath $sfFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $sfCopiedHash = (Get-FileHash -LiteralPath $sfCopied -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sfOriginalHash -ne $sfCopiedHash) { throw "Hash mismatch: $sfRelative; original retained" }
    $sfManifest.Add(@{path=$sfRelative;bytes=$sfFile.Length;sha256=$sfOriginalHash})
    $sfCount++
    if (($sfCount % 100) -eq 0) { Write-Output (@{stage='verify';done=$sfCount;total=$sfFiles.Count;seconds=[math]::Round($sfTimer.Elapsed.TotalSeconds,1)} | ConvertTo-Json -Compress) }
}
$sfAfter = @(Get-ChildItem -LiteralPath $sfSource -File -Recurse)
if ($sfAfter.Count -ne $sfFiles.Count) { throw 'Source changed during copy; original retained' }
foreach ($sfFile in $sfAfter) { if ($sfBefore[$sfFile.FullName] -ne "$($sfFile.Length):$($sfFile.LastWriteTimeUtc.Ticks)") { throw 'Source changed during migration; original retained' } }
if (@(Get-ChildItem -LiteralPath $sfDest -File -Recurse).Count -ne $sfFiles.Count) { throw 'Copied file count differs' }
$sfRecord = [ordered]@{stage='verified_copy';created_at=(Get-Date -Format o);source=$sfSource;destination=$sfDest;files=$sfFiles.Count;bytes=$sfBytes;manifest=$sfManifest;original_retained=$true}
$sfRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $sfAudit -Encoding UTF8
# All resolved targets above are exact task paths. Rename only after the copied
# bytes and unchanged source identity are verified; roll back if link creation fails.
Move-Item -LiteralPath $sfSource -Destination $sfBackup
try { New-Item -ItemType Junction -Path $sfSource -Target $sfDest | Out-Null }
catch { Move-Item -LiteralPath $sfBackup -Destination $sfSource; throw }
if ((Get-Item -LiteralPath $sfSource).LinkType -ne 'Junction') { throw 'Expected a junction; verified backup retained' }
foreach ($sfItem in $sfManifest) {
    if ((Get-Item -LiteralPath (Join-Path $sfSource $sfItem.path)).Length -ne $sfItem.bytes) { throw 'Linked readback failed; verified backup retained' }
}
if ((Get-Item -LiteralPath $sfBackup).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected linked backup; do not remove' }
Remove-Item -LiteralPath $sfBackup -Recurse -Force
New-Item -ItemType Directory -Path $sfNew | Out-Null
New-Item -ItemType Junction -Path $sfLogical -Target $sfNew | Out-Null
$sfRecord.stage='complete';$sfRecord.original_retained=$false
$sfRecord['evidence_preserved_at']=$sfDest;$sfRecord['original_path_preserved_by_junction']=$true
$sfRecord['new_results_logical']=$sfLogical;$sfRecord['new_results_physical']=$sfNew
$sfRecord['D_free_bytes']=(Get-PSDrive D).Free;$sfRecord['C_free_bytes']=(Get-PSDrive C).Free
$sfRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $sfAudit -Encoding UTF8
Write-Output (@{stage='complete';files=$sfFiles.Count;preserved_bytes=$sfBytes;D_free_GiB=[math]::Round((Get-PSDrive D).Free/1GB,2);C_free_GiB=[math]::Round((Get-PSDrive C).Free/1GB,2)} | ConvertTo-Json -Compress)
