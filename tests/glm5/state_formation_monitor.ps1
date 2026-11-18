$ErrorActionPreference='Stop'
$sfMonitorRoot='C:\Users\Admin\.codex\visualizations\2026\11\08\01a1bbc6-0c76-7430-aec3-194dcd9f7933\research-storage\state_formation_v1'
$sfMonitorDeadline=(Get-Date).AddMinutes(90)
while ((Get-Date) -lt $sfMonitorDeadline -and -not (Test-Path -LiteralPath (Join-Path $sfMonitorRoot 'monitor-stop.json'))) {
    $sfProcessIds=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*state_formation_runner.py*' -or $_.CommandLine -like '*state_formation_diagnostic.py*' } | Select-Object -ExpandProperty ProcessId)
    $sfRows=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {
        $sfMatch=[regex]::Match($_.Name,'^pid_(\d+)_')
        $sfMatch.Success -and $sfProcessIds.Contains([uint32]$sfMatch.Groups[1].Value)
    } | Select-Object Name,SharedUsage,DedicatedUsage)
    $sfLine=@{time=(Get-Date -Format o);source='Windows WDDM process counters';process_ids=$sfProcessIds;gpu_process_memory=$sfRows;counter_readable=$true} | ConvertTo-Json -Compress -Depth 4
    Add-Content -LiteralPath (Join-Path $sfMonitorRoot 'gpu-monitor.jsonl') -Value $sfLine -Encoding UTF8
    Write-Output $sfLine
    Start-Sleep -Seconds 30
}
