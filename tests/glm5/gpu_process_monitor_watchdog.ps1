param(
    [ValidateRange(5, 600)]
    [int]$CheckIntervalSeconds = 30
)

$ErrorActionPreference = "Continue"
$monitorScript = Join-Path $PSScriptRoot "gpu_process_monitor.ps1"
$monitorPattern = [regex]::Escape($monitorScript)
$watchdogLog = Join-Path $PSScriptRoot "result\gpu_process_monitor\watchdog_log.txt"
$logDirectory = Split-Path -Parent $watchdogLog
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Write-WatchdogLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') $Message"
    $line | Out-File -LiteralPath $watchdogLog -Append -Encoding utf8
}

function Get-MonitorProcesses {
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -eq "powershell.exe" -and
                $_.CommandLine -match $monitorPattern
            }
    )
}

Write-WatchdogLog "watchdog started; pid=$PID"

while ($true) {
    try {
        $monitors = @(Get-MonitorProcesses)
        if ($monitors.Count -eq 0) {
            $arguments = @(
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-WindowStyle", "Hidden",
                "-File", ('"' + $monitorScript + '"'),
                "-IntervalMinutes", "5",
                "-MinUsagePercent", "10",
                "-SampleSeconds", "3"
            )
            $process = Start-Process `
                -FilePath "powershell.exe" `
                -ArgumentList $arguments `
                -WindowStyle Hidden `
                -PassThru
            Write-WatchdogLog "monitor restarted; pid=$($process.Id)"
        }
        elseif ($monitors.Count -gt 1) {
            $keepers = @($monitors | Sort-Object CreationDate)
            foreach ($duplicate in ($keepers | Select-Object -Skip 1)) {
                Stop-Process -Id $duplicate.ProcessId -Force -ErrorAction SilentlyContinue
                Write-WatchdogLog "duplicate monitor stopped; pid=$($duplicate.ProcessId)"
            }
        }
    }
    catch {
        Write-WatchdogLog "watchdog error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
