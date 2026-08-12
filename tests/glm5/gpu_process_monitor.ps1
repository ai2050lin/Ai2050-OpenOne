param(
    [ValidateRange(1, 1440)]
    [int]$IntervalMinutes = 5,

    [ValidateRange(1, 30)]
    [int]$SampleSeconds = 3,

    [ValidateRange(0, 100)]
    [double]$MinUsagePercent = 10,

    [string]$LogPath = (
        Join-Path $PSScriptRoot "result\gpu_process_monitor\gpu_over_10_percent_log.txt"
    ),

    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Get-NvidiaSmiPath {
    $command = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $defaultPath = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path -LiteralPath $defaultPath) {
        return $defaultPath
    }

    throw "nvidia-smi.exe was not found. Install an NVIDIA driver or add it to PATH."
}

function Get-ProcessDescription {
    param([int]$ProcessId)

    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" `
        -ErrorAction SilentlyContinue
    if (-not $cim) {
        return [pscustomobject]@{
            Name        = "<exited>"
            Path        = ""
            CommandLine = ""
        }
    }

    return [pscustomobject]@{
        Name        = [string]$cim.Name
        Path        = [string]$cim.ExecutablePath
        CommandLine = [string]$cim.CommandLine
    }
}

function Get-GpuEngineUsage {
    param([int]$Seconds)

    try {
        $counterSamples = (
            Get-Counter "\GPU Engine(*)\Utilization Percentage" `
                -SampleInterval 1 -MaxSamples $Seconds
        ).CounterSamples
    }
    catch {
        return @()
    }

    $parsed = foreach ($sample in $counterSamples) {
        if (
            $sample.InstanceName -match
            "^pid_(\d+)_.*_engtype_(.+)$"
        ) {
            [pscustomobject]@{
                ProcessId = [int]$Matches[1]
                Engine    = [string]$Matches[2]
                Usage     = [double]$sample.CookedValue
            }
        }
    }

    $usage = foreach ($group in ($parsed | Group-Object ProcessId, Engine)) {
        $first = $group.Group[0]
        [pscustomobject]@{
            ProcessId = $first.ProcessId
            Engine    = $first.Engine
            Average   = [math]::Round(
                ($group.Group.Usage | Measure-Object -Average).Average, 2
            )
            Peak      = [math]::Round(
                ($group.Group.Usage | Measure-Object -Maximum).Maximum, 2
            )
        }
    }

    return @($usage)
}

function Write-GpuSnapshot {
    param(
        [string]$NvidiaSmi,
        [string]$Destination,
        [int]$Seconds
    )

    $timestamp = Get-Date
    $engineUsage = @(Get-GpuEngineUsage -Seconds $Seconds)

    $allIds = @($engineUsage.ProcessId) |
        Where-Object { $_ -gt 0 } |
        Sort-Object -Unique

    $output = [System.Collections.Generic.List[string]]::new()
    $output.Add("Time: $($timestamp.ToString('yyyy-MM-dd HH:mm:ss zzz'))")

    $qualifying = @()
    foreach ($processId in $allIds) {
        $processUsage = @(
            $engineUsage | Where-Object ProcessId -eq $processId |
                Sort-Object Peak -Descending
        )
        $totalAverage = [math]::Round(
            ($processUsage.Average | Measure-Object -Sum).Sum, 2
        )
        $totalPeak = [math]::Round(
            ($processUsage.Peak | Measure-Object -Sum).Sum, 2
        )
        if (
            $totalAverage -gt $MinUsagePercent -or
            $totalPeak -gt $MinUsagePercent
        ) {
            $qualifying += [pscustomobject]@{
                ProcessId = $processId
                Usage     = $processUsage
                Average   = $totalAverage
                Peak      = $totalPeak
            }
        }
    }

    foreach ($entry in $qualifying) {
            $description = Get-ProcessDescription -ProcessId $entry.ProcessId
            $output.Add(
                "PID=$($entry.ProcessId) Name=$($description.Name) " +
                "GPU_Avg=$($entry.Average)% GPU_Peak=$($entry.Peak)%"
            )
            $output.Add("Path: $($description.Path)")
            $output.Add("Command: $($description.CommandLine)")
            foreach ($item in $entry.Usage) {
                if ($item.Average -gt 0 -or $item.Peak -gt 0) {
                $output.Add(
                    "Engine=$($item.Engine) " +
                    "Avg=$($item.Average)% Peak=$($item.Peak)%"
                )
            }
        }
    }
    $output.Add("")
    [System.IO.File]::AppendAllLines(
        $Destination,
        $output,
        [System.Text.UTF8Encoding]::new($false)
    )
}

$resolvedLogPath = [System.IO.Path]::GetFullPath($LogPath)
$logDirectory = Split-Path -Parent $resolvedLogPath
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$nvidiaSmi = Get-NvidiaSmiPath

do {
    try {
        Write-GpuSnapshot `
            -NvidiaSmi $nvidiaSmi `
            -Destination $resolvedLogPath `
            -Seconds $SampleSeconds
        Write-Host "GPU snapshot appended: $resolvedLogPath"
    }
    catch {
        $errorLines = @(
            ("=" * 100),
            "Time: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))",
            "Monitor error: $($_.Exception.Message)",
            ""
        )
        [System.IO.File]::AppendAllLines(
            $resolvedLogPath,
            $errorLines,
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Warning $_.Exception.Message
    }

    if (-not $Once) {
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
} while (-not $Once)
