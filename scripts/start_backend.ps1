[CmdletBinding()]
param(
    # Detached/idempotent launcher for the AI2050 client backend (server/server.py :5001).
    # Designed for the VSCode folderOpen auto task: survives terminal close, no window,
    # reuses a healthy instance if one is already listening, logs to .run/backend/.
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot '.run\backend'
$pidFile = Join-Path $runDir 'backend.pid'
$logFile = Join-Path $runDir 'backend.log'
$errFile = Join-Path $runDir 'backend.err.log'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$server = Join-Path $repoRoot 'server\server.py'
$port = 5001

function Test-PortOpen([int]$p) {
    $c = New-Object Net.Sockets.TcpClient
    try {
        $t = $c.BeginConnect('127.0.0.1', $p, $null, $null)
        if (-not $t.AsyncWaitHandle.WaitOne(800)) { return $false }
        $c.EndConnect($t); return $true
    } catch { return $false }
    finally { $c.Close() }
}

if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $server)) { throw "Server entry not found: $server" }
if (-not (Test-Path -LiteralPath $runDir)) { New-Item -ItemType Directory -Path $runDir -Force | Out-Null }

if ($Status) {
    if (Test-PortOpen $port) {
        $old = if (Test-Path -LiteralPath $pidFile) { Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue } else { $null }
        Write-Host "[AI2050] Backend :$port is RUNNING (pid file: $old)"
    } else {
        Write-Host "[AI2050] Backend :$port is DOWN"
    }
    exit 0
}

if ($Stop) {
    if (Test-Path -LiteralPath $pidFile) {
        $oldPid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue
        if ($oldPid) {
            try { Stop-Process -Id ([int]$oldPid) -Force -ErrorAction SilentlyContinue } catch { }
        }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[AI2050] Backend :$port stopped (best effort)"
    exit 0
}

# Idempotent: reuse a healthy listener.
if (Test-PortOpen $port) {
    Write-Host "[AI2050] Backend :$port already running - reuse"
    exit 0
}

# Clean up a stale pid file (dead owner) so the pid file stays truthful.
if (Test-Path -LiteralPath $pidFile) {
    $oldPid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue
    if ($oldPid) {
        $alive = $false
        try { $alive = $null -ne (Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue) } catch { }
        if (-not $alive) { Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue }
    }
}

# Start detached: hidden window, output to log files, independent of this terminal.
$env:PYTHONIOENCODING = 'utf-8'
$env:AI2050_SKIP_MODEL_LOAD = '1'

# Some environments carry both HTTPS_PROXY and https_proxy (case-only duplicates),
# which makes Start-Process throw "An item with the same key has already been
# added". Keep the uppercase variant, drop lowercase duplicates before spawning.
$names = @([System.Environment]::GetEnvironmentVariables('Process').Keys) | ForEach-Object { $_ }
$seen = @{}
foreach ($name in $names) {
    $key = $name.ToUpperInvariant()
    if ($seen.ContainsKey($key)) {
        if ($name -ceq $key) {
            # current is ALL-UPPERCASE: drop the previously kept lowercase twin
            [void][System.Environment]::SetEnvironmentVariable($seen[$key], $null, 'Process')
            $seen[$key] = $name
        } else {
            # current is lowercase and an uppercase twin is already kept
            [void][System.Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    } else {
        $seen[$key] = $name
    }
}

$proc = Start-Process -FilePath $python `
    -ArgumentList @("`"$server`"") `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru
$proc.Id | Set-Content -LiteralPath $pidFile -Encoding ascii

# Readiness gate: wait up to 20s for :5001 to accept connections.
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-PortOpen $port) { $ready = $true; break }
    if ($proc.HasExited) { break }
}

if ($ready) {
    Write-Host "[AI2050] Backend :$port started (pid $($proc.Id), log $logFile)"
    exit 0
}

Write-Host "[AI2050] Backend :$port FAILED to start (exit $($proc.ExitCode)). See $errFile" -ForegroundColor Red
exit 1
