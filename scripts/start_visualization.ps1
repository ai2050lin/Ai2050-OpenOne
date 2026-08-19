[CmdletBinding()]
param(
    [ValidateSet('dev', 'build', 'preview', 'lint')]
    [string]$Mode = 'dev',
    [ValidateRange(1, 65535)]
    [int]$Port = 5173,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot 'frontend'

function Get-NodeRuntimeHome {
    $candidates = [System.Collections.Generic.List[string]]::new()

    if ($env:AI2050_NODE_HOME) {
        $candidates.Add($env:AI2050_NODE_HOME)
    }

    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($nodeCommand) {
        $candidates.Add((Split-Path -Parent $nodeCommand.Source))
    }

    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npmCommand) {
        $candidates.Add((Split-Path -Parent $npmCommand.Source))
    }

    $candidates.Add((Join-Path $env:ProgramFiles 'nodejs'))
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\nodejs'))

    $workbuddyRoot = Join-Path $env:USERPROFILE '.workbuddy\binaries\node\versions'
    if (Test-Path -LiteralPath $workbuddyRoot) {
        Get-ChildItem -LiteralPath $workbuddyRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { $candidates.Add($_.FullName) }
    }

    $codexRuntimeRoot = Join-Path $env:LOCALAPPDATA 'OpenAI\Codex\runtimes\cua_node'
    if (Test-Path -LiteralPath $codexRuntimeRoot) {
        Get-ChildItem -LiteralPath $codexRuntimeRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object { $candidates.Add((Join-Path $_.FullName 'bin')) }
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate) { continue }
        $node = Join-Path $candidate 'node.exe'
        $npm = Join-Path $candidate 'npm.cmd'
        if ((Test-Path -LiteralPath $node) -and (Test-Path -LiteralPath $npm)) {
            try {
                [void](Test-SupportedNodeVersion $node)
                return $candidate
            }
            catch {
                Write-Verbose "Skipping unsupported Node.js runtime at ${candidate}: $($_.Exception.Message)"
            }
        }
    }

    return $null
}

function Test-SupportedNodeVersion([string]$NodeExecutable) {
    $versionText = (& $NodeExecutable --version).Trim().TrimStart('v')
    $version = $null
    if (-not [version]::TryParse($versionText, [ref]$version)) {
        throw "Cannot parse Node.js version: $versionText"
    }

    $supported =
        ($version.Major -gt 22) -or
        ($version.Major -eq 22 -and $version.Minor -ge 12) -or
        ($version.Major -eq 20 -and $version.Minor -ge 19)

    if (-not $supported) {
        throw "Vite 7 requires Node.js >=20.19 or >=22.12; current version is v$versionText."
    }

    return $versionText
}

if (-not (Test-Path -LiteralPath $frontendRoot)) {
    throw "Frontend directory not found: $frontendRoot"
}

$nodeHome = Get-NodeRuntimeHome
if (-not $nodeHome) {
    throw @"
No Node.js runtime containing both node.exe and npm.cmd was found.
Install Node.js >=20.19 or >=22.12, or point AI2050_NODE_HOME to a Node.js directory:
  `$env:AI2050_NODE_HOME = 'C:\path\to\nodejs'
"@
}

$nodeExecutable = Join-Path $nodeHome 'node.exe'
$npmExecutable = Join-Path $nodeHome 'npm.cmd'
$nodeVersion = Test-SupportedNodeVersion $nodeExecutable
$env:PATH = "$nodeHome;$env:PATH"

Write-Host "[AI2050] Node.js v$nodeVersion" -ForegroundColor Green
Write-Host "[AI2050] Node home: $nodeHome" -ForegroundColor DarkGray
Write-Host "[AI2050] Frontend: $frontendRoot" -ForegroundColor DarkGray

Push-Location $frontendRoot
try {
    $viteCommand = Join-Path $frontendRoot 'node_modules\.bin\vite.cmd'
    if ($Install -or -not (Test-Path -LiteralPath $viteCommand)) {
        Write-Host '[AI2050] Installing locked frontend dependencies...' -ForegroundColor Yellow
        & $npmExecutable ci
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    switch ($Mode) {
        'dev' {
            Write-Host "[AI2050] Starting visualization at http://localhost:$Port" -ForegroundColor Cyan
            & $npmExecutable run dev -- --port $Port --strictPort
        }
        'preview' {
            if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'dist\index.html'))) {
                & $npmExecutable run build
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            Write-Host "[AI2050] Previewing visualization at http://localhost:$Port" -ForegroundColor Cyan
            & $npmExecutable run preview -- --port $Port --strictPort
        }
        'build' { & $npmExecutable run build }
        'lint' { & $npmExecutable run lint }
    }

    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
