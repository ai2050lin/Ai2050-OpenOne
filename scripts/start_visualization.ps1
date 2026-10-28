[CmdletBinding()]
param(
    # client：启动 vite dev 后打开 Electron 客户端（加载 dev 服务器，热更新），客户端窗口关闭即结束
    [ValidateSet('dev', 'build', 'preview', 'lint', 'client')]
    [string]$Mode = 'dev',
    [ValidateRange(1, 65535)]
    [int]$Port = 5173,
    [switch]$Install,
    # 守护方式（默认：VSCode 终端里自动脱管，普通终端里前台守护）
    #   -Detach     强制脱管：守护进程独立于本终端，VSCode reload / 关终端都不影响服务
    #   -Foreground 前台守护：Ctrl+C 停止，便于现场观察
    [switch]$Detach,
    [switch]$Foreground,
    # 停止 / 查看状态
    [switch]$Stop,
    [switch]$Status
)

# ─────────────────────────────────────────────────────────────────────────────
# 2026-10-19 重构：启动链路改为「守护进程托管」，解决反复出现的
# 「VSCode 里启动客户端，访问不了」。
#
# 旧实现的三个硬伤：
#   ① vite 是一次性前台进程，挂了没人管 → 端口空着，浏览器 ERR_CONNECTION_REFUSED；
#   ② 进程随 VSCode 调试会话/终端消失，且不留日志 → 故障无法定位，只能靠重开碰运气；
#   ③ 浏览器在 vite 真正 bind 前就被打开（脚本先打印 URL）→ 必然撞空端口。
#
# 现在 dev / preview 一律交给 scripts/dev_server.mjs（幂等复用 + 僵死回收 + 就绪门 +
# 退避重启 + 日志落盘）。本脚本只保留它不可替代的部分：Node.js 运行时发现与版本校验。
# 注意：dev/preview 不再经过 npm.cmd（历史上 VSCode 集成终端里 npm 包装层会挂起）。
# ─────────────────────────────────────────────────────────────────────────────

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
        if (-not (Test-Path -LiteralPath $node)) { continue }
        try {
            [void](Test-SupportedNodeVersion $node)
            return $candidate
        }
        catch {
            Write-Verbose "Skipping unsupported Node.js runtime at ${candidate}: $($_.Exception.Message)"
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
No Node.js runtime containing node.exe was found.
Install Node.js >=20.19 or >=22.12, or point AI2050_NODE_HOME to a Node.js directory:
  `$env:AI2050_NODE_HOME = 'C:\path\to\nodejs'
"@
}

$nodeExecutable = Join-Path $nodeHome 'node.exe'
$npmCli = Join-Path $nodeHome 'node_modules\npm\bin\npm-cli.js'
$nodeVersion = Test-SupportedNodeVersion $nodeExecutable
$env:PATH = "$nodeHome;$env:PATH"

Write-Host "[AI2050] Node.js v$nodeVersion" -ForegroundColor Green
Write-Host "[AI2050] Node home: $nodeHome" -ForegroundColor DarkGray
Write-Host "[AI2050] Frontend: $frontendRoot" -ForegroundColor DarkGray

$devServer = Join-Path $PSScriptRoot 'dev_server.mjs'
if (-not (Test-Path -LiteralPath $devServer)) {
    throw "Launcher not found: $devServer"
}

# 停止 / 状态：直接转给守护器（它会核对 PID、等端口真正释放，不谎报成功）
if ($Stop) {
    & $nodeExecutable $devServer --stop --port $Port
    exit $LASTEXITCODE
}
if ($Status) {
    & $nodeExecutable $devServer --status --port $Port
    exit $LASTEXITCODE
}

Push-Location $frontendRoot
try {
    $viteScript = Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'
    $eslintScript = Join-Path $frontendRoot 'node_modules\eslint\bin\eslint.js'
    if ($Install -or -not (Test-Path -LiteralPath $viteScript)) {
        Write-Host '[AI2050] Installing locked frontend dependencies...' -ForegroundColor Yellow
        if (-not (Test-Path -LiteralPath $npmCli)) {
            throw "npm CLI not found beside the selected Node.js runtime: $npmCli"
        }
        & $nodeExecutable $npmCli ci
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    switch ($Mode) {
        'dev' {
            $launcherArgs = @($devServer, '--port', "$Port")
            if ($Foreground) { $launcherArgs += '--foreground' }
            if ($Detach) { $launcherArgs += '--detach' }
            & $nodeExecutable @launcherArgs
            exit $LASTEXITCODE
        }
        'preview' {
            if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'dist\index.html'))) {
                & $nodeExecutable $viteScript build
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            $previewArgs = @($devServer, '--port', "$Port", '--mode', 'preview')
            if ($Foreground) { $previewArgs += '--foreground' }
            if ($Detach) { $previewArgs += '--detach' }
            & $nodeExecutable @previewArgs
            exit $LASTEXITCODE
        }
        'client' {
            # 1) vite dev 由守护器托管（幂等复用、就绪后才返回）
            & $nodeExecutable $devServer --port "$Port" --detach
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            # 2) Electron 二进制：npm ci 跳过下载时补装（electron/install.js 按包内 checksums.json 核对）
            $electronExe = Join-Path $frontendRoot 'node_modules\electron\dist\electron.exe'
            if (-not (Test-Path -LiteralPath $electronExe)) {
                Write-Host '[AI2050] Downloading the Electron binary (first run only)...' -ForegroundColor Yellow
                & $nodeExecutable (Join-Path $frontendRoot 'node_modules\electron\install.js')
                if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $electronExe)) { throw "Electron binary not installed: $electronExe" }
            }
            # 3) 客户端加载 dev 服务器（只在未打包时生效，见 electron/main.js 的 AI2050_CLIENT_DEV_URL）
            $env:AI2050_CLIENT_DEV_URL = "http://localhost:$Port/"
            Write-Host "[AI2050] Electron client -> $($env:AI2050_CLIENT_DEV_URL)" -ForegroundColor Green
            & $electronExe $frontendRoot
            exit $LASTEXITCODE
        }
        'build' { & $nodeExecutable $viteScript build; exit $LASTEXITCODE }
        'lint' {
            if (-not (Test-Path -LiteralPath $eslintScript)) {
                throw "ESLint entry point not found: $eslintScript"
            }
            & $nodeExecutable $eslintScript .
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}
