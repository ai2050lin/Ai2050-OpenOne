param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SmokeArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$userProfile = [Environment]::GetFolderPath("UserProfile")
$candidates = @(
    (Join-Path $repoRoot ".venv-qwen38\Scripts\python.exe"),
    (Join-Path $userProfile ".workbuddy\binaries\python\versions\3.11.9\python.exe")
)

$python = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $candidate -c "import bitsandbytes, torch, transformers; assert torch.cuda.is_available(); assert hasattr(transformers, 'Qwen3_5ForConditionalGeneration')" *> $null
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($probeExitCode -eq 0) {
        $python = $candidate
        break
    }
}

if ($null -eq $python) {
    throw "No compatible Python 3.11 + CUDA + Transformers + bitsandbytes runtime was found."
}

if (-not $SmokeArgs -or $SmokeArgs.Count -eq 0) {
    $SmokeArgs = @("preflight")
}

Write-Host "Qwen3.8-27B runtime: $python"
& $python (Join-Path $PSScriptRoot "qwen38_27b_nf4_smoke.py") @SmokeArgs
exit $LASTEXITCODE
