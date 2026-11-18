$ErrorActionPreference='Stop'
$env:PYTHONPATH='tests/glm5;.'
$env:PYTHONUTF8='1'
foreach ($sfStage in @('confirm','causal','capability')) {
    Write-Output (@{time=(Get-Date -Format o);stage=$sfStage;status='starting'} | ConvertTo-Json -Compress)
    & .venv/Scripts/python.exe -B -u tests/glm5/state_formation_runner.py --stage $sfStage *> ("tests/glm5/result/state_formation_v1/"+$sfStage+'.log')
    if ($LASTEXITCODE -ne 0) { throw "Stage $sfStage stopped with exit $LASTEXITCODE; completed evidence retained" }
    Write-Output (@{time=(Get-Date -Format o);stage=$sfStage;status='completed'} | ConvertTo-Json -Compress)
}
