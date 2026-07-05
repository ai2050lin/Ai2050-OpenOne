$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RootDir

$RoundName = if ($args.Count -ge 1) { $args[0] } else { "color_feature_neuron_atlas" }
$CommonArgs = @(
  "--round-name", $RoundName,
  "--templates-per-object", "4",
  "--layers", "auto",
  "--batch-size", "4",
  "--topk-blockers", "16",
  "--keep-top-channels-per-sample", "128",
  "--keep-channel-rows", "20000",
  "--summary-top-channels", "50",
  "--log-every", "5"
)

foreach ($Model in @("qwen3", "glm4", "deepseek7b")) {
  python tests\glm5\phase941_color_feature_neuron_atlas.py --model $Model @CommonArgs
}

python tests\glm5\phase941_color_feature_neuron_atlas.py --summarize-round --round-name $RoundName
