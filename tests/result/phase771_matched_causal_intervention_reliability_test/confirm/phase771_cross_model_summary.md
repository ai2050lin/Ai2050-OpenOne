# Phase 771 Matched Causal Intervention Reliability Test (confirm)

- Status: `complete`
- Test: matched semantic-clean vs semantic-fail direct source-contribution removal.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Matched Arm Summary

| model | arm | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `clean` | 80 | -0.023 | -0.022 | 0.000 | 0.257 | 0.240 | 0.070 |
| qwen3 | `fail` | 80 | -0.030 | -0.042 | 0.000 | 0.255 | 0.166 | 0.068 |
| glm4 | `clean` | 80 | 0.007 | 0.018 | 0.000 | 0.266 | 0.001 | 0.002 |
| glm4 | `fail` | 80 | -0.002 | 0.009 | 0.000 | 0.268 | 0.001 | 0.001 |
| deepseek7b | `clean` | 80 | 0.025 | -0.006 | 0.000 | 0.272 | 0.058 | 0.131 |
| deepseek7b | `fail` | 80 | 0.013 | 0.002 | 0.013 | 0.277 | 0.098 | 0.118 |

## Semantic Label Summary

| model | semantic label | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `exact_clean` | 80 | -0.023 | -0.022 | 0.000 | 0.257 | 0.240 | 0.070 |
| qwen3 | `semantic_fail` | 80 | -0.030 | -0.042 | 0.000 | 0.255 | 0.166 | 0.068 |
| glm4 | `exact_clean` | 72 | 0.006 | 0.013 | 0.000 | 0.269 | 0.002 | 0.002 |
| glm4 | `semantic_fail` | 80 | -0.002 | 0.009 | 0.000 | 0.268 | 0.001 | 0.001 |
| glm4 | `semantic_only` | 8 | 0.016 | 0.062 | 0.000 | 0.241 | -0.001 | 0.000 |
| deepseek7b | `exact_clean` | 32 | 0.002 | 0.003 | 0.000 | 0.259 | -0.012 | 0.198 |
| deepseek7b | `semantic_fail` | 80 | 0.013 | 0.002 | 0.013 | 0.277 | 0.098 | 0.118 |
| deepseek7b | `semantic_only` | 48 | 0.040 | -0.012 | 0.000 | 0.280 | 0.105 | 0.086 |

## Fiber Bucket Summary

| model | fiber bucket | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_high` | 96 | -0.008 | -0.021 | 0.000 | 0.255 | 0.261 | 0.094 |
| qwen3 | `fiber_low` | 64 | -0.055 | -0.049 | 0.000 | 0.257 | 0.116 | 0.031 |
| glm4 | `fiber_high` | 88 | 0.007 | 0.013 | 0.000 | 0.256 | 0.002 | 0.002 |
| glm4 | `fiber_low` | 72 | -0.003 | 0.015 | 0.000 | 0.282 | -0.000 | 0.001 |
| deepseek7b | `fiber_high` | 72 | 0.001 | 0.000 | 0.000 | 0.260 | 0.018 | 0.177 |
| deepseek7b | `fiber_low` | 88 | 0.034 | -0.004 | 0.011 | 0.286 | 0.126 | 0.081 |

## Strict Interpretation

- If clean rows have larger target/margin drops than fail rows, output-clean states are more causally dependent on the tested source paths.
- If fiber-high rows have larger drops than fiber-low rows, paired fiber stability predicts intervention sensitivity.
- This is still a head/source intervention, not a neuron/channel atlas.
- Because the test uses allowed-value commonsense prompts, it does not prove free-generation closure.
