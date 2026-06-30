# Phase 771 Matched Causal Intervention Reliability Test (main)

- Status: `complete`
- Test: matched semantic-clean vs semantic-fail direct source-contribution removal.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Matched Arm Summary

| model | arm | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `clean` | 24 | -0.010 | -0.065 | 0.000 | 0.252 | 0.217 | 0.074 |
| qwen3 | `fail` | 24 | -0.026 | -0.109 | 0.000 | 0.249 | 0.151 | 0.068 |
| glm4 | `clean` | 24 | 0.021 | 0.018 | 0.000 | 0.251 | 0.000 | 0.004 |
| glm4 | `fail` | 24 | 0.008 | 0.023 | 0.000 | 0.253 | -0.002 | 0.002 |
| deepseek7b | `clean` | 24 | 0.013 | -0.005 | 0.000 | 0.238 | 0.003 | 0.113 |
| deepseek7b | `fail` | 24 | 0.016 | -0.044 | 0.000 | 0.235 | -0.002 | 0.113 |

## Semantic Label Summary

| model | semantic label | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `exact_clean` | 24 | -0.010 | -0.065 | 0.000 | 0.252 | 0.217 | 0.074 |
| qwen3 | `semantic_fail` | 24 | -0.026 | -0.109 | 0.000 | 0.249 | 0.151 | 0.068 |
| glm4 | `exact_clean` | 24 | 0.021 | 0.018 | 0.000 | 0.251 | 0.000 | 0.004 |
| glm4 | `semantic_fail` | 24 | 0.008 | 0.023 | 0.000 | 0.253 | -0.002 | 0.002 |
| deepseek7b | `exact_clean` | 12 | -0.031 | 0.036 | 0.000 | 0.244 | 0.031 | 0.140 |
| deepseek7b | `semantic_fail` | 24 | 0.016 | -0.044 | 0.000 | 0.235 | -0.002 | 0.113 |
| deepseek7b | `semantic_only` | 12 | 0.057 | -0.047 | 0.000 | 0.233 | -0.024 | 0.085 |

## Fiber Bucket Summary

| model | fiber bucket | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_high` | 32 | -0.008 | -0.084 | 0.000 | 0.252 | 0.176 | 0.087 |
| qwen3 | `fiber_low` | 16 | -0.039 | -0.094 | 0.000 | 0.248 | 0.199 | 0.039 |
| glm4 | `fiber_high` | 36 | 0.021 | 0.023 | 0.000 | 0.245 | -0.002 | 0.003 |
| glm4 | `fiber_low` | 12 | -0.005 | 0.016 | 0.000 | 0.273 | 0.001 | 0.001 |
| deepseek7b | `fiber_high` | 28 | 0.002 | 0.007 | 0.000 | 0.236 | 0.011 | 0.163 |
| deepseek7b | `fiber_low` | 20 | 0.031 | -0.069 | 0.000 | 0.238 | -0.014 | 0.043 |

## Strict Interpretation

- If clean rows have larger target/margin drops than fail rows, output-clean states are more causally dependent on the tested source paths.
- If fiber-high rows have larger drops than fiber-low rows, paired fiber stability predicts intervention sensitivity.
- This is still a head/source intervention, not a neuron/channel atlas.
- Because the test uses allowed-value commonsense prompts, it does not prove free-generation closure.
