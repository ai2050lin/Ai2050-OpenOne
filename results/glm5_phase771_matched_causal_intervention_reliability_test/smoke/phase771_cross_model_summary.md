# Phase 771 Matched Causal Intervention Reliability Test (smoke)

- Status: `complete`
- Test: matched semantic-clean vs semantic-fail direct source-contribution removal.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Matched Arm Summary

| model | arm | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `clean` | 2 | -0.062 | -0.406 | 0.000 | 0.497 | -0.017 | 0.252 |
| qwen3 | `fail` | 2 | -0.062 | -0.375 | 0.000 | 0.498 | 0.062 | 0.255 |
| glm4 | `clean` | 2 | -0.031 | -0.031 | 0.000 | 0.467 | -0.013 | 0.012 |
| glm4 | `fail` | 2 | 0.000 | 0.031 | 0.000 | 0.470 | -0.011 | 0.008 |
| deepseek7b | `clean` | 2 | 0.000 | -0.031 | 0.000 | 0.428 | -0.060 | 0.250 |
| deepseek7b | `fail` | 2 | -0.031 | -0.062 | 0.000 | 0.416 | -0.069 | 0.228 |

## Semantic Label Summary

| model | semantic label | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `exact_clean` | 2 | -0.062 | -0.406 | 0.000 | 0.497 | -0.017 | 0.252 |
| qwen3 | `semantic_fail` | 2 | -0.062 | -0.375 | 0.000 | 0.498 | 0.062 | 0.255 |
| glm4 | `exact_clean` | 2 | -0.031 | -0.031 | 0.000 | 0.467 | -0.013 | 0.012 |
| glm4 | `semantic_fail` | 2 | 0.000 | 0.031 | 0.000 | 0.470 | -0.011 | 0.008 |
| deepseek7b | `semantic_fail` | 2 | -0.031 | -0.062 | 0.000 | 0.416 | -0.069 | 0.228 |
| deepseek7b | `semantic_only` | 2 | 0.000 | -0.031 | 0.000 | 0.428 | -0.060 | 0.250 |

## Fiber Bucket Summary

| model | fiber bucket | n rows | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_high` | 2 | -0.062 | -0.406 | 0.000 | 0.497 | -0.017 | 0.252 |
| qwen3 | `fiber_low` | 2 | -0.062 | -0.375 | 0.000 | 0.498 | 0.062 | 0.255 |
| glm4 | `fiber_high` | 4 | -0.016 | 0.000 | 0.000 | 0.469 | -0.012 | 0.010 |
| deepseek7b | `fiber_high` | 4 | -0.016 | -0.047 | 0.000 | 0.422 | -0.064 | 0.239 |

## Strict Interpretation

- If clean rows have larger target/margin drops than fail rows, output-clean states are more causally dependent on the tested source paths.
- If fiber-high rows have larger drops than fiber-low rows, paired fiber stability predicts intervention sensitivity.
- This is still a head/source intervention, not a neuron/channel atlas.
- Because the test uses allowed-value commonsense prompts, it does not prove free-generation closure.
