# Phase 772 Matched Component Discovery Scan (smoke)

- Status: `complete`
- Test: scan layer/head/source components by direct score, then causally remove top components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | scan score | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `scan_top_component` | 2 | 2 | 31.175 | 2.562 | 2.781 | 0.000 | 0.931 | 20.248 | 1.784 |
| glm4 | `scan_top_component` | 2 | 2 | 1.431 | 0.375 | 0.344 | 0.000 | 0.898 | 0.966 | 0.072 |
| deepseek7b | `scan_top_component` | 2 | 2 | 1.541 | -0.250 | -0.094 | 0.000 | 0.752 | 0.600 | 1.156 |

## Fiber Bucket Summary

| model | bucket | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_low` | 1 | 1 | 3.500 | 3.625 | 0.000 | 18.509 | 1.630 |
| qwen3 | `fiber_high` | 1 | 1 | 1.625 | 1.938 | 0.000 | 21.987 | 1.939 |
| glm4 | `fiber_high` | 2 | 2 | 0.375 | 0.344 | 0.000 | 0.966 | 0.072 |
| deepseek7b | `fiber_high` | 2 | 2 | -0.250 | -0.094 | 0.000 | 0.600 | 1.156 |

## Top Components

| model | component | rows | cases | scan score | target drop | margin drop | top1 loss |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `L32:attn_out:H7:instruction` | 2 | 2 | 31.175 | 2.562 | 2.781 | 0.000 |
| glm4 | `L34:attn_out:H13:instruction` | 2 | 2 | 1.431 | 0.375 | 0.344 | 0.000 |
| deepseek7b | `L23:attn_out:H19:question` | 1 | 1 | 1.768 | -0.250 | -0.062 | 0.000 |
| deepseek7b | `L23:attn_out:H20:instruction` | 1 | 1 | 1.314 | -0.250 | -0.125 | 0.000 |

## Strict Interpretation

- Direct-score discovery is only a filter. The causal-removal columns are the primary evidence.
- If scan-top components have weak target/margin drops, component discovery has not found sufficient causal paths.
- If top components recur across cases, they are better atlas candidates than Phase755 global heads.
- This is still head/source-level evidence, not neuron/channel-level evidence.
