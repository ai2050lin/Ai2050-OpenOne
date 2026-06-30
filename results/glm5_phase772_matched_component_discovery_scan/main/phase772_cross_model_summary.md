# Phase 772 Matched Component Discovery Scan (main)

- Status: `complete`
- Test: scan layer/head/source components by direct score, then causally remove top components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | scan score | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `scan_top_component` | 24 | 8 | 10.666 | 0.740 | 0.656 | 0.042 | 0.898 | 7.682 | 0.324 |
| glm4 | `scan_top_component` | 24 | 8 | 0.607 | 0.070 | 0.109 | 0.000 | 0.854 | 0.357 | 0.063 |
| deepseek7b | `scan_top_component` | 24 | 8 | 14.660 | 0.557 | 0.289 | 0.083 | 0.758 | 10.845 | 0.647 |

## Fiber Bucket Summary

| model | bucket | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_low` | 9 | 3 | 0.583 | 0.458 | 0.111 | 9.040 | 0.000 |
| qwen3 | `fiber_high` | 15 | 5 | 0.833 | 0.775 | 0.000 | 6.868 | 0.518 |
| glm4 | `fiber_high` | 18 | 6 | 0.080 | 0.139 | 0.000 | 0.375 | 0.082 |
| glm4 | `fiber_low` | 6 | 2 | 0.042 | 0.021 | 0.000 | 0.303 | 0.005 |
| deepseek7b | `fiber_high` | 12 | 4 | 0.635 | 0.469 | 0.167 | 13.244 | 0.751 |
| deepseek7b | `fiber_low` | 12 | 4 | 0.479 | 0.109 | 0.000 | 8.447 | 0.543 |

## Top Components

| model | component | rows | cases | scan score | target drop | margin drop | top1 loss |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `L32:attn_out:H5:instruction` | 6 | 6 | 23.702 | 2.125 | 1.812 | 0.167 |
| qwen3 | `L32:attn_out:H7:instruction` | 1 | 1 | 33.961 | 1.625 | 1.938 | 0.000 |
| qwen3 | `L30:attn_out:H27:instruction` | 1 | 1 | 7.996 | 0.750 | 0.625 | 0.000 |
| qwen3 | `L33:attn_out:H20:instruction` | 1 | 1 | 12.137 | 0.375 | 0.250 | 0.000 |
| qwen3 | `L33:attn_out:H31:instruction` | 6 | 6 | 3.819 | 0.375 | 0.208 | 0.000 |
| qwen3 | `L33:attn_out:H21:instruction` | 1 | 1 | 1.897 | 0.375 | 0.125 | 0.000 |
| qwen3 | `L32:attn_out:H3:instruction` | 1 | 1 | 6.828 | 0.125 | 0.688 | 0.000 |
| qwen3 | `L33:attn_out:H29:instruction` | 1 | 1 | 5.072 | 0.000 | 0.188 | 0.000 |
| qwen3 | `L30:attn_out:H24:instruction` | 5 | 5 | 3.455 | -0.025 | -0.025 | 0.000 |
| qwen3 | `L32:attn_out:H8:instruction` | 1 | 1 | 5.694 | -0.375 | -0.062 | 0.000 |
| glm4 | `L34:attn_out:H10:instruction` | 1 | 1 | 2.140 | 0.562 | 0.688 | 0.000 |
| glm4 | `L34:attn_out:H13:instruction` | 1 | 1 | 1.243 | 0.312 | 0.312 | 0.000 |
| glm4 | `L35:attn_out:H28:instruction` | 2 | 2 | 0.554 | 0.156 | 0.188 | 0.000 |
| glm4 | `L38:attn_out:H26:instruction` | 1 | 1 | 0.190 | 0.125 | 0.125 | 0.000 |
| glm4 | `L34:attn_out:H1:instruction` | 5 | 5 | 0.933 | 0.087 | 0.075 | 0.000 |
| glm4 | `L34:attn_out:H20:instruction` | 5 | 5 | 0.407 | 0.025 | 0.000 | 0.000 |
| glm4 | `L39:attn_out:H31:question` | 1 | 1 | 0.176 | 0.000 | 0.062 | 0.000 |
| glm4 | `L34:attn_out:H22:instruction` | 1 | 1 | 1.604 | 0.000 | 0.000 | 0.000 |
| glm4 | `L38:attn_out:H6:instruction` | 5 | 5 | 0.131 | -0.013 | -0.025 | 0.000 |
| glm4 | `L34:attn_out:H2:instruction` | 2 | 2 | 0.375 | -0.062 | 0.406 | 0.000 |
| deepseek7b | `L23:attn_out:H11:instruction` | 1 | 1 | 10.429 | 0.938 | 1.375 | 1.000 |
| deepseek7b | `L27:attn_out:H24:instruction` | 1 | 1 | 27.580 | 0.938 | 1.250 | 1.000 |
| deepseek7b | `L27:attn_out:H21:instruction` | 1 | 1 | 36.189 | 1.125 | 0.938 | 0.000 |
| deepseek7b | `L27:attn_out:H17:instruction` | 2 | 2 | 22.238 | 0.781 | 0.719 | 0.000 |
| deepseek7b | `L27:attn_out:H20:instruction` | 1 | 1 | 22.307 | 0.688 | 0.625 | 0.000 |
| deepseek7b | `L25:attn_out:H11:instruction` | 1 | 1 | 27.138 | 0.688 | 0.375 | 0.000 |
| deepseek7b | `L23:attn_out:H19:instruction` | 5 | 5 | 14.010 | 0.550 | 0.087 | 0.000 |
| deepseek7b | `L25:attn_out:H13:instruction` | 1 | 1 | 25.767 | 0.500 | 0.625 | 0.000 |
| deepseek7b | `L27:attn_out:H27:instruction` | 4 | 4 | 6.837 | 0.484 | 0.203 | 0.000 |
| deepseek7b | `L27:attn_out:H23:instruction` | 4 | 4 | 9.532 | 0.469 | -0.344 | 0.000 |
| deepseek7b | `L27:attn_out:H5:instruction` | 1 | 1 | 10.999 | 0.250 | 0.312 | 0.000 |
| deepseek7b | `L27:attn_out:H10:instruction` | 1 | 1 | 7.375 | 0.250 | 0.000 | 0.000 |

## Strict Interpretation

- Direct-score discovery is only a filter. The causal-removal columns are the primary evidence.
- If scan-top components have weak target/margin drops, component discovery has not found sufficient causal paths.
- If top components recur across cases, they are better atlas candidates than Phase755 global heads.
- This is still head/source-level evidence, not neuron/channel-level evidence.
