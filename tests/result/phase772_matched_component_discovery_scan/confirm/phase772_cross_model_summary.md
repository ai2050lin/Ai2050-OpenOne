# Phase 772 Matched Component Discovery Scan (confirm)

- Status: `complete`
- Test: scan layer/head/source components by direct score, then causally remove top components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | scan score | target drop | margin drop | top1 loss | attention | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `scan_top_component` | 10 | 5 | 16.824 | 0.600 | 0.825 | 0.200 | 0.927 | 11.356 | 0.582 |
| qwen3 | `same_layer_control_head` | 10 | 5 | 0.138 | 0.037 | -0.006 | 0.000 | 0.861 | -0.376 | 0.164 |
| glm4 | `scan_top_component` | 10 | 5 | 0.600 | 0.025 | 0.087 | 0.000 | 0.876 | 0.264 | 0.093 |
| glm4 | `same_layer_control_head` | 10 | 5 | 0.062 | 0.013 | 0.009 | 0.000 | 0.656 | -0.002 | 0.041 |
| deepseek7b | `scan_top_component` | 10 | 5 | 21.446 | 0.512 | 0.156 | 0.000 | 0.805 | 15.666 | 0.524 |
| deepseek7b | `same_layer_control_head` | 10 | 5 | 0.522 | 0.044 | 0.025 | 0.000 | 0.433 | -0.308 | 0.872 |

## Fiber Bucket Summary

| model | bucket | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `fiber_high` | 20 | 5 | 0.319 | 0.409 | 0.100 | 5.490 | 0.373 |
| glm4 | `fiber_high` | 20 | 5 | 0.019 | 0.048 | 0.000 | 0.131 | 0.067 |
| deepseek7b | `fiber_high` | 20 | 5 | 0.278 | 0.091 | 0.000 | 7.679 | 0.698 |

## Top Components

| model | component | rows | cases | scan score | target drop | margin drop | top1 loss |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `L32:attn_out:H5:instruction` | 1 | 1 | 22.877 | 1.250 | 1.000 | 1.000 |
| qwen3 | `L33:attn_out:H31:instruction` | 1 | 1 | 3.177 | 0.500 | 0.500 | 1.000 |
| qwen3 | `L32:attn_out:H7:instruction` | 1 | 1 | 33.961 | 1.625 | 1.938 | 0.000 |
| qwen3 | `L33:attn_out:H16:instruction` | 2 | 2 | 26.730 | 1.125 | 2.188 | 0.000 |
| qwen3 | `L33:attn_out:H30:instruction` | 1 | 1 | 9.054 | 0.250 | 0.125 | 0.000 |
| qwen3 | `L33:attn_out:H20:instruction` | 2 | 2 | 6.172 | 0.188 | 0.062 | 0.000 |
| qwen3 | `L33:attn_out:H15:instruction` | 1 | 1 | 13.678 | 0.125 | 0.438 | 0.000 |
| qwen3 | `L33:attn_out:H4:instruction` | 1 | 1 | 0.039 | 0.125 | 0.125 | 0.000 |
| qwen3 | `L30:attn_out:H29:instruction` | 2 | 2 | 0.087 | 0.062 | 0.062 | 0.000 |
| qwen3 | `L33:attn_out:H21:instruction` | 2 | 2 | 0.095 | 0.062 | -0.188 | 0.000 |
| qwen3 | `L33:attn_out:H25:instruction` | 1 | 1 | 0.455 | 0.000 | 0.125 | 0.000 |
| qwen3 | `L32:attn_out:H10:instruction` | 1 | 1 | 0.051 | 0.000 | 0.125 | 0.000 |
| glm4 | `L34:attn_out:H13:instruction` | 1 | 1 | 1.243 | 0.312 | 0.312 | 0.000 |
| glm4 | `L35:attn_out:H28:instruction` | 1 | 1 | 0.451 | 0.125 | 0.125 | 0.000 |
| glm4 | `L38:attn_out:H26:instruction` | 1 | 1 | 0.187 | 0.125 | 0.125 | 0.000 |
| glm4 | `L34:attn_out:H22:instruction` | 1 | 1 | 1.121 | 0.062 | 0.125 | 0.000 |
| glm4 | `L38:attn_out:H31:instruction` | 1 | 1 | 0.135 | 0.062 | 0.062 | 0.000 |
| glm4 | `L34:attn_out:H27:instruction` | 1 | 1 | 0.045 | 0.062 | 0.000 | 0.000 |
| glm4 | `L34:attn_out:H1:instruction` | 2 | 2 | 0.877 | 0.031 | 0.062 | 0.000 |
| glm4 | `L34:attn_out:H20:instruction` | 2 | 2 | 0.383 | 0.031 | 0.031 | 0.000 |
| glm4 | `L38:attn_out:H6:instruction` | 1 | 1 | 0.197 | 0.000 | 0.031 | 0.000 |
| glm4 | `L38:attn_out:H11:instruction` | 1 | 1 | 0.159 | 0.000 | 0.031 | 0.000 |
| glm4 | `L34:attn_out:H7:instruction` | 1 | 1 | 0.064 | 0.000 | 0.000 | 0.000 |
| glm4 | `L35:attn_out:H1:instruction` | 1 | 1 | 0.063 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `L27:attn_out:H21:instruction` | 2 | 2 | 25.321 | 0.844 | 0.688 | 0.000 |
| deepseek7b | `L25:attn_out:H11:instruction` | 1 | 1 | 27.138 | 0.688 | 0.375 | 0.000 |
| deepseek7b | `L27:attn_out:H23:instruction` | 3 | 3 | 26.622 | 0.542 | 0.302 | 0.000 |
| deepseek7b | `L27:attn_out:H17:instruction` | 1 | 1 | 13.999 | 0.500 | 0.375 | 0.000 |
| deepseek7b | `L23:attn_out:H19:instruction` | 2 | 2 | 15.464 | 0.375 | -0.750 | 0.000 |
| deepseek7b | `L25:attn_out:H16:instruction` | 1 | 1 | 0.034 | 0.188 | 0.062 | 0.000 |
| deepseek7b | `L27:attn_out:H26:instruction` | 2 | 2 | 1.635 | 0.125 | 0.125 | 0.000 |
| deepseek7b | `L27:attn_out:H22:instruction` | 1 | 1 | 0.322 | 0.125 | 0.000 | 0.000 |
| deepseek7b | `L23:attn_out:H24:instruction` | 2 | 2 | 0.088 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `L27:attn_out:H10:instruction` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `L27:attn_out:H0:instruction` | 3 | 3 | 0.474 | -0.042 | -0.021 | 0.000 |
| deepseek7b | `L27:attn_out:H5:instruction` | 1 | 1 | 11.893 | -0.125 | 0.031 | 0.000 |

## Strict Interpretation

- Direct-score discovery is only a filter. The causal-removal columns are the primary evidence.
- If scan-top components have weak target/margin drops, component discovery has not found sufficient causal paths.
- If top components recur across cases, they are better atlas candidates than Phase755 global heads.
- This is still head/source-level evidence, not neuron/channel-level evidence.
