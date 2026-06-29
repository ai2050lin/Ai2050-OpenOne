# Phase 749 Suppressor Component Decomposition (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: subunit donor-recipient deltas measured against route-level max logits.

| model | component | subunit | kind | n | donor top1 | target boost | route suppression | coverage | margin gain | delta fraction | effect |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out | L33:attn_out:topH4 | attn_headset | 12 | 0.000 | 1.589 | 3.406 | 3.92 | 2.261 | 0.856 | `global_suppressor_margin_candidate` |
| qwen3 | L32:mlp_out | L32:mlp_out | whole_component | 12 | 0.000 | -0.047 | 3.094 | 3.83 | 0.495 | 1.000 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H9 | attn_head | 2 | 0.000 | 0.000 | 2.688 | 2.50 | 0.525 | 0.375 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out | whole_component | 12 | 0.000 | 2.948 | 2.656 | 2.67 | 3.343 | 1.000 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:topH2 | attn_headset | 12 | 0.000 | 1.089 | 2.260 | 2.83 | 1.492 | 0.720 | `global_suppressor_margin_candidate` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 12 | 0.000 | 0.245 | 1.979 | 4.50 | 0.647 | 0.365 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:topH1 | attn_headset | 12 | 0.000 | 0.609 | 1.844 | 3.08 | 0.964 | 0.434 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H28 | attn_head | 2 | 0.000 | -0.094 | 1.812 | 5.00 | 0.269 | 0.257 | `small_or_no_effect` |
| qwen3 | L33:attn_out | L33:attn_out:H16 | attn_head | 6 | 0.000 | 0.052 | 1.750 | 3.50 | 0.385 | 0.392 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H7 | attn_head | 1 | 0.000 | -0.250 | 1.625 | 4.00 | 0.075 | 0.182 | `small_or_no_effect` |
| qwen3 | L33:attn_out | L33:attn_out:H23 | attn_head | 8 | 0.000 | 1.070 | 1.562 | 2.88 | 1.365 | 0.403 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:topH1 | attn_headset | 12 | 0.000 | 0.703 | 1.542 | 2.83 | 0.977 | 0.480 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H19 | attn_head | 2 | 0.000 | 0.000 | 1.312 | 4.00 | 0.263 | 0.485 | `small_or_no_effect` |
| qwen3 | L32:attn_out | L32:attn_out:H3 | attn_head | 6 | 0.000 | 1.302 | 1.250 | 3.33 | 1.528 | 0.386 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:topH2 | attn_headset | 12 | 0.083 | 0.880 | 1.250 | 2.33 | 1.036 | 0.654 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:H11 | attn_head | 11 | 0.000 | 0.188 | 1.216 | 2.36 | 0.393 | 0.444 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 12 | 0.000 | -0.005 | 1.167 | 3.92 | 0.217 | 0.233 | `small_or_no_effect` |
| qwen3 | L32:attn_out | L32:attn_out:topH4 | attn_headset | 12 | 0.167 | 1.359 | 1.135 | 2.08 | 1.431 | 0.757 | `global_suppressor_margin_candidate` |
| glm4 | L36:mlp_out | L36:mlp_out | whole_component | 12 | 0.333 | -0.344 | 1.086 | 3.25 | -0.359 | 1.000 | `global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:topH4 | attn_headset | 12 | 0.333 | 0.099 | 0.841 | 3.58 | 0.236 | 0.662 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:topH4 | attn_headset | 12 | 0.667 | 0.729 | 0.659 | 2.33 | 0.661 | 0.668 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 12 | 0.333 | -0.026 | 0.654 | 3.92 | 0.089 | 0.299 | `global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:topH2 | attn_headset | 12 | 0.500 | 0.135 | 0.565 | 2.75 | 0.211 | 0.499 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out | whole_component | 12 | 0.667 | 0.552 | 0.529 | 2.83 | 0.552 | 1.000 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out | whole_component | 12 | 0.833 | 1.896 | 0.526 | 2.17 | 1.712 | 1.000 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:H10 | attn_head | 3 | 0.333 | -0.062 | 0.479 | 2.00 | -0.007 | 0.375 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H9 | attn_head | 3 | 0.667 | 0.062 | 0.458 | 4.33 | 0.143 | 0.270 | `global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:topH2 | attn_headset | 12 | 0.667 | 0.630 | 0.383 | 2.00 | 0.546 | 0.501 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:topH1 | attn_headset | 12 | 0.583 | 0.042 | 0.365 | 2.17 | 0.099 | 0.386 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H1 | attn_head | 3 | 1.000 | -0.188 | 0.354 | 1.67 | -0.156 | 0.185 | `global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:H9 | attn_head | 6 | 0.333 | -0.010 | 0.349 | 2.17 | 0.056 | 0.199 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:H2 | attn_head | 6 | 0.333 | -0.208 | 0.349 | 1.83 | -0.165 | 0.259 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:H12 | attn_head | 3 | 0.000 | -0.042 | 0.312 | 3.67 | 0.026 | 0.473 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H7 | attn_head | 6 | 0.667 | 0.167 | 0.302 | 2.00 | 0.207 | 0.393 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:topH1 | attn_headset | 12 | 0.500 | 0.276 | 0.297 | 1.33 | 0.244 | 0.411 | `small_or_no_effect` |
| glm4 | L36:mlp_out | L36:mlp_out:topC16 | mlp_channelset | 12 | 0.500 | -0.016 | 0.279 | 3.08 | 0.027 | 0.193 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out | whole_component | 12 | 0.167 | 1.208 | 3.323 | 2.92 | 1.739 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH4 | attn_headset | 12 | 0.167 | 0.958 | 2.768 | 3.17 | 1.432 | 0.708 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out | whole_component | 12 | 0.167 | 0.354 | 2.292 | 3.42 | 0.719 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH2 | attn_headset | 12 | 0.167 | 0.708 | 2.281 | 3.08 | 1.086 | 0.601 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out | whole_component | 12 | 0.250 | 1.620 | 2.057 | 3.17 | 1.903 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:topH4 | attn_headset | 12 | 0.250 | 1.266 | 1.911 | 3.00 | 1.590 | 0.739 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:H11 | attn_head | 9 | 0.222 | 1.160 | 1.715 | 3.56 | 1.454 | 0.601 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:H1 | attn_head | 11 | 0.273 | 0.869 | 1.568 | 2.82 | 1.056 | 0.536 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH1 | attn_headset | 12 | 0.167 | 0.547 | 1.490 | 2.92 | 0.750 | 0.477 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out:C1435 | mlp_channel | 1 | 0.000 | -0.188 | 1.375 | 5.00 | 0.087 | 0.061 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H7 | attn_head | 5 | 0.000 | 0.138 | 1.363 | 3.80 | 0.392 | 0.264 | `small_or_no_effect` |
| deepseek7b | L23:attn_out | L23:attn_out:topH1 | attn_headset | 12 | 0.250 | 0.807 | 1.193 | 3.17 | 0.991 | 0.513 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out:topC64 | mlp_channelset | 12 | 0.167 | 0.359 | 1.177 | 3.67 | 0.564 | 0.310 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:topH2 | attn_headset | 12 | 0.167 | 0.990 | 1.172 | 2.92 | 1.151 | 0.607 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out:C2139 | mlp_channel | 1 | 0.000 | -0.125 | 1.125 | 4.00 | 0.156 | 0.034 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H22 | attn_head | 3 | 0.000 | 0.042 | 1.125 | 3.67 | 0.270 | 0.283 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H8 | attn_head | 2 | 0.000 | -0.156 | 1.000 | 4.00 | 0.094 | 0.163 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H9 | attn_head | 1 | 0.000 | -0.250 | 1.000 | 3.00 | 0.000 | 0.075 | `small_or_no_effect` |

## Strict Interpretation

- Attention decomposition is head-level o_proj projected delta evidence.
- MLP decomposition is residual output channel evidence, not true neuron evidence.
- A small subunit matching the whole component's route suppression is a localization hint, not proof of natural coding origin.
