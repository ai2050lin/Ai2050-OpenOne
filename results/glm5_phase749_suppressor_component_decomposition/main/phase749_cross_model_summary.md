# Phase 749 Suppressor Component Decomposition (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: subunit donor-recipient deltas measured against route-level max logits.

| model | component | subunit | kind | n | donor top1 | target boost | route suppression | coverage | margin gain | delta fraction | effect |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out | L33:attn_out:topH4 | attn_headset | 8 | 0.000 | 1.656 | 3.516 | 3.38 | 2.391 | 0.850 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H9 | attn_head | 2 | 0.000 | 0.000 | 2.688 | 2.50 | 0.525 | 0.375 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out | whole_component | 8 | 0.000 | 3.406 | 2.656 | 2.38 | 3.781 | 1.000 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:topH2 | attn_headset | 8 | 0.000 | 1.469 | 2.625 | 2.88 | 1.966 | 0.719 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:H11 | attn_head | 6 | 0.000 | 0.146 | 2.104 | 3.17 | 0.535 | 0.561 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H23 | attn_head | 4 | 0.000 | 1.875 | 2.000 | 3.50 | 2.263 | 0.518 | `global_suppressor_margin_candidate` |
| qwen3 | L32:mlp_out | L32:mlp_out | whole_component | 8 | 0.000 | 0.188 | 1.984 | 3.12 | 0.491 | 1.000 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:topH1 | attn_headset | 8 | 0.000 | 0.938 | 1.891 | 2.62 | 1.308 | 0.446 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:topH2 | attn_headset | 8 | 0.125 | 0.469 | 1.891 | 2.00 | 0.714 | 0.557 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 8 | 0.000 | 0.219 | 1.781 | 4.12 | 0.608 | 0.367 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:topH1 | attn_headset | 8 | 0.000 | 0.125 | 1.781 | 3.12 | 0.466 | 0.425 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | L32:attn_out | whole_component | 8 | 0.250 | 2.359 | 1.734 | 1.50 | 2.383 | 1.000 | `booster_candidate` |
| qwen3 | L32:attn_out | L32:attn_out:topH4 | attn_headset | 8 | 0.250 | 0.938 | 1.672 | 1.62 | 1.083 | 0.683 | `route_specific_suppressor_candidate` |
| qwen3 | L33:attn_out | L33:attn_out:H16 | attn_head | 4 | 0.000 | 0.062 | 1.469 | 2.75 | 0.331 | 0.365 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:C1154 | mlp_channel | 1 | 0.000 | -0.125 | 1.000 | 5.00 | 0.075 | 0.049 | `small_or_no_effect` |
| qwen3 | L33:attn_out | L33:attn_out:H29 | attn_head | 2 | 0.000 | -0.188 | 0.875 | 3.50 | -0.013 | 0.399 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 8 | 0.000 | 0.062 | 0.812 | 3.25 | 0.211 | 0.232 | `small_or_no_effect` |
| qwen3 | L33:attn_out | L33:attn_out:H13 | attn_head | 3 | 0.000 | 0.083 | 0.792 | 0.67 | 0.275 | 0.269 | `route_specific_suppressor_candidate` |
| glm4 | L35:attn_out | L35:attn_out:topH4 | attn_headset | 8 | 0.500 | 0.125 | 0.852 | 3.62 | 0.279 | 0.651 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L36:mlp_out | L36:mlp_out | whole_component | 8 | 0.500 | -0.070 | 0.723 | 2.38 | -0.274 | 1.000 | `global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:topH4 | attn_headset | 8 | 0.750 | 0.547 | 0.691 | 2.38 | 0.493 | 0.633 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 8 | 0.500 | -0.047 | 0.605 | 3.75 | 0.061 | 0.283 | `global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:H2 | attn_head | 3 | 1.000 | -0.500 | 0.573 | 1.67 | -0.429 | 0.279 | `global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:topH2 | attn_headset | 8 | 0.750 | 0.203 | 0.559 | 2.50 | 0.275 | 0.511 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out | whole_component | 8 | 1.000 | 1.758 | 0.555 | 2.00 | 1.508 | 1.000 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out | whole_component | 8 | 0.750 | 0.242 | 0.551 | 2.75 | 0.223 | 1.000 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:topH2 | attn_headset | 8 | 0.750 | 0.555 | 0.465 | 2.25 | 0.478 | 0.477 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:H1 | attn_head | 2 | 1.000 | -0.188 | 0.438 | 2.00 | -0.125 | 0.167 | `global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | L35:attn_out:H9 | attn_head | 4 | 0.500 | 0.062 | 0.375 | 3.75 | 0.126 | 0.279 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H29 | attn_head | 2 | 0.000 | -0.031 | 0.344 | 2.00 | 0.010 | 0.386 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:topH1 | attn_headset | 8 | 0.750 | 0.078 | 0.328 | 2.00 | 0.122 | 0.417 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H12 | attn_head | 2 | 1.000 | 0.344 | 0.312 | 1.50 | 0.344 | 0.344 | `booster_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:topH1 | attn_headset | 8 | 0.750 | 0.352 | 0.312 | 1.12 | 0.295 | 0.407 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H7 | attn_head | 6 | 0.667 | 0.167 | 0.302 | 2.00 | 0.207 | 0.393 | `small_or_no_effect` |
| glm4 | L35:attn_out | L35:attn_out:H8 | attn_head | 3 | 1.000 | 0.125 | 0.260 | 1.67 | 0.147 | 0.258 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:H12 | attn_head | 1 | 0.000 | -0.062 | 0.250 | 3.00 | -0.031 | 0.371 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out | whole_component | 8 | 0.250 | 1.133 | 3.195 | 2.88 | 1.587 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH4 | attn_headset | 8 | 0.250 | 0.922 | 2.734 | 3.12 | 1.366 | 0.715 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out | whole_component | 8 | 0.250 | 0.539 | 2.367 | 3.38 | 0.902 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out | whole_component | 8 | 0.250 | 1.727 | 2.344 | 3.12 | 2.021 | 1.000 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH2 | attn_headset | 8 | 0.250 | 0.852 | 2.227 | 3.00 | 1.196 | 0.614 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:topH4 | attn_headset | 8 | 0.250 | 1.344 | 2.195 | 3.12 | 1.714 | 0.764 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:H11 | attn_head | 6 | 0.167 | 1.198 | 1.906 | 3.83 | 1.543 | 0.605 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:H1 | attn_head | 7 | 0.286 | 0.893 | 1.688 | 2.43 | 1.060 | 0.579 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH1 | attn_headset | 8 | 0.250 | 0.805 | 1.523 | 2.38 | 0.959 | 0.543 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | L25:mlp_out:C1435 | mlp_channel | 1 | 0.000 | -0.188 | 1.375 | 5.00 | 0.087 | 0.061 | `small_or_no_effect` |
| deepseek7b | L23:attn_out | L23:attn_out:topH1 | attn_headset | 8 | 0.250 | 0.797 | 1.336 | 3.25 | 1.007 | 0.536 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:topH2 | attn_headset | 8 | 0.250 | 0.930 | 1.312 | 2.62 | 1.112 | 0.611 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:H15 | attn_head | 1 | 0.000 | -0.188 | 1.188 | 5.00 | 0.050 | 0.139 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H7 | attn_head | 4 | 0.000 | 0.188 | 1.156 | 3.50 | 0.397 | 0.280 | `small_or_no_effect` |
| deepseek7b | L25:mlp_out | L25:mlp_out:C2139 | mlp_channel | 1 | 0.000 | -0.125 | 1.125 | 4.00 | 0.156 | 0.034 | `small_or_no_effect` |
| deepseek7b | L25:mlp_out | L25:mlp_out:topC64 | mlp_channelset | 8 | 0.250 | 0.391 | 1.023 | 3.38 | 0.535 | 0.314 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:attn_out | L23:attn_out:H19 | attn_head | 4 | 0.000 | 0.672 | 0.953 | 1.75 | 0.768 | 0.459 | `small_or_no_effect` |
| deepseek7b | L25:mlp_out | L25:mlp_out:C3346 | mlp_channel | 1 | 0.000 | -0.125 | 0.938 | 4.00 | 0.109 | 0.039 | `small_or_no_effect` |

## Strict Interpretation

- Attention decomposition is head-level o_proj projected delta evidence.
- MLP decomposition is residual output channel evidence, not true neuron evidence.
- A small subunit matching the whole component's route suppression is a localization hint, not proof of natural coding origin.
