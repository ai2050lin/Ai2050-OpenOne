# Phase 750 Natural Subunit Suppressor Necessity Test (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: natural forward erase of selected headsets/channelsets, no donor delta installed.

| model | context | site | subunit | kind | n | base top1 | after top1 | top1 loss | target drop | route release | coverage | margin drop | effect |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -1.031 | 2.281 | 3.25 | -0.537 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.500 | 1.844 | 3.75 | -0.084 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.781 | 1.281 | 2.50 | -0.672 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.344 | 1.156 | 2.75 | -0.091 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H26 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.125 | 1.125 | 3.00 | 0.094 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:C0 | mlp_channel | 2 | 1.000 | 1.000 | 0.000 | -0.500 | 0.875 | 4.00 | -0.281 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.312 | 0.781 | 3.25 | -0.139 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H26 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.250 | 0.750 | 3.00 | -0.062 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.281 | 0.719 | 2.50 | -0.094 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.344 | 0.719 | 1.75 | -0.203 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH2 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.281 | 0.688 | 1.50 | -0.383 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:C0 | mlp_channel | 2 | 1.000 | 1.000 | 0.000 | 0.000 | 0.625 | 3.00 | 0.156 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.250 | 0.500 | 2.00 | 0.250 | `natural_suppressor_necessity_candidate` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.062 | 0.469 | 2.25 | -0.011 | `small_or_no_effect` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | 0.219 | 0.312 | 1.25 | 0.145 | `small_or_no_effect` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | 0.312 | 0.312 | 1.50 | 0.125 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | 0.219 | 0.312 | 1.00 | 0.016 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H11 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -0.125 | 0.312 | 2.00 | -0.116 | `small_or_no_effect` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.062 | 0.781 | 3.50 | 0.017 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.172 | 0.703 | 3.75 | 0.022 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH4 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.141 | 0.688 | 3.00 | -0.010 | `small_or_no_effect` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H7 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.250 | 0.625 | 4.00 | 0.375 | `natural_suppressor_necessity_candidate` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH2 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.047 | 0.578 | 2.75 | 0.018 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -0.188 | 0.531 | 4.00 | -0.070 | `small_or_no_effect` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:C2319 | mlp_channel | 4 | 1.000 | 1.000 | 0.000 | -0.125 | 0.500 | 3.50 | -0.019 | `small_or_no_effect` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.141 | 0.469 | 3.50 | -0.054 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:C2319 | mlp_channel | 4 | 1.000 | 1.000 | 0.000 | -0.094 | 0.406 | 3.50 | 0.004 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:topC16 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.141 | 0.406 | 3.00 | -0.035 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH2 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.031 | 0.391 | 1.75 | 0.059 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H7 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.125 | 0.375 | 3.00 | 0.250 | `route_guard_necessity_candidate` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:topC16 | mlp_channelset | 8 | 1.000 | 1.000 | 0.000 | -0.047 | 0.375 | 3.75 | 0.027 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH1 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.078 | 0.266 | 2.00 | -0.093 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH1 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.016 | 0.250 | 2.25 | 0.049 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H13 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -0.062 | 0.250 | 2.00 | -0.008 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH2 | attn_headset | 8 | 1.000 | 1.000 | 0.000 | -0.047 | 0.250 | 1.75 | -0.165 | `small_or_no_effect` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H8 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.125 | 0.250 | 3.00 | -0.344 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.00 | 0.250 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:C2570 | mlp_channel | 4 | 0.750 | 1.000 | 0.000 | -0.719 | 0.906 | 2.75 | -0.532 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_recipient | L25:mlp_out | L25:mlp_out:C2570 | mlp_channel | 4 | 0.750 | 1.000 | 0.000 | -0.719 | 0.906 | 2.75 | -0.532 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H14 | attn_head | 1 | 0.000 | 0.000 | 0.000 | -0.125 | 0.750 | 4.00 | 0.062 | `small_or_no_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:topH4 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.352 | 0.672 | 2.00 | 0.381 | `target_support_necessity_candidate` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH4 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.695 | 0.602 | 1.62 | 0.614 | `target_support_necessity_candidate` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 1.000 | 1.000 | 0.000 | 0.125 | 0.562 | 4.00 | 0.266 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:topC64 | mlp_channelset | 8 | 0.750 | 0.750 | 0.000 | -0.188 | 0.562 | 2.25 | -0.091 | `small_or_no_effect` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:topH4 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.672 | 0.547 | 1.88 | 0.639 | `target_support_necessity_candidate` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH2 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.312 | 0.547 | 1.88 | 0.290 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H6 | attn_head | 1 | 1.000 | 1.000 | 0.000 | 0.125 | 0.500 | 4.00 | 0.138 | `small_or_no_effect` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:C756 | mlp_channel | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | 4.00 | 0.125 | `small_or_no_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:topH1 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.094 | 0.438 | 2.38 | 0.116 | `small_or_no_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H8 | attn_head | 1 | 0.000 | 0.000 | 0.000 | -0.375 | 0.438 | 3.00 | -0.312 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH2 | attn_headset | 8 | 0.750 | 0.625 | 0.125 | 0.445 | 0.430 | 1.62 | 0.430 | `natural_suppressor_necessity_candidate` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H11 | attn_head | 4 | 1.000 | 1.000 | 0.000 | 0.281 | 0.391 | 1.50 | 0.273 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:topC16 | mlp_channelset | 8 | 0.750 | 0.750 | 0.000 | -0.266 | 0.367 | 2.00 | -0.212 | `small_or_no_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH1 | attn_headset | 8 | 0.750 | 0.750 | 0.000 | 0.438 | 0.352 | 1.88 | 0.414 | `target_support_necessity_candidate` |

## Strict Interpretation

- This phase tests natural necessity-like behavior, not donor-recipient patch success.
- Attention erase zeroes selected o_proj input head slices at the final token.
- MLP erase zeroes selected residual output channels at the final token; this is still not neuron-level evidence.
- If erase releases competitor routes or drops the target, the subunit is a natural-route candidate, not yet a complete mechanism proof.
