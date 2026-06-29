# Phase 750 Natural Subunit Suppressor Necessity Test (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: natural forward erase of selected headsets/channelsets, no donor delta installed.

| model | context | site | subunit | kind | n | base top1 | after top1 | top1 loss | target drop | route release | coverage | margin drop | effect |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H25 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -1.812 | 3.062 | 3.50 | -0.979 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH4 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -1.292 | 1.979 | 3.00 | -0.919 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.417 | 1.854 | 3.50 | 0.022 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH4 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -1.062 | 1.833 | 3.00 | -0.736 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:C4 | mlp_channel | 2 | 1.000 | 1.000 | 0.000 | -0.500 | 1.500 | 4.00 | -0.125 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H25 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -1.938 | 1.438 | 2.50 | -1.755 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH2 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.688 | 1.292 | 2.17 | -0.531 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H7 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.250 | 1.250 | 4.00 | 0.062 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.458 | 1.229 | 1.83 | -0.319 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.812 | 1.188 | 2.50 | -0.625 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H26 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.125 | 1.125 | 3.00 | 0.094 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:C4 | mlp_channel | 2 | 1.000 | 1.000 | 0.000 | -0.500 | 1.125 | 3.00 | -0.125 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.312 | 0.938 | 2.50 | -0.049 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.271 | 0.938 | 2.83 | -0.051 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:C0 | mlp_channel | 2 | 1.000 | 1.000 | 0.000 | -0.500 | 0.875 | 4.00 | -0.281 | `erase_improves_or_inverse_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:topC64 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.333 | 0.833 | 1.83 | -0.153 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.688 | 0.792 | 2.33 | -0.592 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H26 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.250 | 0.750 | 3.00 | -0.062 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.198 | 0.969 | 4.33 | 0.023 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH4 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.115 | 0.740 | 3.33 | 0.000 | `small_or_no_effect` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH4 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.115 | 0.677 | 3.00 | -0.048 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:topC16 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.115 | 0.573 | 4.00 | 0.013 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H23 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -0.031 | 0.562 | 4.50 | 0.081 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH2 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.031 | 0.542 | 2.50 | 0.068 | `small_or_no_effect` |
| glm4 | natural_donor | L36:mlp_out | L36:mlp_out:C2319 | mlp_channel | 8 | 1.000 | 1.000 | 0.000 | -0.109 | 0.531 | 4.25 | 0.005 | `small_or_no_effect` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH2 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.083 | 0.510 | 2.50 | -0.027 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | attn_head | 2 | 1.000 | 1.000 | 0.000 | -0.250 | 0.438 | 3.00 | -0.141 | `small_or_no_effect` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:C2319 | mlp_channel | 8 | 1.000 | 1.000 | 0.000 | -0.109 | 0.406 | 3.25 | -0.017 | `small_or_no_effect` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.312 | 0.375 | 3.00 | 0.250 | `natural_suppressor_necessity_candidate` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H9 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.000 | 0.375 | 4.00 | 0.094 | `small_or_no_effect` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:topC64 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.135 | 0.375 | 3.00 | -0.065 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H13 | attn_head | 2 | 1.000 | 1.000 | 0.000 | 0.000 | 0.375 | 4.00 | -0.175 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:topH1 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.042 | 0.354 | 3.17 | 0.034 | `small_or_no_effect` |
| glm4 | natural_recipient | L36:mlp_out | L36:mlp_out:topC16 | mlp_channelset | 12 | 1.000 | 1.000 | 0.000 | -0.062 | 0.333 | 3.33 | 0.004 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH2 | attn_headset | 12 | 1.000 | 1.000 | 0.000 | -0.083 | 0.260 | 1.67 | -0.183 | `small_or_no_effect` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H13 | attn_head | 4 | 1.000 | 1.000 | 0.000 | -0.062 | 0.250 | 2.00 | -0.008 | `small_or_no_effect` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:H0 | attn_head | 3 | 1.000 | 1.000 | 0.000 | -0.042 | 1.208 | 3.00 | 0.214 | `small_or_no_effect` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:C2574 | mlp_channel | 1 | 1.000 | 1.000 | 0.000 | -0.250 | 1.188 | 5.00 | -0.013 | `small_or_no_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H25 | attn_head | 1 | 1.000 | 0.000 | 1.000 | -0.125 | 1.062 | 4.00 | 0.087 | `small_or_no_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.00 | 0.250 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H2 | attn_head | 1 | 1.000 | 1.000 | 0.000 | -0.125 | 0.875 | 4.00 | 0.094 | `small_or_no_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | attn_head | 7 | 0.857 | 0.714 | 0.143 | 0.518 | 0.777 | 2.29 | 0.616 | `natural_suppressor_necessity_candidate` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H14 | attn_head | 1 | 0.000 | 0.000 | 0.000 | -0.125 | 0.750 | 4.00 | 0.062 | `small_or_no_effect` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:topH4 | attn_headset | 12 | 0.750 | 0.667 | 0.083 | 0.542 | 0.740 | 2.08 | 0.542 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:C2570 | mlp_channel | 6 | 0.833 | 1.000 | 0.000 | -0.438 | 0.719 | 2.17 | -0.504 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_recipient | L25:mlp_out | L25:mlp_out:C2570 | mlp_channel | 6 | 0.833 | 1.000 | 0.000 | -0.438 | 0.719 | 2.17 | -0.504 | `erase_improves_or_inverse_effect` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | attn_headset | 12 | 0.750 | 0.500 | 0.250 | 1.438 | 0.682 | 1.58 | 1.423 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH2 | attn_headset | 12 | 0.750 | 0.667 | 0.083 | 0.776 | 0.641 | 1.67 | 0.801 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH1 | attn_headset | 12 | 0.750 | 0.583 | 0.167 | 0.464 | 0.625 | 2.00 | 0.511 | `target_support_necessity_candidate` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:topH2 | attn_headset | 12 | 0.750 | 0.667 | 0.083 | 0.469 | 0.620 | 2.00 | 0.463 | `small_or_no_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:topH4 | attn_headset | 12 | 0.750 | 0.667 | 0.083 | 0.417 | 0.620 | 2.25 | 0.448 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:H11 | attn_head | 5 | 1.000 | 1.000 | 0.000 | 0.250 | 0.613 | 2.00 | 0.291 | `target_support_necessity_candidate` |
| deepseek7b | natural_donor | L25:mlp_out | L25:mlp_out:topC64 | mlp_channelset | 12 | 0.750 | 0.750 | 0.000 | -0.104 | 0.578 | 2.25 | -0.076 | `small_or_no_effect` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:topH1 | attn_headset | 12 | 0.750 | 0.667 | 0.083 | 0.167 | 0.568 | 2.50 | 0.221 | `small_or_no_effect` |

## Strict Interpretation

- This phase tests natural necessity-like behavior, not donor-recipient patch success.
- Attention erase zeroes selected o_proj input head slices at the final token.
- MLP erase zeroes selected residual output channels at the final token; this is still not neuron-level evidence.
- If erase releases competitor routes or drops the target, the subunit is a natural-route candidate, not yet a complete mechanism proof.
