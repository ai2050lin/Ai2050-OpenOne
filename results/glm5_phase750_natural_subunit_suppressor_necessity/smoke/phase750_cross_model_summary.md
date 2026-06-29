# Phase 750 Natural Subunit Suppressor Necessity Test (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: natural forward erase of selected headsets/channelsets, no donor delta installed.

| model | context | site | subunit | kind | n | base top1 | after top1 | top1 loss | target drop | route release | coverage | margin drop | effect |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 1 | 1.000 | 1.000 | 0.000 | -0.250 | 0.625 | 3.00 | -0.094 | `small_or_no_effect` |
| qwen3 | natural_donor | L32:mlp_out | L32:mlp_out:C2464 | mlp_channel | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 0.250 | 2.00 | 0.062 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:topC16 | mlp_channelset | 1 | 1.000 | 1.000 | 0.000 | -0.250 | 0.250 | 2.00 | -0.125 | `small_or_no_effect` |
| qwen3 | natural_recipient | L32:mlp_out | L32:mlp_out:C2464 | mlp_channel | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH1 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | -0.125 | 0.562 | 4.00 | 0.016 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | attn_head | 1 | 1.000 | 1.000 | 0.000 | -0.125 | 0.562 | 4.00 | 0.016 | `small_or_no_effect` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:topH2 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | -0.250 | 0.312 | 2.00 | -0.094 | `small_or_no_effect` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH2 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | -0.062 | 0.250 | 2.00 | -0.281 | `erase_improves_or_inverse_effect` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:topH1 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | -0.125 | 0.188 | 1.00 | -0.062 | `small_or_no_effect` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H2 | attn_head | 1 | 1.000 | 1.000 | 0.000 | -0.125 | 0.188 | 1.00 | -0.062 | `small_or_no_effect` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH2 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | -0.250 | 1.625 | 3.00 | 0.292 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH1 | attn_headset | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.00 | 0.250 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.00 | 0.250 | `route_guard_necessity_candidate` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH1 | attn_headset | 1 | 1.000 | 1.000 | 0.000 | 0.125 | 0.438 | 3.00 | 0.271 | `route_guard_necessity_candidate` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 1.000 | 1.000 | 0.000 | 0.125 | 0.438 | 3.00 | 0.271 | `route_guard_necessity_candidate` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH2 | attn_headset | 1 | 0.000 | 0.000 | 0.000 | 0.250 | 0.312 | 2.00 | 0.297 | `natural_suppressor_necessity_candidate` |

## Strict Interpretation

- This phase tests natural necessity-like behavior, not donor-recipient patch success.
- Attention erase zeroes selected o_proj input head slices at the final token.
- MLP erase zeroes selected residual output channels at the final token; this is still not neuron-level evidence.
- If erase releases competitor routes or drops the target, the subunit is a natural-route candidate, not yet a complete mechanism proof.
