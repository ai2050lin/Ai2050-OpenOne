# Phase 749 Suppressor Component Decomposition (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: subunit donor-recipient deltas measured against route-level max logits.

| model | component | subunit | kind | n | donor top1 | target boost | route suppression | coverage | margin gain | delta fraction | effect |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L32:mlp_out | L32:mlp_out | whole_component | 1 | 0.000 | 0.062 | 2.000 | 3.00 | 0.729 | 1.000 | `global_suppressor_margin_candidate` |
| qwen3 | L32:mlp_out | L32:mlp_out:randC1 | mlp_channelset_control | 1 | 0.000 | 0.000 | 0.250 | 2.00 | 0.083 | 0.003 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:C616 | mlp_channel | 1 | 0.000 | 0.062 | 0.000 | 0.00 | 0.021 | 0.039 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC1 | mlp_channelset | 1 | 0.000 | 0.062 | 0.000 | 0.00 | 0.021 | 0.039 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:randC4 | mlp_channelset_control | 1 | 0.000 | 0.000 | 0.000 | 0.00 | -0.042 | 0.026 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:C1754 | mlp_channel | 1 | 0.000 | 0.000 | 0.000 | 0.00 | -0.125 | 0.040 | `small_or_no_effect` |
| qwen3 | L32:mlp_out | L32:mlp_out:topC4 | mlp_channelset | 1 | 0.000 | 0.125 | 0.000 | 0.00 | 0.000 | 0.076 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out | whole_component | 1 | 1.000 | 0.500 | 0.625 | 3.00 | 0.562 | 1.000 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:topH2 | attn_headset | 1 | 1.000 | 0.125 | 0.500 | 3.00 | 0.219 | 0.316 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L34:attn_out | L34:attn_out:H13 | attn_head | 1 | 1.000 | 0.000 | 0.375 | 1.00 | 0.062 | 0.228 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:H9 | attn_head | 1 | 1.000 | 0.062 | 0.188 | 2.00 | 0.094 | 0.218 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:topH1 | attn_headset | 1 | 1.000 | 0.062 | 0.188 | 2.00 | 0.094 | 0.218 | `small_or_no_effect` |
| glm4 | L34:attn_out | L34:attn_out:H29 | attn_head | 1 | 1.000 | 0.000 | 0.000 | 0.00 | -0.016 | 0.035 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H0 | attn_head | 1 | 0.000 | -0.125 | 0.938 | 4.00 | 0.109 | 0.107 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out | whole_component | 1 | 0.000 | 0.188 | 0.312 | 1.00 | -0.016 | 1.000 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | L22:attn_out:H22 | attn_head | 1 | 0.000 | -0.312 | 0.250 | 1.00 | -0.281 | 0.319 | `harmful_or_competitor_support` |
| deepseek7b | L22:attn_out | L22:attn_out:H7 | attn_head | 1 | 0.000 | 0.250 | 0.188 | 1.00 | 0.203 | 0.291 | `route_specific_suppressor_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH1 | attn_headset | 1 | 0.000 | 0.250 | 0.188 | 1.00 | 0.203 | 0.291 | `route_specific_suppressor_candidate` |
| deepseek7b | L22:attn_out | L22:attn_out:topH2 | attn_headset | 1 | 0.000 | 0.250 | 0.125 | 1.00 | 0.219 | 0.312 | `route_specific_suppressor_candidate` |

## Strict Interpretation

- Attention decomposition is head-level o_proj projected delta evidence.
- MLP decomposition is residual output channel evidence, not true neuron evidence.
- A small subunit matching the whole component's route suppression is a localization hint, not proof of natural coding origin.
