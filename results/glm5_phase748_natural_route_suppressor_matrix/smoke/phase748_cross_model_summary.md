# Phase 748 Natural Route Suppressor Matrix (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: whole-component donor-recipient deltas measured against route-level max logits.

| model | component | n | donor top1 | target boost | route suppression | route coverage | margin gain | selected prob gain | effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L28:attn_out | 1 | 0.000 | 0.938 | 1.125 | 3.00 | 1.312 | 0.000 | `global_suppressor_margin_candidate` |
| qwen3 | L30:mlp_out | 1 | 0.000 | 0.438 | 1.125 | 3.00 | 0.812 | 0.000 | `global_suppressor_margin_candidate` |
| qwen3 | L30:attn_out | 1 | 0.000 | 1.125 | 0.000 | 0.00 | 0.833 | 0.000 | `booster_candidate` |
| qwen3 | L28:mlp_out | 1 | 0.000 | 0.812 | 0.000 | 0.00 | 0.396 | 0.000 | `booster_candidate` |
| glm4 | L34:mlp_out | 1 | 1.000 | -0.688 | 1.250 | 3.00 | -0.406 | -0.091 | `global_suppressor_closure_candidate` |
| glm4 | L34:attn_out | 1 | 1.000 | 0.500 | 0.625 | 3.00 | 0.562 | 0.080 | `mixed_boost_global_suppressor_closure_candidate` |
| glm4 | L35:mlp_out | 1 | 1.000 | -0.438 | 0.562 | 1.00 | -0.312 | -0.066 | `harmful_or_competitor_support` |
| glm4 | L35:attn_out | 1 | 1.000 | 0.375 | 0.500 | 2.00 | 0.359 | 0.070 | `mixed_boost_global_suppressor_closure_candidate` |
| deepseek7b | L22:mlp_out | 1 | 0.000 | 0.250 | 1.000 | 1.00 | 0.438 | 0.002 | `route_specific_suppressor_candidate` |
| deepseek7b | L23:attn_out | 1 | 0.000 | 2.812 | 0.500 | 2.00 | 2.906 | 0.085 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:mlp_out | 1 | 0.000 | 0.188 | 0.500 | 1.00 | 0.094 | 0.000 | `small_or_no_effect` |
| deepseek7b | L22:attn_out | 1 | 0.000 | 0.188 | 0.312 | 1.00 | -0.016 | -0.000 | `small_or_no_effect` |

## Route-Specific Matrix Slices

| model | route | component | n | suppression | positive rate | margin gain | donor top1 | effect counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L30:mlp_out | 1 | 0.250 | 1.000 | 0.688 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | recipient_answer | L28:attn_out | 1 | 0.125 | 1.000 | 1.062 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | recipient_answer | L30:attn_out | 1 | -0.125 | 0.000 | 1.000 | 0.000 | `{"booster_candidate": 1}` |
| qwen3 | recipient_answer | L28:mlp_out | 1 | -0.250 | 0.000 | 0.562 | 0.000 | `{"booster_candidate": 1}` |
| qwen3 | punctuation_or_stop | L28:attn_out | 1 | 0.500 | 1.000 | 1.438 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | punctuation_or_stop | L30:mlp_out | 1 | 0.250 | 1.000 | 0.688 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | punctuation_or_stop | L30:attn_out | 1 | -0.250 | 0.000 | 0.875 | 0.000 | `{"booster_candidate": 1}` |
| qwen3 | punctuation_or_stop | L28:mlp_out | 1 | -0.500 | 0.000 | 0.312 | 0.000 | `{"booster_candidate": 1}` |
| qwen3 | format_or_schema | L30:mlp_out | 1 | 0.625 | 1.000 | 1.062 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | format_or_schema | L28:attn_out | 1 | 0.500 | 1.000 | 1.438 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| qwen3 | format_or_schema | L28:mlp_out | 1 | -0.500 | 0.000 | 0.312 | 0.000 | `{"booster_candidate": 1}` |
| qwen3 | format_or_schema | L30:attn_out | 1 | -0.500 | 0.000 | 0.625 | 0.000 | `{"booster_candidate": 1}` |
| glm4 | recipient_answer | L35:attn_out | 1 | 0.375 | 1.000 | 0.750 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | recipient_answer | L34:mlp_out | 1 | 0.125 | 1.000 | -0.562 | 1.000 | `{"global_suppressor_closure_candidate": 1}` |
| glm4 | recipient_answer | L34:attn_out | 1 | 0.062 | 1.000 | 0.562 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | recipient_answer | L35:mlp_out | 1 | 0.000 | 0.000 | -0.438 | 1.000 | `{"harmful_or_competitor_support": 1}` |
| glm4 | other_semantic_value | L34:attn_out | 1 | 0.062 | 1.000 | 0.562 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | other_semantic_value | L35:mlp_out | 1 | -0.062 | 0.000 | -0.500 | 1.000 | `{"harmful_or_competitor_support": 1}` |
| glm4 | other_semantic_value | L34:mlp_out | 1 | -0.125 | 0.000 | -0.812 | 1.000 | `{"global_suppressor_closure_candidate": 1}` |
| glm4 | other_semantic_value | L35:attn_out | 1 | -0.188 | 0.000 | 0.188 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | format_or_schema | L34:mlp_out | 1 | 0.562 | 1.000 | -0.125 | 1.000 | `{"global_suppressor_closure_candidate": 1}` |
| glm4 | format_or_schema | L35:mlp_out | 1 | 0.562 | 1.000 | 0.125 | 1.000 | `{"harmful_or_competitor_support": 1}` |
| glm4 | format_or_schema | L34:attn_out | 1 | -0.375 | 0.000 | 0.125 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | format_or_schema | L35:attn_out | 1 | -0.375 | 0.000 | 0.000 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L34:mlp_out | 1 | 0.562 | 1.000 | -0.125 | 1.000 | `{"global_suppressor_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L34:attn_out | 1 | 0.500 | 1.000 | 1.000 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L35:attn_out | 1 | 0.125 | 1.000 | 0.500 | 1.000 | `{"mixed_boost_global_suppressor_closure_candidate": 1}` |
| glm4 | echo_object_or_relation | L35:mlp_out | 1 | 0.000 | 0.000 | -0.438 | 1.000 | `{"harmful_or_competitor_support": 1}` |
| deepseek7b | punctuation_or_stop | L22:mlp_out | 1 | 1.000 | 1.000 | 1.250 | 0.000 | `{"route_specific_suppressor_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L23:mlp_out | 1 | 0.500 | 1.000 | 0.688 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L22:attn_out | 1 | 0.312 | 1.000 | 0.500 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L23:attn_out | 1 | 0.125 | 1.000 | 2.938 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| deepseek7b | other_vocab | L22:mlp_out | 1 | -0.125 | 0.000 | 0.125 | 0.000 | `{"route_specific_suppressor_candidate": 1}` |
| deepseek7b | other_vocab | L23:attn_out | 1 | -0.125 | 0.000 | 2.688 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| deepseek7b | other_vocab | L22:attn_out | 1 | -0.500 | 0.000 | -0.312 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | other_vocab | L23:mlp_out | 1 | -0.625 | 0.000 | -0.438 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | format_or_schema | L23:attn_out | 1 | 0.375 | 1.000 | 3.188 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |
| deepseek7b | format_or_schema | L22:attn_out | 1 | -0.125 | 0.000 | 0.062 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | format_or_schema | L22:mlp_out | 1 | 0.000 | 0.000 | 0.250 | 0.000 | `{"route_specific_suppressor_candidate": 1}` |
| deepseek7b | format_or_schema | L23:mlp_out | 1 | 0.000 | 0.000 | 0.188 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | echo_object_or_relation | L22:mlp_out | 1 | -0.125 | 0.000 | 0.125 | 0.000 | `{"route_specific_suppressor_candidate": 1}` |
| deepseek7b | echo_object_or_relation | L23:mlp_out | 1 | -0.250 | 0.000 | -0.062 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | echo_object_or_relation | L22:attn_out | 1 | -0.500 | 0.000 | -0.312 | 0.000 | `{"small_or_no_effect": 1}` |
| deepseek7b | echo_object_or_relation | L23:attn_out | 1 | 0.000 | 0.000 | 2.812 | 0.000 | `{"global_suppressor_margin_candidate": 1}` |

## Strict Interpretation

- `target boost` measures constructive force toward the donor answer.
- `route suppression` measures selective force against measured route maxima; positive values are suppressor evidence.
- `route coverage` estimates whether the component is route-specific or broad/global over measured top-k routes.
- This is still whole-component donor-recipient delta evidence; it does not yet prove a natural neuron-level suppressor or training-origin suppressor.

Atlas graph: nodes=18 edges=15
