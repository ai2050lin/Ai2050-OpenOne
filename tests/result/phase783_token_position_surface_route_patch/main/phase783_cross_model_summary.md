# Phase 783 Token-Position Surface Route Patch (main)

- Status: `complete`
- Test: patch/replace Phase 782 route components over token-position scopes.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-level position fiber test, not head/channel/neuron-level proof.

## Routes And Scopes

| model | route | compare | size | scopes | components |
|---|---|---|---:|---|---|
| qwen3 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `attn:L35, mlp:L35, mlp:L34, mlp:L33, mlp:L32, mlp:L26` |
| qwen3 | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `attn:L34, attn:L31, mlp:L34, mlp:L35, attn:L35, attn:L32` |
| glm4 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `mlp:L38, mlp:L39, mlp:L34, mlp:L27, mlp:L36, mlp:L31` |
| glm4 | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `mlp:L38, attn:L33, attn:L29, attn:L35, attn:L32, mlp:L34` |
| deepseek7b | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `mlp:L27, mlp:L26, mlp:L24, attn:L19, mlp:L22, mlp:L21` |
| deepseek7b | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `answer_site, answer_prefix, format_cue, object_tokens, relation_tokens, semantic_pair, protocol_all, all_pre_answer_plus_answer` | `attn:L26, attn:L27, mlp:L27, attn:L25, attn:L23, attn:L22` |

## Top Sufficiency Fibers

| model | route | scope | size | strict gain | delta margin | gain vs answer | margin vs answer | score | alignment |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `with_candidate_list:route_k6` | `answer_site` | 6 | 1.000 | 9.667 | 0.000 | 0.000 | 9.667 | `{"same_count": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `answer_site` | 6 | 1.000 | 8.729 | 0.000 | 0.000 | 8.729 | `{"same_count": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.833 | 8.333 | -0.167 | -0.396 | 6.944 | `{"mean_broadcast": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `protocol_all` | 6 | 0.667 | 8.271 | -0.333 | -0.458 | 5.514 | `{"mean_broadcast": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.667 | 7.167 | -0.333 | -2.500 | 4.778 | `{"mean_broadcast": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `protocol_all` | 6 | 0.667 | 7.167 | -0.333 | -2.500 | 4.778 | `{"mean_broadcast": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `answer_prefix` | 6 | 0.000 | 0.125 | -1.000 | -9.542 | 0.000 | `{"same_count": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `answer_prefix` | 6 | 0.000 | 0.062 | -1.000 | -8.667 | 0.000 | `{"same_count": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `format_cue` | 6 | 0.000 | 0.021 | -1.000 | -9.646 | 0.000 | `{"same_count": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `object_tokens` | 6 | 0.000 | 0.000 | -1.000 | -9.667 | 0.000 | `{"same_count": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `object_tokens` | 6 | 0.000 | -0.021 | -1.000 | -8.750 | -0.000 | `{"same_count": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `relation_tokens` | 6 | 0.000 | -0.042 | -1.000 | -8.771 | -0.000 | `{"same_count": 6}` |
| glm4 | `with_candidate_list:route_k6` | `answer_site` | 6 | 0.833 | 2.177 | 0.000 | 0.000 | 1.814 | `{"same_count": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `protocol_all` | 6 | 0.333 | 3.328 | 0.000 | 2.443 | 1.109 | `{"mean_broadcast": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.333 | 3.299 | 0.000 | 2.414 | 1.100 | `{"mean_broadcast": 6}` |
| glm4 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.500 | 1.984 | -0.333 | -0.193 | 0.992 | `{"mean_broadcast": 6}` |
| glm4 | `with_candidate_list:route_k6` | `protocol_all` | 6 | 0.500 | 1.943 | -0.333 | -0.234 | 0.971 | `{"mean_broadcast": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `answer_site` | 6 | 0.333 | 0.885 | 0.000 | 0.000 | 0.295 | `{"same_count": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `answer_prefix` | 6 | 0.000 | 0.094 | -0.333 | -0.792 | 0.000 | `{"same_count": 6}` |
| glm4 | `with_candidate_list:route_k6` | `answer_prefix` | 6 | 0.000 | 0.062 | -0.833 | -2.115 | 0.000 | `{"same_count": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `relation_tokens` | 6 | 0.000 | 0.031 | -0.333 | -0.854 | 0.000 | `{"same_count": 6}` |
| glm4 | `with_candidate_list:route_k6` | `semantic_pair` | 6 | 0.000 | 0.031 | -0.833 | -2.146 | 0.000 | `{"same_count": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `format_cue` | 6 | 0.000 | 0.021 | -0.333 | -0.865 | 0.000 | `{"mean_broadcast": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `object_tokens` | 6 | 0.000 | 0.021 | -0.333 | -0.865 | 0.000 | `{"same_count": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.667 | 5.789 | 0.167 | 0.560 | 3.859 | `{"mean_broadcast": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `answer_site` | 6 | 0.500 | 5.229 | 0.000 | 0.000 | 2.615 | `{"same_count": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `protocol_all` | 6 | 0.333 | 5.846 | -0.167 | 0.617 | 1.949 | `{"mean_broadcast": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `answer_site` | 6 | 0.167 | 3.229 | 0.000 | 0.000 | 0.538 | `{"same_count": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `protocol_all` | 6 | 0.000 | 3.391 | -0.167 | 0.161 | 0.000 | `{"mean_broadcast": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | 6 | 0.000 | 3.146 | -0.167 | -0.083 | 0.000 | `{"mean_broadcast": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `answer_prefix` | 6 | 0.000 | 0.113 | -0.167 | -3.117 | 0.000 | `{"same_count": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `format_cue` | 6 | 0.000 | 0.112 | -0.500 | -5.117 | 0.000 | `{"mean_broadcast": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | 6 | 0.000 | 0.093 | -0.500 | -5.136 | 0.000 | `{"same_count": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `semantic_pair` | 6 | 0.000 | 0.089 | -0.167 | -3.141 | 0.000 | `{"same_count": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `relation_tokens` | 6 | 0.000 | 0.085 | -0.500 | -5.145 | 0.000 | `{"same_count": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `relation_tokens` | 6 | 0.000 | 0.073 | -0.167 | -3.156 | 0.000 | `{"same_count": 6}` |

## Top Answer-Site Advantages

| model | route | scope | strict gain vs answer | margin gain vs answer | strict gain | delta margin |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | -0.167 | -0.396 | 0.833 | 8.333 |
| qwen3 | `lowercase_short_value:route_k6` | `protocol_all` | -0.333 | -0.458 | 0.667 | 8.271 |
| qwen3 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | -0.333 | -2.500 | 0.667 | 7.167 |
| qwen3 | `with_candidate_list:route_k6` | `protocol_all` | -0.333 | -2.500 | 0.667 | 7.167 |
| qwen3 | `lowercase_short_value:route_k6` | `answer_prefix` | -1.000 | -8.667 | 0.000 | 0.062 |
| qwen3 | `lowercase_short_value:route_k6` | `object_tokens` | -1.000 | -8.750 | 0.000 | -0.021 |
| qwen3 | `lowercase_short_value:route_k6` | `relation_tokens` | -1.000 | -8.771 | 0.000 | -0.042 |
| qwen3 | `lowercase_short_value:route_k6` | `semantic_pair` | -1.000 | -8.792 | 0.000 | -0.062 |
| qwen3 | `lowercase_short_value:route_k6` | `format_cue` | -1.000 | -8.833 | 0.000 | -0.104 |
| qwen3 | `with_candidate_list:route_k6` | `answer_prefix` | -1.000 | -9.542 | 0.000 | 0.125 |
| qwen3 | `with_candidate_list:route_k6` | `format_cue` | -1.000 | -9.646 | 0.000 | 0.021 |
| qwen3 | `with_candidate_list:route_k6` | `object_tokens` | -1.000 | -9.667 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `protocol_all` | 0.000 | 2.443 | 0.333 | 3.328 |
| glm4 | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | 0.000 | 2.414 | 0.333 | 3.299 |
| glm4 | `lowercase_short_value:route_k6` | `answer_prefix` | -0.333 | -0.792 | 0.000 | 0.094 |
| glm4 | `lowercase_short_value:route_k6` | `relation_tokens` | -0.333 | -0.854 | 0.000 | 0.031 |
| glm4 | `lowercase_short_value:route_k6` | `format_cue` | -0.333 | -0.865 | 0.000 | 0.021 |
| glm4 | `lowercase_short_value:route_k6` | `object_tokens` | -0.333 | -0.865 | 0.000 | 0.021 |
| glm4 | `lowercase_short_value:route_k6` | `semantic_pair` | -0.333 | -0.896 | 0.000 | -0.010 |
| glm4 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | -0.333 | -0.193 | 0.500 | 1.984 |
| glm4 | `with_candidate_list:route_k6` | `protocol_all` | -0.333 | -0.234 | 0.500 | 1.943 |
| glm4 | `with_candidate_list:route_k6` | `answer_prefix` | -0.833 | -2.115 | 0.000 | 0.062 |
| glm4 | `with_candidate_list:route_k6` | `semantic_pair` | -0.833 | -2.146 | 0.000 | 0.031 |
| glm4 | `with_candidate_list:route_k6` | `relation_tokens` | -0.833 | -2.156 | 0.000 | 0.021 |
| deepseek7b | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | 0.167 | 0.560 | 0.667 | 5.789 |
| deepseek7b | `with_candidate_list:route_k6` | `protocol_all` | -0.167 | 0.161 | 0.000 | 3.391 |
| deepseek7b | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | -0.167 | -0.083 | 0.000 | 3.146 |
| deepseek7b | `with_candidate_list:route_k6` | `answer_prefix` | -0.167 | -3.117 | 0.000 | 0.113 |
| deepseek7b | `with_candidate_list:route_k6` | `semantic_pair` | -0.167 | -3.141 | 0.000 | 0.089 |
| deepseek7b | `with_candidate_list:route_k6` | `relation_tokens` | -0.167 | -3.156 | 0.000 | 0.073 |
| deepseek7b | `with_candidate_list:route_k6` | `object_tokens` | -0.167 | -3.201 | 0.000 | 0.029 |
| deepseek7b | `with_candidate_list:route_k6` | `format_cue` | -0.167 | -3.231 | 0.000 | -0.002 |
| deepseek7b | `lowercase_short_value:route_k6` | `protocol_all` | -0.167 | 0.617 | 0.333 | 5.846 |
| deepseek7b | `lowercase_short_value:route_k6` | `format_cue` | -0.500 | -5.117 | 0.000 | 0.112 |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | -0.500 | -5.136 | 0.000 | 0.093 |
| deepseek7b | `lowercase_short_value:route_k6` | `relation_tokens` | -0.500 | -5.145 | 0.000 | 0.085 |

## Top Necessity Fibers

| model | route | scope | intervention | size | strict loss | semantic loss | delta margin | score |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 1.000 | 0.500 | -9.333 | 11.667 |
| qwen3 | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.833 | 0.167 | -8.271 | 7.582 |
| qwen3 | `lowercase_short_value:route_k6` | `protocol_all` | `replace_donor_fiber_with_baseline` | 6 | 0.667 | 0.667 | -2.010 | 2.010 |
| qwen3 | `with_candidate_list:route_k6` | `protocol_all` | `replace_donor_fiber_with_baseline` | 6 | 0.333 | 0.333 | -3.594 | 1.797 |
| qwen3 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | `replace_donor_fiber_with_baseline` | 6 | 0.333 | 0.333 | -2.990 | 1.495 |
| qwen3 | `lowercase_short_value:route_k6` | `all_pre_answer_plus_answer` | `replace_donor_fiber_with_baseline` | 6 | 0.667 | 0.667 | -1.271 | 1.271 |
| qwen3 | `lowercase_short_value:route_k6` | `format_cue` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.562 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `format_cue` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `relation_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `object_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.021 | -0.000 |
| glm4 | `with_candidate_list:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.667 | 0.000 | -2.458 | 1.639 |
| glm4 | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.500 | 0.167 | -0.917 | 0.535 |
| glm4 | `with_candidate_list:route_k6` | `protocol_all` | `replace_donor_fiber_with_baseline` | 6 | 0.333 | 0.167 | -0.969 | 0.404 |
| glm4 | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | `replace_donor_fiber_with_baseline` | 6 | 0.333 | 0.167 | -0.786 | 0.328 |
| glm4 | `with_candidate_list:route_k6` | `answer_prefix` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.073 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `answer_prefix` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.052 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `relation_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.031 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.021 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `format_cue` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `object_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `relation_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `format_cue` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | 0.000 | -0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.500 | 0.000 | -5.004 | 2.502 |
| deepseek7b | `with_candidate_list:route_k6` | `answer_site` | `replace_donor_fiber_with_baseline` | 6 | 0.167 | 0.167 | -3.286 | 0.822 |
| deepseek7b | `with_candidate_list:route_k6` | `all_pre_answer_plus_answer` | `replace_donor_fiber_with_baseline` | 6 | 0.167 | 0.500 | -0.667 | 0.278 |
| deepseek7b | `with_candidate_list:route_k6` | `protocol_all` | `replace_donor_fiber_with_baseline` | 6 | 0.167 | 0.500 | -0.016 | 0.007 |
| deepseek7b | `with_candidate_list:route_k6` | `answer_prefix` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.135 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.062 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `relation_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.052 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `semantic_pair` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.042 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `object_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.042 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `relation_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.036 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `object_tokens` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.021 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `format_cue` | `replace_donor_fiber_with_baseline` | 6 | 0.000 | 0.000 | -0.010 | 0.000 |

## Strict Interpretation

- If non-answer scopes beat answer_site, the route should be treated as a position-component fiber.
- If answer_site remains best, Phase 782 likely captured a readout-side route.
- Mean-broadcast rows are useful boundary probes, but same-count rows are cleaner causal evidence.
