# Phase 782 Multi-Component Surface Route Patch (smoke)

- Status: `complete`
- Test: patch/replace/zero multiple Phase 780 candidate components as a route.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: route-level answer-position patch, not full-token-sequence mechanism.

## Routes

| model | route | compare | size | components |
|---|---|---|---:|---|
| qwen3 | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `attn:L35` |
| qwen3 | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `attn:L35, mlp:L35` |
| qwen3 | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `attn:L34` |
| qwen3 | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `attn:L34, attn:L31` |
| glm4 | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `mlp:L38` |
| glm4 | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `mlp:L38, mlp:L39` |
| glm4 | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `mlp:L38` |
| glm4 | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `mlp:L38, attn:L33` |
| deepseek7b | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `mlp:L27` |
| deepseek7b | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `mlp:L27, mlp:L26` |
| deepseek7b | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `attn:L26` |
| deepseek7b | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `attn:L26, attn:L27` |

## Route Intervention Summary

| model | route | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k1` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k1` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 6.750 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 1.000 | -2.250 | 2.312 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 2 | 1.000 | 1.000 | 1.000 | 4.625 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:route_k1` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 4.812 | -1.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:route_k2` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k2` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 6.750 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 1.000 | 0.375 | 4.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | 2.062 | -4.688 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 4.781 | -1.969 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:route_k1` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k1` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 5.125 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -2.625 | 1.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| qwen3 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 2 | 1.000 | 1.000 | 1.000 | 2.938 | -2.188 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:route_k1` | `zero_donor_route` | 2 | 0.500 | 1.000 | 1.000 | 2.781 | -2.344 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k2` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k2` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 5.125 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -1.750 | 2.812 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| qwen3 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | 1.250 | -3.875 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | 0.500 | 1.000 | 1.000 | 1.219 | -3.906 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `lowercase_short_value:route_k1` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k1` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 0.500 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -0.531 | 0.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 2 | 0.000 | 1.000 | 1.000 | -0.375 | -0.875 | 0.000 | 1.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k1` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 0.719 | 0.219 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| glm4 | `lowercase_short_value:route_k2` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k2` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 0.500 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -0.156 | 1.312 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.000 | 1.000 | 0.500 | -0.781 | -1.281 | 0.000 | 1.000 | 0.000 | 0.500 | `{"case_variant_target_value": 2}` |
| glm4 | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 1.219 | 0.719 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| glm4 | `with_candidate_list:route_k1` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:route_k1` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 1.656 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -0.781 | 0.688 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | 0.875 | -0.781 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:route_k1` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 2.688 | 1.031 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| glm4 | `with_candidate_list:route_k2` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:route_k2` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 1.656 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.500 | 1.000 | 1.000 | 0.125 | 1.594 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | -0.031 | -1.688 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | 1.000 | 1.000 | 1.000 | 1.156 | -0.500 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| deepseek7b | `lowercase_short_value:route_k1` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `normal_donor` | 2 | 0.000 | 0.500 | 0.500 | -0.727 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 0.500 | -2.648 | 2.633 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 2 | 0.000 | 0.500 | 0.500 | -3.274 | -2.548 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `zero_donor_route` | 2 | 0.500 | 0.500 | 0.500 | 0.160 | 0.887 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `normal_donor` | 2 | 0.000 | 0.500 | 0.500 | -0.727 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 0.500 | -1.281 | 4.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.000 | 0.500 | 0.500 | -4.664 | -3.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | 0.500 | 0.500 | 0.500 | 1.049 | 1.775 | 0.500 | 0.000 | 0.000 | 0.000 | `{"target_value": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `normal_donor` | 2 | 0.000 | 1.000 | 0.500 | -1.938 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| deepseek7b | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 0.500 | -4.172 | 1.109 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 2 | 0.000 | 1.000 | 1.000 | -2.594 | -0.656 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| deepseek7b | `with_candidate_list:route_k1` | `zero_donor_route` | 2 | 0.000 | 1.000 | 1.000 | -2.656 | -0.719 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| deepseek7b | `with_candidate_list:route_k2` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `normal_donor` | 2 | 0.000 | 1.000 | 0.500 | -1.938 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| deepseek7b | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 2 | 0.000 | 0.500 | 0.000 | -3.422 | 1.859 | 0.000 | 0.000 | 0.000 | 0.500 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | 0.000 | 0.500 | 1.000 | -3.438 | -1.500 | 0.000 | 0.000 | 0.500 | 0.000 | `{"case_variant_target_value": 1, "punctuation": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | 0.000 | 0.000 | 0.000 | -4.258 | -2.320 | 0.000 | 0.000 | 1.000 | 0.500 | `{"punctuation": 1, "whitespace_or_empty": 1}` |

## Top Sufficiency Routes

| model | route | size | delta margin | strict gain | score |
|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k2` | 2 | 4.938 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:route_k2` | 2 | 2.812 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:route_k1` | 1 | 2.312 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:route_k1` | 1 | 1.938 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:route_k2` | 2 | 1.594 | 0.500 | 0.797 |
| glm4 | `lowercase_short_value:route_k2` | 2 | 1.312 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k1` | 1 | 0.938 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:route_k1` | 1 | 0.688 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k2` | 2 | 4.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k1` | 1 | 2.633 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k2` | 2 | 1.859 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k1` | 1 | 1.109 | 0.000 | 0.000 |

## Top Necessity Routes

| model | route | intervention | size | delta margin | strict loss | semantic loss | score |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -4.688 | 0.500 | 0.000 | 2.344 |
| qwen3 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -3.906 | 0.500 | 0.000 | 1.953 |
| qwen3 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -3.875 | 0.500 | 0.000 | 1.938 |
| qwen3 | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | -2.344 | 0.500 | 0.000 | 1.172 |
| qwen3 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.188 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.125 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | -1.969 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:route_k1` | `zero_donor_route` | 1 | -1.938 | 0.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -1.281 | 1.000 | 0.000 | 1.281 |
| glm4 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -0.875 | 1.000 | 0.000 | 0.875 |
| glm4 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -1.688 | 0.500 | 0.000 | 0.844 |
| glm4 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -0.781 | 0.500 | 0.000 | 0.391 |
| glm4 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -0.500 | 0.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k1` | `zero_donor_route` | 1 | 0.219 | 0.000 | 0.000 | -0.000 |
| glm4 | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | 0.719 | 0.000 | 0.000 | -0.000 |
| glm4 | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | 1.031 | 0.000 | 0.000 | -0.000 |
| deepseek7b | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -2.320 | 0.000 | 1.000 | 1.160 |
| deepseek7b | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -1.500 | 0.000 | 0.500 | 0.375 |
| deepseek7b | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -3.938 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.548 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | -0.719 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -0.656 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k1` | `zero_donor_route` | 1 | 0.887 | 0.000 | 0.000 | -0.000 |
| deepseek7b | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | 1.775 | 0.000 | 0.000 | -0.000 |

## Strict Interpretation

- Route-level patch tests whether candidate combinations beat single-component patch.
- It still patches only the answer position.
- If route patch remains weak, next tests must include token-position ranges and downstream readout closure.
