# Phase 782 Multi-Component Surface Route Patch (main)

- Status: `complete`
- Test: patch/replace/zero multiple Phase 780 candidate components as a route.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: route-level answer-position patch, not full-token-sequence mechanism.

## Routes

| model | route | compare | size | components |
|---|---|---|---:|---|
| qwen3 | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `attn:L35` |
| qwen3 | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `attn:L35, mlp:L35` |
| qwen3 | `lowercase_short_value:route_k4` | `lowercase_short_value` | 4 | `attn:L35, mlp:L35, mlp:L34, mlp:L33` |
| qwen3 | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `attn:L34` |
| qwen3 | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `attn:L34, attn:L31` |
| qwen3 | `with_candidate_list:route_k4` | `with_candidate_list` | 4 | `attn:L34, attn:L31, mlp:L34, mlp:L35` |
| glm4 | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `mlp:L38` |
| glm4 | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `mlp:L38, mlp:L39` |
| glm4 | `lowercase_short_value:route_k4` | `lowercase_short_value` | 4 | `mlp:L38, mlp:L39, mlp:L34, mlp:L27` |
| glm4 | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `mlp:L38` |
| glm4 | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `mlp:L38, attn:L33` |
| glm4 | `with_candidate_list:route_k4` | `with_candidate_list` | 4 | `mlp:L38, attn:L33, attn:L29, attn:L35` |
| deepseek7b | `lowercase_short_value:route_k1` | `lowercase_short_value` | 1 | `mlp:L27` |
| deepseek7b | `lowercase_short_value:route_k2` | `lowercase_short_value` | 2 | `mlp:L27, mlp:L26` |
| deepseek7b | `lowercase_short_value:route_k4` | `lowercase_short_value` | 4 | `mlp:L27, mlp:L26, mlp:L24, attn:L19` |
| deepseek7b | `with_candidate_list:route_k1` | `with_candidate_list` | 1 | `attn:L26` |
| deepseek7b | `with_candidate_list:route_k2` | `with_candidate_list` | 2 | `attn:L26, attn:L27` |
| deepseek7b | `with_candidate_list:route_k4` | `with_candidate_list` | 4 | `attn:L26, attn:L27, mlp:L27, attn:L25` |

## Route Intervention Summary

| model | route | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k1` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k1` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.667 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.000 | 0.833 | 1.000 | -2.354 | 2.438 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.542 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k1` | `zero_donor_route` | 6 | 1.000 | 1.000 | 1.000 | 4.542 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k2` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k2` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.667 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.333 | 0.667 | 1.000 | 0.042 | 4.833 | 0.333 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 3, "target_value": 1, "whitespace_or_empty": 2}` |
| qwen3 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | 2.354 | -4.312 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| qwen3 | `lowercase_short_value:route_k2` | `zero_donor_route` | 6 | 1.000 | 1.000 | 1.000 | 4.177 | -2.490 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k4` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k4` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.667 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.833 | 1.000 | 1.000 | 2.875 | 7.667 | 0.833 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.333 | 0.833 | 1.000 | -0.646 | -7.312 | 0.000 | 0.667 | 0.167 | 0.000 | `{"case_variant_target_value": 4, "lexical_word": 1, "target_value": 1}` |
| qwen3 | `lowercase_short_value:route_k4` | `zero_donor_route` | 6 | 0.833 | 0.833 | 0.833 | 5.526 | -1.141 | 0.000 | 0.167 | 0.167 | 0.167 | `{"target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k1` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k1` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.302 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.000 | 1.000 | 1.000 | -2.812 | 1.979 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| qwen3 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.042 | -2.260 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k1` | `zero_donor_route` | 6 | 0.833 | 1.000 | 1.000 | 4.021 | -2.281 | 0.000 | 0.167 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| qwen3 | `with_candidate_list:route_k2` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k2` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.302 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.333 | 1.000 | 1.000 | -0.958 | 3.833 | 0.333 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 4, "target_value": 2}` |
| qwen3 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.500 | 0.833 | 0.833 | 1.521 | -4.781 | 0.000 | 0.500 | 0.167 | 0.167 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 2, "target_value": 3}` |
| qwen3 | `with_candidate_list:route_k2` | `zero_donor_route` | 6 | 0.500 | 0.833 | 1.000 | 2.083 | -4.219 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_target_value": 2, "lexical_capitalized": 1, "target_value": 3}` |
| qwen3 | `with_candidate_list:route_k4` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k4` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.302 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.833 | 1.000 | 1.000 | 3.042 | 7.833 | 0.833 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| qwen3 | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.167 | 0.500 | 0.667 | -0.979 | -7.281 | 0.000 | 0.833 | 0.500 | 0.333 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 2, "format_or_explanation_word": 1, "target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k4` | `zero_donor_route` | 6 | 1.000 | 1.000 | 1.000 | 4.844 | -1.458 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| glm4 | `lowercase_short_value:route_k1` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k1` | `normal_donor` | 6 | 0.500 | 1.000 | 1.000 | -0.052 | null | null | null | null | null | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.000 | 1.000 | 1.000 | -0.677 | 0.552 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 6 | 0.000 | 1.000 | 1.000 | -0.646 | -0.594 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k1` | `zero_donor_route` | 6 | 0.500 | 0.833 | 1.000 | 0.500 | 0.552 | 0.000 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 1, "lexical_word": 1, "target_value": 4}` |
| glm4 | `lowercase_short_value:route_k2` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k2` | `normal_donor` | 6 | 0.500 | 1.000 | 1.000 | -0.052 | null | null | null | null | null | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.000 | 1.000 | 1.000 | -0.510 | 0.719 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.833 | 0.833 | -0.792 | -0.740 | 0.000 | 0.500 | 0.167 | 0.167 | `{"case_variant_target_value": 5, "lexical_capitalized": 1}` |
| glm4 | `lowercase_short_value:route_k2` | `zero_donor_route` | 6 | 0.667 | 0.667 | 0.833 | 1.419 | 1.471 | 0.167 | 0.000 | 0.333 | 0.167 | `{"lexical_word": 2, "target_value": 4}` |
| glm4 | `lowercase_short_value:route_k4` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k4` | `normal_donor` | 6 | 0.500 | 1.000 | 1.000 | -0.052 | null | null | null | null | null | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `lowercase_short_value:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.167 | 1.000 | 1.000 | -0.385 | 0.844 | 0.167 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.833 | 1.000 | -0.948 | -0.896 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_target_value": 5, "lexical_capitalized": 1}` |
| glm4 | `lowercase_short_value:route_k4` | `zero_donor_route` | 6 | 0.333 | 0.333 | 0.667 | 1.089 | 1.141 | 0.000 | 0.167 | 0.667 | 0.333 | `{"format_or_explanation_word": 1, "lexical_capitalized": 2, "lexical_word": 1, "target_value": 2}` |
| glm4 | `with_candidate_list:route_k1` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k1` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.500 | 1.000 | 1.000 | -0.208 | 1.021 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | 0.625 | -1.146 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k1` | `zero_donor_route` | 6 | 0.833 | 0.833 | 1.000 | 2.562 | 0.792 | 0.000 | 0.167 | 0.167 | 0.000 | `{"lexical_word": 1, "target_value": 5}` |
| glm4 | `with_candidate_list:route_k2` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k2` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.667 | 1.000 | 1.000 | 0.448 | 1.677 | 0.667 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | -0.052 | -1.823 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k2` | `zero_donor_route` | 6 | 0.833 | 0.833 | 1.000 | 1.401 | -0.370 | 0.000 | 0.167 | 0.167 | 0.000 | `{"lexical_word": 1, "target_value": 5}` |
| glm4 | `with_candidate_list:route_k4` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k4` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.667 | 1.000 | 1.000 | 0.740 | 1.969 | 0.667 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | -0.375 | -2.146 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k4` | `zero_donor_route` | 6 | 0.500 | 0.833 | 1.000 | 0.964 | -0.807 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_target_value": 2, "lexical_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:route_k1` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `normal_donor` | 6 | 0.500 | 0.667 | 0.667 | 0.216 | null | null | null | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.000 | 0.833 | 0.833 | -2.477 | 2.690 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.500 | 0.667 | -2.414 | -2.630 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "whitespace_or_empty": 2}` |
| deepseek7b | `lowercase_short_value:route_k1` | `zero_donor_route` | 6 | 0.667 | 0.500 | 0.667 | 0.397 | 0.181 | 0.167 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 1, "contrast_value": 1, "format_or_explanation_word": 1, "lexical_word": 1, "target_value": 2}` |
| deepseek7b | `lowercase_short_value:route_k2` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `normal_donor` | 6 | 0.500 | 0.667 | 0.667 | 0.216 | null | null | null | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.167 | 0.833 | 0.833 | -1.000 | 4.167 | 0.167 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 4, "format_or_explanation_word": 1, "target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.667 | 0.667 | -3.893 | -4.109 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k2` | `zero_donor_route` | 6 | 0.333 | 0.333 | 0.667 | 1.100 | 0.883 | 0.167 | 0.333 | 0.333 | 0.000 | `{"lexical_capitalized": 1, "lexical_word": 1, "target_value": 2, "whitespace_or_empty": 2}` |
| deepseek7b | `lowercase_short_value:route_k4` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k4` | `normal_donor` | 6 | 0.500 | 0.667 | 0.667 | 0.216 | null | null | null | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.333 | 0.667 | 0.500 | 0.068 | 5.234 | 0.333 | 0.000 | 0.167 | 0.167 | `{"case_variant_target_value": 2, "contrast_value": 1, "format_or_explanation_word": 1, "target_value": 2}` |
| deepseek7b | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.500 | 0.667 | -4.777 | -4.993 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "format_or_explanation_word": 1, "lexical_capitalized": 1}` |
| deepseek7b | `lowercase_short_value:route_k4` | `zero_donor_route` | 6 | 0.000 | 0.500 | 0.667 | -0.405 | -0.621 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_target_value": 3, "lexical_capitalized": 1, "lexical_word": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `normal_donor` | 6 | 0.167 | 1.000 | 0.667 | -1.948 | null | null | null | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `patch_baseline_from_donor_route` | 6 | 0.000 | 0.667 | 0.667 | -4.203 | 0.964 | 0.000 | 0.000 | 0.167 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 6 | 0.000 | 1.000 | 0.833 | -2.646 | -0.698 | 0.000 | 0.167 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| deepseek7b | `with_candidate_list:route_k1` | `zero_donor_route` | 6 | 0.167 | 1.000 | 0.833 | -2.542 | -0.594 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `normal_donor` | 6 | 0.167 | 1.000 | 0.667 | -1.948 | null | null | null | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `patch_baseline_from_donor_route` | 6 | 0.000 | 0.667 | 0.333 | -3.370 | 1.797 | 0.000 | 0.000 | 0.167 | 0.333 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.833 | 1.000 | -3.604 | -1.656 | 0.000 | 0.167 | 0.167 | 0.000 | `{"case_variant_target_value": 5, "punctuation": 1}` |
| deepseek7b | `with_candidate_list:route_k2` | `zero_donor_route` | 6 | 0.000 | 0.000 | 0.333 | -4.319 | -2.371 | 0.000 | 0.167 | 1.000 | 0.333 | `{"lexical_capitalized": 1, "punctuation": 2, "whitespace_or_empty": 3}` |
| deepseek7b | `with_candidate_list:route_k4` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k4` | `normal_donor` | 6 | 0.167 | 1.000 | 0.667 | -1.948 | null | null | null | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k4` | `patch_baseline_from_donor_route` | 6 | 0.000 | 0.667 | 0.500 | -2.365 | 2.802 | 0.000 | 0.000 | 0.333 | 0.333 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 4}` |
| deepseek7b | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 6 | 0.000 | 0.667 | 0.833 | -4.667 | -2.719 | 0.000 | 0.167 | 0.333 | 0.000 | `{"case_variant_target_value": 4, "punctuation": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `with_candidate_list:route_k4` | `zero_donor_route` | 6 | 0.000 | 0.833 | 0.833 | -2.708 | -0.760 | 0.000 | 0.167 | 0.167 | 0.000 | `{"case_variant_target_value": 5, "punctuation": 1}` |

## Top Sufficiency Routes

| model | route | size | delta margin | strict gain | score |
|---|---|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k4` | 4 | 7.833 | 0.833 | 6.528 |
| qwen3 | `lowercase_short_value:route_k4` | 4 | 7.667 | 0.833 | 6.389 |
| qwen3 | `lowercase_short_value:route_k2` | 2 | 4.833 | 0.333 | 1.611 |
| qwen3 | `with_candidate_list:route_k2` | 2 | 3.833 | 0.333 | 1.278 |
| qwen3 | `lowercase_short_value:route_k1` | 1 | 2.438 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:route_k1` | 1 | 1.979 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:route_k4` | 4 | 1.969 | 0.667 | 1.312 |
| glm4 | `with_candidate_list:route_k2` | 2 | 1.677 | 0.667 | 1.118 |
| glm4 | `with_candidate_list:route_k1` | 1 | 1.021 | 0.500 | 0.510 |
| glm4 | `lowercase_short_value:route_k4` | 4 | 0.844 | 0.167 | 0.141 |
| glm4 | `lowercase_short_value:route_k2` | 2 | 0.719 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:route_k1` | 1 | 0.552 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k4` | 4 | 5.234 | 0.333 | 1.745 |
| deepseek7b | `lowercase_short_value:route_k2` | 2 | 4.167 | 0.167 | 0.694 |
| deepseek7b | `with_candidate_list:route_k4` | 4 | 2.802 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k1` | 1 | 2.690 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k2` | 2 | 1.797 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:route_k1` | 1 | 0.964 | 0.000 | 0.000 |

## Top Necessity Routes

| model | route | intervention | size | delta margin | strict loss | semantic loss | score |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 4 | -7.281 | 0.833 | 0.500 | 7.888 |
| qwen3 | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 4 | -7.312 | 0.667 | 0.167 | 5.484 |
| qwen3 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -4.781 | 0.500 | 0.167 | 2.789 |
| qwen3 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -4.219 | 0.500 | 0.167 | 2.461 |
| qwen3 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -4.312 | 0.333 | 0.000 | 1.438 |
| qwen3 | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | -2.281 | 0.167 | 0.000 | 0.380 |
| qwen3 | `lowercase_short_value:route_k4` | `zero_donor_route` | 4 | -1.141 | 0.167 | 0.167 | 0.285 |
| qwen3 | `lowercase_short_value:route_k2` | `zero_donor_route` | 2 | -2.490 | 0.000 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.260 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.125 | 0.000 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 4 | -2.146 | 0.333 | 0.000 | 0.715 |
| glm4 | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -1.823 | 0.333 | 0.000 | 0.608 |
| glm4 | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 4 | -0.896 | 0.500 | 0.167 | 0.523 |
| glm4 | `with_candidate_list:route_k4` | `zero_donor_route` | 4 | -0.807 | 0.500 | 0.167 | 0.471 |
| glm4 | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -0.740 | 0.500 | 0.167 | 0.431 |
| glm4 | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -1.146 | 0.333 | 0.000 | 0.382 |
| glm4 | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -0.594 | 0.500 | 0.000 | 0.297 |
| glm4 | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -0.370 | 0.167 | 0.167 | 0.092 |
| glm4 | `lowercase_short_value:route_k1` | `zero_donor_route` | 1 | 0.552 | 0.000 | 0.167 | -0.046 |
| glm4 | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | 0.792 | 0.167 | 0.167 | -0.198 |
| deepseek7b | `lowercase_short_value:route_k4` | `replace_donor_route_with_baseline` | 4 | -4.993 | 0.500 | 0.167 | 2.913 |
| deepseek7b | `lowercase_short_value:route_k2` | `replace_donor_route_with_baseline` | 2 | -4.109 | 0.500 | 0.000 | 2.055 |
| deepseek7b | `with_candidate_list:route_k2` | `zero_donor_route` | 2 | -2.371 | 0.167 | 1.000 | 1.581 |
| deepseek7b | `lowercase_short_value:route_k1` | `replace_donor_route_with_baseline` | 1 | -2.630 | 0.500 | 0.167 | 1.534 |
| deepseek7b | `with_candidate_list:route_k4` | `replace_donor_route_with_baseline` | 4 | -2.719 | 0.167 | 0.333 | 0.906 |
| deepseek7b | `with_candidate_list:route_k2` | `replace_donor_route_with_baseline` | 2 | -1.656 | 0.167 | 0.167 | 0.414 |
| deepseek7b | `lowercase_short_value:route_k4` | `zero_donor_route` | 4 | -0.621 | 0.500 | 0.167 | 0.362 |
| deepseek7b | `with_candidate_list:route_k4` | `zero_donor_route` | 4 | -0.760 | 0.167 | 0.167 | 0.190 |
| deepseek7b | `with_candidate_list:route_k1` | `replace_donor_route_with_baseline` | 1 | -0.698 | 0.167 | 0.000 | 0.116 |
| deepseek7b | `with_candidate_list:route_k1` | `zero_donor_route` | 1 | -0.594 | 0.000 | 0.000 | 0.000 |

## Strict Interpretation

- Route-level patch tests whether candidate combinations beat single-component patch.
- It still patches only the answer position.
- If route patch remains weak, next tests must include token-position ranges and downstream readout closure.
