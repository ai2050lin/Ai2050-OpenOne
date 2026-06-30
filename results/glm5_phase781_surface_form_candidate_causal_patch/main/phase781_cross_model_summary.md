# Phase 781 Surface-Form Candidate Causal Patch and Ablation (main)

- Status: `complete`
- Test: patch baseline component from donor repair prompt; replace/zero donor component for necessity.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-level causal evidence, not head/neuron-level proof.

## Candidates

| model | candidate | compare | kind | layer | phase780 score |
|---|---|---|---|---:|---:|
| qwen3 | `with_candidate_list:attn:L34` | `with_candidate_list` | `attn` | 34 | 7.250 |
| qwen3 | `lowercase_short_value:attn:L35` | `lowercase_short_value` | `attn` | 35 | 6.546 |
| qwen3 | `with_candidate_list:attn:L31` | `with_candidate_list` | `attn` | 31 | 6.199 |
| qwen3 | `lowercase_short_value:mlp:L35` | `lowercase_short_value` | `mlp` | 35 | 6.047 |
| glm4 | `with_candidate_list:mlp:L38` | `with_candidate_list` | `mlp` | 38 | 1.520 |
| glm4 | `with_candidate_list:attn:L33` | `with_candidate_list` | `attn` | 33 | 1.341 |
| glm4 | `lowercase_short_value:mlp:L38` | `lowercase_short_value` | `mlp` | 38 | 0.428 |
| glm4 | `with_candidate_list:attn:L29` | `with_candidate_list` | `attn` | 29 | 0.408 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `lowercase_short_value` | `mlp` | 27 | 17.339 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `lowercase_short_value` | `mlp` | 26 | 8.681 |
| deepseek7b | `with_candidate_list:attn:L26` | `with_candidate_list` | `attn` | 26 | 7.761 |
| deepseek7b | `with_candidate_list:attn:L27` | `with_candidate_list` | `attn` | 27 | 3.609 |

## Intervention Summary

| model | candidate | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:attn:L35` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:attn:L35` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.667 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:attn:L35` | `patch_baseline_from_donor` | 6 | 0.000 | 0.833 | 1.000 | -2.354 | 2.438 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:attn:L35` | `replace_donor_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.542 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:attn:L35` | `zero_donor_component` | 6 | 1.000 | 1.000 | 1.000 | 4.542 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:mlp:L35` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:mlp:L35` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.667 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:mlp:L35` | `patch_baseline_from_donor` | 6 | 0.000 | 0.667 | 1.000 | -2.125 | 2.667 | 0.000 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 4, "lexical_capitalized": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:mlp:L35` | `replace_donor_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.625 | -2.042 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:mlp:L35` | `zero_donor_component` | 6 | 1.000 | 1.000 | 1.000 | 5.042 | -1.625 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L31` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:attn:L31` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.302 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L31` | `patch_baseline_from_donor` | 6 | 0.000 | 0.833 | 1.000 | -2.667 | 2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:attn:L31` | `replace_donor_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.312 | -1.990 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L31` | `zero_donor_component` | 6 | 1.000 | 1.000 | 1.000 | 4.448 | -1.854 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L34` | `normal_baseline` | 6 | 0.000 | 0.833 | 1.000 | -4.792 | null | null | null | null | null | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:attn:L34` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 6.302 | null | null | null | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L34` | `patch_baseline_from_donor` | 6 | 0.000 | 1.000 | 1.000 | -2.812 | 1.979 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L34` | `replace_donor_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 4.042 | -2.260 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:attn:L34` | `zero_donor_component` | 6 | 0.833 | 1.000 | 1.000 | 4.021 | -2.281 | 0.000 | 0.167 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| glm4 | `lowercase_short_value:mlp:L38` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:mlp:L38` | `normal_donor` | 6 | 0.500 | 1.000 | 1.000 | -0.052 | null | null | null | null | null | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `lowercase_short_value:mlp:L38` | `patch_baseline_from_donor` | 6 | 0.000 | 1.000 | 1.000 | -0.677 | 0.552 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:mlp:L38` | `replace_donor_with_baseline` | 6 | 0.000 | 1.000 | 1.000 | -0.646 | -0.594 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:mlp:L38` | `zero_donor_component` | 6 | 0.500 | 0.833 | 1.000 | 0.500 | 0.552 | 0.000 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 1, "lexical_word": 1, "target_value": 4}` |
| glm4 | `with_candidate_list:attn:L29` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:attn:L29` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:attn:L29` | `patch_baseline_from_donor` | 6 | 0.000 | 1.000 | 1.000 | -1.083 | 0.146 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:attn:L29` | `replace_donor_with_baseline` | 6 | 1.000 | 1.000 | 1.000 | 1.708 | -0.062 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:attn:L29` | `zero_donor_component` | 6 | 1.000 | 1.000 | 1.000 | 1.688 | -0.083 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:attn:L33` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:attn:L33` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:attn:L33` | `patch_baseline_from_donor` | 6 | 0.333 | 1.000 | 1.000 | -0.521 | 0.708 | 0.333 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `with_candidate_list:attn:L33` | `replace_donor_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | 0.906 | -0.865 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:attn:L33` | `zero_donor_component` | 6 | 0.667 | 1.000 | 1.000 | 0.896 | -0.875 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:mlp:L38` | `normal_baseline` | 6 | 0.000 | 1.000 | 1.000 | -1.229 | null | null | null | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:mlp:L38` | `normal_donor` | 6 | 1.000 | 1.000 | 1.000 | 1.771 | null | null | null | null | null | `{"target_value": 6}` |
| glm4 | `with_candidate_list:mlp:L38` | `patch_baseline_from_donor` | 6 | 0.500 | 1.000 | 1.000 | -0.208 | 1.021 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `with_candidate_list:mlp:L38` | `replace_donor_with_baseline` | 6 | 0.667 | 1.000 | 1.000 | 0.625 | -1.146 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:mlp:L38` | `zero_donor_component` | 6 | 0.833 | 0.833 | 1.000 | 2.562 | 0.792 | 0.000 | 0.167 | 0.167 | 0.000 | `{"lexical_word": 1, "target_value": 5}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `normal_donor` | 6 | 0.500 | 0.667 | 0.667 | 0.216 | null | null | null | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `patch_baseline_from_donor` | 6 | 0.000 | 0.833 | 0.667 | -3.286 | 1.880 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `replace_donor_with_baseline` | 6 | 0.167 | 0.667 | 0.667 | -1.410 | -1.626 | 0.000 | 0.333 | 0.000 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "format_or_explanation_word": 1, "target_value": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `zero_donor_component` | 6 | 0.500 | 0.667 | 0.667 | 0.172 | -0.044 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "target_value": 3, "whitespace_or_empty": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `normal_donor` | 6 | 0.500 | 0.667 | 0.667 | 0.216 | null | null | null | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `patch_baseline_from_donor` | 6 | 0.000 | 0.833 | 0.833 | -2.477 | 2.690 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `replace_donor_with_baseline` | 6 | 0.000 | 0.500 | 0.667 | -2.414 | -2.630 | 0.000 | 0.500 | 0.167 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "whitespace_or_empty": 2}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `zero_donor_component` | 6 | 0.667 | 0.500 | 0.667 | 0.397 | 0.181 | 0.167 | 0.000 | 0.167 | 0.000 | `{"case_variant_target_value": 1, "contrast_value": 1, "format_or_explanation_word": 1, "lexical_word": 1, "target_value": 2}` |
| deepseek7b | `with_candidate_list:attn:L26` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:attn:L26` | `normal_donor` | 6 | 0.167 | 1.000 | 0.667 | -1.948 | null | null | null | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:attn:L26` | `patch_baseline_from_donor` | 6 | 0.000 | 0.667 | 0.667 | -4.203 | 0.964 | 0.000 | 0.000 | 0.167 | 0.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:attn:L26` | `replace_donor_with_baseline` | 6 | 0.000 | 1.000 | 0.833 | -2.646 | -0.698 | 0.000 | 0.167 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| deepseek7b | `with_candidate_list:attn:L26` | `zero_donor_component` | 6 | 0.167 | 1.000 | 0.833 | -2.542 | -0.594 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:attn:L27` | `normal_baseline` | 6 | 0.000 | 0.833 | 0.667 | -5.167 | null | null | null | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:attn:L27` | `normal_donor` | 6 | 0.167 | 1.000 | 0.667 | -1.948 | null | null | null | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `with_candidate_list:attn:L27` | `patch_baseline_from_donor` | 6 | 0.000 | 0.667 | 0.333 | -4.260 | 0.906 | 0.000 | 0.000 | 0.167 | 0.333 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:attn:L27` | `replace_donor_with_baseline` | 6 | 0.000 | 0.833 | 0.667 | -2.667 | -0.719 | 0.000 | 0.167 | 0.167 | 0.000 | `{"case_variant_target_value": 5, "punctuation": 1}` |
| deepseek7b | `with_candidate_list:attn:L27` | `zero_donor_component` | 6 | 0.000 | 0.500 | 0.500 | -3.583 | -1.635 | 0.000 | 0.167 | 0.500 | 0.167 | `{"case_variant_target_value": 3, "punctuation": 2, "whitespace_or_empty": 1}` |

## Top Sufficiency Candidates

| model | candidate | intervention | cases | delta margin | strict gain | score |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:mlp:L35` | `patch_baseline_from_donor` | 6 | 2.667 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:attn:L35` | `patch_baseline_from_donor` | 6 | 2.438 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:attn:L31` | `patch_baseline_from_donor` | 6 | 2.125 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:attn:L34` | `patch_baseline_from_donor` | 6 | 1.979 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:mlp:L38` | `patch_baseline_from_donor` | 6 | 1.021 | 0.500 | 0.510 |
| glm4 | `with_candidate_list:attn:L33` | `patch_baseline_from_donor` | 6 | 0.708 | 0.333 | 0.236 |
| glm4 | `lowercase_short_value:mlp:L38` | `patch_baseline_from_donor` | 6 | 0.552 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:attn:L29` | `patch_baseline_from_donor` | 6 | 0.146 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `patch_baseline_from_donor` | 6 | 2.690 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `patch_baseline_from_donor` | 6 | 1.880 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:attn:L26` | `patch_baseline_from_donor` | 6 | 0.964 | 0.000 | 0.000 |
| deepseek7b | `with_candidate_list:attn:L27` | `patch_baseline_from_donor` | 6 | 0.906 | 0.000 | 0.000 |

## Top Necessity Candidates

| model | candidate | intervention | cases | delta margin | strict loss | semantic loss | score |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:attn:L34` | `zero_donor_component` | 6 | -2.281 | 0.167 | 0.000 | 0.380 |
| qwen3 | `with_candidate_list:attn:L34` | `replace_donor_with_baseline` | 6 | -2.260 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:attn:L35` | `replace_donor_with_baseline` | 6 | -2.125 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:attn:L35` | `zero_donor_component` | 6 | -2.125 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:mlp:L35` | `replace_donor_with_baseline` | 6 | -2.042 | 0.000 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:attn:L31` | `replace_donor_with_baseline` | 6 | -1.990 | 0.000 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:attn:L31` | `zero_donor_component` | 6 | -1.854 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:mlp:L35` | `zero_donor_component` | 6 | -1.625 | 0.000 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:mlp:L38` | `replace_donor_with_baseline` | 6 | -1.146 | 0.333 | 0.000 | 0.382 |
| glm4 | `lowercase_short_value:mlp:L38` | `replace_donor_with_baseline` | 6 | -0.594 | 0.500 | 0.000 | 0.297 |
| glm4 | `with_candidate_list:attn:L33` | `zero_donor_component` | 6 | -0.875 | 0.333 | 0.000 | 0.292 |
| glm4 | `with_candidate_list:attn:L33` | `replace_donor_with_baseline` | 6 | -0.865 | 0.333 | 0.000 | 0.288 |
| glm4 | `with_candidate_list:attn:L29` | `zero_donor_component` | 6 | -0.083 | 0.000 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:attn:L29` | `replace_donor_with_baseline` | 6 | -0.062 | 0.000 | 0.000 | 0.000 |
| glm4 | `lowercase_short_value:mlp:L38` | `zero_donor_component` | 6 | 0.552 | 0.000 | 0.167 | -0.046 |
| glm4 | `with_candidate_list:mlp:L38` | `zero_donor_component` | 6 | 0.792 | 0.167 | 0.167 | -0.198 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `replace_donor_with_baseline` | 6 | -2.630 | 0.500 | 0.167 | 1.534 |
| deepseek7b | `with_candidate_list:attn:L27` | `zero_donor_component` | 6 | -1.635 | 0.167 | 0.500 | 0.681 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `replace_donor_with_baseline` | 6 | -1.626 | 0.333 | 0.000 | 0.542 |
| deepseek7b | `with_candidate_list:attn:L27` | `replace_donor_with_baseline` | 6 | -0.719 | 0.167 | 0.167 | 0.180 |
| deepseek7b | `with_candidate_list:attn:L26` | `replace_donor_with_baseline` | 6 | -0.698 | 0.167 | 0.000 | 0.116 |
| deepseek7b | `with_candidate_list:attn:L26` | `zero_donor_component` | 6 | -0.594 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `zero_donor_component` | 6 | -0.044 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `zero_donor_component` | 6 | 0.181 | 0.000 | 0.167 | -0.015 |

## Strict Interpretation

- `patch_baseline_from_donor` tests sufficiency at block granularity.
- `replace_donor_with_baseline` and `zero_donor_component` test necessity at block granularity.
- Weak single-block patch is still compatible with a distributed multi-component route.
