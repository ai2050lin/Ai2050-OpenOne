# Phase 784 Answer-Site Route Channel Budget (main)

- Status: `complete`
- Test: answer-site route channel/subspace budget.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-output dimension budget, not final head/neuron atlas.

## Routes

| model | route | compare | size | components |
|---|---|---|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `attn:L35, mlp:L35, mlp:L34, mlp:L33, mlp:L32, mlp:L26` |
| qwen3 | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `attn:L34, attn:L31, mlp:L34, mlp:L35, attn:L35, attn:L32` |
| glm4 | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `mlp:L38, mlp:L39, mlp:L34, mlp:L27, mlp:L36, mlp:L31` |
| glm4 | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `mlp:L38, attn:L33, attn:L29, attn:L35, attn:L32, mlp:L34` |
| deepseek7b | `lowercase_short_value:route_k6` | `lowercase_short_value` | 6 | `mlp:L27, mlp:L26, mlp:L24, attn:L19, mlp:L22, mlp:L21` |
| deepseek7b | `with_candidate_list:route_k6` | `with_candidate_list` | 6 | `attn:L26, attn:L27, mlp:L27, attn:L25, attn:L23, attn:L22` |

## Budget Intervention Summary

| model | route | budget | intervention | cases | dims | frac | score cover | strict gain | delta margin | gain/full | margin/full | top1 classes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.048 | 0.000 | 2.042 | 0.000 | 0.234 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.004 | 0.102 | 0.000 | 3.771 | 0.000 | 0.432 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.017 | 0.234 | 0.667 | 7.375 | 0.667 | 0.845 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `lowercase_short_value:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.067 | 0.514 | 1.000 | 14.135 | 1.000 | 1.619 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 15360.000 | 1.000 | 1.000 | 1.000 | 8.729 | 1.000 | 1.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.048 | 0.000 | -1.917 | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.004 | 0.102 | 0.000 | -3.396 | null | null | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.017 | 0.234 | 0.000 | -6.771 | null | null | `{"case_variant_target_value": 3, "target_value": 3}` |
| qwen3 | `lowercase_short_value:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.067 | 0.514 | 0.000 | -13.583 | null | null | `{"case_variant_target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 15360.000 | 1.000 | 1.000 | 0.000 | -8.271 | null | null | `{"case_variant_target_value": 4, "lexical_word": 1, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.035 | 0.167 | 1.438 | 0.167 | 0.149 | `{"case_variant_target_value": 5, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.004 | 0.085 | 0.167 | 3.042 | 0.167 | 0.315 | `{"case_variant_target_value": 5, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.017 | 0.211 | 0.667 | 6.729 | 0.667 | 0.696 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `with_candidate_list:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.067 | 0.488 | 1.000 | 14.260 | 1.000 | 1.475 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 15360.000 | 1.000 | 1.000 | 1.000 | 9.667 | 1.000 | 1.000 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.035 | 0.000 | -1.385 | null | null | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.004 | 0.085 | 0.000 | -2.885 | null | null | `{"contrast_value": 1, "target_value": 5}` |
| qwen3 | `with_candidate_list:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.017 | 0.211 | 0.000 | -6.510 | null | null | `{"case_variant_target_value": 2, "contrast_value": 1, "lexical_capitalized": 2, "target_value": 1}` |
| qwen3 | `with_candidate_list:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.067 | 0.488 | 0.000 | -14.552 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 5}` |
| qwen3 | `with_candidate_list:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 15360.000 | 1.000 | 1.000 | 0.000 | -9.333 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "punctuation": 1, "whitespace_or_empty": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.075 | 0.000 | 0.688 | 0.000 | 0.776 | `{"case_variant_target_value": 5, "lexical_capitalized": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.003 | 0.137 | 0.667 | 1.333 | 2.000 | 1.506 | `{"case_variant_target_value": 1, "lexical_capitalized": 1, "target_value": 4}` |
| glm4 | `lowercase_short_value:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.010 | 0.270 | 0.667 | 2.635 | 2.000 | 2.976 | `{"lexical_capitalized": 1, "punctuation": 1, "target_value": 4}` |
| glm4 | `lowercase_short_value:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.042 | 0.534 | 0.833 | 5.130 | 2.500 | 5.794 | `{"lexical_capitalized": 1, "target_value": 5}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 24576.000 | 1.000 | 1.000 | 0.333 | 0.885 | 1.000 | 1.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.075 | 0.000 | -0.667 | null | null | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.003 | 0.137 | 0.000 | -1.344 | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.010 | 0.270 | 0.000 | -2.589 | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.042 | 0.534 | 0.000 | -4.932 | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 24576.000 | 1.000 | 1.000 | 0.000 | -0.917 | null | null | `{"case_variant_target_value": 5, "lexical_capitalized": 1}` |
| glm4 | `with_candidate_list:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.046 | 0.167 | 0.427 | 0.200 | 0.196 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `with_candidate_list:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.003 | 0.111 | 0.500 | 1.104 | 0.600 | 0.507 | `{"case_variant_target_value": 2, "lexical_capitalized": 1, "target_value": 3}` |
| glm4 | `with_candidate_list:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.010 | 0.261 | 1.000 | 2.771 | 1.200 | 1.273 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.042 | 0.523 | 1.000 | 5.833 | 1.200 | 2.679 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 24576.000 | 1.000 | 1.000 | 0.833 | 2.177 | 1.000 | 1.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| glm4 | `with_candidate_list:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.046 | 0.000 | -0.510 | null | null | `{"case_variant_target_value": 1, "target_value": 5}` |
| glm4 | `with_candidate_list:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.003 | 0.111 | 0.000 | -1.219 | null | null | `{"case_variant_target_value": 2, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.010 | 0.261 | 0.000 | -2.885 | null | null | `{"case_variant_target_value": 5, "contrast_value": 1}` |
| glm4 | `with_candidate_list:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.042 | 0.523 | 0.000 | -5.573 | null | null | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 24576.000 | 1.000 | 1.000 | 0.000 | -2.458 | null | null | `{"case_variant_target_value": 4, "target_value": 2}` |
| deepseek7b | `lowercase_short_value:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.056 | 0.000 | 2.688 | 0.000 | 0.514 | `{"case_variant_target_value": 5, "punctuation": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.003 | 0.111 | 0.167 | 4.401 | 0.333 | 0.842 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 3, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.012 | 0.230 | 0.667 | 7.516 | 1.333 | 1.437 | `{"punctuation": 2, "target_value": 4}` |
| deepseek7b | `lowercase_short_value:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.048 | 0.478 | 1.000 | 12.927 | 2.000 | 2.472 | `{"target_value": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 21504.000 | 1.000 | 1.000 | 0.500 | 5.229 | 1.000 | 1.000 | `{"case_variant_target_value": 2, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `lowercase_short_value:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.056 | 0.000 | -2.620 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.003 | 0.111 | 0.000 | -4.258 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.012 | 0.230 | 0.000 | -7.221 | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.048 | 0.478 | 0.000 | -12.424 | null | null | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 21504.000 | 1.000 | 1.000 | 0.000 | -5.004 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `16` | `patch_baseline_from_donor_channel_budget` | 6 | 16.000 | 0.001 | 0.032 | 0.000 | 1.229 | 0.000 | 0.381 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `64` | `patch_baseline_from_donor_channel_budget` | 6 | 64.000 | 0.003 | 0.076 | 0.000 | 2.224 | 0.000 | 0.689 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `256` | `patch_baseline_from_donor_channel_budget` | 6 | 256.000 | 0.012 | 0.182 | 0.167 | 4.419 | 1.000 | 1.369 | `{"case_variant_target_value": 3, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `1024` | `patch_baseline_from_donor_channel_budget` | 6 | 1024.000 | 0.048 | 0.423 | 0.833 | 8.729 | 5.000 | 2.703 | `{"format_or_explanation_word": 1, "target_value": 5}` |
| deepseek7b | `with_candidate_list:route_k6` | `all` | `patch_baseline_from_donor_channel_budget` | 6 | 21504.000 | 1.000 | 1.000 | 0.167 | 3.229 | 1.000 | 1.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `16` | `replace_donor_channel_budget_with_baseline` | 6 | 16.000 | 0.001 | 0.032 | 0.000 | -1.083 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 5}` |
| deepseek7b | `with_candidate_list:route_k6` | `64` | `replace_donor_channel_budget_with_baseline` | 6 | 64.000 | 0.003 | 0.076 | 0.000 | -2.010 | null | null | `{"case_variant_contrast_value": 1, "case_variant_target_value": 5}` |
| deepseek7b | `with_candidate_list:route_k6` | `256` | `replace_donor_channel_budget_with_baseline` | 6 | 256.000 | 0.012 | 0.182 | 0.000 | -4.099 | null | null | `{"case_variant_target_value": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `1024` | `replace_donor_channel_budget_with_baseline` | 6 | 1024.000 | 0.048 | 0.423 | 0.000 | -8.391 | null | null | `{"case_variant_target_value": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `all` | `replace_donor_channel_budget_with_baseline` | 6 | 21504.000 | 1.000 | 1.000 | 0.000 | -3.286 | null | null | `{"case_variant_target_value": 5, "punctuation": 1}` |

## Low-Budget Successes

| model | route | budget | dims | frac | score cover | strict gain | delta margin |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k6` | `16` | 16.000 | 0.001 | 0.035 | 0.167 | 1.438 |
| qwen3 | `with_candidate_list:route_k6` | `64` | 64.000 | 0.004 | 0.085 | 0.167 | 3.042 |
| qwen3 | `lowercase_short_value:route_k6` | `256` | 256.000 | 0.017 | 0.234 | 0.667 | 7.375 |
| qwen3 | `with_candidate_list:route_k6` | `256` | 256.000 | 0.017 | 0.211 | 0.667 | 6.729 |
| qwen3 | `with_candidate_list:route_k6` | `1024` | 1024.000 | 0.067 | 0.488 | 1.000 | 14.260 |
| qwen3 | `lowercase_short_value:route_k6` | `1024` | 1024.000 | 0.067 | 0.514 | 1.000 | 14.135 |
| glm4 | `with_candidate_list:route_k6` | `16` | 16.000 | 0.001 | 0.046 | 0.167 | 0.427 |
| glm4 | `lowercase_short_value:route_k6` | `64` | 64.000 | 0.003 | 0.137 | 0.667 | 1.333 |
| glm4 | `with_candidate_list:route_k6` | `64` | 64.000 | 0.003 | 0.111 | 0.500 | 1.104 |
| glm4 | `with_candidate_list:route_k6` | `256` | 256.000 | 0.010 | 0.261 | 1.000 | 2.771 |
| glm4 | `lowercase_short_value:route_k6` | `256` | 256.000 | 0.010 | 0.270 | 0.667 | 2.635 |
| glm4 | `with_candidate_list:route_k6` | `1024` | 1024.000 | 0.042 | 0.523 | 1.000 | 5.833 |
| glm4 | `lowercase_short_value:route_k6` | `1024` | 1024.000 | 0.042 | 0.534 | 0.833 | 5.130 |
| deepseek7b | `lowercase_short_value:route_k6` | `64` | 64.000 | 0.003 | 0.111 | 0.167 | 4.401 |
| deepseek7b | `lowercase_short_value:route_k6` | `256` | 256.000 | 0.012 | 0.230 | 0.667 | 7.516 |
| deepseek7b | `with_candidate_list:route_k6` | `256` | 256.000 | 0.012 | 0.182 | 0.167 | 4.419 |
| deepseek7b | `lowercase_short_value:route_k6` | `1024` | 1024.000 | 0.048 | 0.478 | 1.000 | 12.927 |
| deepseek7b | `with_candidate_list:route_k6` | `1024` | 1024.000 | 0.048 | 0.423 | 0.833 | 8.729 |

## Strict Interpretation

- `all` should approximate the Phase 782 full answer-site route patch.
- Small-budget success means the readout-side route has sparse/channel-like support under the current ranking rule.
- Small-budget failure means the route is distributed or the ranking rule is incomplete.
- This does not yet identify attention heads or biological neurons.
