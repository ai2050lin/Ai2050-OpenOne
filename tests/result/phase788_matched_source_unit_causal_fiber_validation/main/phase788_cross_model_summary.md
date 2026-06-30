# Phase 788 Matched Source Unit Causal Fiber Validation (main)

- Status: `complete`
- Test: patch donor source units into baseline, with matched and random controls.
- Attention source units are o_proj input head slices.
- MLP source units are down_proj input activation channels.
- Matched controls are selected inside the same layer/component by source magnitude and activation-delta norm.

## Cross-Model Intervention Summary

| model | source | subspace | selection | intervention | cases | strict gain | strict loss | delta margin | source signed | top1 classes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.229 | -1.561 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.035 | -0.807 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 1.611 | -7.032 | `{"case_variant_target_value": 15, "lexical_capitalized": 1, "whitespace_or_empty": 2}` |
| qwen3 | `attention_head_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.250 | -1.561 | `{"contrast_value": 1, "target_value": 17}` |
| qwen3 | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.056 | -1.389 | -7.032 | `{"contrast_value": 1, "target_value": 17}` |
| qwen3 | `attention_head_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.083 | 1.635 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.021 | 0.725 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 2.104 | 9.383 | `{"case_variant_target_value": 14, "lexical_capitalized": 1, "target_value": 1, "whitespace_or_empty": 2}` |
| qwen3 | `attention_head_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.021 | 1.635 | `{"target_value": 18}` |
| qwen3 | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.056 | -1.913 | 9.383 | `{"target_value": 18}` |
| qwen3 | `mlp_channel_set` | `negative` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.448 | -2.083 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 20, "whitespace_or_empty": 3}` |
| qwen3 | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.005 | -0.007 | `{"case_variant_target_value": 20, "whitespace_or_empty": 4}` |
| qwen3 | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.042 | 0.000 | 0.401 | -8.319 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 20, "target_value": 1, "whitespace_or_empty": 2}` |
| qwen3 | `mlp_channel_set` | `negative` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.260 | -2.083 | `{"target_value": 24}` |
| qwen3 | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -0.185 | -8.319 | `{"contrast_value": 1, "lexical_capitalized": 1, "lexical_word": 1, "target_value": 21}` |
| qwen3 | `mlp_channel_set` | `positive` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.036 | 1.958 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 19, "whitespace_or_empty": 4}` |
| qwen3 | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.010 | 0.025 | `{"case_variant_target_value": 20, "whitespace_or_empty": 4}` |
| qwen3 | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.083 | 0.000 | 1.818 | 9.224 | `{"case_variant_target_value": 20, "target_value": 2, "whitespace_or_empty": 2}` |
| qwen3 | `mlp_channel_set` | `positive` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | 0.083 | 1.958 | `{"target_value": 24}` |
| qwen3 | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -1.562 | 9.224 | `{"contrast_value": 2, "lexical_word": 1, "target_value": 21}` |
| glm4 | `attention_head_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.021 | -0.065 | `{"case_variant_target_value": 12}` |
| glm4 | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.026 | -0.020 | `{"case_variant_target_value": 11, "lexical_capitalized": 1}` |
| glm4 | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.167 | 0.000 | 0.458 | -0.515 | `{"case_variant_target_value": 10, "target_value": 2}` |
| glm4 | `attention_head_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | 0.026 | -0.065 | `{"target_value": 12}` |
| glm4 | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.167 | -0.490 | -0.515 | `{"case_variant_target_value": 2, "target_value": 10}` |
| glm4 | `attention_head_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.021 | 0.027 | `{"case_variant_target_value": 12}` |
| glm4 | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.005 | 0.013 | `{"case_variant_target_value": 11, "lexical_capitalized": 1}` |
| glm4 | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.167 | 0.000 | 0.484 | 0.443 | `{"case_variant_target_value": 10, "target_value": 2}` |
| glm4 | `attention_head_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | 0.031 | 0.027 | `{"target_value": 12}` |
| glm4 | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.167 | -0.536 | 0.443 | `{"case_variant_target_value": 2, "target_value": 10}` |
| glm4 | `mlp_channel_set` | `negative` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.036 | -0.548 | `{"case_variant_target_value": 23, "lexical_capitalized": 1}` |
| glm4 | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.000 | -0.012 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.008 | -1.625 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `negative` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.042 | -0.031 | -0.548 | `{"case_variant_target_value": 7, "target_value": 17}` |
| glm4 | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | 0.008 | -1.625 | `{"case_variant_target_value": 5, "lexical_capitalized": 1, "target_value": 18}` |
| glm4 | `mlp_channel_set` | `positive` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.049 | 0.480 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.003 | 0.012 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.042 | 0.000 | 0.263 | 1.961 | `{"case_variant_target_value": 23, "target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.042 | -0.049 | 0.480 | `{"case_variant_target_value": 7, "target_value": 17}` |
| glm4 | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -0.276 | 1.961 | `{"case_variant_target_value": 10, "target_value": 14}` |
| deepseek7b | `attention_head_set` | `negative` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.118 | -0.795 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.107 | -0.427 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 1.467 | -8.763 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 13, "format_or_explanation_word": 3, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.152 | -0.795 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 11, "format_or_explanation_word": 1, "target_value": 5}` |
| deepseek7b | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.278 | -1.338 | -8.763 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 13, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `matched8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.064 | 1.244 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.018 | 0.574 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 1.628 | 10.566 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 12, "format_or_explanation_word": 3, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `matched8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.056 | -0.039 | 1.244 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 11, "format_or_explanation_word": 1, "lexical_word": 1, "target_value": 4}` |
| deepseek7b | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.222 | -1.546 | 10.566 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 14, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.634 | -6.052 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.023 | -0.075 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.264 | -19.515 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `negative` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.520 | -6.052 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 8, "format_or_explanation_word": 2, "target_value": 6}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.255 | -19.515 | `{"case_variant_contrast_value": 3, "case_variant_target_value": 6, "format_or_explanation_word": 2, "target_value": 7}` |
| deepseek7b | `mlp_channel_set` | `positive` | `matched32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.131 | 6.621 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.031 | 0.099 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 1.490 | 18.180 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `positive` | `matched32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.107 | 6.621 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 7, "contrast_value": 1, "format_or_explanation_word": 2, "target_value": 7}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.278 | -1.302 | 18.180 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 11, "format_or_explanation_word": 2, "lexical_word": 1, "target_value": 2}` |

## Top-Minus-Matched Specificity

| model | source | subspace | intervention | set | top gain | matched gain | top delta | matched delta | gap | matched distance |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attention_head_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.056 | 0.000 | 2.104 | -0.083 | 2.188 | 9.337 |
| qwen3 | `attention_head_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -1.913 | -0.021 | -1.892 | 9.337 |
| qwen3 | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 32 | 0.083 | 0.000 | 1.818 | 0.036 | 1.781 | 193.218 |
| qwen3 | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | -1.562 | 0.083 | -1.646 | 193.218 |
| qwen3 | `attention_head_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.000 | 0.000 | 1.611 | 0.229 | 1.382 | 8.011 |
| qwen3 | `attention_head_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -1.389 | -0.250 | -1.139 | 8.011 |
| qwen3 | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | -0.185 | -0.260 | 0.076 | 188.438 |
| qwen3 | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 32 | 0.042 | 0.000 | 0.401 | 0.448 | -0.047 | 188.438 |
| glm4 | `attention_head_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -0.536 | 0.031 | -0.568 | 14.392 |
| glm4 | `attention_head_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -0.490 | 0.026 | -0.516 | 13.184 |
| glm4 | `attention_head_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.167 | 0.000 | 0.484 | -0.021 | 0.505 | 14.392 |
| glm4 | `attention_head_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.167 | 0.000 | 0.458 | -0.021 | 0.479 | 13.184 |
| glm4 | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | -0.276 | -0.049 | -0.227 | 171.687 |
| glm4 | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 32 | 0.042 | 0.000 | 0.263 | 0.049 | 0.214 | 171.687 |
| glm4 | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | 0.008 | -0.031 | 0.039 | 180.065 |
| glm4 | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 32 | 0.000 | 0.000 | 0.008 | 0.036 | -0.029 | 180.065 |
| deepseek7b | `attention_head_set` | `positive` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -1.546 | -0.039 | -1.507 | 13.907 |
| deepseek7b | `attention_head_set` | `positive` | `patch_baseline_from_donor_source_units` | 8 | 0.056 | 0.000 | 1.628 | 0.064 | 1.564 | 13.907 |
| deepseek7b | `mlp_channel_set` | `positive` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | -1.302 | -0.107 | -1.194 | 369.098 |
| deepseek7b | `attention_head_set` | `negative` | `replace_donor_source_units_with_baseline` | 8 | 0.000 | 0.000 | -1.338 | -0.152 | -1.186 | 15.158 |
| deepseek7b | `attention_head_set` | `negative` | `patch_baseline_from_donor_source_units` | 8 | 0.056 | 0.000 | 1.467 | 0.118 | 1.350 | 15.158 |
| deepseek7b | `mlp_channel_set` | `positive` | `patch_baseline_from_donor_source_units` | 32 | 0.000 | 0.000 | 1.490 | 0.131 | 1.358 | 369.098 |
| deepseek7b | `mlp_channel_set` | `negative` | `patch_baseline_from_donor_source_units` | 32 | 0.000 | 0.000 | 0.264 | 0.634 | -0.370 | 379.135 |
| deepseek7b | `mlp_channel_set` | `negative` | `replace_donor_source_units_with_baseline` | 32 | 0.000 | 0.000 | -0.255 | -0.520 | 0.265 | 379.135 |

## Top Sufficiency Components

| model | route | component | source | subspace | selection | cases | strict gain | delta margin | source signed |
|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k6` | `attn:L31` | `attention_head_set` | `positive` | `top8` | 1 | 1.000 | 4.250 | 7.772 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.167 | 1.854 | 11.113 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.167 | 1.417 | 7.525 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.167 | 0.396 | -8.085 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `top8` | 6 | 0.000 | 2.375 | 10.169 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 2.271 | 12.301 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `attention_head_set` | `positive` | `top8` | 6 | 0.000 | 2.250 | 10.753 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `top8` | 6 | 0.000 | 2.229 | -6.781 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 1.729 | 5.958 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `attention_head_set` | `negative` | `top8` | 6 | 0.000 | 1.688 | -8.606 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `top8` | 4 | 0.000 | 1.250 | 7.901 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `attention_head_set` | `negative` | `top8` | 1 | 0.000 | 1.125 | -6.499 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `negative` | `top8` | 5 | 0.000 | 0.875 | -5.550 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `positive` | `top8` | 1 | 0.000 | 0.875 | 3.989 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.583 | -1.665 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.479 | -2.353 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.458 | -10.582 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.458 | -9.572 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `attention_head_set` | `negative` | `matched8` | 6 | 0.000 | 0.438 | -2.230 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.396 | -2.308 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `attention_head_set` | `negative` | `top8` | 6 | 0.333 | 0.740 | -0.596 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `attention_head_set` | `positive` | `top8` | 6 | 0.333 | 0.740 | 0.624 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L38` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.167 | 0.469 | 2.526 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 0.354 | 1.279 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `negative` | `top8` | 3 | 0.000 | 0.250 | -0.136 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `positive` | `top8` | 4 | 0.000 | 0.250 | 0.103 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `positive` | `top8` | 2 | 0.000 | 0.188 | 0.579 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.146 | -0.284 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 0.125 | 0.276 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `negative` | `top8` | 3 | 0.000 | 0.104 | -0.733 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 0.104 | 3.762 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L38` | `mlp_channel_set` | `positive` | `matched32` | 6 | 0.000 | 0.083 | 0.915 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.062 | -0.943 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L38` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.062 | -0.848 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `positive` | `matched32` | 6 | 0.000 | 0.052 | 0.596 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `positive` | `matched8` | 4 | 0.000 | 0.047 | 0.005 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `negative` | `matched8` | 3 | 0.000 | 0.042 | -0.008 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `positive` | `matched8` | 2 | 0.000 | 0.031 | 0.064 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `positive` | `random8` | 2 | 0.000 | 0.031 | 0.044 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.031 | -1.158 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `top8` | 6 | 0.167 | 3.016 | 0.518 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `top8` | 6 | 0.167 | 3.001 | -0.505 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 1.969 | 13.526 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 1.559 | 24.077 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `positive` | `top8` | 6 | 0.000 | 1.109 | 17.604 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 0.941 | 16.938 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `attention_head_set` | `positive` | `top8` | 5 | 0.000 | 0.850 | 13.982 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `negative` | `top8` | 5 | 0.000 | 0.844 | -13.781 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `attention_head_set` | `negative` | `top8` | 5 | 0.000 | 0.838 | -12.994 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.828 | -7.960 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.656 | -14.057 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.579 | -3.673 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `matched32` | 6 | 0.000 | 0.494 | -6.523 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `matched32` | 6 | 0.000 | 0.324 | 9.912 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `attention_head_set` | `positive` | `top8` | 1 | 0.000 | 0.312 | 11.553 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.224 | -26.833 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `random8` | 6 | 0.000 | 0.217 | -0.011 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `negative` | `matched8` | 5 | 0.000 | 0.173 | -1.002 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `matched8` | 6 | 0.000 | 0.141 | 0.022 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `attention_head_set` | `negative` | `matched8` | 2 | 0.000 | 0.125 | -0.373 |

## Interpretation Boundary

- This validates answer-site source-unit specificity against matched controls, not full Q/K/V path or cross-position semantic fibers.
- Matched controls are still approximate: they do not fully match token source, attention pattern, or upstream causal history.
- MLP channel sets are activation channels, closer to neuron-level than residual channels but still not biological neurons.
