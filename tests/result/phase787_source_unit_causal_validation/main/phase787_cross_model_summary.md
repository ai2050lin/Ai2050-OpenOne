# Phase 787 Source Unit Causal Validation (main)

- Status: `complete`
- Test: patch donor source units into baseline, with random controls.
- Attention source units are o_proj input head slices.
- MLP source units are down_proj input activation channels.

## Cross-Model Intervention Summary

| model | source | subspace | selection | intervention | cases | strict gain | strict loss | delta margin | source signed | top1 classes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.035 | -0.807 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 1.611 | -7.032 | `{"case_variant_target_value": 15, "lexical_capitalized": 1, "whitespace_or_empty": 2}` |
| qwen3 | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.056 | -1.389 | -7.032 | `{"contrast_value": 1, "target_value": 17}` |
| qwen3 | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.021 | 0.725 | `{"case_variant_target_value": 15, "whitespace_or_empty": 3}` |
| qwen3 | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 2.104 | 9.383 | `{"case_variant_target_value": 14, "lexical_capitalized": 1, "target_value": 1, "whitespace_or_empty": 2}` |
| qwen3 | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.056 | -1.913 | 9.383 | `{"target_value": 18}` |
| qwen3 | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.005 | -0.007 | `{"case_variant_target_value": 20, "whitespace_or_empty": 4}` |
| qwen3 | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.042 | 0.000 | 0.401 | -8.319 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 20, "target_value": 1, "whitespace_or_empty": 2}` |
| qwen3 | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -0.185 | -8.319 | `{"contrast_value": 1, "lexical_capitalized": 1, "lexical_word": 1, "target_value": 21}` |
| qwen3 | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.010 | 0.025 | `{"case_variant_target_value": 20, "whitespace_or_empty": 4}` |
| qwen3 | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.083 | 0.000 | 1.818 | 9.224 | `{"case_variant_target_value": 20, "target_value": 2, "whitespace_or_empty": 2}` |
| qwen3 | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -1.562 | 9.224 | `{"contrast_value": 2, "lexical_word": 1, "target_value": 21}` |
| glm4 | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.026 | -0.020 | `{"case_variant_target_value": 11, "lexical_capitalized": 1}` |
| glm4 | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.167 | 0.000 | 0.458 | -0.515 | `{"case_variant_target_value": 10, "target_value": 2}` |
| glm4 | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.167 | -0.490 | -0.515 | `{"case_variant_target_value": 2, "target_value": 10}` |
| glm4 | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.005 | 0.013 | `{"case_variant_target_value": 11, "lexical_capitalized": 1}` |
| glm4 | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.167 | 0.000 | 0.484 | 0.443 | `{"case_variant_target_value": 10, "target_value": 2}` |
| glm4 | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.167 | -0.536 | 0.443 | `{"case_variant_target_value": 2, "target_value": 10}` |
| glm4 | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.000 | -0.012 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.008 | -1.625 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | 0.008 | -1.625 | `{"case_variant_target_value": 5, "lexical_capitalized": 1, "target_value": 18}` |
| glm4 | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | -0.003 | 0.012 | `{"case_variant_target_value": 24}` |
| glm4 | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.042 | 0.000 | 0.263 | 1.961 | `{"case_variant_target_value": 23, "target_value": 1}` |
| glm4 | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.125 | -0.276 | 1.961 | `{"case_variant_target_value": 10, "target_value": 14}` |
| deepseek7b | `attention_head_set` | `negative` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.107 | -0.427 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `negative` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 1.467 | -8.763 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 13, "format_or_explanation_word": 3, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `negative` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.278 | -1.338 | -8.763 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 13, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `random8` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.018 | 0.574 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `attention_head_set` | `positive` | `top8` | `patch_baseline_from_donor_source_units` | 6 | 0.056 | 0.000 | 1.628 | 10.566 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 12, "format_or_explanation_word": 3, "target_value": 1}` |
| deepseek7b | `attention_head_set` | `positive` | `top8` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.222 | -1.546 | 10.566 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 14, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `mlp_channel_set` | `negative` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.023 | -0.075 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.264 | -19.515 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `negative` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.000 | -0.255 | -19.515 | `{"case_variant_contrast_value": 3, "case_variant_target_value": 6, "format_or_explanation_word": 2, "target_value": 7}` |
| deepseek7b | `mlp_channel_set` | `positive` | `random32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 0.031 | 0.099 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top32` | `patch_baseline_from_donor_source_units` | 6 | 0.000 | 0.000 | 1.490 | 18.180 | `{"case_variant_target_value": 15, "format_or_explanation_word": 3}` |
| deepseek7b | `mlp_channel_set` | `positive` | `top32` | `replace_donor_source_units_with_baseline` | 6 | 0.000 | 0.278 | -1.302 | 18.180 | `{"case_variant_contrast_value": 2, "case_variant_target_value": 11, "format_or_explanation_word": 2, "lexical_word": 1, "target_value": 2}` |

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
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.458 | -10.582 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.458 | -9.572 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.292 | -5.038 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `attention_head_set` | `negative` | `random8` | 6 | 0.000 | 0.146 | -1.474 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `attention_head_set` | `positive` | `random8` | 6 | 0.000 | 0.083 | 1.267 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `attention_head_set` | `positive` | `random8` | 6 | 0.000 | 0.062 | 0.563 |
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
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `attention_head_set` | `positive` | `random8` | 2 | 0.000 | 0.031 | 0.044 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.031 | -1.158 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L38` | `mlp_channel_set` | `negative` | `random32` | 6 | 0.000 | 0.021 | -0.032 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `mlp_channel_set` | `negative` | `random32` | 6 | 0.000 | 0.010 | -0.007 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `positive` | `random32` | 6 | 0.000 | 0.010 | -0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `negative` | `random8` | 3 | 0.000 | 0.000 | -0.004 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `attention_head_set` | `positive` | `random8` | 4 | 0.000 | 0.000 | -0.002 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `attention_head_set` | `positive` | `random8` | 6 | 0.000 | 0.000 | 0.013 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `mlp_channel_set` | `negative` | `random32` | 6 | 0.000 | 0.000 | -0.001 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `top8` | 6 | 0.167 | 3.016 | 0.518 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `top8` | 6 | 0.167 | 3.001 | -0.505 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 1.969 | 13.526 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 1.559 | 24.077 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `positive` | `top8` | 6 | 0.000 | 1.109 | 17.604 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `top32` | 6 | 0.000 | 0.941 | 16.938 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `attention_head_set` | `positive` | `top8` | 5 | 0.000 | 0.850 | 13.982 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `negative` | `top8` | 5 | 0.000 | 0.844 | -13.781 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `attention_head_set` | `negative` | `top8` | 5 | 0.000 | 0.838 | -12.994 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.656 | -14.057 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `attention_head_set` | `positive` | `top8` | 1 | 0.000 | 0.312 | 11.553 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `top32` | 6 | 0.000 | 0.224 | -26.833 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `negative` | `random8` | 6 | 0.000 | 0.217 | -0.011 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `attention_head_set` | `negative` | `random8` | 5 | 0.000 | 0.071 | -0.417 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `attention_head_set` | `negative` | `random8` | 2 | 0.000 | 0.062 | -0.156 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `attention_head_set` | `positive` | `random8` | 1 | 0.000 | 0.062 | 0.320 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `attention_head_set` | `positive` | `random8` | 5 | 0.000 | 0.055 | 1.422 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `negative` | `random32` | 6 | 0.000 | 0.055 | 0.004 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `mlp_channel_set` | `positive` | `random32` | 6 | 0.000 | 0.046 | 0.162 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `attention_head_set` | `positive` | `random8` | 6 | 0.000 | 0.044 | 0.014 |

## Interpretation Boundary

- This validates answer-site source-unit effects, not full Q/K/V path or cross-position semantic fibers.
- Random controls are matched by unit count but not by activation norm.
- MLP channel sets are activation channels, closer to neuron-level than residual channels but still not biological neurons.
