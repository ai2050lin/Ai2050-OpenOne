# Phase 785 Positive-Negative Subspace Split (main)

- Status: `complete`
- Test: split answer-site route dimensions by signed readout contribution.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-output channel evidence, not final head/neuron atlas.

## Top Sufficiency Subspaces

| model | route | mode | budget | cases | dims | strict gain | delta margin | pos cover | neg cover | abs cover | top1 classes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `with_candidate_list:route_k6` | `all_positive` | `all_positive` | 6 | 8163.000 | 1.000 | 25.260 | 1.000 | 0.000 | 0.579 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `all_positive` | `all_positive` | 6 | 8028.167 | 1.000 | 22.234 | 1.000 | 0.000 | 0.574 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 1.000 | 14.260 | 0.488 | 0.000 | 0.283 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 1.000 | 14.135 | 0.514 | 0.000 | 0.295 | `{"target_value": 6}` |
| qwen3 | `with_candidate_list:route_k6` | `all` | `all` | 6 | 15360.000 | 1.000 | 9.667 | 1.000 | 1.000 | 1.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `all` | `all` | 6 | 15360.000 | 1.000 | 8.729 | 1.000 | 1.000 | 1.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 0.667 | 7.375 | 0.234 | 0.000 | 0.135 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `with_candidate_list:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 0.667 | 6.729 | 0.211 | 0.000 | 0.122 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `lowercase_short_value:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.667 | 6.146 | 0.394 | 0.305 | 0.356 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `with_candidate_list:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.667 | 5.312 | 0.372 | 0.301 | 0.342 | `{"case_variant_target_value": 2, "target_value": 4}` |
| qwen3 | `lowercase_short_value:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.333 | 4.250 | 0.184 | 0.111 | 0.153 | `{"case_variant_target_value": 4, "target_value": 2}` |
| qwen3 | `with_candidate_list:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.333 | 2.854 | 0.158 | 0.119 | 0.142 | `{"case_variant_target_value": 4, "target_value": 2}` |
| qwen3 | `lowercase_short_value:route_k6` | `random` | `random_1024` | 6 | 1024.000 | 0.000 | 0.688 | 0.068 | 0.070 | 0.069 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:route_k6` | `random` | `random_1024` | 6 | 1024.000 | 0.000 | 0.500 | 0.062 | 0.066 | 0.064 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `random` | `random_256` | 6 | 256.000 | 0.000 | 0.146 | 0.015 | 0.015 | 0.015 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:route_k6` | `neutral` | `neutral_1024` | 6 | 1024.000 | 0.000 | 0.083 | 0.001 | 0.001 | 0.001 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| glm4 | `with_candidate_list:route_k6` | `all_positive` | `all_positive` | 6 | 12837.333 | 1.000 | 10.414 | 1.000 | 0.000 | 0.547 | `{"target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `all_positive` | `all_positive` | 6 | 12335.667 | 1.000 | 8.276 | 1.000 | 0.000 | 0.515 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 1.000 | 5.833 | 0.523 | 0.000 | 0.286 | `{"target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 0.833 | 5.130 | 0.534 | 0.000 | 0.275 | `{"lexical_capitalized": 1, "target_value": 5}` |
| glm4 | `with_candidate_list:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 1.000 | 2.771 | 0.261 | 0.000 | 0.143 | `{"target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `all` | `all` | 6 | 24576.000 | 0.833 | 2.177 | 1.000 | 1.000 | 1.000 | `{"case_variant_target_value": 1, "target_value": 5}` |
| glm4 | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 0.667 | 2.635 | 0.270 | 0.000 | 0.139 | `{"lexical_capitalized": 1, "punctuation": 1, "target_value": 4}` |
| glm4 | `with_candidate_list:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.500 | 1.000 | 0.398 | 0.390 | 0.395 | `{"case_variant_target_value": 3, "target_value": 3}` |
| glm4 | `lowercase_short_value:route_k6` | `all` | `all` | 6 | 24576.000 | 0.333 | 0.885 | 1.000 | 1.000 | 1.000 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `with_candidate_list:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.167 | 0.615 | 0.187 | 0.170 | 0.179 | `{"case_variant_target_value": 5, "target_value": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.000 | 0.604 | 0.388 | 0.367 | 0.378 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.000 | 0.573 | 0.194 | 0.163 | 0.179 | `{"case_variant_target_value": 5, "lexical_capitalized": 1}` |
| glm4 | `with_candidate_list:route_k6` | `random` | `random_1024` | 6 | 1024.000 | 0.000 | 0.146 | 0.043 | 0.041 | 0.042 | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `random` | `random_256` | 6 | 256.000 | 0.000 | 0.062 | 0.011 | 0.010 | 0.011 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `random` | `random_256` | 6 | 256.000 | 0.000 | 0.021 | 0.010 | 0.011 | 0.010 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `neutral` | `neutral_256` | 6 | 256.000 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all_positive` | `all_positive` | 6 | 10866.833 | 1.000 | 18.074 | 1.000 | 0.000 | 0.536 | `{"target_value": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `all_positive` | `all_positive` | 6 | 10925.333 | 1.000 | 17.703 | 1.000 | 0.000 | 0.528 | `{"target_value": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 1.000 | 12.927 | 0.478 | 0.000 | 0.256 | `{"target_value": 6}` |
| deepseek7b | `with_candidate_list:route_k6` | `positive` | `positive_1024` | 6 | 1024.000 | 0.833 | 8.729 | 0.423 | 0.000 | 0.223 | `{"format_or_explanation_word": 1, "target_value": 5}` |
| deepseek7b | `lowercase_short_value:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 0.667 | 7.516 | 0.230 | 0.000 | 0.123 | `{"punctuation": 2, "target_value": 4}` |
| deepseek7b | `lowercase_short_value:route_k6` | `all` | `all` | 6 | 21504.000 | 0.500 | 5.229 | 1.000 | 1.000 | 1.000 | `{"case_variant_target_value": 2, "format_or_explanation_word": 1, "target_value": 3}` |
| deepseek7b | `with_candidate_list:route_k6` | `positive` | `positive_256` | 6 | 256.000 | 0.167 | 4.419 | 0.182 | 0.000 | 0.096 | `{"case_variant_target_value": 3, "format_or_explanation_word": 1, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `all` | `all` | 6 | 21504.000 | 0.167 | 3.229 | 1.000 | 1.000 | 1.000 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.000 | 3.799 | 0.352 | 0.290 | 0.323 | `{"case_variant_target_value": 4, "punctuation": 1, "target_value": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.000 | 3.115 | 0.174 | 0.112 | 0.145 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 4, "punctuation": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `abs` | `abs_1024` | 6 | 1024.000 | 0.000 | 1.807 | 0.296 | 0.277 | 0.288 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `abs` | `abs_256` | 6 | 256.000 | 0.000 | 1.231 | 0.124 | 0.118 | 0.122 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `random` | `random_1024` | 6 | 1024.000 | 0.000 | 0.608 | 0.047 | 0.049 | 0.048 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `random` | `random_1024` | 6 | 1024.000 | 0.000 | 0.251 | 0.049 | 0.049 | 0.049 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `random` | `random_256` | 6 | 256.000 | 0.000 | 0.136 | 0.011 | 0.012 | 0.012 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `random` | `random_256` | 6 | 256.000 | 0.000 | 0.087 | 0.013 | 0.011 | 0.012 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |

## Negative Patch Effects

| model | route | budget | dims | strict gain | delta margin | neg cover | top1 classes |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | `with_candidate_list:route_k6` | `negative_1024` | 1024.000 | 0.000 | -11.266 | 0.529 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 5}` |
| qwen3 | `lowercase_short_value:route_k6` | `negative_1024` | 1024.000 | 0.000 | -9.521 | 0.527 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 5}` |
| qwen3 | `with_candidate_list:route_k6` | `negative_256` | 256.000 | 0.000 | -4.792 | 0.234 | `{"case_variant_target_value": 6}` |
| qwen3 | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -4.052 | 0.231 | `{"case_variant_target_value": 6}` |
| glm4 | `lowercase_short_value:route_k6` | `negative_1024` | 1024.000 | 0.000 | -4.682 | 0.526 | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `negative_1024` | 1024.000 | 0.000 | -3.794 | 0.557 | `{"case_variant_target_value": 4, "format_or_explanation_word": 1, "lexical_capitalized": 1}` |
| glm4 | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -2.193 | 0.250 | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list:route_k6` | `negative_256` | 256.000 | 0.000 | -2.047 | 0.284 | `{"case_variant_target_value": 6}` |
| deepseek7b | `lowercase_short_value:route_k6` | `negative_1024` | 1024.000 | 0.000 | -9.682 | 0.459 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `negative_1024` | 1024.000 | 0.000 | -6.995 | 0.439 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:route_k6` | `negative_256` | 256.000 | 0.000 | -5.120 | 0.201 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list:route_k6` | `negative_256` | 256.000 | 0.000 | -3.112 | 0.193 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |

## Random Controls

| model | route | budget | dims | strict gain | delta margin | abs cover |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.688 | 0.069 |
| qwen3 | `with_candidate_list:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.500 | 0.064 |
| qwen3 | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.146 | 0.015 |
| qwen3 | `with_candidate_list:route_k6` | `random_256` | 256.000 | 0.000 | 0.062 | 0.017 |
| glm4 | `with_candidate_list:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.146 | 0.042 |
| glm4 | `with_candidate_list:route_k6` | `random_256` | 256.000 | 0.000 | 0.062 | 0.011 |
| glm4 | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.021 | 0.010 |
| glm4 | `lowercase_short_value:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.010 | 0.044 |
| deepseek7b | `lowercase_short_value:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.608 | 0.048 |
| deepseek7b | `with_candidate_list:route_k6` | `random_1024` | 1024.000 | 0.000 | 0.251 | 0.049 |
| deepseek7b | `lowercase_short_value:route_k6` | `random_256` | 256.000 | 0.000 | 0.136 | 0.012 |
| deepseek7b | `with_candidate_list:route_k6` | `random_256` | 256.000 | 0.000 | 0.087 | 0.012 |

## Strict Interpretation

- Positive subspace success supports a readout-supporting subspace.
- Negative-only patch with negative margin supports an interfering subspace.
- Random/neutral controls estimate whether success comes from score ranking rather than arbitrary channel count.
- This phase still does not split attention heads or MLP activation neurons.
