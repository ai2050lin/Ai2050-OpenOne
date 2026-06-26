# Phase 666 Cross-Model Summary

目标：在 Phase 665 找到的最早 token0->token1 边界上，比较 baseline / self_restore / zero_remove / mismatch_restore / correct_restore，审计 token1 转移门是否具有语义特异性。

## qwen3

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_qwen3_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 5 / rows: 50 / total_time_min: 0.10
- boundaries: `[{'layer': 22, 'component': 'attn_out', 'label': 'L22_attn_out'}, {'layer': 23, 'component': 'layer_input', 'label': 'L23_layer_input'}]`
- interventions: `['baseline', 'self_restore', 'zero_remove', 'mismatch_restore', 'correct_restore']`

### Boundary Specificity

| pair_task | site | combo | boundary | n | correct_delta | mismatch_delta | zero_delta | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| explanation_required | separator_input_edge | top2 | L23_layer_input | 2 | 0.875 | -2.688 | -4.062 | 3.562 | 4.938 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | 3 | 0.208 | -2.042 | -5.146 | 2.250 | 5.354 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | 2 | 1.875 | 1.875 | 0.875 | 0.000 | 1.000 |
| explanation_required | separator_input_edge | top1 | L22_attn_out | 3 | 0.833 | 0.833 | 0.333 | 0.000 | 0.500 |

### Intervention Summary

| pair_task | site | combo | boundary | intervention | n | top1_rate | mean_rank | mean_margin | mean_rank_delta | mean_margin_delta | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top1 | L22_attn_out | baseline | 3 | 0.333 | 1.67 | -0.833 | 0.00 | 0.000 | 2:2, 0:1 |
| explanation_required | separator_input_edge | top1 | L22_attn_out | self_restore | 3 | 0.667 | 1.33 | -0.417 | 0.33 | 0.417 | 4:1, 2:1, 0:1 |
| explanation_required | separator_input_edge | top1 | L22_attn_out | zero_remove | 3 | 0.333 | 1.67 | -0.500 | 0.00 | 0.333 | 2:2, 0:1 |
| explanation_required | separator_input_edge | top1 | L22_attn_out | mismatch_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | L22_attn_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | baseline | 3 | 0.333 | 1.67 | -0.833 | 0.00 | 0.000 | 2:2, 0:1 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | self_restore | 3 | 0.667 | 1.33 | -0.417 | 0.33 | 0.417 | 4:1, 2:1, 0:1 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | zero_remove | 3 | 0.000 | 20.67 | -5.979 | -19.00 | -5.146 | .:3 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | mismatch_restore | 3 | 0.333 | 2.33 | -2.875 | -0.67 | -2.042 | 0:3 |
| explanation_required | separator_input_edge | top1 | L23_layer_input | correct_restore | 3 | 0.667 | 1.33 | -0.625 | 0.33 | 0.208 | 4:1, 2:1, 0:1 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | baseline | 2 | 0.000 | 2.00 | -1.875 | 0.00 | 0.000 | 2:2 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | self_restore | 2 | 0.000 | 2.00 | -1.000 | 0.00 | 0.875 | 2:2 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | zero_remove | 2 | 0.000 | 2.00 | -1.000 | 0.00 | 0.875 | 2:2 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | mismatch_restore | 2 | 1.000 | 1.00 | 0.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | L22_attn_out | correct_restore | 2 | 1.000 | 1.00 | 0.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | L23_layer_input | baseline | 2 | 0.000 | 2.00 | -1.875 | 0.00 | 0.000 | 2:2 |
| explanation_required | separator_input_edge | top2 | L23_layer_input | self_restore | 2 | 0.000 | 2.00 | -0.562 | 0.00 | 1.312 | 2:2 |
| explanation_required | separator_input_edge | top2 | L23_layer_input | zero_remove | 2 | 0.000 | 17.50 | -5.938 | -15.50 | -4.062 | .:2 |
| explanation_required | separator_input_edge | top2 | L23_layer_input | mismatch_restore | 2 | 0.000 | 3.00 | -4.562 | -1.00 | -2.688 | 0:2 |
| explanation_required | separator_input_edge | top2 | L23_layer_input | correct_restore | 2 | 0.500 | 1.50 | -1.000 | 0.50 | 0.875 | 4:1, 2:1 |

## glm4

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_glm4_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 4 / rows: 60 / total_time_min: 0.16
- boundaries: `[{'layer': 22, 'component': 'layer_input', 'label': 'L22_layer_input'}, {'layer': 22, 'component': 'attn_out', 'label': 'L22_attn_out'}, {'layer': 22, 'component': 'layer_out', 'label': 'L22_layer_out'}]`
- interventions: `['baseline', 'self_restore', 'zero_remove', 'mismatch_restore', 'correct_restore']`

### Boundary Specificity

| pair_task | site | combo | boundary | n | correct_delta | mismatch_delta | zero_delta | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | 3 | 0.375 | -0.125 | 0.375 | 0.500 | 0.000 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | 3 | 0.375 | -0.042 | -3.354 | 0.417 | 3.729 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | 3 | 0.375 | 0.375 | -11.047 | 0.000 | 11.422 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | 1 | 0.000 | 0.000 | -9.695 | 0.000 | 9.695 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | 1 | 0.000 | 0.000 | -4.312 | 0.000 | 4.312 |
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Intervention Summary

| pair_task | site | combo | boundary | intervention | n | top1_rate | mean_rank | mean_margin | mean_rank_delta | mean_margin_delta | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | baseline | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | self_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | zero_remove | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | mismatch_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_attn_out | correct_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | baseline | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | self_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | zero_remove | 1 | 0.000 | 44.00 | -4.312 | -43.00 | -4.312 |  ?:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | mismatch_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_input | correct_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | baseline | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | self_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | zero_remove | 1 | 0.000 | 12631.00 | -9.695 | -12630.00 | -9.695 |  hard:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | mismatch_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | L22_layer_out | correct_restore | 1 | 1.000 | 1.00 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | baseline | 3 | 0.333 | 1.67 | -0.375 | 0.00 | 0.000 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | self_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | zero_remove | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:2, 05:1 |
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | mismatch_restore | 3 | 0.333 | 1.67 | -0.500 | 0.00 | -0.125 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_attn_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | baseline | 3 | 0.333 | 1.67 | -0.375 | 0.00 | 0.000 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | self_restore | 3 | 0.333 | 1.67 | -0.375 | 0.00 | 0.000 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | zero_remove | 3 | 0.000 | 88.33 | -3.729 | -86.67 | -3.354 | ?:2,  of:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | mismatch_restore | 3 | 0.667 | 1.33 | -0.417 | 0.33 | -0.042 | 22:2, 05:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_input | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | baseline | 3 | 0.333 | 1.67 | -0.375 | 0.00 | 0.000 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | self_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | zero_remove | 3 | 0.000 | 12382.67 | -11.422 | -12381.00 | -11.047 |  heavy:2,  hard:1 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | mismatch_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | L22_layer_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.375 | 22:3 |

## deepseek7b

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_deepseek7b_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 12 / rows: 90 / total_time_min: 0.23
- boundaries: `[{'layer': 21, 'component': 'layer_out', 'label': 'L21_layer_out'}, {'layer': 22, 'component': 'layer_input', 'label': 'L22_layer_input'}, {'layer': 22, 'component': 'layer_out', 'label': 'L22_layer_out'}]`
- interventions: `['baseline', 'self_restore', 'zero_remove', 'mismatch_restore', 'correct_restore']`

### Boundary Specificity

| pair_task | site | combo | boundary | n | correct_delta | mismatch_delta | zero_delta | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | 3 | 1.042 | -2.271 | -7.688 | 3.312 | 8.729 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | 3 | 1.042 | -2.271 | -7.688 | 3.312 | 8.729 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | 3 | 0.771 | -2.479 | -7.833 | 3.250 | 8.604 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | 3 | 0.771 | -2.479 | -7.833 | 3.250 | 8.604 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | 3 | 0.771 | -2.312 | -6.042 | 3.083 | 6.812 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | 3 | 1.042 | -2.042 | -5.771 | 3.083 | 6.812 |

### Intervention Summary

| pair_task | site | combo | boundary | intervention | n | top1_rate | mean_rank | mean_margin | mean_rank_delta | mean_margin_delta | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | baseline | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | self_restore | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | zero_remove | 3 | 0.000 | 377.00 | -8.729 | -375.00 | -7.688 | ,:2,  the:1 |
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | mismatch_restore | 3 | 0.000 | 6.00 | -3.312 | -4.00 | -2.271 | 2:2, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | L21_layer_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | baseline | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | self_restore | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | zero_remove | 3 | 0.000 | 377.00 | -8.729 | -375.00 | -7.688 | ,:2,  the:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | mismatch_restore | 3 | 0.000 | 6.00 | -3.312 | -4.00 | -2.271 | 2:2, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_input | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | baseline | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | self_restore | 3 | 0.333 | 2.00 | -1.042 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | zero_remove | 3 | 0.000 | 80.33 | -6.812 | -78.33 | -5.771 |  o:2, ,:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | mismatch_restore | 3 | 0.000 | 6.00 | -3.083 | -4.00 | -2.042 | 2:2, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | L22_layer_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | baseline | 3 | 0.333 | 1.67 | -0.771 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | self_restore | 3 | 0.333 | 1.67 | -1.021 | 0.00 | -0.250 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | zero_remove | 3 | 0.000 | 536.67 | -8.604 | -535.00 | -7.833 | ,:2,  the:1 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | mismatch_restore | 3 | 0.000 | 6.00 | -3.250 | -4.33 | -2.479 | 2:2, 4:1 |
| explanation_required | late_peak_layer_out | top2 | L21_layer_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | baseline | 3 | 0.333 | 1.67 | -0.771 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | self_restore | 3 | 0.333 | 1.67 | -1.021 | 0.00 | -0.250 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | zero_remove | 3 | 0.000 | 536.67 | -8.604 | -535.00 | -7.833 | ,:2,  the:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | mismatch_restore | 3 | 0.000 | 6.00 | -3.250 | -4.33 | -2.479 | 2:2, 4:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_input | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | baseline | 3 | 0.333 | 1.67 | -0.771 | 0.00 | 0.000 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | self_restore | 3 | 0.333 | 2.00 | -1.042 | -0.33 | -0.271 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | zero_remove | 3 | 0.000 | 80.33 | -6.812 | -78.67 | -6.042 |  o:2, ,:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | mismatch_restore | 3 | 0.000 | 6.00 | -3.083 | -4.33 | -2.312 | 2:2, 4:1 |
| explanation_required | late_peak_layer_out | top2 | L22_layer_out | correct_restore | 3 | 1.000 | 1.00 | 0.000 | 0.67 | 0.771 | 4:2, 2:1 |
