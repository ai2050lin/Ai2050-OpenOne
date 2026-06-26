# Phase 668 Cross-Model Summary

目标：比较 full boundary state、top-head ensemble、component ensemble 对 token1 transition 的闭合能力。

## qwen3

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_qwen3_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 5 / rows: 60 / total_time_min: 0.16
- ensembles: `[{'name': 'full_L23_layer_input', 'kind': 'component_set', 'components': [[23, 'layer_input']]}, {'name': 'full_L22_layer_out', 'kind': 'component_set', 'components': [[22, 'layer_out']]}, {'name': 'L22_attn_mlp', 'kind': 'component_set', 'components': [[22, 'attn_out'], [22, 'mlp_out']]}, {'name': 'L22_heads10_11', 'kind': 'head_set', 'layer': 22, 'heads': [10, 11]}]`

### Ensemble Specificity

| pair_task | site | combo | ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| explanation_required | separator_input_edge | top2 | L22_heads10_11 | head_set | 2 | 0.500 | 0.000 | 4.688 | 1.375 |
| explanation_required | separator_input_edge | top2 | full_L22_layer_out | component_set | 2 | 0.500 | 0.000 | 3.562 | 4.938 |
| explanation_required | separator_input_edge | top2 | full_L23_layer_input | component_set | 2 | 0.500 | 0.000 | 3.562 | 4.938 |
| explanation_required | separator_input_edge | top1 | L22_heads10_11 | head_set | 3 | 0.667 | 0.333 | 3.000 | 0.792 |
| explanation_required | separator_input_edge | top1 | full_L22_layer_out | component_set | 3 | 0.667 | 0.333 | 2.250 | 5.354 |
| explanation_required | separator_input_edge | top1 | full_L23_layer_input | component_set | 3 | 0.667 | 0.333 | 2.250 | 5.354 |
| explanation_required | separator_input_edge | top2 | L22_attn_mlp | component_set | 2 | 0.000 | 0.000 | 1.000 | -2.438 |
| explanation_required | separator_input_edge | top1 | L22_attn_mlp | component_set | 3 | 0.333 | 0.333 | 0.750 | -1.125 |

### Intervention Summary

| pair_task | site | combo | ensemble | intervention | n | top1_rate | rank_delta | margin_delta |
|---|---|---|---|---|---:|---:|---:|---:|
| explanation_required | separator_input_edge | top1 | L22_attn_mlp | correct_restore | 3 | 0.333 | 0.00 | -1.458 |
| explanation_required | separator_input_edge | top1 | L22_attn_mlp | mismatch_restore | 3 | 0.333 | -0.67 | -2.208 |
| explanation_required | separator_input_edge | top1 | L22_attn_mlp | zero_remove | 3 | 0.333 | 0.00 | -0.333 |
| explanation_required | separator_input_edge | top1 | L22_heads10_11 | correct_restore | 3 | 0.667 | 0.33 | 0.792 |
| explanation_required | separator_input_edge | top1 | L22_heads10_11 | mismatch_restore | 3 | 0.333 | 0.00 | -2.208 |
| explanation_required | separator_input_edge | top1 | L22_heads10_11 | zero_remove | 3 | 0.333 | 0.00 | 0.000 |
| explanation_required | separator_input_edge | top1 | full_L22_layer_out | correct_restore | 3 | 0.667 | 0.33 | 0.208 |
| explanation_required | separator_input_edge | top1 | full_L22_layer_out | mismatch_restore | 3 | 0.333 | -0.67 | -2.042 |
| explanation_required | separator_input_edge | top1 | full_L22_layer_out | zero_remove | 3 | 0.000 | -19.00 | -5.146 |
| explanation_required | separator_input_edge | top1 | full_L23_layer_input | correct_restore | 3 | 0.667 | 0.33 | 0.208 |
| explanation_required | separator_input_edge | top1 | full_L23_layer_input | mismatch_restore | 3 | 0.333 | -0.67 | -2.042 |
| explanation_required | separator_input_edge | top1 | full_L23_layer_input | zero_remove | 3 | 0.000 | -19.00 | -5.146 |
| explanation_required | separator_input_edge | top2 | L22_attn_mlp | correct_restore | 2 | 0.000 | 0.00 | -1.938 |
| explanation_required | separator_input_edge | top2 | L22_attn_mlp | mismatch_restore | 2 | 0.000 | -1.00 | -2.938 |
| explanation_required | separator_input_edge | top2 | L22_attn_mlp | zero_remove | 2 | 0.000 | 0.00 | 0.500 |
| explanation_required | separator_input_edge | top2 | L22_heads10_11 | correct_restore | 2 | 0.500 | 0.50 | 1.438 |
| explanation_required | separator_input_edge | top2 | L22_heads10_11 | mismatch_restore | 2 | 0.000 | 0.00 | -3.250 |
| explanation_required | separator_input_edge | top2 | L22_heads10_11 | zero_remove | 2 | 0.000 | 0.00 | 0.062 |
| explanation_required | separator_input_edge | top2 | full_L22_layer_out | correct_restore | 2 | 0.500 | 0.50 | 0.875 |
| explanation_required | separator_input_edge | top2 | full_L22_layer_out | mismatch_restore | 2 | 0.000 | -1.00 | -2.688 |
| explanation_required | separator_input_edge | top2 | full_L22_layer_out | zero_remove | 2 | 0.000 | -15.50 | -4.062 |
| explanation_required | separator_input_edge | top2 | full_L23_layer_input | correct_restore | 2 | 0.500 | 0.50 | 0.875 |
| explanation_required | separator_input_edge | top2 | full_L23_layer_input | mismatch_restore | 2 | 0.000 | -1.00 | -2.688 |
| explanation_required | separator_input_edge | top2 | full_L23_layer_input | zero_remove | 2 | 0.000 | -15.50 | -4.062 |

## glm4

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_glm4_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 4 / rows: 48 / total_time_min: 0.24
- ensembles: `[{'name': 'full_L22_attn_out', 'kind': 'component_set', 'components': [[22, 'attn_out']]}, {'name': 'full_L22_layer_input', 'kind': 'component_set', 'components': [[22, 'layer_input']]}, {'name': 'L21_layer_out_L22_attn_mlp', 'kind': 'component_set', 'components': [[21, 'layer_out'], [22, 'attn_out'], [22, 'mlp_out']]}, {'name': 'L22_heads7_13', 'kind': 'head_set', 'layer': 22, 'heads': [7, 13]}]`

### Ensemble Specificity

| pair_task | site | combo | ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| explanation_required | late_peak_layer_out | top1 | L22_heads7_13 | head_set | 3 | 1.000 | 0.333 | 0.604 | 0.292 |
| explanation_required | late_peak_layer_out | top1 | full_L22_attn_out | component_set | 3 | 1.000 | 0.333 | 0.500 | 0.000 |
| explanation_required | late_peak_layer_out | top1 | full_L22_layer_input | component_set | 3 | 1.000 | 0.667 | 0.417 | 3.729 |
| explanation_required | l22_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | component_set | 1 | 1.000 | 1.000 | 0.000 | 9.695 |
| explanation_required | late_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | component_set | 3 | 1.000 | 1.000 | 0.000 | 11.422 |
| explanation_required | l22_peak_layer_out | top1 | L22_heads7_13 | head_set | 1 | 1.000 | 1.000 | 0.000 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_attn_out | component_set | 1 | 1.000 | 1.000 | 0.000 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_layer_input | component_set | 1 | 1.000 | 1.000 | 0.000 | 4.312 |

### Intervention Summary

| pair_task | site | combo | ensemble | intervention | n | top1_rate | rank_delta | margin_delta |
|---|---|---|---|---|---:|---:|---:|---:|
| explanation_required | l22_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | correct_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | mismatch_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | zero_remove | 1 | 0.000 | -12630.00 | -9.695 |
| explanation_required | l22_peak_layer_out | top1 | L22_heads7_13 | correct_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | L22_heads7_13 | mismatch_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | L22_heads7_13 | zero_remove | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_attn_out | correct_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_attn_out | mismatch_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_attn_out | zero_remove | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_layer_input | correct_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_layer_input | mismatch_restore | 1 | 1.000 | 0.00 | 0.000 |
| explanation_required | l22_peak_layer_out | top1 | full_L22_layer_input | zero_remove | 1 | 0.000 | -43.00 | -4.312 |
| explanation_required | late_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | correct_restore | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | mismatch_restore | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | L21_layer_out_L22_attn_mlp | zero_remove | 3 | 0.000 | -12381.00 | -11.047 |
| explanation_required | late_peak_layer_out | top1 | L22_heads7_13 | correct_restore | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | L22_heads7_13 | mismatch_restore | 3 | 0.333 | 0.00 | -0.229 |
| explanation_required | late_peak_layer_out | top1 | L22_heads7_13 | zero_remove | 3 | 0.333 | 0.00 | 0.083 |
| explanation_required | late_peak_layer_out | top1 | full_L22_attn_out | correct_restore | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | full_L22_attn_out | mismatch_restore | 3 | 0.333 | 0.00 | -0.125 |
| explanation_required | late_peak_layer_out | top1 | full_L22_attn_out | zero_remove | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | full_L22_layer_input | correct_restore | 3 | 1.000 | 0.67 | 0.375 |
| explanation_required | late_peak_layer_out | top1 | full_L22_layer_input | mismatch_restore | 3 | 0.667 | 0.33 | -0.042 |
| explanation_required | late_peak_layer_out | top1 | full_L22_layer_input | zero_remove | 3 | 0.000 | -86.67 | -3.354 |

## deepseek7b

- source_phase665: `results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_deepseek7b_autoregressive_continuation_controller_localization_confirm.json`
- failures_tested: 12 / rows: 72 / total_time_min: 0.40
- ensembles: `[{'name': 'full_L21_layer_out', 'kind': 'component_set', 'components': [[21, 'layer_out']]}, {'name': 'full_L22_layer_input', 'kind': 'component_set', 'components': [[22, 'layer_input']]}, {'name': 'L21_attn_mlp', 'kind': 'component_set', 'components': [[21, 'attn_out'], [21, 'mlp_out']]}, {'name': 'L21_heads14_17', 'kind': 'head_set', 'layer': 21, 'heads': [14, 17]}]`

### Ensemble Specificity

| pair_task | site | combo | ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| explanation_required | l22_peak_layer_out | top2 | full_L21_layer_out | component_set | 3 | 1.000 | 0.000 | 3.312 | 8.729 |
| explanation_required | l22_peak_layer_out | top2 | full_L22_layer_input | component_set | 3 | 1.000 | 0.000 | 3.312 | 8.729 |
| explanation_required | late_peak_layer_out | top2 | full_L21_layer_out | component_set | 3 | 1.000 | 0.000 | 3.250 | 8.604 |
| explanation_required | late_peak_layer_out | top2 | full_L22_layer_input | component_set | 3 | 1.000 | 0.000 | 3.250 | 8.604 |
| explanation_required | l22_peak_layer_out | top2 | L21_attn_mlp | component_set | 3 | 0.333 | 0.000 | 0.542 | 0.271 |
| explanation_required | late_peak_layer_out | top2 | L21_attn_mlp | component_set | 3 | 0.333 | 0.000 | 0.479 | 0.312 |
| explanation_required | l22_peak_layer_out | top2 | L21_heads14_17 | head_set | 3 | 0.333 | 0.333 | 0.292 | 0.021 |
| explanation_required | late_peak_layer_out | top2 | L21_heads14_17 | head_set | 3 | 0.333 | 0.333 | 0.292 | 0.062 |

### Intervention Summary

| pair_task | site | combo | ensemble | intervention | n | top1_rate | rank_delta | margin_delta |
|---|---|---|---|---|---:|---:|---:|---:|
| explanation_required | l22_peak_layer_out | top2 | L21_attn_mlp | correct_restore | 3 | 0.333 | -0.33 | -0.042 |
| explanation_required | l22_peak_layer_out | top2 | L21_attn_mlp | mismatch_restore | 3 | 0.000 | -0.33 | -0.583 |
| explanation_required | l22_peak_layer_out | top2 | L21_attn_mlp | zero_remove | 3 | 0.333 | -0.33 | -0.312 |
| explanation_required | l22_peak_layer_out | top2 | L21_heads14_17 | correct_restore | 3 | 0.333 | -0.33 | -0.188 |
| explanation_required | l22_peak_layer_out | top2 | L21_heads14_17 | mismatch_restore | 3 | 0.333 | -0.33 | -0.479 |
| explanation_required | l22_peak_layer_out | top2 | L21_heads14_17 | zero_remove | 3 | 0.333 | -0.33 | -0.208 |
| explanation_required | l22_peak_layer_out | top2 | full_L21_layer_out | correct_restore | 3 | 1.000 | 1.00 | 1.042 |
| explanation_required | l22_peak_layer_out | top2 | full_L21_layer_out | mismatch_restore | 3 | 0.000 | -4.00 | -2.271 |
| explanation_required | l22_peak_layer_out | top2 | full_L21_layer_out | zero_remove | 3 | 0.000 | -375.00 | -7.688 |
| explanation_required | l22_peak_layer_out | top2 | full_L22_layer_input | correct_restore | 3 | 1.000 | 1.00 | 1.042 |
| explanation_required | l22_peak_layer_out | top2 | full_L22_layer_input | mismatch_restore | 3 | 0.000 | -4.00 | -2.271 |
| explanation_required | l22_peak_layer_out | top2 | full_L22_layer_input | zero_remove | 3 | 0.000 | -375.00 | -7.688 |
| explanation_required | late_peak_layer_out | top2 | L21_attn_mlp | correct_restore | 3 | 0.333 | -0.33 | -0.292 |
| explanation_required | late_peak_layer_out | top2 | L21_attn_mlp | mismatch_restore | 3 | 0.000 | -0.67 | -0.771 |
| explanation_required | late_peak_layer_out | top2 | L21_attn_mlp | zero_remove | 3 | 0.333 | -0.67 | -0.604 |
| explanation_required | late_peak_layer_out | top2 | L21_heads14_17 | correct_restore | 3 | 0.333 | 0.00 | -0.104 |
| explanation_required | late_peak_layer_out | top2 | L21_heads14_17 | mismatch_restore | 3 | 0.333 | -0.67 | -0.396 |
| explanation_required | late_peak_layer_out | top2 | L21_heads14_17 | zero_remove | 3 | 0.333 | 0.00 | -0.167 |
| explanation_required | late_peak_layer_out | top2 | full_L21_layer_out | correct_restore | 3 | 1.000 | 0.67 | 0.771 |
| explanation_required | late_peak_layer_out | top2 | full_L21_layer_out | mismatch_restore | 3 | 0.000 | -4.33 | -2.479 |
| explanation_required | late_peak_layer_out | top2 | full_L21_layer_out | zero_remove | 3 | 0.000 | -535.00 | -7.833 |
| explanation_required | late_peak_layer_out | top2 | full_L22_layer_input | correct_restore | 3 | 1.000 | 0.67 | 0.771 |
| explanation_required | late_peak_layer_out | top2 | full_L22_layer_input | mismatch_restore | 3 | 0.000 | -4.33 | -2.479 |
| explanation_required | late_peak_layer_out | top2 | full_L22_layer_input | zero_remove | 3 | 0.000 | -535.00 | -7.833 |
