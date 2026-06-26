# Phase 654 Cross-Model Summary

目标：固定 Phase 653 的强峰 restore patch，审计 support_value 到 generate_value 之间的桥接失败。重点观察 rank 进入前 15 但短生成仍不输出正确值的样本。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / time: 0.97 min
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- sites: `['separator_input_edge', 'early_peak_layer_out']`

### By Mode

| pair_task | eval_task | direction | site | layers | components | n | mean_rank | mean_margin_vs_top | exact | tok0 | support_no_gen | final_l2 | top0_category | gen_first_text |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | explanation_required |  |  |  |  | 20 | 1.80 | -0.613 | 9/20 | 10/20 | 0/20 | 0.000 | explanation:10, correct_prefix:10 |  The:10,  v:10 |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 14,15,16,17 | layer_out | 20 | 4.35 | -1.700 | 7/20 | 8/20 | 12/20 | 31.302 | space:11, correct_prefix:8, word:1 |  :12,  v:7,  o:1 |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 14 | layer_input | 20 | 3.10 | -1.137 | 4/20 | 4/20 | 16/20 | 20.142 | explanation:12, space:4, correct_prefix:4 |  The:11,  v:5,  :4 |
| explanation_required | short_value_allowed |  |  |  |  | 20 | 9.20 | -4.963 | 0/20 | 0/20 | 0/20 | 0.000 | space:20 |  :20 |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 14,15,16,17 | layer_out | 20 | 6.85 | -3.438 | 5/20 | 5/20 | 14/20 | 24.208 | space:15, correct_prefix:5 |  :15,  v:5 |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 14 | layer_input | 20 | 8.60 | -4.756 | 1/20 | 1/20 | 15/20 | 13.916 | space:19, correct_prefix:1 |  :19,  v:1 |
| yes_no_required | short_value_allowed |  |  |  |  | 20 | 9.20 | -4.963 | 0/20 | 0/20 | 0/20 | 0.000 | space:20 |  :20 |
| yes_no_required | short_value_allowed | task_to_value | early_peak_layer_out | 14,15,16,17 | layer_out | 20 | 4.40 | -1.887 | 9/20 | 9/20 | 11/20 | 36.514 | space:10, correct_prefix:9, explanation:1 |  :10,  v:9,  The:1 |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 14 | layer_input | 20 | 7.65 | -3.894 | 4/20 | 4/20 | 16/20 | 25.498 | space:16, correct_prefix:4 |  :16,  v:4 |
| yes_no_required | yes_no_required |  |  |  |  | 20 | 12.95 | -2.922 | 0/20 | 0/20 | 0/20 | 0.000 | explanation:20 |  Yes:20 |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 14,15,16,17 | layer_out | 20 | 4.15 | -2.219 | 5/20 | 5/20 | 15/20 | 42.123 | space:15, correct_prefix:5 |  :15,  v:5 |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 14 | layer_input | 20 | 1.85 | -0.606 | 11/20 | 11/20 | 9/20 | 30.716 | correct_prefix:11, space:8, explanation:1 |  v:11,  :8,  Yes:1 |

### Bridge Failures: rank <= 15 and exact false

| pair_task | eval_task | direction | site | prefix_rank | margin_vs_top | top0 | gen_first | generation_text |
|---|---|---|---|---:|---:|---|---|---|
| explanation_required | explanation_required | value_to_task | separator_input_edge | 1 | 0.000 |  v |  v |  v22<nl><nl>Wait, |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 1 | 0.000 |  v |   |  48<nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 2 | -0.125 |  The |  The |  The value is 48 |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 2 | -0.250 |   |   |  48<nl><nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 2 | -0.250 |   |   |  48<nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 2 | -0.375 |  The |  The |  The value is 48 |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 2 | -0.375 |   |   |  48<nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -0.375 |   |   |  22<nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 2 | -0.500 |  The |  The |  The value is 22 |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 2 | -0.500 |   |   |  48<nl>Okay, |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -0.500 |   |   |  48<nl><nl>Okay, |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -0.500 |   |   |  22<nl><nl>Wait, |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -0.625 |   |   |  48<nl><nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 2 | -0.625 |  The |  The |  The value is 48 |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -0.625 |   |   |  22<nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 2 | -0.750 |  The |  The |  The value is 48 |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -0.750 |   |   |  48<nl><nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -0.875 |   |   |  22<nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 2 | -0.875 |   |   |  48<nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 2 | -0.875 |   |   |  48<nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -1.000 |   |   |  22<nl>Okay, |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -1.000 |   |   |  22<nl><nl>Wait, |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 2 | -1.125 |   |   |  48<nl><nl>The answer |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 2 | -1.375 |   |   |  48<nl><nl>Okay, |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -1.500 |   |   |  48<nl><nl>Okay, |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 2 | -1.500 |   |   |  48<nl><nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -1.625 |   |   |  05<nl>Okay, |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 2 | -1.750 |   |   |  22<nl><nl>Okay, |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -1.875 |   |   |  48<nl><nl>Okay, |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -1.875 |   |   |  48<nl><nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 2 | -2.000 |   |   |  22<nl>Okay, |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 2 | -2.250 |   |   |  48<nl><nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 3 | -0.250 |  The |  The |  The answer is v91 |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 3 | -0.750 |  o |  o |  o17<nl>Reason: |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 3 | -1.000 |   |   |  48<nl><nl>The answer |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 3 | -1.125 |  The |  The |  The value is 48 |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 3 | -1.250 |   |   |  22<nl><nl>Okay, |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 3 | -1.500 |  The |  The |  The value is 22 |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 3 | -2.125 |   |   |  48<nl><nl>Okay, |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 3 | -2.250 |   |   |  48<nl>Okay, |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / time: 1.45 min
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- sites: `['l22_peak_layer_out', 'late_peak_layer_out']`

### By Mode

| pair_task | eval_task | direction | site | layers | components | n | mean_rank | mean_margin_vs_top | exact | tok0 | support_no_gen | final_l2 | top0_category | gen_first_text |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | explanation_required |  |  |  |  | 20 | 58.30 | -5.239 | 0/20 | 0/20 | 0/20 | 0.000 | explanation:20 |  The:20 |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 22 | layer_out | 20 | 2.15 | -0.803 | 5/20 | 5/20 | 15/20 | 120.985 | space:15, correct_prefix:5 |  :15,  v:5 |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 21,22 | layer_out | 20 | 2.15 | -0.803 | 5/20 | 5/20 | 15/20 | 120.985 | space:15, correct_prefix:5 |  :15,  v:5 |
| explanation_required | short_value_allowed |  |  |  |  | 20 | 2.20 | -1.031 | 3/20 | 3/20 | 0/20 | 0.000 | space:17, correct_prefix:3 |  :17,  v:3 |
| explanation_required | short_value_allowed | task_to_value | l22_peak_layer_out | 22 | layer_out | 20 | 44.15 | -5.069 | 0/20 | 0/20 | 4/20 | 118.829 | explanation:20 |  The:20 |
| explanation_required | short_value_allowed | task_to_value | late_peak_layer_out | 21,22 | layer_out | 20 | 44.15 | -5.069 | 0/20 | 0/20 | 4/20 | 118.829 | explanation:20 |  The:20 |
| yes_no_required | short_value_allowed |  |  |  |  | 20 | 2.20 | -1.031 | 3/20 | 3/20 | 0/20 | 0.000 | space:17, correct_prefix:3 |  :17,  v:3 |
| yes_no_required | short_value_allowed | task_to_value | l22_peak_layer_out | 22 | layer_out | 20 | 135.25 | -6.723 | 0/20 | 0/20 | 0/20 | 102.387 | explanation:16, word:4 |  Yes:16,  c:4 |
| yes_no_required | short_value_allowed | task_to_value | late_peak_layer_out | 21,22 | layer_out | 20 | 135.25 | -6.723 | 0/20 | 0/20 | 0/20 | 102.387 | explanation:16, word:4 |  Yes:16,  c:4 |
| yes_no_required | yes_no_required |  |  |  |  | 20 | 188.10 | -9.116 | 0/20 | 0/20 | 0/20 | 0.000 | explanation:20 |  Yes:12,  yes:8 |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 22 | layer_out | 20 | 3.45 | -1.134 | 3/20 | 3/20 | 17/20 | 90.066 | space:15, correct_prefix:3, explanation:2 |  :15,  v:3,  no:1,  yes:1 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 21,22 | layer_out | 20 | 3.45 | -1.134 | 3/20 | 3/20 | 17/20 | 90.066 | space:15, correct_prefix:3, explanation:2 |  :15,  v:3,  no:1,  yes:1 |

### Bridge Failures: rank <= 15 and exact false

| pair_task | eval_task | direction | site | prefix_rank | margin_vs_top | top0 | gen_first | generation_text |
|---|---|---|---|---:|---:|---|---|---|
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.062 |   |   |  91<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.062 |   |   |  91<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.188 |   |   |  48.<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.188 |   |   |  48<nl><nl>Reason: According |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -0.312 |   |   |  48.<nl><nl>o71 belongs |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -0.312 |   |   |  48.<nl><nl>o71 belongs |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.312 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.312 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.500 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.500 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.562 |   |   |  0<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.562 |   |   |  48<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.625 |   |   |  0<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.625 |   |   |  48<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.625 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.625 |   |   |  22<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.688 |   |   |  48<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.688 |   |   |  48<nl><nl>Reason: The |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -0.812 |   |   |  48.<nl><nl>c33 r |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -0.812 |   |   |  48.<nl><nl>c33 r |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -0.875 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -0.875 |   |   |  22.<nl><nl>Question: c |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -1.000 |   |   |  22<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -1.000 |   |   |  22<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -1.000 |   |   |  48.<nl><nl>Reason: According |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -1.000 |   |   |  48<nl><nl>Reason: According |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -1.125 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -1.125 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -1.188 |   |   |  48.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -1.188 |   |   |  48.yesno.<nl><nl> |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -1.250 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -1.250 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 3 | -0.188 |   |   |  91.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 3 | -0.188 |   |   |  91.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 3 | -0.375 |   |   |  91.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 3 | -0.375 |   |   |  91.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 3 | -1.000 |   |   |  22.<nl><nl>Question: c |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 3 | -1.000 |   |   |  22.<nl><nl>Question: c |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.062 |   |   |  22<nl><nl>Reason: The |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.062 |   |   |  22<nl><nl>Reason: The |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / time: 1.24 min
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- sites: `['l22_peak_layer_out', 'late_peak_layer_out']`

### By Mode

| pair_task | eval_task | direction | site | layers | components | n | mean_rank | mean_margin_vs_top | exact | tok0 | support_no_gen | final_l2 | top0_category | gen_first_text |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | explanation_required |  |  |  |  | 20 | 77.20 | -5.575 | 0/20 | 0/20 | 0/20 | 0.000 | word:10, newline:9, explanation:1 |  c:11,  

:5,  ?

:4 |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 22 | layer_out | 20 | 8.30 | -2.194 | 3/20 | 2/20 | 14/20 | 48.312 | space:10, newline:8, correct_prefix:2 |  :11,  ?

:6,  v:3 |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 20,21,22 | layer_out | 20 | 8.30 | -2.194 | 3/20 | 2/20 | 14/20 | 48.312 | space:10, newline:8, correct_prefix:2 |  :11,  ?

:6,  v:3 |
| explanation_required | short_value_allowed |  |  |  |  | 20 | 8.00 | -2.325 | 0/20 | 0/20 | 0/20 | 0.000 | newline:10, space:10 |  ?

:10,  :10 |
| explanation_required | short_value_allowed | task_to_value | l22_peak_layer_out | 22 | layer_out | 20 | 56.15 | -5.244 | 0/20 | 0/20 | 1/20 | 48.178 | word:11, newline:8, explanation:1 |  c:12,  

:5,  ?

:3 |
| explanation_required | short_value_allowed | task_to_value | late_peak_layer_out | 20,21,22 | layer_out | 20 | 56.15 | -5.244 | 0/20 | 0/20 | 1/20 | 48.178 | word:11, newline:8, explanation:1 |  c:12,  

:5,  ?

:3 |
| yes_no_required | short_value_allowed |  |  |  |  | 20 | 8.00 | -2.325 | 0/20 | 0/20 | 0/20 | 0.000 | newline:10, space:10 |  ?

:10,  :10 |
| yes_no_required | short_value_allowed | task_to_value | l22_peak_layer_out | 22 | layer_out | 20 | 61.75 | -5.388 | 0/20 | 0/20 | 4/20 | 49.914 | newline:17, explanation:3 |  ?

:16,  Yes:4 |
| yes_no_required | short_value_allowed | task_to_value | late_peak_layer_out | 20,21,22 | layer_out | 20 | 61.75 | -5.388 | 0/20 | 0/20 | 4/20 | 49.914 | newline:17, explanation:3 |  ?

:16,  Yes:4 |
| yes_no_required | yes_no_required |  |  |  |  | 20 | 295.55 | -9.172 | 0/20 | 0/20 | 0/20 | 0.000 | explanation:20 |  yes:20 |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 22 | layer_out | 20 | 13.85 | -2.812 | 0/20 | 0/20 | 15/20 | 50.882 | newline:11, space:9 |  :11,  ?

:9 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 20,21,22 | layer_out | 20 | 13.85 | -2.812 | 0/20 | 0/20 | 15/20 | 50.882 | newline:11, space:9 |  :11,  ?

:9 |

### Bridge Failures: rank <= 15 and exact false

| pair_task | eval_task | direction | site | prefix_rank | margin_vs_top | top0 | gen_first | generation_text |
|---|---|---|---|---:|---:|---|---|---|
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 1 | 0.000 |   |   |  48.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 1 | 0.000 |   |   |  48.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -0.125 |  ?\n\n |  ?

 |  ?<nl><nl>Question: c33 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -0.125 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I'm |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 2 | -0.375 |  ?\n\n |  ?

 |  ?<nl><nl>Question: c12 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 2 | -0.375 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I'm |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 2 | -0.875 |   |   |  05.<nl><nl>But why |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 2 | -0.875 |   |   |  05.<nl><nl>But why |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -0.875 |   |   |  91<nl>Because c |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -0.875 |   |   |  91<nl>Because c |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.000 |   |   |  91<nl>Because:<nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.000 |   |   |  91<nl>Because:<nl><nl> |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.125 |   |   |  48<nl></think><nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.125 |   |   |  48.<nl><nl>So, |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.125 |   |   |  48<nl></think><nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.125 |   |   |  48.<nl><nl>Explanation: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 3 | -1.250 |   |   |  91.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 3 | -1.250 |   |   |  91.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 3 | -1.250 |   |   |  05.<nl><nl></think><nl><nl> |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 3 | -1.250 |   |   |  05.<nl><nl></think><nl><nl> |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.500 |   |   |  05. Why?<nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.500 |   |   |  05. Why?<nl><nl> |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 3 | -1.625 |   |   |  48<nl></think><nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 3 | -1.625 |   |   |  48.<nl><nl>Explanation: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 4 | -1.750 |   |   |  43.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 4 | -1.750 |   |   |  48.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 4 | -1.750 |   |   |  48.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 4 | -1.750 |   |   |  48.<nl><nl>Question: |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 4 | -1.750 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I'm |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 4 | -1.750 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I'm |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 4 | -1.812 |   |   |  05. Why?<nl><nl> |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 4 | -1.812 |   |   |  05. Why?<nl><nl> |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 5 | -1.375 |   |   |  91.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 5 | -1.375 |   |   |  91.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 5 | -2.250 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I need |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 5 | -2.250 |  ?\n\n |  ?

 |  ?<nl><nl>Okay, so I need |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 6 | -2.250 |   |   |  48.<nl><nl>Question: |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 6 | -2.250 |   |   |  48.<nl><nl>Question: |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 6 | -2.312 |   |   |  22<nl>Reason: |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 6 | -2.312 |   |   |  22<nl>Reason: |
