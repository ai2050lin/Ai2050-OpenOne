# Phase 663 Cross-Model Summary

目标：对 Phase 662 的 projection barrier 诊断做读出端反事实干预验证，区分 norm advantage、hidden direction alignment 和 continuation failure。

## qwen3

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 0.72
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 32, 'target_failure_seen': 3, 'fallback_used': 0, 'scanned': 33}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- direction_scales: `[0.5, 1.0, 1.5, 2.0]`

### Plus-Last-Writers Actual State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | separator_input_edge | top1 | 32 | 0.719 | 0.812 | 1.44 | 0.184 | correct_prefix:26, word:5, space:1 | exact_correct:23, first_token_competition_failure:6, correct_prefix_but_generation_wrong:3 |
| explanation_required | separator_input_edge | top2 | 32 | 0.875 | 0.938 | 1.06 | 0.023 | correct_prefix:30, word:2 | exact_correct:28, correct_prefix_but_generation_wrong:2, first_token_competition_failure:2 |
| yes_no_required | early_peak_layer_out | top2 | 32 | 1.000 | 1.000 | 1.00 | 0.000 | correct_prefix:32 | exact_correct:32 |
| yes_no_required | early_peak_layer_out | top3 | 32 | 0.906 | 0.875 | 1.12 | 0.016 | correct_prefix:28, explanation:3, newline:1 | exact_correct:29, first_token_competition_failure:3 |

### Norm-Neutralized Pair Readout

| pair_task | top1_category | n | top1_text | actual_gap | neutral_cos_gap | neutral_flip_rate | correct_cos | competitor_cos | norm_adv | needed_delta |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | word | 7 |  o:7 | 0.857 | 0.0103 | 0.000 | 0.1104 | 0.1208 | -0.0166 | 0.6502 |
| yes_no_required | explanation | 2 |  yes:2 | 0.156 | -0.0074 | 1.000 | 0.0946 | 0.0871 | 0.1077 | 0.1012 |
| explanation_required | space | 1 |  :1 | 0.625 | -0.0029 | 1.000 | 0.0996 | 0.0967 | 0.1040 | 0.4653 |
| yes_no_required | newline | 1 |  \n\n:1 | 0.188 | -0.0013 | 1.000 | 0.0937 | 0.0924 | 0.0415 | 0.1276 |

### Direction Correction by Scale

| pair_task | top1_category | scale | n | correct_top1_rate | mean_rank | mean_gap | top1_after |
|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | space | 0.5 | 1 | 0.000 | 2.00 | 0.375 | space:1 |
| explanation_required | space | 1.0 | 1 | 0.000 | 2.00 | 0.062 | space:1 |
| explanation_required | space | 1.5 | 1 | 1.000 | 1.00 | 0.000 | correct_prefix:1 |
| explanation_required | space | 2.0 | 1 | 1.000 | 1.00 | 0.000 | correct_prefix:1 |
| explanation_required | word | 0.5 | 7 | 0.000 | 2.14 | 0.429 | word:7 |
| explanation_required | word | 1.0 | 7 | 0.143 | 1.14 | 0.027 | word:6, correct_prefix:1 |
| explanation_required | word | 1.5 | 7 | 1.000 | 1.00 | 0.000 | correct_prefix:7 |
| explanation_required | word | 2.0 | 7 | 1.000 | 1.00 | 0.000 | correct_prefix:7 |
| yes_no_required | explanation | 0.5 | 2 | 0.000 | 2.50 | 0.125 | explanation:2 |
| yes_no_required | explanation | 1.0 | 2 | 0.000 | 1.00 | 0.000 | explanation:2 |
| yes_no_required | explanation | 1.5 | 2 | 0.500 | 1.00 | 0.000 | correct_prefix:1, explanation:1 |
| yes_no_required | explanation | 2.0 | 2 | 1.000 | 1.00 | 0.000 | correct_prefix:2 |
| yes_no_required | newline | 0.5 | 1 | 0.000 | 2.00 | 0.188 | newline:1 |
| yes_no_required | newline | 1.0 | 1 | 0.000 | 2.00 | 0.062 | newline:1 |
| yes_no_required | newline | 1.5 | 1 | 1.000 | 1.00 | 0.000 | correct_prefix:1 |
| yes_no_required | newline | 2.0 | 1 | 1.000 | 1.00 | 0.000 | correct_prefix:1 |

### Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| explanation_required | separator_input_edge | top1 | 3 |  v22\n\nWait,:1,  v22\n\nOkay,:1,  05\n\nThe answer:1 |
| explanation_required | separator_input_edge | top2 | 2 |  v22\n\nWait,:1,  v22\n\nOkay,:1 |

## glm4

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 1.11
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 33, 'target_failure_seen': 2, 'fallback_used': 0, 'scanned': 34}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- direction_scales: `[0.5, 1.0, 1.5, 2.0]`

### Plus-Last-Writers Actual State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top1 | 32 | 0.781 | 0.812 | 1.28 | 0.131 | correct_prefix:26, space:5, word:1 | exact_correct:25, first_token_competition_failure:6, correct_prefix_but_generation_wrong:1 |
| explanation_required | late_peak_layer_out | top1 | 32 | 0.719 | 0.812 | 1.28 | 0.131 | correct_prefix:26, space:5, word:1 | exact_correct:23, first_token_competition_failure:6, correct_prefix_but_generation_wrong:3 |
| yes_no_required | l22_peak_layer_out | top2 | 32 | 0.688 | 0.688 | 1.44 | 0.164 | correct_prefix:22, space:5, word:5 | exact_correct:22, first_token_competition_failure:10 |
| yes_no_required | late_peak_layer_out | top2 | 32 | 0.688 | 0.688 | 1.44 | 0.164 | correct_prefix:22, space:5, word:5 | exact_correct:22, first_token_competition_failure:10 |

### Norm-Neutralized Pair Readout

| pair_task | top1_category | n | top1_text | actual_gap | neutral_cos_gap | neutral_flip_rate | correct_cos | competitor_cos | norm_adv | needed_delta |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | space | 10 |  :10 | 0.825 | 0.0068 | 0.000 | 0.0872 | 0.0940 | -0.0045 | 0.8939 |
| yes_no_required | space | 10 |  :10 | 0.787 | 0.0067 | 0.000 | 0.0897 | 0.0964 | -0.0045 | 0.8406 |
| yes_no_required | word | 10 |  o:10 | 0.263 | 0.0153 | 0.000 | 0.1041 | 0.1193 | -0.0808 | 0.2924 |
| explanation_required | word | 2 |  o:2 | 0.062 | 0.0130 | 0.000 | 0.1046 | 0.1176 | -0.0808 | 0.0696 |

### Direction Correction by Scale

| pair_task | top1_category | scale | n | correct_top1_rate | mean_rank | mean_gap | top1_after |
|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | space | 0.5 | 10 | 0.000 | 2.20 | 0.425 | space:10 |
| explanation_required | space | 1.0 | 10 | 0.600 | 1.40 | 0.037 | correct_prefix:6, space:4 |
| explanation_required | space | 1.5 | 10 | 1.000 | 1.00 | 0.000 | correct_prefix:10 |
| explanation_required | space | 2.0 | 10 | 0.800 | 1.00 | 0.000 | correct_prefix:8, space:2 |
| explanation_required | word | 0.5 | 2 | 0.000 | 1.00 | 0.000 | word:2 |
| explanation_required | word | 1.0 | 2 | 0.000 | 1.00 | 0.000 | word:2 |
| explanation_required | word | 1.5 | 2 | 1.000 | 1.00 | 0.000 | correct_prefix:2 |
| explanation_required | word | 2.0 | 2 | 1.000 | 1.00 | 0.000 | correct_prefix:2 |
| yes_no_required | space | 0.5 | 10 | 0.000 | 2.20 | 0.412 | space:10 |
| yes_no_required | space | 1.0 | 10 | 0.200 | 1.20 | 0.013 | space:8, correct_prefix:2 |
| yes_no_required | space | 1.5 | 10 | 0.800 | 1.00 | 0.000 | correct_prefix:8, space:2 |
| yes_no_required | space | 2.0 | 10 | 0.800 | 1.00 | 0.000 | correct_prefix:8, space:2 |
| yes_no_required | word | 0.5 | 10 | 0.000 | 2.40 | 0.175 | word:10 |
| yes_no_required | word | 1.0 | 10 | 0.200 | 1.80 | 0.025 | word:8, correct_prefix:2 |
| yes_no_required | word | 1.5 | 10 | 0.800 | 1.00 | 0.000 | correct_prefix:8, space:2 |
| yes_no_required | word | 2.0 | 10 | 1.000 | 1.00 | 0.000 | correct_prefix:10 |

### Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| explanation_required | late_peak_layer_out | top1 | 3 |  v05\n\nReason: According:2,  22\n\nReason: The:1 |
| explanation_required | l22_peak_layer_out | top1 | 1 |  22\n\nReason: The:1 |

## deepseek7b

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 0.99
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 33, 'target_failure_seen': 10, 'fallback_used': 0, 'scanned': 37}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`
- direction_scales: `[0.5, 1.0, 1.5, 2.0]`

### Plus-Last-Writers Actual State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top2 | 32 | 0.500 | 0.500 | 2.59 | 0.727 | space:16, correct_prefix:16 | first_token_competition_failure:16, exact_correct:16 |
| explanation_required | late_peak_layer_out | top2 | 32 | 0.500 | 0.500 | 2.59 | 0.727 | space:16, correct_prefix:16 | first_token_competition_failure:16, exact_correct:16 |
| yes_no_required | l22_peak_layer_out | top1 | 32 | 0.469 | 0.500 | 3.38 | 1.014 | correct_prefix:16, space:12, newline:4 | first_token_competition_failure:16, exact_correct:15, correct_prefix_but_generation_wrong:1 |
| yes_no_required | late_peak_layer_out | top1 | 32 | 0.469 | 0.500 | 3.38 | 1.014 | correct_prefix:16, space:12, newline:4 | first_token_competition_failure:16, exact_correct:15, correct_prefix_but_generation_wrong:1 |

### Norm-Neutralized Pair Readout

| pair_task | top1_category | n | top1_text | actual_gap | neutral_cos_gap | neutral_flip_rate | correct_cos | competitor_cos | norm_adv | needed_delta |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | space | 32 |  :32 | 1.453 | -0.0133 | 1.000 | 0.0871 | 0.0738 | 0.2603 | 1.0942 |
| yes_no_required | space | 24 |  :24 | 1.750 | -0.0132 | 0.917 | 0.0905 | 0.0772 | 0.2603 | 1.3177 |
| yes_no_required | newline | 8 |  ?\n\n:8 | 2.859 | 0.0197 | 0.000 | 0.0832 | 0.1029 | -0.0408 | 2.2180 |

### Direction Correction by Scale

| pair_task | top1_category | scale | n | correct_top1_rate | mean_rank | mean_gap | top1_after |
|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | space | 0.5 | 32 | 0.000 | 3.25 | 0.777 | space:28, newline:4 |
| explanation_required | space | 1.0 | 32 | 0.188 | 1.94 | 0.312 | space:14, newline:12, correct_prefix:6 |
| explanation_required | space | 1.5 | 32 | 0.625 | 1.44 | 0.176 | correct_prefix:20, newline:10, space:2 |
| explanation_required | space | 2.0 | 32 | 0.750 | 1.19 | 0.086 | correct_prefix:24, newline:6, space:2 |
| yes_no_required | newline | 0.5 | 8 | 0.000 | 5.25 | 1.812 | newline:4, space:4 |
| yes_no_required | newline | 1.0 | 8 | 0.000 | 2.75 | 1.406 | space:8 |
| yes_no_required | newline | 1.5 | 8 | 0.000 | 2.00 | 1.000 | space:8 |
| yes_no_required | newline | 2.0 | 8 | 0.250 | 1.75 | 0.688 | space:6, correct_prefix:2 |
| yes_no_required | space | 0.5 | 24 | 0.000 | 3.75 | 1.208 | space:12, newline:12 |
| yes_no_required | space | 1.0 | 24 | 0.000 | 2.25 | 0.792 | newline:18, space:6 |
| yes_no_required | space | 1.5 | 24 | 0.417 | 1.75 | 0.625 | newline:14, correct_prefix:10 |
| yes_no_required | space | 2.0 | 24 | 0.417 | 1.67 | 0.448 | newline:14, correct_prefix:10 |

### Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| yes_no_required | l22_peak_layer_out | top1 | 1 |  48.\n\nQuestion::1 |
| yes_no_required | late_peak_layer_out | top1 | 1 |  48.\n\nQuestion::1 |
