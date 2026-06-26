# Phase 662 Cross-Model Summary

目标：固定 Phase 661 partially repaired state，审计剩余失败是否来自 final_norm output direction 不足或 lm_head/unembedding projection advantage。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 160 / total_time_min: 0.75
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top2', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}, {'layer': 20, 'component': 'attn_out', 'phase656_dtop': 1.53125, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.9500000000000002}], 'phase658_delta_exact': 14, 'phase658_delta_tok0': 15, 'phase658_rank_improvement': 2.1}, {'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top1', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}], 'phase658_delta_exact': 8, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 1.7000000000000002}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 11, 'phase658_rank_improvement': 3.0000000000000004}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top3', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}, {'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 1.4666666666666666, 'phase657_delta_exact': 2, 'phase657_delta_tok0': 3, 'phase657_rank_improvement': 0.7000000000000002}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 2.5500000000000003}]`
- last_writer_map: `{"('explanation_required', 'separator_input_edge', 'top1')": [{'layer': 35, 'component': 'mlp_out', 'phase660_gap_delta': 0.171875, 'phase660_rank_delta': 0.1499999999999999}, {'layer': 32, 'component': 'mlp_out', 'phase660_gap_delta': 0.1375, 'phase660_rank_delta': 0.1499999999999999}], "('yes_no_required', 'early_peak_layer_out', 'top3')": [{'layer': 35, 'component': 'mlp_out', 'phase660_gap_delta': 0.14687499999999998, 'phase660_rank_delta': 0.30000000000000004}, {'layer': 32, 'component': 'mlp_out', 'phase660_gap_delta': 0.10625, 'phase660_rank_delta': 0.15000000000000013}], "('yes_no_required', 'early_peak_layer_out', 'top2')": [{'layer': 35, 'component': 'mlp_out', 'phase660_gap_delta': 0.06875, 'phase660_rank_delta': 0.09999999999999987}, {'layer': 32, 'component': 'mlp_out', 'phase660_gap_delta': 0.03125, 'phase660_rank_delta': 0.04999999999999982}]}`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### By Mode

| pair_task | site | combo | mode | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top1 | phase658_combo | 20 | 0.600 | 0.700 | 1.40 | 0.294 | correct_prefix:14, space:6 |
| explanation_required | separator_input_edge | top1 | plus_last_writers | 20 | 0.800 | 0.950 | 1.10 | 0.031 | correct_prefix:19, word:1 |
| explanation_required | separator_input_edge | top2 | phase658_combo | 20 | 0.900 | 1.000 | 1.00 | 0.000 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | plus_last_writers | 20 | 0.900 | 1.000 | 1.00 | 0.000 | correct_prefix:20 |
| yes_no_required | early_peak_layer_out | top2 | phase658_combo | 20 | 0.800 | 0.800 | 1.15 | 0.081 | correct_prefix:16, space:3, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | plus_last_writers | 20 | 1.000 | 1.000 | 1.00 | 0.000 | correct_prefix:20 |
| yes_no_required | early_peak_layer_out | top3 | phase658_combo | 20 | 0.800 | 0.700 | 1.60 | 0.212 | correct_prefix:14, newline:5, explanation:1 |
| yes_no_required | early_peak_layer_out | top3 | plus_last_writers | 20 | 0.950 | 0.950 | 1.05 | 0.009 | correct_prefix:19, newline:1 |

### Plus-Last-Writers Remaining Failure Projection

| pair_task | top1_category | n | top1_text | post_gap | pre_gap | norm_gap_change | needed_unit_delta | diff_alignment | correct_cos | competitor_cos | competitor_norm_advantage |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | correct_prefix | 5 |  v:5 | 0.000 | 36.275 | -36.275 | 0.0000 | 0.0000 | 0.1296 | 0.1296 | 0.0000 |
| explanation_required | word | 1 |  o:1 | 0.625 | 56.562 | -55.938 | 0.4741 | 0.0053 | 0.0977 | 0.1061 | -0.0166 |
| yes_no_required | newline | 1 |  \n\n:1 | 0.188 | 62.938 | -62.750 | 0.1276 | 0.0017 | 0.0937 | 0.0924 | 0.0415 |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 160 / total_time_min: 1.10
- combo_specs: `[{'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}]`
- last_writer_map: `{"('explanation_required', 'l22_peak_layer_out', 'top1')": [{'layer': 36, 'component': 'attn_out', 'phase660_gap_delta': 0.184375, 'phase660_rank_delta': 0.34999999999999987}, {'layer': 36, 'component': 'mlp_out', 'phase660_gap_delta': 0.128125, 'phase660_rank_delta': 0.34999999999999987}], "('explanation_required', 'late_peak_layer_out', 'top1')": [{'layer': 36, 'component': 'attn_out', 'phase660_gap_delta': 0.184375, 'phase660_rank_delta': 0.34999999999999987}, {'layer': 36, 'component': 'mlp_out', 'phase660_gap_delta': 0.128125, 'phase660_rank_delta': 0.34999999999999987}], "('yes_no_required', 'l22_peak_layer_out', 'top2')": [{'layer': 36, 'component': 'attn_out', 'phase660_gap_delta': 0.14375000000000004, 'phase660_rank_delta': 0.25}, {'layer': 36, 'component': 'mlp_out', 'phase660_gap_delta': 0.125, 'phase660_rank_delta': 0.3999999999999999}], "('yes_no_required', 'late_peak_layer_out', 'top2')": [{'layer': 36, 'component': 'attn_out', 'phase660_gap_delta': 0.14375000000000004, 'phase660_rank_delta': 0.25}, {'layer': 36, 'component': 'mlp_out', 'phase660_gap_delta': 0.125, 'phase660_rank_delta': 0.3999999999999999}]}`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### By Mode

| pair_task | site | combo | mode | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top1 | phase658_combo | 20 | 0.300 | 0.350 | 1.90 | 0.453 | space:11, correct_prefix:7, word:2 |
| explanation_required | l22_peak_layer_out | top1 | plus_last_writers | 20 | 0.700 | 0.750 | 1.40 | 0.206 | correct_prefix:15, space:5 |
| explanation_required | late_peak_layer_out | top1 | phase658_combo | 20 | 0.300 | 0.350 | 1.90 | 0.453 | space:11, correct_prefix:7, word:2 |
| explanation_required | late_peak_layer_out | top1 | plus_last_writers | 20 | 0.600 | 0.750 | 1.40 | 0.206 | correct_prefix:15, space:5 |
| yes_no_required | l22_peak_layer_out | top2 | phase658_combo | 20 | 0.400 | 0.400 | 2.00 | 0.431 | space:10, correct_prefix:8, word:2 |
| yes_no_required | l22_peak_layer_out | top2 | plus_last_writers | 20 | 0.650 | 0.650 | 1.50 | 0.219 | correct_prefix:13, space:5, word:2 |
| yes_no_required | late_peak_layer_out | top2 | phase658_combo | 20 | 0.400 | 0.400 | 2.00 | 0.431 | space:10, correct_prefix:8, word:2 |
| yes_no_required | late_peak_layer_out | top2 | plus_last_writers | 20 | 0.650 | 0.650 | 1.50 | 0.219 | correct_prefix:13, space:5, word:2 |

### Plus-Last-Writers Remaining Failure Projection

| pair_task | top1_category | n | top1_text | post_gap | pre_gap | norm_gap_change | needed_unit_delta | diff_alignment | correct_cos | competitor_cos | competitor_norm_advantage |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | space | 10 |  :10 | 0.825 | 25.646 | -24.821 | 0.8806 | 0.0049 | 0.0872 | 0.0940 | -0.0045 |
| yes_no_required | space | 10 |  :10 | 0.787 | 27.910 | -27.123 | 0.8406 | 0.0048 | 0.0897 | 0.0964 | -0.0045 |
| explanation_required | correct_prefix | 4 |  v:4 | 0.000 | 24.748 | -24.748 | 0.0000 | 0.0000 | 0.0936 | 0.0936 | 0.0000 |
| yes_no_required | word | 4 |  o:4 | 0.219 | 29.830 | -29.611 | 0.2437 | 0.0014 | 0.1042 | 0.1188 | -0.0808 |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 160 / total_time_min: 0.94
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}]`
- last_writer_map: `{"('yes_no_required', 'l22_peak_layer_out', 'top1')": [{'layer': 25, 'component': 'attn_out', 'phase660_gap_delta': 0.3531249999999999, 'phase660_rank_delta': 2.5}, {'layer': 26, 'component': 'mlp_out', 'phase660_gap_delta': 0.2906249999999999, 'phase660_rank_delta': 1.0}], "('yes_no_required', 'late_peak_layer_out', 'top1')": [{'layer': 25, 'component': 'attn_out', 'phase660_gap_delta': 0.3531249999999999, 'phase660_rank_delta': 2.5}, {'layer': 26, 'component': 'mlp_out', 'phase660_gap_delta': 0.2906249999999999, 'phase660_rank_delta': 1.0}], "('explanation_required', 'l22_peak_layer_out', 'top2')": [{'layer': 26, 'component': 'mlp_out', 'phase660_gap_delta': 0.171875, 'phase660_rank_delta': 0.6499999999999999}], "('explanation_required', 'late_peak_layer_out', 'top2')": [{'layer': 26, 'component': 'mlp_out', 'phase660_gap_delta': 0.171875, 'phase660_rank_delta': 0.6499999999999999}]}`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### By Mode

| pair_task | site | combo | mode | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | top1_category |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top2 | phase658_combo | 20 | 0.150 | 0.150 | 3.65 | 0.978 | space:15, correct_prefix:3, newline:2 |
| explanation_required | l22_peak_layer_out | top2 | plus_last_writers | 20 | 0.450 | 0.450 | 3.00 | 0.806 | space:11, correct_prefix:9 |
| explanation_required | late_peak_layer_out | top2 | phase658_combo | 20 | 0.150 | 0.150 | 3.65 | 0.978 | space:15, correct_prefix:3, newline:2 |
| explanation_required | late_peak_layer_out | top2 | plus_last_writers | 20 | 0.450 | 0.450 | 3.00 | 0.806 | space:11, correct_prefix:9 |
| yes_no_required | l22_peak_layer_out | top1 | phase658_combo | 20 | 0.000 | 0.000 | 9.60 | 2.306 | newline:16, space:4 |
| yes_no_required | l22_peak_layer_out | top1 | plus_last_writers | 20 | 0.450 | 0.450 | 4.05 | 1.166 | correct_prefix:9, space:7, newline:4 |
| yes_no_required | late_peak_layer_out | top1 | phase658_combo | 20 | 0.000 | 0.000 | 9.60 | 2.306 | newline:16, space:4 |
| yes_no_required | late_peak_layer_out | top1 | plus_last_writers | 20 | 0.450 | 0.450 | 4.05 | 1.166 | correct_prefix:9, space:7, newline:4 |

### Plus-Last-Writers Remaining Failure Projection

| pair_task | top1_category | n | top1_text | post_gap | pre_gap | norm_gap_change | needed_unit_delta | diff_alignment | correct_cos | competitor_cos | competitor_norm_advantage |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| explanation_required | space | 22 |  :22 | 1.466 | 385.670 | -384.205 | 1.1038 | 0.0054 | 0.0863 | 0.0733 | 0.2603 |
| yes_no_required | space | 14 |  :14 | 1.696 | 395.143 | -393.446 | 1.2774 | 0.0058 | 0.0903 | 0.0769 | 0.2603 |
| yes_no_required | newline | 8 |  ?\n\n:8 | 2.859 | 454.562 | -451.703 | 2.2180 | 0.0107 | 0.0832 | 0.1029 | -0.0408 |
