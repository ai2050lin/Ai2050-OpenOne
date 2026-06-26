# Phase 659 Cross-Model Summary

目标：固定 Phase 658 best combo，比较 baseline_task / site_restore / combo_ablation 下 correct_prefix 与剩余 top1 competitor 的距离，定位最后 top1 barrier。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 0.37
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top2', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}, {'layer': 20, 'component': 'attn_out', 'phase656_dtop': 1.53125, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.9500000000000002}], 'phase658_delta_exact': 14, 'phase658_delta_tok0': 15, 'phase658_rank_improvement': 2.1}, {'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top1', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}], 'phase658_delta_exact': 8, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 1.7000000000000002}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 11, 'phase658_rank_improvement': 3.0000000000000004}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top3', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}, {'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 1.4666666666666666, 'phase657_delta_exact': 2, 'phase657_delta_tok0': 3, 'phase657_rank_improvement': 0.7000000000000002}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 2.5500000000000003}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Barrier Effects

| pair_task | site | combo | components | n | rank site->combo | rank_improvement | gap site->combo | gap_reduction | correct_top1 site->combo | delta | site_top1_category | combo_top1_category | combo_top1_text |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | separator_input_edge | top2 | L16 attn_out, L20 attn_out | 20 | 3.10->1.00 | 2.10 | 1.137->0.000 | 1.137 | 4->20 | 16 | word:12, space:4, correct_prefix:4 | correct_prefix:20 |  v:20 |
| yes_no_required | early_peak_layer_out | top2 | L18 attn_out, L21 mlp_out | 20 | 4.15->1.15 | 3.00 | 2.219->0.081 | 2.138 | 5->17 | 12 | space:15, correct_prefix:5 | correct_prefix:17, space:2, newline:1 |  v:17,  :2,  \n\n:1 |
| explanation_required | separator_input_edge | top1 | L16 attn_out | 20 | 3.10->1.40 | 1.70 | 1.137->0.294 | 0.844 | 4->14 | 10 | word:12, space:4, correct_prefix:4 | correct_prefix:14, space:6 |  v:14,  :6 |
| yes_no_required | early_peak_layer_out | top3 | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 4.15->1.60 | 2.55 | 2.219->0.212 | 2.006 | 5->14 | 9 | space:15, correct_prefix:5 | correct_prefix:14, newline:5, explanation:1 |  v:14,  \n\n:5,  Yes:1 |

### By Mode

| pair_task | site | combo | mode | components | n | mean_rank | mean_top1_gap | correct_top1_rate | top1_category | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| explanation_required | separator_input_edge | top1 | baseline_task | L16 attn_out | 20 | 1.80 | 0.613 | 0.500 | word:10, correct_prefix:10 |  The:10,  v:10 |
| explanation_required | separator_input_edge | top1 | combo_ablation | L16 attn_out | 20 | 1.40 | 0.294 | 0.700 | correct_prefix:14, space:6 |  v:14,  :6 |
| explanation_required | separator_input_edge | top1 | site_restore | L16 attn_out | 20 | 3.10 | 1.137 | 0.200 | word:12, space:4, correct_prefix:4 |  The:12,  :4,  v:4 |
| explanation_required | separator_input_edge | top2 | baseline_task | L16 attn_out, L20 attn_out | 20 | 1.80 | 0.613 | 0.500 | word:10, correct_prefix:10 |  The:10,  v:10 |
| explanation_required | separator_input_edge | top2 | combo_ablation | L16 attn_out, L20 attn_out | 20 | 1.00 | 0.000 | 1.000 | correct_prefix:20 |  v:20 |
| explanation_required | separator_input_edge | top2 | site_restore | L16 attn_out, L20 attn_out | 20 | 3.10 | 1.137 | 0.200 | word:12, space:4, correct_prefix:4 |  The:12,  :4,  v:4 |
| yes_no_required | early_peak_layer_out | top2 | baseline_task | L18 attn_out, L21 mlp_out | 20 | 12.95 | 2.922 | 0.000 | explanation:20 |  Yes:20 |
| yes_no_required | early_peak_layer_out | top2 | combo_ablation | L18 attn_out, L21 mlp_out | 20 | 1.15 | 0.081 | 0.850 | correct_prefix:17, space:2, newline:1 |  v:17,  :2,  \n\n:1 |
| yes_no_required | early_peak_layer_out | top2 | site_restore | L18 attn_out, L21 mlp_out | 20 | 4.15 | 2.219 | 0.250 | space:15, correct_prefix:5 |  :15,  v:5 |
| yes_no_required | early_peak_layer_out | top3 | baseline_task | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 12.95 | 2.922 | 0.000 | explanation:20 |  Yes:20 |
| yes_no_required | early_peak_layer_out | top3 | combo_ablation | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 1.60 | 0.212 | 0.700 | correct_prefix:14, newline:5, explanation:1 |  v:14,  \n\n:5,  Yes:1 |
| yes_no_required | early_peak_layer_out | top3 | site_restore | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 4.15 | 2.219 | 0.250 | space:15, correct_prefix:5 |  :15,  v:5 |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 0.54
- combo_specs: `[{'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Barrier Effects

| pair_task | site | combo | components | n | rank site->combo | rank_improvement | gap site->combo | gap_reduction | correct_top1 site->combo | delta | site_top1_category | combo_top1_category | combo_top1_text |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| yes_no_required | l22_peak_layer_out | top2 | L27 attn_out, L23 attn_out | 20 | 3.45->2.00 | 1.45 | 1.134->0.431 | 0.703 | 3->8 | 5 | space:15, correct_prefix:3, explanation:2 | space:11, correct_prefix:8, word:1 |  :11,  v:8,  o:1 |
| yes_no_required | late_peak_layer_out | top2 | L27 attn_out, L23 attn_out | 20 | 3.45->2.00 | 1.45 | 1.134->0.431 | 0.703 | 3->8 | 5 | space:15, correct_prefix:3, explanation:2 | space:11, correct_prefix:8, word:1 |  :11,  v:8,  o:1 |
| explanation_required | l22_peak_layer_out | top1 | L23 attn_out | 20 | 2.15->1.90 | 0.25 | 0.803->0.453 | 0.350 | 5->7 | 2 | space:15, correct_prefix:5 | space:11, correct_prefix:7, word:2 |  :11,  v:7,  o:2 |
| explanation_required | late_peak_layer_out | top1 | L23 attn_out | 20 | 2.15->1.90 | 0.25 | 0.803->0.453 | 0.350 | 5->7 | 2 | space:15, correct_prefix:5 | space:11, correct_prefix:7, word:2 |  :11,  v:7,  o:2 |

### By Mode

| pair_task | site | combo | mode | components | n | mean_rank | mean_top1_gap | correct_top1_rate | top1_category | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top1 | baseline_task | L23 attn_out | 20 | 58.30 | 5.239 | 0.000 | word:20 |  The:20 |
| explanation_required | l22_peak_layer_out | top1 | combo_ablation | L23 attn_out | 20 | 1.90 | 0.453 | 0.350 | space:11, correct_prefix:7, word:2 |  :11,  v:7,  o:2 |
| explanation_required | l22_peak_layer_out | top1 | site_restore | L23 attn_out | 20 | 2.15 | 0.803 | 0.250 | space:15, correct_prefix:5 |  :15,  v:5 |
| explanation_required | late_peak_layer_out | top1 | baseline_task | L23 attn_out | 20 | 58.30 | 5.239 | 0.000 | word:20 |  The:20 |
| explanation_required | late_peak_layer_out | top1 | combo_ablation | L23 attn_out | 20 | 1.90 | 0.453 | 0.350 | space:11, correct_prefix:7, word:2 |  :11,  v:7,  o:2 |
| explanation_required | late_peak_layer_out | top1 | site_restore | L23 attn_out | 20 | 2.15 | 0.803 | 0.250 | space:15, correct_prefix:5 |  :15,  v:5 |
| yes_no_required | l22_peak_layer_out | top2 | baseline_task | L27 attn_out, L23 attn_out | 20 | 188.10 | 9.116 | 0.000 | explanation:20 |  yes:10,  Yes:10 |
| yes_no_required | l22_peak_layer_out | top2 | combo_ablation | L27 attn_out, L23 attn_out | 20 | 2.00 | 0.431 | 0.400 | space:11, correct_prefix:8, word:1 |  :11,  v:8,  o:1 |
| yes_no_required | l22_peak_layer_out | top2 | site_restore | L27 attn_out, L23 attn_out | 20 | 3.45 | 1.134 | 0.150 | space:15, correct_prefix:3, explanation:2 |  :15,  v:3,  no:1,  yes:1 |
| yes_no_required | late_peak_layer_out | top2 | baseline_task | L27 attn_out, L23 attn_out | 20 | 188.10 | 9.116 | 0.000 | explanation:20 |  yes:10,  Yes:10 |
| yes_no_required | late_peak_layer_out | top2 | combo_ablation | L27 attn_out, L23 attn_out | 20 | 2.00 | 0.431 | 0.400 | space:11, correct_prefix:8, word:1 |  :11,  v:8,  o:1 |
| yes_no_required | late_peak_layer_out | top2 | site_restore | L27 attn_out, L23 attn_out | 20 | 3.45 | 1.134 | 0.150 | space:15, correct_prefix:3, explanation:2 |  :15,  v:3,  no:1,  yes:1 |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 0.47
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Barrier Effects

| pair_task | site | combo | components | n | rank site->combo | rank_improvement | gap site->combo | gap_reduction | correct_top1 site->combo | delta | site_top1_category | combo_top1_category | combo_top1_text |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | l22_peak_layer_out | top2 | L23 mlp_out, L24 mlp_out | 20 | 8.30->3.65 | 4.65 | 2.194->0.978 | 1.216 | 2->3 | 1 | newline:10, space:8, correct_prefix:2 | space:15, correct_prefix:3, newline:2 |  :15,  v:3,  ?\n\n:2 |
| explanation_required | late_peak_layer_out | top2 | L23 mlp_out, L24 mlp_out | 20 | 8.30->3.65 | 4.65 | 2.194->0.978 | 1.216 | 2->3 | 1 | newline:10, space:8, correct_prefix:2 | space:15, correct_prefix:3, newline:2 |  :15,  v:3,  ?\n\n:2 |
| yes_no_required | l22_peak_layer_out | top1 | L24 mlp_out | 20 | 13.85->9.60 | 4.25 | 2.812->2.306 | 0.506 | 0->0 | 0 | newline:11, space:9 | newline:15, space:5 |  ?\n\n:15,  :5 |
| yes_no_required | late_peak_layer_out | top1 | L24 mlp_out | 20 | 13.85->9.60 | 4.25 | 2.812->2.306 | 0.506 | 0->0 | 0 | newline:11, space:9 | newline:15, space:5 |  ?\n\n:15,  :5 |

### By Mode

| pair_task | site | combo | mode | components | n | mean_rank | mean_top1_gap | correct_top1_rate | top1_category | top1_text |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top2 | baseline_task | L23 mlp_out, L24 mlp_out | 20 | 77.20 | 5.575 | 0.000 | word:11, newline:9 |  c:10,  ?\n\n:5,  \n\n:4,  The:1 |
| explanation_required | l22_peak_layer_out | top2 | combo_ablation | L23 mlp_out, L24 mlp_out | 20 | 3.65 | 0.978 | 0.150 | space:15, correct_prefix:3, newline:2 |  :15,  v:3,  ?\n\n:2 |
| explanation_required | l22_peak_layer_out | top2 | site_restore | L23 mlp_out, L24 mlp_out | 20 | 8.30 | 2.194 | 0.100 | newline:10, space:8, correct_prefix:2 |  ?\n\n:10,  :8,  v:2 |
| explanation_required | late_peak_layer_out | top2 | baseline_task | L23 mlp_out, L24 mlp_out | 20 | 77.20 | 5.575 | 0.000 | word:11, newline:9 |  c:10,  ?\n\n:5,  \n\n:4,  The:1 |
| explanation_required | late_peak_layer_out | top2 | combo_ablation | L23 mlp_out, L24 mlp_out | 20 | 3.65 | 0.978 | 0.150 | space:15, correct_prefix:3, newline:2 |  :15,  v:3,  ?\n\n:2 |
| explanation_required | late_peak_layer_out | top2 | site_restore | L23 mlp_out, L24 mlp_out | 20 | 8.30 | 2.194 | 0.100 | newline:10, space:8, correct_prefix:2 |  ?\n\n:10,  :8,  v:2 |
| yes_no_required | l22_peak_layer_out | top1 | baseline_task | L24 mlp_out | 20 | 295.55 | 9.172 | 0.000 | explanation:20 |  yes:20 |
| yes_no_required | l22_peak_layer_out | top1 | combo_ablation | L24 mlp_out | 20 | 9.60 | 2.306 | 0.000 | newline:15, space:5 |  ?\n\n:15,  :5 |
| yes_no_required | l22_peak_layer_out | top1 | site_restore | L24 mlp_out | 20 | 13.85 | 2.812 | 0.000 | newline:11, space:9 |  ?\n\n:11,  :9 |
| yes_no_required | late_peak_layer_out | top1 | baseline_task | L24 mlp_out | 20 | 295.55 | 9.172 | 0.000 | explanation:20 |  yes:20 |
| yes_no_required | late_peak_layer_out | top1 | combo_ablation | L24 mlp_out | 20 | 9.60 | 2.306 | 0.000 | newline:15, space:5 |  ?\n\n:15,  :5 |
| yes_no_required | late_peak_layer_out | top1 | site_restore | L24 mlp_out | 20 | 13.85 | 2.812 | 0.000 | newline:11, space:9 |  ?\n\n:11,  :9 |
