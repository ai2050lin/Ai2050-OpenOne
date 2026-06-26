# Phase 657 Cross-Model Summary

目标：读取 Phase 656 的格式先验写入候选，在固定 intent-gate restore patch 后，对候选组件做最终位置消融，并用短贪婪生成验证 margin-level 候选是否进入 generation-level。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 0.97
- max_cases: 20 / max_new_tokens: 6
- candidate_specs: `[{'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'layer': 21, 'component': 'mlp_out', 'baseline_top0_category': 'space', 'phase656_n': 15, 'phase656_dtop': 1.8916666666666666, 'phase656_drank': 2.466666666666667, 'phase656_flip': 5}, {'pair_task': 'explanation_required', 'site': 'early_peak_layer_out', 'layer': 21, 'component': 'mlp_out', 'baseline_top0_category': 'space', 'phase656_n': 11, 'phase656_dtop': 1.75, 'phase656_drank': 1.4545454545454546, 'phase656_flip': 2}, {'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'layer': 20, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 4, 'phase656_dtop': 1.53125, 'phase656_drank': 1.75, 'phase656_flip': 0}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'layer': 18, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 15, 'phase656_dtop': 1.5, 'phase656_drank': 2.6666666666666665, 'phase656_flip': 5}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'layer': 23, 'component': 'mlp_out', 'baseline_top0_category': 'space', 'phase656_n': 15, 'phase656_dtop': 1.4666666666666666, 'phase656_drank': 0.9333333333333333, 'phase656_flip': 3}, {'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'layer': 16, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 4, 'phase656_dtop': 1.4375, 'phase656_drank': 3.0, 'phase656_flip': 1}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Generation Effects

| pair_task | site | layer | component | n | phase656_dtop | exact base->ablate | delta_exact | tok0 base->ablate | delta_tok0 | rank base->ablate | rank_improvement | base_top0 | ablation_top0 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | separator_input_edge | L16 | attn_out | 20 | 1.438 | 4->12 | 8 | 5->14 | 9 | 3.10->1.40 | 1.70 | explanation:11, correct_prefix:5, space:4 | correct_prefix:14, space:6 |
| yes_no_required | early_peak_layer_out | L18 | attn_out | 20 | 1.500 | 5->10 | 5 | 5->10 | 5 | 4.15->2.15 | 2.00 | space:15, correct_prefix:5 | space:10, correct_prefix:10 |
| yes_no_required | early_peak_layer_out | L21 | mlp_out | 20 | 1.892 | 5->9 | 4 | 5->10 | 5 | 4.15->2.30 | 1.85 | space:15, correct_prefix:5 | correct_prefix:10, space:8, newline:2 |
| yes_no_required | early_peak_layer_out | L23 | mlp_out | 20 | 1.467 | 5->7 | 2 | 5->8 | 3 | 4.15->3.45 | 0.70 | space:15, correct_prefix:5 | space:10, correct_prefix:8, newline:2 |
| explanation_required | early_peak_layer_out | L21 | mlp_out | 20 | 1.750 | 7->9 | 2 | 8->10 | 2 | 4.35->3.65 | 0.70 | space:11, correct_prefix:8, newline:1 | correct_prefix:10, space:7, newline:3 |
| explanation_required | separator_input_edge | L20 | attn_out | 20 | 1.531 | 4->5 | 1 | 5->7 | 2 | 3.10->2.15 | 0.95 | explanation:11, correct_prefix:5, space:4 | explanation:12, correct_prefix:7, word:1 |

### By Mode

| pair_task | site | layer | component | mode | n | phase656_dtop | exact_rate | tok0_rate | mean_rank | top0_category | generation_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | early_peak_layer_out | L21 | mlp_out | candidate_ablation | 20 | 1.750 | 0.450 | 0.500 | 3.65 | correct_prefix:10, space:7, newline:3 |  v48\n\nOkay,:4;  22\n\nOkay,:3;  v48\nReason::2;  05\nOkay,:2 |
| explanation_required | early_peak_layer_out | L21 | mlp_out | site_restore | 20 | 1.750 | 0.350 | 0.400 | 4.35 | space:11, correct_prefix:8, newline:1 |  22\n\nOkay,:3;  48\nOkay,:3;  v48\n\nOkay,:3;  48\n\nThe answer:2 |
| explanation_required | separator_input_edge | L16 | attn_out | candidate_ablation | 20 | 1.438 | 0.600 | 0.700 | 1.40 | correct_prefix:14, space:6 |  v22\n\nOkay,:3;  v48\n\nThe question:3;  v48\n\nOkay,:3;  05\n\nThe answer:2 |
| explanation_required | separator_input_edge | L16 | attn_out | site_restore | 20 | 1.438 | 0.200 | 0.250 | 3.10 | explanation:11, correct_prefix:5, space:4 |  The value is 48:5;  The value is 22:4;  v48\n\nOkay,:2;  05\n\nThe answer:2 |
| explanation_required | separator_input_edge | L20 | attn_out | candidate_ablation | 20 | 1.531 | 0.250 | 0.350 | 2.15 | explanation:12, correct_prefix:7, word:1 |  The value is 22:4;  The value is 48:4;  v48\n\nThe question:2;  The value is 05:2 |
| explanation_required | separator_input_edge | L20 | attn_out | site_restore | 20 | 1.531 | 0.200 | 0.250 | 3.10 | explanation:11, correct_prefix:5, space:4 |  The value is 48:5;  The value is 22:4;  v48\n\nOkay,:2;  05\n\nThe answer:2 |
| yes_no_required | early_peak_layer_out | L18 | attn_out | candidate_ablation | 20 | 1.500 | 0.500 | 0.500 | 2.15 | space:10, correct_prefix:10 |  v48\n\nOkay,:6;  22\nOkay,:4;  v48\nOkay,:3;  48\nOkay,:2 |
| yes_no_required | early_peak_layer_out | L18 | attn_out | site_restore | 20 | 1.500 | 0.250 | 0.250 | 4.15 | space:15, correct_prefix:5 |  48\nOkay,:7;  22\nOkay,:4;  v48\n\nOkay,:2;  v48\nOkay,:2 |
| yes_no_required | early_peak_layer_out | L21 | mlp_out | candidate_ablation | 20 | 1.892 | 0.450 | 0.500 | 2.30 | correct_prefix:10, space:8, newline:2 |  v48\n\nOkay,:5;  22\nOkay,:4;  48\nOkay,:3;  v48\nOkay,:3 |
| yes_no_required | early_peak_layer_out | L21 | mlp_out | site_restore | 20 | 1.892 | 0.250 | 0.250 | 4.15 | space:15, correct_prefix:5 |  48\nOkay,:7;  22\nOkay,:4;  v48\n\nOkay,:2;  v48\nOkay,:2 |
| yes_no_required | early_peak_layer_out | L23 | mlp_out | candidate_ablation | 20 | 1.467 | 0.350 | 0.400 | 3.45 | space:10, correct_prefix:8, newline:2 |  22\nOkay,:4;  48\nOkay,:4;  v48\n\nOkay,:4;  \n\nOkay, let's try:2 |
| yes_no_required | early_peak_layer_out | L23 | mlp_out | site_restore | 20 | 1.467 | 0.250 | 0.250 | 4.15 | space:15, correct_prefix:5 |  48\nOkay,:7;  22\nOkay,:4;  v48\n\nOkay,:2;  v48\nOkay,:2 |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 1.42
- max_cases: 20 / max_new_tokens: 6
- candidate_specs: `[{'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'layer': 27, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 14, 'phase656_dtop': 0.6339285714285714, 'phase656_drank': 1.0, 'phase656_flip': 1}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'layer': 27, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 14, 'phase656_dtop': 0.6339285714285714, 'phase656_drank': 1.0, 'phase656_flip': 1}, {'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'layer': 23, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 14, 'phase656_dtop': 0.5044642857142857, 'phase656_drank': 0.8571428571428571, 'phase656_flip': 1}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'layer': 23, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 14, 'phase656_dtop': 0.5044642857142857, 'phase656_drank': 0.8571428571428571, 'phase656_flip': 1}, {'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'layer': 23, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 15, 'phase656_dtop': 0.4875, 'phase656_drank': 0.4, 'phase656_flip': 3}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'layer': 23, 'component': 'attn_out', 'baseline_top0_category': 'space', 'phase656_n': 15, 'phase656_dtop': 0.4875, 'phase656_drank': 0.4, 'phase656_flip': 3}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Generation Effects

| pair_task | site | layer | component | n | phase656_dtop | exact base->ablate | delta_exact | tok0 base->ablate | delta_tok0 | rank base->ablate | rank_improvement | base_top0 | ablation_top0 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | L23 | attn_out | 20 | 0.487 | 5->6 | 1 | 5->7 | 2 | 2.15->1.90 | 0.25 | space:15, correct_prefix:5 | space:11, correct_prefix:7, word:2 |
| explanation_required | late_peak_layer_out | L23 | attn_out | 20 | 0.487 | 5->6 | 1 | 5->7 | 2 | 2.15->1.90 | 0.25 | space:15, correct_prefix:5 | space:11, correct_prefix:7, word:2 |
| yes_no_required | l22_peak_layer_out | L27 | attn_out | 20 | 0.634 | 3->4 | 1 | 3->4 | 1 | 3.45->2.60 | 0.85 | space:14, correct_prefix:3, explanation:2, word:1 | space:13, correct_prefix:4, explanation:2, word:1 |
| yes_no_required | late_peak_layer_out | L27 | attn_out | 20 | 0.634 | 3->4 | 1 | 3->4 | 1 | 3.45->2.60 | 0.85 | space:14, correct_prefix:3, explanation:2, word:1 | space:13, correct_prefix:4, explanation:2, word:1 |
| yes_no_required | l22_peak_layer_out | L23 | attn_out | 20 | 0.504 | 3->4 | 1 | 3->4 | 1 | 3.45->2.65 | 0.80 | space:14, correct_prefix:3, explanation:2, word:1 | space:15, correct_prefix:4, word:1 |
| yes_no_required | late_peak_layer_out | L23 | attn_out | 20 | 0.504 | 3->4 | 1 | 3->4 | 1 | 3.45->2.65 | 0.80 | space:14, correct_prefix:3, explanation:2, word:1 | space:15, correct_prefix:4, word:1 |

### By Mode

| pair_task | site | layer | component | mode | n | phase656_dtop | exact_rate | tok0_rate | mean_rank | top0_category | generation_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | L23 | attn_out | candidate_ablation | 20 | 0.487 | 0.300 | 0.350 | 1.90 | space:11, correct_prefix:7, word:2 |  v48.\n\nReason: According:4;  22\n\nReason: The:2;  48\n\nReason: The:2;  0\n\nReason: According:2 |
| explanation_required | l22_peak_layer_out | L23 | attn_out | site_restore | 20 | 0.487 | 0.250 | 0.250 | 2.15 | space:15, correct_prefix:5 |  22\n\nReason: According:3;  22\n\nReason: The:2;  48\n\nReason: The:2;  48.\n\nReason: The:2 |
| explanation_required | late_peak_layer_out | L23 | attn_out | candidate_ablation | 20 | 0.487 | 0.300 | 0.350 | 1.90 | space:11, correct_prefix:7, word:2 |  v48\n\nReason: According:5;  48\n\nReason: According:4;  22\n\nReason: The:2;  48\n\nReason: The:2 |
| explanation_required | late_peak_layer_out | L23 | attn_out | site_restore | 20 | 0.487 | 0.250 | 0.250 | 2.15 | space:15, correct_prefix:5 |  48\n\nReason: According:5;  v48\n\nReason: According:3;  22\n\nReason: According:3;  22\n\nReason: The:2 |
| yes_no_required | l22_peak_layer_out | L23 | attn_out | candidate_ablation | 20 | 0.504 | 0.200 | 0.200 | 2.65 | space:15, correct_prefix:4, word:1 |  48.\n\nQuestion: c:5;  22.\n\nQuestion: c:4;  v48.\n\nQuestion: c:3;  48.\n\nc12 r:2 |
| yes_no_required | l22_peak_layer_out | L23 | attn_out | site_restore | 20 | 0.504 | 0.150 | 0.150 | 3.45 | space:14, correct_prefix:3, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:4;  v48.\n\nQuestion: c:2;  05.\n\nQuestion: c:2 |
| yes_no_required | l22_peak_layer_out | L27 | attn_out | candidate_ablation | 20 | 0.634 | 0.200 | 0.200 | 2.60 | space:13, correct_prefix:4, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:4;  v48.\n\nQuestion: c:3;  05.\n\nQuestion: c:2 |
| yes_no_required | l22_peak_layer_out | L27 | attn_out | site_restore | 20 | 0.634 | 0.150 | 0.150 | 3.45 | space:14, correct_prefix:3, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:4;  v48.\n\nQuestion: c:2;  05.\n\nQuestion: c:2 |
| yes_no_required | late_peak_layer_out | L23 | attn_out | candidate_ablation | 20 | 0.504 | 0.200 | 0.200 | 2.65 | space:15, correct_prefix:4, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:4;  v48.\n\nQuestion: c:3;  48.\n\nc12 r:2 |
| yes_no_required | late_peak_layer_out | L23 | attn_out | site_restore | 20 | 0.504 | 0.150 | 0.150 | 3.45 | space:14, correct_prefix:3, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:3;  v48.\n\nQuestion: c:2;  05.\n\nQuestion: c:2 |
| yes_no_required | late_peak_layer_out | L27 | attn_out | candidate_ablation | 20 | 0.634 | 0.200 | 0.200 | 2.60 | space:13, correct_prefix:4, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:3;  v48.\n\nQuestion: c:3;  05.\n\nQuestion: c:2 |
| yes_no_required | late_peak_layer_out | L27 | attn_out | site_restore | 20 | 0.634 | 0.150 | 0.150 | 3.45 | space:14, correct_prefix:3, explanation:2, word:1 |  22.\n\nQuestion: c:4;  48.\n\nQuestion: c:3;  v48.\n\nQuestion: c:2;  05.\n\nQuestion: c:2 |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 240 / total_time_min: 1.20
- max_cases: 20 / max_new_tokens: 6
- candidate_specs: `[{'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'layer': 24, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 7, 'phase656_dtop': 0.8035714285714286, 'phase656_drank': 7.0, 'phase656_flip': 0}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'layer': 24, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 7, 'phase656_dtop': 0.8035714285714286, 'phase656_drank': 7.0, 'phase656_flip': 0}, {'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'layer': 23, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 7, 'phase656_dtop': 0.6607142857142857, 'phase656_drank': 1.1428571428571428, 'phase656_flip': 1}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'layer': 23, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 7, 'phase656_dtop': 0.6607142857142857, 'phase656_drank': 1.1428571428571428, 'phase656_flip': 1}, {'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'layer': 24, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 11, 'phase656_dtop': 0.5511363636363636, 'phase656_drank': 7.2727272727272725, 'phase656_flip': 0}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'layer': 24, 'component': 'mlp_out', 'baseline_top0_category': 'newline', 'phase656_n': 11, 'phase656_dtop': 0.5511363636363636, 'phase656_drank': 7.2727272727272725, 'phase656_flip': 0}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Generation Effects

| pair_task | site | layer | component | n | phase656_dtop | exact base->ablate | delta_exact | tok0 base->ablate | delta_tok0 | rank base->ablate | rank_improvement | base_top0 | ablation_top0 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | L23 | mlp_out | 20 | 0.661 | 3->3 | 0 | 2->3 | 1 | 8.30->7.40 | 0.90 | space:11, newline:7, correct_prefix:2 | space:17, correct_prefix:3 |
| explanation_required | late_peak_layer_out | L23 | mlp_out | 20 | 0.661 | 3->3 | 0 | 2->3 | 1 | 8.30->7.40 | 0.90 | space:11, newline:7, correct_prefix:2 | space:17, correct_prefix:3 |
| yes_no_required | l22_peak_layer_out | L24 | mlp_out | 20 | 0.551 | 0->0 | 0 | 0->0 | 0 | 13.85->9.60 | 4.25 | newline:11, space:9 | newline:16, space:4 |
| yes_no_required | late_peak_layer_out | L24 | mlp_out | 20 | 0.551 | 0->0 | 0 | 0->0 | 0 | 13.85->9.60 | 4.25 | newline:11, space:9 | newline:16, space:4 |
| explanation_required | l22_peak_layer_out | L24 | mlp_out | 20 | 0.804 | 3->3 | 0 | 2->2 | 0 | 8.30->5.30 | 3.00 | space:11, newline:7, correct_prefix:2 | newline:12, space:6, correct_prefix:2 |
| explanation_required | late_peak_layer_out | L24 | mlp_out | 20 | 0.804 | 3->3 | 0 | 2->2 | 0 | 8.30->5.30 | 3.00 | space:11, newline:7, correct_prefix:2 | newline:12, space:6, correct_prefix:2 |

### By Mode

| pair_task | site | layer | component | mode | n | phase656_dtop | exact_rate | tok0_rate | mean_rank | top0_category | generation_text |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | L23 | mlp_out | candidate_ablation | 20 | 0.661 | 0.150 | 0.150 | 7.40 | space:17, correct_prefix:3 |  22\nReason::3;  48\n</think>\n\n:3;  48\nExplanation::3;  05. Why?\n\n:2 |
| explanation_required | l22_peak_layer_out | L23 | mlp_out | site_restore | 20 | 0.661 | 0.150 | 0.100 | 8.30 | space:11, newline:7, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  48\n</think>\n\n:3;  22\nReason::2;  05. Why?\n\n:2 |
| explanation_required | l22_peak_layer_out | L24 | mlp_out | candidate_ablation | 20 | 0.804 | 0.150 | 0.100 | 5.30 | newline:12, space:6, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  48\n</think>\n\n:3;  ?\n\nOkay, so I have:2;  ?\n\nTo solve this, I:2 |
| explanation_required | l22_peak_layer_out | L24 | mlp_out | site_restore | 20 | 0.804 | 0.150 | 0.100 | 8.30 | space:11, newline:7, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  48\n</think>\n\n:3;  22\nReason::2;  05. Why?\n\n:2 |
| explanation_required | late_peak_layer_out | L23 | mlp_out | candidate_ablation | 20 | 0.661 | 0.150 | 0.150 | 7.40 | space:17, correct_prefix:3 |  48.\n\nExplanation::5;  22\nReason::2;  05. Why?\n\n:2;  22\nBut why:1 |
| explanation_required | late_peak_layer_out | L23 | mlp_out | site_restore | 20 | 0.661 | 0.150 | 0.100 | 8.30 | space:11, newline:7, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  48.\n\nExplanation::2;  05. Why?\n\n:2;  ?\n\nOkay, so I have:1 |
| explanation_required | late_peak_layer_out | L24 | mlp_out | candidate_ablation | 20 | 0.804 | 0.150 | 0.100 | 5.30 | newline:12, space:6, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  ?\n\nOkay, so I have:2;  48.\n\nExplanation::2;  ?\n\nTo solve this, I:2 |
| explanation_required | late_peak_layer_out | L24 | mlp_out | site_restore | 20 | 0.804 | 0.150 | 0.100 | 8.30 | space:11, newline:7, correct_prefix:2 |  ?\n\nOkay, so I'm:5;  48.\n\nExplanation::2;  05. Why?\n\n:2;  ?\n\nOkay, so I have:1 |
| yes_no_required | l22_peak_layer_out | L24 | mlp_out | candidate_ablation | 20 | 0.551 | 0.000 | 0.000 | 9.60 | newline:16, space:4 |  ?\n\nOkay, so I need:5;  ?\n\nQuestion: c12:3;  ?\n\nQuestion: c77:3;  ?\n\nQuestion: c33:2 |
| yes_no_required | l22_peak_layer_out | L24 | mlp_out | site_restore | 20 | 0.551 | 0.000 | 0.000 | 13.85 | newline:11, space:9 |  ?\n\nOkay, so I need:5;  48.\n\nQuestion::4;  05.\n\nQuestion::2;  91.\n\nQuestion::2 |
| yes_no_required | late_peak_layer_out | L24 | mlp_out | candidate_ablation | 20 | 0.551 | 0.000 | 0.000 | 9.60 | newline:16, space:4 |  ?\n\nOkay, so I need:8;  ?\n\nOkay, so I'm:4;  ?\n\nQuestion: c12:2;  05.\n\n</think>\n\n:2 |
| yes_no_required | late_peak_layer_out | L24 | mlp_out | site_restore | 20 | 0.551 | 0.000 | 0.000 | 13.85 | newline:11, space:9 |  ?\n\nOkay, so I need:6;  48.\n\nQuestion::4;  ?\n\nOkay, so I'm:3;  22\n</think>\n\n:2 |
