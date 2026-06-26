# Phase 660 Cross-Model Summary

目标：固定 Phase 659 best combo，审计 space/newline final top1 barrier 来自 residual state、final_norm shift、lm_head projection，还是最后残差写入器残留。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 880 / total_time_min: 0.93
- last_layers: `[32, 33, 34, 35]`
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top2', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}, {'layer': 20, 'component': 'attn_out', 'phase656_dtop': 1.53125, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.9500000000000002}], 'phase658_delta_exact': 14, 'phase658_delta_tok0': 15, 'phase658_rank_improvement': 2.1}, {'pair_task': 'explanation_required', 'site': 'separator_input_edge', 'combo_name': 'top1', 'components': [{'layer': 16, 'component': 'attn_out', 'phase656_dtop': 1.4375, 'phase657_delta_exact': 8, 'phase657_delta_tok0': 9, 'phase657_rank_improvement': 1.7000000000000002}], 'phase658_delta_exact': 8, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 1.7000000000000002}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 11, 'phase658_rank_improvement': 3.0000000000000004}, {'pair_task': 'yes_no_required', 'site': 'early_peak_layer_out', 'combo_name': 'top3', 'components': [{'layer': 18, 'component': 'attn_out', 'phase656_dtop': 1.5, 'phase657_delta_exact': 5, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 2.0000000000000004}, {'layer': 21, 'component': 'mlp_out', 'phase656_dtop': 1.8916666666666666, 'phase657_delta_exact': 4, 'phase657_delta_tok0': 5, 'phase657_rank_improvement': 1.8500000000000005}, {'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 1.4666666666666666, 'phase657_delta_exact': 2, 'phase657_delta_tok0': 3, 'phase657_rank_improvement': 0.7000000000000002}], 'phase658_delta_exact': 11, 'phase658_delta_tok0': 9, 'phase658_rank_improvement': 2.5500000000000003}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Source Effects

| pair_task | site | combo | components | n | post_gap site->combo | gap_reduction | rank site->combo | rank_improvement | combo_norm_gap_shift | site_top1 | combo_top1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| yes_no_required | early_peak_layer_out | top2 | L18 attn_out, L21 mlp_out | 20 | 2.219->0.081 | 2.138 | 4.15->1.15 | 3.00 | -19.919 | space:15, correct_prefix:5 | correct_prefix:16, space:3, newline:1 |
| yes_no_required | early_peak_layer_out | top3 | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 2.219->0.212 | 2.006 | 4.15->1.60 | 2.55 | -23.425 | space:15, correct_prefix:5 | correct_prefix:16, newline:4 |
| explanation_required | separator_input_edge | top2 | L16 attn_out, L20 attn_out | 20 | 1.137->0.000 | 1.137 | 3.10->1.00 | 2.10 | -12.312 | word:12, space:4, correct_prefix:4 | correct_prefix:20 |
| explanation_required | separator_input_edge | top1 | L16 attn_out | 20 | 1.137->0.294 | 0.844 | 3.10->1.40 | 1.70 | -13.994 | word:12, space:4, correct_prefix:4 | correct_prefix:14, space:6 |

### Last Writer Effects

| pair_task | site | combo | last_writer_mode | components | n | gap combo->ablated | gap_delta_vs_combo | rank combo->ablated | rank_delta_vs_combo | ablated_norm_shift | ablated_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top1 | last4_L35_mlp_out | L16 attn_out | 20 | 0.294->0.122 | 0.172 | 1.40->1.25 | 0.15 | -53.659 | correct_prefix:16, space:3, word:1 |
| yes_no_required | early_peak_layer_out | top3 | last4_L35_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.066 | 0.147 | 1.60->1.30 | 0.30 | -58.553 | correct_prefix:18, explanation:2 |
| explanation_required | separator_input_edge | top1 | last4_L32_mlp_out | L16 attn_out | 20 | 0.294->0.156 | 0.138 | 1.40->1.25 | 0.15 | -11.656 | correct_prefix:15, space:5 |
| yes_no_required | early_peak_layer_out | top3 | last4_L32_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.106 | 0.106 | 1.60->1.45 | 0.15 | -15.594 | correct_prefix:16, newline:3, explanation:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L35_mlp_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.013 | 0.069 | 1.15->1.05 | 0.10 | -56.144 | correct_prefix:19, space:1 |
| yes_no_required | early_peak_layer_out | top3 | last4_L32_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.175 | 0.038 | 1.60->1.50 | 0.10 | -13.238 | correct_prefix:16, newline:3, explanation:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L32_mlp_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.050 | 0.031 | 1.15->1.10 | 0.05 | -13.000 | correct_prefix:18, space:1, newline:1 |
| explanation_required | separator_input_edge | top1 | last4_L33_mlp_out | L16 attn_out | 20 | 0.294->0.275 | 0.019 | 1.40->1.40 | 0.00 | -12.562 | correct_prefix:15, space:4, newline:1 |
| explanation_required | separator_input_edge | top2 | last4_L32_attn_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.000 | 0.000 | 1.00->1.00 | 0.00 | -9.000 | correct_prefix:19, space:1 |
| explanation_required | separator_input_edge | top2 | last4_L32_mlp_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.000 | 0.000 | 1.00->1.00 | 0.00 | -9.500 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L33_mlp_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.000 | 0.000 | 1.00->1.00 | 0.00 | -10.875 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L34_attn_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.000 | 0.000 | 1.00->1.00 | 0.00 | -9.588 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L33_attn_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.013 | -0.013 | 1.00->1.05 | -0.05 | -12.438 | correct_prefix:19, space:1 |
| explanation_required | separator_input_edge | top1 | last4_L33_attn_out | L16 attn_out | 20 | 0.294->0.312 | -0.019 | 1.40->1.70 | -0.30 | -14.400 | correct_prefix:14, space:5, word:1 |
| explanation_required | separator_input_edge | top2 | last4_L35_mlp_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.022 | -0.022 | 1.00->1.05 | -0.05 | -53.247 | correct_prefix:19, word:1 |
| explanation_required | separator_input_edge | top2 | last4_L35_attn_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.031 | -0.031 | 1.00->1.10 | -0.10 | -14.606 | correct_prefix:17, space:3 |
| explanation_required | separator_input_edge | top1 | last4_L34_attn_out | L16 attn_out | 20 | 0.294->0.338 | -0.044 | 1.40->1.50 | -0.10 | -10.275 | correct_prefix:14, space:6 |
| explanation_required | separator_input_edge | top2 | last4_L34_mlp_out | L16 attn_out, L20 attn_out | 20 | 0.000->0.050 | -0.050 | 1.00->1.15 | -0.15 | -3.650 | correct_prefix:17, space:2, word:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L33_mlp_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.138 | -0.056 | 1.15->1.35 | -0.20 | -15.113 | correct_prefix:14, newline:6 |
| explanation_required | separator_input_edge | top1 | last4_L32_attn_out | L16 attn_out | 20 | 0.294->0.375 | -0.081 | 1.40->1.35 | 0.05 | -10.850 | correct_prefix:13, space:7 |
| yes_no_required | early_peak_layer_out | top2 | last4_L32_attn_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.175 | -0.094 | 1.15->1.45 | -0.30 | -11.725 | correct_prefix:14, space:5, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L33_attn_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.188 | -0.106 | 1.15->1.65 | -0.50 | -21.413 | correct_prefix:12, space:5, newline:3 |
| explanation_required | separator_input_edge | top1 | last4_L35_attn_out | L16 attn_out | 20 | 0.294->0.450 | -0.156 | 1.40->1.45 | -0.05 | -14.850 | correct_prefix:12, space:8 |
| explanation_required | separator_input_edge | top1 | last4_L34_mlp_out | L16 attn_out | 20 | 0.294->0.450 | -0.156 | 1.40->1.85 | -0.45 | -5.100 | correct_prefix:12, space:8 |
| yes_no_required | early_peak_layer_out | top2 | last4_L34_attn_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.237 | -0.156 | 1.15->1.60 | -0.45 | -15.912 | correct_prefix:12, space:5, newline:3 |
| yes_no_required | early_peak_layer_out | top3 | last4_L34_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.394 | -0.181 | 1.60->2.05 | -0.45 | -21.769 | correct_prefix:11, newline:7, space:2 |
| yes_no_required | early_peak_layer_out | top2 | last4_L35_attn_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.356 | -0.275 | 1.15->1.75 | -0.60 | -16.494 | correct_prefix:10, space:8, newline:2 |
| yes_no_required | early_peak_layer_out | top3 | last4_L33_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.531 | -0.319 | 1.60->2.10 | -0.50 | -16.681 | newline:17, correct_prefix:3 |
| yes_no_required | early_peak_layer_out | top3 | last4_L35_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.600 | -0.387 | 1.60->3.55 | -1.95 | -22.363 | explanation:11, correct_prefix:4, newline:3, space:2 |
| yes_no_required | early_peak_layer_out | top3 | last4_L33_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->0.637 | -0.425 | 1.60->2.55 | -0.95 | -23.137 | newline:17, correct_prefix:3 |
| yes_no_required | early_peak_layer_out | top2 | last4_L34_mlp_out | L18 attn_out, L21 mlp_out | 20 | 0.081->0.506 | -0.425 | 1.15->2.30 | -1.15 | -14.794 | correct_prefix:11, space:4, explanation:4, newline:1 |
| yes_no_required | early_peak_layer_out | top3 | last4_L34_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 0.212->2.188 | -1.975 | 1.60->5.35 | -3.75 | -21.887 | explanation:20 |

### By Mode

| pair_task | site | combo | mode | components | n | pre_gap | post_gap | norm_gap_shift | post_rank | correct_top1_rate | pre_top1 | post_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | separator_input_edge | top1 | baseline_task | L16 attn_out | 20 | 9.812 | 0.613 | -9.200 | 1.80 | 0.500 | newline:11, word:9 | word:10, correct_prefix:10 |
| explanation_required | separator_input_edge | top1 | combo_ablation | L16 attn_out | 20 | 14.287 | 0.294 | -13.994 | 1.40 | 0.700 | newline:11, word:9 | correct_prefix:14, space:6 |
| explanation_required | separator_input_edge | top1 | last4_L32_attn_out | L16 attn_out | 20 | 11.225 | 0.375 | -10.850 | 1.35 | 0.650 | newline:11, word:9 | correct_prefix:13, space:7 |
| explanation_required | separator_input_edge | top1 | last4_L32_mlp_out | L16 attn_out | 20 | 11.812 | 0.156 | -11.656 | 1.25 | 0.750 | newline:12, word:8 | correct_prefix:15, space:5 |
| explanation_required | separator_input_edge | top1 | last4_L33_attn_out | L16 attn_out | 20 | 14.713 | 0.312 | -14.400 | 1.70 | 0.700 | newline:20 | correct_prefix:14, space:5, word:1 |
| explanation_required | separator_input_edge | top1 | last4_L33_mlp_out | L16 attn_out | 20 | 12.838 | 0.275 | -12.562 | 1.40 | 0.750 | newline:12, word:8 | correct_prefix:15, space:4, newline:1 |
| explanation_required | separator_input_edge | top1 | last4_L34_attn_out | L16 attn_out | 20 | 10.613 | 0.338 | -10.275 | 1.50 | 0.700 | word:14, newline:6 | correct_prefix:14, space:6 |
| explanation_required | separator_input_edge | top1 | last4_L34_mlp_out | L16 attn_out | 20 | 5.550 | 0.450 | -5.100 | 1.85 | 0.600 | newline:16, correct_prefix:4 | correct_prefix:12, space:8 |
| explanation_required | separator_input_edge | top1 | last4_L35_attn_out | L16 attn_out | 20 | 15.300 | 0.450 | -14.850 | 1.45 | 0.600 | newline:10, word:10 | correct_prefix:12, space:8 |
| explanation_required | separator_input_edge | top1 | last4_L35_mlp_out | L16 attn_out | 20 | 53.781 | 0.122 | -53.659 | 1.25 | 0.800 | word:20 | correct_prefix:16, space:3, word:1 |
| explanation_required | separator_input_edge | top1 | site_restore | L16 attn_out | 20 | 15.500 | 1.137 | -14.363 | 3.10 | 0.200 | newline:12, word:8 | word:12, space:4, correct_prefix:4 |
| explanation_required | separator_input_edge | top2 | baseline_task | L16 attn_out, L20 attn_out | 20 | 9.812 | 0.613 | -9.200 | 1.80 | 0.500 | newline:11, word:9 | word:10, correct_prefix:10 |
| explanation_required | separator_input_edge | top2 | combo_ablation | L16 attn_out, L20 attn_out | 20 | 12.312 | 0.000 | -12.312 | 1.00 | 1.000 | word:12, newline:8 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L32_attn_out | L16 attn_out, L20 attn_out | 20 | 9.000 | 0.000 | -9.000 | 1.00 | 0.950 | word:11, newline:9 | correct_prefix:19, space:1 |
| explanation_required | separator_input_edge | top2 | last4_L32_mlp_out | L16 attn_out, L20 attn_out | 20 | 9.500 | 0.000 | -9.500 | 1.00 | 1.000 | newline:12, word:8 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L33_attn_out | L16 attn_out, L20 attn_out | 20 | 12.450 | 0.013 | -12.438 | 1.05 | 0.950 | newline:20 | correct_prefix:19, space:1 |
| explanation_required | separator_input_edge | top2 | last4_L33_mlp_out | L16 attn_out, L20 attn_out | 20 | 10.875 | 0.000 | -10.875 | 1.00 | 1.000 | newline:11, word:9 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L34_attn_out | L16 attn_out, L20 attn_out | 20 | 9.588 | 0.000 | -9.588 | 1.00 | 1.000 | word:15, newline:5 | correct_prefix:20 |
| explanation_required | separator_input_edge | top2 | last4_L34_mlp_out | L16 attn_out, L20 attn_out | 20 | 3.700 | 0.050 | -3.650 | 1.15 | 0.850 | newline:12, correct_prefix:6, word:2 | correct_prefix:17, space:2, word:1 |
| explanation_required | separator_input_edge | top2 | last4_L35_attn_out | L16 attn_out, L20 attn_out | 20 | 14.637 | 0.031 | -14.606 | 1.10 | 0.850 | word:15, newline:5 | correct_prefix:17, space:3 |
| explanation_required | separator_input_edge | top2 | last4_L35_mlp_out | L16 attn_out, L20 attn_out | 20 | 53.269 | 0.022 | -53.247 | 1.05 | 0.950 | word:20 | correct_prefix:19, word:1 |
| explanation_required | separator_input_edge | top2 | site_restore | L16 attn_out, L20 attn_out | 20 | 15.500 | 1.137 | -14.363 | 3.10 | 0.200 | newline:12, word:8 | word:12, space:4, correct_prefix:4 |
| yes_no_required | early_peak_layer_out | top2 | baseline_task | L18 attn_out, L21 mlp_out | 20 | 30.975 | 2.922 | -28.053 | 12.95 | 0.000 | explanation:17, newline:3 | explanation:20 |
| yes_no_required | early_peak_layer_out | top2 | combo_ablation | L18 attn_out, L21 mlp_out | 20 | 20.000 | 0.081 | -19.919 | 1.15 | 0.800 | word:17, newline:3 | correct_prefix:16, space:3, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L32_attn_out | L18 attn_out, L21 mlp_out | 20 | 11.900 | 0.175 | -11.725 | 1.45 | 0.700 | word:13, newline:7 | correct_prefix:14, space:5, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L32_mlp_out | L18 attn_out, L21 mlp_out | 20 | 13.050 | 0.050 | -13.000 | 1.10 | 0.900 | word:11, newline:9 | correct_prefix:18, space:1, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L33_attn_out | L18 attn_out, L21 mlp_out | 20 | 21.600 | 0.188 | -21.413 | 1.65 | 0.600 | newline:15, word:5 | correct_prefix:12, space:5, newline:3 |
| yes_no_required | early_peak_layer_out | top2 | last4_L33_mlp_out | L18 attn_out, L21 mlp_out | 20 | 15.250 | 0.138 | -15.113 | 1.35 | 0.700 | word:15, newline:5 | correct_prefix:14, newline:6 |
| yes_no_required | early_peak_layer_out | top2 | last4_L34_attn_out | L18 attn_out, L21 mlp_out | 20 | 16.150 | 0.237 | -15.912 | 1.60 | 0.600 | newline:11, word:9 | correct_prefix:12, space:5, newline:3 |
| yes_no_required | early_peak_layer_out | top2 | last4_L34_mlp_out | L18 attn_out, L21 mlp_out | 20 | 15.300 | 0.506 | -14.794 | 2.30 | 0.550 | explanation:17, newline:3 | correct_prefix:11, space:4, explanation:4, newline:1 |
| yes_no_required | early_peak_layer_out | top2 | last4_L35_attn_out | L18 attn_out, L21 mlp_out | 20 | 16.850 | 0.356 | -16.494 | 1.75 | 0.500 | word:13, newline:7 | correct_prefix:10, space:8, newline:2 |
| yes_no_required | early_peak_layer_out | top2 | last4_L35_mlp_out | L18 attn_out, L21 mlp_out | 20 | 56.156 | 0.013 | -56.144 | 1.05 | 0.950 | word:20 | correct_prefix:19, space:1 |
| yes_no_required | early_peak_layer_out | top2 | site_restore | L18 attn_out, L21 mlp_out | 20 | 20.350 | 2.219 | -18.131 | 4.15 | 0.250 | word:10, newline:9, explanation:1 | space:15, correct_prefix:5 |
| yes_no_required | early_peak_layer_out | top3 | baseline_task | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 30.975 | 2.922 | -28.053 | 12.95 | 0.000 | explanation:17, newline:3 | explanation:20 |
| yes_no_required | early_peak_layer_out | top3 | combo_ablation | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 23.637 | 0.212 | -23.425 | 1.60 | 0.800 | word:14, newline:6 | correct_prefix:16, newline:4 |
| yes_no_required | early_peak_layer_out | top3 | last4_L32_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 13.412 | 0.175 | -13.238 | 1.50 | 0.800 | word:12, explanation:8 | correct_prefix:16, newline:3, explanation:1 |
| yes_no_required | early_peak_layer_out | top3 | last4_L32_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 15.700 | 0.106 | -15.594 | 1.45 | 0.800 | newline:13, word:7 | correct_prefix:16, newline:3, explanation:1 |
| yes_no_required | early_peak_layer_out | top3 | last4_L33_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 23.775 | 0.637 | -23.137 | 2.55 | 0.150 | newline:15, word:5 | newline:17, correct_prefix:3 |
| yes_no_required | early_peak_layer_out | top3 | last4_L33_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 17.212 | 0.531 | -16.681 | 2.10 | 0.150 | word:14, newline:5, explanation:1 | newline:17, correct_prefix:3 |
| yes_no_required | early_peak_layer_out | top3 | last4_L34_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 22.163 | 0.394 | -21.769 | 2.05 | 0.550 | newline:16, word:4 | correct_prefix:11, newline:7, space:2 |
| yes_no_required | early_peak_layer_out | top3 | last4_L34_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 24.075 | 2.188 | -21.887 | 5.35 | 0.000 | explanation:20 | explanation:20 |
| yes_no_required | early_peak_layer_out | top3 | last4_L35_attn_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 22.962 | 0.600 | -22.363 | 3.55 | 0.200 | newline:11, word:8, explanation:1 | explanation:11, correct_prefix:4, newline:3, space:2 |
| yes_no_required | early_peak_layer_out | top3 | last4_L35_mlp_out | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 58.619 | 0.066 | -58.553 | 1.30 | 0.900 | word:18, newline:2 | correct_prefix:18, explanation:2 |
| yes_no_required | early_peak_layer_out | top3 | site_restore | L18 attn_out, L21 mlp_out, L23 mlp_out | 20 | 20.350 | 2.219 | -18.131 | 4.15 | 0.250 | word:10, newline:9, explanation:1 | space:15, correct_prefix:5 |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 880 / total_time_min: 1.22
- last_layers: `[36, 37, 38, 39]`
- combo_specs: `[{'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 27, 'component': 'attn_out', 'phase656_dtop': 0.6339285714285714, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8500000000000001}, {'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.5044642857142857, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.8000000000000003}], 'phase658_delta_exact': 5, 'phase658_delta_tok0': 5, 'phase658_rank_improvement': 1.4500000000000002}, {'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 23, 'component': 'attn_out', 'phase656_dtop': 0.4875, 'phase657_delta_exact': 1, 'phase657_delta_tok0': 2, 'phase657_rank_improvement': 0.25}], 'phase658_delta_exact': 1, 'phase658_delta_tok0': 2, 'phase658_rank_improvement': 0.25}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Source Effects

| pair_task | site | combo | components | n | post_gap site->combo | gap_reduction | rank site->combo | rank_improvement | combo_norm_gap_shift | site_top1 | combo_top1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| yes_no_required | l22_peak_layer_out | top2 | L27 attn_out, L23 attn_out | 20 | 1.134->0.431 | 0.703 | 3.45->2.00 | 1.45 | -36.404 | space:15, correct_prefix:3, explanation:2 | space:11, correct_prefix:8, word:1 |
| yes_no_required | late_peak_layer_out | top2 | L27 attn_out, L23 attn_out | 20 | 1.134->0.431 | 0.703 | 3.45->2.00 | 1.45 | -36.404 | space:15, correct_prefix:3, explanation:2 | space:11, correct_prefix:8, word:1 |
| explanation_required | l22_peak_layer_out | top1 | L23 attn_out | 20 | 0.803->0.453 | 0.350 | 2.15->1.90 | 0.25 | -32.706 | space:15, correct_prefix:5 | space:12, correct_prefix:6, word:2 |
| explanation_required | late_peak_layer_out | top1 | L23 attn_out | 20 | 0.803->0.453 | 0.350 | 2.15->1.90 | 0.25 | -32.706 | space:15, correct_prefix:5 | space:12, correct_prefix:6, word:2 |

### Last Writer Effects

| pair_task | site | combo | last_writer_mode | components | n | gap combo->ablated | gap_delta_vs_combo | rank combo->ablated | rank_delta_vs_combo | ablated_norm_shift | ablated_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top1 | last4_L36_attn_out | L23 attn_out | 20 | 0.453->0.269 | 0.184 | 1.90->1.55 | 0.35 | -27.356 | space:10, correct_prefix:10 |
| explanation_required | late_peak_layer_out | top1 | last4_L36_attn_out | L23 attn_out | 20 | 0.453->0.269 | 0.184 | 1.90->1.55 | 0.35 | -27.356 | space:10, correct_prefix:10 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L36_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.287 | 0.144 | 2.00->1.75 | 0.25 | -30.513 | correct_prefix:11, space:6, word:3 |
| yes_no_required | late_peak_layer_out | top2 | last4_L36_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.287 | 0.144 | 2.00->1.75 | 0.25 | -30.513 | correct_prefix:11, space:6, word:3 |
| explanation_required | l22_peak_layer_out | top1 | last4_L36_mlp_out | L23 attn_out | 20 | 0.453->0.325 | 0.128 | 1.90->1.55 | 0.35 | -30.617 | correct_prefix:10, space:8, word:2 |
| explanation_required | late_peak_layer_out | top1 | last4_L36_mlp_out | L23 attn_out | 20 | 0.453->0.325 | 0.128 | 1.90->1.55 | 0.35 | -30.617 | correct_prefix:10, space:8, word:2 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L36_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.306 | 0.125 | 2.00->1.60 | 0.40 | -34.126 | correct_prefix:12, space:6, word:2 |
| yes_no_required | late_peak_layer_out | top2 | last4_L36_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.306 | 0.125 | 2.00->1.60 | 0.40 | -34.126 | correct_prefix:12, space:6, word:2 |
| explanation_required | l22_peak_layer_out | top1 | last4_L37_attn_out | L23 attn_out | 20 | 0.453->0.344 | 0.109 | 1.90->1.70 | 0.20 | -27.600 | space:12, correct_prefix:8 |
| explanation_required | late_peak_layer_out | top1 | last4_L37_attn_out | L23 attn_out | 20 | 0.453->0.344 | 0.109 | 1.90->1.70 | 0.20 | -27.600 | space:12, correct_prefix:8 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L37_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.350 | 0.081 | 2.00->1.85 | 0.15 | -30.860 | space:9, correct_prefix:9, word:2 |
| yes_no_required | late_peak_layer_out | top2 | last4_L37_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.350 | 0.081 | 2.00->1.85 | 0.15 | -30.860 | space:9, correct_prefix:9, word:2 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L39_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.375 | 0.056 | 2.00->1.90 | 0.10 | -30.210 | space:10, correct_prefix:9, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L39_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.375 | 0.056 | 2.00->1.90 | 0.10 | -30.210 | space:10, correct_prefix:9, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L38_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.431 | 0.000 | 2.00->1.70 | 0.30 | -43.224 | space:10, correct_prefix:9, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L38_attn_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.431 | 0.000 | 2.00->1.70 | 0.30 | -43.224 | space:10, correct_prefix:9, word:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L39_attn_out | L23 attn_out | 20 | 0.453->0.491 | -0.037 | 1.90->1.95 | -0.05 | -25.373 | space:15, correct_prefix:4, word:1 |
| explanation_required | late_peak_layer_out | top1 | last4_L39_attn_out | L23 attn_out | 20 | 0.453->0.491 | -0.037 | 1.90->1.95 | -0.05 | -25.373 | space:15, correct_prefix:4, word:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L38_attn_out | L23 attn_out | 20 | 0.453->0.631 | -0.178 | 1.90->1.90 | 0.00 | -40.305 | space:16, correct_prefix:4 |
| explanation_required | late_peak_layer_out | top1 | last4_L38_attn_out | L23 attn_out | 20 | 0.453->0.631 | -0.178 | 1.90->1.90 | 0.00 | -40.305 | space:16, correct_prefix:4 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L38_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.634 | -0.203 | 2.00->2.30 | -0.30 | -33.410 | space:14, word:3, correct_prefix:3 |
| yes_no_required | late_peak_layer_out | top2 | last4_L38_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.634 | -0.203 | 2.00->2.30 | -0.30 | -33.410 | space:14, word:3, correct_prefix:3 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L37_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.809 | -0.378 | 2.00->2.80 | -0.80 | -33.257 | space:13, correct_prefix:4, word:2, explanation:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L37_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->0.809 | -0.378 | 2.00->2.80 | -0.80 | -33.257 | space:13, correct_prefix:4, word:2, explanation:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L37_mlp_out | L23 attn_out | 20 | 0.453->0.887 | -0.434 | 1.90->2.45 | -0.55 | -32.073 | space:14, correct_prefix:4, word:2 |
| explanation_required | late_peak_layer_out | top1 | last4_L37_mlp_out | L23 attn_out | 20 | 0.453->0.887 | -0.434 | 1.90->2.45 | -0.55 | -32.073 | space:14, correct_prefix:4, word:2 |
| explanation_required | l22_peak_layer_out | top1 | last4_L38_mlp_out | L23 attn_out | 20 | 0.453->1.025 | -0.572 | 1.90->2.40 | -0.50 | -29.467 | space:19, correct_prefix:1 |
| explanation_required | late_peak_layer_out | top1 | last4_L38_mlp_out | L23 attn_out | 20 | 0.453->1.025 | -0.572 | 1.90->2.40 | -0.50 | -29.467 | space:19, correct_prefix:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L39_mlp_out | L23 attn_out | 20 | 0.453->1.325 | -0.872 | 1.90->2.50 | -0.60 | -44.299 | space:17, correct_prefix:2, word:1 |
| explanation_required | late_peak_layer_out | top1 | last4_L39_mlp_out | L23 attn_out | 20 | 0.453->1.325 | -0.872 | 1.90->2.50 | -0.60 | -44.299 | space:17, correct_prefix:2, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L39_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->1.419 | -0.987 | 2.00->3.25 | -1.25 | -46.973 | space:11, explanation:5, correct_prefix:3, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L39_mlp_out | L27 attn_out, L23 attn_out | 20 | 0.431->1.419 | -0.987 | 2.00->3.25 | -1.25 | -46.973 | space:11, explanation:5, correct_prefix:3, word:1 |

### By Mode

| pair_task | site | combo | mode | components | n | pre_gap | post_gap | norm_gap_shift | post_rank | correct_top1_rate | pre_top1 | post_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top1 | baseline_task | L23 attn_out | 20 | 34.743 | 5.239 | -29.504 | 58.30 | 0.000 | word:20 | word:20 |
| explanation_required | l22_peak_layer_out | top1 | combo_ablation | L23 attn_out | 20 | 33.160 | 0.453 | -32.706 | 1.90 | 0.300 | word:20 | space:12, correct_prefix:6, word:2 |
| explanation_required | l22_peak_layer_out | top1 | last4_L36_attn_out | L23 attn_out | 20 | 27.624 | 0.269 | -27.356 | 1.55 | 0.500 | word:20 | space:10, correct_prefix:10 |
| explanation_required | l22_peak_layer_out | top1 | last4_L36_mlp_out | L23 attn_out | 20 | 30.942 | 0.325 | -30.617 | 1.55 | 0.500 | word:20 | correct_prefix:10, space:8, word:2 |
| explanation_required | l22_peak_layer_out | top1 | last4_L37_attn_out | L23 attn_out | 20 | 27.943 | 0.344 | -27.600 | 1.70 | 0.400 | word:20 | space:12, correct_prefix:8 |
| explanation_required | l22_peak_layer_out | top1 | last4_L37_mlp_out | L23 attn_out | 20 | 32.960 | 0.887 | -32.073 | 2.45 | 0.200 | word:20 | space:14, correct_prefix:4, word:2 |
| explanation_required | l22_peak_layer_out | top1 | last4_L38_attn_out | L23 attn_out | 20 | 40.937 | 0.631 | -40.305 | 1.90 | 0.200 | word:20 | space:16, correct_prefix:4 |
| explanation_required | l22_peak_layer_out | top1 | last4_L38_mlp_out | L23 attn_out | 20 | 30.492 | 1.025 | -29.467 | 2.40 | 0.050 | word:20 | space:19, correct_prefix:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L39_attn_out | L23 attn_out | 20 | 25.864 | 0.491 | -25.373 | 1.95 | 0.200 | word:20 | space:15, correct_prefix:4, word:1 |
| explanation_required | l22_peak_layer_out | top1 | last4_L39_mlp_out | L23 attn_out | 20 | 45.624 | 1.325 | -44.299 | 2.50 | 0.100 | word:20 | space:17, correct_prefix:2, word:1 |
| explanation_required | l22_peak_layer_out | top1 | site_restore | L23 attn_out | 20 | 30.676 | 0.803 | -29.873 | 2.15 | 0.250 | word:20 | space:15, correct_prefix:5 |
| explanation_required | late_peak_layer_out | top1 | baseline_task | L23 attn_out | 20 | 34.743 | 5.239 | -29.504 | 58.30 | 0.000 | word:20 | word:20 |
| explanation_required | late_peak_layer_out | top1 | combo_ablation | L23 attn_out | 20 | 33.160 | 0.453 | -32.706 | 1.90 | 0.300 | word:20 | space:12, correct_prefix:6, word:2 |
| explanation_required | late_peak_layer_out | top1 | last4_L36_attn_out | L23 attn_out | 20 | 27.624 | 0.269 | -27.356 | 1.55 | 0.500 | word:20 | space:10, correct_prefix:10 |
| explanation_required | late_peak_layer_out | top1 | last4_L36_mlp_out | L23 attn_out | 20 | 30.942 | 0.325 | -30.617 | 1.55 | 0.500 | word:20 | correct_prefix:10, space:8, word:2 |
| explanation_required | late_peak_layer_out | top1 | last4_L37_attn_out | L23 attn_out | 20 | 27.943 | 0.344 | -27.600 | 1.70 | 0.400 | word:20 | space:12, correct_prefix:8 |
| explanation_required | late_peak_layer_out | top1 | last4_L37_mlp_out | L23 attn_out | 20 | 32.960 | 0.887 | -32.073 | 2.45 | 0.200 | word:20 | space:14, correct_prefix:4, word:2 |
| explanation_required | late_peak_layer_out | top1 | last4_L38_attn_out | L23 attn_out | 20 | 40.937 | 0.631 | -40.305 | 1.90 | 0.200 | word:20 | space:16, correct_prefix:4 |
| explanation_required | late_peak_layer_out | top1 | last4_L38_mlp_out | L23 attn_out | 20 | 30.492 | 1.025 | -29.467 | 2.40 | 0.050 | word:20 | space:19, correct_prefix:1 |
| explanation_required | late_peak_layer_out | top1 | last4_L39_attn_out | L23 attn_out | 20 | 25.864 | 0.491 | -25.373 | 1.95 | 0.200 | word:20 | space:15, correct_prefix:4, word:1 |
| explanation_required | late_peak_layer_out | top1 | last4_L39_mlp_out | L23 attn_out | 20 | 45.624 | 1.325 | -44.299 | 2.50 | 0.100 | word:20 | space:17, correct_prefix:2, word:1 |
| explanation_required | late_peak_layer_out | top1 | site_restore | L23 attn_out | 20 | 30.676 | 0.803 | -29.873 | 2.15 | 0.250 | word:20 | space:15, correct_prefix:5 |
| yes_no_required | l22_peak_layer_out | top2 | baseline_task | L27 attn_out, L23 attn_out | 20 | 62.234 | 9.116 | -53.118 | 188.10 | 0.000 | word:20 | explanation:20 |
| yes_no_required | l22_peak_layer_out | top2 | combo_ablation | L27 attn_out, L23 attn_out | 20 | 36.836 | 0.431 | -36.404 | 2.00 | 0.400 | word:20 | space:11, correct_prefix:8, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L36_attn_out | L27 attn_out, L23 attn_out | 20 | 30.800 | 0.287 | -30.513 | 1.75 | 0.550 | word:20 | correct_prefix:11, space:6, word:3 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L36_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.432 | 0.306 | -34.126 | 1.60 | 0.600 | word:20 | correct_prefix:12, space:6, word:2 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L37_attn_out | L27 attn_out, L23 attn_out | 20 | 31.210 | 0.350 | -30.860 | 1.85 | 0.450 | word:20 | space:9, correct_prefix:9, word:2 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L37_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.066 | 0.809 | -33.257 | 2.80 | 0.200 | word:20 | space:13, correct_prefix:4, word:2, explanation:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L38_attn_out | L27 attn_out, L23 attn_out | 20 | 43.655 | 0.431 | -43.224 | 1.70 | 0.450 | word:20 | space:10, correct_prefix:9, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L38_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.045 | 0.634 | -33.410 | 2.30 | 0.150 | word:20 | space:14, word:3, correct_prefix:3 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L39_attn_out | L27 attn_out, L23 attn_out | 20 | 30.585 | 0.375 | -30.210 | 1.90 | 0.450 | word:20 | space:10, correct_prefix:9, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | last4_L39_mlp_out | L27 attn_out, L23 attn_out | 20 | 48.392 | 1.419 | -46.973 | 3.25 | 0.150 | word:20 | space:11, explanation:5, correct_prefix:3, word:1 |
| yes_no_required | l22_peak_layer_out | top2 | site_restore | L27 attn_out, L23 attn_out | 20 | 38.727 | 1.134 | -37.593 | 3.45 | 0.150 | word:20 | space:15, correct_prefix:3, explanation:2 |
| yes_no_required | late_peak_layer_out | top2 | baseline_task | L27 attn_out, L23 attn_out | 20 | 62.234 | 9.116 | -53.118 | 188.10 | 0.000 | word:20 | explanation:20 |
| yes_no_required | late_peak_layer_out | top2 | combo_ablation | L27 attn_out, L23 attn_out | 20 | 36.836 | 0.431 | -36.404 | 2.00 | 0.400 | word:20 | space:11, correct_prefix:8, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L36_attn_out | L27 attn_out, L23 attn_out | 20 | 30.800 | 0.287 | -30.513 | 1.75 | 0.550 | word:20 | correct_prefix:11, space:6, word:3 |
| yes_no_required | late_peak_layer_out | top2 | last4_L36_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.432 | 0.306 | -34.126 | 1.60 | 0.600 | word:20 | correct_prefix:12, space:6, word:2 |
| yes_no_required | late_peak_layer_out | top2 | last4_L37_attn_out | L27 attn_out, L23 attn_out | 20 | 31.210 | 0.350 | -30.860 | 1.85 | 0.450 | word:20 | space:9, correct_prefix:9, word:2 |
| yes_no_required | late_peak_layer_out | top2 | last4_L37_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.066 | 0.809 | -33.257 | 2.80 | 0.200 | word:20 | space:13, correct_prefix:4, word:2, explanation:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L38_attn_out | L27 attn_out, L23 attn_out | 20 | 43.655 | 0.431 | -43.224 | 1.70 | 0.450 | word:20 | space:10, correct_prefix:9, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L38_mlp_out | L27 attn_out, L23 attn_out | 20 | 34.045 | 0.634 | -33.410 | 2.30 | 0.150 | word:20 | space:14, word:3, correct_prefix:3 |
| yes_no_required | late_peak_layer_out | top2 | last4_L39_attn_out | L27 attn_out, L23 attn_out | 20 | 30.585 | 0.375 | -30.210 | 1.90 | 0.450 | word:20 | space:10, correct_prefix:9, word:1 |
| yes_no_required | late_peak_layer_out | top2 | last4_L39_mlp_out | L27 attn_out, L23 attn_out | 20 | 48.392 | 1.419 | -46.973 | 3.25 | 0.150 | word:20 | space:11, explanation:5, correct_prefix:3, word:1 |
| yes_no_required | late_peak_layer_out | top2 | site_restore | L27 attn_out, L23 attn_out | 20 | 38.727 | 1.134 | -37.593 | 3.45 | 0.150 | word:20 | space:15, correct_prefix:3, explanation:2 |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 880 / total_time_min: 1.11
- last_layers: `[24, 25, 26, 27]`
- combo_specs: `[{'pair_task': 'explanation_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'explanation_required', 'site': 'late_peak_layer_out', 'combo_name': 'top2', 'components': [{'layer': 23, 'component': 'mlp_out', 'phase656_dtop': 0.6607142857142857, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 1, 'phase657_rank_improvement': 0.9000000000000004}, {'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.8035714285714286, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 3.000000000000001}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 1, 'phase658_rank_improvement': 4.65}, {'pair_task': 'yes_no_required', 'site': 'l22_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}, {'pair_task': 'yes_no_required', 'site': 'late_peak_layer_out', 'combo_name': 'top1', 'components': [{'layer': 24, 'component': 'mlp_out', 'phase656_dtop': 0.5511363636363636, 'phase657_delta_exact': 0, 'phase657_delta_tok0': 0, 'phase657_rank_improvement': 4.25}], 'phase658_delta_exact': 0, 'phase658_delta_tok0': 0, 'phase658_rank_improvement': 4.25}]`
- selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Source Effects

| pair_task | site | combo | components | n | post_gap site->combo | gap_reduction | rank site->combo | rank_improvement | combo_norm_gap_shift | site_top1 | combo_top1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top2 | L23 mlp_out, L24 mlp_out | 20 | 2.194->0.978 | 1.216 | 8.30->3.65 | 4.65 | -394.966 | space:10, newline:8, correct_prefix:2 | space:15, correct_prefix:3, newline:2 |
| explanation_required | late_peak_layer_out | top2 | L23 mlp_out, L24 mlp_out | 20 | 2.194->0.978 | 1.216 | 8.30->3.65 | 4.65 | -394.966 | space:10, newline:8, correct_prefix:2 | space:15, correct_prefix:3, newline:2 |
| yes_no_required | l22_peak_layer_out | top1 | L24 mlp_out | 20 | 2.812->2.306 | 0.506 | 13.85->9.60 | 4.25 | -518.003 | newline:11, space:9 | newline:17, space:3 |
| yes_no_required | late_peak_layer_out | top1 | L24 mlp_out | 20 | 2.812->2.306 | 0.506 | 13.85->9.60 | 4.25 | -518.003 | newline:11, space:9 | newline:17, space:3 |

### Last Writer Effects

| pair_task | site | combo | last_writer_mode | components | n | gap combo->ablated | gap_delta_vs_combo | rank combo->ablated | rank_delta_vs_combo | ablated_norm_shift | ablated_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| yes_no_required | l22_peak_layer_out | top1 | last4_L25_attn_out | L24 mlp_out | 20 | 2.306->1.953 | 0.353 | 9.60->7.10 | 2.50 | -417.947 | space:11, newline:8, correct_prefix:1 |
| yes_no_required | late_peak_layer_out | top1 | last4_L25_attn_out | L24 mlp_out | 20 | 2.306->1.953 | 0.353 | 9.60->7.10 | 2.50 | -417.947 | space:11, newline:8, correct_prefix:1 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L26_mlp_out | L24 mlp_out | 20 | 2.306->2.016 | 0.291 | 9.60->8.60 | 1.00 | -499.728 | space:10, newline:7, correct_prefix:3 |
| yes_no_required | late_peak_layer_out | top1 | last4_L26_mlp_out | L24 mlp_out | 20 | 2.306->2.016 | 0.291 | 9.60->8.60 | 1.00 | -499.728 | space:10, newline:7, correct_prefix:3 |
| explanation_required | l22_peak_layer_out | top2 | last4_L26_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->0.806 | 0.172 | 3.65->3.00 | 0.65 | -392.900 | space:11, correct_prefix:9 |
| explanation_required | late_peak_layer_out | top2 | last4_L26_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->0.806 | 0.172 | 3.65->3.00 | 0.65 | -392.900 | space:11, correct_prefix:9 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L25_mlp_out | L24 mlp_out | 20 | 2.306->2.253 | 0.053 | 9.60->8.50 | 1.10 | -422.123 | space:18, newline:2 |
| yes_no_required | late_peak_layer_out | top1 | last4_L25_mlp_out | L24 mlp_out | 20 | 2.306->2.253 | 0.053 | 9.60->8.50 | 1.10 | -422.123 | space:18, newline:2 |
| explanation_required | l22_peak_layer_out | top2 | last4_L24_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->0.978 | 0.000 | 3.65->3.65 | 0.00 | -394.966 | space:15, correct_prefix:3, newline:2 |
| explanation_required | late_peak_layer_out | top2 | last4_L24_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->0.978 | 0.000 | 3.65->3.65 | 0.00 | -394.966 | space:15, correct_prefix:3, newline:2 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L24_mlp_out | L24 mlp_out | 20 | 2.306->2.306 | 0.000 | 9.60->9.60 | 0.00 | -518.003 | newline:17, space:3 |
| yes_no_required | late_peak_layer_out | top1 | last4_L24_mlp_out | L24 mlp_out | 20 | 2.306->2.306 | 0.000 | 9.60->9.60 | 0.00 | -518.003 | newline:17, space:3 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L27_mlp_out | L24 mlp_out | 20 | 2.306->2.789 | -0.483 | 9.60->7.90 | 1.70 | -938.350 | space:20 |
| yes_no_required | late_peak_layer_out | top1 | last4_L27_mlp_out | L24 mlp_out | 20 | 2.306->2.789 | -0.483 | 9.60->7.90 | 1.70 | -938.350 | space:20 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L27_attn_out | L24 mlp_out | 20 | 2.306->2.831 | -0.525 | 9.60->30.30 | -20.70 | -131.369 | newline:19, symbol:1 |
| yes_no_required | late_peak_layer_out | top1 | last4_L27_attn_out | L24 mlp_out | 20 | 2.306->2.831 | -0.525 | 9.60->30.30 | -20.70 | -131.369 | newline:19, symbol:1 |
| explanation_required | l22_peak_layer_out | top2 | last4_L26_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.712 | -0.734 | 3.65->7.80 | -4.15 | -292.600 | space:12, newline:5, correct_prefix:3 |
| explanation_required | late_peak_layer_out | top2 | last4_L26_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.712 | -0.734 | 3.65->7.80 | -4.15 | -292.600 | space:12, newline:5, correct_prefix:3 |
| explanation_required | l22_peak_layer_out | top2 | last4_L27_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.759 | -0.781 | 3.65->2.90 | 0.75 | -868.477 | space:18, correct_prefix:2 |
| explanation_required | late_peak_layer_out | top2 | last4_L27_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.759 | -0.781 | 3.65->2.90 | 0.75 | -868.477 | space:18, correct_prefix:2 |
| explanation_required | l22_peak_layer_out | top2 | last4_L24_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.925 | -0.947 | 3.65->7.75 | -4.10 | -416.402 | space:15, newline:4, correct_prefix:1 |
| explanation_required | late_peak_layer_out | top2 | last4_L24_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->1.925 | -0.947 | 3.65->7.75 | -4.10 | -416.402 | space:15, newline:4, correct_prefix:1 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L26_attn_out | L24 mlp_out | 20 | 2.306->3.291 | -0.984 | 9.60->21.65 | -12.05 | -409.810 | newline:14, space:6 |
| yes_no_required | late_peak_layer_out | top1 | last4_L26_attn_out | L24 mlp_out | 20 | 2.306->3.291 | -0.984 | 9.60->21.65 | -12.05 | -409.810 | newline:14, space:6 |
| explanation_required | l22_peak_layer_out | top2 | last4_L25_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->2.206 | -1.228 | 3.65->5.05 | -1.40 | -313.859 | space:20 |
| explanation_required | late_peak_layer_out | top2 | last4_L25_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->2.206 | -1.228 | 3.65->5.05 | -1.40 | -313.859 | space:20 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L24_attn_out | L24 mlp_out | 20 | 2.306->3.544 | -1.238 | 9.60->18.20 | -8.60 | -512.258 | newline:14, space:6 |
| yes_no_required | late_peak_layer_out | top1 | last4_L24_attn_out | L24 mlp_out | 20 | 2.306->3.544 | -1.238 | 9.60->18.20 | -8.60 | -512.258 | newline:14, space:6 |
| explanation_required | l22_peak_layer_out | top2 | last4_L25_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->2.350 | -1.372 | 3.65->5.40 | -1.75 | -312.319 | space:20 |
| explanation_required | late_peak_layer_out | top2 | last4_L25_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->2.350 | -1.372 | 3.65->5.40 | -1.75 | -312.319 | space:20 |
| explanation_required | l22_peak_layer_out | top2 | last4_L27_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->3.825 | -2.847 | 3.65->43.70 | -40.05 | -29.600 | newline:19, symbol:1 |
| explanation_required | late_peak_layer_out | top2 | last4_L27_attn_out | L23 mlp_out, L24 mlp_out | 20 | 0.978->3.825 | -2.847 | 3.65->43.70 | -40.05 | -29.600 | newline:19, symbol:1 |

### By Mode

| pair_task | site | combo | mode | components | n | pre_gap | post_gap | norm_gap_shift | post_rank | correct_top1_rate | pre_top1 | post_top1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | l22_peak_layer_out | top2 | baseline_task | L23 mlp_out, L24 mlp_out | 20 | 636.296 | 5.575 | -630.721 | 77.20 | 0.000 | word:20 | word:11, newline:9 |
| explanation_required | l22_peak_layer_out | top2 | combo_ablation | L23 mlp_out, L24 mlp_out | 20 | 395.944 | 0.978 | -394.966 | 3.65 | 0.150 | word:20 | space:15, correct_prefix:3, newline:2 |
| explanation_required | l22_peak_layer_out | top2 | last4_L24_attn_out | L23 mlp_out, L24 mlp_out | 20 | 418.327 | 1.925 | -416.402 | 7.75 | 0.050 | word:20 | space:15, newline:4, correct_prefix:1 |
| explanation_required | l22_peak_layer_out | top2 | last4_L24_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 395.944 | 0.978 | -394.966 | 3.65 | 0.150 | word:20 | space:15, correct_prefix:3, newline:2 |
| explanation_required | l22_peak_layer_out | top2 | last4_L25_attn_out | L23 mlp_out, L24 mlp_out | 20 | 314.669 | 2.350 | -312.319 | 5.40 | 0.000 | word:20 | space:20 |
| explanation_required | l22_peak_layer_out | top2 | last4_L25_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 316.066 | 2.206 | -313.859 | 5.05 | 0.000 | word:20 | space:20 |
| explanation_required | l22_peak_layer_out | top2 | last4_L26_attn_out | L23 mlp_out, L24 mlp_out | 20 | 294.312 | 1.712 | -292.600 | 7.80 | 0.150 | word:20 | space:12, newline:5, correct_prefix:3 |
| explanation_required | l22_peak_layer_out | top2 | last4_L26_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 393.706 | 0.806 | -392.900 | 3.00 | 0.450 | word:20 | space:11, correct_prefix:9 |
| explanation_required | l22_peak_layer_out | top2 | last4_L27_attn_out | L23 mlp_out, L24 mlp_out | 20 | 33.425 | 3.825 | -29.600 | 43.70 | 0.000 | space:11, word:4, newline:3, symbol:2 | newline:19, symbol:1 |
| explanation_required | l22_peak_layer_out | top2 | last4_L27_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 870.236 | 1.759 | -868.477 | 2.90 | 0.100 | word:20 | space:18, correct_prefix:2 |
| explanation_required | l22_peak_layer_out | top2 | site_restore | L23 mlp_out, L24 mlp_out | 20 | 622.595 | 2.194 | -620.402 | 8.30 | 0.100 | word:20 | space:10, newline:8, correct_prefix:2 |
| explanation_required | late_peak_layer_out | top2 | baseline_task | L23 mlp_out, L24 mlp_out | 20 | 636.296 | 5.575 | -630.721 | 77.20 | 0.000 | word:20 | word:11, newline:9 |
| explanation_required | late_peak_layer_out | top2 | combo_ablation | L23 mlp_out, L24 mlp_out | 20 | 395.944 | 0.978 | -394.966 | 3.65 | 0.150 | word:20 | space:15, correct_prefix:3, newline:2 |
| explanation_required | late_peak_layer_out | top2 | last4_L24_attn_out | L23 mlp_out, L24 mlp_out | 20 | 418.327 | 1.925 | -416.402 | 7.75 | 0.050 | word:20 | space:15, newline:4, correct_prefix:1 |
| explanation_required | late_peak_layer_out | top2 | last4_L24_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 395.944 | 0.978 | -394.966 | 3.65 | 0.150 | word:20 | space:15, correct_prefix:3, newline:2 |
| explanation_required | late_peak_layer_out | top2 | last4_L25_attn_out | L23 mlp_out, L24 mlp_out | 20 | 314.669 | 2.350 | -312.319 | 5.40 | 0.000 | word:20 | space:20 |
| explanation_required | late_peak_layer_out | top2 | last4_L25_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 316.066 | 2.206 | -313.859 | 5.05 | 0.000 | word:20 | space:20 |
| explanation_required | late_peak_layer_out | top2 | last4_L26_attn_out | L23 mlp_out, L24 mlp_out | 20 | 294.312 | 1.712 | -292.600 | 7.80 | 0.150 | word:20 | space:12, newline:5, correct_prefix:3 |
| explanation_required | late_peak_layer_out | top2 | last4_L26_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 393.706 | 0.806 | -392.900 | 3.00 | 0.450 | word:20 | space:11, correct_prefix:9 |
| explanation_required | late_peak_layer_out | top2 | last4_L27_attn_out | L23 mlp_out, L24 mlp_out | 20 | 33.425 | 3.825 | -29.600 | 43.70 | 0.000 | space:11, word:4, newline:3, symbol:2 | newline:19, symbol:1 |
| explanation_required | late_peak_layer_out | top2 | last4_L27_mlp_out | L23 mlp_out, L24 mlp_out | 20 | 870.236 | 1.759 | -868.477 | 2.90 | 0.100 | word:20 | space:18, correct_prefix:2 |
| explanation_required | late_peak_layer_out | top2 | site_restore | L23 mlp_out, L24 mlp_out | 20 | 622.595 | 2.194 | -620.402 | 8.30 | 0.100 | word:20 | space:10, newline:8, correct_prefix:2 |
| yes_no_required | l22_peak_layer_out | top1 | baseline_task | L24 mlp_out | 20 | 769.390 | 9.172 | -760.218 | 295.55 | 0.000 | word:20 | explanation:20 |
| yes_no_required | l22_peak_layer_out | top1 | combo_ablation | L24 mlp_out | 20 | 520.309 | 2.306 | -518.003 | 9.60 | 0.000 | word:20 | newline:17, space:3 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L24_attn_out | L24 mlp_out | 20 | 515.802 | 3.544 | -512.258 | 18.20 | 0.000 | word:20 | newline:14, space:6 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L24_mlp_out | L24 mlp_out | 20 | 520.309 | 2.306 | -518.003 | 9.60 | 0.000 | word:20 | newline:17, space:3 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L25_attn_out | L24 mlp_out | 20 | 419.900 | 1.953 | -417.947 | 7.10 | 0.050 | word:20 | space:11, newline:8, correct_prefix:1 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L25_mlp_out | L24 mlp_out | 20 | 424.376 | 2.253 | -422.123 | 8.50 | 0.000 | word:20 | space:18, newline:2 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L26_attn_out | L24 mlp_out | 20 | 413.101 | 3.291 | -409.810 | 21.65 | 0.000 | word:20 | newline:14, space:6 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L26_mlp_out | L24 mlp_out | 20 | 501.744 | 2.016 | -499.728 | 8.60 | 0.150 | word:20 | space:10, newline:7, correct_prefix:3 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L27_attn_out | L24 mlp_out | 20 | 134.200 | 2.831 | -131.369 | 30.30 | 0.000 | word:20 | newline:19, symbol:1 |
| yes_no_required | l22_peak_layer_out | top1 | last4_L27_mlp_out | L24 mlp_out | 20 | 941.139 | 2.789 | -938.350 | 7.90 | 0.000 | word:20 | space:20 |
| yes_no_required | l22_peak_layer_out | top1 | site_restore | L24 mlp_out | 20 | 626.797 | 2.812 | -623.984 | 13.85 | 0.000 | word:20 | newline:11, space:9 |
| yes_no_required | late_peak_layer_out | top1 | baseline_task | L24 mlp_out | 20 | 769.390 | 9.172 | -760.218 | 295.55 | 0.000 | word:20 | explanation:20 |
| yes_no_required | late_peak_layer_out | top1 | combo_ablation | L24 mlp_out | 20 | 520.309 | 2.306 | -518.003 | 9.60 | 0.000 | word:20 | newline:17, space:3 |
| yes_no_required | late_peak_layer_out | top1 | last4_L24_attn_out | L24 mlp_out | 20 | 515.802 | 3.544 | -512.258 | 18.20 | 0.000 | word:20 | newline:14, space:6 |
| yes_no_required | late_peak_layer_out | top1 | last4_L24_mlp_out | L24 mlp_out | 20 | 520.309 | 2.306 | -518.003 | 9.60 | 0.000 | word:20 | newline:17, space:3 |
| yes_no_required | late_peak_layer_out | top1 | last4_L25_attn_out | L24 mlp_out | 20 | 419.900 | 1.953 | -417.947 | 7.10 | 0.050 | word:20 | space:11, newline:8, correct_prefix:1 |
| yes_no_required | late_peak_layer_out | top1 | last4_L25_mlp_out | L24 mlp_out | 20 | 424.376 | 2.253 | -422.123 | 8.50 | 0.000 | word:20 | space:18, newline:2 |
| yes_no_required | late_peak_layer_out | top1 | last4_L26_attn_out | L24 mlp_out | 20 | 413.101 | 3.291 | -409.810 | 21.65 | 0.000 | word:20 | newline:14, space:6 |
| yes_no_required | late_peak_layer_out | top1 | last4_L26_mlp_out | L24 mlp_out | 20 | 501.744 | 2.016 | -499.728 | 8.60 | 0.150 | word:20 | space:10, newline:7, correct_prefix:3 |
| yes_no_required | late_peak_layer_out | top1 | last4_L27_attn_out | L24 mlp_out | 20 | 134.200 | 2.831 | -131.369 | 30.30 | 0.000 | word:20 | newline:19, symbol:1 |
| yes_no_required | late_peak_layer_out | top1 | last4_L27_mlp_out | L24 mlp_out | 20 | 941.139 | 2.789 | -938.350 | 7.90 | 0.000 | word:20 | space:20 |
| yes_no_required | late_peak_layer_out | top1 | site_restore | L24 mlp_out | 20 | 626.797 | 2.812 | -623.984 | 13.85 | 0.000 | word:20 | newline:11, space:9 |
