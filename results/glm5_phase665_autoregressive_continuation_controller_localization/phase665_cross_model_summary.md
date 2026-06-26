# Phase 665 Cross-Model Summary

目标：定位 correct_prefix top1 but exact wrong 后的真实自回归续写控制器，扫描 token1/token2 的 continuation-position source patch。

## qwen3

- raw_cases: 512 / selected_items: 64 / continuation_failures: 5 / rows: 650 / total_time_min: 1.76
- scan_layers: `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`
- scan_components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- selection: `{'mode_v_correct_seen': 64, 'repair_correct_seen': 64, 'target_failure_seen': 4, 'fallback_used': 0, 'scanned': 65}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'short_answer_ids': 0}`

### Selected Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| explanation_required | separator_input_edge | top1 | 3 |  v22\n\nWait,:1,  v22\n\nOkay,:1,  05\n\nThe answer:1 |
| explanation_required | separator_input_edge | top2 | 2 |  v22\n\nWait,:1,  v22\n\nOkay,:1 |

### Continuation Baselines

| pair_task | site | combo | step | n | expected_top1_rate | mean_expected_rank | mean_expected_minus_top1 | top1_text |
|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top1 | 1 | 3 | 0.333 | 1.67 | -0.833 | 2:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 2 | 3 | 1.000 | 1.00 | 0.000 | 8:2, 5:1 |
| explanation_required | separator_input_edge | top2 | 1 | 2 | 0.000 | 2.00 | -1.875 | 2:2 |
| explanation_required | separator_input_edge | top2 | 2 | 2 | 1.000 | 1.00 | 0.000 | 8:2 |

### Top Component Patch Candidates

| pair_task | site | combo | step | layer | component | n | flip_rate | mean_rank_improvement | mean_margin_delta | patched_top1 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top2 | 1 | 22 | attn_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 23 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 24 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 24 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 25 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 25 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 26 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 26 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 27 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 27 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 28 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 28 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 29 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 29 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 30 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 30 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 31 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 31 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 32 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 32 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 33 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 33 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 34 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 34 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 35 | layer_input | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 35 | layer_out | 2 | 1.000 | 1.00 | 1.875 | 4:2 |
| explanation_required | separator_input_edge | top2 | 1 | 23 | attn_out | 2 | 0.500 | 0.50 | 1.688 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 29 | attn_out | 2 | 0.500 | 0.50 | 1.688 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 24 | attn_out | 2 | 0.500 | 0.50 | 1.500 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 34 | mlp_out | 2 | 0.500 | 0.50 | 1.500 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 25 | mlp_out | 2 | 0.500 | 0.50 | 0.938 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 22 | layer_out | 2 | 0.500 | 0.50 | 0.875 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 23 | layer_input | 2 | 0.500 | 0.50 | 0.875 | 4:1, 2:1 |
| explanation_required | separator_input_edge | top2 | 1 | 30 | attn_out | 2 | 0.000 | 0.00 | 0.875 | 2:2 |
| explanation_required | separator_input_edge | top1 | 1 | 22 | attn_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 23 | attn_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 23 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 24 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 24 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 25 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 25 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 26 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 26 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 27 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 27 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 28 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 28 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 29 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 29 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 30 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 30 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 31 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 31 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 32 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 32 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 33 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 33 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 34 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 34 | layer_out | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |
| explanation_required | separator_input_edge | top1 | 1 | 35 | layer_input | 3 | 0.667 | 0.67 | 0.833 | 4:2, 0:1 |

## glm4

- raw_cases: 512 / selected_items: 64 / continuation_failures: 4 / rows: 292 / total_time_min: 2.27
- scan_layers: `[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]`
- scan_components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- selection: `{'mode_v_correct_seen': 64, 'repair_correct_seen': 65, 'target_failure_seen': 4, 'fallback_used': 0, 'scanned': 66}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'short_answer_ids': 0}`

### Selected Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| explanation_required | late_peak_layer_out | top1 | 3 |  v05\n\nReason: According:2,  22\n\nReason: The:1 |
| explanation_required | l22_peak_layer_out | top1 | 1 |  22\n\nReason: The:1 |

### Continuation Baselines

| pair_task | site | combo | step | n | expected_top1_rate | mean_expected_rank | mean_expected_minus_top1 | top1_text |
|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top1 | 1 | 1 | 1.000 | 1.00 | 0.000 | 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 3 | 0.333 | 1.67 | -0.375 | 05:2, 22:1 |

### Top Component Patch Candidates

| pair_task | site | combo | step | layer | component | n | flip_rate | mean_rank_improvement | mean_margin_delta | patched_top1 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| explanation_required | late_peak_layer_out | top1 | 1 | 22 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 22 | attn_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 22 | mlp_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 22 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 23 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 23 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 24 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 24 | attn_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 24 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 25 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 25 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 26 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 26 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 27 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 27 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 28 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 28 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 29 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 29 | attn_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 29 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 30 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 30 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 31 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 31 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 32 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 32 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 33 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 33 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 34 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 34 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 35 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 35 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 36 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 36 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 37 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 37 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 38 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 38 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 39 | layer_input | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 39 | layer_out | 3 | 0.667 | 0.67 | 0.375 | 22:3 |
| explanation_required | late_peak_layer_out | top1 | 1 | 37 | mlp_out | 3 | 0.333 | 0.33 | 0.333 | 22:2, 05:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 28 | attn_out | 3 | 0.333 | 0.33 | 0.292 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 25 | mlp_out | 3 | 0.000 | 0.00 | 0.292 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 32 | attn_out | 3 | 0.000 | 0.00 | 0.250 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 26 | mlp_out | 3 | 0.000 | 0.00 | 0.229 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 33 | mlp_out | 3 | 0.000 | 0.00 | 0.229 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 34 | mlp_out | 3 | 0.000 | 0.00 | 0.229 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 24 | mlp_out | 3 | 0.000 | 0.00 | 0.188 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 35 | mlp_out | 3 | 0.000 | 0.00 | 0.188 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 32 | mlp_out | 3 | 0.000 | 0.00 | 0.125 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 27 | mlp_out | 3 | 0.000 | 0.00 | 0.083 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 38 | mlp_out | 3 | 0.000 | 0.00 | 0.083 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 39 | mlp_out | 3 | 0.000 | 0.00 | 0.083 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 29 | mlp_out | 3 | 0.000 | 0.00 | 0.062 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 28 | mlp_out | 3 | 0.000 | 0.00 | 0.042 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 36 | mlp_out | 3 | 0.000 | 0.00 | 0.021 | 05:2, 22:1 |
| explanation_required | late_peak_layer_out | top1 | 1 | 38 | attn_out | 3 | 0.000 | 0.00 | 0.021 | 05:2, 22:1 |
| explanation_required | l22_peak_layer_out | top1 | 1 | 22 | layer_input | 1 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | 1 | 22 | attn_out | 1 | 0.000 | 0.00 | 0.000 | 22:1 |
| explanation_required | l22_peak_layer_out | top1 | 1 | 22 | mlp_out | 1 | 0.000 | 0.00 | 0.000 | 22:1 |

## deepseek7b

- raw_cases: 512 / selected_items: 64 / continuation_failures: 12 / rows: 696 / total_time_min: 2.08
- scan_layers: `[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- scan_components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- selection: `{'mode_v_correct_seen': 64, 'repair_correct_seen': 66, 'target_failure_seen': 20, 'fallback_used': 0, 'scanned': 71}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'short_answer_ids': 0}`

### Selected Continuation Failures

| pair_task | site | combo | n | generation_text |
|---|---|---|---:|---|
| explanation_required | l22_peak_layer_out | top2 | 3 |  22\nBut why:1,  v05.\n\nBut wait:1,  v05 or v4:1 |
| explanation_required | late_peak_layer_out | top2 | 3 |  22\nBut why:1,  v05.\n\nBut wait:1,  v05 or v4:1 |
| yes_no_required | l22_peak_layer_out | top1 | 3 |  48.\n\nQuestion::3 |
| yes_no_required | late_peak_layer_out | top1 | 3 |  48.\n\nQuestion::3 |

### Continuation Baselines

| pair_task | site | combo | step | n | expected_top1_rate | mean_expected_rank | mean_expected_minus_top1 | top1_text |
|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top2 | 1 | 3 | 0.333 | 2.00 | -1.042 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 2 | 3 | 1.000 | 1.00 | 0.000 | 8:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 3 | 0.333 | 1.67 | -0.771 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 2 | 3 | 1.000 | 1.00 | 0.000 | 8:2, 2:1 |
| yes_no_required | l22_peak_layer_out | top1 | 1 | 3 | 0.667 | 1.33 | -0.417 | 4:2, 0:1 |
| yes_no_required | l22_peak_layer_out | top1 | 2 | 3 | 1.000 | 1.00 | 0.000 | 8:3 |
| yes_no_required | late_peak_layer_out | top1 | 1 | 3 | 0.667 | 1.33 | -0.292 | 4:2, 0:1 |
| yes_no_required | late_peak_layer_out | top1 | 2 | 3 | 1.000 | 1.00 | 0.000 | 8:3 |

### Top Component Patch Candidates

| pair_task | site | combo | step | layer | component | n | flip_rate | mean_rank_improvement | mean_margin_delta | patched_top1 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| explanation_required | l22_peak_layer_out | top2 | 1 | 21 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 22 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 22 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 23 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 23 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 24 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 24 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 25 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 4:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 25 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 26 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 26 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 27 | layer_input | 3 | 0.667 | 1.00 | 1.042 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 27 | layer_out | 3 | 0.667 | 1.00 | 1.042 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 20 | layer_out | 3 | 0.333 | 0.67 | 0.958 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 21 | layer_input | 3 | 0.333 | 0.67 | 0.958 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 18 | layer_out | 3 | 0.333 | 0.67 | 0.938 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 19 | layer_input | 3 | 0.333 | 0.67 | 0.938 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 19 | layer_out | 3 | 0.333 | 0.67 | 0.854 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 20 | layer_input | 3 | 0.333 | 0.67 | 0.854 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 14 | layer_input | 3 | 0.000 | 0.33 | 0.812 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 17 | layer_out | 3 | 0.000 | 0.33 | 0.792 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 18 | layer_input | 3 | 0.000 | 0.33 | 0.792 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 18 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 19 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 20 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 21 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 21 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 22 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 22 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 23 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 23 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 24 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 24 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 25 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 4:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 25 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 26 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 26 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 27 | layer_input | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 27 | layer_out | 3 | 0.667 | 0.67 | 0.771 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 14 | layer_out | 3 | 0.000 | 0.33 | 0.771 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 15 | layer_input | 3 | 0.000 | 0.33 | 0.771 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 14 | layer_input | 3 | 0.333 | 0.33 | 0.729 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 19 | layer_out | 3 | 0.333 | 0.33 | 0.729 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 20 | layer_input | 3 | 0.333 | 0.33 | 0.729 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 15 | layer_out | 3 | 0.000 | 0.33 | 0.729 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 16 | layer_input | 3 | 0.000 | 0.33 | 0.729 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 16 | layer_out | 3 | 0.000 | 0.33 | 0.729 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 17 | layer_input | 3 | 0.000 | 0.33 | 0.729 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 14 | layer_out | 3 | 0.333 | 0.33 | 0.708 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 15 | layer_input | 3 | 0.333 | 0.33 | 0.708 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 15 | layer_out | 3 | 0.333 | 0.33 | 0.688 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 16 | layer_input | 3 | 0.333 | 0.33 | 0.688 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 16 | layer_out | 3 | 0.333 | 0.33 | 0.667 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 17 | layer_input | 3 | 0.333 | 0.33 | 0.667 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 17 | layer_out | 3 | 0.333 | 0.33 | 0.667 | 2:1, 0:1, 4:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 18 | layer_input | 3 | 0.333 | 0.33 | 0.667 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 18 | mlp_out | 3 | 0.000 | 0.33 | 0.625 | 0:2, 2:1 |
| explanation_required | late_peak_layer_out | top2 | 1 | 18 | mlp_out | 3 | 0.333 | 0.33 | 0.604 | 2:1, 0:1, 4:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 23 | attn_out | 3 | 0.000 | 0.33 | 0.604 | 0:2, 2:1 |
| explanation_required | l22_peak_layer_out | top2 | 1 | 22 | attn_out | 3 | 0.000 | 0.33 | 0.500 | 0:2, 2:1 |
