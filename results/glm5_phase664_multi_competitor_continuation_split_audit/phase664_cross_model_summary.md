# Phase 664 Cross-Model Summary

目标：从 pairwise projection intervention 推进到 multi-competitor readout margin，并审计 correct-prefix-top1 之后的 continuation failure。

## qwen3

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 0.73
- target_categories: `['space', 'newline', 'word', 'explanation']`
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 32, 'target_failure_seen': 3, 'fallback_used': 0, 'scanned': 33}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Actual Multi-Competitor State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_category | max_competitor | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | separator_input_edge | top1 | 32 | 0.719 | 0.781 | 1.44 | 0.184 | -0.184 | correct_prefix:25, word:5, space:2 | none:26, word:5, space:1 | exact_correct:23, first_token_competition_failure:7, correct_prefix_but_generation_wrong:2 |
| explanation_required | separator_input_edge | top2 | 32 | 0.875 | 0.938 | 1.06 | 0.023 | -0.023 | correct_prefix:30, word:2 | none:30, word:2 | exact_correct:28, correct_prefix_but_generation_wrong:2, first_token_competition_failure:2 |
| yes_no_required | early_peak_layer_out | top2 | 32 | 1.000 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:32 | none:32 | exact_correct:32 |
| yes_no_required | early_peak_layer_out | top3 | 32 | 0.906 | 0.906 | 1.12 | 0.016 | -0.016 | correct_prefix:29, explanation:2, newline:1 | none:29, explanation:2, newline:1 | exact_correct:29, first_token_competition_failure:3 |

### Multi-Competitor Failures

| pair_task | max_competitor | n | mean_multi_margin | winner_sets |
|---|---|---:|---:|---|
| explanation_required | word | 7 | -0.857 | word:6, space+word:1 |
| yes_no_required | explanation | 2 | -0.156 | explanation:2 |
| explanation_required | space | 1 | -0.625 | space:1 |
| yes_no_required | newline | 1 | -0.188 | newline:1 |

### Multi-Correction by Scale

| pair_task | max_competitor | scale | n | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_after | max_comp_after |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | space | 0.5 | 1 | 0.000 | 2.00 | 0.375 | -0.375 | space:1 | space:1 |
| explanation_required | space | 1.0 | 1 | 0.000 | 2.00 | 0.062 | -0.062 | space:1 | space:1 |
| explanation_required | space | 1.5 | 1 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:1 | none:1 |
| explanation_required | space | 2.0 | 1 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:1 | none:1 |
| explanation_required | word | 0.5 | 7 | 0.143 | 2.14 | 0.420 | -0.420 | word:6, correct_prefix:1 | word:6, none:1 |
| explanation_required | word | 1.0 | 7 | 0.571 | 1.14 | 0.027 | -0.027 | correct_prefix:4, word:3 | none:6, word:1 |
| explanation_required | word | 1.5 | 7 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:7 | none:7 |
| explanation_required | word | 2.0 | 7 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:7 | none:7 |
| yes_no_required | explanation | 0.5 | 2 | 0.000 | 2.50 | 0.125 | -0.125 | explanation:2 | explanation:2 |
| yes_no_required | explanation | 1.0 | 2 | 0.000 | 1.00 | 0.000 | 0.000 | explanation:2 | none:2 |
| yes_no_required | explanation | 1.5 | 2 | 0.500 | 1.00 | 0.000 | 0.000 | correct_prefix:1, explanation:1 | none:2 |
| yes_no_required | explanation | 2.0 | 2 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:2 | none:2 |
| yes_no_required | newline | 0.5 | 1 | 0.000 | 2.00 | 0.188 | -0.188 | newline:1 | newline:1 |
| yes_no_required | newline | 1.0 | 1 | 0.000 | 2.00 | 0.062 | -0.062 | newline:1 | newline:1 |
| yes_no_required | newline | 1.5 | 1 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:1 | none:1 |
| yes_no_required | newline | 2.0 | 1 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:1 | none:1 |

### Continuation Audit

| pair_task | site | combo | n | token1_match_rate | token2_match_rate | mean_token1_expected_rank | mean_token2_expected_rank | generated_text |
|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | separator_input_edge | top2 | 2 | 0.000 | 0.000 | 2.00 | 6.50 | 22\n\n:2 |
| explanation_required | separator_input_edge | top1 | 2 | 0.000 | 0.000 | 2.00 | 6.50 | 22\n\n:2 |

## glm4

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 1.11
- target_categories: `['space', 'newline', 'word', 'explanation']`
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 33, 'target_failure_seen': 2, 'fallback_used': 0, 'scanned': 34}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Actual Multi-Competitor State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_category | max_competitor | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | l22_peak_layer_out | top1 | 32 | 0.781 | 0.781 | 1.28 | 0.131 | -0.131 | correct_prefix:25, space:6, word:1 | none:26, space:5, word:1 | exact_correct:25, first_token_competition_failure:7 |
| explanation_required | late_peak_layer_out | top1 | 32 | 0.719 | 0.781 | 1.28 | 0.131 | -0.131 | correct_prefix:25, space:6, word:1 | none:26, space:5, word:1 | exact_correct:23, first_token_competition_failure:7, correct_prefix_but_generation_wrong:2 |
| yes_no_required | l22_peak_layer_out | top2 | 32 | 0.688 | 0.719 | 1.44 | 0.164 | -0.164 | correct_prefix:23, word:5, space:4 | none:23, word:5, space:4 | exact_correct:22, first_token_competition_failure:9, correct_prefix_but_generation_wrong:1 |
| yes_no_required | late_peak_layer_out | top2 | 32 | 0.688 | 0.719 | 1.44 | 0.164 | -0.164 | correct_prefix:23, word:5, space:4 | none:23, word:5, space:4 | exact_correct:22, first_token_competition_failure:9, correct_prefix_but_generation_wrong:1 |

### Multi-Competitor Failures

| pair_task | max_competitor | n | mean_multi_margin | winner_sets |
|---|---|---:|---:|---|
| explanation_required | space | 10 | -0.825 | space+word:6, space:4 |
| yes_no_required | word | 10 | -0.263 | word:8, space+word:2 |
| yes_no_required | space | 8 | -0.984 | space+word:6, space:2 |
| explanation_required | word | 2 | -0.062 | word:2 |

### Multi-Correction by Scale

| pair_task | max_competitor | scale | n | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_after | max_comp_after |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | space | 0.5 | 10 | 0.000 | 2.00 | 0.375 | -0.375 | space:10 | space:10 |
| explanation_required | space | 1.0 | 10 | 0.600 | 1.40 | 0.037 | -0.037 | correct_prefix:6, space:4 | none:6, space:4 |
| explanation_required | space | 1.5 | 10 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:10 | none:10 |
| explanation_required | space | 2.0 | 10 | 0.800 | 1.00 | 0.000 | 0.000 | correct_prefix:8, space:2 | none:10 |
| explanation_required | word | 0.5 | 2 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:2 | none:2 |
| explanation_required | word | 1.0 | 2 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:2 | none:2 |
| explanation_required | word | 1.5 | 2 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:2 | none:2 |
| explanation_required | word | 2.0 | 2 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:2 | none:2 |
| yes_no_required | space | 0.5 | 8 | 0.000 | 2.00 | 0.453 | -0.453 | space:8 | space:8 |
| yes_no_required | space | 1.0 | 8 | 0.750 | 1.25 | 0.016 | -0.016 | correct_prefix:6, space:2 | none:6, space:2 |
| yes_no_required | space | 1.5 | 8 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:8 | none:8 |
| yes_no_required | space | 2.0 | 8 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:8 | none:8 |
| yes_no_required | word | 0.5 | 10 | 0.000 | 2.40 | 0.175 | -0.175 | word:10 | word:10 |
| yes_no_required | word | 1.0 | 10 | 0.400 | 1.40 | 0.013 | -0.013 | correct_prefix:4, word:4, space:2 | none:8, space:2 |
| yes_no_required | word | 1.5 | 10 | 0.800 | 1.00 | 0.000 | 0.000 | correct_prefix:8, word:2 | none:10 |
| yes_no_required | word | 2.0 | 10 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:10 | none:10 |

### Continuation Audit

| pair_task | site | combo | n | token1_match_rate | token2_match_rate | mean_token1_expected_rank | mean_token2_expected_rank | generated_text |
|---|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | late_peak_layer_out | top1 | 2 | 0.000 | 0.000 | 2.00 | 0.00 | 05\n\nReason:2 |
| yes_no_required | l22_peak_layer_out | top2 | 1 | 1.000 | 0.000 | 1.00 | 0.00 | 22.\n\nQuestion:1 |
| yes_no_required | late_peak_layer_out | top2 | 1 | 1.000 | 0.000 | 1.00 | 0.00 | 22.\n\nQuestion:1 |

## deepseek7b

- raw_cases: 384 / selected_items: 32 / rows: 128 / total_time_min: 0.96
- target_categories: `['space', 'newline', 'word', 'explanation']`
- selection: `{'mode_v_correct_seen': 32, 'repair_correct_seen': 33, 'target_failure_seen': 10, 'fallback_used': 0, 'scanned': 37}` / filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}`

### Actual Multi-Competitor State

| pair_task | site | combo | n | exact_rate | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_category | max_competitor | continuation_tag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | l22_peak_layer_out | top2 | 32 | 0.500 | 0.500 | 2.59 | 0.727 | -0.727 | space:16, correct_prefix:16 | space:16, none:16 | first_token_competition_failure:16, exact_correct:16 |
| explanation_required | late_peak_layer_out | top2 | 32 | 0.500 | 0.500 | 2.59 | 0.727 | -0.727 | space:16, correct_prefix:16 | space:16, none:16 | first_token_competition_failure:16, exact_correct:16 |
| yes_no_required | l22_peak_layer_out | top1 | 32 | 0.469 | 0.469 | 3.38 | 1.014 | -1.014 | correct_prefix:15, space:13, newline:4 | none:16, space:12, newline:4 | first_token_competition_failure:17, exact_correct:15 |
| yes_no_required | late_peak_layer_out | top1 | 32 | 0.469 | 0.469 | 3.38 | 1.014 | -1.014 | correct_prefix:15, space:13, newline:4 | none:16, space:12, newline:4 | first_token_competition_failure:17, exact_correct:15 |

### Multi-Competitor Failures

| pair_task | max_competitor | n | mean_multi_margin | winner_sets |
|---|---|---:|---:|---|
| explanation_required | space | 32 | -1.453 | space:14, space+newline:10, space+newline+word:8 |
| yes_no_required | space | 24 | -1.750 | space+newline:16, space+newline+word+explanation:4, space:2, space+newline+explanation:2 |
| yes_no_required | newline | 8 | -2.859 | space+newline+explanation:4, space+newline+word+explanation:4 |

### Multi-Correction by Scale

| pair_task | max_competitor | scale | n | correct_top1_rate | mean_rank | mean_gap | mean_multi_margin | top1_after | max_comp_after |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| explanation_required | space | 0.5 | 32 | 0.000 | 2.50 | 0.641 | -0.641 | space:32 | space:30, none:2 |
| explanation_required | space | 1.0 | 32 | 0.562 | 1.12 | 0.016 | -0.016 | correct_prefix:18, space:14 | none:28, space:4 |
| explanation_required | space | 1.5 | 32 | 0.938 | 1.00 | 0.000 | 0.000 | correct_prefix:30, space:2 | none:32 |
| explanation_required | space | 2.0 | 32 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:32 | none:32 |
| yes_no_required | newline | 0.5 | 8 | 0.000 | 3.00 | 0.625 | -0.625 | newline:8 | newline:6, space:2 |
| yes_no_required | newline | 1.0 | 8 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:8 | none:8 |
| yes_no_required | newline | 1.5 | 8 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:8 | none:8 |
| yes_no_required | newline | 2.0 | 8 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:8 | none:8 |
| yes_no_required | space | 0.5 | 24 | 0.000 | 2.58 | 0.573 | -0.573 | space:20, newline:4 | space:24 |
| yes_no_required | space | 1.0 | 24 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:24 | none:24 |
| yes_no_required | space | 1.5 | 24 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:24 | none:24 |
| yes_no_required | space | 2.0 | 24 | 1.000 | 1.00 | 0.000 | 0.000 | correct_prefix:24 | none:24 |

### Continuation Audit

| pair_task | site | combo | n | token1_match_rate | token2_match_rate | mean_token1_expected_rank | mean_token2_expected_rank | generated_text |
|---|---|---|---:|---:|---:|---:|---:|---|
