# Phase 650 Cross-Model Summary

目标：把 Phase 649 的 label/separator/relation_tail protocol field 放入跨模板与副作用边界，检查目标修复、模板泛化、以及对非目标语言状态的旧值吸附风险。

说明：relation_changed / explanation_needed / non_value 的 exact 表示仍输出旧正确值，在这些 split 中应视为短答值吸附风险，不是正向成功率。

## qwen3

- raw_cases: 320 / selected_items: 40 / mode_rows: 8880
- max_per_split: 8 / templates: `['Answer', 'Response', 'Value']`
- positions: `['label_aligned', 'label_colon', 'separator', 'relation_tail']` / interval_specs: `[{'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'layer_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'attn_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'mlp_out'}]`
- selection_stats: `{'target_failure_seen': 8, 'original_correct_seen': 112, 'counts': {'target_failure': 8, 'original_correct': 8, 'relation_changed': 8, 'explanation_needed': 8, 'non_value': 8}}`
- filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}` / total_time_min: 17.45

### Baselines

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | original | 8 |  |  |  |  | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 1.047 | correct_prefix:5, space:3 |
| target_failure | Answer | inline | 8 |  |  |  |  | 0/8 | 0/8 | 1/8 | 4/8 | 4/8 | 0/8 | 4.4 | -1.234 | newline:4, space:3, correct_prefix:1 |
| target_failure | Response | original | 8 |  |  |  |  | 4/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 4.078 | correct_prefix:6, space:2 |
| target_failure | Response | inline | 8 |  |  |  |  | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.062 | correct_prefix:5, space:3 |
| target_failure | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 7/8 | 2/8 | 0/8 | 11.8 | -5.453 | newline:7, space:1 |
| target_failure | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.4 | -5.766 | newline:8 |
| original_correct | Answer | original | 8 |  |  |  |  | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.828 | correct_prefix:6, space:2 |
| original_correct | Answer | inline | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 2/8 | 6/8 | 0/8 | 3.4 | -0.641 | space:3, correct_prefix:3, newline:2 |
| original_correct | Response | original | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| original_correct | Response | inline | 8 |  |  |  |  | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 3.344 | space:4, correct_prefix:4 |
| original_correct | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 8.0 | -4.859 | newline:8 |
| original_correct | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 11.1 | -4.953 | newline:8 |
| relation_changed | Answer | original | 8 |  |  |  |  | 4/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.828 | correct_prefix:6, space:2 |
| relation_changed | Answer | inline | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 2/8 | 6/8 | 0/8 | 3.4 | -0.641 | space:3, correct_prefix:3, newline:2 |
| relation_changed | Response | original | 8 |  |  |  |  | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| relation_changed | Response | inline | 8 |  |  |  |  | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 3.344 | space:4, correct_prefix:4 |
| relation_changed | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 8.0 | -4.859 | newline:8 |
| relation_changed | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 11.1 | -4.953 | newline:8 |
| explanation_needed | Answer | original | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.234 | correct_prefix:8 |
| explanation_needed | Answer | inline | 8 |  |  |  |  | 1/8 | 0/8 | 1/8 | 7/8 | 5/8 | 0/8 | 2.1 | -0.219 | newline:7, correct_prefix:1 |
| explanation_needed | Response | original | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | inline | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.469 | correct_prefix:8 |
| explanation_needed | Value | original | 8 |  |  |  |  | 1/8 | 0/8 | 1/8 | 0/8 | 8/8 | 0/8 | 3.4 | -1.047 | space:7, correct_prefix:1 |
| explanation_needed | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 8.4 | -4.266 | newline:8 |
| non_value | Answer | original | 8 |  |  |  |  | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 2.5 | 0.719 | explanation:7, correct_prefix:1 |
| non_value | Answer | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 8/8 | 0/8 | 12.1 | -3.141 | newline:8 |
| non_value | Response | original | 8 |  |  |  |  | 2/8 | 0/8 | 1/8 | 0/8 | 4/8 | 0/8 | 2.5 | 0.078 | explanation:5, space:2, correct_prefix:1 |
| non_value | Response | inline | 8 |  |  |  |  | 2/8 | 0/8 | 1/8 | 1/8 | 8/8 | 0/8 | 3.6 | -0.500 | space:5, newline:1, correct_prefix:1, explanation:1 |
| non_value | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.0 | 3.906 | space:8 |
| non_value | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.0 | 2.109 | space:8 |

### Target Failure Best Sufficiency

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.375 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.375 | correct_prefix:8 |
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.578 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.266 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.266 | correct_prefix:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.859 | correct_prefix:7, space:1 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.859 | correct_prefix:7, space:1 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.672 | correct_prefix:7, word:1 |
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 0/8 | 8/8 | 0/8 | 5/8 | 0/8 | 1.0 | 3.906 | correct_prefix:8 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.703 | correct_prefix:6, word:2 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.062 | correct_prefix:5, space:3 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.062 | correct_prefix:5, space:3 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.031 | correct_prefix:4, space:4 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.031 | correct_prefix:4, space:4 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.984 | correct_prefix:5, space:3 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.8 | 1.453 | space:5, correct_prefix:3 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.250 | word:5, correct_prefix:3 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.250 | word:5, correct_prefix:3 |
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.1 | 0.828 | word:8 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.2 | 0.609 | word:8 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 1/8 | 7/8 | 0/8 | 8.4 | -3.000 | word:7, newline:1 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 2/8 | 6/8 | 0/8 | 9.2 | -2.969 | word:6, newline:2 |
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 1/8 | 4/8 | 4/8 | 0/8 | 4.4 | -1.234 | newline:4, space:3, correct_prefix:1 |
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 1/8 | 4/8 | 4/8 | 0/8 | 4.4 | -1.234 | newline:4, space:3, correct_prefix:1 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 7/8 | 1/8 | 0/8 | 11.1 | -5.547 | newline:7, space:1 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 7/8 | 1/8 | 0/8 | 11.1 | -5.547 | newline:7, space:1 |
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 1/8 | 0/8 | 3.9 | -1.234 | newline:8 |
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 1/8 | 0/8 | 3.9 | -1.234 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 9.1 | -3.203 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 9.1 | -3.203 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 14.6 | -5.266 | newline:8 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.4 | -5.766 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.4 | -5.766 | newline:8 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.6 | -5.359 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 17.2 | -6.734 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 17.2 | -6.734 | newline:8 |

### Largest Old-Value Side Effects

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Answer | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.656 | correct_prefix:8 |
| original_correct | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.219 | correct_prefix:8 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.500 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.141 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.500 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.141 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| explanation_needed | Answer | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.391 | correct_prefix:8 |
| explanation_needed | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.234 | correct_prefix:8 |
| explanation_needed | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.141 | correct_prefix:8 |
| explanation_needed | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.234 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.344 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.547 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.047 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.984 | correct_prefix:8 |
| explanation_needed | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.344 | correct_prefix:8 |
| explanation_needed | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.547 | correct_prefix:8 |
| explanation_needed | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.047 | correct_prefix:8 |
| explanation_needed | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.984 | correct_prefix:8 |
| explanation_needed | Response | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 0.797 | correct_prefix:8 |
| explanation_needed | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.641 | correct_prefix:8 |
| explanation_needed | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.156 | correct_prefix:8 |
| explanation_needed | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.469 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.469 | correct_prefix:8 |
| original_correct | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.203 | correct_prefix:8 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.078 | correct_prefix:8 |
| original_correct | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| original_correct | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| original_correct | Response | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 2.750 | correct_prefix:7, space:1 |
| original_correct | Response | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 2.750 | correct_prefix:7, space:1 |
| original_correct | Response | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.859 | correct_prefix:7, word:1 |
| original_correct | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.547 | correct_prefix:7, word:1 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.016 | correct_prefix:7, word:1 |
| original_correct | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.578 | correct_prefix:7, word:1 |
| explanation_needed | Answer | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.750 | correct_prefix:7, space:1 |
| original_correct | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.828 | correct_prefix:6, space:2 |
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.828 | correct_prefix:6, space:2 |
| explanation_needed | Answer | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 2.766 | correct_prefix:6, word:2 |
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.188 | correct_prefix:6, space:2 |
| explanation_needed | Response | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 0.844 | correct_prefix:6, space:2 |
| relation_changed | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 2/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.203 | correct_prefix:8 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 5/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.391 | correct_prefix:6, word:2 |
| explanation_needed | Answer | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 2.609 | correct_prefix:5, word:3 |
| explanation_needed | Response | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.047 | correct_prefix:5, word:3 |

### Position Overview

#### label_aligned

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.375 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.266 | correct_prefix:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.859 | correct_prefix:7, space:1 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.031 | correct_prefix:4, space:4 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.250 | word:5, correct_prefix:3 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 7/8 | 1/8 | 0/8 | 11.1 | -5.547 | newline:7, space:1 |
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 1/8 | 0/8 | 3.9 | -1.234 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 9.1 | -3.203 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 17.2 | -6.734 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| relation_changed | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| relation_changed | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| explanation_needed | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.500 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.141 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| non_value | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 4/8 | 1.5 | 2.000 | correct_prefix:4, word:4 |
| non_value | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 0/8 | 3/8 | 2/8 | 8/8 | 0/8 | 2.5 | -0.297 | correct_prefix:3, space:2, newline:2, explanation:1 |
| non_value | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 1/8 | 0/8 | 0/8 | 0/8 | 1/8 | 5/8 | 2.6 | 3.312 | word:8 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.0 | 2.375 | space:8 |

#### label_colon

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.375 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.266 | correct_prefix:8 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.859 | correct_prefix:7, space:1 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.031 | correct_prefix:4, space:4 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.250 | word:5, correct_prefix:3 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 7/8 | 1/8 | 0/8 | 11.1 | -5.547 | newline:7, space:1 |
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 1/8 | 0/8 | 3.9 | -1.234 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 9.1 | -3.203 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 17.2 | -6.734 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| original_correct | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| relation_changed | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.734 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.453 | correct_prefix:8 |
| relation_changed | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.500 | correct_prefix:7, space:1 |
| explanation_needed | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.500 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.141 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| explanation_needed | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| non_value | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 4/8 | 1.5 | 2.000 | correct_prefix:4, word:4 |
| non_value | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 0/8 | 3/8 | 2/8 | 8/8 | 0/8 | 2.5 | -0.297 | correct_prefix:3, space:2, newline:2, explanation:1 |
| non_value | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 1/8 | 0/8 | 0/8 | 0/8 | 1/8 | 5/8 | 2.6 | 3.312 | word:8 |
| non_value | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.0 | 2.375 | space:8 |

#### separator

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.578 | correct_prefix:8 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.672 | correct_prefix:7, word:1 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.062 | correct_prefix:5, space:3 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.8 | 1.453 | space:5, correct_prefix:3 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.2 | 0.609 | word:8 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 2/8 | 6/8 | 0/8 | 9.2 | -2.969 | word:6, newline:2 |
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 1/8 | 4/8 | 4/8 | 0/8 | 4.4 | -1.234 | newline:4, space:3, correct_prefix:1 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.4 | -5.766 | newline:8 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.6 | -5.359 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.656 | correct_prefix:8 |
| original_correct | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.219 | correct_prefix:8 |
| relation_changed | Answer | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.656 | correct_prefix:8 |
| relation_changed | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.531 | correct_prefix:8 |
| relation_changed | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| relation_changed | Response | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.219 | correct_prefix:8 |
| explanation_needed | Answer | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.391 | correct_prefix:8 |
| explanation_needed | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.234 | correct_prefix:8 |
| explanation_needed | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 5.141 | correct_prefix:8 |
| explanation_needed | Response | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 0.797 | correct_prefix:8 |
| non_value | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 3/8 | 0/8 | 1.9 | 3.031 | word:5, correct_prefix:3 |
| non_value | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 2/8 | 6/8 | 2.4 | 0.406 | word:6, correct_prefix:2 |
| non_value | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 2/8 | 0/8 | 1/8 | 0/8 | 4/8 | 0/8 | 2.5 | 0.078 | explanation:5, space:2, correct_prefix:1 |
| non_value | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 2/8 | 0/8 | 1/8 | 1/8 | 8/8 | 0/8 | 3.6 | -0.500 | space:5, newline:1, correct_prefix:1, explanation:1 |

#### relation_tail

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 0/8 | 8/8 | 0/8 | 5/8 | 0/8 | 1.0 | 3.906 | correct_prefix:8 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.703 | correct_prefix:6, word:2 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.062 | correct_prefix:5, space:3 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.984 | correct_prefix:5, space:3 |
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.1 | 0.828 | word:8 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 1/8 | 7/8 | 0/8 | 8.4 | -3.000 | word:7, newline:1 |
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 1/8 | 4/8 | 4/8 | 0/8 | 4.4 | -1.234 | newline:4, space:3, correct_prefix:1 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 14.6 | -5.266 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 15.4 | -5.766 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| original_correct | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.203 | correct_prefix:8 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.016 | correct_prefix:7, word:1 |
| relation_changed | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 5/8 | 2/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.203 | correct_prefix:8 |
| relation_changed | Answer | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| relation_changed | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| relation_changed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 4/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 4.578 | correct_prefix:7, word:1 |
| explanation_needed | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.234 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.469 | correct_prefix:8 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.688 | correct_prefix:8 |
| non_value | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 1.9 | 3.328 | word:4, correct_prefix:4 |
| non_value | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 2/8 | 0/8 | 1/8 | 0/8 | 4/8 | 0/8 | 2.5 | 0.078 | explanation:5, space:2, correct_prefix:1 |
| non_value | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 2/8 | 0/8 | 1/8 | 1/8 | 8/8 | 0/8 | 3.6 | -0.500 | space:5, newline:1, correct_prefix:1, explanation:1 |
| non_value | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 2/8 | 0/8 | 2/8 | 2/8 | 2/8 | 4/8 | 2.4 | 0.266 | word:4, newline:2, correct_prefix:2 |

## glm4

- raw_cases: 320 / selected_items: 40 / mode_rows: 8880
- max_per_split: 8 / templates: `['Answer', 'Response', 'Value']`
- positions: `['label_aligned', 'label_colon', 'separator', 'relation_tail']` / interval_specs: `[{'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'layer_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'attn_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'mlp_out'}]`
- selection_stats: `{'target_failure_seen': 8, 'original_correct_seen': 73, 'counts': {'target_failure': 8, 'original_correct': 8, 'relation_changed': 8, 'explanation_needed': 8, 'non_value': 8}}`
- filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}` / total_time_min: 20.07

### Baselines

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | original | 8 |  |  |  |  | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 75.414 | correct_prefix:7, word:1 |
| target_failure | Answer | inline | 8 |  |  |  |  | 5/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 51.719 | correct_prefix:7, word:1 |
| target_failure | Response | original | 8 |  |  |  |  | 5/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:7, explanation:1 |
| target_failure | Response | inline | 8 |  |  |  |  | 4/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:6, word:2 |
| target_failure | Value | original | 8 |  |  |  |  | 5/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.629 | correct_prefix:7, word:1 |
| target_failure | Value | inline | 8 |  |  |  |  | 4/8 | 1/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.375 | correct_prefix:6, space:2 |
| original_correct | Answer | original | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Answer | inline | 8 |  |  |  |  | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| original_correct | Response | original | 8 |  |  |  |  | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| original_correct | Response | inline | 8 |  |  |  |  | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.4 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| original_correct | Value | original | 8 |  |  |  |  | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Value | inline | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| relation_changed | Answer | original | 8 |  |  |  |  | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Answer | inline | 8 |  |  |  |  | 3/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Response | original | 8 |  |  |  |  | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| relation_changed | Response | inline | 8 |  |  |  |  | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.4 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| relation_changed | Value | original | 8 |  |  |  |  | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| relation_changed | Value | inline | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| explanation_needed | Answer | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 | 75.1 | 99.000 | explanation:7, word:1 |
| explanation_needed | Answer | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 167.4 | 99.000 | explanation:8 |
| explanation_needed | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 | 89.2 | 99.000 | explanation:7, word:1 |
| explanation_needed | Response | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 148.6 | 99.000 | explanation:8 |
| explanation_needed | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 14.6 | 86.891 | space:8 |
| explanation_needed | Value | inline | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 51.344 | space:5, correct_prefix:3 |
| non_value | Answer | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 209.5 | 47.693 | explanation:8 |
| non_value | Answer | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 216.9 | 47.758 | explanation:8 |
| non_value | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 219.0 | 73.361 | explanation:8 |
| non_value | Response | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 252.1 | 47.658 | explanation:8 |
| non_value | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 13.5 | 74.605 | explanation:8 |
| non_value | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 22.0 | 24.793 | explanation:8 |

### Target Failure Best Sufficiency

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 51.719 | correct_prefix:7, word:1 |
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 51.719 | correct_prefix:7, word:1 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:6, word:2 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.375 | correct_prefix:6, space:2 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.375 | correct_prefix:6, space:2 |
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 75.387 | correct_prefix:7, explanation:1 |
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 75.387 | correct_prefix:7, explanation:1 |
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.9 | 99.000 | correct_prefix:7, explanation:1 |
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.9 | 99.000 | correct_prefix:7, explanation:1 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.2 | 39.188 | correct_prefix:5, space:3 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.2 | 39.188 | correct_prefix:5, space:3 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:6, word:2 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.336 | space:4, correct_prefix:4 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.336 | space:4, correct_prefix:4 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 3.148 | correct_prefix:5, space:2, word:1 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 3/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 3.062 | space:4, correct_prefix:4 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 1/8 | 1/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.5 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 1/8 | 1/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.5 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 8/8 | 0/8 | 2.6 | 99.000 | word:7, correct_prefix:1 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 1/8 | 0/8 | 2/8 | 0/8 | 2/8 | 0/8 | 3.0 | 99.000 | explanation:6, correct_prefix:2 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 4.9 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 10.5 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 10.5 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 1/8 | 0/8 | 8/8 | 0/8 | 2.4 | 99.000 | space:7, correct_prefix:1 |
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | word:8 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:7, explanation:1 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:7, explanation:1 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 5.2 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 | 7.5 | 99.000 | explanation:7, word:1 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 1/8 | 1/8 | 0/8 | 5/8 | 0/8 | 15.2 | 99.000 | word:4, explanation:3, correct_prefix:1 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 1/8 | 1/8 | 0/8 | 5/8 | 0/8 | 15.2 | 99.000 | word:4, explanation:3, correct_prefix:1 |

### Largest Old-Value Side Effects

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| original_correct | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| original_correct | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| original_correct | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| original_correct | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| original_correct | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| original_correct | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| original_correct | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 99.000 | correct_prefix:5, word:2, explanation:1 |
| original_correct | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 99.000 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| relation_changed | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| relation_changed | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| relation_changed | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| relation_changed | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| relation_changed | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| relation_changed | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| relation_changed | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| original_correct | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| original_correct | Response | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 99.000 | correct_prefix:4, word:3, explanation:1 |
| original_correct | Response | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 99.000 | correct_prefix:4, word:3, explanation:1 |
| relation_changed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| relation_changed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| original_correct | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.6 | 99.000 | explanation:4, correct_prefix:4 |
| relation_changed | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.6 | 99.000 | explanation:4, correct_prefix:4 |
| original_correct | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.8 | 99.000 | explanation:4, correct_prefix:4 |
| relation_changed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.8 | 99.000 | explanation:4, correct_prefix:4 |
| relation_changed | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 3/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 3/8 | 2/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 99.000 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 3/8 | 2/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 99.000 | correct_prefix:5, word:2, explanation:1 |
| original_correct | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| original_correct | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| relation_changed | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| relation_changed | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.375 | space:5, correct_prefix:3 |
| explanation_needed | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 51.344 | space:5, correct_prefix:3 |
| explanation_needed | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 51.344 | space:5, correct_prefix:3 |
| original_correct | Value | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 15.656 | space:5, correct_prefix:3 |
| original_correct | Value | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 15.656 | space:5, correct_prefix:3 |
| relation_changed | Response | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 2/8 | 2/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 99.000 | correct_prefix:4, word:3, explanation:1 |
| relation_changed | Response | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 2/8 | 2/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 99.000 | correct_prefix:4, word:3, explanation:1 |
| relation_changed | Value | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 15.656 | space:5, correct_prefix:3 |
| relation_changed | Value | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 2/8 | 1/8 | 3/8 | 0/8 | 8/8 | 0/8 | 1.6 | 15.656 | space:5, correct_prefix:3 |
| original_correct | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 1.8 | 3.117 | space:6, correct_prefix:2 |
| original_correct | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 1.8 | 3.117 | space:6, correct_prefix:2 |
| relation_changed | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 1.8 | 3.117 | space:6, correct_prefix:2 |
| relation_changed | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 1.8 | 3.117 | space:6, correct_prefix:2 |
| original_correct | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.0 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| original_correct | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 2/8 | 0/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.0 | 99.000 | word:5, correct_prefix:2, explanation:1 |

### Position Overview

#### label_aligned

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 75.387 | correct_prefix:7, explanation:1 |
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.9 | 99.000 | correct_prefix:7, explanation:1 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.2 | 39.188 | correct_prefix:5, space:3 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.336 | space:4, correct_prefix:4 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 1/8 | 1/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.5 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 10.5 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:7, explanation:1 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 1/8 | 1/8 | 0/8 | 5/8 | 0/8 | 15.2 | 99.000 | word:4, explanation:3, correct_prefix:1 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| original_correct | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| original_correct | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| relation_changed | Answer | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| relation_changed | Response | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 4/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| relation_changed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| explanation_needed | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 4.4 | 75.219 | space:8 |
| explanation_needed | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 7.1 | 25.961 | space:8 |
| explanation_needed | Value | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 7.1 | 99.000 | space:8 |
| explanation_needed | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 10.9 | 99.000 | space:8 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_layer_out_restore | 8 | label_aligned | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.4 | 62.629 | explanation:8 |
| non_value | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.5 | 99.000 | word:8 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_attn_out_restore | 8 | label_aligned | remove_from_inline | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 24.2 | 99.000 | explanation:8 |
| non_value | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 30.6 | 49.098 | explanation:8 |

#### label_colon

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.4 | 75.387 | correct_prefix:7, explanation:1 |
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.9 | 99.000 | correct_prefix:7, explanation:1 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.2 | 39.188 | correct_prefix:5, space:3 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.5 | 2.336 | space:4, correct_prefix:4 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 1/8 | 1/8 | 2/8 | 0/8 | 8/8 | 0/8 | 2.5 | 99.000 | word:5, correct_prefix:2, explanation:1 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 10.5 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:7, explanation:1 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 1/8 | 1/8 | 0/8 | 5/8 | 0/8 | 15.2 | 99.000 | word:4, explanation:3, correct_prefix:1 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| original_correct | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 6/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| original_correct | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| relation_changed | Answer | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 75.523 | correct_prefix:8 |
| relation_changed | Response | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 4/8 | 2/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.523 | correct_prefix:7, space:1 |
| relation_changed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.6 | 87.234 | correct_prefix:6, explanation:1, word:1 |
| explanation_needed | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 4.4 | 75.219 | space:8 |
| explanation_needed | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 7.1 | 25.961 | space:8 |
| explanation_needed | Value | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 7.1 | 99.000 | space:8 |
| explanation_needed | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 10.9 | 99.000 | space:8 |
| non_value | Value | label_colon_remove_from_inline_L17_20_layer_out_restore | 8 | label_colon | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.4 | 62.629 | explanation:8 |
| non_value | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.5 | 99.000 | word:8 |
| non_value | Value | label_colon_remove_from_inline_L17_20_attn_out_restore | 8 | label_colon | remove_from_inline | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 24.2 | 99.000 | explanation:8 |
| non_value | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 30.6 | 49.098 | explanation:8 |

#### separator

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 6/8 | 1/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 51.719 | correct_prefix:7, word:1 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.375 | correct_prefix:6, space:2 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 2/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:6, word:2 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 3.148 | correct_prefix:5, space:2, word:1 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 8/8 | 0/8 | 2.6 | 99.000 | word:7, correct_prefix:1 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 1/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 4.9 | 99.000 | explanation:7, correct_prefix:1 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 1/8 | 0/8 | 8/8 | 0/8 | 2.4 | 99.000 | space:7, correct_prefix:1 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:8 |
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 | 0/8 | 7.5 | 99.000 | explanation:7, word:1 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| original_correct | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Answer | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Response | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| relation_changed | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| relation_changed | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.6 | 99.000 | explanation:4, correct_prefix:4 |
| explanation_needed | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 51.344 | space:5, correct_prefix:3 |
| explanation_needed | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.8 | 99.000 | space:8 |
| explanation_needed | Value | separator_remove_from_inline_L17_20_mlp_out_restore | 8 | separator | remove_from_inline | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 14.5 | 99.000 | space:8 |
| explanation_needed | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 14.6 | 86.891 | space:8 |
| non_value | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.1 | 99.000 | space:4, word:4 |
| non_value | Value | separator_remove_from_inline_L17_20_layer_out_restore | 8 | separator | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 13.5 | 74.605 | explanation:8 |
| non_value | Value | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 17.2 | 86.680 | explanation:6, space:1, word:1 |
| non_value | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 22.0 | 24.793 | explanation:8 |

#### relation_tail

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 51.719 | correct_prefix:7, word:1 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 99.000 | correct_prefix:6, word:2 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.1 | 1.375 | correct_prefix:6, space:2 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 3/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.6 | 3.062 | space:4, correct_prefix:4 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 1/8 | 0/8 | 2/8 | 0/8 | 2/8 | 0/8 | 3.0 | 99.000 | explanation:6, correct_prefix:2 |
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | word:8 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 2.8 | 99.000 | space:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 3.5 | 99.000 | word:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 1/8 | 0/8 | 1/8 | 0/8 | 5.2 | 99.000 | explanation:7, correct_prefix:1 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| original_correct | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| original_correct | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| original_correct | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.5 | 87.227 | correct_prefix:5, word:2, explanation:1 |
| relation_changed | Answer | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:8 |
| relation_changed | Response | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 3/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.0 | 99.000 | correct_prefix:7, word:1 |
| relation_changed | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 4/8 | 1/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 4.266 | correct_prefix:5, space:3 |
| relation_changed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 4/8 | 0/8 | 3.8 | 99.000 | explanation:4, correct_prefix:4 |
| explanation_needed | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 51.344 | space:5, correct_prefix:3 |
| explanation_needed | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 10.8 | 74.789 | space:8 |
| explanation_needed | Value | relation_tail_remove_from_inline_L17_20_mlp_out_restore | 8 | relation_tail | remove_from_inline | mlp_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 13.2 | 99.000 | space:8 |
| explanation_needed | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 14.6 | 86.891 | space:8 |
| non_value | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 9.9 | 99.000 | space:4, word:4 |
| non_value | Value | relation_tail_remove_from_inline_L17_20_layer_out_restore | 8 | relation_tail | remove_from_inline | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 13.5 | 74.605 | explanation:8 |
| non_value | Value | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 18.9 | 74.367 | explanation:7, space:1 |
| non_value | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 22.0 | 24.793 | explanation:8 |

## deepseek7b

- raw_cases: 320 / selected_items: 40 / mode_rows: 8880
- max_per_split: 8 / templates: `['Answer', 'Response', 'Value']`
- positions: `['label_aligned', 'label_colon', 'separator', 'relation_tail']` / interval_specs: `[{'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'layer_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'attn_out'}, {'interval': 'L17_20', 'layers': [17, 18, 19, 20], 'component': 'mlp_out'}]`
- selection_stats: `{'target_failure_seen': 8, 'original_correct_seen': 17, 'counts': {'target_failure': 8, 'original_correct': 8, 'relation_changed': 8, 'explanation_needed': 8, 'non_value': 8}}`
- filtered: `{'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0}` / total_time_min: 21.02

### Baselines

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 7/8 | 1/8 | 0/8 | 13.6 | -2.477 | newline:7, word:1 |
| target_failure | Answer | inline | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.312 | correct_prefix:8 |
| target_failure | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 111.8 | -4.289 | word:8 |
| target_failure | Response | inline | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.062 | correct_prefix:8 |
| target_failure | Value | original | 8 |  |  |  |  | 2/8 | 0/8 | 2/8 | 6/8 | 2/8 | 0/8 | 3.0 | -1.828 | newline:6, correct_prefix:2 |
| target_failure | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.5 | -3.094 | newline:8 |
| original_correct | Answer | original | 8 |  |  |  |  | 2/8 | 0/8 | 2/8 | 6/8 | 2/8 | 0/8 | 8.8 | -1.664 | newline:6, correct_prefix:2 |
| original_correct | Answer | inline | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.281 | correct_prefix:8 |
| original_correct | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 80.4 | -3.016 | word:8 |
| original_correct | Response | inline | 8 |  |  |  |  | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.320 | correct_prefix:8 |
| original_correct | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 3.2 | -2.242 | newline:8 |
| original_correct | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 3.6 | -2.094 | newline:8 |
| relation_changed | Answer | original | 8 |  |  |  |  | 1/8 | 0/8 | 1/8 | 7/8 | 1/8 | 0/8 | 9.0 | -2.109 | newline:7, correct_prefix:1 |
| relation_changed | Answer | inline | 8 |  |  |  |  | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| relation_changed | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 75.2 | -3.164 | word:8 |
| relation_changed | Response | inline | 8 |  |  |  |  | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.648 | correct_prefix:8 |
| relation_changed | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 3.8 | -2.711 | newline:8 |
| relation_changed | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.1 | -2.625 | newline:8 |
| explanation_needed | Answer | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 5/8 | 0/8 | 62.8 | -4.133 | word:4, explanation:4 |
| explanation_needed | Answer | inline | 8 |  |  |  |  | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.4 | 1.031 | correct_prefix:6, newline:2 |
| explanation_needed | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 38.6 | -4.008 | word:8 |
| explanation_needed | Response | inline | 8 |  |  |  |  | 1/8 | 0/8 | 1/8 | 0/8 | 3/8 | 0/8 | 3.1 | 0.789 | explanation:5, word:2, correct_prefix:1 |
| explanation_needed | Value | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 25.1 | -7.609 | newline:8 |
| explanation_needed | Value | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.1 | -2.922 | newline:8 |
| non_value | Answer | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 4/8 | 0/8 | 0/8 | 164.9 | -5.789 | newline:4, explanation:4 |
| non_value | Answer | inline | 8 |  |  |  |  | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.6 | 0.656 | correct_prefix:5, explanation:2, newline:1 |
| non_value | Response | original | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 198.8 | -5.688 | explanation:8 |
| non_value | Response | inline | 8 |  |  |  |  | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3.1 | 0.477 | explanation:8 |
| non_value | Value | original | 8 |  |  |  |  | 3/8 | 0/8 | 3/8 | 5/8 | 3/8 | 0/8 | 2.9 | -1.000 | newline:5, correct_prefix:3 |
| non_value | Value | inline | 8 |  |  |  |  | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.672 | correct_prefix:5, space:3 |

### Target Failure Best Sufficiency

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.523 | correct_prefix:8 |
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.523 | correct_prefix:8 |
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.312 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.484 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.484 | correct_prefix:8 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.062 | correct_prefix:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.727 | correct_prefix:8 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.727 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 27.859 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 27.859 | correct_prefix:8 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.062 | correct_prefix:8 |
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.516 | correct_prefix:6, word:2 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 6/8 | 1/8 | 7/8 | 1/8 | 7/8 | 0/8 | 1.1 | 1.047 | correct_prefix:7, newline:1 |
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.312 | correct_prefix:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 2/8 | 6/8 | 0/8 | 1.9 | 0.445 | correct_prefix:5, newline:2, word:1 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.062 | correct_prefix:8 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.2 | 0.703 | correct_prefix:4, space:2, word:2 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.2 | 0.703 | correct_prefix:4, space:2, word:2 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.4 | 0.812 | correct_prefix:4, space:2, word:2 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 2/8 | 2/8 | 0/8 | 8/8 | 0/8 | 3.5 | 0.836 | word:4, space:2, correct_prefix:2 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 8.2 | -2.531 | word:5, newline:3 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 8.2 | -2.531 | word:5, newline:3 |
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 9.6 | -2.438 | word:5, newline:3 |
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 11.4 | -2.578 | word:5, newline:3 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 2.8 | -1.422 | newline:6, word:2 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 2.8 | -1.422 | newline:6, word:2 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 3.4 | -1.562 | newline:6, word:2 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 3.5 | -1.500 | newline:6, word:2 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.5 | -3.094 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.5 | -3.094 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 5.0 | -3.734 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 5.0 | -3.734 | newline:8 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 18.0 | -7.453 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 18.2 | -7.711 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 24.5 | -8.750 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 24.5 | -8.750 | newline:8 |

### Largest Old-Value Side Effects

| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.484 | correct_prefix:8 |
| original_correct | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.484 | correct_prefix:8 |
| original_correct | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.281 | correct_prefix:8 |
| original_correct | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.281 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.805 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 16.805 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.805 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 16.805 | correct_prefix:8 |
| original_correct | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.320 | correct_prefix:8 |
| original_correct | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| original_correct | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.320 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.875 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.875 | correct_prefix:8 |
| explanation_needed | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.180 | correct_prefix:8 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.867 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.383 | correct_prefix:8 |
| non_value | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| non_value | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 7/8 | 0/8 | 1.1 | 1.438 | correct_prefix:7, explanation:1 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.484 | correct_prefix:6, space:2 |
| explanation_needed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.484 | correct_prefix:6, space:2 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.2 | 1.000 | correct_prefix:6, newline:2 |
| non_value | Value | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.2 | 1.000 | correct_prefix:6, newline:2 |
| explanation_needed | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.4 | 1.031 | correct_prefix:6, newline:2 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.4 | 1.031 | correct_prefix:6, newline:2 |
| original_correct | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.648 | correct_prefix:6, word:2 |
| original_correct | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.992 | correct_prefix:5, word:3 |
| original_correct | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.992 | correct_prefix:5, word:3 |
| non_value | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.125 | correct_prefix:5, space:3 |
| non_value | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.125 | correct_prefix:5, space:3 |
| non_value | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.672 | correct_prefix:5, space:3 |
| original_correct | Response | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.031 | correct_prefix:5, space:3 |
| original_correct | Response | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.6 | 2.031 | correct_prefix:5, space:3 |
| non_value | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 1.4 | 2.273 | correct_prefix:5, explanation:3 |
| non_value | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 1.4 | 2.273 | correct_prefix:5, explanation:3 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_attn_out_restore | 8 | label_aligned | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.625 | correct_prefix:5, word:2, newline:1 |
| non_value | Value | label_colon_remove_from_inline_L17_20_attn_out_restore | 8 | label_colon | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.625 | correct_prefix:5, word:2, newline:1 |
| non_value | Value | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.836 | correct_prefix:5, word:2, newline:1 |
| non_value | Value | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 4/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.508 | correct_prefix:4, word:3, newline:1 |
| original_correct | Answer | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 3/8 | 5/8 | 0/8 | 1.6 | 0.344 | correct_prefix:5, newline:3 |
| original_correct | Answer | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 5/8 | 0/8 | 5/8 | 3/8 | 5/8 | 0/8 | 1.6 | 0.344 | correct_prefix:5, newline:3 |
| non_value | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.6 | 0.656 | correct_prefix:5, explanation:2, newline:1 |
| non_value | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 2.4 | 0.359 | correct_prefix:5, explanation:3 |
| non_value | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 2.4 | 0.359 | correct_prefix:5, explanation:3 |
| relation_changed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.172 | correct_prefix:8 |
| relation_changed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.172 | correct_prefix:8 |
| relation_changed | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| relation_changed | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.094 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.523 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.094 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.523 | correct_prefix:8 |
| relation_changed | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.648 | correct_prefix:8 |
| relation_changed | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.742 | correct_prefix:8 |
| relation_changed | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.648 | correct_prefix:8 |
| non_value | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.8 | 2.789 | correct_prefix:4, space:4 |
| non_value | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 4/8 | 0/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.8 | 2.789 | correct_prefix:4, space:4 |
| original_correct | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 1/8 | 7/8 | 0/8 | 2.4 | 0.508 | correct_prefix:4, word:3, newline:1 |

### Position Overview

#### label_aligned

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.523 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.484 | correct_prefix:8 |
| target_failure | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.727 | correct_prefix:8 |
| target_failure | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 27.859 | correct_prefix:8 |
| target_failure | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.2 | 0.703 | correct_prefix:4, space:2, word:2 |
| target_failure | Answer | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 8.2 | -2.531 | word:5, newline:3 |
| target_failure | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 2.8 | -1.422 | newline:6, word:2 |
| target_failure | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 5.0 | -3.734 | newline:8 |
| target_failure | Value | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 24.5 | -8.750 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.484 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.805 | correct_prefix:8 |
| original_correct | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 16.805 | correct_prefix:8 |
| original_correct | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.992 | correct_prefix:5, word:3 |
| relation_changed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.172 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.094 | correct_prefix:8 |
| relation_changed | Response | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.523 | correct_prefix:8 |
| relation_changed | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 3/8 | 3/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.859 | correct_prefix:6, word:2 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.875 | correct_prefix:8 |
| explanation_needed | Response | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.484 | correct_prefix:6, space:2 |
| explanation_needed | Value | label_aligned_to_original_L17_20_attn_out_restore | 8 | label_aligned | to_original | attn_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.2 | 1.984 | correct_prefix:3, word:3, space:2 |
| non_value | Value | label_aligned_remove_from_inline_L17_20_mlp_out_restore | 8 | label_aligned | remove_from_inline | mlp_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.2 | 1.000 | correct_prefix:6, newline:2 |
| non_value | Answer | label_aligned_to_original_L17_20_mlp_out_restore | 8 | label_aligned | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 1.4 | 2.273 | correct_prefix:5, explanation:3 |
| non_value | Value | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.125 | correct_prefix:5, space:3 |
| non_value | Answer | label_aligned_to_original_L17_20_layer_out_restore | 8 | label_aligned | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 2.4 | 0.359 | correct_prefix:5, explanation:3 |

#### label_colon

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.523 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.484 | correct_prefix:8 |
| target_failure | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.727 | correct_prefix:8 |
| target_failure | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 27.859 | correct_prefix:8 |
| target_failure | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.2 | 0.703 | correct_prefix:4, space:2, word:2 |
| target_failure | Answer | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 8.2 | -2.531 | word:5, newline:3 |
| target_failure | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 2.8 | -1.422 | newline:6, word:2 |
| target_failure | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 5.0 | -3.734 | newline:8 |
| target_failure | Value | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 24.5 | -8.750 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.484 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.805 | correct_prefix:8 |
| original_correct | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 16.805 | correct_prefix:8 |
| original_correct | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.992 | correct_prefix:5, word:3 |
| relation_changed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.172 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.094 | correct_prefix:8 |
| relation_changed | Response | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.523 | correct_prefix:8 |
| relation_changed | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 3/8 | 3/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 1.859 | correct_prefix:6, word:2 |
| explanation_needed | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.875 | correct_prefix:8 |
| explanation_needed | Response | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.484 | correct_prefix:6, space:2 |
| explanation_needed | Value | label_colon_to_original_L17_20_attn_out_restore | 8 | label_colon | to_original | attn_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.2 | 1.984 | correct_prefix:3, word:3, space:2 |
| non_value | Value | label_colon_remove_from_inline_L17_20_mlp_out_restore | 8 | label_colon | remove_from_inline | mlp_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.2 | 1.000 | correct_prefix:6, newline:2 |
| non_value | Answer | label_colon_to_original_L17_20_mlp_out_restore | 8 | label_colon | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 1.4 | 2.273 | correct_prefix:5, explanation:3 |
| non_value | Value | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.125 | correct_prefix:5, space:3 |
| non_value | Answer | label_colon_to_original_L17_20_layer_out_restore | 8 | label_colon | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 5/8 | 0/8 | 2.4 | 0.359 | correct_prefix:5, explanation:3 |

#### separator

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.312 | correct_prefix:8 |
| target_failure | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.062 | correct_prefix:8 |
| target_failure | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 7/8 | 1/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.062 | correct_prefix:8 |
| target_failure | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 6/8 | 1/8 | 7/8 | 1/8 | 7/8 | 0/8 | 1.1 | 1.047 | correct_prefix:7, newline:1 |
| target_failure | Value | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 3.4 | 0.812 | correct_prefix:4, space:2, word:2 |
| target_failure | Answer | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 11.4 | -2.578 | word:5, newline:3 |
| target_failure | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 3.4 | -1.562 | newline:6, word:2 |
| target_failure | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.5 | -3.094 | newline:8 |
| target_failure | Value | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 18.0 | -7.453 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.281 | correct_prefix:8 |
| original_correct | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.320 | correct_prefix:8 |
| original_correct | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 4.281 | correct_prefix:8 |
| original_correct | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 2/8 | 6/8 | 0/8 | 1.8 | 1.094 | correct_prefix:4, newline:2, word:2 |
| relation_changed | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| relation_changed | Response | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.648 | correct_prefix:8 |
| relation_changed | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.742 | correct_prefix:8 |
| relation_changed | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 3/8 | 2/8 | 5/8 | 2/8 | 6/8 | 0/8 | 1.6 | 1.086 | correct_prefix:5, newline:2, word:1 |
| explanation_needed | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.180 | correct_prefix:8 |
| explanation_needed | Response | separator_to_original_L17_20_attn_out_restore | 8 | separator | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.438 | correct_prefix:8 |
| explanation_needed | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.4 | 1.031 | correct_prefix:6, newline:2 |
| explanation_needed | Response | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 3/8 | 0/8 | 3/8 | 0/8 | 8/8 | 0/8 | 2.6 | 3.125 | word:5, correct_prefix:3 |
| non_value | Answer | separator_to_original_L17_20_mlp_out_restore | 8 | separator | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| non_value | Value | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.672 | correct_prefix:5, space:3 |
| non_value | Value | separator_remove_from_inline_L17_20_attn_out_restore | 8 | separator | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.836 | correct_prefix:5, word:2, newline:1 |
| non_value | Answer | separator_to_original_L17_20_layer_out_restore | 8 | separator | to_original | layer_out | restore | 5/8 | 0/8 | 5/8 | 1/8 | 5/8 | 0/8 | 1.6 | 0.656 | correct_prefix:5, explanation:2, newline:1 |

#### relation_tail

Target repair candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| target_failure | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 6/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.4 | 3.516 | correct_prefix:6, word:2 |
| target_failure | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 5/8 | 3/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.312 | correct_prefix:8 |
| target_failure | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 5/8 | 0/8 | 5/8 | 2/8 | 6/8 | 0/8 | 1.9 | 0.445 | correct_prefix:5, newline:2, word:1 |
| target_failure | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.062 | correct_prefix:8 |
| target_failure | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 2/8 | 2/8 | 0/8 | 8/8 | 0/8 | 3.5 | 0.836 | word:4, space:2, correct_prefix:2 |
| target_failure | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 3/8 | 5/8 | 0/8 | 9.6 | -2.438 | word:5, newline:3 |
| target_failure | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 0/8 | 0/8 | 0/8 | 6/8 | 2/8 | 0/8 | 3.5 | -1.500 | newline:6, word:2 |
| target_failure | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 4.5 | -3.094 | newline:8 |
| target_failure | Value | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 18.2 | -7.711 | newline:8 |

Side-effect candidates:
| split | template | mode | n | position | direction | component | control | exact | wrong_exact | tok0 | newline | gen_short | gen_expl | rank | prefix-newline | top0_category |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_correct | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.281 | correct_prefix:8 |
| original_correct | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 3.320 | correct_prefix:8 |
| original_correct | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 5/8 | 0/8 | 6/8 | 0/8 | 8/8 | 0/8 | 1.2 | 3.648 | correct_prefix:6, word:2 |
| original_correct | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 4/8 | 0/8 | 4/8 | 1/8 | 7/8 | 0/8 | 2.4 | 0.508 | correct_prefix:4, word:3, newline:1 |
| relation_changed | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.992 | correct_prefix:8 |
| relation_changed | Response | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 4/8 | 4/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 2.648 | correct_prefix:8 |
| relation_changed | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 3/8 | 2/8 | 5/8 | 1/8 | 7/8 | 0/8 | 2.1 | 0.422 | correct_prefix:5, word:2, newline:1 |
| relation_changed | Response | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 2/8 | 4/8 | 7/8 | 0/8 | 8/8 | 0/8 | 1.1 | 3.055 | correct_prefix:7, word:1 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.867 | correct_prefix:8 |
| explanation_needed | Response | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 8/8 | 0/8 | 8/8 | 0/8 | 8/8 | 0/8 | 1.0 | 1.383 | correct_prefix:8 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 6/8 | 0/8 | 6/8 | 2/8 | 6/8 | 0/8 | 1.4 | 1.031 | correct_prefix:6, newline:2 |
| explanation_needed | Answer | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 2/8 | 0/8 | 2/8 | 4/8 | 2/8 | 2/8 | 15.6 | -1.367 | newline:4, word:2, correct_prefix:2 |
| non_value | Answer | relation_tail_to_original_L17_20_mlp_out_restore | 8 | relation_tail | to_original | mlp_out | restore | 7/8 | 0/8 | 7/8 | 0/8 | 7/8 | 0/8 | 1.1 | 1.438 | correct_prefix:7, explanation:1 |
| non_value | Value | relation_tail_remove_from_inline_L17_20_attn_out_restore | 8 | relation_tail | remove_from_inline | attn_out | restore | 5/8 | 0/8 | 4/8 | 1/8 | 5/8 | 0/8 | 1.5 | 3.508 | correct_prefix:4, word:3, newline:1 |
| non_value | Value | relation_tail_to_original_L17_20_layer_out_restore | 8 | relation_tail | to_original | layer_out | restore | 3/8 | 2/8 | 5/8 | 0/8 | 8/8 | 0/8 | 1.4 | 1.672 | correct_prefix:5, space:3 |
| non_value | Value | relation_tail_to_original_L17_20_attn_out_restore | 8 | relation_tail | to_original | attn_out | restore | 3/8 | 1/8 | 4/8 | 0/8 | 8/8 | 0/8 | 1.9 | 2.891 | correct_prefix:4, space:4 |
