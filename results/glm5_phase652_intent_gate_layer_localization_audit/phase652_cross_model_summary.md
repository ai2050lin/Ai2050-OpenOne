# Phase 652 Cross-Model Summary

目标：把 Phase651 的 L14-L22 区间结果收缩到单层、单位置、单组件。主指标是 correct value prefix rank 的改善或压制。

## qwen3

- raw_cases: 320 / selected_items: 20 / mode_rows: 10160 / time: 6.65 min
- layers: `[14, 15, 16, 17, 18, 19, 20, 21, 22]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- tasks: `['explanation_required', 'yes_no_required']` / positions: `['intent_word', 'instruction_span', 'label_aligned', 'separator', 'relation_tail']`
- filtered: `{'position_missing': 40, 'position_len_mismatch': 20, 'empty_patch': 0}` / selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}`

### Baselines

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | explanation_required |  |  |  |  |  | 20 | 1.8->1.80 | 0.0 | 10->10 | 0/20 | explanation:10, correct_prefix:10 |
| explanation_required | short_value_allowed |  |  |  |  |  | 20 | 9.2->9.20 | 0.0 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed |  |  |  |  |  | 20 | 9.2->9.20 | 0.0 | 0->0 | 0/20 | space:20 |
| yes_no_required | yes_no_required |  |  |  |  |  | 20 | 12.95->12.95 | 0.0 | 0->0 | 0/20 | explanation:20 |

### Strongest Absorption: value_to_task

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_out | restore | 20 | 12.95->1.65 | 11.299999999999999 | 0->10 | 0/20 | space:10, correct_prefix:10 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_input | restore | 20 | 12.95->1.65 | 11.299999999999999 | 0->10 | 0/20 | space:10, correct_prefix:10 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_out | restore | 20 | 12.95->1.85 | 11.1 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_input | restore | 20 | 12.95->1.85 | 11.1 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_input | restore | 20 | 12.95->1.85 | 11.1 | 0->11 | 0/20 | correct_prefix:11, space:8, explanation:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_input | restore | 20 | 12.95->1.85 | 11.1 | 0->11 | 0/20 | correct_prefix:11, space:7, explanation:2 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_out | restore | 20 | 12.95->1.95 | 11.0 | 0->11 | 0/20 | correct_prefix:11, space:8, explanation:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_input | restore | 20 | 12.95->1.95 | 11.0 | 0->11 | 0/20 | correct_prefix:11, space:8, explanation:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_out | restore | 20 | 12.95->2.00 | 10.95 | 0->11 | 1/20 | correct_prefix:11, space:7, explanation:1, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_input | restore | 20 | 12.95->2.00 | 10.95 | 0->11 | 1/20 | correct_prefix:11, space:7, explanation:1, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_out | restore | 20 | 12.95->2.10 | 10.85 | 0->10 | 0/20 | space:10, correct_prefix:10 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_input | restore | 20 | 12.95->2.10 | 10.85 | 0->10 | 0/20 | space:10, correct_prefix:10 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_out | restore | 20 | 12.95->2.30 | 10.649999999999999 | 0->9 | 0/20 | space:10, correct_prefix:9, explanation:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_input | restore | 20 | 12.95->2.30 | 10.649999999999999 | 0->9 | 0/20 | space:10, correct_prefix:9, explanation:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_out | restore | 20 | 12.95->2.85 | 10.1 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_input | restore | 20 | 12.95->2.85 | 10.1 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_out | restore | 20 | 12.95->2.90 | 10.049999999999999 | 0->6 | 0/20 | space:14, correct_prefix:6 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_input | restore | 20 | 12.95->2.90 | 10.049999999999999 | 0->6 | 0/20 | space:14, correct_prefix:6 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_out | restore | 20 | 12.95->3.85 | 9.1 | 0->6 | 1/20 | space:13, correct_prefix:6, newline:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_input | restore | 20 | 12.95->3.85 | 9.1 | 0->6 | 1/20 | space:13, correct_prefix:6, newline:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | mlp_out | restore | 20 | 12.95->3.85 | 9.1 | 0->5 | 0/20 | explanation:10, space:5, correct_prefix:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | layer_out | restore | 20 | 12.95->3.95 | 9.0 | 0->8 | 0/20 | correct_prefix:8, explanation:7, space:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_input | restore | 20 | 12.95->3.95 | 9.0 | 0->8 | 0/20 | correct_prefix:8, explanation:7, space:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | mlp_out | restore | 20 | 12.95->4.00 | 8.95 | 0->6 | 0/20 | explanation:9, correct_prefix:6, space:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | attn_out | restore | 20 | 12.95->4.05 | 8.899999999999999 | 0->7 | 0/20 | explanation:8, correct_prefix:7, space:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | attn_out | restore | 20 | 12.95->4.15 | 8.799999999999999 | 0->8 | 0/20 | correct_prefix:8, explanation:7, space:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_out | restore | 20 | 12.95->4.15 | 8.799999999999999 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_input | restore | 20 | 12.95->4.15 | 8.799999999999999 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | attn_out | restore | 20 | 12.95->4.20 | 8.75 | 0->7 | 0/20 | explanation:8, correct_prefix:7, space:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_out | restore | 20 | 12.95->4.35 | 8.6 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_input | restore | 20 | 12.95->4.35 | 8.6 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | mlp_out | restore | 20 | 12.95->4.35 | 8.6 | 0->5 | 0/20 | explanation:10, space:5, correct_prefix:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_out | restore | 20 | 12.95->4.60 | 8.35 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_input | restore | 20 | 12.95->4.60 | 8.35 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | attn_out | restore | 20 | 12.95->4.90 | 8.049999999999999 | 0->5 | 0/20 | explanation:15, correct_prefix:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_out | restore | 20 | 12.95->5.40 | 7.549999999999999 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_input | restore | 20 | 12.95->5.40 | 7.549999999999999 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | attn_out | restore | 20 | 12.95->5.50 | 7.449999999999999 | 0->5 | 0/20 | explanation:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_out | restore | 20 | 12.95->5.55 | 7.3999999999999995 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_input | restore | 20 | 12.95->5.55 | 7.3999999999999995 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 14 | layer_input | restore | 20 | 12.95->6.00 | 6.949999999999999 | 0->6 | 0/20 | explanation:11, correct_prefix:6, space:3 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | mlp_out | restore | 20 | 12.95->6.05 | 6.8999999999999995 | 0->3 | 0/20 | explanation:13, space:4, correct_prefix:3 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_out | restore | 20 | 12.95->6.30 | 6.6499999999999995 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_input | restore | 20 | 12.95->6.30 | 6.6499999999999995 | 0->5 | 0/20 | space:15, correct_prefix:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_out | restore | 20 | 12.95->6.30 | 6.6499999999999995 | 0->2 | 0/20 | space:18, correct_prefix:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_input | restore | 20 | 12.95->6.30 | 6.6499999999999995 | 0->2 | 0/20 | space:18, correct_prefix:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 14 | layer_out | restore | 20 | 12.95->6.60 | 6.35 | 0->6 | 0/20 | explanation:12, correct_prefix:6, space:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | layer_input | restore | 20 | 12.95->6.60 | 6.35 | 0->6 | 0/20 | explanation:12, correct_prefix:6, space:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | mlp_out | restore | 20 | 12.95->6.65 | 6.299999999999999 | 0->4 | 0/20 | explanation:12, correct_prefix:4, space:4 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_out | restore | 20 | 12.95->6.75 | 6.199999999999999 | 0->4 | 0/20 | space:16, correct_prefix:4 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_input | restore | 20 | 12.95->6.75 | 6.199999999999999 | 0->4 | 0/20 | space:16, correct_prefix:4 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | attn_out | restore | 20 | 12.95->7.00 | 5.949999999999999 | 0->4 | 0/20 | explanation:16, correct_prefix:4 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | mlp_out | restore | 20 | 12.95->7.30 | 5.6499999999999995 | 0->2 | 0/20 | explanation:13, space:5, correct_prefix:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_out | restore | 20 | 12.95->7.30 | 5.6499999999999995 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_input | restore | 20 | 12.95->7.30 | 5.6499999999999995 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_out | restore | 20 | 12.95->7.60 | 5.35 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | mlp_out | restore | 20 | 12.95->8.65 | 4.299999999999999 | 0->4 | 0/20 | explanation:16, correct_prefix:4 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | mlp_out | restore | 20 | 12.95->8.65 | 4.299999999999999 | 0->2 | 0/20 | explanation:18, correct_prefix:2 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_out | restore | 20 | 12.95->9.10 | 3.8499999999999996 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_input | restore | 20 | 12.95->9.10 | 3.8499999999999996 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_out | restore | 20 | 12.95->9.20 | 3.75 | 0->0 | 0/20 | space:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | attn_out | restore | 20 | 12.95->9.35 | 3.5999999999999996 | 0->1 | 0/20 | explanation:19, correct_prefix:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_out | restore | 20 | 12.95->9.40 | 3.549999999999999 | 0->0 | 0/20 | space:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_input | restore | 20 | 12.95->9.40 | 3.549999999999999 | 0->0 | 0/20 | space:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_out | restore | 20 | 12.95->9.60 | 3.3499999999999996 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_input | restore | 20 | 12.95->9.60 | 3.3499999999999996 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_out | restore | 20 | 12.95->9.70 | 3.25 | 0->0 | 0/20 | space:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | mlp_out | restore | 20 | 12.95->9.80 | 3.1499999999999986 | 0->2 | 0/20 | explanation:18, correct_prefix:2 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_out | restore | 20 | 12.95->9.90 | 3.049999999999999 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_input | restore | 20 | 12.95->9.90 | 3.049999999999999 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | mlp_out | restore | 20 | 12.95->9.90 | 3.049999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 14 | mlp_out | restore | 20 | 12.95->10.05 | 2.8999999999999986 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | attn_out | restore | 20 | 12.95->10.30 | 2.6499999999999986 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | attn_out | restore | 20 | 12.95->10.55 | 2.3999999999999986 | 0->1 | 0/20 | explanation:19, correct_prefix:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | attn_out | restore | 20 | 12.95->10.80 | 2.1499999999999986 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | mlp_out | restore | 20 | 12.95->10.90 | 2.049999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | attn_out | restore | 20 | 12.95->11.90 | 1.049999999999999 | 0->1 | 0/20 | explanation:19, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | attn_out | restore | 20 | 12.95->12.05 | 0.8999999999999986 | 0->0 | 0/20 | explanation:20 |
| explanation_required | explanation_required | label_aligned | value_to_task | 16 | mlp_out | restore | 20 | 1.8->1.10 | 0.7 | 10->18 | 0/20 | correct_prefix:18, space:2 |
| explanation_required | explanation_required | label_aligned | value_to_task | 16 | attn_out | restore | 20 | 1.8->1.15 | 0.6500000000000001 | 10->17 | 0/20 | correct_prefix:17, space:2, word:1 |

### Strongest Suppression: task_to_value

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 18 | mlp_out | restore | 20 | 9.2->16.15 | -6.949999999999999 | 0->0 | 3/20 | space:17, newline:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | attn_out | restore | 20 | 9.2->14.20 | -5.0 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | attn_out | restore | 20 | 9.2->14.15 | -4.950000000000001 | 0->0 | 1/20 | space:19, newline:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 18 | mlp_out | restore | 20 | 9.2->13.90 | -4.700000000000001 | 0->1 | 2/20 | space:17, newline:2, correct_prefix:1 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 18 | mlp_out | restore | 20 | 9.2->13.80 | -4.600000000000001 | 0->0 | 3/20 | space:17, newline:3 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | attn_out | restore | 20 | 9.2->13.55 | -4.350000000000001 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 18 | mlp_out | restore | 20 | 9.2->12.40 | -3.200000000000001 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 21 | attn_out | restore | 20 | 9.2->12.30 | -3.1000000000000014 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | mlp_out | restore | 20 | 9.2->12.25 | -3.0500000000000007 | 0->0 | 4/20 | space:16, newline:4 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 20 | mlp_out | restore | 20 | 9.2->12.25 | -3.0500000000000007 | 0->0 | 2/20 | space:17, newline:2, explanation:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 21 | attn_out | restore | 20 | 9.2->12.25 | -3.0500000000000007 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | mlp_out | restore | 20 | 9.2->12.15 | -2.950000000000001 | 0->0 | 3/20 | space:17, newline:3 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 21 | attn_out | restore | 20 | 9.2->12.10 | -2.9000000000000004 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 18 | mlp_out | restore | 20 | 9.2->11.95 | -2.75 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | mlp_out | restore | 20 | 9.2->11.80 | -2.6000000000000014 | 0->0 | 4/20 | space:16, newline:4 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | mlp_out | restore | 20 | 9.2->11.75 | -2.5500000000000007 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 18 | mlp_out | restore | 20 | 9.2->11.50 | -2.3000000000000007 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | attn_out | restore | 20 | 9.2->11.25 | -2.0500000000000007 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | attn_out | restore | 20 | 9.2->11.25 | -2.0500000000000007 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | attn_out | restore | 20 | 9.2->11.15 | -1.950000000000001 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 22 | mlp_out | restore | 20 | 9.2->11.00 | -1.8000000000000007 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 19 | mlp_out | restore | 20 | 9.2->11.00 | -1.8000000000000007 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | mlp_out | restore | 20 | 9.2->11.00 | -1.8000000000000007 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 20 | mlp_out | restore | 20 | 9.2->10.90 | -1.700000000000001 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | mlp_out | restore | 20 | 9.2->10.75 | -1.5500000000000007 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 20 | mlp_out | restore | 20 | 9.2->10.70 | -1.5 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | mlp_out | restore | 20 | 9.2->10.50 | -1.3000000000000007 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | attn_out | restore | 20 | 9.2->10.45 | -1.25 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 14 | attn_out | restore | 20 | 9.2->10.10 | -0.9000000000000004 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_out | restore | 20 | 9.2->10.05 | -0.8500000000000014 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_input | restore | 20 | 9.2->10.05 | -0.8500000000000014 | 0->1 | 0/20 | space:19, correct_prefix:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 22 | attn_out | restore | 20 | 9.2->10.00 | -0.8000000000000007 | 0->0 | 1/20 | space:19, newline:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 16 | mlp_out | restore | 20 | 9.2->10.00 | -0.8000000000000007 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | attn_out | restore | 20 | 9.2->9.95 | -0.75 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | mlp_out | restore | 20 | 9.2->9.95 | -0.75 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 20 | layer_out | restore | 20 | 9.2->9.95 | -0.75 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 21 | layer_input | restore | 20 | 9.2->9.95 | -0.75 | 0->1 | 1/20 | space:18, correct_prefix:1, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 21 | mlp_out | restore | 20 | 9.2->9.90 | -0.7000000000000011 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 22 | attn_out | restore | 20 | 9.2->9.90 | -0.7000000000000011 | 0->0 | 1/20 | space:19, newline:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 16 | mlp_out | restore | 20 | 9.2->9.80 | -0.6000000000000014 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 22 | attn_out | restore | 20 | 9.2->9.75 | -0.5500000000000007 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | attn_out | restore | 20 | 9.2->9.70 | -0.5 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 21 | layer_out | restore | 20 | 9.2->9.65 | -0.45000000000000107 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 22 | layer_input | restore | 20 | 9.2->9.65 | -0.45000000000000107 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 18 | mlp_out | restore | 20 | 9.2->9.55 | -0.3500000000000014 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 14 | attn_out | restore | 20 | 9.2->9.55 | -0.3500000000000014 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 14 | mlp_out | restore | 20 | 9.2->9.55 | -0.3500000000000014 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 14 | mlp_out | restore | 20 | 9.2->9.55 | -0.3500000000000014 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 16 | mlp_out | restore | 20 | 9.2->9.55 | -0.3500000000000014 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | mlp_out | restore | 20 | 9.2->9.50 | -0.3000000000000007 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 14 | mlp_out | restore | 20 | 9.2->9.50 | -0.3000000000000007 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 14 | attn_out | restore | 20 | 9.2->9.45 | -0.25 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 15 | attn_out | restore | 20 | 9.2->9.40 | -0.20000000000000107 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 17 | attn_out | restore | 20 | 9.2->9.40 | -0.20000000000000107 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 16 | mlp_out | restore | 20 | 9.2->9.35 | -0.15000000000000036 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 21 | attn_out | restore | 20 | 9.2->9.35 | -0.15000000000000036 | 0->0 | 1/20 | space:19, newline:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_out | restore | 20 | 9.2->9.35 | -0.15000000000000036 | 0->2 | 3/20 | space:12, newline:3, explanation:3, correct_prefix:2 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_input | restore | 20 | 9.2->9.35 | -0.15000000000000036 | 0->2 | 3/20 | space:12, newline:3, explanation:3, correct_prefix:2 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 15 | mlp_out | restore | 20 | 9.2->9.30 | -0.10000000000000142 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | mlp_out | restore | 20 | 9.2->9.30 | -0.10000000000000142 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | mlp_out | restore | 20 | 9.2->9.30 | -0.10000000000000142 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 14 | attn_out | restore | 20 | 9.2->9.25 | -0.05000000000000071 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 19 | mlp_out | restore | 20 | 9.2->9.20 | 0.0 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 20 | mlp_out | restore | 20 | 9.2->9.20 | 0.0 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 15 | attn_out | restore | 20 | 9.2->9.15 | 0.049999999999998934 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 15 | attn_out | restore | 20 | 9.2->9.15 | 0.049999999999998934 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 18 | layer_out | restore | 20 | 9.2->9.15 | 0.049999999999998934 | 0->5 | 1/20 | space:14, correct_prefix:5, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 19 | layer_input | restore | 20 | 9.2->9.15 | 0.049999999999998934 | 0->5 | 1/20 | space:14, correct_prefix:5, newline:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 15 | layer_out | restore | 20 | 9.2->9.10 | 0.09999999999999964 | 0->3 | 0/20 | space:16, correct_prefix:3, explanation:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_input | restore | 20 | 9.2->9.10 | 0.09999999999999964 | 0->3 | 0/20 | space:16, correct_prefix:3, explanation:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 19 | attn_out | restore | 20 | 9.2->9.05 | 0.14999999999999858 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 20 | attn_out | restore | 20 | 9.2->9.05 | 0.14999999999999858 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 14 | layer_input | restore | 20 | 9.2->9.05 | 0.14999999999999858 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 15 | attn_out | restore | 20 | 9.2->9.00 | 0.1999999999999993 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 19 | attn_out | restore | 20 | 9.2->9.00 | 0.1999999999999993 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | attn_out | restore | 20 | 9.2->9.00 | 0.1999999999999993 | 0->0 | 1/20 | space:19, newline:1 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 16 | attn_out | restore | 20 | 9.2->8.95 | 0.25 | 0->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 22 | layer_out | restore | 20 | 9.2->8.95 | 0.25 | 0->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 15 | attn_out | restore | 20 | 9.2->8.95 | 0.25 | 0->0 | 1/20 | space:19, newline:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | attn_out | restore | 20 | 9.2->8.95 | 0.25 | 0->0 | 0/20 | space:20 |

## glm4

- raw_cases: 320 / selected_items: 20 / mode_rows: 10160 / time: 9.20 min
- layers: `[14, 15, 16, 17, 18, 19, 20, 21, 22]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- tasks: `['explanation_required', 'yes_no_required']` / positions: `['intent_word', 'instruction_span', 'label_aligned', 'separator', 'relation_tail']`
- filtered: `{'position_missing': 40, 'position_len_mismatch': 20, 'empty_patch': 0}` / selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 0, 'fallback_used': 0, 'scanned': 20}`

### Baselines

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | explanation_required |  |  |  |  |  | 20 | 58.3->58.30 | 0.0 | 0->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed |  |  |  |  |  | 20 | 2.2->2.20 | 0.0 | 3->3 | 0/20 | space:17, correct_prefix:3 |
| yes_no_required | short_value_allowed |  |  |  |  |  | 20 | 2.2->2.20 | 0.0 | 3->3 | 0/20 | space:17, correct_prefix:3 |
| yes_no_required | yes_no_required |  |  |  |  |  | 20 | 188.1->188.10 | 0.0 | 0->0 | 0/20 | explanation:20 |

### Strongest Absorption: value_to_task

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_out | restore | 20 | 188.1->3.35 | 184.75 | 0->3 | 0/20 | space:15, correct_prefix:3, explanation:2 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_out | restore | 20 | 188.1->3.45 | 184.65 | 0->3 | 0/20 | space:15, correct_prefix:3, explanation:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_out | restore | 20 | 188.1->3.85 | 184.25 | 0->4 | 0/20 | space:9, explanation:7, correct_prefix:4 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_out | restore | 20 | 188.1->4.10 | 184.0 | 0->3 | 0/20 | explanation:10, space:7, correct_prefix:3 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_input | restore | 20 | 188.1->4.10 | 184.0 | 0->3 | 0/20 | explanation:10, space:7, correct_prefix:3 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_out | restore | 20 | 188.1->4.30 | 183.79999999999998 | 0->3 | 0/20 | explanation:10, space:7, correct_prefix:3 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_input | restore | 20 | 188.1->4.30 | 183.79999999999998 | 0->3 | 0/20 | explanation:10, space:7, correct_prefix:3 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_out | restore | 20 | 188.1->5.25 | 182.85 | 0->1 | 0/20 | explanation:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_input | restore | 20 | 188.1->5.25 | 182.85 | 0->1 | 0/20 | explanation:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_out | restore | 20 | 188.1->5.60 | 182.5 | 0->0 | 0/20 | explanation:14, space:6 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_input | restore | 20 | 188.1->5.60 | 182.5 | 0->0 | 0/20 | explanation:14, space:6 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_out | restore | 20 | 188.1->8.30 | 179.79999999999998 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_input | restore | 20 | 188.1->8.30 | 179.79999999999998 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_out | restore | 20 | 188.1->8.55 | 179.54999999999998 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_input | restore | 20 | 188.1->8.55 | 179.54999999999998 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_out | restore | 20 | 188.1->8.60 | 179.5 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_input | restore | 20 | 188.1->8.60 | 179.5 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_out | restore | 20 | 188.1->9.85 | 178.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_input | restore | 20 | 188.1->9.85 | 178.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_out | restore | 20 | 188.1->15.85 | 172.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_input | restore | 20 | 188.1->15.85 | 172.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_out | restore | 20 | 188.1->16.25 | 171.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_input | restore | 20 | 188.1->16.25 | 171.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_out | restore | 20 | 188.1->16.50 | 171.6 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_input | restore | 20 | 188.1->16.50 | 171.6 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_out | restore | 20 | 188.1->32.55 | 155.55 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_input | restore | 20 | 188.1->32.55 | 155.55 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_out | restore | 20 | 188.1->33.85 | 154.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_input | restore | 20 | 188.1->33.85 | 154.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_out | restore | 20 | 188.1->35.05 | 153.05 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_input | restore | 20 | 188.1->35.05 | 153.05 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_out | restore | 20 | 188.1->37.85 | 150.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_input | restore | 20 | 188.1->37.85 | 150.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_out | restore | 20 | 188.1->40.25 | 147.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_input | restore | 20 | 188.1->40.25 | 147.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_out | restore | 20 | 188.1->50.65 | 137.45 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_input | restore | 20 | 188.1->50.65 | 137.45 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_out | restore | 20 | 188.1->52.75 | 135.35 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_input | restore | 20 | 188.1->52.75 | 135.35 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | attn_out | restore | 20 | 188.1->53.30 | 134.8 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | attn_out | restore | 20 | 188.1->54.00 | 134.1 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | attn_out | restore | 20 | 188.1->54.10 | 134.0 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_out | restore | 20 | 188.1->63.30 | 124.8 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_input | restore | 20 | 188.1->63.30 | 124.8 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_out | restore | 20 | 188.1->77.70 | 110.39999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_input | restore | 20 | 188.1->77.70 | 110.39999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | attn_out | restore | 20 | 188.1->93.00 | 95.1 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | layer_out | restore | 20 | 188.1->95.60 | 92.5 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_input | restore | 20 | 188.1->95.60 | 92.5 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_out | restore | 20 | 188.1->96.95 | 91.14999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_input | restore | 20 | 188.1->96.95 | 91.14999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | attn_out | restore | 20 | 188.1->97.75 | 90.35 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_input | restore | 20 | 188.1->98.90 | 89.19999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_out | restore | 20 | 188.1->101.30 | 86.8 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_input | restore | 20 | 188.1->101.30 | 86.8 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_input | restore | 20 | 188.1->101.45 | 86.64999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | attn_out | restore | 20 | 188.1->102.55 | 85.55 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | mlp_out | restore | 20 | 188.1->105.65 | 82.44999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | mlp_out | restore | 20 | 188.1->106.25 | 81.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | mlp_out | restore | 20 | 188.1->109.25 | 78.85 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | mlp_out | restore | 20 | 188.1->109.50 | 78.6 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | mlp_out | restore | 20 | 188.1->110.15 | 77.94999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | mlp_out | restore | 20 | 188.1->114.50 | 73.6 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | mlp_out | restore | 20 | 188.1->115.20 | 72.89999999999999 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | attn_out | restore | 20 | 188.1->130.65 | 57.44999999999999 | 0->0 | 0/20 | explanation:20 |
| explanation_required | explanation_required | label_aligned | value_to_task | 19 | layer_out | restore | 20 | 58.3->1.50 | 56.8 | 0->15 | 0/20 | correct_prefix:15, space:4, word:1 |
| explanation_required | explanation_required | label_aligned | value_to_task | 20 | layer_input | restore | 20 | 58.3->1.50 | 56.8 | 0->15 | 0/20 | correct_prefix:15, space:4, word:1 |
| explanation_required | explanation_required | separator | value_to_task | 17 | layer_out | restore | 20 | 58.3->1.60 | 56.699999999999996 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| explanation_required | explanation_required | separator | value_to_task | 18 | layer_input | restore | 20 | 58.3->1.60 | 56.699999999999996 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| explanation_required | explanation_required | relation_tail | value_to_task | 17 | layer_out | restore | 20 | 58.3->1.60 | 56.699999999999996 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| explanation_required | explanation_required | relation_tail | value_to_task | 18 | layer_input | restore | 20 | 58.3->1.60 | 56.699999999999996 | 0->12 | 0/20 | correct_prefix:12, space:8 |
| explanation_required | explanation_required | separator | value_to_task | 16 | layer_out | restore | 20 | 58.3->1.75 | 56.55 | 0->11 | 0/20 | correct_prefix:11, space:9 |
| explanation_required | explanation_required | separator | value_to_task | 17 | layer_input | restore | 20 | 58.3->1.75 | 56.55 | 0->11 | 0/20 | correct_prefix:11, space:9 |
| explanation_required | explanation_required | separator | value_to_task | 15 | layer_out | restore | 20 | 58.3->1.75 | 56.55 | 0->10 | 0/20 | correct_prefix:10, space:9, word:1 |
| explanation_required | explanation_required | separator | value_to_task | 16 | layer_input | restore | 20 | 58.3->1.75 | 56.55 | 0->10 | 0/20 | correct_prefix:10, space:9, word:1 |
| explanation_required | explanation_required | relation_tail | value_to_task | 18 | layer_out | restore | 20 | 58.3->1.80 | 56.5 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| explanation_required | explanation_required | relation_tail | value_to_task | 19 | layer_input | restore | 20 | 58.3->1.80 | 56.5 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| explanation_required | explanation_required | relation_tail | value_to_task | 19 | layer_out | restore | 20 | 58.3->1.80 | 56.5 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| explanation_required | explanation_required | relation_tail | value_to_task | 20 | layer_input | restore | 20 | 58.3->1.80 | 56.5 | 0->7 | 0/20 | space:13, correct_prefix:7 |
| explanation_required | explanation_required | label_aligned | value_to_task | 20 | layer_out | restore | 20 | 58.3->1.90 | 56.4 | 0->13 | 0/20 | correct_prefix:13, space:5, word:2 |

### Strongest Suppression: task_to_value

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_out | restore | 20 | 2.2->136.00 | -133.8 | 3->0 | 0/20 | explanation:15, word:5 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_input | restore | 20 | 2.2->136.00 | -133.8 | 3->0 | 0/20 | explanation:15, word:5 |
| yes_no_required | short_value_allowed | separator | task_to_value | 22 | layer_out | restore | 20 | 2.2->135.65 | -133.45000000000002 | 3->0 | 0/20 | explanation:16, word:4 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_out | restore | 20 | 2.2->135.25 | -133.05 | 3->0 | 0/20 | explanation:16, word:4 |
| yes_no_required | short_value_allowed | separator | task_to_value | 21 | layer_out | restore | 20 | 2.2->132.30 | -130.10000000000002 | 3->0 | 0/20 | explanation:15, word:5 |
| yes_no_required | short_value_allowed | separator | task_to_value | 22 | layer_input | restore | 20 | 2.2->132.30 | -130.10000000000002 | 3->0 | 0/20 | explanation:15, word:5 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_out | restore | 20 | 2.2->92.15 | -89.95 | 3->0 | 0/20 | word:13, explanation:7 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_out | restore | 20 | 2.2->85.95 | -83.75 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_input | restore | 20 | 2.2->85.95 | -83.75 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 20 | layer_out | restore | 20 | 2.2->82.15 | -79.95 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 21 | layer_input | restore | 20 | 2.2->82.15 | -79.95 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_out | restore | 20 | 2.2->62.05 | -59.849999999999994 | 3->0 | 0/20 | word:16, space:3, explanation:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_input | restore | 20 | 2.2->62.05 | -59.849999999999994 | 3->0 | 0/20 | word:16, space:3, explanation:1 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_out | restore | 20 | 2.2->61.00 | -58.8 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_input | restore | 20 | 2.2->61.00 | -58.8 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | layer_out | restore | 20 | 2.2->59.30 | -57.099999999999994 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | separator | task_to_value | 20 | layer_input | restore | 20 | 2.2->59.30 | -57.099999999999994 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_out | restore | 20 | 2.2->44.15 | -41.949999999999996 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 22 | layer_out | restore | 20 | 2.2->43.20 | -41.0 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_out | restore | 20 | 2.2->37.15 | -34.949999999999996 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_out | restore | 20 | 2.2->34.35 | -32.15 | 3->0 | 0/20 | space:19, word:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_input | restore | 20 | 2.2->34.35 | -32.15 | 3->0 | 0/20 | space:19, word:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_out | restore | 20 | 2.2->30.70 | -28.5 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_input | restore | 20 | 2.2->30.70 | -28.5 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 21 | layer_out | restore | 20 | 2.2->26.15 | -23.95 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 22 | layer_input | restore | 20 | 2.2->26.15 | -23.95 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_out | restore | 20 | 2.2->24.35 | -22.150000000000002 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_input | restore | 20 | 2.2->24.35 | -22.150000000000002 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 20 | layer_out | restore | 20 | 2.2->21.10 | -18.900000000000002 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 21 | layer_input | restore | 20 | 2.2->21.10 | -18.900000000000002 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_out | restore | 20 | 2.2->17.40 | -15.2 | 3->0 | 0/20 | space:20 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_input | restore | 20 | 2.2->17.40 | -15.2 | 3->0 | 0/20 | space:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_out | restore | 20 | 2.2->16.55 | -14.350000000000001 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_input | restore | 20 | 2.2->16.55 | -14.350000000000001 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | layer_out | restore | 20 | 2.2->15.40 | -13.2 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | separator | task_to_value | 20 | layer_input | restore | 20 | 2.2->15.40 | -13.2 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_out | restore | 20 | 2.2->12.25 | -10.05 | 3->0 | 0/20 | explanation:19, word:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_input | restore | 20 | 2.2->12.25 | -10.05 | 3->0 | 0/20 | explanation:19, word:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_out | restore | 20 | 2.2->11.50 | -9.3 | 3->0 | 0/20 | explanation:20 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_input | restore | 20 | 2.2->11.50 | -9.3 | 3->0 | 0/20 | explanation:20 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_out | restore | 20 | 2.2->11.50 | -9.3 | 3->0 | 0/20 | space:9, explanation:6, word:5 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_input | restore | 20 | 2.2->11.50 | -9.3 | 3->0 | 0/20 | space:9, explanation:6, word:5 |
| yes_no_required | short_value_allowed | separator | task_to_value | 18 | layer_out | restore | 20 | 2.2->10.10 | -7.8999999999999995 | 3->0 | 0/20 | space:9, word:6, explanation:5 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | layer_input | restore | 20 | 2.2->10.10 | -7.8999999999999995 | 3->0 | 0/20 | space:9, word:6, explanation:5 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_out | restore | 20 | 2.2->7.25 | -5.05 | 3->0 | 0/20 | explanation:12, word:8 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_input | restore | 20 | 2.2->7.25 | -5.05 | 3->0 | 0/20 | explanation:12, word:8 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_out | restore | 20 | 2.2->6.85 | -4.6499999999999995 | 3->0 | 0/20 | explanation:18, space:2 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_input | restore | 20 | 2.2->6.85 | -4.6499999999999995 | 3->0 | 0/20 | explanation:18, space:2 |
| explanation_required | short_value_allowed | separator | task_to_value | 18 | layer_out | restore | 20 | 2.2->6.10 | -3.8999999999999995 | 3->0 | 0/20 | explanation:12, word:8 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | layer_input | restore | 20 | 2.2->6.10 | -3.8999999999999995 | 3->0 | 0/20 | explanation:12, word:8 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 18 | layer_out | restore | 20 | 2.2->4.65 | -2.45 | 3->0 | 0/20 | space:12, explanation:5, word:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_input | restore | 20 | 2.2->4.65 | -2.45 | 3->0 | 0/20 | space:12, explanation:5, word:3 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_out | restore | 20 | 2.2->4.50 | -2.3 | 3->0 | 0/20 | space:14, word:6 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_input | restore | 20 | 2.2->4.50 | -2.3 | 3->0 | 0/20 | space:14, word:6 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | layer_out | restore | 20 | 2.2->4.30 | -2.0999999999999996 | 3->0 | 0/20 | space:15, word:5 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_input | restore | 20 | 2.2->4.30 | -2.0999999999999996 | 3->0 | 0/20 | space:15, word:5 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_out | restore | 20 | 2.2->4.25 | -2.05 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_input | restore | 20 | 2.2->4.25 | -2.05 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | layer_out | restore | 20 | 2.2->4.20 | -2.0 | 3->0 | 0/20 | space:15, word:5 |
| explanation_required | short_value_allowed | separator | task_to_value | 18 | layer_input | restore | 20 | 2.2->4.20 | -2.0 | 3->0 | 0/20 | space:15, word:5 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_out | restore | 20 | 2.2->4.15 | -1.9500000000000002 | 3->0 | 0/20 | space:17, word:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 18 | layer_input | restore | 20 | 2.2->4.15 | -1.9500000000000002 | 3->0 | 0/20 | space:17, word:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 15 | layer_out | restore | 20 | 2.2->4.10 | -1.8999999999999995 | 3->0 | 0/20 | space:17, word:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_input | restore | 20 | 2.2->4.10 | -1.8999999999999995 | 3->0 | 0/20 | space:17, word:3 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | layer_out | restore | 20 | 2.2->4.05 | -1.8499999999999996 | 3->0 | 0/20 | space:14, word:6 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | layer_input | restore | 20 | 2.2->4.05 | -1.8499999999999996 | 3->0 | 0/20 | space:14, word:6 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_out | restore | 20 | 2.2->4.00 | -1.7999999999999998 | 3->0 | 0/20 | space:18, word:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_input | restore | 20 | 2.2->4.00 | -1.7999999999999998 | 3->0 | 0/20 | space:18, word:2 |
| explanation_required | short_value_allowed | separator | task_to_value | 15 | layer_out | restore | 20 | 2.2->3.95 | -1.75 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | layer_input | restore | 20 | 2.2->3.95 | -1.75 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 15 | layer_out | restore | 20 | 2.2->3.90 | -1.6999999999999997 | 3->0 | 0/20 | space:14, word:6 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | layer_input | restore | 20 | 2.2->3.90 | -1.6999999999999997 | 3->0 | 0/20 | space:14, word:6 |
| yes_no_required | short_value_allowed | separator | task_to_value | 17 | layer_out | restore | 20 | 2.2->3.55 | -1.3499999999999996 | 3->0 | 0/20 | space:18, word:2 |
| yes_no_required | short_value_allowed | separator | task_to_value | 18 | layer_input | restore | 20 | 2.2->3.55 | -1.3499999999999996 | 3->0 | 0/20 | space:18, word:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 16 | layer_out | restore | 20 | 2.2->3.50 | -1.2999999999999998 | 3->0 | 0/20 | space:19, word:1 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_input | restore | 20 | 2.2->3.50 | -1.2999999999999998 | 3->0 | 0/20 | space:19, word:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 14 | layer_out | restore | 20 | 2.2->3.35 | -1.15 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 15 | layer_input | restore | 20 | 2.2->3.35 | -1.15 | 3->0 | 0/20 | space:16, word:4 |
| explanation_required | short_value_allowed | separator | task_to_value | 14 | layer_out | restore | 20 | 2.2->3.35 | -1.15 | 3->0 | 0/20 | space:15, word:5 |
| explanation_required | short_value_allowed | separator | task_to_value | 15 | layer_input | restore | 20 | 2.2->3.35 | -1.15 | 3->0 | 0/20 | space:15, word:5 |

## deepseek7b

- raw_cases: 320 / selected_items: 20 / mode_rows: 10160 / time: 7.86 min
- layers: `[14, 15, 16, 17, 18, 19, 20, 21, 22]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']`
- tasks: `['explanation_required', 'yes_no_required']` / positions: `['intent_word', 'instruction_span', 'label_aligned', 'separator', 'relation_tail']`
- filtered: `{'position_missing': 40, 'position_len_mismatch': 20, 'empty_patch': 0}` / selection: `{'mode_v_correct_seen': 20, 'repair_correct_seen': 20, 'target_failure_seen': 6, 'fallback_used': 0, 'scanned': 23}`

### Baselines

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| explanation_required | explanation_required |  |  |  |  |  | 20 | 77.2->77.20 | 0.0 | 0->0 | 9/20 | word:10, newline:9, explanation:1 |
| explanation_required | short_value_allowed |  |  |  |  |  | 20 | 8.0->8.00 | 0.0 | 0->0 | 10/20 | newline:10, space:10 |
| yes_no_required | short_value_allowed |  |  |  |  |  | 20 | 8.0->8.00 | 0.0 | 0->0 | 10/20 | newline:10, space:10 |
| yes_no_required | yes_no_required |  |  |  |  |  | 20 | 295.55->295.55 | 0.0 | 0->0 | 0/20 | explanation:20 |

### Strongest Absorption: value_to_task

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_out | restore | 20 | 295.55->13.85 | 281.7 | 0->0 | 11/20 | newline:11, space:9 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_out | restore | 20 | 295.55->14.10 | 281.45 | 0->0 | 10/20 | newline:10, space:10 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_out | restore | 20 | 295.55->14.65 | 280.90000000000003 | 0->0 | 7/20 | space:13, newline:7 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_out | restore | 20 | 295.55->19.35 | 276.2 | 0->0 | 12/20 | newline:12, space:8 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | layer_input | restore | 20 | 295.55->19.35 | 276.2 | 0->0 | 12/20 | newline:12, space:8 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_out | restore | 20 | 295.55->19.55 | 276.0 | 0->1 | 13/20 | newline:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_input | restore | 20 | 295.55->19.55 | 276.0 | 0->1 | 13/20 | newline:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | layer_out | restore | 20 | 295.55->19.70 | 275.85 | 0->0 | 12/20 | newline:12, space:8 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | layer_input | restore | 20 | 295.55->19.70 | 275.85 | 0->0 | 12/20 | newline:12, space:8 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_out | restore | 20 | 295.55->20.10 | 275.45 | 0->1 | 13/20 | newline:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | layer_input | restore | 20 | 295.55->20.10 | 275.45 | 0->1 | 13/20 | newline:13, space:6, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_out | restore | 20 | 295.55->20.90 | 274.65000000000003 | 0->1 | 12/20 | newline:12, space:7, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_input | restore | 20 | 295.55->20.90 | 274.65000000000003 | 0->1 | 12/20 | newline:12, space:7, correct_prefix:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | layer_out | restore | 20 | 295.55->21.75 | 273.8 | 0->0 | 10/20 | newline:10, space:10 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | layer_input | restore | 20 | 295.55->21.75 | 273.8 | 0->0 | 10/20 | newline:10, space:10 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_out | restore | 20 | 295.55->22.30 | 273.25 | 0->0 | 9/20 | explanation:11, newline:9 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_input | restore | 20 | 295.55->22.30 | 273.25 | 0->0 | 9/20 | explanation:11, newline:9 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | layer_out | restore | 20 | 295.55->22.45 | 273.1 | 0->0 | 15/20 | newline:15, space:5 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | layer_input | restore | 20 | 295.55->22.45 | 273.1 | 0->0 | 15/20 | newline:15, space:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_out | restore | 20 | 295.55->22.65 | 272.90000000000003 | 0->0 | 17/20 | newline:17, space:2, explanation:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_input | restore | 20 | 295.55->22.65 | 272.90000000000003 | 0->0 | 17/20 | newline:17, space:2, explanation:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_out | restore | 20 | 295.55->23.05 | 272.5 | 0->0 | 17/20 | newline:17, space:2, explanation:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_input | restore | 20 | 295.55->23.05 | 272.5 | 0->0 | 17/20 | newline:17, space:2, explanation:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | layer_out | restore | 20 | 295.55->26.70 | 268.85 | 0->0 | 15/20 | newline:15, space:5 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | layer_input | restore | 20 | 295.55->26.70 | 268.85 | 0->0 | 15/20 | newline:15, space:5 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | layer_out | restore | 20 | 295.55->27.15 | 268.40000000000003 | 0->0 | 16/20 | newline:16, space:4 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | layer_input | restore | 20 | 295.55->27.15 | 268.40000000000003 | 0->0 | 16/20 | newline:16, space:4 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_out | restore | 20 | 295.55->32.25 | 263.3 | 0->0 | 2/20 | explanation:18, newline:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | layer_input | restore | 20 | 295.55->32.25 | 263.3 | 0->0 | 2/20 | explanation:18, newline:2 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_out | restore | 20 | 295.55->32.95 | 262.6 | 0->0 | 7/20 | explanation:13, newline:7 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | layer_input | restore | 20 | 295.55->32.95 | 262.6 | 0->0 | 7/20 | explanation:13, newline:7 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_out | restore | 20 | 295.55->34.50 | 261.05 | 0->0 | 8/20 | explanation:12, newline:8 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | layer_input | restore | 20 | 295.55->34.50 | 261.05 | 0->0 | 8/20 | explanation:12, newline:8 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_out | restore | 20 | 295.55->40.05 | 255.5 | 0->0 | 4/20 | explanation:16, newline:4 |
| yes_no_required | yes_no_required | separator | value_to_task | 17 | layer_input | restore | 20 | 295.55->40.05 | 255.5 | 0->0 | 4/20 | explanation:16, newline:4 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_out | restore | 20 | 295.55->40.95 | 254.60000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 17 | layer_input | restore | 20 | 295.55->40.95 | 254.60000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_out | restore | 20 | 295.55->45.70 | 249.85000000000002 | 0->0 | 3/20 | explanation:16, newline:3, space:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 17 | layer_input | restore | 20 | 295.55->45.70 | 249.85000000000002 | 0->0 | 3/20 | explanation:16, newline:3, space:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | layer_out | restore | 20 | 295.55->64.65 | 230.9 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | layer_input | restore | 20 | 295.55->64.65 | 230.9 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_out | restore | 20 | 295.55->66.70 | 228.85000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_input | restore | 20 | 295.55->66.70 | 228.85000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 15 | layer_out | restore | 20 | 295.55->71.20 | 224.35000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | layer_input | restore | 20 | 295.55->71.20 | 224.35000000000002 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_out | restore | 20 | 295.55->71.50 | 224.05 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_input | restore | 20 | 295.55->71.50 | 224.05 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 14 | layer_out | restore | 20 | 295.55->76.20 | 219.35000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | layer_input | restore | 20 | 295.55->76.20 | 219.35000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 15 | layer_out | restore | 20 | 295.55->77.15 | 218.4 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | layer_input | restore | 20 | 295.55->77.15 | 218.4 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | layer_input | restore | 20 | 295.55->82.35 | 213.20000000000002 | 0->0 | 2/20 | explanation:18, newline:2 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | layer_input | restore | 20 | 295.55->86.20 | 209.35000000000002 | 0->0 | 2/20 | explanation:18, newline:2 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 14 | layer_input | restore | 20 | 295.55->92.50 | 203.05 | 0->0 | 1/20 | explanation:19, newline:1 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 19 | mlp_out | restore | 20 | 295.55->104.80 | 190.75 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | mlp_out | restore | 20 | 295.55->108.95 | 186.60000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 19 | mlp_out | restore | 20 | 295.55->111.80 | 183.75 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | mlp_out | restore | 20 | 295.55->142.90 | 152.65 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | mlp_out | restore | 20 | 295.55->143.30 | 152.25 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | mlp_out | restore | 20 | 295.55->144.80 | 150.75 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 21 | mlp_out | restore | 20 | 295.55->147.25 | 148.3 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 18 | attn_out | restore | 20 | 295.55->148.35 | 147.20000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 18 | attn_out | restore | 20 | 295.55->148.50 | 147.05 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 21 | mlp_out | restore | 20 | 295.55->148.55 | 147.0 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | attn_out | restore | 20 | 295.55->152.70 | 142.85000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 21 | mlp_out | restore | 20 | 295.55->154.65 | 140.9 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 16 | mlp_out | restore | 20 | 295.55->160.20 | 135.35000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 16 | mlp_out | restore | 20 | 295.55->161.80 | 133.75 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 16 | mlp_out | restore | 20 | 295.55->167.10 | 128.45000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 22 | attn_out | restore | 20 | 295.55->175.25 | 120.30000000000001 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 22 | attn_out | restore | 20 | 295.55->175.70 | 119.85000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 22 | attn_out | restore | 20 | 295.55->175.90 | 119.65 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 14 | mlp_out | restore | 20 | 295.55->184.35 | 111.20000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | relation_tail | value_to_task | 20 | attn_out | restore | 20 | 295.55->185.55 | 110.0 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 20 | attn_out | restore | 20 | 295.55->190.80 | 104.75 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 19 | attn_out | restore | 20 | 295.55->194.65 | 100.9 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 20 | attn_out | restore | 20 | 295.55->195.25 | 100.30000000000001 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | separator | value_to_task | 14 | mlp_out | restore | 20 | 295.55->195.45 | 100.10000000000002 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 15 | attn_out | restore | 20 | 295.55->198.05 | 97.5 | 0->0 | 0/20 | explanation:20 |
| yes_no_required | yes_no_required | label_aligned | value_to_task | 18 | mlp_out | restore | 20 | 295.55->202.50 | 93.05000000000001 | 0->0 | 0/20 | explanation:20 |

### Strongest Suppression: task_to_value

| task | eval_task | position | direction | layer | component | control | n | rank base->patch | rank_improve | tok0 base->patch | newline | top0_category |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| yes_no_required | short_value_allowed | separator | task_to_value | 22 | layer_out | restore | 20 | 8.0->61.80 | -53.8 | 0->0 | 17/20 | newline:17, explanation:3 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_out | restore | 20 | 8.0->61.75 | -53.75 | 0->0 | 17/20 | newline:17, explanation:3 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_out | restore | 20 | 8.0->60.35 | -52.35 | 0->0 | 17/20 | newline:17, explanation:3 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_out | restore | 20 | 8.0->58.25 | -50.25 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_input | restore | 20 | 8.0->58.25 | -50.25 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | layer_out | restore | 20 | 8.0->57.70 | -49.7 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | separator | task_to_value | 20 | layer_input | restore | 20 | 8.0->57.70 | -49.7 | 0->0 | 18/20 | newline:18, explanation:2 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_out | restore | 20 | 8.0->56.15 | -48.15 | 0->0 | 8/20 | word:11, newline:8, explanation:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 22 | layer_out | restore | 20 | 8.0->55.30 | -47.3 | 0->0 | 8/20 | word:12, newline:8 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_out | restore | 20 | 8.0->54.65 | -46.65 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_input | restore | 20 | 8.0->54.65 | -46.65 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | separator | task_to_value | 20 | layer_out | restore | 20 | 8.0->54.50 | -46.5 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | separator | task_to_value | 21 | layer_input | restore | 20 | 8.0->54.50 | -46.5 | 0->0 | 18/20 | newline:18, explanation:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_out | restore | 20 | 8.0->54.25 | -46.25 | 0->0 | 14/20 | newline:14, explanation:6 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_input | restore | 20 | 8.0->54.25 | -46.25 | 0->0 | 14/20 | newline:14, explanation:6 |
| yes_no_required | short_value_allowed | separator | task_to_value | 21 | layer_out | restore | 20 | 8.0->54.20 | -46.2 | 0->0 | 13/20 | newline:13, explanation:7 |
| yes_no_required | short_value_allowed | separator | task_to_value | 22 | layer_input | restore | 20 | 8.0->54.20 | -46.2 | 0->0 | 13/20 | newline:13, explanation:7 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_out | restore | 20 | 8.0->54.05 | -46.05 | 0->0 | 19/20 | newline:19, explanation:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_input | restore | 20 | 8.0->54.05 | -46.05 | 0->0 | 19/20 | newline:19, explanation:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_out | restore | 20 | 8.0->53.55 | -45.55 | 0->0 | 9/20 | word:11, newline:9 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_out | restore | 20 | 8.0->51.10 | -43.1 | 0->0 | 17/20 | newline:17, explanation:3 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_input | restore | 20 | 8.0->51.10 | -43.1 | 0->0 | 17/20 | newline:17, explanation:3 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_out | restore | 20 | 8.0->50.70 | -42.7 | 0->0 | 19/20 | newline:19, explanation:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_input | restore | 20 | 8.0->50.70 | -42.7 | 0->0 | 19/20 | newline:19, explanation:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_out | restore | 20 | 8.0->41.25 | -33.25 | 0->0 | 6/20 | word:14, newline:6 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 22 | layer_input | restore | 20 | 8.0->41.25 | -33.25 | 0->0 | 6/20 | word:14, newline:6 |
| explanation_required | short_value_allowed | separator | task_to_value | 21 | layer_out | restore | 20 | 8.0->40.20 | -32.2 | 0->0 | 6/20 | word:14, newline:6 |
| explanation_required | short_value_allowed | separator | task_to_value | 22 | layer_input | restore | 20 | 8.0->40.20 | -32.2 | 0->0 | 6/20 | word:14, newline:6 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_out | restore | 20 | 8.0->39.05 | -31.049999999999997 | 0->0 | 8/20 | word:11, newline:8, explanation:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 21 | layer_input | restore | 20 | 8.0->39.05 | -31.049999999999997 | 0->0 | 8/20 | word:11, newline:8, explanation:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_out | restore | 20 | 8.0->38.70 | -30.700000000000003 | 0->0 | 6/20 | word:13, newline:6, explanation:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 22 | layer_input | restore | 20 | 8.0->38.70 | -30.700000000000003 | 0->0 | 6/20 | word:13, newline:6, explanation:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 20 | layer_out | restore | 20 | 8.0->38.20 | -30.200000000000003 | 0->0 | 7/20 | word:12, newline:7, explanation:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 21 | layer_input | restore | 20 | 8.0->38.20 | -30.200000000000003 | 0->0 | 7/20 | word:12, newline:7, explanation:1 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_out | restore | 20 | 8.0->37.20 | -29.200000000000003 | 0->0 | 19/20 | newline:19, space:1 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_input | restore | 20 | 8.0->37.20 | -29.200000000000003 | 0->0 | 19/20 | newline:19, space:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 18 | layer_out | restore | 20 | 8.0->36.75 | -28.75 | 0->0 | 19/20 | newline:19, space:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 19 | layer_input | restore | 20 | 8.0->36.75 | -28.75 | 0->0 | 19/20 | newline:19, space:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_out | restore | 20 | 8.0->34.60 | -26.6 | 0->0 | 7/20 | word:13, newline:7 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 21 | layer_input | restore | 20 | 8.0->34.60 | -26.6 | 0->0 | 7/20 | word:13, newline:7 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_out | restore | 20 | 8.0->32.30 | -24.299999999999997 | 0->0 | 8/20 | word:10, newline:8, explanation:2 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 20 | layer_input | restore | 20 | 8.0->32.30 | -24.299999999999997 | 0->0 | 8/20 | word:10, newline:8, explanation:2 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | layer_out | restore | 20 | 8.0->31.95 | -23.95 | 0->0 | 8/20 | newline:8, word:8, explanation:4 |
| explanation_required | short_value_allowed | separator | task_to_value | 20 | layer_input | restore | 20 | 8.0->31.95 | -23.95 | 0->0 | 8/20 | newline:8, word:8, explanation:4 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 18 | layer_out | restore | 20 | 8.0->30.35 | -22.35 | 0->0 | 19/20 | newline:19, space:1 |
| yes_no_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_input | restore | 20 | 8.0->30.35 | -22.35 | 0->0 | 19/20 | newline:19, space:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | mlp_out | restore | 20 | 8.0->29.85 | -21.85 | 0->0 | 16/20 | newline:16, space:4 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | mlp_out | restore | 20 | 8.0->28.15 | -20.15 | 0->0 | 14/20 | newline:14, space:6 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | mlp_out | restore | 20 | 8.0->27.35 | -19.35 | 0->0 | 15/20 | newline:15, space:5 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_out | restore | 20 | 8.0->25.25 | -17.25 | 0->0 | 8/20 | newline:8, word:7, explanation:5 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 19 | layer_input | restore | 20 | 8.0->25.25 | -17.25 | 0->0 | 8/20 | newline:8, word:7, explanation:5 |
| explanation_required | short_value_allowed | separator | task_to_value | 18 | layer_out | restore | 20 | 8.0->25.15 | -17.15 | 0->0 | 8/20 | newline:8, word:7, explanation:5 |
| explanation_required | short_value_allowed | separator | task_to_value | 19 | layer_input | restore | 20 | 8.0->25.15 | -17.15 | 0->0 | 8/20 | newline:8, word:7, explanation:5 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_out | restore | 20 | 8.0->24.60 | -16.6 | 0->0 | 8/20 | word:9, newline:8, explanation:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 20 | layer_input | restore | 20 | 8.0->24.60 | -16.6 | 0->0 | 8/20 | word:9, newline:8, explanation:3 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_out | restore | 20 | 8.0->24.45 | -16.45 | 0->0 | 18/20 | newline:18, space:2 |
| yes_no_required | short_value_allowed | relation_tail | task_to_value | 18 | layer_input | restore | 20 | 8.0->24.45 | -16.45 | 0->0 | 18/20 | newline:18, space:2 |
| explanation_required | short_value_allowed | instruction_span | task_to_value | 14 | layer_input | restore | 20 | 8.0->24.30 | -16.3 | 0->0 | 9/20 | newline:9, space:6, word:5 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 18 | layer_out | restore | 20 | 8.0->22.25 | -14.25 | 0->0 | 8/20 | word:10, newline:8, explanation:2 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 19 | layer_input | restore | 20 | 8.0->22.25 | -14.25 | 0->0 | 8/20 | word:10, newline:8, explanation:2 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 15 | layer_out | restore | 20 | 8.0->22.20 | -14.2 | 0->0 | 11/20 | newline:11, explanation:8, word:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | layer_input | restore | 20 | 8.0->22.20 | -14.2 | 0->0 | 11/20 | newline:11, explanation:8, word:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 17 | layer_out | restore | 20 | 8.0->22.20 | -14.2 | 0->0 | 19/20 | newline:19, space:1 |
| yes_no_required | short_value_allowed | separator | task_to_value | 18 | layer_input | restore | 20 | 8.0->22.20 | -14.2 | 0->0 | 19/20 | newline:19, space:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 15 | layer_out | restore | 20 | 8.0->21.70 | -13.7 | 0->0 | 11/20 | newline:11, explanation:6, word:3 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_input | restore | 20 | 8.0->21.70 | -13.7 | 0->0 | 11/20 | newline:11, explanation:6, word:3 |
| explanation_required | short_value_allowed | separator | task_to_value | 15 | layer_out | restore | 20 | 8.0->21.65 | -13.649999999999999 | 0->0 | 12/20 | newline:12, explanation:8 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | layer_input | restore | 20 | 8.0->21.65 | -13.649999999999999 | 0->0 | 12/20 | newline:12, explanation:8 |
| explanation_required | short_value_allowed | separator | task_to_value | 14 | layer_out | restore | 20 | 8.0->21.60 | -13.600000000000001 | 0->0 | 13/20 | newline:13, explanation:6, word:1 |
| explanation_required | short_value_allowed | separator | task_to_value | 15 | layer_input | restore | 20 | 8.0->21.60 | -13.600000000000001 | 0->0 | 13/20 | newline:13, explanation:6, word:1 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 14 | layer_out | restore | 20 | 8.0->20.95 | -12.95 | 0->0 | 13/20 | newline:13, explanation:7 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 15 | layer_input | restore | 20 | 8.0->20.95 | -12.95 | 0->0 | 13/20 | newline:13, explanation:7 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 16 | layer_out | restore | 20 | 8.0->20.95 | -12.95 | 0->0 | 6/20 | explanation:9, newline:6, word:5 |
| explanation_required | short_value_allowed | relation_tail | task_to_value | 17 | layer_input | restore | 20 | 8.0->20.95 | -12.95 | 0->0 | 6/20 | explanation:9, newline:6, word:5 |
| explanation_required | short_value_allowed | separator | task_to_value | 16 | layer_out | restore | 20 | 8.0->20.85 | -12.850000000000001 | 0->0 | 6/20 | word:8, newline:6, explanation:6 |
| explanation_required | short_value_allowed | separator | task_to_value | 17 | layer_input | restore | 20 | 8.0->20.85 | -12.850000000000001 | 0->0 | 6/20 | word:8, newline:6, explanation:6 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 16 | layer_out | restore | 20 | 8.0->20.30 | -12.3 | 0->0 | 8/20 | word:11, newline:8, explanation:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_input | restore | 20 | 8.0->20.30 | -12.3 | 0->0 | 8/20 | word:11, newline:8, explanation:1 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 17 | layer_out | restore | 20 | 8.0->20.25 | -12.25 | 0->0 | 8/20 | newline:8, word:8, explanation:4 |
| explanation_required | short_value_allowed | label_aligned | task_to_value | 18 | layer_input | restore | 20 | 8.0->20.25 | -12.25 | 0->0 | 8/20 | newline:8, word:8, explanation:4 |
