# Phase 648 Cross-Model Summary

目标：把 Phase647 的 separator writer candidate graph 扩展到多个提示边界位置，检查 value_short_answer_protocol 是单边界现象还是多位置协议场。

## qwen3

- raw_cases: 320 / target_seen: 26 / cases_written: 26 / mode_rows: 2964
- layers: `[17, 18, 19, 20]` / positions: `['separator', 'answer_label', 'prompt_last', 'question_mark_answer', 'relation_tail']` / target_only: True
- filtered: `{'not_target': 294, 'position_missing': 0, 'position_len_mismatch': 26, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 6.58

### Baselines

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| original | 26 |  |  | baseline |  |  |  | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| inline | 26 |  |  | baseline |  |  |  | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |

### Position Best Rows

| position | best sufficiency | exact | newline | rank | best necessity/remove | exact | newline | rank |
|---|---|---:|---:|---:|---|---:|---:|---:|
| separator | separator_to_original_interval_L18_19_attn_out_restore | 21/26 | 0/26 | 1.1 | separator_remove_from_inline_interval_L18_19_mlp_out_restore | 0/26 | 26/26 | 3.7 |
| answer_label |  |  |  |  |  |  |  |  |
| prompt_last | prompt_last_to_original_interval_L18_19_attn_out_restore | 23/26 | 0/26 | 1.0 | prompt_last_remove_from_inline_L17_layer_input_restore | 0/26 | 14/26 | 4.5 |
| question_mark_answer | question_mark_answer_to_original_interval_L17_20_attn_out_restore | 23/26 | 0/26 | 1.0 | question_mark_answer_remove_from_inline_interval_L18_19_mlp_out_restore | 7/26 | 16/26 | 2.2 |
| relation_tail | relation_tail_to_original_interval_L18_19_attn_out_restore | 19/26 | 0/26 | 1.1 | relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 9/26 | 12/26 | 4.9 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_to_original_interval_L18_19_attn_out_restore | 26 | separator | to_original | interval | L18_19 |  | attn_out | 21/26 | 24/26 | 0/26 | 1.1 | 2.697 | correct_prefix:24, space:2 |  v05:10,  v22:6,  v91:6,  v48:2,  91:1,  22:1 |
| separator_to_original_interval_L17_20_attn_out_restore | 26 | separator | to_original | interval | L17_20 |  | attn_out | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |  v05:11,  v22:6,  v91:4,  91:2,  22:1,  v?\nLet:1,  v48:1 |
| separator_to_original_interval_L17_20_mlp_out_restore | 26 | separator | to_original | interval | L17_20 |  | mlp_out | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |  v05:10,  22:5,  v91:4,  91:2,  v48:2,  v22:2,  48:1 |
| separator_to_original_interval_L18_19_mlp_out_restore | 26 | separator | to_original | interval | L18_19 |  | mlp_out | 16/26 | 18/26 | 2/26 | 1.3 | 1.087 | correct_prefix:18, space:6, newline:2 |  v05:6,  v91:6,  22:4,  v48:3,  v22:3,  91:2,  ?\n\nOkay,:2 |
| separator_to_original_L18_layer_out_restore | 26 | separator | to_original | single_layer |  | 18 | layer_out | 4/26 | 4/26 | 19/26 | 3.6 | -1.173 | newline:19, correct_prefix:4, space:3 |  ?\n\nOkay,:19,  91:2,  v05:1,  v48:1,  22:1,  v22:1,  v91:1 |
| separator_to_original_L19_layer_input_restore | 26 | separator | to_original | single_layer |  | 19 | layer_input | 4/26 | 4/26 | 19/26 | 3.6 | -1.173 | newline:19, correct_prefix:4, space:3 |  ?\n\nOkay,:19,  91:2,  v05:1,  v48:1,  22:1,  v22:1,  v91:1 |
| separator_to_original_L17_layer_out_restore | 26 | separator | to_original | single_layer |  | 17 | layer_out | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |  ?\n\nOkay,:19,  v22:2,  v05:1,  v48:1,  91:1,  22:1,  v91:1 |
| separator_to_original_L18_layer_input_restore | 26 | separator | to_original | single_layer |  | 18 | layer_input | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |  ?\n\nOkay,:19,  v22:2,  v05:1,  v48:1,  91:1,  22:1,  v91:1 |
| separator_to_original_interval_L18_19_layer_out_restore | 26 | separator | to_original | interval | L18_19 |  | layer_out | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |  ?\n\nOkay,:21,  91:1,  v48:1,  22:1,  v22:1,  v91:1 |
| separator_to_original_L19_layer_out_restore | 26 | separator | to_original | single_layer |  | 19 | layer_out | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |  ?\n\nOkay,:21,  91:1,  v48:1,  22:1,  v22:1,  v91:1 |
| separator_to_original_L20_layer_input_restore | 26 | separator | to_original | single_layer |  | 20 | layer_input | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |  ?\n\nOkay,:21,  91:1,  v48:1,  22:1,  v22:1,  v91:1 |
| separator_to_original_interval_L17_20_layer_out_restore | 26 | separator | to_original | interval | L17_20 |  | layer_out | 1/26 | 1/26 | 24/26 | 3.7 | -1.380 | newline:24, space:1, correct_prefix:1 |  ?\n\nOkay,:22,  91:1,  22:1,  v22:1,  v91:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_remove_from_inline_interval_L18_19_mlp_out_restore | 26 | separator | remove_from_inline | interval | L18_19 |  | mlp_out | 0/26 | 0/26 | 26/26 | 3.7 | -1.413 | newline:26 |  ?\n\nOkay,:20,  ?\n\nTo solve:6 |
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 26 | separator | remove_from_inline | interval | L17_20 |  | attn_out | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |  v05:5,  ?\n\nTo solve:4,  22:4,  91:4,  05:3,  ?\n\nOkay,:2,  v22:2,  48:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 26 | separator | remove_from_inline | interval | L18_19 |  | layer_out | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| separator_remove_from_inline_L18_layer_out_restore | 26 | separator | remove_from_inline | single_layer |  | 18 | layer_out | 8/26 | 11/26 | 2/26 | 2.6 | 0.019 | space:13, correct_prefix:11, newline:2 |  22:6,  v05:5,  05:4,  91:4,  v48:2,  ?\n\nOkay,:2,  v22:2,  v91:1 |
| separator_remove_from_inline_L19_layer_input_restore | 26 | separator | remove_from_inline | single_layer |  | 19 | layer_input | 8/26 | 11/26 | 2/26 | 2.6 | 0.019 | space:13, correct_prefix:11, newline:2 |  22:6,  v05:5,  05:4,  91:4,  v48:2,  ?\n\nOkay,:2,  v22:2,  v91:1 |
| separator_remove_from_inline_L17_layer_out_restore | 26 | separator | remove_from_inline | single_layer |  | 17 | layer_out | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |  05:5,  22:5,  91:5,  v05:4,  v22:3,  v48:2,  ?\n\nOkay,:1,  v91:1 |
| separator_remove_from_inline_L18_layer_input_restore | 26 | separator | remove_from_inline | single_layer |  | 18 | layer_input | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |  05:5,  22:5,  91:5,  v05:4,  v22:3,  v48:2,  ?\n\nOkay,:1,  v91:1 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 26 | separator | remove_from_inline | interval | L17_20 |  | layer_out | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |  91:7,  05:5,  22:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| separator_remove_from_inline_L20_layer_out_restore | 26 | separator | remove_from_inline | single_layer |  | 20 | layer_out | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |  91:7,  05:5,  22:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| separator_remove_from_inline_L19_layer_out_restore | 26 | separator | remove_from_inline | single_layer |  | 19 | layer_out | 8/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v91:2,  v22:1 |
| separator_remove_from_inline_L20_layer_input_restore | 26 | separator | remove_from_inline | single_layer |  | 20 | layer_input | 8/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v91:2,  v22:1 |
| separator_remove_from_inline_L17_layer_input_restore | 26 | separator | remove_from_inline | single_layer |  | 17 | layer_input | 10/26 | 11/26 | 2/26 | 2.5 | 0.240 | space:13, correct_prefix:11, newline:2 |  v05:7,  22:5,  91:5,  05:4,  v48:2,  v22:2,  ?\n\nOkay,:1 |

### answer_label

#### Best Sufficiency Restore

No rows.

#### Best Necessity Remove

No rows.

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_to_original_interval_L18_19_attn_out_restore | 26 | prompt_last | to_original | interval | L18_19 |  | attn_out | 23/26 | 25/26 | 0/26 | 1.0 | 2.774 | correct_prefix:25, space:1 |  v05:10,  v22:7,  v91:5,  v48:3,  91:1 |
| prompt_last_to_original_interval_L17_20_mlp_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | mlp_out | 21/26 | 22/26 | 0/26 | 1.2 | 3.264 | correct_prefix:22, space:4 |  v05:11,  v22:5,  v91:4,  91:3,  48:1,  v48:1,  22:1 |
| prompt_last_to_original_interval_L17_20_attn_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | attn_out | 18/26 | 21/26 | 0/26 | 1.3 | 3.303 | correct_prefix:21, space:3, word:2 |  v05:9,  v22:6,  v91:4,  05:2,  What is the:2,  22:1,  v?\n\nLet:1,  v48:1 |
| prompt_last_to_original_L17_layer_input_restore | 26 | prompt_last | to_original | single_layer |  | 17 | layer_input | 15/26 | 17/26 | 3/26 | 1.4 | 0.822 | correct_prefix:17, space:6, newline:3 |  v05:7,  v22:5,  22:3,  v48:3,  v91:3,  05:2,  91:1,  ?\n\nOkay,:1 |
| prompt_last_to_original_interval_L18_19_mlp_out_restore | 26 | prompt_last | to_original | interval | L18_19 |  | mlp_out | 14/26 | 17/26 | 0/26 | 1.3 | 1.279 | correct_prefix:17, space:9 |  v05:5,  v91:5,  22:4,  v48:3,  91:3,  v22:3,  05:3 |
| prompt_last_to_original_L17_layer_out_restore | 26 | prompt_last | to_original | single_layer |  | 17 | layer_out | 12/26 | 14/26 | 7/26 | 1.8 | 0.413 | correct_prefix:14, newline:7, space:5 |  v05:6,  v22:4,  \n\nOkay,:3,  22:3,  v48:3,  ?\n\nOkay,:2,  05:2,  v91:2 |
| prompt_last_to_original_L18_layer_input_restore | 26 | prompt_last | to_original | single_layer |  | 18 | layer_input | 12/26 | 14/26 | 7/26 | 1.8 | 0.413 | correct_prefix:14, newline:7, space:5 |  v05:6,  v22:4,  \n\nOkay,:3,  22:3,  v48:3,  ?\n\nOkay,:2,  05:2,  v91:2 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | layer_out | 9/26 | 10/26 | 4/26 | 2.3 | 0.279 | space:12, correct_prefix:10, newline:4 |  v05:5,  22:5,  ?\n\nOkay,:4,  05:4,  91:2,  v48:2,  v22:2,  48:1 |
| prompt_last_to_original_L20_layer_out_restore | 26 | prompt_last | to_original | single_layer |  | 20 | layer_out | 9/26 | 10/26 | 4/26 | 2.3 | 0.279 | space:12, correct_prefix:10, newline:4 |  v05:5,  22:5,  ?\n\nOkay,:4,  05:4,  91:2,  v48:2,  v22:2,  48:1 |
| prompt_last_to_original_interval_L18_19_layer_out_restore | 26 | prompt_last | to_original | interval | L18_19 |  | layer_out | 7/26 | 9/26 | 3/26 | 2.4 | 0.337 | space:14, correct_prefix:9, newline:3 |  22:5,  05:5,  91:4,  v05:3,  ?\n\nOkay,:3,  v22:3,  v48:2,  v91:1 |
| prompt_last_to_original_L19_layer_out_restore | 26 | prompt_last | to_original | single_layer |  | 19 | layer_out | 7/26 | 9/26 | 3/26 | 2.4 | 0.337 | space:14, correct_prefix:9, newline:3 |  22:5,  91:4,  05:4,  v05:3,  ?\n\nOkay,:3,  v22:3,  v48:2,  48:1 |
| prompt_last_to_original_L20_layer_input_restore | 26 | prompt_last | to_original | single_layer |  | 20 | layer_input | 7/26 | 9/26 | 3/26 | 2.4 | 0.337 | space:14, correct_prefix:9, newline:3 |  22:5,  91:4,  05:4,  v05:3,  ?\n\nOkay,:3,  v22:3,  v48:2,  48:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_remove_from_inline_L17_layer_input_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_input | 0/26 | 1/26 | 14/26 | 4.5 | -1.260 | newline:14, space:11, correct_prefix:1 |  ?\n\nOkay,:13,  22:5,  05:3,  91:3,  48:1,  v22:1 |
| prompt_last_remove_from_inline_interval_L18_19_mlp_out_restore | 26 | prompt_last | remove_from_inline | interval | L18_19 |  | mlp_out | 2/26 | 3/26 | 17/26 | 3.0 | -0.649 | newline:17, space:6, correct_prefix:3 |  ?\n\nOkay,:16,  22:4,  v05:2,  91:1,  48:1,  ?\n\nTo solve:1,  v22:1 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | 3/26 | 2/26 | 23/26 | 4.4 | -1.697 | newline:23, correct_prefix:2, space:1 |  ?\n\nOkay,:20,  v05:2,  v22:2,  ?\n\nTo solve:1,  05:1 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 26 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | 3/26 | 4/26 | 13/26 | 2.9 | -0.462 | newline:13, space:9, correct_prefix:4 |  ?\n\nOkay,:10,  22:5,  91:4,  v05:2,  v22:2,  05:1,  ?\n\nTo solve:1,  v48:1 |
| prompt_last_remove_from_inline_L19_layer_out_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_out | 3/26 | 4/26 | 13/26 | 2.9 | -0.462 | newline:13, space:9, correct_prefix:4 |  ?\n\nOkay,:11,  22:5,  91:3,  v05:2,  v22:2,  05:1,  48:1,  v48:1 |
| prompt_last_remove_from_inline_L20_layer_input_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 20 | layer_input | 3/26 | 4/26 | 13/26 | 2.9 | -0.462 | newline:13, space:9, correct_prefix:4 |  ?\n\nOkay,:11,  22:5,  91:3,  v05:2,  v22:2,  05:1,  48:1,  v48:1 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | 3/26 | 4/26 | 10/26 | 3.3 | -0.510 | space:12, newline:10, correct_prefix:4 |  ?\n\nOkay,:10,  22:6,  91:5,  05:2,  v05:1,  v48:1,  v91:1 |
| prompt_last_remove_from_inline_L20_layer_out_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 20 | layer_out | 3/26 | 4/26 | 10/26 | 3.3 | -0.510 | space:12, newline:10, correct_prefix:4 |  ?\n\nOkay,:10,  22:6,  91:4,  05:2,  v05:1,  48:1,  v48:1,  v91:1 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | 4/26 | 4/26 | 11/26 | 3.2 | -0.447 | newline:11, space:11, correct_prefix:4 |  ?\n\nOkay,:8,  22:6,  91:4,  05:3,  v05:2,  v48:2,  v22:1 |
| prompt_last_remove_from_inline_L18_layer_input_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_input | 4/26 | 4/26 | 11/26 | 3.2 | -0.447 | newline:11, space:11, correct_prefix:4 |  ?\n\nOkay,:8,  22:6,  91:4,  05:3,  v05:2,  v48:2,  v22:1 |
| prompt_last_remove_from_inline_L18_layer_out_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_out | 5/26 | 8/26 | 8/26 | 2.8 | -0.341 | space:10, correct_prefix:8, newline:8 |  ?\n\nOkay,:8,  22:6,  91:4,  v05:3,  v48:2,  v22:2,  05:1 |
| prompt_last_remove_from_inline_L19_layer_input_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_input | 5/26 | 8/26 | 8/26 | 2.8 | -0.341 | space:10, correct_prefix:8, newline:8 |  ?\n\nOkay,:8,  22:6,  91:4,  v05:3,  v48:2,  v22:2,  05:1 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_to_original_interval_L17_20_attn_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | 23/26 | 26/26 | 0/26 | 1.0 | 4.567 | correct_prefix:26 |  v05:11,  v22:7,  v91:6,  v?\nLet:1,  v48:1 |
| question_mark_answer_to_original_interval_L18_19_attn_out_restore | 26 | question_mark_answer | to_original | interval | L18_19 |  | attn_out | 22/26 | 24/26 | 0/26 | 1.1 | 3.010 | correct_prefix:24, space:2 |  v05:11,  v22:6,  v91:6,  91:1,  22:1,  v48:1 |
| question_mark_answer_to_original_interval_L18_19_mlp_out_restore | 26 | question_mark_answer | to_original | interval | L18_19 |  | mlp_out | 14/26 | 16/26 | 1/26 | 1.7 | 0.673 | correct_prefix:16, space:9, newline:1 |  v05:6,  22:5,  v91:5,  91:4,  v48:2,  v22:2,  ?\n\nOkay,:1,  05:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | 5/26 | 6/26 | 0/26 | 2.2 | 1.106 | space:18, correct_prefix:6, word:2 |  22:12,  48:5,  v05:4,  c12:1,  c33:1,  v91:1,  91:1,  05:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:5,  22:4,  48:1,  05:1,  v22:1 |
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 26 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:5,  22:4,  05:2,  v22:1 |
| question_mark_answer_to_original_L17_layer_input_restore | 26 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| question_mark_answer_to_original_L17_layer_out_restore | 26 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| question_mark_answer_to_original_L18_layer_input_restore | 26 | question_mark_answer | to_original | single_layer |  | 18 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| question_mark_answer_to_original_L18_layer_out_restore | 26 | question_mark_answer | to_original | single_layer |  | 18 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  48:2,  05:1,  v22:1 |
| question_mark_answer_to_original_L19_layer_input_restore | 26 | question_mark_answer | to_original | single_layer |  | 19 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  48:2,  05:1,  v22:1 |
| question_mark_answer_to_original_L19_layer_out_restore | 26 | question_mark_answer | to_original | single_layer |  | 19 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:5,  22:4,  48:1,  05:1,  v22:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_remove_from_inline_interval_L18_19_mlp_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | mlp_out | 7/26 | 7/26 | 16/26 | 2.2 | -0.327 | newline:16, correct_prefix:7, space:3 |  ?\n\nOkay,:14,  v05:3,  v48:2,  v22:2,  22:2,  v91:2,  48:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_attn_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | attn_out | 11/26 | 14/26 | 1/26 | 1.6 | 0.837 | correct_prefix:14, space:11, newline:1 |  v05:5,  22:5,  05:4,  v22:4,  v91:3,  91:2,  v48:2,  \n\nOkay,:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | 12/26 | 14/26 | 9/26 | 3.2 | 0.279 | correct_prefix:14, newline:9, space:2, word:1 |  v05:8,  ?\n\nTo solve:6,  ?\n\nOkay,:3,  v22:3,  22:2,  v91:2,  c33:1,  v48:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:10,  v22:6,  v91:4,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L18_layer_input_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_input | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L18_layer_out_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_out | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L19_layer_input_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 19 | layer_input | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L19_layer_out_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 19 | layer_out | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |
| question_mark_answer_remove_from_inline_L20_layer_input_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 20 | layer_input | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:6,  v91:5,  v48:3,  22:2,  91:1 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_to_original_interval_L18_19_attn_out_restore | 26 | relation_tail | to_original | interval | L18_19 |  | attn_out | 19/26 | 24/26 | 0/26 | 1.1 | 2.678 | correct_prefix:24, space:2 |  v22:8,  v05:8,  v91:6,  v48:2,  91:1,  22:1 |
| relation_tail_to_original_interval_L18_19_mlp_out_restore | 26 | relation_tail | to_original | interval | L18_19 |  | mlp_out | 14/26 | 15/26 | 4/26 | 1.7 | 0.745 | correct_prefix:15, space:7, newline:4 |  v05:6,  91:5,  22:4,  v91:4,  v48:2,  v22:2,  o43:1,  ?\n\nOkay,:1 |
| relation_tail_to_original_interval_L17_20_attn_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | attn_out | 13/26 | 25/26 | 0/26 | 1.1 | 4.183 | correct_prefix:25, word:1 |  v?\nLet:10,  v05:8,  v22:4,  v91:1,  v48:1,  v?\nQuestion:1,  What is the:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | mlp_out | 8/26 | 12/26 | 0/26 | 1.7 | 1.736 | correct_prefix:12, space:12, word:2 |  22:8,  v05:5,  v91:3,  v22:3,  48:3,  91:2,  v48:1,  c33:1 |
| relation_tail_to_original_interval_L17_20_layer_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| relation_tail_to_original_interval_L18_19_layer_out_restore | 26 | relation_tail | to_original | interval | L18_19 |  | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  22:4,  91:3,  05:3,  48:1,  v22:1 |
| relation_tail_to_original_L17_layer_input_restore | 26 | relation_tail | to_original | single_layer |  | 17 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  48:2,  05:1,  v22:1 |
| relation_tail_to_original_L17_layer_out_restore | 26 | relation_tail | to_original | single_layer |  | 17 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| relation_tail_to_original_L18_layer_input_restore | 26 | relation_tail | to_original | single_layer |  | 18 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| relation_tail_to_original_L18_layer_out_restore | 26 | relation_tail | to_original | single_layer |  | 18 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  22:4,  91:3,  48:2,  05:2,  v22:1 |
| relation_tail_to_original_L19_layer_input_restore | 26 | relation_tail | to_original | single_layer |  | 19 | layer_input | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  22:4,  91:3,  48:2,  05:2,  v22:1 |
| relation_tail_to_original_L19_layer_out_restore | 26 | relation_tail | to_original | single_layer |  | 19 | layer_out | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  22:4,  05:3,  48:2,  91:2,  v22:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | 9/26 | 11/26 | 12/26 | 4.9 | -0.673 | newline:12, correct_prefix:11, word:3 |  ?\n\nTo solve:6,  v05:5,  ?\n\nOkay,:5,  v22:4,  What is the:2,  c33:1,  v91:1,  v48:1 |
| relation_tail_remove_from_inline_interval_L18_19_attn_out_restore | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | attn_out | 10/26 | 13/26 | 3/26 | 1.8 | 0.644 | correct_prefix:13, space:10, newline:3 |  22:5,  v05:4,  05:4,  v22:3,  \n\nOkay,:3,  v91:3,  91:2,  v48:2 |
| relation_tail_remove_from_inline_interval_L18_19_mlp_out_restore | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | mlp_out | 13/26 | 15/26 | 9/26 | 2.0 | -0.202 | correct_prefix:15, newline:9, space:2 |  ?\n\nOkay,:8,  v22:5,  v05:4,  v48:3,  v91:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | 16/26 | 26/26 | 0/26 | 1.0 | 4.361 | correct_prefix:26 |  v22:12,  v05:11,  v48:2,  v91:1 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L18_layer_input_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_input | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L18_layer_out_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_out | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L19_layer_input_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 19 | layer_input | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L19_layer_out_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 19 | layer_out | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_L20_layer_input_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 20 | layer_input | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:10,  v22:7,  v91:4,  22:2,  v48:2,  91:1 |

### Global Top Notes

- Top sufficiency: question_mark_answer_to_original_interval_L17_20_attn_out_restore exact=23/26 newline=0/26; prompt_last_to_original_interval_L18_19_attn_out_restore exact=23/26 newline=0/26; question_mark_answer_to_original_interval_L18_19_attn_out_restore exact=22/26 newline=0/26; separator_to_original_interval_L18_19_attn_out_restore exact=21/26 newline=0/26; prompt_last_to_original_interval_L17_20_mlp_out_restore exact=21/26 newline=0/26; separator_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26; relation_tail_to_original_interval_L18_19_attn_out_restore exact=19/26 newline=0/26; prompt_last_to_original_interval_L17_20_attn_out_restore exact=18/26 newline=0/26
- Top necessity/remove: separator_remove_from_inline_interval_L18_19_mlp_out_restore exact=0/26 newline=26/26; prompt_last_remove_from_inline_L17_layer_input_restore exact=0/26 newline=14/26; prompt_last_remove_from_inline_interval_L18_19_mlp_out_restore exact=2/26 newline=17/26; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=3/26 newline=23/26; prompt_last_remove_from_inline_interval_L18_19_layer_out_restore exact=3/26 newline=13/26; prompt_last_remove_from_inline_L19_layer_out_restore exact=3/26 newline=13/26; prompt_last_remove_from_inline_L20_layer_input_restore exact=3/26 newline=13/26; prompt_last_remove_from_inline_interval_L17_20_layer_out_restore exact=3/26 newline=10/26

## glm4

- raw_cases: 320 / target_seen: 36 / cases_written: 36 / mode_rows: 4104
- layers: `[17, 18, 19, 20]` / positions: `['separator', 'answer_label', 'prompt_last', 'question_mark_answer', 'relation_tail']` / target_only: True
- filtered: `{'not_target': 284, 'position_missing': 0, 'position_len_mismatch': 36, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 10.41

### Baselines

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| original | 36 |  |  | baseline |  |  |  | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:8,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| inline | 36 |  |  | baseline |  |  |  | 27/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:7,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2 |

### Position Best Rows

| position | best sufficiency | exact | newline | rank | best necessity/remove | exact | newline | rank |
|---|---|---:|---:|---:|---|---:|---:|---:|
| separator | separator_to_original_interval_L18_19_attn_out_restore | 30/36 | 0/36 | 1.1 | separator_remove_from_inline_interval_L17_20_mlp_out_restore | 10/36 | 0/36 | 2.3 |
| answer_label |  |  |  |  |  |  |  |  |
| prompt_last | prompt_last_to_original_interval_L18_19_attn_out_restore | 30/36 | 0/36 | 1.0 | prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 16/36 | 0/36 | 3.3 |
| question_mark_answer | question_mark_answer_to_original_interval_L18_19_attn_out_restore | 34/36 | 0/36 | 1.0 | question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 2/36 | 0/36 | 3.4 |
| relation_tail | relation_tail_to_original_interval_L17_20_layer_out_restore | 29/36 | 0/36 | 1.5 | relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 0/36 | 0/36 | 3.5 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_to_original_interval_L18_19_attn_out_restore | 36 | separator | to_original | interval | L18_19 |  | attn_out | 30/36 | 34/36 | 0/36 | 1.1 | 88.524 | correct_prefix:34, word:1, explanation:1 |  v91:13,  v05:10,  v48:9,  v22:2,  c77:1,  No,:1 |
| separator_to_original_L17_layer_input_restore | 36 | separator | to_original | single_layer |  | 17 | layer_input | 28/36 | 30/36 | 0/36 | 1.6 | 88.558 | correct_prefix:30, explanation:4, word:2 |  v91:10,  v05:9,  v48:8,  v22:3,  No.\n\n:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L19_layer_out_restore | 36 | separator | to_original | single_layer |  | 19 | layer_out | 28/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v05:10,  v91:10,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L20_layer_input_restore | 36 | separator | to_original | single_layer |  | 20 | layer_input | 28/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v05:10,  v91:10,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L20_layer_out_restore | 36 | separator | to_original | single_layer |  | 20 | layer_out | 28/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:9,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L18_layer_out_restore | 36 | separator | to_original | single_layer |  | 18 | layer_out | 28/36 | 30/36 | 0/36 | 1.9 | 85.951 | correct_prefix:30, explanation:4, word:2 |  v05:9,  v91:9,  v48:9,  v22:2,  Yes.\n\n:2,  c12:2,  No.\n\n:1,  c59:1 |
| separator_to_original_L19_layer_input_restore | 36 | separator | to_original | single_layer |  | 19 | layer_input | 28/36 | 30/36 | 0/36 | 1.9 | 85.951 | correct_prefix:30, explanation:4, word:2 |  v05:9,  v91:9,  v48:9,  v22:2,  Yes.\n\n:2,  c12:2,  No.\n\n:1,  c59:1 |
| separator_to_original_interval_L18_19_layer_out_restore | 36 | separator | to_original | interval | L18_19 |  | layer_out | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_interval_L17_20_layer_out_restore | 36 | separator | to_original | interval | L17_20 |  | layer_out | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L17_layer_out_restore | 36 | separator | to_original | single_layer |  | 17 | layer_out | 26/36 | 30/36 | 0/36 | 1.9 | 77.990 | correct_prefix:30, explanation:3, word:3 |  v05:11,  v91:11,  v48:6,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_L18_layer_input_restore | 36 | separator | to_original | single_layer |  | 18 | layer_input | 26/36 | 30/36 | 0/36 | 1.9 | 77.990 | correct_prefix:30, explanation:3, word:3 |  v05:11,  v91:11,  v48:6,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| separator_to_original_interval_L18_19_mlp_out_restore | 36 | separator | to_original | interval | L18_19 |  | mlp_out | 20/36 | 22/36 | 0/36 | 1.9 | 91.161 | correct_prefix:22, word:12, explanation:2 |  v91:9,  v05:8,  v48:5,  c12:4,  c33:3,  c59:3,  No.\n\n:2,  c77:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |  c12:8,  c59:7,  c33:6,  v05:5,  v91:4,  c77:3,  v48:2,  Yes.\n\n:1 |
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | attn_out | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |  v91:7,  v05:6,  v48:5,  c12:4,  o43:2,  o82:2,  o17:2,  c77:1 |
| separator_remove_from_inline_interval_L18_19_mlp_out_restore | 36 | separator | remove_from_inline | interval | L18_19 |  | mlp_out | 29/36 | 31/36 | 0/36 | 1.3 | 32.868 | correct_prefix:31, word:3, explanation:2 |  v91:10,  v05:9,  v48:9,  v22:3,  c77:2,  c33:1,  Yes.\n\n:1,  Yes.\n:1 |
| separator_remove_from_inline_interval_L18_19_attn_out_restore | 36 | separator | remove_from_inline | interval | L18_19 |  | attn_out | 30/36 | 33/36 | 0/36 | 1.3 | 75.506 | correct_prefix:33, explanation:3 |  v05:11,  v91:11,  v48:9,  v22:2,  Yes.\n:2,  Yes.\n\n:1 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | layer_out | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L20_layer_out_restore | 36 | separator | remove_from_inline | single_layer |  | 20 | layer_out | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 36 | separator | remove_from_inline | interval | L18_19 |  | layer_out | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L17_layer_out_restore | 36 | separator | remove_from_inline | single_layer |  | 17 | layer_out | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L18_layer_input_restore | 36 | separator | remove_from_inline | single_layer |  | 18 | layer_input | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L19_layer_out_restore | 36 | separator | remove_from_inline | single_layer |  | 19 | layer_out | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L20_layer_input_restore | 36 | separator | remove_from_inline | single_layer |  | 20 | layer_input | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| separator_remove_from_inline_L17_layer_input_restore | 36 | separator | remove_from_inline | single_layer |  | 17 | layer_input | 31/36 | 34/36 | 0/36 | 1.2 | 70.291 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |

### answer_label

#### Best Sufficiency Restore

No rows.

#### Best Necessity Remove

No rows.

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_to_original_interval_L18_19_attn_out_restore | 36 | prompt_last | to_original | interval | L18_19 |  | attn_out | 30/36 | 35/36 | 0/36 | 1.0 | 78.103 | correct_prefix:35, word:1 |  v91:13,  v05:10,  v48:8,  v22:4,  c77:1 |
| prompt_last_to_original_L19_layer_out_restore | 36 | prompt_last | to_original | single_layer |  | 19 | layer_out | 26/36 | 28/36 | 0/36 | 1.6 | 75.367 | correct_prefix:28, word:6, explanation:2 |  v91:10,  v48:8,  v05:7,  c12:3,  No.\n\n:2,  v22:2,  c59:2,  c33:1 |
| prompt_last_to_original_L20_layer_input_restore | 36 | prompt_last | to_original | single_layer |  | 20 | layer_input | 26/36 | 28/36 | 0/36 | 1.6 | 75.367 | correct_prefix:28, word:6, explanation:2 |  v91:10,  v48:8,  v05:7,  c12:3,  No.\n\n:2,  v22:2,  c59:2,  c33:1 |
| prompt_last_to_original_interval_L18_19_layer_out_restore | 36 | prompt_last | to_original | interval | L18_19 |  | layer_out | 25/36 | 28/36 | 0/36 | 1.6 | 75.367 | correct_prefix:28, word:6, explanation:2 |  v91:11,  v05:7,  v48:7,  c12:3,  No.\n\n:2,  v22:2,  c59:2,  c33:1 |
| prompt_last_to_original_L17_layer_input_restore | 36 | prompt_last | to_original | single_layer |  | 17 | layer_input | 25/36 | 28/36 | 0/36 | 1.6 | 77.969 | correct_prefix:28, word:6, explanation:2 |  v91:10,  v05:8,  v48:6,  v22:4,  c12:3,  c59:2,  No.\n\n:1,  c77:1 |
| prompt_last_to_original_L18_layer_out_restore | 36 | prompt_last | to_original | single_layer |  | 18 | layer_out | 24/36 | 27/36 | 0/36 | 1.6 | 78.015 | correct_prefix:27, word:7, explanation:2 |  v91:11,  v05:9,  v48:5,  c12:3,  No.\n\n:2,  v22:2,  c59:2,  c33:1 |
| prompt_last_to_original_L19_layer_input_restore | 36 | prompt_last | to_original | single_layer |  | 19 | layer_input | 24/36 | 27/36 | 0/36 | 1.6 | 78.015 | correct_prefix:27, word:7, explanation:2 |  v91:11,  v05:9,  v48:5,  c12:3,  No.\n\n:2,  v22:2,  c59:2,  c33:1 |
| prompt_last_to_original_L17_layer_out_restore | 36 | prompt_last | to_original | single_layer |  | 17 | layer_out | 24/36 | 28/36 | 0/36 | 1.6 | 75.314 | correct_prefix:28, word:6, explanation:2 |  v91:10,  v05:8,  v48:6,  v22:3,  c12:3,  c59:3,  No.\n\n:2,  c77:1 |
| prompt_last_to_original_L18_layer_input_restore | 36 | prompt_last | to_original | single_layer |  | 18 | layer_input | 24/36 | 28/36 | 0/36 | 1.6 | 75.314 | correct_prefix:28, word:6, explanation:2 |  v91:10,  v05:8,  v48:6,  v22:3,  c12:3,  c59:3,  No.\n\n:2,  c77:1 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 36 | prompt_last | to_original | interval | L17_20 |  | layer_out | 24/36 | 27/36 | 0/36 | 1.6 | 75.378 | correct_prefix:27, word:7, explanation:2 |  v91:10,  v05:8,  v48:6,  c12:3,  c59:3,  No.\n\n:2,  v22:2,  c33:1 |
| prompt_last_to_original_L20_layer_out_restore | 36 | prompt_last | to_original | single_layer |  | 20 | layer_out | 24/36 | 27/36 | 0/36 | 1.6 | 75.378 | correct_prefix:27, word:7, explanation:2 |  v91:10,  v05:8,  v48:6,  c12:3,  c59:3,  No.\n\n:2,  v22:2,  c33:1 |
| prompt_last_to_original_interval_L18_19_mlp_out_restore | 36 | prompt_last | to_original | interval | L18_19 |  | mlp_out | 14/36 | 16/36 | 0/36 | 2.0 | 96.384 | word:18, correct_prefix:16, explanation:2 |  v91:7,  c12:7,  v05:6,  c59:5,  c77:3,  c33:3,  v48:3,  No.\n\n:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | 16/36 | 19/36 | 0/36 | 3.3 | 96.380 | correct_prefix:19, word:16, explanation:1 |  v91:6,  v05:5,  v48:5,  c33:4,  c12:4,  c59:3,  c77:2,  o43:2 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | 19/36 | 21/36 | 0/36 | 1.8 | 99.000 | correct_prefix:21, word:14, explanation:1 |  v05:7,  v91:7,  v48:6,  c12:5,  c33:4,  c59:3,  c77:2,  True.\n\n:1 |
| prompt_last_remove_from_inline_interval_L18_19_mlp_out_restore | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | mlp_out | 28/36 | 29/36 | 0/36 | 1.6 | 56.825 | correct_prefix:29, explanation:7 |  v05:10,  v91:9,  v48:7,  Yes.\n:3,  v22:3,  Yes.\n\n:3,  c33:1 |
| prompt_last_remove_from_inline_interval_L18_19_attn_out_restore | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | attn_out | 29/36 | 33/36 | 0/36 | 1.4 | 83.318 | correct_prefix:33, explanation:3 |  v91:12,  v05:10,  v48:9,  v22:2,  Yes.\n\n:2,  Yes,:1 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | 30/36 | 33/36 | 0/36 | 1.2 | 64.969 | correct_prefix:33, word:2, explanation:1 |  v91:12,  v48:10,  v05:9,  v22:2,  o17:1,  c77:1,  Yes.\n\n:1 |
| prompt_last_remove_from_inline_L18_layer_out_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_out | 30/36 | 32/36 | 0/36 | 1.2 | 70.254 | correct_prefix:32, word:2, explanation:2 |  v91:10,  v48:10,  v05:9,  v22:3,  c77:2,  Yes.\n\n:2 |
| prompt_last_remove_from_inline_L19_layer_input_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_input | 30/36 | 32/36 | 0/36 | 1.2 | 70.254 | correct_prefix:32, word:2, explanation:2 |  v91:10,  v48:10,  v05:9,  v22:3,  c77:2,  Yes.\n\n:2 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | 30/36 | 32/36 | 0/36 | 1.2 | 72.832 | correct_prefix:32, word:2, explanation:2 |  v91:10,  v48:10,  v05:9,  v22:3,  c77:2,  Yes.\n\n:2 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | 30/36 | 33/36 | 0/36 | 1.2 | 75.510 | correct_prefix:33, explanation:2, word:1 |  v91:11,  v05:10,  v48:9,  v22:3,  Yes.\n\n:2,  c77:1 |
| prompt_last_remove_from_inline_L18_layer_input_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_input | 30/36 | 33/36 | 0/36 | 1.2 | 75.510 | correct_prefix:33, explanation:2, word:1 |  v91:11,  v05:10,  v48:9,  v22:3,  Yes.\n\n:2,  c77:1 |
| prompt_last_remove_from_inline_L19_layer_out_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_out | 30/36 | 32/36 | 0/36 | 1.2 | 72.832 | correct_prefix:32, word:2, explanation:2 |  v05:10,  v91:10,  v48:9,  v22:3,  c77:2,  Yes.\n\n:2 |
| prompt_last_remove_from_inline_L20_layer_input_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 20 | layer_input | 30/36 | 32/36 | 0/36 | 1.2 | 72.832 | correct_prefix:32, word:2, explanation:2 |  v05:10,  v91:10,  v48:9,  v22:3,  c77:2,  Yes.\n\n:2 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_to_original_interval_L18_19_attn_out_restore | 36 | question_mark_answer | to_original | interval | L18_19 |  | attn_out | 34/36 | 35/36 | 0/36 | 1.0 | 96.392 | correct_prefix:35, word:1 |  v91:11,  v48:11,  v05:10,  v22:3,  c77:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 36 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 36 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L17_layer_input_restore | 36 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L17_layer_out_restore | 36 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L18_layer_input_restore | 36 | question_mark_answer | to_original | single_layer |  | 18 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L19_layer_out_restore | 36 | question_mark_answer | to_original | single_layer |  | 19 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L20_layer_input_restore | 36 | question_mark_answer | to_original | single_layer |  | 20 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L20_layer_out_restore | 36 | question_mark_answer | to_original | single_layer |  | 20 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L18_layer_out_restore | 36 | question_mark_answer | to_original | single_layer |  | 18 | layer_out | 27/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:11,  v05:9,  v48:7,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_L19_layer_input_restore | 36 | question_mark_answer | to_original | single_layer |  | 19 | layer_input | 27/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:11,  v05:9,  v48:7,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| question_mark_answer_to_original_interval_L18_19_mlp_out_restore | 36 | question_mark_answer | to_original | interval | L18_19 |  | mlp_out | 8/36 | 8/36 | 0/36 | 2.3 | 93.765 | word:27, correct_prefix:8, explanation:1 |  c12:10,  c33:6,  c59:6,  c77:5,  v91:5,  v05:2,  v48:1,  Yes.\n\n:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | 2/36 | 2/36 | 0/36 | 3.4 | 77.670 | word:34, correct_prefix:2 |  c12:11,  c59:9,  c33:8,  c77:3,  True\n:2,  v48:1,  v05:1,  True.\n\n:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | 14/36 | 17/36 | 0/36 | 9.0 | 96.366 | correct_prefix:17, word:17, explanation:2 |  v91:6,  c12:6,  v05:5,  v48:4,  o17:3,  o43:2,  c59:2,  c77:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_mlp_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L18_19 |  | mlp_out | 26/36 | 27/36 | 0/36 | 1.3 | 24.839 | correct_prefix:27, word:5, explanation:4 |  v05:10,  v91:9,  v48:6,  c12:3,  Yes.\n\n:3,  v22:2,  Yes.\n:1,  c33:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | 27/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:10,  v48:6,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L20_layer_out_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 20 | layer_out | 27/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:6,  v22:4,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L19_layer_out_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 19 | layer_out | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:6,  v22:4,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L20_layer_input_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 20 | layer_input | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:6,  v22:4,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L18_layer_input_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_input | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| question_mark_answer_remove_from_inline_L18_layer_out_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_to_original_interval_L17_20_layer_out_restore | 36 | relation_tail | to_original | interval | L17_20 |  | layer_out | 29/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v05:10,  v91:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_interval_L18_19_layer_out_restore | 36 | relation_tail | to_original | interval | L18_19 |  | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L17_layer_input_restore | 36 | relation_tail | to_original | single_layer |  | 17 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  c77:2,  v22:2,  Yes.\n\n:2,  c12:2,  Yes,:1 |
| relation_tail_to_original_L17_layer_out_restore | 36 | relation_tail | to_original | single_layer |  | 17 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L18_layer_input_restore | 36 | relation_tail | to_original | single_layer |  | 18 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L18_layer_out_restore | 36 | relation_tail | to_original | single_layer |  | 18 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L19_layer_input_restore | 36 | relation_tail | to_original | single_layer |  | 19 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L19_layer_out_restore | 36 | relation_tail | to_original | single_layer |  | 19 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L20_layer_input_restore | 36 | relation_tail | to_original | single_layer |  | 20 | layer_input | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_L20_layer_out_restore | 36 | relation_tail | to_original | single_layer |  | 20 | layer_out | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:8,  Yes.\n\n:3,  c77:2,  v22:2,  c12:2 |
| relation_tail_to_original_interval_L18_19_attn_out_restore | 36 | relation_tail | to_original | interval | L18_19 |  | attn_out | 23/36 | 34/36 | 0/36 | 1.1 | 93.789 | correct_prefix:34, word:2 |  v05:13,  v91:10,  v48:8,  v22:3,  c12:1,  c77:1 |
| relation_tail_to_original_interval_L18_19_mlp_out_restore | 36 | relation_tail | to_original | interval | L18_19 |  | mlp_out | 7/36 | 9/36 | 0/36 | 2.4 | 93.755 | word:26, correct_prefix:9, explanation:1 |  c12:10,  c59:7,  c33:6,  c77:5,  v91:5,  v48:1,  v05:1,  Yes.\n\n:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | 0/36 | 0/36 | 0/36 | 3.5 | 83.009 | word:36 |  c12:11,  c33:10,  c59:9,  c77:4,  True\n:1,  True.\n\n:1 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | 12/36 | 15/36 | 0/36 | 8.8 | 93.736 | word:21, correct_prefix:15 |  v91:6,  v05:5,  c12:5,  c33:3,  v48:3,  o17:3,  c77:2,  o43:2 |
| relation_tail_remove_from_inline_interval_L18_19_attn_out_restore | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | attn_out | 24/36 | 34/36 | 0/36 | 1.1 | 83.362 | correct_prefix:34, word:1, explanation:1 |  v05:12,  v91:10,  v48:10,  v22:2,  c77:1,  Yes.\n:1 |
| relation_tail_remove_from_inline_interval_L18_19_mlp_out_restore | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | mlp_out | 26/36 | 28/36 | 0/36 | 1.4 | 35.481 | correct_prefix:28, explanation:4, word:4 |  v05:11,  v91:8,  v48:7,  c12:3,  Yes.\n:2,  v22:2,  Yes.\n\n:2,  c33:1 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:11,  v91:10,  v48:6,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L19_layer_out_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 19 | layer_out | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:11,  v91:10,  v48:6,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L20_layer_input_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 20 | layer_input | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:11,  v91:10,  v48:6,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:11,  v91:10,  v48:7,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L17_layer_input_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_input | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L18_layer_input_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_input | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| relation_tail_remove_from_inline_L18_layer_out_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_out | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v05:10,  v91:10,  v48:7,  v22:3,  c77:2,  c12:2,  c59:1,  Yes.\n\n:1 |

### Global Top Notes

- Top sufficiency: question_mark_answer_to_original_interval_L18_19_attn_out_restore exact=34/36 newline=0/36; prompt_last_to_original_interval_L18_19_attn_out_restore exact=30/36 newline=0/36; separator_to_original_interval_L18_19_attn_out_restore exact=30/36 newline=0/36; relation_tail_to_original_interval_L17_20_layer_out_restore exact=29/36 newline=0/36; question_mark_answer_to_original_interval_L17_20_layer_out_restore exact=28/36 newline=0/36; question_mark_answer_to_original_interval_L18_19_layer_out_restore exact=28/36 newline=0/36; question_mark_answer_to_original_L17_layer_input_restore exact=28/36 newline=0/36; question_mark_answer_to_original_L17_layer_out_restore exact=28/36 newline=0/36
- Top necessity/remove: relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore exact=0/36 newline=0/36; question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore exact=2/36 newline=0/36; separator_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36; relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=12/36 newline=0/36; question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=14/36 newline=0/36; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=16/36 newline=0/36; separator_remove_from_inline_interval_L17_20_attn_out_restore exact=17/36 newline=0/36; prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore exact=19/36 newline=0/36

## deepseek7b

- raw_cases: 320 / target_seen: 48 / cases_written: 48 / mode_rows: 5472
- layers: `[17, 18, 19, 20]` / positions: `['separator', 'answer_label', 'prompt_last', 'question_mark_answer', 'relation_tail']` / target_only: True
- filtered: `{'not_target': 88, 'position_missing': 0, 'position_len_mismatch': 48, 'empty_patch': 0, 'case_cap': 1}` / total_time_min: 13.09

### Baselines

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| original | 48 |  |  | baseline |  |  |  | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:26,  ?\n\nI think:8,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| inline | 48 |  |  | baseline |  |  |  | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |

### Position Best Rows

| position | best sufficiency | exact | newline | rank | best necessity/remove | exact | newline | rank |
|---|---|---:|---:|---:|---|---:|---:|---:|
| separator | separator_to_original_L17_layer_input_restore | 46/48 | 0/48 | 1.0 | separator_remove_from_inline_interval_L17_20_attn_out_restore | 8/48 | 0/48 | 2.4 |
| answer_label |  |  |  |  |  |  |  |  |
| prompt_last | prompt_last_to_original_L17_layer_out_restore | 46/48 | 0/48 | 1.1 | prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 5/48 | 2/48 | 2.7 |
| question_mark_answer | question_mark_answer_to_original_interval_L18_19_layer_out_restore | 45/48 | 0/48 | 1.0 | question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 5/48 | 0/48 | 2.6 |
| relation_tail | relation_tail_to_original_interval_L18_19_layer_out_restore | 45/48 | 0/48 | 1.0 | relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 1/48 | 0/48 | 2.7 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_to_original_L17_layer_input_restore | 48 | separator | to_original | single_layer |  | 17 | layer_input | 46/48 | 46/48 | 0/48 | 1.0 | 1.835 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  64:2 |
| separator_to_original_L18_layer_out_restore | 48 | separator | to_original | single_layer |  | 18 | layer_out | 46/48 | 46/48 | 0/48 | 1.0 | 2.216 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| separator_to_original_L19_layer_input_restore | 48 | separator | to_original | single_layer |  | 19 | layer_input | 46/48 | 46/48 | 0/48 | 1.0 | 2.216 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| separator_to_original_L17_layer_out_restore | 48 | separator | to_original | single_layer |  | 17 | layer_out | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:1,  64:1 |
| separator_to_original_L18_layer_input_restore | 48 | separator | to_original | single_layer |  | 18 | layer_input | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:1,  64:1 |
| separator_to_original_L20_layer_out_restore | 48 | separator | to_original | single_layer |  | 20 | layer_out | 45/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| separator_to_original_interval_L18_19_layer_out_restore | 48 | separator | to_original | interval | L18_19 |  | layer_out | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| separator_to_original_L19_layer_out_restore | 48 | separator | to_original | single_layer |  | 19 | layer_out | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| separator_to_original_L20_layer_input_restore | 48 | separator | to_original | single_layer |  | 20 | layer_input | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| separator_to_original_interval_L17_20_layer_out_restore | 48 | separator | to_original | interval | L17_20 |  | layer_out | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |  v48:14,  v05:13,  v22:11,  v91:7,  22:2,  05:1 |
| separator_to_original_interval_L17_20_mlp_out_restore | 48 | separator | to_original | interval | L17_20 |  | mlp_out | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |  v05:13,  v22:10,  v48:9,  v91:5,  ?\n\nTo solve:4,  c33:2,  c77:2,  o71:1 |
| separator_to_original_interval_L18_19_mlp_out_restore | 48 | separator | to_original | interval | L18_19 |  | mlp_out | 20/48 | 21/48 | 7/48 | 2.2 | 0.677 | correct_prefix:21, word:14, newline:7, space:6 |  v48:8,  v05:7,  c77:5,  v91:4,  c59:4,  ?\n\nTo solve:4,  ?\n\nI think:3,  c12:3 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | attn_out | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |  c77:14,  c33:9,  c59:6,  c12:5,  v48:3,  v05:3,  v22:2,  o43:2 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | layer_out | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |  ?\n\nTo solve:35,  v48:7,  v05:4,  c77:1,  v22:1 |
| separator_remove_from_inline_L20_layer_out_restore | 48 | separator | remove_from_inline | single_layer |  | 20 | layer_out | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |  ?\n\nTo solve:33,  v48:7,  v05:4,  ?\n\nI think:2,  c77:1,  v22:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 48 | separator | remove_from_inline | interval | L18_19 |  | layer_out | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| separator_remove_from_inline_L19_layer_out_restore | 48 | separator | remove_from_inline | single_layer |  | 19 | layer_out | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| separator_remove_from_inline_L20_layer_input_restore | 48 | separator | remove_from_inline | single_layer |  | 20 | layer_input | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| separator_remove_from_inline_L18_layer_out_restore | 48 | separator | remove_from_inline | single_layer |  | 18 | layer_out | 14/48 | 15/48 | 22/48 | 2.8 | -0.395 | newline:22, correct_prefix:15, space:10, word:1 |  ?\n\nTo solve:21,  v48:7,  22:6,  v05:4,  48:4,  v91:2,  c77:1,  05:1 |
| separator_remove_from_inline_L19_layer_input_restore | 48 | separator | remove_from_inline | single_layer |  | 19 | layer_input | 14/48 | 15/48 | 22/48 | 2.8 | -0.395 | newline:22, correct_prefix:15, space:10, word:1 |  ?\n\nTo solve:21,  v48:7,  22:6,  v05:4,  48:4,  v91:2,  c77:1,  05:1 |
| separator_remove_from_inline_L17_layer_out_restore | 48 | separator | remove_from_inline | single_layer |  | 17 | layer_out | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |  ?\n\nTo solve:16,  22:8,  v48:8,  v05:4,  05:3,  48:3,  v91:2,  91:1 |
| separator_remove_from_inline_L18_layer_input_restore | 48 | separator | remove_from_inline | single_layer |  | 18 | layer_input | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |  ?\n\nTo solve:16,  22:8,  v48:8,  v05:4,  05:3,  48:3,  v91:2,  91:1 |
| separator_remove_from_inline_L17_layer_input_restore | 48 | separator | remove_from_inline | single_layer |  | 17 | layer_input | 15/48 | 16/48 | 15/48 | 2.4 | 0.103 | space:17, correct_prefix:16, newline:15 |  ?\n\nTo solve:11,  22:9,  v48:7,  48:5,  v05:4,  05:4,  v91:3,  c77:1 |
| separator_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |  ?\n\nTo solve:19,  v05:12,  v48:6,  v22:4,  v91:4,  ?\n\nI think:3 |

### answer_label

#### Best Sufficiency Restore

No rows.

#### Best Necessity Remove

No rows.

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_to_original_L17_layer_out_restore | 48 | prompt_last | to_original | single_layer |  | 17 | layer_out | 46/48 | 46/48 | 0/48 | 1.1 | 1.667 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  64:2 |
| prompt_last_to_original_L18_layer_input_restore | 48 | prompt_last | to_original | single_layer |  | 18 | layer_input | 46/48 | 46/48 | 0/48 | 1.1 | 1.667 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  64:2 |
| prompt_last_to_original_interval_L18_19_layer_out_restore | 48 | prompt_last | to_original | interval | L18_19 |  | layer_out | 45/48 | 45/48 | 0/48 | 1.1 | 1.868 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  05:1,  22:1,  64:1 |
| prompt_last_to_original_L19_layer_out_restore | 48 | prompt_last | to_original | single_layer |  | 19 | layer_out | 45/48 | 45/48 | 0/48 | 1.1 | 1.868 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  05:1,  22:1,  64:1 |
| prompt_last_to_original_L20_layer_input_restore | 48 | prompt_last | to_original | single_layer |  | 20 | layer_input | 45/48 | 45/48 | 0/48 | 1.1 | 1.868 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  05:1,  22:1,  64:1 |
| prompt_last_to_original_L20_layer_out_restore | 48 | prompt_last | to_original | single_layer |  | 20 | layer_out | 45/48 | 45/48 | 0/48 | 1.1 | 2.102 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| prompt_last_to_original_L18_layer_out_restore | 48 | prompt_last | to_original | single_layer |  | 18 | layer_out | 45/48 | 44/48 | 1/48 | 1.0 | 1.590 | correct_prefix:44, space:3, newline:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| prompt_last_to_original_L19_layer_input_restore | 48 | prompt_last | to_original | single_layer |  | 19 | layer_input | 45/48 | 44/48 | 1/48 | 1.0 | 1.590 | correct_prefix:44, space:3, newline:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| prompt_last_to_original_L17_layer_input_restore | 48 | prompt_last | to_original | single_layer |  | 17 | layer_input | 44/48 | 43/48 | 3/48 | 1.2 | 1.422 | correct_prefix:43, newline:3, space:2 |  v48:16,  v05:12,  v22:11,  v91:5,  ?\n\nTo solve:2,  64:2 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 48 | prompt_last | to_original | interval | L17_20 |  | layer_out | 43/48 | 45/48 | 0/48 | 1.1 | 2.102 | correct_prefix:45, space:3 |  v48:14,  v05:13,  v22:11,  v91:7,  22:2,  05:1 |
| prompt_last_to_original_interval_L17_20_mlp_out_restore | 48 | prompt_last | to_original | interval | L17_20 |  | mlp_out | 31/48 | 32/48 | 14/48 | 1.6 | 0.570 | correct_prefix:32, newline:14, word:1, space:1 |  ?\n\nTo solve:12,  v48:12,  v22:8,  v05:8,  v91:4,  ?\n\nI think:2,  c33:1,  71:1 |
| prompt_last_to_original_interval_L18_19_mlp_out_restore | 48 | prompt_last | to_original | interval | L18_19 |  | mlp_out | 19/48 | 20/48 | 10/48 | 2.3 | 0.327 | correct_prefix:20, newline:10, space:10, word:8 |  ?\n\nTo solve:9,  v48:8,  v05:7,  64:5,  c77:5,  v91:4,  22:2,  48:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | 5/48 | 7/48 | 2/48 | 2.7 | 0.118 | word:39, correct_prefix:7, newline:2 |  c77:15,  c33:9,  c59:5,  c12:5,  v05:3,  ?\n\nTo solve:2,  o43:2,  o58:2 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | 12/48 | 14/48 | 29/48 | 3.2 | -0.624 | newline:29, correct_prefix:14, space:3, word:2 |  ?\n\nTo solve:28,  v48:6,  v05:6,  c77:2,  48:2,  05:1,  ?\n\nI think:1,  v91:1 |
| prompt_last_remove_from_inline_L20_layer_out_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 20 | layer_out | 14/48 | 14/48 | 29/48 | 3.2 | -0.624 | newline:29, correct_prefix:14, space:3, word:2 |  ?\n\nTo solve:27,  v48:8,  v05:4,  c77:2,  48:2,  ?\n\nI think:2,  05:1,  v91:1 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 48 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | 20/48 | 19/48 | 23/48 | 2.2 | -0.073 | newline:23, correct_prefix:19, space:4, word:2 |  ?\n\nTo solve:20,  v48:9,  v05:7,  v91:3,  22:2,  c77:2,  ?\n\nI think:2,  05:1 |
| prompt_last_remove_from_inline_L19_layer_out_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_out | 20/48 | 19/48 | 23/48 | 2.2 | -0.073 | newline:23, correct_prefix:19, space:4, word:2 |  ?\n\nTo solve:20,  v48:9,  v05:7,  v91:3,  22:2,  c77:2,  ?\n\nI think:2,  05:1 |
| prompt_last_remove_from_inline_L20_layer_input_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 20 | layer_input | 20/48 | 19/48 | 23/48 | 2.2 | -0.073 | newline:23, correct_prefix:19, space:4, word:2 |  ?\n\nTo solve:20,  v48:9,  v05:7,  v91:3,  22:2,  c77:2,  ?\n\nI think:2,  05:1 |
| prompt_last_remove_from_inline_L18_layer_out_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_out | 21/48 | 23/48 | 15/48 | 2.1 | 0.281 | correct_prefix:23, newline:15, space:8, word:2 |  ?\n\nTo solve:12,  v48:9,  v05:8,  22:7,  c77:3,  v91:3,  48:2,  ?\n\nI think:2 |
| prompt_last_remove_from_inline_L19_layer_input_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 19 | layer_input | 21/48 | 23/48 | 15/48 | 2.1 | 0.281 | correct_prefix:23, newline:15, space:8, word:2 |  ?\n\nTo solve:12,  v48:9,  v05:8,  22:7,  c77:3,  v91:3,  48:2,  ?\n\nI think:2 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | 23/48 | 23/48 | 15/48 | 2.0 | 0.337 | correct_prefix:23, newline:15, space:10 |  ?\n\nTo solve:11,  22:9,  v48:9,  v05:9,  v91:4,  48:2,  05:1,  c77:1 |
| prompt_last_remove_from_inline_L18_layer_input_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 18 | layer_input | 23/48 | 23/48 | 15/48 | 2.0 | 0.337 | correct_prefix:23, newline:15, space:10 |  ?\n\nTo solve:11,  22:9,  v48:9,  v05:9,  v91:4,  48:2,  05:1,  c77:1 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | 25/48 | 27/48 | 21/48 | 1.7 | 0.276 | correct_prefix:27, newline:21 |  ?\n\nTo solve:17,  v05:13,  v48:7,  v22:5,  v91:4,  ?\n\nI think:2 |
| prompt_last_remove_from_inline_L17_layer_input_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_input | 25/48 | 26/48 | 9/48 | 1.8 | 0.616 | correct_prefix:26, space:11, newline:9, word:2 |  v48:10,  v05:10,  22:9,  ?\n\nTo solve:8,  v91:4,  48:2,  c77:2,  05:1 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 48 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L17_layer_input_restore | 48 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L17_layer_out_restore | 48 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L18_layer_input_restore | 48 | question_mark_answer | to_original | single_layer |  | 18 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L18_layer_out_restore | 48 | question_mark_answer | to_original | single_layer |  | 18 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L19_layer_input_restore | 48 | question_mark_answer | to_original | single_layer |  | 19 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L19_layer_out_restore | 48 | question_mark_answer | to_original | single_layer |  | 19 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L20_layer_input_restore | 48 | question_mark_answer | to_original | single_layer |  | 20 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_L20_layer_out_restore | 48 | question_mark_answer | to_original | single_layer |  | 20 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 48 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | 42/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v05:14,  v48:13,  v22:11,  v91:7,  22:2,  05:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_restore | 48 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | 29/48 | 34/48 | 9/48 | 1.5 | 0.993 | correct_prefix:34, newline:9, word:5 |  v05:12,  v22:9,  v48:8,  ?\n\nTo solve:7,  v91:5,  ?\n\nI think:2,  c33:2,  c77:1 |
| question_mark_answer_to_original_interval_L18_19_mlp_out_restore | 48 | question_mark_answer | to_original | interval | L18_19 |  | mlp_out | 20/48 | 21/48 | 5/48 | 2.0 | 0.939 | correct_prefix:21, word:14, space:8, newline:5 |  v48:8,  v05:7,  c77:5,  ?\n\nI think:4,  v91:4,  c59:4,  64:3,  c12:3 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | 5/48 | 5/48 | 0/48 | 2.6 | 0.469 | word:43, correct_prefix:5 |  c77:16,  c33:9,  c59:7,  c12:5,  v05:3,  o43:2,  o58:2,  v48:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | 11/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:6,  v05:5,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:33,  v48:7,  v05:4,  c77:1,  48:1,  ?\n\nI think:1,  v22:1 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L18_layer_input_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L18_layer_out_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 18 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L19_layer_input_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 19 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L19_layer_out_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 19 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L20_layer_input_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 20 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_L20_layer_out_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 20 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | 12/48 | 16/48 | 31/48 | 2.6 | -0.728 | newline:31, correct_prefix:16, word:1 |  ?\n\nTo solve:18,  ?\n\nI think:13,  v05:8,  v48:3,  v91:3,  v22:2,  o71:1 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_to_original_interval_L18_19_layer_out_restore | 48 | relation_tail | to_original | interval | L18_19 |  | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L17_layer_input_restore | 48 | relation_tail | to_original | single_layer |  | 17 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L17_layer_out_restore | 48 | relation_tail | to_original | single_layer |  | 17 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L18_layer_input_restore | 48 | relation_tail | to_original | single_layer |  | 18 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L18_layer_out_restore | 48 | relation_tail | to_original | single_layer |  | 18 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L19_layer_input_restore | 48 | relation_tail | to_original | single_layer |  | 19 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L19_layer_out_restore | 48 | relation_tail | to_original | single_layer |  | 19 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L20_layer_input_restore | 48 | relation_tail | to_original | single_layer |  | 20 | layer_input | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_L20_layer_out_restore | 48 | relation_tail | to_original | single_layer |  | 20 | layer_out | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| relation_tail_to_original_interval_L17_20_layer_out_restore | 48 | relation_tail | to_original | interval | L17_20 |  | layer_out | 31/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v05:15,  v48:14,  v22:10,  v91:6,  22:2,  05:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_restore | 48 | relation_tail | to_original | interval | L17_20 |  | mlp_out | 21/48 | 25/48 | 12/48 | 2.1 | 0.346 | correct_prefix:25, newline:12, word:9, space:2 |  ?\n\nTo solve:10,  v22:7,  v48:7,  v05:6,  v91:4,  c33:3,  o71:2,  ?\n\nI think:2 |
| relation_tail_to_original_interval_L18_19_mlp_out_restore | 48 | relation_tail | to_original | interval | L18_19 |  | mlp_out | 16/48 | 19/48 | 5/48 | 2.3 | 0.854 | correct_prefix:19, word:13, space:11, newline:5 |  v05:7,  ?\n\nI think:5,  v48:5,  c77:5,  64:4,  v91:4,  c59:4,  48:3 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | 1/48 | 4/48 | 0/48 | 2.7 | 0.404 | word:44, correct_prefix:4 |  c77:16,  c33:9,  c59:8,  c12:5,  o58:2,  v48:2,  o29:1,  o06:1 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | 4/48 | 7/48 | 39/48 | 4.6 | -1.646 | newline:39, correct_prefix:7, word:2 |  ?\n\nTo solve:25,  ?\n\nI think:14,  v05:4,  v91:2,  v22:1,  o71:1,  o17:1 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | 7/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v05:6,  v22:3,  v48:2,  c77:1,  v91:1,  48:1 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 48 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L17_layer_input_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L18_layer_input_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L18_layer_out_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 18 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L19_layer_input_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 19 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L19_layer_out_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 19 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L20_layer_input_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 20 | layer_input | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| relation_tail_remove_from_inline_L20_layer_out_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 20 | layer_out | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:34,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |

### Global Top Notes

- Top sufficiency: separator_to_original_L17_layer_input_restore exact=46/48 newline=0/48; separator_to_original_L18_layer_out_restore exact=46/48 newline=0/48; separator_to_original_L19_layer_input_restore exact=46/48 newline=0/48; separator_to_original_L17_layer_out_restore exact=46/48 newline=0/48; separator_to_original_L18_layer_input_restore exact=46/48 newline=0/48; prompt_last_to_original_L17_layer_out_restore exact=46/48 newline=0/48; prompt_last_to_original_L18_layer_input_restore exact=46/48 newline=0/48; question_mark_answer_to_original_interval_L18_19_layer_out_restore exact=45/48 newline=0/48
- Top necessity/remove: relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=1/48 newline=0/48; relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore exact=4/48 newline=39/48; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=2/48; question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=0/48; relation_tail_remove_from_inline_interval_L17_20_layer_out_restore exact=7/48 newline=34/48; separator_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48; question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore exact=11/48 newline=34/48; separator_remove_from_inline_interval_L17_20_layer_out_restore exact=12/48 newline=35/48
