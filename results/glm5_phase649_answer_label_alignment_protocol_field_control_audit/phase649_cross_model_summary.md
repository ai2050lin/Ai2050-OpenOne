# Phase 649 Cross-Model Summary

目标：修复 Phase648 的 answer_label 对齐缺口，并对最强 protocol field 候选加入 restore/random/reverse controls。

## qwen3

- raw_cases: 320 / target_seen: 26 / cases_written: 26 / mode_rows: 5876
- positions: `['answer_word', 'colon', 'answer_colon', 'answer_label_aligned', 'separator', 'prompt_last', 'question_mark_answer', 'relation_tail']`
- filtered: `{'not_target': 294, 'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 12.05

### Baselines

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| original | 26 |  |  | baseline |  |  |  |  | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| inline | 26 |  |  | baseline |  |  |  |  | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |

### Position Control Overview

| position | best restore sufficiency | best restore necessity/remove | best random | best reverse |
|---|---|---|---|---|
| answer_word | answer_word_to_original_interval_L17_20_attn_out_restore exact=25/26 newline=0/26 rank=1.0 | answer_word_remove_from_inline_interval_L17_20_mlp_out_restore exact=0/26 newline=15/26 rank=5.5 | answer_word_to_original_interval_L18_19_layer_out_random exact=17/26 newline=2/26 rank=1.3 | answer_word_to_original_interval_L17_20_layer_out_reverse exact=19/26 newline=0/26 rank=1.2 |
| colon | colon_to_original_interval_L17_20_mlp_out_restore exact=21/26 newline=0/26 rank=1.2 | colon_remove_from_inline_L17_layer_input_restore exact=0/26 newline=14/26 rank=4.5 | colon_to_original_interval_L17_20_mlp_out_random exact=18/26 newline=0/26 rank=1.7 | colon_to_original_interval_L17_20_mlp_out_reverse exact=22/26 newline=0/26 rank=1.1 |
| answer_colon | answer_colon_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2 | answer_colon_remove_from_inline_interval_L17_20_attn_out_restore exact=7/26 newline=7/26 rank=3.6 | answer_colon_to_original_interval_L17_20_mlp_out_random exact=17/26 newline=0/26 rank=1.5 | answer_colon_to_original_interval_L17_20_layer_out_reverse exact=18/26 newline=0/26 rank=1.3 |
| answer_label_aligned | answer_label_aligned_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2 | answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore exact=7/26 newline=7/26 rank=3.6 | answer_label_aligned_to_original_interval_L17_20_mlp_out_random exact=18/26 newline=0/26 rank=1.3 | answer_label_aligned_to_original_interval_L17_20_layer_out_reverse exact=18/26 newline=0/26 rank=1.3 |
| separator | separator_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2 | separator_remove_from_inline_interval_L17_20_attn_out_restore exact=7/26 newline=7/26 rank=3.6 | separator_to_original_interval_L17_20_mlp_out_random exact=19/26 newline=0/26 rank=1.3 | separator_to_original_interval_L17_20_layer_out_reverse exact=18/26 newline=0/26 rank=1.3 |
| prompt_last | prompt_last_to_original_interval_L17_20_mlp_out_restore exact=21/26 newline=0/26 rank=1.2 | prompt_last_remove_from_inline_L17_layer_input_restore exact=0/26 newline=14/26 rank=4.5 | prompt_last_to_original_interval_L17_20_mlp_out_random exact=17/26 newline=0/26 rank=1.4 | prompt_last_to_original_interval_L17_20_mlp_out_reverse exact=22/26 newline=0/26 rank=1.1 |
| question_mark_answer | question_mark_answer_to_original_interval_L17_20_attn_out_restore exact=23/26 newline=0/26 rank=1.0 | question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=12/26 newline=9/26 rank=3.2 | question_mark_answer_remove_from_inline_interval_L17_20_attn_out_random exact=16/26 newline=2/26 rank=1.6 | question_mark_answer_remove_from_inline_interval_L17_20_attn_out_reverse exact=17/26 newline=0/26 rank=1.0 |
| relation_tail | relation_tail_to_original_interval_L17_20_attn_out_restore exact=13/26 newline=0/26 rank=1.1 | relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=9/26 newline=12/26 rank=4.9 | relation_tail_remove_from_inline_interval_L17_20_mlp_out_random exact=13/26 newline=0/26 rank=1.5 | relation_tail_to_original_interval_L17_20_layer_out_reverse exact=14/26 newline=0/26 rank=1.5 |

### answer_word

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L17_20_attn_out_restore | 26 | answer_word | to_original | interval | L17_20 |  | attn_out | restore | 25/26 | 26/26 | 0/26 | 1.0 | 2.144 | correct_prefix:26 |
| answer_word_to_original_interval_L17_20_mlp_out_restore | 26 | answer_word | to_original | interval | L17_20 |  | mlp_out | restore | 16/26 | 18/26 | 6/26 | 1.5 | 0.769 | correct_prefix:18, newline:6, space:2 |
| answer_word_to_original_interval_L18_19_layer_out_restore | 26 | answer_word | to_original | interval | L18_19 |  | layer_out | restore | 9/26 | 10/26 | 14/26 | 2.0 | -0.226 | newline:14, correct_prefix:10, space:2 |
| answer_word_to_original_L17_layer_out_restore | 26 | answer_word | to_original | single_layer |  | 17 | layer_out | restore | 6/26 | 8/26 | 16/26 | 2.2 | -0.433 | newline:16, correct_prefix:8, space:2 |
| answer_word_to_original_interval_L17_20_layer_out_restore | 26 | answer_word | to_original | interval | L17_20 |  | layer_out | restore | 5/26 | 5/26 | 19/26 | 2.3 | -0.413 | newline:19, correct_prefix:5, space:2 |
| answer_word_to_original_L17_layer_input_restore | 26 | answer_word | to_original | single_layer |  | 17 | layer_input | restore | 0/26 | 0/26 | 24/26 | 3.8 | -1.409 | newline:24, space:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 0/26 | 0/26 | 15/26 | 5.5 | -1.952 | newline:15, space:11 |
| answer_word_remove_from_inline_L17_layer_out_restore | 26 | answer_word | remove_from_inline | single_layer |  | 17 | layer_out | restore | 4/26 | 5/26 | 3/26 | 4.3 | -0.688 | space:18, correct_prefix:5, newline:3 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_restore | 26 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | restore | 4/26 | 4/26 | 0/26 | 3.5 | -0.091 | space:22, correct_prefix:4 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_restore | 26 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | restore | 5/26 | 6/26 | 9/26 | 3.4 | -0.822 | space:11, newline:9, correct_prefix:6 |
| answer_word_remove_from_inline_L17_layer_input_restore | 26 | answer_word | remove_from_inline | single_layer |  | 17 | layer_input | restore | 5/26 | 5/26 | 2/26 | 3.6 | -0.202 | space:19, correct_prefix:5, newline:2 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_restore | 26 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | restore | 5/26 | 6/26 | 0/26 | 2.9 | 0.255 | space:20, correct_prefix:6 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L18_19_layer_out_random | 26 | answer_word | to_original | interval | L18_19 |  | layer_out | random | 17/26 | 20/26 | 2/26 | 1.3 | 1.303 | correct_prefix:20, space:4, newline:2 |
| answer_word_to_original_interval_L17_20_attn_out_random | 26 | answer_word | to_original | interval | L17_20 |  | attn_out | random | 17/26 | 18/26 | 6/26 | 1.4 | 0.899 | correct_prefix:18, newline:6, space:1, word:1 |
| answer_word_to_original_interval_L17_20_layer_out_random | 26 | answer_word | to_original | interval | L17_20 |  | layer_out | random | 14/26 | 16/26 | 4/26 | 1.4 | 1.240 | correct_prefix:16, space:6, newline:4 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_random | 26 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | random | 8/26 | 9/26 | 3/26 | 2.5 | 0.231 | space:14, correct_prefix:9, newline:3 |
| answer_word_to_original_interval_L17_20_mlp_out_random | 26 | answer_word | to_original | interval | L17_20 |  | mlp_out | random | 7/26 | 11/26 | 10/26 | 2.3 | 0.135 | correct_prefix:11, newline:10, space:5 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_random | 26 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | random | 1/26 | 2/26 | 9/26 | 4.2 | -1.178 | space:15, newline:9, correct_prefix:2 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_random | 26 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | random | 0/26 | 1/26 | 14/26 | 4.6 | -1.399 | newline:14, space:11, correct_prefix:1 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_random | 26 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | random | 0/26 | 0/26 | 15/26 | 5.8 | -1.942 | newline:15, space:11 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L17_20_layer_out_reverse | 26 | answer_word | to_original | interval | L17_20 |  | layer_out | reverse | 19/26 | 23/26 | 0/26 | 1.2 | 2.615 | correct_prefix:23, space:2, word:1 |
| answer_word_to_original_interval_L18_19_layer_out_reverse | 26 | answer_word | to_original | interval | L18_19 |  | layer_out | reverse | 13/26 | 17/26 | 0/26 | 1.3 | 1.894 | correct_prefix:17, space:9 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 10/26 | 11/26 | 0/26 | 2.0 | 0.750 | space:15, correct_prefix:11 |
| answer_word_to_original_interval_L17_20_attn_out_reverse | 26 | answer_word | to_original | interval | L17_20 |  | attn_out | reverse | 10/26 | 11/26 | 8/26 | 2.4 | -0.038 | correct_prefix:11, newline:8, space:4, word:3 |
| answer_word_to_original_interval_L17_20_mlp_out_reverse | 26 | answer_word | to_original | interval | L17_20 |  | mlp_out | reverse | 3/26 | 4/26 | 2/26 | 2.7 | -0.115 | space:20, correct_prefix:4, newline:2 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/26 | 0/26 | 13/26 | 5.7 | -1.625 | newline:13, space:13 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 17/26 | 4.7 | -1.851 | newline:17, space:9 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 24/26 | 5.1 | -2.034 | newline:24, space:2 |

### colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L17_20_mlp_out_restore | 26 | colon | to_original | interval | L17_20 |  | mlp_out | restore | 21/26 | 22/26 | 0/26 | 1.2 | 3.264 | correct_prefix:22, space:4 |
| colon_to_original_interval_L17_20_attn_out_restore | 26 | colon | to_original | interval | L17_20 |  | attn_out | restore | 18/26 | 21/26 | 0/26 | 1.3 | 3.303 | correct_prefix:21, space:3, word:2 |
| colon_to_original_L17_layer_input_restore | 26 | colon | to_original | single_layer |  | 17 | layer_input | restore | 15/26 | 17/26 | 3/26 | 1.4 | 0.822 | correct_prefix:17, space:6, newline:3 |
| colon_to_original_L17_layer_out_restore | 26 | colon | to_original | single_layer |  | 17 | layer_out | restore | 12/26 | 14/26 | 7/26 | 1.8 | 0.413 | correct_prefix:14, newline:7, space:5 |
| colon_to_original_interval_L17_20_layer_out_restore | 26 | colon | to_original | interval | L17_20 |  | layer_out | restore | 9/26 | 10/26 | 4/26 | 2.3 | 0.279 | space:12, correct_prefix:10, newline:4 |
| colon_to_original_interval_L18_19_layer_out_restore | 26 | colon | to_original | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 3/26 | 2.4 | 0.337 | space:14, correct_prefix:9, newline:3 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_remove_from_inline_L17_layer_input_restore | 26 | colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 0/26 | 1/26 | 14/26 | 4.5 | -1.260 | newline:14, space:11, correct_prefix:1 |
| colon_remove_from_inline_interval_L17_20_attn_out_restore | 26 | colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 3/26 | 2/26 | 23/26 | 4.4 | -1.697 | newline:23, correct_prefix:2, space:1 |
| colon_remove_from_inline_interval_L18_19_layer_out_restore | 26 | colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 3/26 | 4/26 | 13/26 | 2.9 | -0.462 | newline:13, space:9, correct_prefix:4 |
| colon_remove_from_inline_interval_L17_20_layer_out_restore | 26 | colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 3/26 | 4/26 | 10/26 | 3.3 | -0.510 | space:12, newline:10, correct_prefix:4 |
| colon_remove_from_inline_L17_layer_out_restore | 26 | colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 11/26 | 3.2 | -0.447 | newline:11, space:11, correct_prefix:4 |
| colon_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 20/26 | 21/26 | 0/26 | 1.2 | 2.750 | correct_prefix:21, space:5 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L17_20_mlp_out_random | 26 | colon | to_original | interval | L17_20 |  | mlp_out | random | 18/26 | 20/26 | 0/26 | 1.7 | 3.101 | correct_prefix:20, space:5, word:1 |
| colon_to_original_interval_L17_20_layer_out_random | 26 | colon | to_original | interval | L17_20 |  | layer_out | random | 16/26 | 19/26 | 4/26 | 1.7 | 0.803 | correct_prefix:19, newline:4, space:3 |
| colon_to_original_interval_L18_19_layer_out_random | 26 | colon | to_original | interval | L18_19 |  | layer_out | random | 14/26 | 15/26 | 3/26 | 1.7 | 0.692 | correct_prefix:15, space:7, newline:3, word:1 |
| colon_remove_from_inline_interval_L17_20_mlp_out_random | 26 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 10/26 | 10/26 | 0/26 | 2.0 | 1.538 | space:16, correct_prefix:10 |
| colon_remove_from_inline_interval_L17_20_attn_out_random | 26 | colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 8/26 | 8/26 | 13/26 | 3.4 | -0.601 | newline:13, correct_prefix:8, space:5 |
| colon_to_original_interval_L17_20_attn_out_random | 26 | colon | to_original | interval | L17_20 |  | attn_out | random | 7/26 | 9/26 | 10/26 | 4.5 | -0.688 | newline:10, correct_prefix:9, space:4, word:2, explanation:1 |
| colon_remove_from_inline_interval_L18_19_layer_out_random | 26 | colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 1/26 | 1/26 | 13/26 | 5.0 | -1.562 | newline:13, space:12, correct_prefix:1 |
| colon_remove_from_inline_interval_L17_20_layer_out_random | 26 | colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 0/26 | 0/26 | 17/26 | 5.2 | -1.716 | newline:17, space:9 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L17_20_mlp_out_reverse | 26 | colon | to_original | interval | L17_20 |  | mlp_out | reverse | 22/26 | 24/26 | 0/26 | 1.1 | 4.188 | correct_prefix:24, word:2 |
| colon_to_original_interval_L17_20_layer_out_reverse | 26 | colon | to_original | interval | L17_20 |  | layer_out | reverse | 18/26 | 21/26 | 1/26 | 1.2 | 1.500 | correct_prefix:21, space:2, word:2, newline:1 |
| colon_to_original_interval_L18_19_layer_out_reverse | 26 | colon | to_original | interval | L18_19 |  | layer_out | reverse | 17/26 | 19/26 | 1/26 | 1.3 | 1.351 | correct_prefix:19, space:3, word:3, newline:1 |
| colon_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 8/26 | 11/26 | 14/26 | 2.2 | 0.231 | newline:14, correct_prefix:11, space:1 |
| colon_to_original_interval_L17_20_attn_out_reverse | 26 | colon | to_original | interval | L17_20 |  | attn_out | reverse | 7/26 | 8/26 | 16/26 | 6.7 | -1.745 | newline:16, correct_prefix:8, space:2 |
| colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 5/26 | 6/26 | 0/26 | 2.2 | 1.188 | space:20, correct_prefix:6 |
| colon_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 1/26 | 19/26 | 5.7 | -1.851 | newline:19, space:6, correct_prefix:1 |
| colon_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 24/26 | 6.3 | -2.197 | newline:24, space:2 |

### answer_colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_interval_L17_20_attn_out_restore | 26 | answer_colon | to_original | interval | L17_20 |  | attn_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |
| answer_colon_to_original_interval_L17_20_mlp_out_restore | 26 | answer_colon | to_original | interval | L17_20 |  | mlp_out | restore | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |
| answer_colon_to_original_L17_layer_out_restore | 26 | answer_colon | to_original | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |
| answer_colon_to_original_interval_L18_19_layer_out_restore | 26 | answer_colon | to_original | interval | L18_19 |  | layer_out | restore | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |
| answer_colon_to_original_interval_L17_20_layer_out_restore | 26 | answer_colon | to_original | interval | L17_20 |  | layer_out | restore | 1/26 | 1/26 | 24/26 | 3.7 | -1.380 | newline:24, space:1, correct_prefix:1 |
| answer_colon_to_original_L17_layer_input_restore | 26 | answer_colon | to_original | single_layer |  | 17 | layer_input | restore | 1/26 | 1/26 | 25/26 | 4.0 | -1.457 | newline:25, correct_prefix:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L17_20_attn_out_restore | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_restore | 26 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |
| answer_colon_remove_from_inline_L17_layer_out_restore | 26 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_restore | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |
| answer_colon_remove_from_inline_L17_layer_input_restore | 26 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 10/26 | 11/26 | 2/26 | 2.5 | 0.240 | space:13, correct_prefix:11, newline:2 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 26/26 | 26/26 | 0/26 | 1.0 | 3.341 | correct_prefix:26 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_interval_L17_20_mlp_out_random | 26 | answer_colon | to_original | interval | L17_20 |  | mlp_out | random | 17/26 | 18/26 | 0/26 | 1.5 | 3.846 | correct_prefix:18, word:7, space:1 |
| answer_colon_to_original_interval_L17_20_layer_out_random | 26 | answer_colon | to_original | interval | L17_20 |  | layer_out | random | 14/26 | 14/26 | 6/26 | 2.0 | 0.688 | correct_prefix:14, newline:6, space:6 |
| answer_colon_to_original_interval_L17_20_attn_out_random | 26 | answer_colon | to_original | interval | L17_20 |  | attn_out | random | 11/26 | 11/26 | 3/26 | 3.3 | 0.827 | word:12, correct_prefix:11, newline:3 |
| answer_colon_to_original_interval_L18_19_layer_out_random | 26 | answer_colon | to_original | interval | L18_19 |  | layer_out | random | 11/26 | 12/26 | 5/26 | 2.0 | 0.769 | correct_prefix:12, space:6, newline:5, word:2, explanation:1 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_random | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 10/26 | 10/26 | 0/26 | 1.7 | 1.620 | space:16, correct_prefix:10 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_random | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 10/26 | 12/26 | 4/26 | 2.5 | 0.750 | correct_prefix:12, space:10, newline:4 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_random | 26 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 0/26 | 1/26 | 14/26 | 5.1 | -1.697 | newline:14, space:11, correct_prefix:1 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_random | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 0/26 | 0/26 | 17/26 | 4.9 | -1.620 | newline:17, space:9 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_interval_L17_20_layer_out_reverse | 26 | answer_colon | to_original | interval | L17_20 |  | layer_out | reverse | 18/26 | 20/26 | 0/26 | 1.3 | 2.880 | correct_prefix:20, word:3, space:2, explanation:1 |
| answer_colon_to_original_interval_L17_20_mlp_out_reverse | 26 | answer_colon | to_original | interval | L17_20 |  | mlp_out | reverse | 15/26 | 17/26 | 0/26 | 1.3 | 5.019 | correct_prefix:17, word:9 |
| answer_colon_to_original_interval_L18_19_layer_out_reverse | 26 | answer_colon | to_original | interval | L18_19 |  | layer_out | reverse | 15/26 | 18/26 | 0/26 | 1.4 | 1.846 | correct_prefix:18, space:5, word:3 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 13/26 | 17/26 | 1/26 | 1.5 | 2.173 | correct_prefix:17, space:8, newline:1 |
| answer_colon_to_original_interval_L17_20_attn_out_reverse | 26 | answer_colon | to_original | interval | L17_20 |  | attn_out | reverse | 7/26 | 7/26 | 1/26 | 6.1 | -0.889 | word:17, correct_prefix:7, space:1, newline:1 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 2/26 | 2/26 | 0/26 | 4.1 | 0.351 | space:24, correct_prefix:2 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 22/26 | 6.1 | -2.519 | newline:22, space:4 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 25/26 | 5.8 | -2.361 | newline:25, space:1 |

### answer_label_aligned

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_interval_L17_20_attn_out_restore | 26 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_restore | 26 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | restore | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |
| answer_label_aligned_to_original_L17_layer_out_restore | 26 | answer_label_aligned | to_original | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_restore | 26 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | restore | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_restore | 26 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | restore | 1/26 | 1/26 | 24/26 | 3.7 | -1.380 | newline:24, space:1, correct_prefix:1 |
| answer_label_aligned_to_original_L17_layer_input_restore | 26 | answer_label_aligned | to_original | single_layer |  | 17 | layer_input | restore | 1/26 | 1/26 | 25/26 | 4.0 | -1.457 | newline:25, correct_prefix:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | restore | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_restore | 26 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |
| answer_label_aligned_remove_from_inline_L17_layer_out_restore | 26 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_out | restore | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_restore | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | restore | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |
| answer_label_aligned_remove_from_inline_L17_layer_input_restore | 26 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_input | restore | 10/26 | 11/26 | 2/26 | 2.5 | 0.240 | space:13, correct_prefix:11, newline:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 26/26 | 26/26 | 0/26 | 1.0 | 3.341 | correct_prefix:26 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_interval_L17_20_mlp_out_random | 26 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | random | 18/26 | 19/26 | 0/26 | 1.3 | 3.755 | correct_prefix:19, word:5, space:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_random | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | random | 11/26 | 11/26 | 2/26 | 2.5 | 1.120 | space:13, correct_prefix:11, newline:2 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_random | 26 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | random | 11/26 | 12/26 | 4/26 | 2.5 | 0.514 | correct_prefix:12, space:10, newline:4 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_random | 26 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | random | 11/26 | 13/26 | 7/26 | 2.3 | 0.231 | correct_prefix:13, newline:7, space:5, word:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_random | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | random | 10/26 | 11/26 | 0/26 | 1.6 | 1.851 | space:15, correct_prefix:11 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_random | 26 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | random | 10/26 | 10/26 | 2/26 | 4.7 | 0.798 | correct_prefix:10, word:9, space:5, newline:2 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random | 26 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | random | 2/26 | 2/26 | 16/26 | 5.1 | -1.668 | newline:16, space:8, correct_prefix:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_random | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | random | 1/26 | 1/26 | 17/26 | 5.3 | -1.688 | newline:17, space:8, correct_prefix:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_interval_L17_20_layer_out_reverse | 26 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | reverse | 18/26 | 20/26 | 0/26 | 1.3 | 2.880 | correct_prefix:20, word:3, space:2, explanation:1 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_reverse | 26 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | reverse | 15/26 | 17/26 | 0/26 | 1.3 | 5.019 | correct_prefix:17, word:9 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_reverse | 26 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | reverse | 15/26 | 18/26 | 0/26 | 1.4 | 1.846 | correct_prefix:18, space:5, word:3 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 13/26 | 17/26 | 1/26 | 1.5 | 2.173 | correct_prefix:17, space:8, newline:1 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_reverse | 26 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | reverse | 7/26 | 7/26 | 1/26 | 6.1 | -0.889 | word:17, correct_prefix:7, space:1, newline:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 2/26 | 2/26 | 0/26 | 4.1 | 0.351 | space:24, correct_prefix:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 22/26 | 6.1 | -2.519 | newline:22, space:4 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 25/26 | 5.8 | -2.361 | newline:25, space:1 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_interval_L17_20_attn_out_restore | 26 | separator | to_original | interval | L17_20 |  | attn_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |
| separator_to_original_interval_L17_20_mlp_out_restore | 26 | separator | to_original | interval | L17_20 |  | mlp_out | restore | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |
| separator_to_original_L17_layer_out_restore | 26 | separator | to_original | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |
| separator_to_original_interval_L18_19_layer_out_restore | 26 | separator | to_original | interval | L18_19 |  | layer_out | restore | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |
| separator_to_original_interval_L17_20_layer_out_restore | 26 | separator | to_original | interval | L17_20 |  | layer_out | restore | 1/26 | 1/26 | 24/26 | 3.7 | -1.380 | newline:24, space:1, correct_prefix:1 |
| separator_to_original_L17_layer_input_restore | 26 | separator | to_original | single_layer |  | 17 | layer_input | restore | 1/26 | 1/26 | 25/26 | 4.0 | -1.457 | newline:25, correct_prefix:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 26 | separator | remove_from_inline | interval | L17_20 |  | attn_out | restore | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 26 | separator | remove_from_inline | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |
| separator_remove_from_inline_L17_layer_out_restore | 26 | separator | remove_from_inline | single_layer |  | 17 | layer_out | restore | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 26 | separator | remove_from_inline | interval | L17_20 |  | layer_out | restore | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |
| separator_remove_from_inline_L17_layer_input_restore | 26 | separator | remove_from_inline | single_layer |  | 17 | layer_input | restore | 10/26 | 11/26 | 2/26 | 2.5 | 0.240 | space:13, correct_prefix:11, newline:2 |
| separator_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 26/26 | 26/26 | 0/26 | 1.0 | 3.341 | correct_prefix:26 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_interval_L17_20_mlp_out_random | 26 | separator | to_original | interval | L17_20 |  | mlp_out | random | 19/26 | 20/26 | 0/26 | 1.3 | 3.635 | correct_prefix:20, word:5, space:1 |
| separator_to_original_interval_L17_20_layer_out_random | 26 | separator | to_original | interval | L17_20 |  | layer_out | random | 11/26 | 14/26 | 6/26 | 2.0 | 0.356 | correct_prefix:14, newline:6, space:5, explanation:1 |
| separator_remove_from_inline_interval_L17_20_attn_out_random | 26 | separator | remove_from_inline | interval | L17_20 |  | attn_out | random | 10/26 | 12/26 | 2/26 | 2.1 | 0.962 | correct_prefix:12, space:12, newline:2 |
| separator_remove_from_inline_interval_L17_20_mlp_out_random | 26 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | random | 9/26 | 10/26 | 0/26 | 1.6 | 1.678 | space:16, correct_prefix:10 |
| separator_to_original_interval_L18_19_layer_out_random | 26 | separator | to_original | interval | L18_19 |  | layer_out | random | 9/26 | 11/26 | 4/26 | 2.5 | 0.361 | correct_prefix:11, space:10, newline:4, word:1 |
| separator_to_original_interval_L17_20_attn_out_random | 26 | separator | to_original | interval | L17_20 |  | attn_out | random | 6/26 | 7/26 | 4/26 | 4.7 | 0.087 | word:13, correct_prefix:7, newline:4, space:2 |
| separator_remove_from_inline_interval_L17_20_layer_out_random | 26 | separator | remove_from_inline | interval | L17_20 |  | layer_out | random | 1/26 | 2/26 | 14/26 | 5.3 | -1.712 | newline:14, space:10, correct_prefix:2 |
| separator_remove_from_inline_interval_L18_19_layer_out_random | 26 | separator | remove_from_inline | interval | L18_19 |  | layer_out | random | 0/26 | 1/26 | 17/26 | 5.3 | -1.827 | newline:17, space:8, correct_prefix:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_interval_L17_20_layer_out_reverse | 26 | separator | to_original | interval | L17_20 |  | layer_out | reverse | 18/26 | 20/26 | 0/26 | 1.3 | 2.880 | correct_prefix:20, word:3, space:2, explanation:1 |
| separator_to_original_interval_L17_20_mlp_out_reverse | 26 | separator | to_original | interval | L17_20 |  | mlp_out | reverse | 15/26 | 17/26 | 0/26 | 1.3 | 5.019 | correct_prefix:17, word:9 |
| separator_to_original_interval_L18_19_layer_out_reverse | 26 | separator | to_original | interval | L18_19 |  | layer_out | reverse | 15/26 | 18/26 | 0/26 | 1.4 | 1.846 | correct_prefix:18, space:5, word:3 |
| separator_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | separator | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 13/26 | 17/26 | 1/26 | 1.5 | 2.173 | correct_prefix:17, space:8, newline:1 |
| separator_to_original_interval_L17_20_attn_out_reverse | 26 | separator | to_original | interval | L17_20 |  | attn_out | reverse | 7/26 | 7/26 | 1/26 | 6.1 | -0.889 | word:17, correct_prefix:7, space:1, newline:1 |
| separator_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 2/26 | 2/26 | 0/26 | 4.1 | 0.351 | space:24, correct_prefix:2 |
| separator_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | separator | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 22/26 | 6.1 | -2.519 | newline:22, space:4 |
| separator_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | separator | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 25/26 | 5.8 | -2.361 | newline:25, space:1 |

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L17_20_mlp_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | mlp_out | restore | 21/26 | 22/26 | 0/26 | 1.2 | 3.264 | correct_prefix:22, space:4 |
| prompt_last_to_original_interval_L17_20_attn_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | attn_out | restore | 18/26 | 21/26 | 0/26 | 1.3 | 3.303 | correct_prefix:21, space:3, word:2 |
| prompt_last_to_original_L17_layer_input_restore | 26 | prompt_last | to_original | single_layer |  | 17 | layer_input | restore | 15/26 | 17/26 | 3/26 | 1.4 | 0.822 | correct_prefix:17, space:6, newline:3 |
| prompt_last_to_original_L17_layer_out_restore | 26 | prompt_last | to_original | single_layer |  | 17 | layer_out | restore | 12/26 | 14/26 | 7/26 | 1.8 | 0.413 | correct_prefix:14, newline:7, space:5 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 26 | prompt_last | to_original | interval | L17_20 |  | layer_out | restore | 9/26 | 10/26 | 4/26 | 2.3 | 0.279 | space:12, correct_prefix:10, newline:4 |
| prompt_last_to_original_interval_L18_19_layer_out_restore | 26 | prompt_last | to_original | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 3/26 | 2.4 | 0.337 | space:14, correct_prefix:9, newline:3 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_remove_from_inline_L17_layer_input_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_input | restore | 0/26 | 1/26 | 14/26 | 4.5 | -1.260 | newline:14, space:11, correct_prefix:1 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | restore | 3/26 | 2/26 | 23/26 | 4.4 | -1.697 | newline:23, correct_prefix:2, space:1 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 26 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | restore | 3/26 | 4/26 | 13/26 | 2.9 | -0.462 | newline:13, space:9, correct_prefix:4 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | restore | 3/26 | 4/26 | 10/26 | 3.3 | -0.510 | space:12, newline:10, correct_prefix:4 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 26 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 11/26 | 3.2 | -0.447 | newline:11, space:11, correct_prefix:4 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 20/26 | 21/26 | 0/26 | 1.2 | 2.750 | correct_prefix:21, space:5 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L17_20_mlp_out_random | 26 | prompt_last | to_original | interval | L17_20 |  | mlp_out | random | 17/26 | 19/26 | 0/26 | 1.4 | 3.024 | correct_prefix:19, word:4, space:3 |
| prompt_last_to_original_interval_L18_19_layer_out_random | 26 | prompt_last | to_original | interval | L18_19 |  | layer_out | random | 14/26 | 17/26 | 4/26 | 1.7 | 1.038 | correct_prefix:17, space:5, newline:4 |
| prompt_last_to_original_interval_L17_20_layer_out_random | 26 | prompt_last | to_original | interval | L17_20 |  | layer_out | random | 12/26 | 13/26 | 4/26 | 1.7 | 0.750 | correct_prefix:13, space:8, newline:4, word:1 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_random | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | random | 11/26 | 11/26 | 0/26 | 1.7 | 1.639 | space:15, correct_prefix:11 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_random | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | random | 11/26 | 11/26 | 11/26 | 3.0 | -0.476 | correct_prefix:11, newline:11, space:4 |
| prompt_last_to_original_interval_L17_20_attn_out_random | 26 | prompt_last | to_original | interval | L17_20 |  | attn_out | random | 10/26 | 10/26 | 14/26 | 4.1 | -0.495 | newline:14, correct_prefix:10, space:2 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_random | 26 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | random | 3/26 | 4/26 | 14/26 | 4.9 | -1.500 | newline:14, space:8, correct_prefix:4 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_random | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | random | 0/26 | 0/26 | 18/26 | 5.3 | -1.736 | newline:18, space:8 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L17_20_mlp_out_reverse | 26 | prompt_last | to_original | interval | L17_20 |  | mlp_out | reverse | 22/26 | 24/26 | 0/26 | 1.1 | 4.188 | correct_prefix:24, word:2 |
| prompt_last_to_original_interval_L17_20_layer_out_reverse | 26 | prompt_last | to_original | interval | L17_20 |  | layer_out | reverse | 18/26 | 21/26 | 1/26 | 1.2 | 1.500 | correct_prefix:21, space:2, word:2, newline:1 |
| prompt_last_to_original_interval_L18_19_layer_out_reverse | 26 | prompt_last | to_original | interval | L18_19 |  | layer_out | reverse | 17/26 | 19/26 | 1/26 | 1.3 | 1.351 | correct_prefix:19, space:3, word:3, newline:1 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 8/26 | 11/26 | 14/26 | 2.2 | 0.231 | newline:14, correct_prefix:11, space:1 |
| prompt_last_to_original_interval_L17_20_attn_out_reverse | 26 | prompt_last | to_original | interval | L17_20 |  | attn_out | reverse | 7/26 | 8/26 | 16/26 | 6.7 | -1.745 | newline:16, correct_prefix:8, space:2 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 5/26 | 6/26 | 0/26 | 2.2 | 1.188 | space:20, correct_prefix:6 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 1/26 | 19/26 | 5.7 | -1.851 | newline:19, space:6, correct_prefix:1 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 24/26 | 6.3 | -2.197 | newline:24, space:2 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_to_original_interval_L17_20_attn_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | restore | 23/26 | 26/26 | 0/26 | 1.0 | 4.567 | correct_prefix:26 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | restore | 5/26 | 6/26 | 0/26 | 2.2 | 1.106 | space:18, correct_prefix:6, word:2 |
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 26 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 26 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| question_mark_answer_to_original_L17_layer_input_restore | 26 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| question_mark_answer_to_original_L17_layer_out_restore | 26 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | restore | 12/26 | 14/26 | 9/26 | 3.2 | 0.279 | correct_prefix:14, newline:9, space:2, word:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | restore | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | restore | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 26 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 24/26 | 26/26 | 0/26 | 1.0 | 4.428 | correct_prefix:26 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_random | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | random | 16/26 | 20/26 | 2/26 | 1.6 | 2.793 | correct_prefix:20, space:4, newline:2 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_random | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | random | 12/26 | 14/26 | 0/26 | 1.9 | 1.596 | correct_prefix:14, space:8, word:4 |
| question_mark_answer_to_original_interval_L17_20_layer_out_random | 26 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | random | 10/26 | 11/26 | 6/26 | 2.2 | 0.178 | correct_prefix:11, space:8, newline:6, word:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_random | 26 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | random | 8/26 | 11/26 | 0/26 | 1.7 | 3.192 | word:13, correct_prefix:11, space:2 |
| question_mark_answer_to_original_interval_L18_19_layer_out_random | 26 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | random | 6/26 | 7/26 | 9/26 | 2.9 | -0.067 | space:10, newline:9, correct_prefix:7 |
| question_mark_answer_to_original_interval_L17_20_attn_out_random | 26 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | random | 5/26 | 6/26 | 1/26 | 8.0 | -0.115 | word:15, correct_prefix:6, space:3, newline:1, explanation:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_random | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | random | 1/26 | 2/26 | 16/26 | 5.2 | -1.788 | newline:16, space:8, correct_prefix:2 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_random | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | random | 0/26 | 0/26 | 16/26 | 5.4 | -1.865 | newline:16, space:10 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 17/26 | 25/26 | 0/26 | 1.0 | 3.716 | correct_prefix:25, space:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_reverse | 26 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | reverse | 14/26 | 17/26 | 0/26 | 1.5 | 2.966 | correct_prefix:17, word:9 |
| question_mark_answer_to_original_interval_L18_19_layer_out_reverse | 26 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | reverse | 7/26 | 9/26 | 0/26 | 2.0 | 2.010 | word:16, correct_prefix:9, space:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_reverse | 26 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | reverse | 4/26 | 5/26 | 0/26 | 2.0 | 4.106 | word:21, correct_prefix:5 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/26 | 0/26 | 0/26 | 4.6 | 0.375 | space:25, word:1 |
| question_mark_answer_to_original_interval_L17_20_attn_out_reverse | 26 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | reverse | 0/26 | 0/26 | 1/26 | 27.1 | -4.291 | word:25, newline:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 23/26 | 5.7 | -2.067 | newline:23, space:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 24/26 | 6.4 | -2.438 | newline:24, space:2 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_to_original_interval_L17_20_attn_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | attn_out | restore | 13/26 | 25/26 | 0/26 | 1.1 | 4.183 | correct_prefix:25, word:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | mlp_out | restore | 8/26 | 12/26 | 0/26 | 1.7 | 1.736 | correct_prefix:12, space:12, word:2 |
| relation_tail_to_original_interval_L17_20_layer_out_restore | 26 | relation_tail | to_original | interval | L17_20 |  | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| relation_tail_to_original_interval_L18_19_layer_out_restore | 26 | relation_tail | to_original | interval | L18_19 |  | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| relation_tail_to_original_L17_layer_input_restore | 26 | relation_tail | to_original | single_layer |  | 17 | layer_input | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |
| relation_tail_to_original_L17_layer_out_restore | 26 | relation_tail | to_original | single_layer |  | 17 | layer_out | restore | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | restore | 9/26 | 11/26 | 12/26 | 4.9 | -0.673 | newline:12, correct_prefix:11, word:3 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 16/26 | 26/26 | 0/26 | 1.0 | 4.361 | correct_prefix:26 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | restore | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | restore | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |
| relation_tail_remove_from_inline_L17_layer_input_restore | 26 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_input | restore | 20/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_random | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | random | 13/26 | 18/26 | 0/26 | 1.5 | 2.183 | correct_prefix:18, space:7, word:1 |
| relation_tail_to_original_interval_L18_19_layer_out_random | 26 | relation_tail | to_original | interval | L18_19 |  | layer_out | random | 12/26 | 14/26 | 4/26 | 2.0 | 0.683 | correct_prefix:14, space:7, newline:4, word:1 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_random | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | random | 11/26 | 21/26 | 2/26 | 1.7 | 2.236 | correct_prefix:21, word:2, newline:2, space:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_random | 26 | relation_tail | to_original | interval | L17_20 |  | mlp_out | random | 10/26 | 14/26 | 0/26 | 1.7 | 3.226 | correct_prefix:14, word:11, space:1 |
| relation_tail_to_original_interval_L17_20_layer_out_random | 26 | relation_tail | to_original | interval | L17_20 |  | layer_out | random | 9/26 | 9/26 | 7/26 | 2.7 | 0.173 | space:10, correct_prefix:9, newline:7 |
| relation_tail_to_original_interval_L17_20_attn_out_random | 26 | relation_tail | to_original | interval | L17_20 |  | attn_out | random | 5/26 | 5/26 | 2/26 | 13.0 | -1.471 | word:19, correct_prefix:5, newline:2 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_random | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | random | 1/26 | 2/26 | 15/26 | 4.6 | -1.399 | newline:15, space:9, correct_prefix:2 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_random | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | random | 1/26 | 2/26 | 16/26 | 4.5 | -1.418 | newline:16, space:8, correct_prefix:2 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_to_original_interval_L17_20_layer_out_reverse | 26 | relation_tail | to_original | interval | L17_20 |  | layer_out | reverse | 14/26 | 17/26 | 0/26 | 1.5 | 2.966 | correct_prefix:17, word:9 |
| relation_tail_to_original_interval_L18_19_layer_out_reverse | 26 | relation_tail | to_original | interval | L18_19 |  | layer_out | reverse | 7/26 | 9/26 | 0/26 | 2.0 | 2.010 | word:16, correct_prefix:9, space:1 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_reverse | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 4/26 | 23/26 | 0/26 | 1.2 | 3.327 | correct_prefix:23, word:3 |
| relation_tail_to_original_interval_L17_20_mlp_out_reverse | 26 | relation_tail | to_original | interval | L17_20 |  | mlp_out | reverse | 2/26 | 3/26 | 0/26 | 2.3 | 3.942 | word:23, correct_prefix:3 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_reverse | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 1/26 | 2/26 | 0/26 | 3.7 | 0.798 | space:22, correct_prefix:2, word:2 |
| relation_tail_to_original_interval_L17_20_attn_out_reverse | 26 | relation_tail | to_original | interval | L17_20 |  | attn_out | reverse | 0/26 | 0/26 | 1/26 | 31.8 | -4.981 | word:25, newline:1 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_reverse | 26 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 0/26 | 0/26 | 23/26 | 5.7 | -2.067 | newline:23, space:3 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_reverse | 26 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 0/26 | 0/26 | 24/26 | 6.4 | -2.438 | newline:24, space:2 |

### Global Top Notes

- Top sufficiency: answer_word_to_original_interval_L17_20_attn_out_restore exact=25/26 newline=0/26 rank=1.0; question_mark_answer_to_original_interval_L17_20_attn_out_restore exact=23/26 newline=0/26 rank=1.0; colon_to_original_interval_L17_20_mlp_out_restore exact=21/26 newline=0/26 rank=1.2; prompt_last_to_original_interval_L17_20_mlp_out_restore exact=21/26 newline=0/26 rank=1.2; answer_colon_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2; answer_label_aligned_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2; separator_to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26 rank=1.2; colon_to_original_interval_L17_20_attn_out_restore exact=18/26 newline=0/26 rank=1.3; prompt_last_to_original_interval_L17_20_attn_out_restore exact=18/26 newline=0/26 rank=1.3; answer_colon_to_original_interval_L17_20_mlp_out_restore exact=17/26 newline=0/26 rank=1.3
- Top necessity/remove: answer_word_remove_from_inline_interval_L17_20_mlp_out_restore exact=0/26 newline=15/26 rank=5.5; colon_remove_from_inline_L17_layer_input_restore exact=0/26 newline=14/26 rank=4.5; prompt_last_remove_from_inline_L17_layer_input_restore exact=0/26 newline=14/26 rank=4.5; colon_remove_from_inline_interval_L17_20_attn_out_restore exact=3/26 newline=23/26 rank=4.4; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=3/26 newline=23/26 rank=4.4; colon_remove_from_inline_interval_L18_19_layer_out_restore exact=3/26 newline=13/26 rank=2.9; prompt_last_remove_from_inline_interval_L18_19_layer_out_restore exact=3/26 newline=13/26 rank=2.9; colon_remove_from_inline_interval_L17_20_layer_out_restore exact=3/26 newline=10/26 rank=3.3; prompt_last_remove_from_inline_interval_L17_20_layer_out_restore exact=3/26 newline=10/26 rank=3.3; colon_remove_from_inline_L17_layer_out_restore exact=4/26 newline=11/26 rank=3.2

## glm4

- raw_cases: 320 / target_seen: 36 / cases_written: 36 / mode_rows: 8136
- positions: `['answer_word', 'colon', 'answer_colon', 'answer_label_aligned', 'separator', 'prompt_last', 'question_mark_answer', 'relation_tail']`
- filtered: `{'not_target': 284, 'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 19.16

### Baselines

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| original | 36 |  |  | baseline |  |  |  |  | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| inline | 36 |  |  | baseline |  |  |  |  | 27/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |

### Position Control Overview

| position | best restore sufficiency | best restore necessity/remove | best random | best reverse |
|---|---|---|---|---|
| answer_word | answer_word_to_original_interval_L18_19_layer_out_restore exact=30/36 newline=0/36 rank=1.5 | answer_word_remove_from_inline_interval_L17_20_mlp_out_restore exact=26/36 newline=0/36 rank=1.6 | answer_word_to_original_interval_L18_19_layer_out_random exact=28/36 newline=0/36 rank=1.4 | answer_word_to_original_interval_L17_20_mlp_out_reverse exact=28/36 newline=0/36 rank=1.3 |
| colon | colon_to_original_interval_L18_19_layer_out_restore exact=25/36 newline=0/36 rank=1.6 | colon_remove_from_inline_interval_L17_20_attn_out_restore exact=16/36 newline=0/36 rank=3.3 | colon_to_original_interval_L18_19_layer_out_random exact=28/36 newline=0/36 rank=1.4 | colon_to_original_interval_L18_19_layer_out_reverse exact=25/36 newline=0/36 rank=1.6 |
| answer_colon | answer_colon_to_original_L17_layer_input_restore exact=28/36 newline=0/36 rank=1.6 | answer_colon_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3 | answer_colon_remove_from_inline_interval_L17_20_layer_out_random exact=27/36 newline=0/36 rank=1.7 | answer_colon_to_original_interval_L17_20_layer_out_reverse exact=23/36 newline=0/36 rank=1.5 |
| answer_label_aligned | answer_label_aligned_to_original_L17_layer_input_restore exact=28/36 newline=0/36 rank=1.6 | answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3 | answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random exact=27/36 newline=0/36 rank=1.8 | answer_label_aligned_to_original_interval_L17_20_layer_out_reverse exact=23/36 newline=0/36 rank=1.5 |
| separator | separator_to_original_L17_layer_input_restore exact=28/36 newline=0/36 rank=1.6 | separator_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3 | separator_to_original_interval_L17_20_layer_out_random exact=26/36 newline=0/36 rank=1.5 | separator_to_original_interval_L17_20_layer_out_reverse exact=23/36 newline=0/36 rank=1.5 |
| prompt_last | prompt_last_to_original_interval_L18_19_layer_out_restore exact=25/36 newline=0/36 rank=1.6 | prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=16/36 newline=0/36 rank=3.3 | prompt_last_to_original_interval_L17_20_layer_out_random exact=27/36 newline=0/36 rank=1.5 | prompt_last_to_original_interval_L18_19_layer_out_reverse exact=25/36 newline=0/36 rank=1.6 |
| question_mark_answer | question_mark_answer_to_original_interval_L17_20_layer_out_restore exact=28/36 newline=0/36 rank=1.5 | question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore exact=2/36 newline=0/36 rank=3.4 | question_mark_answer_remove_from_inline_interval_L18_19_layer_out_random exact=24/36 newline=0/36 rank=1.8 | question_mark_answer_to_original_interval_L17_20_attn_out_reverse exact=9/36 newline=0/36 rank=10.2 |
| relation_tail | relation_tail_to_original_interval_L17_20_layer_out_restore exact=29/36 newline=0/36 rank=1.5 | relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore exact=0/36 newline=0/36 rank=3.5 | relation_tail_remove_from_inline_interval_L18_19_layer_out_random exact=24/36 newline=0/36 rank=1.9 | relation_tail_to_original_interval_L17_20_attn_out_reverse exact=10/36 newline=0/36 rank=15.5 |

### answer_word

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L18_19_layer_out_restore | 36 | answer_word | to_original | interval | L18_19 |  | layer_out | restore | 30/36 | 31/36 | 0/36 | 1.5 | 83.336 | correct_prefix:31, word:3, explanation:2 |
| answer_word_to_original_L17_layer_input_restore | 36 | answer_word | to_original | single_layer |  | 17 | layer_input | restore | 29/36 | 31/36 | 0/36 | 1.5 | 91.155 | correct_prefix:31, word:3, explanation:2 |
| answer_word_to_original_L17_layer_out_restore | 36 | answer_word | to_original | single_layer |  | 17 | layer_out | restore | 29/36 | 31/36 | 0/36 | 1.5 | 80.715 | correct_prefix:31, word:3, explanation:2 |
| answer_word_to_original_interval_L17_20_layer_out_restore | 36 | answer_word | to_original | interval | L17_20 |  | layer_out | restore | 26/36 | 29/36 | 0/36 | 1.6 | 80.719 | correct_prefix:29, explanation:4, word:3 |
| answer_word_to_original_interval_L17_20_attn_out_restore | 36 | answer_word | to_original | interval | L17_20 |  | attn_out | restore | 25/36 | 27/36 | 0/36 | 2.0 | 93.738 | correct_prefix:27, word:6, explanation:3 |
| answer_word_to_original_interval_L17_20_mlp_out_restore | 36 | answer_word | to_original | interval | L17_20 |  | mlp_out | restore | 13/36 | 13/36 | 0/36 | 2.4 | 77.938 | word:21, correct_prefix:13, explanation:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 26/36 | 29/36 | 0/36 | 1.6 | 56.914 | correct_prefix:29, explanation:5, word:2 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_restore | 36 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.4 | 64.947 | correct_prefix:30, word:4, explanation:2 |
| answer_word_remove_from_inline_L17_layer_input_restore | 36 | answer_word | remove_from_inline | single_layer |  | 17 | layer_input | restore | 28/36 | 32/36 | 0/36 | 1.4 | 64.930 | correct_prefix:32, word:2, explanation:2 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_restore | 36 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | restore | 29/36 | 31/36 | 0/36 | 1.3 | 70.268 | correct_prefix:31, word:3, explanation:2 |
| answer_word_remove_from_inline_L17_layer_out_restore | 36 | answer_word | remove_from_inline | single_layer |  | 17 | layer_out | restore | 29/36 | 32/36 | 0/36 | 1.3 | 62.350 | correct_prefix:32, word:2, explanation:2 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_restore | 36 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | restore | 30/36 | 32/36 | 0/36 | 1.9 | 77.931 | correct_prefix:32, explanation:4 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L18_19_layer_out_random | 36 | answer_word | to_original | interval | L18_19 |  | layer_out | random | 28/36 | 31/36 | 0/36 | 1.4 | 77.955 | correct_prefix:31, word:4, explanation:1 |
| answer_word_to_original_interval_L17_20_mlp_out_random | 36 | answer_word | to_original | interval | L17_20 |  | mlp_out | random | 26/36 | 28/36 | 0/36 | 1.5 | 62.135 | correct_prefix:28, word:6, explanation:2 |
| answer_word_to_original_interval_L17_20_layer_out_random | 36 | answer_word | to_original | interval | L17_20 |  | layer_out | random | 26/36 | 28/36 | 0/36 | 1.5 | 72.793 | correct_prefix:28, word:6, explanation:2 |
| answer_word_to_original_interval_L17_20_attn_out_random | 36 | answer_word | to_original | interval | L17_20 |  | attn_out | random | 26/36 | 27/36 | 0/36 | 1.8 | 88.522 | correct_prefix:27, word:7, explanation:2 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_random | 36 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | random | 26/36 | 27/36 | 0/36 | 1.8 | 64.832 | correct_prefix:27, word:5, explanation:4 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_random | 36 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | random | 25/36 | 28/36 | 0/36 | 1.7 | 56.869 | correct_prefix:28, word:4, explanation:4 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_random | 36 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | random | 25/36 | 27/36 | 0/36 | 2.4 | 67.460 | correct_prefix:27, explanation:6, word:3 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_random | 36 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | random | 17/36 | 17/36 | 0/36 | 2.2 | 40.882 | correct_prefix:17, word:15, explanation:4 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L17_20_mlp_out_reverse | 36 | answer_word | to_original | interval | L17_20 |  | mlp_out | reverse | 28/36 | 32/36 | 0/36 | 1.3 | 48.992 | correct_prefix:32, explanation:2, word:2 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 26/36 | 28/36 | 0/36 | 1.7 | 46.275 | correct_prefix:28, word:4, explanation:4 |
| answer_word_to_original_interval_L17_20_layer_out_reverse | 36 | answer_word | to_original | interval | L17_20 |  | layer_out | reverse | 25/36 | 27/36 | 0/36 | 1.3 | 72.754 | correct_prefix:27, word:9 |
| answer_word_to_original_interval_L18_19_layer_out_reverse | 36 | answer_word | to_original | interval | L18_19 |  | layer_out | reverse | 24/36 | 27/36 | 0/36 | 1.4 | 72.731 | correct_prefix:27, word:9 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 24/36 | 29/36 | 0/36 | 2.0 | 38.198 | correct_prefix:29, explanation:4, word:3 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 19/36 | 19/36 | 0/36 | 3.8 | 77.895 | correct_prefix:19, word:9, explanation:8 |
| answer_word_to_original_interval_L17_20_attn_out_reverse | 36 | answer_word | to_original | interval | L17_20 |  | attn_out | reverse | 16/36 | 19/36 | 0/36 | 1.9 | 78.036 | correct_prefix:19, word:17 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 6/36 | 7/36 | 0/36 | 3.9 | 46.154 | word:24, correct_prefix:7, explanation:5 |

### colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L18_19_layer_out_restore | 36 | colon | to_original | interval | L18_19 |  | layer_out | restore | 25/36 | 28/36 | 0/36 | 1.6 | 75.367 | correct_prefix:28, word:6, explanation:2 |
| colon_to_original_L17_layer_input_restore | 36 | colon | to_original | single_layer |  | 17 | layer_input | restore | 25/36 | 28/36 | 0/36 | 1.6 | 77.969 | correct_prefix:28, word:6, explanation:2 |
| colon_to_original_L17_layer_out_restore | 36 | colon | to_original | single_layer |  | 17 | layer_out | restore | 24/36 | 28/36 | 0/36 | 1.6 | 75.314 | correct_prefix:28, word:6, explanation:2 |
| colon_to_original_interval_L17_20_layer_out_restore | 36 | colon | to_original | interval | L17_20 |  | layer_out | restore | 24/36 | 27/36 | 0/36 | 1.6 | 75.378 | correct_prefix:27, word:7, explanation:2 |
| colon_to_original_interval_L17_20_mlp_out_restore | 36 | colon | to_original | interval | L17_20 |  | mlp_out | restore | 8/36 | 12/36 | 0/36 | 2.2 | 99.000 | word:23, correct_prefix:12, explanation:1 |
| colon_to_original_interval_L17_20_attn_out_restore | 36 | colon | to_original | interval | L17_20 |  | attn_out | restore | 2/36 | 2/36 | 0/36 | 10.1 | 99.000 | word:34, correct_prefix:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_remove_from_inline_interval_L17_20_attn_out_restore | 36 | colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 16/36 | 19/36 | 0/36 | 3.3 | 96.380 | correct_prefix:19, word:16, explanation:1 |
| colon_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 19/36 | 21/36 | 0/36 | 1.8 | 99.000 | correct_prefix:21, word:14, explanation:1 |
| colon_remove_from_inline_interval_L17_20_layer_out_restore | 36 | colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 30/36 | 33/36 | 0/36 | 1.2 | 64.969 | correct_prefix:33, word:2, explanation:1 |
| colon_remove_from_inline_interval_L18_19_layer_out_restore | 36 | colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 30/36 | 32/36 | 0/36 | 1.2 | 72.832 | correct_prefix:32, word:2, explanation:2 |
| colon_remove_from_inline_L17_layer_out_restore | 36 | colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 30/36 | 33/36 | 0/36 | 1.2 | 75.510 | correct_prefix:33, explanation:2, word:1 |
| colon_remove_from_inline_L17_layer_input_restore | 36 | colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 30/36 | 32/36 | 0/36 | 1.4 | 75.493 | correct_prefix:32, word:2, explanation:2 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L18_19_layer_out_random | 36 | colon | to_original | interval | L18_19 |  | layer_out | random | 28/36 | 29/36 | 0/36 | 1.4 | 78.014 | correct_prefix:29, word:5, explanation:2 |
| colon_to_original_interval_L17_20_layer_out_random | 36 | colon | to_original | interval | L17_20 |  | layer_out | random | 28/36 | 29/36 | 0/36 | 1.4 | 85.930 | correct_prefix:29, word:5, explanation:2 |
| colon_remove_from_inline_interval_L18_19_layer_out_random | 36 | colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 28/36 | 28/36 | 0/36 | 1.6 | 70.089 | correct_prefix:28, word:4, explanation:4 |
| colon_remove_from_inline_interval_L17_20_layer_out_random | 36 | colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 28/36 | 30/36 | 0/36 | 1.6 | 75.493 | correct_prefix:30, explanation:4, word:2 |
| colon_remove_from_inline_interval_L17_20_mlp_out_random | 36 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 9/36 | 10/36 | 0/36 | 2.9 | 99.000 | word:23, correct_prefix:10, explanation:3 |
| colon_remove_from_inline_interval_L17_20_attn_out_random | 36 | colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 9/36 | 9/36 | 0/36 | 8.2 | 99.000 | word:22, correct_prefix:9, explanation:5 |
| colon_to_original_interval_L17_20_attn_out_random | 36 | colon | to_original | interval | L17_20 |  | attn_out | random | 7/36 | 8/36 | 0/36 | 4.0 | 99.000 | word:27, correct_prefix:8, explanation:1 |
| colon_to_original_interval_L17_20_mlp_out_random | 36 | colon | to_original | interval | L17_20 |  | mlp_out | random | 5/36 | 5/36 | 0/36 | 2.4 | 99.000 | word:31, correct_prefix:5 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_interval_L18_19_layer_out_reverse | 36 | colon | to_original | interval | L18_19 |  | layer_out | reverse | 25/36 | 26/36 | 0/36 | 1.6 | 91.174 | correct_prefix:26, word:10 |
| colon_to_original_interval_L17_20_layer_out_reverse | 36 | colon | to_original | interval | L17_20 |  | layer_out | reverse | 25/36 | 28/36 | 0/36 | 1.6 | 91.143 | correct_prefix:28, word:8 |
| colon_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 20/36 | 20/36 | 0/36 | 2.8 | 56.808 | correct_prefix:20, word:12, explanation:4 |
| colon_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 17/36 | 18/36 | 0/36 | 3.5 | 85.724 | correct_prefix:18, word:12, explanation:6 |
| colon_to_original_interval_L17_20_attn_out_reverse | 36 | colon | to_original | interval | L17_20 |  | attn_out | reverse | 10/36 | 12/36 | 0/36 | 4.1 | 64.398 | word:24, correct_prefix:12 |
| colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 6/36 | 6/36 | 0/36 | 4.3 | 99.000 | word:21, explanation:9, correct_prefix:6 |
| colon_to_original_interval_L17_20_mlp_out_reverse | 36 | colon | to_original | interval | L17_20 |  | mlp_out | reverse | 3/36 | 3/36 | 0/36 | 2.8 | 99.000 | word:33, correct_prefix:3 |
| colon_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 2/36 | 2/36 | 0/36 | 14.8 | 99.000 | word:29, explanation:5, correct_prefix:2 |

### answer_colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_L17_layer_input_restore | 36 | answer_colon | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.6 | 88.558 | correct_prefix:30, explanation:4, word:2 |
| answer_colon_to_original_interval_L18_19_layer_out_restore | 36 | answer_colon | to_original | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |
| answer_colon_to_original_interval_L17_20_layer_out_restore | 36 | answer_colon | to_original | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |
| answer_colon_to_original_L17_layer_out_restore | 36 | answer_colon | to_original | single_layer |  | 17 | layer_out | restore | 26/36 | 30/36 | 0/36 | 1.9 | 77.990 | correct_prefix:30, explanation:3, word:3 |
| answer_colon_to_original_interval_L17_20_mlp_out_restore | 36 | answer_colon | to_original | interval | L17_20 |  | mlp_out | restore | 3/36 | 3/36 | 0/36 | 4.0 | 99.000 | word:32, correct_prefix:3, explanation:1 |
| answer_colon_to_original_interval_L17_20_attn_out_restore | 36 | answer_colon | to_original | interval | L17_20 |  | attn_out | restore | 1/36 | 2/36 | 0/36 | 28.7 | 99.000 | word:28, explanation:6, correct_prefix:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_restore | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_restore | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_restore | 36 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |
| answer_colon_remove_from_inline_L17_layer_out_restore | 36 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |
| answer_colon_remove_from_inline_L17_layer_input_restore | 36 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 31/36 | 34/36 | 0/36 | 1.2 | 70.291 | correct_prefix:34, word:1, explanation:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L17_20_layer_out_random | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 27/36 | 29/36 | 0/36 | 1.7 | 59.517 | correct_prefix:29, explanation:4, word:3 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_random | 36 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 27/36 | 28/36 | 0/36 | 1.7 | 59.516 | correct_prefix:28, word:5, explanation:3 |
| answer_colon_to_original_interval_L17_20_layer_out_random | 36 | answer_colon | to_original | interval | L17_20 |  | layer_out | random | 26/36 | 28/36 | 0/36 | 1.6 | 75.332 | correct_prefix:28, word:8 |
| answer_colon_to_original_interval_L18_19_layer_out_random | 36 | answer_colon | to_original | interval | L18_19 |  | layer_out | random | 25/36 | 26/36 | 0/36 | 1.5 | 75.405 | correct_prefix:26, word:10 |
| answer_colon_to_original_interval_L17_20_attn_out_random | 36 | answer_colon | to_original | interval | L17_20 |  | attn_out | random | 9/36 | 11/36 | 0/36 | 13.9 | 96.337 | word:24, correct_prefix:11, explanation:1 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_random | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 4/36 | 4/36 | 0/36 | 15.7 | 96.316 | word:26, explanation:6, correct_prefix:4 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_random | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 3/36 | 3/36 | 0/36 | 4.6 | 83.174 | word:30, correct_prefix:3, explanation:3 |
| answer_colon_to_original_interval_L17_20_mlp_out_random | 36 | answer_colon | to_original | interval | L17_20 |  | mlp_out | random | 2/36 | 3/36 | 0/36 | 2.9 | 99.000 | word:33, correct_prefix:3 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_interval_L17_20_layer_out_reverse | 36 | answer_colon | to_original | interval | L17_20 |  | layer_out | reverse | 23/36 | 26/36 | 0/36 | 1.5 | 93.741 | correct_prefix:26, word:10 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 21/36 | 21/36 | 0/36 | 3.3 | 24.661 | correct_prefix:21, word:11, explanation:4 |
| answer_colon_to_original_interval_L18_19_layer_out_reverse | 36 | answer_colon | to_original | interval | L18_19 |  | layer_out | reverse | 16/36 | 19/36 | 0/36 | 1.7 | 96.380 | correct_prefix:19, word:17 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 12/36 | 13/36 | 0/36 | 4.4 | 43.346 | word:16, correct_prefix:13, explanation:7 |
| answer_colon_to_original_interval_L17_20_attn_out_reverse | 36 | answer_colon | to_original | interval | L17_20 |  | attn_out | reverse | 8/36 | 11/36 | 0/36 | 9.3 | 61.461 | word:25, correct_prefix:11 |
| answer_colon_to_original_interval_L17_20_mlp_out_reverse | 36 | answer_colon | to_original | interval | L17_20 |  | mlp_out | reverse | 6/36 | 6/36 | 0/36 | 2.5 | 93.729 | word:30, correct_prefix:6 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/36 | 1/36 | 0/36 | 24.7 | 96.289 | word:24, explanation:11, correct_prefix:1 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/36 | 0/36 | 0/36 | 7.4 | 96.317 | word:28, explanation:8 |

### answer_label_aligned

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_L17_layer_input_restore | 36 | answer_label_aligned | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.6 | 88.558 | correct_prefix:30, explanation:4, word:2 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_restore | 36 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_restore | 36 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |
| answer_label_aligned_to_original_L17_layer_out_restore | 36 | answer_label_aligned | to_original | single_layer |  | 17 | layer_out | restore | 26/36 | 30/36 | 0/36 | 1.9 | 77.990 | correct_prefix:30, explanation:3, word:3 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_restore | 36 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | restore | 3/36 | 3/36 | 0/36 | 4.0 | 99.000 | word:32, correct_prefix:3, explanation:1 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_restore | 36 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | restore | 1/36 | 2/36 | 0/36 | 28.7 | 99.000 | word:28, explanation:6, correct_prefix:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | restore | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_restore | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_restore | 36 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |
| answer_label_aligned_remove_from_inline_L17_layer_out_restore | 36 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |
| answer_label_aligned_remove_from_inline_L17_layer_input_restore | 36 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_input | restore | 31/36 | 34/36 | 0/36 | 1.2 | 70.291 | correct_prefix:34, word:1, explanation:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random | 36 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | random | 27/36 | 28/36 | 0/36 | 1.8 | 62.267 | correct_prefix:28, word:5, explanation:3 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_random | 36 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | random | 26/36 | 28/36 | 0/36 | 1.4 | 72.669 | correct_prefix:28, word:7, explanation:1 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_random | 36 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | random | 25/36 | 26/36 | 0/36 | 1.5 | 75.416 | correct_prefix:26, word:10 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_random | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | random | 24/36 | 27/36 | 0/36 | 1.8 | 59.550 | correct_prefix:27, word:5, explanation:4 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_random | 36 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | random | 9/36 | 10/36 | 0/36 | 10.0 | 96.384 | word:25, correct_prefix:10, explanation:1 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_random | 36 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | random | 5/36 | 5/36 | 0/36 | 2.4 | 99.000 | word:30, correct_prefix:5, explanation:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_random | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | random | 5/36 | 5/36 | 0/36 | 3.1 | 96.352 | word:30, correct_prefix:5, explanation:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_random | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | random | 5/36 | 5/36 | 0/36 | 13.8 | 88.222 | word:23, explanation:8, correct_prefix:5 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_interval_L17_20_layer_out_reverse | 36 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | reverse | 23/36 | 26/36 | 0/36 | 1.5 | 93.741 | correct_prefix:26, word:10 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 21/36 | 21/36 | 0/36 | 3.3 | 24.661 | correct_prefix:21, word:11, explanation:4 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_reverse | 36 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | reverse | 16/36 | 19/36 | 0/36 | 1.7 | 96.380 | correct_prefix:19, word:17 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 12/36 | 13/36 | 0/36 | 4.4 | 43.346 | word:16, correct_prefix:13, explanation:7 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_reverse | 36 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | reverse | 8/36 | 11/36 | 0/36 | 9.3 | 61.461 | word:25, correct_prefix:11 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_reverse | 36 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | reverse | 6/36 | 6/36 | 0/36 | 2.5 | 93.729 | word:30, correct_prefix:6 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/36 | 1/36 | 0/36 | 24.7 | 96.289 | word:24, explanation:11, correct_prefix:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/36 | 0/36 | 0/36 | 7.4 | 96.317 | word:28, explanation:8 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_L17_layer_input_restore | 36 | separator | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.6 | 88.558 | correct_prefix:30, explanation:4, word:2 |
| separator_to_original_interval_L18_19_layer_out_restore | 36 | separator | to_original | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |
| separator_to_original_interval_L17_20_layer_out_restore | 36 | separator | to_original | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |
| separator_to_original_L17_layer_out_restore | 36 | separator | to_original | single_layer |  | 17 | layer_out | restore | 26/36 | 30/36 | 0/36 | 1.9 | 77.990 | correct_prefix:30, explanation:3, word:3 |
| separator_to_original_interval_L17_20_mlp_out_restore | 36 | separator | to_original | interval | L17_20 |  | mlp_out | restore | 3/36 | 3/36 | 0/36 | 4.0 | 99.000 | word:32, correct_prefix:3, explanation:1 |
| separator_to_original_interval_L17_20_attn_out_restore | 36 | separator | to_original | interval | L17_20 |  | attn_out | restore | 1/36 | 2/36 | 0/36 | 28.7 | 99.000 | word:28, explanation:6, correct_prefix:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | attn_out | restore | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 36 | separator | remove_from_inline | interval | L17_20 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 36 | separator | remove_from_inline | interval | L18_19 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |
| separator_remove_from_inline_L17_layer_out_restore | 36 | separator | remove_from_inline | single_layer |  | 17 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |
| separator_remove_from_inline_L17_layer_input_restore | 36 | separator | remove_from_inline | single_layer |  | 17 | layer_input | restore | 31/36 | 34/36 | 0/36 | 1.2 | 70.291 | correct_prefix:34, word:1, explanation:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_interval_L17_20_layer_out_random | 36 | separator | to_original | interval | L17_20 |  | layer_out | random | 26/36 | 26/36 | 0/36 | 1.5 | 72.797 | correct_prefix:26, word:10 |
| separator_remove_from_inline_interval_L17_20_layer_out_random | 36 | separator | remove_from_inline | interval | L17_20 |  | layer_out | random | 26/36 | 28/36 | 0/36 | 1.6 | 62.243 | correct_prefix:28, word:5, explanation:3 |
| separator_to_original_interval_L18_19_layer_out_random | 36 | separator | to_original | interval | L18_19 |  | layer_out | random | 25/36 | 26/36 | 0/36 | 1.5 | 77.970 | correct_prefix:26, word:9, explanation:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_random | 36 | separator | remove_from_inline | interval | L18_19 |  | layer_out | random | 25/36 | 27/36 | 0/36 | 1.7 | 51.441 | correct_prefix:27, word:5, explanation:4 |
| separator_to_original_interval_L17_20_mlp_out_random | 36 | separator | to_original | interval | L17_20 |  | mlp_out | random | 7/36 | 8/36 | 0/36 | 2.6 | 93.731 | word:27, correct_prefix:8, explanation:1 |
| separator_remove_from_inline_interval_L17_20_mlp_out_random | 36 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | random | 7/36 | 7/36 | 0/36 | 3.7 | 91.123 | word:24, correct_prefix:7, explanation:5 |
| separator_to_original_interval_L17_20_attn_out_random | 36 | separator | to_original | interval | L17_20 |  | attn_out | random | 7/36 | 7/36 | 0/36 | 8.1 | 96.285 | word:27, correct_prefix:7, explanation:2 |
| separator_remove_from_inline_interval_L17_20_attn_out_random | 36 | separator | remove_from_inline | interval | L17_20 |  | attn_out | random | 4/36 | 4/36 | 0/36 | 9.9 | 93.694 | word:25, explanation:7, correct_prefix:4 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_interval_L17_20_layer_out_reverse | 36 | separator | to_original | interval | L17_20 |  | layer_out | reverse | 23/36 | 26/36 | 0/36 | 1.5 | 93.741 | correct_prefix:26, word:10 |
| separator_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | separator | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 21/36 | 21/36 | 0/36 | 3.3 | 24.661 | correct_prefix:21, word:11, explanation:4 |
| separator_to_original_interval_L18_19_layer_out_reverse | 36 | separator | to_original | interval | L18_19 |  | layer_out | reverse | 16/36 | 19/36 | 0/36 | 1.7 | 96.380 | correct_prefix:19, word:17 |
| separator_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | separator | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 12/36 | 13/36 | 0/36 | 4.4 | 43.346 | word:16, correct_prefix:13, explanation:7 |
| separator_to_original_interval_L17_20_attn_out_reverse | 36 | separator | to_original | interval | L17_20 |  | attn_out | reverse | 8/36 | 11/36 | 0/36 | 9.3 | 61.461 | word:25, correct_prefix:11 |
| separator_to_original_interval_L17_20_mlp_out_reverse | 36 | separator | to_original | interval | L17_20 |  | mlp_out | reverse | 6/36 | 6/36 | 0/36 | 2.5 | 93.729 | word:30, correct_prefix:6 |
| separator_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | separator | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/36 | 1/36 | 0/36 | 24.7 | 96.289 | word:24, explanation:11, correct_prefix:1 |
| separator_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/36 | 0/36 | 0/36 | 7.4 | 96.317 | word:28, explanation:8 |

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L18_19_layer_out_restore | 36 | prompt_last | to_original | interval | L18_19 |  | layer_out | restore | 25/36 | 28/36 | 0/36 | 1.6 | 75.367 | correct_prefix:28, word:6, explanation:2 |
| prompt_last_to_original_L17_layer_input_restore | 36 | prompt_last | to_original | single_layer |  | 17 | layer_input | restore | 25/36 | 28/36 | 0/36 | 1.6 | 77.969 | correct_prefix:28, word:6, explanation:2 |
| prompt_last_to_original_L17_layer_out_restore | 36 | prompt_last | to_original | single_layer |  | 17 | layer_out | restore | 24/36 | 28/36 | 0/36 | 1.6 | 75.314 | correct_prefix:28, word:6, explanation:2 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 36 | prompt_last | to_original | interval | L17_20 |  | layer_out | restore | 24/36 | 27/36 | 0/36 | 1.6 | 75.378 | correct_prefix:27, word:7, explanation:2 |
| prompt_last_to_original_interval_L17_20_mlp_out_restore | 36 | prompt_last | to_original | interval | L17_20 |  | mlp_out | restore | 8/36 | 12/36 | 0/36 | 2.2 | 99.000 | word:23, correct_prefix:12, explanation:1 |
| prompt_last_to_original_interval_L17_20_attn_out_restore | 36 | prompt_last | to_original | interval | L17_20 |  | attn_out | restore | 2/36 | 2/36 | 0/36 | 10.1 | 99.000 | word:34, correct_prefix:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | restore | 16/36 | 19/36 | 0/36 | 3.3 | 96.380 | correct_prefix:19, word:16, explanation:1 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 19/36 | 21/36 | 0/36 | 1.8 | 99.000 | correct_prefix:21, word:14, explanation:1 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | restore | 30/36 | 33/36 | 0/36 | 1.2 | 64.969 | correct_prefix:33, word:2, explanation:1 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | restore | 30/36 | 32/36 | 0/36 | 1.2 | 72.832 | correct_prefix:32, word:2, explanation:2 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | restore | 30/36 | 33/36 | 0/36 | 1.2 | 75.510 | correct_prefix:33, explanation:2, word:1 |
| prompt_last_remove_from_inline_L17_layer_input_restore | 36 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_input | restore | 30/36 | 32/36 | 0/36 | 1.4 | 75.493 | correct_prefix:32, word:2, explanation:2 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L17_20_layer_out_random | 36 | prompt_last | to_original | interval | L17_20 |  | layer_out | random | 27/36 | 29/36 | 0/36 | 1.5 | 80.633 | correct_prefix:29, word:5, explanation:2 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_random | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | random | 27/36 | 27/36 | 0/36 | 1.7 | 70.061 | correct_prefix:27, word:5, explanation:4 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_random | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | random | 27/36 | 28/36 | 0/36 | 1.7 | 77.994 | correct_prefix:28, word:4, explanation:4 |
| prompt_last_to_original_interval_L18_19_layer_out_random | 36 | prompt_last | to_original | interval | L18_19 |  | layer_out | random | 25/36 | 28/36 | 0/36 | 1.5 | 80.578 | correct_prefix:28, word:7, explanation:1 |
| prompt_last_to_original_interval_L17_20_attn_out_random | 36 | prompt_last | to_original | interval | L17_20 |  | attn_out | random | 12/36 | 13/36 | 0/36 | 3.7 | 99.000 | word:22, correct_prefix:13, explanation:1 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_random | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | random | 10/36 | 13/36 | 0/36 | 2.8 | 99.000 | word:18, correct_prefix:13, explanation:5 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_random | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | random | 9/36 | 9/36 | 0/36 | 6.2 | 99.000 | word:24, correct_prefix:9, explanation:3 |
| prompt_last_to_original_interval_L17_20_mlp_out_random | 36 | prompt_last | to_original | interval | L17_20 |  | mlp_out | random | 5/36 | 7/36 | 0/36 | 2.5 | 99.000 | word:29, correct_prefix:7 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_interval_L18_19_layer_out_reverse | 36 | prompt_last | to_original | interval | L18_19 |  | layer_out | reverse | 25/36 | 26/36 | 0/36 | 1.6 | 91.174 | correct_prefix:26, word:10 |
| prompt_last_to_original_interval_L17_20_layer_out_reverse | 36 | prompt_last | to_original | interval | L17_20 |  | layer_out | reverse | 25/36 | 28/36 | 0/36 | 1.6 | 91.143 | correct_prefix:28, word:8 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 20/36 | 20/36 | 0/36 | 2.8 | 56.808 | correct_prefix:20, word:12, explanation:4 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 17/36 | 18/36 | 0/36 | 3.5 | 85.724 | correct_prefix:18, word:12, explanation:6 |
| prompt_last_to_original_interval_L17_20_attn_out_reverse | 36 | prompt_last | to_original | interval | L17_20 |  | attn_out | reverse | 10/36 | 12/36 | 0/36 | 4.1 | 64.398 | word:24, correct_prefix:12 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 6/36 | 6/36 | 0/36 | 4.3 | 99.000 | word:21, explanation:9, correct_prefix:6 |
| prompt_last_to_original_interval_L17_20_mlp_out_reverse | 36 | prompt_last | to_original | interval | L17_20 |  | mlp_out | reverse | 3/36 | 3/36 | 0/36 | 2.8 | 99.000 | word:33, correct_prefix:3 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 2/36 | 2/36 | 0/36 | 14.8 | 99.000 | word:29, explanation:5, correct_prefix:2 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 36 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 36 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| question_mark_answer_to_original_L17_layer_input_restore | 36 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| question_mark_answer_to_original_L17_layer_out_restore | 36 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| question_mark_answer_to_original_interval_L17_20_attn_out_restore | 36 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | restore | 2/36 | 3/36 | 0/36 | 22.9 | 99.000 | explanation:17, word:16, correct_prefix:3 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_restore | 36 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | restore | 0/36 | 0/36 | 0/36 | 3.8 | 99.000 | word:36 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 2/36 | 2/36 | 0/36 | 3.4 | 77.670 | word:34, correct_prefix:2 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | restore | 14/36 | 17/36 | 0/36 | 9.0 | 96.366 | correct_prefix:17, word:17, explanation:2 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 36 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 36 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_random | 36 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | random | 24/36 | 24/36 | 0/36 | 1.8 | 62.111 | correct_prefix:24, word:9, explanation:3 |
| question_mark_answer_to_original_interval_L17_20_layer_out_random | 36 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | random | 23/36 | 24/36 | 0/36 | 1.6 | 75.267 | correct_prefix:24, word:11, explanation:1 |
| question_mark_answer_to_original_interval_L18_19_layer_out_random | 36 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | random | 23/36 | 25/36 | 0/36 | 1.7 | 80.536 | correct_prefix:25, word:11 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_random | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | random | 19/36 | 21/36 | 0/36 | 2.3 | 69.999 | correct_prefix:21, word:11, explanation:4 |
| question_mark_answer_to_original_interval_L17_20_attn_out_random | 36 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | random | 9/36 | 9/36 | 0/36 | 7.1 | 93.740 | word:25, correct_prefix:9, explanation:2 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_random | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | random | 6/36 | 6/36 | 0/36 | 12.6 | 99.000 | word:23, explanation:7, correct_prefix:6 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_random | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | random | 2/36 | 2/36 | 0/36 | 4.4 | 93.704 | word:33, correct_prefix:2, explanation:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_random | 36 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | random | 1/36 | 1/36 | 0/36 | 3.4 | 99.000 | word:35, correct_prefix:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_to_original_interval_L17_20_attn_out_reverse | 36 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | reverse | 9/36 | 11/36 | 0/36 | 10.2 | 90.964 | word:25, correct_prefix:11 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 7/36 | 9/36 | 0/36 | 6.9 | 59.171 | word:21, correct_prefix:9, explanation:6 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 6/36 | 7/36 | 0/36 | 9.6 | 40.152 | word:22, correct_prefix:7, explanation:7 |
| question_mark_answer_to_original_interval_L17_20_layer_out_reverse | 36 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | reverse | 5/36 | 6/36 | 0/36 | 2.4 | 99.000 | word:30, correct_prefix:6 |
| question_mark_answer_to_original_interval_L18_19_layer_out_reverse | 36 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | reverse | 2/36 | 2/36 | 0/36 | 2.5 | 99.000 | word:34, correct_prefix:2 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_reverse | 36 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | reverse | 1/36 | 1/36 | 0/36 | 3.4 | 99.000 | word:35, correct_prefix:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/36 | 0/36 | 0/36 | 7.8 | 99.000 | word:36 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 0/36 | 0/36 | 0/36 | 25.2 | 99.000 | word:23, explanation:13 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_to_original_interval_L17_20_layer_out_restore | 36 | relation_tail | to_original | interval | L17_20 |  | layer_out | restore | 29/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| relation_tail_to_original_interval_L18_19_layer_out_restore | 36 | relation_tail | to_original | interval | L18_19 |  | layer_out | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| relation_tail_to_original_L17_layer_input_restore | 36 | relation_tail | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| relation_tail_to_original_L17_layer_out_restore | 36 | relation_tail | to_original | single_layer |  | 17 | layer_out | restore | 28/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |
| relation_tail_to_original_interval_L17_20_attn_out_restore | 36 | relation_tail | to_original | interval | L17_20 |  | attn_out | restore | 3/36 | 6/36 | 0/36 | 18.2 | 99.000 | explanation:17, word:13, correct_prefix:6 |
| relation_tail_to_original_interval_L17_20_mlp_out_restore | 36 | relation_tail | to_original | interval | L17_20 |  | mlp_out | restore | 0/36 | 0/36 | 0/36 | 3.9 | 99.000 | word:36 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 0/36 | 0/36 | 0/36 | 3.5 | 83.009 | word:36 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | restore | 12/36 | 15/36 | 0/36 | 8.8 | 93.736 | word:21, correct_prefix:15 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | restore | 28/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| relation_tail_remove_from_inline_L17_layer_input_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_input | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 36 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | restore | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L18_19_layer_out_random | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | random | 24/36 | 25/36 | 0/36 | 1.9 | 59.424 | correct_prefix:25, word:8, explanation:3 |
| relation_tail_to_original_interval_L17_20_layer_out_random | 36 | relation_tail | to_original | interval | L17_20 |  | layer_out | random | 23/36 | 23/36 | 0/36 | 1.6 | 75.279 | correct_prefix:23, word:13 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_random | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | random | 23/36 | 23/36 | 0/36 | 1.9 | 59.457 | correct_prefix:23, word:10, explanation:3 |
| relation_tail_to_original_interval_L18_19_layer_out_random | 36 | relation_tail | to_original | interval | L18_19 |  | layer_out | random | 22/36 | 24/36 | 0/36 | 1.6 | 69.957 | correct_prefix:24, word:10, explanation:2 |
| relation_tail_to_original_interval_L17_20_attn_out_random | 36 | relation_tail | to_original | interval | L17_20 |  | attn_out | random | 8/36 | 11/36 | 0/36 | 9.3 | 99.000 | word:20, correct_prefix:11, explanation:5 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_random | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | random | 3/36 | 7/36 | 0/36 | 22.4 | 93.709 | word:22, explanation:7, correct_prefix:7 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_random | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | random | 1/36 | 1/36 | 0/36 | 6.2 | 93.617 | word:34, explanation:1, correct_prefix:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_random | 36 | relation_tail | to_original | interval | L17_20 |  | mlp_out | random | 0/36 | 0/36 | 0/36 | 3.6 | 99.000 | word:36 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_to_original_interval_L17_20_attn_out_reverse | 36 | relation_tail | to_original | interval | L17_20 |  | attn_out | reverse | 10/36 | 12/36 | 0/36 | 15.5 | 85.476 | word:24, correct_prefix:12 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_reverse | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 7/36 | 9/36 | 0/36 | 6.9 | 59.171 | word:21, correct_prefix:9, explanation:6 |
| relation_tail_to_original_interval_L17_20_layer_out_reverse | 36 | relation_tail | to_original | interval | L17_20 |  | layer_out | reverse | 6/36 | 6/36 | 0/36 | 2.4 | 99.000 | word:30, correct_prefix:6 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_reverse | 36 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 6/36 | 7/36 | 0/36 | 9.6 | 40.152 | word:22, correct_prefix:7, explanation:7 |
| relation_tail_to_original_interval_L18_19_layer_out_reverse | 36 | relation_tail | to_original | interval | L18_19 |  | layer_out | reverse | 2/36 | 2/36 | 0/36 | 2.5 | 99.000 | word:34, correct_prefix:2 |
| relation_tail_to_original_interval_L17_20_mlp_out_reverse | 36 | relation_tail | to_original | interval | L17_20 |  | mlp_out | reverse | 1/36 | 1/36 | 0/36 | 3.4 | 99.000 | word:35, correct_prefix:1 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_reverse | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 0/36 | 0/36 | 0/36 | 9.0 | 99.000 | word:36 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_reverse | 36 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 0/36 | 0/36 | 0/36 | 24.2 | 99.000 | word:23, explanation:13 |

### Global Top Notes

- Top sufficiency: answer_word_to_original_interval_L18_19_layer_out_restore exact=30/36 newline=0/36 rank=1.5; answer_word_to_original_L17_layer_input_restore exact=29/36 newline=0/36 rank=1.5; answer_word_to_original_L17_layer_out_restore exact=29/36 newline=0/36 rank=1.5; relation_tail_to_original_interval_L17_20_layer_out_restore exact=29/36 newline=0/36 rank=1.5; question_mark_answer_to_original_interval_L17_20_layer_out_restore exact=28/36 newline=0/36 rank=1.5; question_mark_answer_to_original_interval_L18_19_layer_out_restore exact=28/36 newline=0/36 rank=1.5; question_mark_answer_to_original_L17_layer_input_restore exact=28/36 newline=0/36 rank=1.5; question_mark_answer_to_original_L17_layer_out_restore exact=28/36 newline=0/36 rank=1.5; relation_tail_to_original_interval_L18_19_layer_out_restore exact=28/36 newline=0/36 rank=1.5; relation_tail_to_original_L17_layer_input_restore exact=28/36 newline=0/36 rank=1.5
- Top necessity/remove: relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore exact=0/36 newline=0/36 rank=3.5; question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore exact=2/36 newline=0/36 rank=3.4; answer_colon_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3; answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3; separator_remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36 rank=2.3; relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=12/36 newline=0/36 rank=8.8; question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=14/36 newline=0/36 rank=9.0; colon_remove_from_inline_interval_L17_20_attn_out_restore exact=16/36 newline=0/36 rank=3.3; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=16/36 newline=0/36 rank=3.3; answer_colon_remove_from_inline_interval_L17_20_attn_out_restore exact=17/36 newline=0/36 rank=4.6

## deepseek7b

- raw_cases: 320 / target_seen: 48 / cases_written: 48 / mode_rows: 10848
- positions: `['answer_word', 'colon', 'answer_colon', 'answer_label_aligned', 'separator', 'prompt_last', 'question_mark_answer', 'relation_tail']`
- filtered: `{'not_target': 88, 'position_missing': 0, 'position_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 1}` / total_time_min: 25.50

### Baselines

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| original | 48 |  |  | baseline |  |  |  |  | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| inline | 48 |  |  | baseline |  |  |  |  | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |

### Position Control Overview

| position | best restore sufficiency | best restore necessity/remove | best random | best reverse |
|---|---|---|---|---|
| answer_word | answer_word_to_original_interval_L17_20_layer_out_restore exact=26/48 newline=21/48 rank=2.6 | answer_word_remove_from_inline_L17_layer_input_restore exact=43/48 newline=0/48 rank=1.1 | answer_word_remove_from_inline_interval_L17_20_mlp_out_random exact=48/48 newline=0/48 rank=1.0 | answer_word_remove_from_inline_interval_L17_20_attn_out_reverse exact=48/48 newline=0/48 rank=1.0 |
| colon | colon_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.1 | colon_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=2/48 rank=2.7 | colon_remove_from_inline_interval_L18_19_layer_out_random exact=46/48 newline=0/48 rank=1.0 | colon_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.0 |
| answer_colon | answer_colon_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0 | answer_colon_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4 | answer_colon_remove_from_inline_interval_L18_19_layer_out_random exact=44/48 newline=1/48 rank=1.1 | answer_colon_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.0 |
| answer_label_aligned | answer_label_aligned_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0 | answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4 | answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random exact=45/48 newline=0/48 rank=1.0 | answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.0 |
| separator | separator_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0 | separator_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4 | separator_remove_from_inline_interval_L18_19_layer_out_random exact=46/48 newline=0/48 rank=1.0 | separator_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.0 |
| prompt_last | prompt_last_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.1 | prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=2/48 rank=2.7 | prompt_last_remove_from_inline_interval_L18_19_layer_out_random exact=45/48 newline=0/48 rank=1.0 | prompt_last_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.0 |
| question_mark_answer | question_mark_answer_to_original_interval_L18_19_layer_out_restore exact=45/48 newline=0/48 rank=1.0 | question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=0/48 rank=2.6 | question_mark_answer_remove_from_inline_interval_L18_19_layer_out_random exact=47/48 newline=0/48 rank=1.0 | question_mark_answer_remove_from_inline_interval_L18_19_layer_out_reverse exact=45/48 newline=0/48 rank=1.1 |
| relation_tail | relation_tail_to_original_interval_L18_19_layer_out_restore exact=45/48 newline=0/48 rank=1.0 | relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=1/48 newline=0/48 rank=2.7 | relation_tail_remove_from_inline_interval_L18_19_layer_out_random exact=44/48 newline=0/48 rank=1.1 | relation_tail_remove_from_inline_interval_L18_19_layer_out_reverse exact=44/48 newline=0/48 rank=1.1 |

### answer_word

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_to_original_interval_L17_20_layer_out_restore | 48 | answer_word | to_original | interval | L17_20 |  | layer_out | restore | 26/48 | 26/48 | 21/48 | 2.6 | 0.035 | correct_prefix:26, newline:21, word:1 |
| answer_word_to_original_L17_layer_input_restore | 48 | answer_word | to_original | single_layer |  | 17 | layer_input | restore | 21/48 | 20/48 | 27/48 | 3.5 | -0.505 | newline:27, correct_prefix:20, word:1 |
| answer_word_to_original_interval_L17_20_mlp_out_restore | 48 | answer_word | to_original | interval | L17_20 |  | mlp_out | restore | 20/48 | 23/48 | 12/48 | 2.9 | 0.163 | correct_prefix:23, newline:12, word:7, space:6 |
| answer_word_to_original_interval_L18_19_layer_out_restore | 48 | answer_word | to_original | interval | L18_19 |  | layer_out | restore | 20/48 | 18/48 | 25/48 | 3.4 | -0.310 | newline:25, correct_prefix:18, word:3, space:2 |
| answer_word_to_original_L17_layer_out_restore | 48 | answer_word | to_original | single_layer |  | 17 | layer_out | restore | 18/48 | 16/48 | 30/48 | 3.9 | -0.603 | newline:30, correct_prefix:16, word:1, space:1 |
| answer_word_to_original_interval_L17_20_attn_out_restore | 48 | answer_word | to_original | interval | L17_20 |  | attn_out | restore | 12/48 | 11/48 | 36/48 | 4.8 | -1.374 | newline:36, correct_prefix:11, word:1 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_remove_from_inline_L17_layer_input_restore | 48 | answer_word | remove_from_inline | single_layer |  | 17 | layer_input | restore | 43/48 | 45/48 | 0/48 | 1.1 | 2.087 | correct_prefix:45, space:3 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_restore | 48 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | restore | 43/48 | 44/48 | 0/48 | 1.1 | 1.987 | correct_prefix:44, space:4 |
| answer_word_remove_from_inline_L17_layer_out_restore | 48 | answer_word | remove_from_inline | single_layer |  | 17 | layer_out | restore | 43/48 | 43/48 | 0/48 | 1.1 | 2.040 | correct_prefix:43, space:5 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_restore | 48 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.156 | correct_prefix:45, space:3 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 47/48 | 48/48 | 0/48 | 1.0 | 2.478 | correct_prefix:48 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_restore | 48 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | restore | 48/48 | 48/48 | 0/48 | 1.0 | 2.339 | correct_prefix:48 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_remove_from_inline_interval_L17_20_mlp_out_random | 48 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | random | 48/48 | 48/48 | 0/48 | 1.0 | 2.846 | correct_prefix:48 |
| answer_word_remove_from_inline_interval_L17_20_attn_out_random | 48 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | random | 48/48 | 47/48 | 1/48 | 1.0 | 1.934 | correct_prefix:47, newline:1 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_random | 48 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | random | 46/48 | 48/48 | 0/48 | 1.0 | 2.629 | correct_prefix:48 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_random | 48 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | random | 45/48 | 45/48 | 0/48 | 1.0 | 2.526 | correct_prefix:45, space:3 |
| answer_word_to_original_interval_L17_20_mlp_out_random | 48 | answer_word | to_original | interval | L17_20 |  | mlp_out | random | 21/48 | 20/48 | 26/48 | 3.4 | -0.145 | newline:26, correct_prefix:20, word:1, space:1 |
| answer_word_to_original_interval_L17_20_layer_out_random | 48 | answer_word | to_original | interval | L17_20 |  | layer_out | random | 18/48 | 18/48 | 28/48 | 4.5 | -0.651 | newline:28, correct_prefix:18, word:1, space:1 |
| answer_word_to_original_interval_L17_20_attn_out_random | 48 | answer_word | to_original | interval | L17_20 |  | attn_out | random | 14/48 | 14/48 | 30/48 | 4.6 | -1.111 | newline:30, correct_prefix:14, word:4 |
| answer_word_to_original_interval_L18_19_layer_out_random | 48 | answer_word | to_original | interval | L18_19 |  | layer_out | random | 13/48 | 13/48 | 31/48 | 6.4 | -1.090 | newline:31, correct_prefix:13, word:2, space:2 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_word_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | answer_word | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 48/48 | 48/48 | 0/48 | 1.0 | 1.714 | correct_prefix:48 |
| answer_word_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | answer_word | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 46/48 | 48/48 | 0/48 | 1.0 | 3.146 | correct_prefix:48 |
| answer_word_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | answer_word | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 46/48 | 47/48 | 0/48 | 1.0 | 2.939 | correct_prefix:47, space:1 |
| answer_word_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | answer_word | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 46/48 | 47/48 | 0/48 | 1.0 | 2.863 | correct_prefix:47, space:1 |
| answer_word_to_original_interval_L17_20_mlp_out_reverse | 48 | answer_word | to_original | interval | L17_20 |  | mlp_out | reverse | 22/48 | 22/48 | 25/48 | 3.8 | -0.423 | newline:25, correct_prefix:22, word:1 |
| answer_word_to_original_interval_L17_20_layer_out_reverse | 48 | answer_word | to_original | interval | L17_20 |  | layer_out | reverse | 13/48 | 11/48 | 35/48 | 8.6 | -1.411 | newline:35, correct_prefix:11, word:1, explanation:1 |
| answer_word_to_original_interval_L17_20_attn_out_reverse | 48 | answer_word | to_original | interval | L17_20 |  | attn_out | reverse | 12/48 | 12/48 | 19/48 | 5.0 | -0.995 | newline:19, word:17, correct_prefix:12 |
| answer_word_to_original_interval_L18_19_layer_out_reverse | 48 | answer_word | to_original | interval | L18_19 |  | layer_out | reverse | 10/48 | 8/48 | 35/48 | 12.2 | -1.962 | newline:35, correct_prefix:8, word:3, explanation:2 |

### colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_to_original_L17_layer_out_restore | 48 | colon | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.1 | 1.667 | correct_prefix:46, space:2 |
| colon_to_original_interval_L18_19_layer_out_restore | 48 | colon | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 1.868 | correct_prefix:45, space:3 |
| colon_to_original_L17_layer_input_restore | 48 | colon | to_original | single_layer |  | 17 | layer_input | restore | 44/48 | 43/48 | 3/48 | 1.2 | 1.422 | correct_prefix:43, newline:3, space:2 |
| colon_to_original_interval_L17_20_layer_out_restore | 48 | colon | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 45/48 | 0/48 | 1.1 | 2.102 | correct_prefix:45, space:3 |
| colon_to_original_interval_L17_20_mlp_out_restore | 48 | colon | to_original | interval | L17_20 |  | mlp_out | restore | 31/48 | 32/48 | 14/48 | 1.6 | 0.570 | correct_prefix:32, newline:14, word:1, space:1 |
| colon_to_original_interval_L17_20_attn_out_restore | 48 | colon | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 22/48 | 19.1 | -3.509 | word:26, newline:22 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_remove_from_inline_interval_L17_20_attn_out_restore | 48 | colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 5/48 | 7/48 | 2/48 | 2.7 | 0.118 | word:39, correct_prefix:7, newline:2 |
| colon_remove_from_inline_interval_L17_20_layer_out_restore | 48 | colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 14/48 | 29/48 | 3.2 | -0.624 | newline:29, correct_prefix:14, space:3, word:2 |
| colon_remove_from_inline_interval_L18_19_layer_out_restore | 48 | colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 20/48 | 19/48 | 23/48 | 2.2 | -0.073 | newline:23, correct_prefix:19, space:4, word:2 |
| colon_remove_from_inline_L17_layer_out_restore | 48 | colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 23/48 | 23/48 | 15/48 | 2.0 | 0.337 | correct_prefix:23, newline:15, space:10 |
| colon_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 25/48 | 27/48 | 21/48 | 1.7 | 0.276 | correct_prefix:27, newline:21 |
| colon_remove_from_inline_L17_layer_input_restore | 48 | colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 25/48 | 26/48 | 9/48 | 1.8 | 0.616 | correct_prefix:26, space:11, newline:9, word:2 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_remove_from_inline_interval_L18_19_layer_out_random | 48 | colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 46/48 | 46/48 | 0/48 | 1.0 | 2.195 | correct_prefix:46, space:2 |
| colon_remove_from_inline_interval_L17_20_layer_out_random | 48 | colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 43/48 | 45/48 | 0/48 | 1.1 | 2.320 | correct_prefix:45, space:3 |
| colon_remove_from_inline_interval_L17_20_mlp_out_random | 48 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 41/48 | 44/48 | 2/48 | 1.2 | 1.738 | correct_prefix:44, newline:2, space:1, word:1 |
| colon_to_original_interval_L17_20_mlp_out_random | 48 | colon | to_original | interval | L17_20 |  | mlp_out | random | 12/48 | 12/48 | 30/48 | 9.0 | -1.777 | newline:30, correct_prefix:12, space:3, word:3 |
| colon_to_original_interval_L17_20_layer_out_random | 48 | colon | to_original | interval | L17_20 |  | layer_out | random | 10/48 | 11/48 | 26/48 | 10.2 | -1.551 | newline:26, correct_prefix:11, space:6, word:3, explanation:2 |
| colon_to_original_interval_L18_19_layer_out_random | 48 | colon | to_original | interval | L18_19 |  | layer_out | random | 9/48 | 9/48 | 30/48 | 8.5 | -1.639 | newline:30, correct_prefix:9, word:4, space:3, explanation:2 |
| colon_remove_from_inline_interval_L17_20_attn_out_random | 48 | colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 1/48 | 5.3 | -0.754 | word:45, newline:1, correct_prefix:1, explanation:1 |
| colon_to_original_interval_L17_20_attn_out_random | 48 | colon | to_original | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 10/48 | 31.2 | -2.899 | word:35, newline:10, space:1, correct_prefix:1, explanation:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| colon_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 46/48 | 0/48 | 1.0 | 3.530 | correct_prefix:46, space:2 |
| colon_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 43/48 | 46/48 | 0/48 | 1.0 | 4.089 | correct_prefix:46, space:2 |
| colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 33/48 | 36/48 | 0/48 | 1.4 | 10.859 | correct_prefix:36, space:9, word:2, number:1 |
| colon_to_original_interval_L17_20_mlp_out_reverse | 48 | colon | to_original | interval | L17_20 |  | mlp_out | reverse | 5/48 | 5/48 | 34/48 | 29.3 | -3.303 | newline:34, word:5, correct_prefix:5, space:4 |
| colon_to_original_interval_L17_20_attn_out_reverse | 48 | colon | to_original | interval | L17_20 |  | attn_out | reverse | 2/48 | 2/48 | 0/48 | 23.7 | -2.172 | word:44, explanation:2, correct_prefix:2 |
| colon_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 2/48 | 12.6 | -1.409 | word:43, newline:2, explanation:2, correct_prefix:1 |
| colon_to_original_interval_L18_19_layer_out_reverse | 48 | colon | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 1/48 | 24/48 | 232.0 | -6.393 | newline:24, word:13, space:10, correct_prefix:1 |
| colon_to_original_interval_L17_20_layer_out_reverse | 48 | colon | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 33/48 | 266.1 | -6.672 | newline:33, word:12, space:2, correct_prefix:1 |

### answer_colon

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_to_original_L17_layer_input_restore | 48 | answer_colon | to_original | single_layer |  | 17 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 1.835 | correct_prefix:46, space:2 |
| answer_colon_to_original_L17_layer_out_restore | 48 | answer_colon | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |
| answer_colon_to_original_interval_L18_19_layer_out_restore | 48 | answer_colon | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |
| answer_colon_to_original_interval_L17_20_layer_out_restore | 48 | answer_colon | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |
| answer_colon_to_original_interval_L17_20_mlp_out_restore | 48 | answer_colon | to_original | interval | L17_20 |  | mlp_out | restore | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |
| answer_colon_to_original_interval_L17_20_attn_out_restore | 48 | answer_colon | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 17/48 | 19.0 | -3.152 | word:31, newline:17 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L17_20_attn_out_restore | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | restore | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_restore | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |
| answer_colon_remove_from_inline_interval_L18_19_layer_out_restore | 48 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |
| answer_colon_remove_from_inline_L17_layer_out_restore | 48 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_out | restore | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |
| answer_colon_remove_from_inline_L17_layer_input_restore | 48 | answer_colon | remove_from_inline | single_layer |  | 17 | layer_input | restore | 15/48 | 16/48 | 15/48 | 2.4 | 0.103 | space:17, correct_prefix:16, newline:15 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L18_19_layer_out_random | 48 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | random | 44/48 | 44/48 | 1/48 | 1.1 | 2.229 | correct_prefix:44, space:3, newline:1 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_random | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | random | 42/48 | 45/48 | 0/48 | 1.0 | 2.352 | correct_prefix:45, space:3 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_random | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | random | 38/48 | 42/48 | 3/48 | 1.1 | 2.350 | correct_prefix:42, newline:3, word:2, space:1 |
| answer_colon_to_original_interval_L18_19_layer_out_random | 48 | answer_colon | to_original | interval | L18_19 |  | layer_out | random | 14/48 | 14/48 | 31/48 | 6.8 | -1.303 | newline:31, correct_prefix:14, word:2, space:1 |
| answer_colon_to_original_interval_L17_20_mlp_out_random | 48 | answer_colon | to_original | interval | L17_20 |  | mlp_out | random | 12/48 | 13/48 | 24/48 | 6.1 | -1.118 | newline:24, correct_prefix:13, word:8, space:2, symbol:1 |
| answer_colon_to_original_interval_L17_20_layer_out_random | 48 | answer_colon | to_original | interval | L17_20 |  | layer_out | random | 9/48 | 11/48 | 31/48 | 7.1 | -1.296 | newline:31, correct_prefix:11, word:3, space:3 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_random | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | random | 4/48 | 4/48 | 5/48 | 6.7 | -0.600 | word:39, newline:5, correct_prefix:4 |
| answer_colon_to_original_interval_L17_20_attn_out_random | 48 | answer_colon | to_original | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 5/48 | 14.2 | -2.046 | word:42, newline:5, correct_prefix:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_colon_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | answer_colon | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 46/48 | 0/48 | 1.0 | 3.880 | correct_prefix:46, space:2 |
| answer_colon_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 44/48 | 46/48 | 0/48 | 1.0 | 4.368 | correct_prefix:46, space:2 |
| answer_colon_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 35/48 | 40/48 | 0/48 | 1.2 | 19.751 | correct_prefix:40, number:4, space:4 |
| answer_colon_to_original_interval_L17_20_mlp_out_reverse | 48 | answer_colon | to_original | interval | L17_20 |  | mlp_out | reverse | 4/48 | 4/48 | 35/48 | 34.2 | -3.736 | newline:35, word:5, space:4, correct_prefix:4 |
| answer_colon_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | answer_colon | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 2/48 | 2/48 | 4/48 | 15.0 | -1.477 | word:40, newline:4, correct_prefix:2, explanation:2 |
| answer_colon_to_original_interval_L17_20_attn_out_reverse | 48 | answer_colon | to_original | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 0/48 | 18.4 | -1.172 | word:47, correct_prefix:1 |
| answer_colon_to_original_interval_L18_19_layer_out_reverse | 48 | answer_colon | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 0/48 | 30/48 | 306.8 | -6.904 | newline:30, word:12, space:4, explanation:2 |
| answer_colon_to_original_interval_L17_20_layer_out_reverse | 48 | answer_colon | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 35/48 | 337.2 | -7.105 | newline:35, word:10, correct_prefix:1, space:1, explanation:1 |

### answer_label_aligned

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_to_original_L17_layer_input_restore | 48 | answer_label_aligned | to_original | single_layer |  | 17 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 1.835 | correct_prefix:46, space:2 |
| answer_label_aligned_to_original_L17_layer_out_restore | 48 | answer_label_aligned | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_restore | 48 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_restore | 48 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_restore | 48 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | restore | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_restore | 48 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 17/48 | 19.0 | -3.152 | word:31, newline:17 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | restore | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_restore | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_restore | 48 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |
| answer_label_aligned_remove_from_inline_L17_layer_out_restore | 48 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_out | restore | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |
| answer_label_aligned_remove_from_inline_L17_layer_input_restore | 48 | answer_label_aligned | remove_from_inline | single_layer |  | 17 | layer_input | restore | 15/48 | 16/48 | 15/48 | 2.4 | 0.103 | space:17, correct_prefix:16, newline:15 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random | 48 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | random | 45/48 | 47/48 | 0/48 | 1.0 | 2.275 | correct_prefix:47, space:1 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_random | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | random | 44/48 | 46/48 | 0/48 | 1.0 | 2.352 | correct_prefix:46, space:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_random | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | random | 42/48 | 46/48 | 2/48 | 1.1 | 2.293 | correct_prefix:46, newline:2 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_random | 48 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | random | 13/48 | 13/48 | 24/48 | 5.5 | -1.182 | newline:24, correct_prefix:13, word:10, space:1 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_random | 48 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | random | 12/48 | 12/48 | 25/48 | 10.2 | -1.366 | newline:25, correct_prefix:12, word:5, space:5, explanation:1 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_random | 48 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | random | 11/48 | 11/48 | 29/48 | 7.5 | -1.363 | newline:29, correct_prefix:11, word:6, space:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_random | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | random | 2/48 | 2/48 | 5/48 | 6.9 | -0.892 | word:41, newline:5, correct_prefix:2 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_random | 48 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | random | 0/48 | 0/48 | 4/48 | 17.9 | -2.374 | word:43, newline:4, explanation:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | answer_label_aligned | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 46/48 | 0/48 | 1.0 | 3.880 | correct_prefix:46, space:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 44/48 | 46/48 | 0/48 | 1.0 | 4.368 | correct_prefix:46, space:2 |
| answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 35/48 | 40/48 | 0/48 | 1.2 | 19.751 | correct_prefix:40, number:4, space:4 |
| answer_label_aligned_to_original_interval_L17_20_mlp_out_reverse | 48 | answer_label_aligned | to_original | interval | L17_20 |  | mlp_out | reverse | 4/48 | 4/48 | 35/48 | 34.2 | -3.736 | newline:35, word:5, space:4, correct_prefix:4 |
| answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | answer_label_aligned | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 2/48 | 2/48 | 4/48 | 15.0 | -1.477 | word:40, newline:4, correct_prefix:2, explanation:2 |
| answer_label_aligned_to_original_interval_L17_20_attn_out_reverse | 48 | answer_label_aligned | to_original | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 0/48 | 18.4 | -1.172 | word:47, correct_prefix:1 |
| answer_label_aligned_to_original_interval_L18_19_layer_out_reverse | 48 | answer_label_aligned | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 0/48 | 30/48 | 306.8 | -6.904 | newline:30, word:12, space:4, explanation:2 |
| answer_label_aligned_to_original_interval_L17_20_layer_out_reverse | 48 | answer_label_aligned | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 35/48 | 337.2 | -7.105 | newline:35, word:10, correct_prefix:1, space:1, explanation:1 |

### separator

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_to_original_L17_layer_input_restore | 48 | separator | to_original | single_layer |  | 17 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 1.835 | correct_prefix:46, space:2 |
| separator_to_original_L17_layer_out_restore | 48 | separator | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |
| separator_to_original_interval_L18_19_layer_out_restore | 48 | separator | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |
| separator_to_original_interval_L17_20_layer_out_restore | 48 | separator | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |
| separator_to_original_interval_L17_20_mlp_out_restore | 48 | separator | to_original | interval | L17_20 |  | mlp_out | restore | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |
| separator_to_original_interval_L17_20_attn_out_restore | 48 | separator | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 17/48 | 19.0 | -3.152 | word:31, newline:17 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_remove_from_inline_interval_L17_20_attn_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | attn_out | restore | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |
| separator_remove_from_inline_interval_L17_20_layer_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |
| separator_remove_from_inline_interval_L18_19_layer_out_restore | 48 | separator | remove_from_inline | interval | L18_19 |  | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |
| separator_remove_from_inline_L17_layer_out_restore | 48 | separator | remove_from_inline | single_layer |  | 17 | layer_out | restore | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |
| separator_remove_from_inline_L17_layer_input_restore | 48 | separator | remove_from_inline | single_layer |  | 17 | layer_input | restore | 15/48 | 16/48 | 15/48 | 2.4 | 0.103 | space:17, correct_prefix:16, newline:15 |
| separator_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_remove_from_inline_interval_L18_19_layer_out_random | 48 | separator | remove_from_inline | interval | L18_19 |  | layer_out | random | 46/48 | 46/48 | 0/48 | 1.0 | 2.263 | correct_prefix:46, space:2 |
| separator_remove_from_inline_interval_L17_20_layer_out_random | 48 | separator | remove_from_inline | interval | L17_20 |  | layer_out | random | 43/48 | 45/48 | 0/48 | 1.1 | 2.404 | correct_prefix:45, space:3 |
| separator_remove_from_inline_interval_L17_20_mlp_out_random | 48 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | random | 39/48 | 43/48 | 1/48 | 1.1 | 2.353 | correct_prefix:43, word:2, space:2, newline:1 |
| separator_to_original_interval_L17_20_mlp_out_random | 48 | separator | to_original | interval | L17_20 |  | mlp_out | random | 11/48 | 12/48 | 24/48 | 5.2 | -1.056 | newline:24, correct_prefix:12, word:9, space:3 |
| separator_to_original_interval_L17_20_layer_out_random | 48 | separator | to_original | interval | L17_20 |  | layer_out | random | 11/48 | 12/48 | 27/48 | 10.3 | -1.438 | newline:27, correct_prefix:12, word:5, space:4 |
| separator_to_original_interval_L18_19_layer_out_random | 48 | separator | to_original | interval | L18_19 |  | layer_out | random | 10/48 | 11/48 | 32/48 | 9.6 | -1.445 | newline:32, correct_prefix:11, word:3, explanation:1, space:1 |
| separator_remove_from_inline_interval_L17_20_attn_out_random | 48 | separator | remove_from_inline | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 4/48 | 6.5 | 1.302 | word:42, newline:4, correct_prefix:1, explanation:1 |
| separator_to_original_interval_L17_20_attn_out_random | 48 | separator | to_original | interval | L17_20 |  | attn_out | random | 0/48 | 0/48 | 5/48 | 18.2 | -2.277 | word:42, newline:5, explanation:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| separator_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | separator | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 46/48 | 0/48 | 1.0 | 3.880 | correct_prefix:46, space:2 |
| separator_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | separator | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 44/48 | 46/48 | 0/48 | 1.0 | 4.368 | correct_prefix:46, space:2 |
| separator_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | separator | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 35/48 | 40/48 | 0/48 | 1.2 | 19.751 | correct_prefix:40, number:4, space:4 |
| separator_to_original_interval_L17_20_mlp_out_reverse | 48 | separator | to_original | interval | L17_20 |  | mlp_out | reverse | 4/48 | 4/48 | 35/48 | 34.2 | -3.736 | newline:35, word:5, space:4, correct_prefix:4 |
| separator_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | separator | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 2/48 | 2/48 | 4/48 | 15.0 | -1.477 | word:40, newline:4, correct_prefix:2, explanation:2 |
| separator_to_original_interval_L17_20_attn_out_reverse | 48 | separator | to_original | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 0/48 | 18.4 | -1.172 | word:47, correct_prefix:1 |
| separator_to_original_interval_L18_19_layer_out_reverse | 48 | separator | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 0/48 | 30/48 | 306.8 | -6.904 | newline:30, word:12, space:4, explanation:2 |
| separator_to_original_interval_L17_20_layer_out_reverse | 48 | separator | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 35/48 | 337.2 | -7.105 | newline:35, word:10, correct_prefix:1, space:1, explanation:1 |

### prompt_last

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_to_original_L17_layer_out_restore | 48 | prompt_last | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.1 | 1.667 | correct_prefix:46, space:2 |
| prompt_last_to_original_interval_L18_19_layer_out_restore | 48 | prompt_last | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 1.868 | correct_prefix:45, space:3 |
| prompt_last_to_original_L17_layer_input_restore | 48 | prompt_last | to_original | single_layer |  | 17 | layer_input | restore | 44/48 | 43/48 | 3/48 | 1.2 | 1.422 | correct_prefix:43, newline:3, space:2 |
| prompt_last_to_original_interval_L17_20_layer_out_restore | 48 | prompt_last | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 45/48 | 0/48 | 1.1 | 2.102 | correct_prefix:45, space:3 |
| prompt_last_to_original_interval_L17_20_mlp_out_restore | 48 | prompt_last | to_original | interval | L17_20 |  | mlp_out | restore | 31/48 | 32/48 | 14/48 | 1.6 | 0.570 | correct_prefix:32, newline:14, word:1, space:1 |
| prompt_last_to_original_interval_L17_20_attn_out_restore | 48 | prompt_last | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 22/48 | 19.1 | -3.509 | word:26, newline:22 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_remove_from_inline_interval_L17_20_attn_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | restore | 5/48 | 7/48 | 2/48 | 2.7 | 0.118 | word:39, correct_prefix:7, newline:2 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 14/48 | 29/48 | 3.2 | -0.624 | newline:29, correct_prefix:14, space:3, word:2 |
| prompt_last_remove_from_inline_interval_L18_19_layer_out_restore | 48 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | restore | 20/48 | 19/48 | 23/48 | 2.2 | -0.073 | newline:23, correct_prefix:19, space:4, word:2 |
| prompt_last_remove_from_inline_L17_layer_out_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_out | restore | 23/48 | 23/48 | 15/48 | 2.0 | 0.337 | correct_prefix:23, newline:15, space:10 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 25/48 | 27/48 | 21/48 | 1.7 | 0.276 | correct_prefix:27, newline:21 |
| prompt_last_remove_from_inline_L17_layer_input_restore | 48 | prompt_last | remove_from_inline | single_layer |  | 17 | layer_input | restore | 25/48 | 26/48 | 9/48 | 1.8 | 0.616 | correct_prefix:26, space:11, newline:9, word:2 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_remove_from_inline_interval_L18_19_layer_out_random | 48 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | random | 45/48 | 47/48 | 0/48 | 1.0 | 2.173 | correct_prefix:47, space:1 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_random | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | random | 42/48 | 45/48 | 0/48 | 1.1 | 2.341 | correct_prefix:45, space:3 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_random | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | random | 37/48 | 40/48 | 4/48 | 1.2 | 1.676 | correct_prefix:40, space:4, newline:4 |
| prompt_last_to_original_interval_L17_20_layer_out_random | 48 | prompt_last | to_original | interval | L17_20 |  | layer_out | random | 11/48 | 12/48 | 29/48 | 9.4 | -1.533 | newline:29, correct_prefix:12, space:5, word:2 |
| prompt_last_to_original_interval_L18_19_layer_out_random | 48 | prompt_last | to_original | interval | L18_19 |  | layer_out | random | 10/48 | 10/48 | 29/48 | 8.3 | -1.534 | newline:29, correct_prefix:10, word:4, space:3, explanation:2 |
| prompt_last_to_original_interval_L17_20_mlp_out_random | 48 | prompt_last | to_original | interval | L17_20 |  | mlp_out | random | 7/48 | 8/48 | 30/48 | 9.7 | -1.786 | newline:30, correct_prefix:8, word:6, space:4 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_random | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | random | 3/48 | 3/48 | 5/48 | 6.3 | -0.922 | word:39, newline:5, correct_prefix:3, space:1 |
| prompt_last_to_original_interval_L17_20_attn_out_random | 48 | prompt_last | to_original | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 11/48 | 27.3 | -3.064 | word:35, newline:11, correct_prefix:1, explanation:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| prompt_last_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | prompt_last | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 46/48 | 0/48 | 1.0 | 3.530 | correct_prefix:46, space:2 |
| prompt_last_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 43/48 | 46/48 | 0/48 | 1.0 | 4.089 | correct_prefix:46, space:2 |
| prompt_last_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 33/48 | 36/48 | 0/48 | 1.4 | 10.859 | correct_prefix:36, space:9, word:2, number:1 |
| prompt_last_to_original_interval_L17_20_mlp_out_reverse | 48 | prompt_last | to_original | interval | L17_20 |  | mlp_out | reverse | 5/48 | 5/48 | 34/48 | 29.3 | -3.303 | newline:34, word:5, correct_prefix:5, space:4 |
| prompt_last_to_original_interval_L17_20_attn_out_reverse | 48 | prompt_last | to_original | interval | L17_20 |  | attn_out | reverse | 2/48 | 2/48 | 0/48 | 23.7 | -2.172 | word:44, explanation:2, correct_prefix:2 |
| prompt_last_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | prompt_last | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 2/48 | 12.6 | -1.409 | word:43, newline:2, explanation:2, correct_prefix:1 |
| prompt_last_to_original_interval_L18_19_layer_out_reverse | 48 | prompt_last | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 1/48 | 24/48 | 232.0 | -6.393 | newline:24, word:13, space:10, correct_prefix:1 |
| prompt_last_to_original_interval_L17_20_layer_out_reverse | 48 | prompt_last | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 33/48 | 266.1 | -6.672 | newline:33, word:12, space:2, correct_prefix:1 |

### question_mark_answer

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_to_original_interval_L18_19_layer_out_restore | 48 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| question_mark_answer_to_original_L17_layer_input_restore | 48 | question_mark_answer | to_original | single_layer |  | 17 | layer_input | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| question_mark_answer_to_original_L17_layer_out_restore | 48 | question_mark_answer | to_original | single_layer |  | 17 | layer_out | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_restore | 48 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | restore | 42/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_restore | 48 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | restore | 29/48 | 34/48 | 9/48 | 1.5 | 0.993 | correct_prefix:34, newline:9, word:5 |
| question_mark_answer_to_original_interval_L17_20_attn_out_restore | 48 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 16/48 | 21.6 | -3.247 | word:30, newline:16, explanation:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | restore | 5/48 | 5/48 | 0/48 | 2.6 | 0.469 | word:43, correct_prefix:5 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | restore | 11/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| question_mark_answer_remove_from_inline_L17_layer_input_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_input | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| question_mark_answer_remove_from_inline_L17_layer_out_restore | 48 | question_mark_answer | remove_from_inline | single_layer |  | 17 | layer_out | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 12/48 | 16/48 | 31/48 | 2.6 | -0.728 | newline:31, correct_prefix:16, word:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_random | 48 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | random | 47/48 | 47/48 | 0/48 | 1.0 | 2.203 | correct_prefix:47, space:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_random | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | random | 43/48 | 45/48 | 0/48 | 1.1 | 2.279 | correct_prefix:45, space:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_random | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | random | 31/48 | 34/48 | 12/48 | 2.0 | 1.323 | correct_prefix:34, newline:12, word:2 |
| question_mark_answer_to_original_interval_L18_19_layer_out_random | 48 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | random | 10/48 | 9/48 | 26/48 | 7.2 | -1.366 | newline:26, correct_prefix:9, word:6, space:5, explanation:2 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_random | 48 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | random | 9/48 | 10/48 | 29/48 | 7.0 | -1.638 | newline:29, correct_prefix:10, word:8, space:1 |
| question_mark_answer_to_original_interval_L17_20_layer_out_random | 48 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | random | 7/48 | 9/48 | 32/48 | 6.9 | -1.434 | newline:32, correct_prefix:9, space:4, word:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_random | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 4/48 | 7.3 | 1.118 | word:43, newline:4, correct_prefix:1 |
| question_mark_answer_to_original_interval_L17_20_attn_out_random | 48 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | random | 1/48 | 1/48 | 4/48 | 20.4 | 0.029 | word:43, newline:4, correct_prefix:1 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| question_mark_answer_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | question_mark_answer | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 45/48 | 45/48 | 0/48 | 1.1 | 3.842 | correct_prefix:45, space:3 |
| question_mark_answer_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 44/48 | 46/48 | 0/48 | 1.0 | 4.285 | correct_prefix:46, space:2 |
| question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 29/48 | 32/48 | 0/48 | 1.6 | 11.073 | correct_prefix:32, space:10, number:5, word:1 |
| question_mark_answer_to_original_interval_L17_20_mlp_out_reverse | 48 | question_mark_answer | to_original | interval | L17_20 |  | mlp_out | reverse | 4/48 | 4/48 | 34/48 | 39.8 | -3.765 | newline:34, word:6, space:4, correct_prefix:4 |
| question_mark_answer_to_original_interval_L17_20_attn_out_reverse | 48 | question_mark_answer | to_original | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 0/48 | 20.4 | -1.413 | word:47, correct_prefix:1 |
| question_mark_answer_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | question_mark_answer | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 2/48 | 18.2 | -1.871 | word:41, explanation:4, newline:2, correct_prefix:1 |
| question_mark_answer_to_original_interval_L18_19_layer_out_reverse | 48 | question_mark_answer | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 0/48 | 18/48 | 407.2 | -7.005 | newline:18, word:13, explanation:9, space:8 |
| question_mark_answer_to_original_interval_L17_20_layer_out_reverse | 48 | question_mark_answer | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 23/48 | 383.5 | -6.718 | newline:23, word:15, space:5, explanation:4, correct_prefix:1 |

### relation_tail

#### Best Sufficiency Restore

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_to_original_interval_L18_19_layer_out_restore | 48 | relation_tail | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| relation_tail_to_original_L17_layer_input_restore | 48 | relation_tail | to_original | single_layer |  | 17 | layer_input | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| relation_tail_to_original_L17_layer_out_restore | 48 | relation_tail | to_original | single_layer |  | 17 | layer_out | restore | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| relation_tail_to_original_interval_L17_20_layer_out_restore | 48 | relation_tail | to_original | interval | L17_20 |  | layer_out | restore | 31/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_restore | 48 | relation_tail | to_original | interval | L17_20 |  | mlp_out | restore | 21/48 | 25/48 | 12/48 | 2.1 | 0.346 | correct_prefix:25, newline:12, word:9, space:2 |
| relation_tail_to_original_interval_L17_20_attn_out_restore | 48 | relation_tail | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 16/48 | 20.0 | -3.184 | word:30, newline:16, explanation:2 |

#### Best Necessity Remove

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L17_20_attn_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | restore | 1/48 | 4/48 | 0/48 | 2.7 | 0.404 | word:44, correct_prefix:4 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 4/48 | 7/48 | 39/48 | 4.6 | -1.646 | newline:39, correct_prefix:7, word:2 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_restore | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | restore | 7/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| relation_tail_remove_from_inline_interval_L18_19_layer_out_restore | 48 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| relation_tail_remove_from_inline_L17_layer_input_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_input | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |
| relation_tail_remove_from_inline_L17_layer_out_restore | 48 | relation_tail | remove_from_inline | single_layer |  | 17 | layer_out | restore | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |

#### Best Random Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L18_19_layer_out_random | 48 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | random | 44/48 | 45/48 | 0/48 | 1.1 | 2.014 | correct_prefix:45, space:3 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_random | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | random | 34/48 | 47/48 | 0/48 | 1.0 | 2.181 | correct_prefix:47, space:1 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_random | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | random | 24/48 | 36/48 | 5/48 | 1.9 | 1.073 | correct_prefix:36, newline:5, word:4, space:2, number:1 |
| relation_tail_to_original_interval_L18_19_layer_out_random | 48 | relation_tail | to_original | interval | L18_19 |  | layer_out | random | 7/48 | 8/48 | 29/48 | 11.1 | -1.549 | newline:29, correct_prefix:8, word:6, space:4, explanation:1 |
| relation_tail_to_original_interval_L17_20_layer_out_random | 48 | relation_tail | to_original | interval | L17_20 |  | layer_out | random | 7/48 | 9/48 | 32/48 | 8.6 | -1.426 | newline:32, correct_prefix:9, space:5, word:2 |
| relation_tail_to_original_interval_L17_20_attn_out_random | 48 | relation_tail | to_original | interval | L17_20 |  | attn_out | random | 2/48 | 3/48 | 2/48 | 15.0 | -2.139 | word:43, correct_prefix:3, newline:2 |
| relation_tail_to_original_interval_L17_20_mlp_out_random | 48 | relation_tail | to_original | interval | L17_20 |  | mlp_out | random | 2/48 | 3/48 | 27/48 | 9.4 | -2.193 | newline:27, word:14, space:4, correct_prefix:3 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_random | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | random | 1/48 | 3/48 | 4/48 | 7.5 | -0.823 | word:41, newline:4, correct_prefix:3 |

#### Best Reverse Controls

| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |
|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| relation_tail_remove_from_inline_interval_L18_19_layer_out_reverse | 48 | relation_tail | remove_from_inline | interval | L18_19 |  | layer_out | reverse | 44/48 | 45/48 | 0/48 | 1.1 | 3.842 | correct_prefix:45, space:3 |
| relation_tail_remove_from_inline_interval_L17_20_layer_out_reverse | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | layer_out | reverse | 36/48 | 46/48 | 0/48 | 1.0 | 4.285 | correct_prefix:46, space:2 |
| relation_tail_remove_from_inline_interval_L17_20_mlp_out_reverse | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | mlp_out | reverse | 21/48 | 30/48 | 0/48 | 1.8 | 8.823 | correct_prefix:30, space:13, number:5 |
| relation_tail_to_original_interval_L17_20_attn_out_reverse | 48 | relation_tail | to_original | interval | L17_20 |  | attn_out | reverse | 1/48 | 1/48 | 0/48 | 18.4 | -1.303 | word:47, correct_prefix:1 |
| relation_tail_to_original_interval_L17_20_mlp_out_reverse | 48 | relation_tail | to_original | interval | L17_20 |  | mlp_out | reverse | 1/48 | 2/48 | 38/48 | 55.6 | -4.610 | newline:38, space:4, word:4, correct_prefix:2 |
| relation_tail_remove_from_inline_interval_L17_20_attn_out_reverse | 48 | relation_tail | remove_from_inline | interval | L17_20 |  | attn_out | reverse | 0/48 | 1/48 | 3/48 | 16.9 | -1.863 | word:39, explanation:5, newline:3, correct_prefix:1 |
| relation_tail_to_original_interval_L18_19_layer_out_reverse | 48 | relation_tail | to_original | interval | L18_19 |  | layer_out | reverse | 0/48 | 0/48 | 18/48 | 407.2 | -7.005 | newline:18, word:13, explanation:9, space:8 |
| relation_tail_to_original_interval_L17_20_layer_out_reverse | 48 | relation_tail | to_original | interval | L17_20 |  | layer_out | reverse | 0/48 | 1/48 | 23/48 | 383.5 | -6.718 | newline:23, word:15, space:5, explanation:4, correct_prefix:1 |

### Global Top Notes

- Top sufficiency: answer_colon_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0; answer_label_aligned_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0; separator_to_original_L17_layer_input_restore exact=46/48 newline=0/48 rank=1.0; answer_colon_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.0; answer_label_aligned_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.0; separator_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.0; colon_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.1; prompt_last_to_original_L17_layer_out_restore exact=46/48 newline=0/48 rank=1.1; question_mark_answer_to_original_interval_L18_19_layer_out_restore exact=45/48 newline=0/48 rank=1.0; question_mark_answer_to_original_L17_layer_input_restore exact=45/48 newline=0/48 rank=1.0
- Top necessity/remove: relation_tail_remove_from_inline_interval_L17_20_attn_out_restore exact=1/48 newline=0/48 rank=2.7; relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore exact=4/48 newline=39/48 rank=4.6; colon_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=2/48 rank=2.7; prompt_last_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=2/48 rank=2.7; question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore exact=5/48 newline=0/48 rank=2.6; relation_tail_remove_from_inline_interval_L17_20_layer_out_restore exact=7/48 newline=34/48 rank=6.9; answer_colon_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4; answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4; separator_remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48 rank=2.4; question_mark_answer_remove_from_inline_interval_L17_20_layer_out_restore exact=11/48 newline=34/48 rank=6.9
