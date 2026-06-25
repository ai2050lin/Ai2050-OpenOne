# Phase 645 Cross-Model Summary

目标：审计 Phase 643/644 的 L17-L20 middle protocol trajectory 是否只在目标失败样本上有效，以及它对原本正确样本、关系变化、解释任务和非值任务的副作用边界。

注意：relation_changed / explanation_needed / non_value 的 exact 不是正向成功率，而是旧 value 吸附或过短回答的风险指标。

## qwen3

- raw_cases: 320 / selected_items: 219 / mode_rows: 1752
- component: `layer_out` / layers: `[18, 19]` / top_k: 20
- max_per_split: 48 / max_new_tokens: 3
- selection_stats: `{'counts': {'target_failure': 26, 'original_correct': 48, 'inline_bad': 1, 'relation_changed': 48, 'explanation_needed': 48, 'non_value': 48}, 'target_failure_seen': 26, 'original_correct_seen': 289, 'inline_bad_seen': 1, 'relation_pool_size': 2}`
- filtered: `{'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 4.49

### target_failure

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 26 | 23/26 | 19/26 | 4/26 | 0/26 | 0/26 | 26/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| inline | 26 | 1/26 | 0/26 | 1/26 | 15/26 | 14/26 | 12/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| to_original_middle_restore | 26 | 2/26 | 2/26 | 1/26 | 22/26 | 21/26 | 5/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |  ?\n\nOkay,:21,  91:1,  v48:1,  22:1,  v22:1,  v91:1 |
| to_original_middle_random | 26 | 11/26 | 10/26 | 1/26 | 4/26 | 4/26 | 22/26 | 2.0 | 0.798 | correct_prefix:11, space:10, newline:4, word:1 |  v05:4,  22:4,  ?\n\nOkay,:4,  05:3,  91:3,  v22:3,  v48:2,  v91:2 |
| to_original_middle_reverse | 26 | 18/26 | 15/26 | 2/26 | 0/26 | 0/26 | 26/26 | 1.4 | 1.846 | correct_prefix:18, space:5, word:3 |  v05:7,  v22:4,  22:3,  v48:3,  91:3,  v91:3,  o43:1,  o05:1 |
| remove_from_inline_middle_restore | 26 | 9/26 | 7/26 | 2/26 | 0/26 | 0/26 | 26/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| remove_from_inline_middle_random | 26 | 0/26 | 0/26 | 0/26 | 20/26 | 20/26 | 6/26 | 5.3 | -1.774 | newline:20, space:6 |  ?\n\nOkay,:19,  05:2,  91:2,  22:2,  ?\n\nTo solve:1 |
| remove_from_inline_middle_reverse | 26 | 0/26 | 0/26 | 0/26 | 25/26 | 25/26 | 1/26 | 5.8 | -2.361 | newline:25, space:1 |  ?\n\nOkay,:24,  ?\nQuestion::1,  91:1 |

### original_correct

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 30/48 | 28/48 | 1/48 | 0/48 | 0/48 | 48/48 | 1.5 | 1.560 | correct_prefix:30, space:17, word:1 |  v48:18,  22:9,  v22:6,  91:4,  05:3,  v05:3,  v91:2,  48:2 |
| inline | 48 | 6/48 | 7/48 | 0/48 | 24/48 | 21/48 | 27/48 | 4.4 | -1.328 | newline:24, space:18, correct_prefix:6 |  ?\n\nOkay,:21,  22:10,  v48:7,  05:4,  48:3,  91:3 |
| to_original_middle_restore | 48 | 8/48 | 8/48 | 0/48 | 36/48 | 36/48 | 12/48 | 3.6 | -1.247 | newline:36, correct_prefix:8, space:4 |  ?\n\nOkay,:36,  v48:8,  05:3,  91:1 |
| to_original_middle_random | 48 | 23/48 | 23/48 | 1/48 | 4/48 | 2/48 | 47/48 | 1.8 | 1.094 | correct_prefix:23, space:20, newline:4, word:1 |  v48:15,  22:8,  v22:5,  48:5,  05:4,  91:4,  v05:2,  v91:2 |
| to_original_middle_reverse | 48 | 31/48 | 29/48 | 1/48 | 0/48 | 0/48 | 48/48 | 1.6 | 2.143 | correct_prefix:31, space:13, word:4 |  v48:18,  v22:8,  22:7,  v05:3,  05:2,  o17:2,  48:2,  91:2 |
| remove_from_inline_middle_restore | 48 | 15/48 | 14/48 | 0/48 | 1/48 | 1/48 | 48/48 | 2.6 | 0.383 | space:32, correct_prefix:15, newline:1 |  22:14,  v48:13,  48:8,  91:6,  05:5,  v05:1,  \n\nOkay,:1 |
| remove_from_inline_middle_random | 48 | 7/48 | 7/48 | 0/48 | 21/48 | 21/48 | 27/48 | 4.6 | -1.453 | newline:21, space:20, correct_prefix:7 |  ?\n\nOkay,:20,  22:8,  48:7,  v48:5,  05:3,  91:2,  v05:1,  v91:1 |
| remove_from_inline_middle_reverse | 48 | 1/48 | 1/48 | 0/48 | 46/48 | 46/48 | 2/48 | 5.3 | -2.341 | newline:46, space:1, correct_prefix:1 |  ?\n\nOkay,:46,  48:1,  v48:1 |

### inline_bad

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 1.750 | correct_prefix:1 |  v91:1 |
| inline | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 7.0 | -3.250 | space:1 |  91:1 |
| to_original_middle_restore | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 4.0 | -1.625 | space:1 |  91:1 |
| to_original_middle_random | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 4.250 | correct_prefix:1 |  v91:1 |
| to_original_middle_reverse | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 3.375 | correct_prefix:1 |  v91:1 |
| remove_from_inline_middle_restore | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 7.0 | -1.125 | space:1 |  91:1 |
| remove_from_inline_middle_random | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 12.0 | -4.750 | newline:1 |  ?\n\nOkay,:1 |
| remove_from_inline_middle_reverse | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 9.0 | -4.125 | space:1 |  91:1 |

### relation_changed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 30/48 | 11/48 | 12/48 | 0/48 | 0/48 | 48/48 | 1.5 | 1.568 | correct_prefix:30, space:17, word:1 |  v48:17,  22:10,  v05:7,  v22:3,  05:3,  91:3,  v91:2,  48:2 |
| inline | 48 | 6/48 | 5/48 | 2/48 | 26/48 | 24/48 | 24/48 | 4.2 | -1.232 | newline:26, space:16, correct_prefix:6 |  ?\n\nOkay,:24,  22:8,  v48:7,  05:4,  48:3,  91:2 |
| to_original_middle_restore | 48 | 8/48 | 6/48 | 2/48 | 36/48 | 36/48 | 12/48 | 3.4 | -1.138 | newline:36, correct_prefix:8, space:4 |  ?\n\nOkay,:36,  v48:8,  05:3,  91:1 |
| to_original_middle_random | 48 | 30/48 | 12/48 | 12/48 | 1/48 | 1/48 | 47/48 | 1.8 | 1.255 | correct_prefix:30, space:17, newline:1 |  v48:17,  22:8,  v22:5,  v05:5,  05:4,  91:4,  48:3,  v91:1 |
| to_original_middle_reverse | 48 | 30/48 | 11/48 | 14/48 | 0/48 | 0/48 | 48/48 | 1.6 | 2.112 | correct_prefix:30, space:14, word:4 |  v48:17,  22:8,  v05:7,  v22:5,  05:2,  o17:2,  91:2,  48:2 |
| remove_from_inline_middle_restore | 48 | 17/48 | 10/48 | 6/48 | 1/48 | 1/48 | 48/48 | 2.6 | 0.417 | space:30, correct_prefix:17, newline:1 |  22:12,  v48:12,  48:8,  05:6,  91:5,  v05:4,  \n\nOkay,:1 |
| remove_from_inline_middle_random | 48 | 6/48 | 5/48 | 0/48 | 27/48 | 27/48 | 21/48 | 4.3 | -1.409 | newline:27, space:15, correct_prefix:6 |  ?\n\nOkay,:27,  22:7,  v48:5,  05:4,  91:2,  48:2,  v91:1 |
| remove_from_inline_middle_reverse | 48 | 1/48 | 1/48 | 0/48 | 46/48 | 46/48 | 2/48 | 5.1 | -2.253 | newline:46, space:1, correct_prefix:1 |  ?\n\nOkay,:46,  48:1,  v48:1 |

### explanation_needed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 32/48 | 32/48 | 2/48 | 0/48 | 0/48 | 46/48 | 1.3 | 3.461 | correct_prefix:32, word:12, explanation:4 |  v48:17,  v22:12,  v05:5,  o06:3,  o82:2,  The value is:2,  o29:2,  o17:1 |
| inline | 48 | 6/48 | 8/48 | 1/48 | 42/48 | 39/48 | 40/48 | 3.9 | -1.023 | newline:42, correct_prefix:6 |  \n\nAnswer::27,  ?\n\nOkay,:7,  v48:5,  \n\nOkay,:4,  v05:2,  v22:2,  \n\nTo determine:1 |
| to_original_middle_restore | 48 | 2/48 | 2/48 | 0/48 | 46/48 | 46/48 | 23/48 | 5.8 | -3.250 | newline:46, correct_prefix:2 |  \n\nAnswer::20,  \n\nThe value:13,  \n\nThe answer:8,  \n\nThe question:4,  v48:1,  \n\nOkay,:1,  v05:1 |
| to_original_middle_random | 48 | 17/48 | 15/48 | 2/48 | 3/48 | 3/48 | 27/48 | 2.5 | 2.073 | explanation:18, correct_prefix:17, word:9, newline:3, space:1 |  The value is:15,  v22:10,  v05:4,  v48:3,  The value of:2,  o17:2,  ?\n\nOkay,:2,  o06:2 |
| to_original_middle_reverse | 48 | 38/48 | 34/48 | 2/48 | 0/48 | 0/48 | 48/48 | 1.2 | 4.841 | correct_prefix:38, word:10 |  v48:16,  v22:14,  o06:4,  v05:4,  v91:2,  o29:2,  o43:1,  o17:1 |
| remove_from_inline_middle_restore | 48 | 48/48 | 47/48 | 1/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.815 | correct_prefix:48 |  v48:20,  v22:13,  v05:9,  v91:6 |
| remove_from_inline_middle_random | 48 | 10/48 | 9/48 | 1/48 | 33/48 | 33/48 | 27/48 | 6.5 | -1.799 | newline:33, correct_prefix:10, space:5 |  ?\n\nOkay,:17,  \n\nAnswer::12,  v48:6,  22:4,  ?\nOkay,:4,  v05:2,  05:1,  v91:1 |
| remove_from_inline_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 48/48 | 48/48 | 19/48 | 35.2 | -5.245 | newline:48 |  \n\nAnswer::19,  \n\nThe question:15,  ?\n\nOkay,:14 |

### non_value

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 1/48 | 1/48 | 0/48 | 0/48 | 47/48 | 1/48 | 19.2 | -2.055 | explanation:47, correct_prefix:1 |  Yes.\nExplanation:30,  Yes.\nWait:7,  Yes.\nOkay:6,  Yes\nExplanation:4,  v48:1 |
| inline | 48 | 0/48 | 0/48 | 0/48 | 48/48 | 48/48 | 44/48 | 24.2 | -4.220 | newline:48 |  \n\nOkay,:44,  ?\n\nOkay,:4 |
| to_original_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 48/48 | 48/48 | 48/48 | 30.1 | -5.544 | newline:48 |  \n\nOkay,:48 |
| to_original_middle_random | 48 | 4/48 | 4/48 | 0/48 | 0/48 | 43/48 | 5/48 | 28.5 | -2.574 | explanation:43, correct_prefix:4, space:1 |  Yes.\nExplanation:16,  Yes.\nOkay:8,  Yes\nExplanation:6,  Yes.\n\nWait:5,  Yes.\nWait:4,  v48:3,  yes\n\nOkay:2,  05:1 |
| to_original_middle_reverse | 48 | 6/48 | 8/48 | 0/48 | 0/48 | 34/48 | 14/48 | 6.2 | 2.285 | explanation:36, space:6, correct_prefix:6 |  Yes.\nExplanation:15,  Yes\nExplanation:13,  v48:7,  22:3,  05:2,  Yes\n\nWait:2,  Yes\n\nOkay:1,  Yes\n\nExplanation:1 |
| remove_from_inline_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 48/48 | 0/48 | 29.8 | -3.510 | explanation:48 |  Yes.\nWait:28,  Yes.\nOkay:16,  Yes.\nQuestion:4 |
| remove_from_inline_middle_random | 48 | 2/48 | 2/48 | 0/48 | 37/48 | 44/48 | 21/48 | 23.9 | -3.622 | newline:37, explanation:7, correct_prefix:2, word:1, space:1 |  ?\n\nOkay,:19,  \n\nOkay,:18,  yes.\nOkay:3,  Yes.\nQuestion:2,  v48:1,  Is the value:1,  48:1,  Yes.\nOkay:1 |
| remove_from_inline_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 44/48 | 48/48 | 12/48 | 20.6 | -2.879 | newline:44, explanation:4 |  ?\n\nOkay,:29,  \n\nOkay,:12,  Yes.\n\nWait:4,  \n\nThe question:2,  Yes.\nQuestion:1 |

### Boundary Notes

- target_failure: original exact/old=19/26, to_original_middle_restore exact/old=2/26, newline 0->22
- target_failure: inline exact/old=0/26, remove_from_inline_middle_restore exact/old=7/26, newline 15->0
- original_correct: original exact/old=28/48, to_original_middle_restore exact/old=8/48, newline 0->36
- original_correct: inline exact/old=7/48, remove_from_inline_middle_restore exact/old=14/48, newline 24->1
- inline_bad: original exact/old=0/1, to_original_middle_restore exact/old=0/1, newline 0->0
- inline_bad: inline exact/old=0/1, remove_from_inline_middle_restore exact/old=0/1, newline 0->0
- relation_changed: original exact/old=11/48, to_original_middle_restore exact/old=6/48, newline 0->36
- relation_changed: inline exact/old=5/48, remove_from_inline_middle_restore exact/old=10/48, newline 26->1
- explanation_needed: original exact/old=32/48, to_original_middle_restore exact/old=2/48, newline 0->46
- explanation_needed: inline exact/old=8/48, remove_from_inline_middle_restore exact/old=47/48, newline 42->0
- non_value: original exact/old=1/48, to_original_middle_restore exact/old=0/48, newline 0->48
- non_value: inline exact/old=0/48, remove_from_inline_middle_restore exact/old=0/48, newline 48->0

## glm4

- raw_cases: 320 / selected_items: 234 / mode_rows: 1872
- component: `layer_out` / layers: `[18, 19]` / top_k: 20
- max_per_split: 48 / max_new_tokens: 2
- selection_stats: `{'counts': {'target_failure': 36, 'original_correct': 48, 'inline_bad': 6, 'relation_changed': 48, 'explanation_needed': 48, 'non_value': 48}, 'target_failure_seen': 36, 'original_correct_seen': 257, 'inline_bad_seen': 6, 'relation_pool_size': 2}`
- filtered: `{'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 5.91

### target_failure

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 36 | 30/36 | 29/36 | 1/36 | 0/36 | 1/36 | 36/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:8,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| inline | 36 | 28/36 | 27/36 | 1/36 | 0/36 | 3/36 | 36/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:7,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2 |
| to_original_middle_restore | 36 | 30/36 | 27/36 | 3/36 | 0/36 | 3/36 | 36/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_middle_random | 36 | 28/36 | 25/36 | 1/36 | 0/36 | 1/36 | 36/36 | 1.4 | 72.707 | correct_prefix:28, word:7, explanation:1 |  v91:10,  v05:8,  v48:6,  c77:3,  c12:3,  v22:2,  c59:2,  c33:1 |
| to_original_middle_reverse | 36 | 19/36 | 16/36 | 1/36 | 0/36 | 0/36 | 36/36 | 1.7 | 96.380 | correct_prefix:19, word:17 |  c33:6,  v48:6,  v91:6,  c77:5,  v05:5,  c12:4,  c59:3,  o17:1 |
| remove_from_inline_middle_restore | 36 | 34/36 | 31/36 | 2/36 | 0/36 | 1/36 | 36/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_middle_random | 36 | 25/36 | 23/36 | 1/36 | 0/36 | 4/36 | 36/36 | 2.0 | 62.119 | correct_prefix:25, word:6, explanation:5 |  v05:8,  v91:8,  v48:6,  v22:3,  c33:3,  Yes.\n\n:2,  c12:2,  c77:1 |
| remove_from_inline_middle_reverse | 36 | 21/36 | 21/36 | 0/36 | 0/36 | 4/36 | 36/36 | 3.3 | 24.661 | correct_prefix:21, word:11, explanation:4 |  v05:7,  v91:7,  v48:5,  c33:4,  Yes.\n:4,  c12:3,  c77:2,  c59:2 |

### original_correct

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 41/48 | 40/48 | 0/48 | 0/48 | 1/48 | 48/48 | 1.2 | 73.541 | correct_prefix:41, word:6, explanation:1 |  v48:17,  v22:11,  v91:7,  v05:5,  c12:3,  c33:2,  c77:2,  No.\n\n:1 |
| inline | 48 | 34/48 | 33/48 | 0/48 | 0/48 | 8/48 | 48/48 | 1.5 | 69.506 | correct_prefix:34, explanation:8, word:6 |  v48:11,  v22:10,  v91:7,  Yes.\n:5,  v05:5,  c59:3,  No.\n:3,  c12:2 |
| to_original_middle_restore | 48 | 33/48 | 33/48 | 0/48 | 0/48 | 4/48 | 48/48 | 1.6 | 75.512 | correct_prefix:33, word:10, explanation:5 |  v48:12,  v22:9,  v91:7,  v05:5,  c12:4,  Yes.\n\n:3,  c59:2,  c77:2 |
| to_original_middle_random | 48 | 34/48 | 33/48 | 0/48 | 0/48 | 1/48 | 48/48 | 1.4 | 75.438 | correct_prefix:34, word:13, explanation:1 |  v22:11,  v48:10,  v91:7,  v05:5,  c33:4,  c59:3,  c77:3,  c12:3 |
| to_original_middle_reverse | 48 | 22/48 | 21/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.6 | 87.250 | word:26, correct_prefix:22 |  c12:8,  v48:7,  c77:7,  v22:6,  v91:6,  c59:4,  c33:4,  v05:2 |
| remove_from_inline_middle_restore | 48 | 44/48 | 44/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.1 | 71.477 | correct_prefix:44, word:4 |  v48:18,  v22:14,  v91:7,  v05:5,  c33:2,  o82:1,  c77:1 |
| remove_from_inline_middle_random | 48 | 28/48 | 28/48 | 0/48 | 0/48 | 5/48 | 48/48 | 1.9 | 57.551 | correct_prefix:28, word:14, explanation:6 |  v22:8,  v48:8,  v91:7,  c59:5,  Yes.\n:5,  v05:5,  c33:4,  c12:4 |
| remove_from_inline_middle_reverse | 48 | 23/48 | 22/48 | 0/48 | 0/48 | 5/48 | 48/48 | 2.7 | 27.182 | correct_prefix:23, word:14, explanation:11 |  c59:8,  v48:7,  v22:7,  c77:5,  v91:5,  Yes.\n:4,  No.:4,  v05:3 |

### inline_bad

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 6 | 6/6 | 3/6 | 3/6 | 0/6 | 0/6 | 6/6 | 1.0 | 67.594 | correct_prefix:6 |  v05:2,  v48:2,  v22:2 |
| inline | 6 | 6/6 | 1/6 | 5/6 | 0/6 | 0/6 | 6/6 | 1.0 | 83.422 | correct_prefix:6 |  v91:3,  v05:2,  v48:1 |
| to_original_middle_restore | 6 | 6/6 | 3/6 | 3/6 | 0/6 | 0/6 | 6/6 | 1.0 | 67.458 | correct_prefix:6 |  v05:2,  v48:2,  v22:2 |
| to_original_middle_random | 6 | 6/6 | 2/6 | 4/6 | 0/6 | 0/6 | 6/6 | 1.0 | 83.156 | correct_prefix:6 |  v05:2,  v48:2,  v22:1,  v91:1 |
| to_original_middle_reverse | 6 | 5/6 | 2/6 | 3/6 | 0/6 | 0/6 | 6/6 | 1.2 | 67.620 | correct_prefix:5, word:1 |  v22:2,  c33:1,  v91:1,  v05:1,  v48:1 |
| remove_from_inline_middle_restore | 6 | 6/6 | 2/6 | 4/6 | 0/6 | 0/6 | 6/6 | 1.0 | 67.927 | correct_prefix:6 |  v91:3,  v05:1,  v22:1,  v48:1 |
| remove_from_inline_middle_random | 6 | 5/6 | 2/6 | 3/6 | 0/6 | 0/6 | 6/6 | 1.2 | 51.714 | correct_prefix:5, word:1 |  v91:3,  c33:1,  v22:1,  v48:1 |
| remove_from_inline_middle_reverse | 6 | 3/6 | 0/6 | 3/6 | 0/6 | 0/6 | 6/6 | 2.3 | 35.260 | word:3, correct_prefix:3 |  c33:2,  v91:2,  v05:1,  c59:1 |

### relation_changed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 40/48 | 11/48 | 14/48 | 0/48 | 1/48 | 48/48 | 1.2 | 75.470 | correct_prefix:40, word:7, explanation:1 |  v48:16,  v22:9,  v91:8,  v05:6,  c77:3,  c12:3,  c33:2,  No.\n\n:1 |
| inline | 48 | 33/48 | 7/48 | 11/48 | 0/48 | 8/48 | 48/48 | 1.5 | 69.482 | correct_prefix:33, explanation:8, word:7 |  v48:10,  v22:8,  v91:8,  v05:6,  Yes.\n:5,  c59:3,  No.\n:3,  c77:2 |
| to_original_middle_restore | 48 | 32/48 | 8/48 | 11/48 | 0/48 | 5/48 | 48/48 | 1.7 | 75.467 | correct_prefix:32, word:10, explanation:6 |  v48:11,  v91:8,  v22:7,  v05:6,  c12:4,  Yes.\n\n:3,  c59:2,  c77:2 |
| to_original_middle_random | 48 | 34/48 | 9/48 | 10/48 | 0/48 | 0/48 | 48/48 | 1.5 | 71.493 | correct_prefix:34, word:14 |  v48:12,  v91:8,  v22:7,  v05:6,  c77:5,  c59:3,  c33:3,  c12:3 |
| to_original_middle_reverse | 48 | 21/48 | 2/48 | 8/48 | 0/48 | 0/48 | 48/48 | 1.6 | 85.288 | word:27, correct_prefix:21 |  c77:9,  c12:7,  v48:6,  v22:5,  v91:5,  v05:4,  c59:4,  c33:4 |
| remove_from_inline_middle_restore | 48 | 44/48 | 13/48 | 16/48 | 0/48 | 0/48 | 48/48 | 1.1 | 71.475 | correct_prefix:44, word:4 |  v48:17,  v22:12,  v91:8,  v05:7,  c33:2,  o82:1,  c77:1 |
| remove_from_inline_middle_random | 48 | 29/48 | 5/48 | 11/48 | 0/48 | 8/48 | 48/48 | 1.6 | 65.417 | correct_prefix:29, word:11, explanation:8 |  v48:11,  v91:7,  Yes.\n:6,  v05:5,  v22:5,  c59:4,  c12:4,  c77:3 |
| remove_from_inline_middle_reverse | 48 | 20/48 | 5/48 | 8/48 | 0/48 | 5/48 | 48/48 | 2.8 | 29.112 | correct_prefix:20, word:17, explanation:11 |  c59:8,  v48:6,  c77:6,  v22:5,  Yes.\n:4,  v91:4,  v05:4,  No.:4 |

### explanation_needed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 22/48 | 39.2 | 99.000 | explanation:29, word:19 |  The combination:8,  c12:7,  The relationship:6,  c59:6,  c77:5,  The cell:4,  The c:4,  c33:4 |
| inline | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 11/48 | 109.7 | 99.000 | explanation:38, word:10 |  The relationship:12,  The question:11,  The answer:5,  The given:3,  c12:3,  c59:3,  The instruction:2, The relationship:2 |
| to_original_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 2/48 | 11/48 | 229.6 | 99.000 | explanation:37, word:11 |  The instruction:10,  The expression:10,  The relationship:7,  The cell:4,  The combination:3,  Explanation::2,  The query:2,  c12:2 |
| to_original_middle_random | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 27/48 | 69.8 | 94.826 | word:26, explanation:22 |  c12:8,  c33:7,  c77:6,  c59:6,  The combination:5,  The relationship:4,  The expression:3,  The instruction:2 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 24/48 | 73.7 | 96.906 | explanation:26, word:22 |  The combination:9,  c59:8,  The relationship:7,  c77:7,  c33:6,  The expression:4,  c12:3,  The cell:2 |
| remove_from_inline_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 13/48 | 44.1 | 99.000 | explanation:36, word:12 |  The relationship:17,  The given:7,  The question:5,  c12:4,  The answer:3,  c59:3,  c77:3,  c33:3 |
| remove_from_inline_middle_random | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 5/48 | 28/48 | 119.9 | 99.000 | explanation:27, word:21 |  c12:7,  The question:5,  The given:5, Yes,:4,  c59:4, The notation:4,  The answer:3,  c77:3 |
| remove_from_inline_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 45/48 | 48/48 | 234.1 | 94.952 | word:45, explanation:3 | Explanation:\n:30, Explanation:\n\n:8,  Explanation:\n\n:7, Yes,:3 |

### non_value

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 36/48 | 48/48 | 163.5 | 41.630 | explanation:48 |  Yes\n\n:35,  Yes.:12,  Yes.\n\n:1 |
| inline | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 43/48 | 48/48 | 178.3 | 50.108 | explanation:48 |  Yes\n\n:18,  Yes.\n\n:12,  Yes\n:10,  Yes.:5,  Yes.\n:2, Yes\n\n:1 |
| to_original_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 23/48 | 48/48 | 328.8 | 94.712 | explanation:48 |  Yes.:25,  Yes.\n\n:15,  Yes\n\n:7, Yes.\n\n:1 |
| to_original_middle_random | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 38/48 | 48/48 | 149.5 | 62.891 | explanation:48 |  Yes\n\n:33,  Yes.:10,  Yes.\n\n:4,  yes\n\n:1 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 48/48 | 48/48 | 88.2 | 99.000 | explanation:48 |  Yes\n\n:40,  yes\n\n:7, Yes\n\n:1 |
| remove_from_inline_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 48/48 | 48/48 | 121.7 | 14.271 | explanation:48 |  Yes\n\n:48 |
| remove_from_inline_middle_random | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 41/48 | 48/48 | 180.6 | 52.204 | explanation:48 |  Yes\n\n:17,  Yes.\n\n:13,  Yes\n:9,  Yes.:6, Yes\n\n:1, Yes.:1,  Yes.\n:1 |
| remove_from_inline_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 38/48 | 48/48 | 211.9 | 39.322 | explanation:48 |  Yes.\n\n:20,  Yes.:10,  Yes.\n:7, Yes\n\n:4,  Yes\n:4,  Yes\n\n:3 |

### Boundary Notes

- target_failure: original exact/old=29/36, to_original_middle_restore exact/old=27/36, newline 0->0
- target_failure: inline exact/old=27/36, remove_from_inline_middle_restore exact/old=31/36, newline 0->0
- original_correct: original exact/old=40/48, to_original_middle_restore exact/old=33/48, newline 0->0
- original_correct: inline exact/old=33/48, remove_from_inline_middle_restore exact/old=44/48, newline 0->0
- inline_bad: original exact/old=3/6, to_original_middle_restore exact/old=3/6, newline 0->0
- inline_bad: inline exact/old=1/6, remove_from_inline_middle_restore exact/old=2/6, newline 0->0
- relation_changed: original exact/old=11/48, to_original_middle_restore exact/old=8/48, newline 0->0
- relation_changed: inline exact/old=7/48, remove_from_inline_middle_restore exact/old=13/48, newline 0->0
- explanation_needed: original exact/old=0/48, to_original_middle_restore exact/old=0/48, newline 0->0
- explanation_needed: inline exact/old=0/48, remove_from_inline_middle_restore exact/old=0/48, newline 0->0
- non_value: original exact/old=0/48, to_original_middle_restore exact/old=0/48, newline 0->0
- non_value: inline exact/old=0/48, remove_from_inline_middle_restore exact/old=0/48, newline 0->0

## deepseek7b

- raw_cases: 320 / selected_items: 241 / mode_rows: 1928
- component: `layer_out` / layers: `[18, 19]` / top_k: 20
- max_per_split: 48 / max_new_tokens: 3
- selection_stats: `{'counts': {'target_failure': 48, 'original_correct': 48, 'inline_bad': 1, 'relation_changed': 48, 'explanation_needed': 48, 'non_value': 48}, 'target_failure_seen': 74, 'original_correct_seen': 146, 'inline_bad_seen': 1, 'relation_pool_size': 2}`
- filtered: `{'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 5.65

### target_failure

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 12/48 | 12/48 | 0/48 | 34/48 | 34/48 | 14/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:26,  ?\n\nI think:8,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| inline | 48 | 47/48 | 45/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_middle_restore | 48 | 45/48 | 45/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_middle_random | 48 | 11/48 | 10/48 | 1/48 | 24/48 | 24/48 | 23/48 | 7.6 | -1.478 | newline:24, correct_prefix:11, space:6, word:6, explanation:1 |  ?\n\nTo solve:15,  ?\n\nI think:9,  64:6,  v48:5,  v05:4,  c77:3,  o71:2,  v91:2 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 30/48 | 27/48 | 20/48 | 306.8 | -6.904 | newline:30, word:12, space:4, explanation:2 |  ?\n\nTo solve:26,  c59:6,  64:4,  c33:3,  c77:2,  yes, because:2,  11:1,  71:1 |
| remove_from_inline_middle_restore | 48 | 15/48 | 14/48 | 0/48 | 30/48 | 29/48 | 19/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_middle_random | 48 | 47/48 | 47/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.273 | correct_prefix:47, space:1 |  v48:16,  v22:12,  v05:12,  v91:7,  22:1 |
| remove_from_inline_middle_reverse | 48 | 46/48 | 45/48 | 1/48 | 0/48 | 0/48 | 48/48 | 1.0 | 3.880 | correct_prefix:46, space:2 |  v48:15,  v05:13,  v22:11,  v91:7,  22:2 |

### original_correct

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 9/48 | 8/48 | 0/48 | 36/48 | 37/48 | 11/48 | 5.9 | -1.243 | newline:36, correct_prefix:9, explanation:2, word:1 |  ?\n\nTo solve:24,  ?\n\nI think:11,  v48:5,  v05:3,  c77:1,  yes.\n\nc:1,  o06:1,  yes.\n\nQuestion:1 |
| inline | 48 | 47/48 | 47/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.327 | correct_prefix:47, space:1 |  v48:28,  v22:9,  v05:7,  v91:3,  22:1 |
| to_original_middle_restore | 48 | 48/48 | 48/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.413 | correct_prefix:48 |  v48:28,  v22:10,  v05:7,  v91:3 |
| to_original_middle_random | 48 | 8/48 | 9/48 | 0/48 | 31/48 | 30/48 | 18/48 | 7.8 | -1.363 | newline:31, correct_prefix:8, word:4, explanation:3, space:2 |  ?\n\nTo solve:19,  ?\n\nI think:7,  v48:6,  o58:2,  c12:2,  yes.\n\nc:2,  64:2,  v05:1 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 34/48 | 35/48 | 12/48 | 304.0 | -6.594 | newline:34, word:8, space:5, explanation:1 |  ?\n\nTo solve:27,  ?\n\nI think:7,  c12:3,  64:3,  c77:2,  belongs to c:1,  c59:1,  Yes.\n\nc:1 |
| remove_from_inline_middle_restore | 48 | 15/48 | 14/48 | 0/48 | 31/48 | 31/48 | 17/48 | 3.3 | -0.727 | newline:31, correct_prefix:15, word:1, space:1 |  ?\n\nTo solve:30,  v48:9,  v05:3,  v91:2,  c77:1,  o06:1,  91:1,  ?\n\nI think:1 |
| remove_from_inline_middle_random | 48 | 44/48 | 44/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.1 | 2.211 | correct_prefix:44, space:3, word:1 |  v48:28,  v22:8,  v05:6,  v91:2,  22:2,  o06:1,  91:1 |
| remove_from_inline_middle_reverse | 48 | 46/48 | 46/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.0 | 3.961 | correct_prefix:46, space:2 |  v48:28,  v22:8,  v05:7,  v91:3,  22:2 |

### inline_bad

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 3.0 | -1.250 | newline:1 |  ?\n\nTo solve:1 |
| inline | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 2.312 | correct_prefix:1 |  v22:1 |
| to_original_middle_restore | 1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 | 1.0 | 2.500 | correct_prefix:1 |  v91:1 |
| to_original_middle_random | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 3.0 | -0.750 | space:1 |  82:1 |
| to_original_middle_reverse | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 190.0 | -7.188 | newline:1 |  ?\n\nTo solve:1 |
| remove_from_inline_middle_restore | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/1 | 3.0 | -1.000 | newline:1 |  ?\n\nTo solve:1 |
| remove_from_inline_middle_random | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 2.062 | correct_prefix:1 |  v22:1 |
| remove_from_inline_middle_reverse | 1 | 1/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1.0 | 3.938 | correct_prefix:1 |  v22:1 |

### relation_changed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 11/48 | 4/48 | 5/48 | 32/48 | 34/48 | 14/48 | 8.6 | -1.536 | newline:32, correct_prefix:11, word:3, explanation:2 |  ?\n\nTo solve:25,  ?\n\nI think:7,  v48:6,  v05:4,  c77:2,  yes.\n\nc:1,  o95:1,  yes.\n\nQuestion:1 |
| inline | 48 | 48/48 | 17/48 | 22/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.355 | correct_prefix:48 |  v48:20,  v22:11,  v05:11,  v91:6 |
| to_original_middle_restore | 48 | 48/48 | 17/48 | 23/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.486 | correct_prefix:48 |  v48:21,  v22:11,  v05:10,  v91:6 |
| to_original_middle_random | 48 | 13/48 | 8/48 | 4/48 | 27/48 | 27/48 | 21/48 | 8.8 | -1.426 | newline:27, correct_prefix:13, word:5, space:2, explanation:1 |  ?\n\nTo solve:22,  v48:9,  ?\n\nI think:4,  v05:4,  64:2,  c77:2,  o58:1,  c12:1 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 34/48 | 35/48 | 12/48 | 379.4 | -7.225 | newline:34, word:11, space:2, explanation:1 |  ?\n\nTo solve:29,  ?\n\nI think:5,  c77:3,  c59:3,  c12:2,  64:2,  c33:2,  belongs to c:1 |
| remove_from_inline_middle_restore | 48 | 15/48 | 6/48 | 6/48 | 30/48 | 30/48 | 18/48 | 3.8 | -0.913 | newline:30, correct_prefix:15, word:2, space:1 |  ?\n\nTo solve:29,  v48:8,  v05:4,  c77:2,  v91:2,  91:1,  o06:1,  ?\n\nI think:1 |
| remove_from_inline_middle_random | 48 | 45/48 | 16/48 | 21/48 | 2/48 | 1/48 | 47/48 | 1.0 | 2.366 | correct_prefix:45, newline:2, space:1 |  v48:19,  v22:11,  v05:10,  v91:6,  ?\n\nTo solve:1,  22:1 |
| remove_from_inline_middle_reverse | 48 | 46/48 | 16/48 | 21/48 | 0/48 | 0/48 | 48/48 | 1.0 | 4.010 | correct_prefix:46, space:2 |  v48:20,  v22:10,  v05:10,  v91:6,  22:2 |

### explanation_needed

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 34/48 | 44.4 | -3.805 | word:31, explanation:17 |  c33:11,  c59:10,  c12:8,  The value at:7,  c77:5,  The value of:2,  The radius of:2,  The answer is:1 |
| inline | 48 | 46/48 | 44/48 | 1/48 | 2/48 | 2/48 | 46/48 | 1.1 | 1.531 | correct_prefix:46, newline:2 |  v48:18,  v22:12,  v05:10,  v91:5,  ?\n\nTo solve:2,  48:1 |
| to_original_middle_restore | 48 | 43/48 | 43/48 | 0/48 | 0/48 | 0/48 | 48/48 | 1.2 | 1.801 | correct_prefix:43, space:5 |  v48:17,  v22:12,  v05:10,  v91:4,  48:3,  91:2 |
| to_original_middle_random | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 26/48 | 47.5 | -3.581 | explanation:26, word:22 |  The value at:12,  c33:8,  c59:7,  c12:5,  c77:5,  The r6:3,  The question is:3,  The value of:2 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 25/48 | 726.1 | -9.122 | explanation:30, word:18 |  c59:10,  The question is:7,  c12:7,  The value at:6,  c33:5,  The r3:4,  c77:3,  The radius of:2 |
| remove_from_inline_middle_restore | 48 | 0/48 | 0/48 | 0/48 | 2/48 | 1/48 | 5/48 | 12.1 | -2.029 | explanation:43, word:3, newline:2 |  The value at:25,  The value is:12,  c33:3,  The question is:2,  The value associated:2,  The value of:2,  \n\n</think>\n\n:1,  c59:1 |
| remove_from_inline_middle_random | 48 | 41/48 | 40/48 | 1/48 | 3/48 | 3/48 | 45/48 | 1.2 | 1.250 | correct_prefix:41, space:4, newline:3 |  v48:17,  v22:11,  v05:9,  v91:4,  ?\n\nTo solve:3,  48:2,  22:1,  91:1 |
| remove_from_inline_middle_reverse | 48 | 48/48 | 47/48 | 1/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.521 | correct_prefix:48 |  v48:21,  v22:12,  v05:10,  v91:5 |

### non_value

| mode | n | tok0 | exact/old_exact | wrong_exact | newline_top0 | gen_newline | gen_short | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 48 | 0/48 | 0/48 | 0/48 | 28/48 | 44/48 | 0/48 | 55.2 | -4.135 | newline:28, explanation:20 |  ?\n\nOkay,:26,  yes.\n\nQuestion:10,  yes\nQuestion:5,  yes\n</think>:3,  yes, because:2,  yes or no:2 |
| inline | 48 | 38/48 | 36/48 | 2/48 | 8/48 | 9/48 | 38/48 | 1.3 | 0.846 | correct_prefix:38, newline:8, explanation:2 |  v48:16,  v22:8,  v05:8,  ?\n\nOkay,:7,  v91:6,  yes.\n\nQuestion:2,  yes or no:1 |
| to_original_middle_restore | 48 | 23/48 | 23/48 | 2/48 | 8/48 | 23/48 | 25/48 | 2.3 | 0.293 | correct_prefix:23, explanation:17, newline:8 |  yes.\n\nQuestion:15,  v48:11,  ?\n\nOkay,:8,  v22:5,  v05:5,  v91:4 |
| to_original_middle_random | 48 | 0/48 | 0/48 | 0/48 | 26/48 | 41/48 | 0/48 | 48.3 | -4.011 | newline:26, explanation:22 |  ?\n\nOkay,:25,  yes.\n\nQuestion:8,  yes\nQuestion:5,  yes or no:5,  yes\n</think>:3,  yes, because:2 |
| to_original_middle_reverse | 48 | 0/48 | 0/48 | 0/48 | 34/48 | 42/48 | 0/48 | 414.2 | -9.229 | newline:34, explanation:14 |  ?\n\nOkay,:33,  yes\nQuestion:7,  yes, because:3,  yes or no:3,  yes\n</think>:2 |
| remove_from_inline_middle_restore | 48 | 4/48 | 4/48 | 0/48 | 34/48 | 36/48 | 4/48 | 14.2 | -2.241 | newline:34, explanation:10, correct_prefix:4 |  ?\n\nOkay,:30,  yes or no:8,  yes.\n\nQuestion:5,  v48:4,  ?\n\nTo solve:1 |
| remove_from_inline_middle_random | 48 | 33/48 | 33/48 | 2/48 | 13/48 | 12/48 | 35/48 | 1.4 | 0.621 | correct_prefix:33, newline:13, explanation:2 |  v48:16,  ?\n\nOkay,:11,  v05:7,  v22:6,  v91:6,  yes or no:1,  yes.\n\nQuestion:1 |
| remove_from_inline_middle_reverse | 48 | 48/48 | 45/48 | 3/48 | 0/48 | 0/48 | 48/48 | 1.0 | 2.779 | correct_prefix:48 |  v48:22,  v22:11,  v05:9,  v91:6 |

### Boundary Notes

- target_failure: original exact/old=12/48, to_original_middle_restore exact/old=45/48, newline 34->0
- target_failure: inline exact/old=45/48, remove_from_inline_middle_restore exact/old=14/48, newline 0->30
- original_correct: original exact/old=8/48, to_original_middle_restore exact/old=48/48, newline 36->0
- original_correct: inline exact/old=47/48, remove_from_inline_middle_restore exact/old=14/48, newline 0->31
- inline_bad: original exact/old=0/1, to_original_middle_restore exact/old=1/1, newline 1->0
- inline_bad: inline exact/old=0/1, remove_from_inline_middle_restore exact/old=0/1, newline 0->1
- relation_changed: original exact/old=4/48, to_original_middle_restore exact/old=17/48, newline 32->0
- relation_changed: inline exact/old=17/48, remove_from_inline_middle_restore exact/old=6/48, newline 0->30
- explanation_needed: original exact/old=0/48, to_original_middle_restore exact/old=43/48, newline 0->0
- explanation_needed: inline exact/old=44/48, remove_from_inline_middle_restore exact/old=0/48, newline 2->2
- non_value: original exact/old=0/48, to_original_middle_restore exact/old=23/48, newline 28->8
- non_value: inline exact/old=36/48, remove_from_inline_middle_restore exact/old=4/48, newline 8->34
