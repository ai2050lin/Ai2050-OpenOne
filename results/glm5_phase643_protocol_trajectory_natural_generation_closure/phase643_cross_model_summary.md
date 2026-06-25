# Phase 643 Cross-Model Summary

目标：把 Phase 642 的 L17-L20 protocol trajectory patch 压到 greedy natural generation，检查 exact generation、newline/explanation tendency 和生成文本分布。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 17 / mode_rows: 170
- target_only: True / top_k: 20 / max_new_tokens: 3
- component: `layer_out` / interval: `L17_20`
- full_layers: `[17, 18, 19, 20]` / middle_layers: `[18, 19]`
- filtered: `{'not_target': 239, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 1.15

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 11/17 | 3/17 | 0/17 | 1.2 | 1.272 | correct_prefix:14, space:3 |  v05:7,  v22:4,  v48:3,  22:2,  91:1 |
| inline | 17 | 1/17 | 0/17 | 0/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |  ?\n\nOkay,:9,  91:3,  05:2,  22:2,  48:1 |
| to_original_full_restore | 17 | 0/17 | 0/17 | 0/17 | 16/17 | 3.8 | -1.375 | newline:16, space:1 |  ?\n\nOkay,:15,  91:1,  22:1 |
| to_original_middle_restore | 17 | 1/17 | 1/17 | 0/17 | 14/17 | 3.6 | -1.228 | newline:14, space:2, correct_prefix:1 |  ?\n\nOkay,:14,  91:1,  v48:1,  22:1 |
| to_original_full_random | 17 | 9/17 | 6/17 | 1/17 | 4/17 | 1.9 | 1.007 | correct_prefix:9, newline:4, word:2, space:2 |  v05:5,  \n\nOkay,:3,  o58:1,  22:1,  05:1,  91:1,  v91:1,  o43:1,  48:1,  ?\n\nOkay,:1 |
| to_original_full_reverse | 17 | 11/17 | 10/17 | 1/17 | 0/17 | 1.5 | 2.787 | correct_prefix:11, word:3, space:2, explanation:1 |  v05:5,  v48:3,  22:2,  v22:2,  91:1,  v91:1,  o43:1,  o05:1,  o17:1 |
| remove_from_inline_full_restore | 17 | 8/17 | 6/17 | 1/17 | 0/17 | 2.4 | 0.346 | space:9, correct_prefix:8 |  v05:4,  22:4,  05:3,  91:3,  v48:2,  v22:1 |
| remove_from_inline_middle_restore | 17 | 7/17 | 6/17 | 1/17 | 0/17 | 2.7 | 0.191 | space:10, correct_prefix:7 |  v05:4,  22:4,  05:3,  91:3,  v48:2,  v22:1 |
| remove_from_inline_full_random | 17 | 1/17 | 1/17 | 0/17 | 10/17 | 5.2 | -1.779 | newline:10, space:6, correct_prefix:1 |  ?\n\nOkay,:7,  05:3,  91:2,  ?\n\nTo solve:2,  22:1,  48:1,  v05:1 |
| remove_from_inline_full_reverse | 17 | 0/17 | 0/17 | 0/17 | 15/17 | 6.2 | -2.618 | newline:15, space:2 |  ?\n\nOkay,:14,  91:1,  22:1,  ?\n\nTo solve:1 |

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 31 / mode_rows: 310
- target_only: True / top_k: 20 / max_new_tokens: 2
- component: `layer_out` / interval: `L17_20`
- full_layers: `[17, 18, 19, 20]` / middle_layers: `[18, 19]`
- filtered: `{'not_target': 225, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 1.98

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 28/31 | 1/31 | 0/31 | 1.1 | 80.722 | correct_prefix:29, word:2 |  v91:10,  v05:9,  v48:8,  v22:2,  c77:1,  c59:1 |
| inline | 31 | 27/31 | 26/31 | 1/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |  v05:9,  v91:9,  v48:7,  v22:3,  Yes.\n\n:2,  c77:1 |
| to_original_full_restore | 31 | 29/31 | 26/31 | 3/31 | 0/31 | 1.2 | 89.874 | correct_prefix:29, explanation:2 |  v05:10,  v91:10,  v48:7,  v22:2,  No.\n\n:1,  Yes.\n\n:1 |
| to_original_middle_restore | 31 | 29/31 | 26/31 | 3/31 | 0/31 | 1.2 | 77.691 | correct_prefix:29, explanation:2 |  v05:10,  v91:10,  v48:7,  v22:2,  No.\n\n:1,  Yes.\n\n:1 |
| to_original_full_random | 31 | 27/31 | 24/31 | 3/31 | 0/31 | 1.2 | 77.630 | correct_prefix:27, word:4 |  v91:10,  v05:9,  v48:7,  c77:1,  c33:1,  v22:1,  c12:1,  c59:1 |
| to_original_full_reverse | 31 | 25/31 | 22/31 | 1/31 | 0/31 | 1.3 | 95.929 | correct_prefix:25, word:6 |  v91:8,  v05:7,  v48:7,  c77:2,  c12:2,  c59:2,  c33:1,  v22:1,  o17:1 |
| remove_from_inline_full_restore | 31 | 31/31 | 28/31 | 2/31 | 0/31 | 1.0 | 65.643 | correct_prefix:31 |  v91:11,  v05:10,  v48:7,  v22:3 |
| remove_from_inline_middle_restore | 31 | 31/31 | 28/31 | 2/31 | 0/31 | 1.0 | 65.584 | correct_prefix:31 |  v91:11,  v05:10,  v48:7,  v22:3 |
| remove_from_inline_full_random | 31 | 25/31 | 23/31 | 0/31 | 0/31 | 1.4 | 56.181 | correct_prefix:25, explanation:4, word:2 |  v05:9,  v91:7,  v48:5,  v22:3,  Yes.\n\n:3,  c33:2,  Yes.\n:1,  c59:1 |
| remove_from_inline_full_reverse | 31 | 12/31 | 11/31 | 0/31 | 0/31 | 2.9 | 40.469 | word:13, correct_prefix:12, explanation:6 |  c33:6,  v05:5,  c12:4,  c59:3,  v91:3,  v22:2,  Yes.\n:2,  No.:1,  v48:1,  Yes.\n\n:1 |

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 82 / mode_rows: 820
- target_only: True / top_k: 20 / max_new_tokens: 3
- component: `layer_out` / interval: `L17_20`
- full_layers: `[17, 18, 19, 20]` / middle_layers: `[18, 19]`
- filtered: `{'not_target': 174, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 3.01

| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | generation_text |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| original | 82 | 20/82 | 20/82 | 0/82 | 57/82 | 9.4 | -1.704 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\nTo solve:42,  ?\n\nI think:14,  v48:12,  v05:6,  c77:1,  48:1,  v22:1,  o06:1,  o71:1,  o17:1 |
| inline | 82 | 75/82 | 72/82 | 0/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |  v48:25,  v22:20,  v05:18,  v91:9,  05:5,  22:4,  91:1 |
| to_original_full_restore | 82 | 77/82 | 72/82 | 3/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |  v48:24,  v22:21,  v05:21,  v91:9,  05:3,  22:2,  64:1,  91:1 |
| to_original_middle_restore | 82 | 76/82 | 76/82 | 0/82 | 1/82 | 1.1 | 2.281 | correct_prefix:76, space:5, newline:1 |  v48:25,  v22:22,  v05:19,  v91:10,  05:3,  22:2,  ?\n\nTo solve:1 |
| to_original_full_random | 82 | 14/82 | 13/82 | 1/82 | 51/82 | 10.6 | -1.708 | newline:51, correct_prefix:14, word:8, space:8, explanation:1 |  ?\n\nTo solve:47,  64:7,  v48:6,  v05:6,  c77:2,  c59:2,  48:2,  ?\n\nI think:2,  o71:1,  71:1 |
| to_original_full_reverse | 82 | 1/82 | 0/82 | 1/82 | 61/82 | 367.0 | -7.345 | newline:61, word:15, explanation:4, correct_prefix:1, space:1 |  ?\n\nTo solve:55,  c59:4,  yes, because:4,  belongs to which:4,  ?\n\nI think:3,  c77:2,  c33:2,  c12:2,  ?\n\nThe question:2,  o17:1 |
| remove_from_inline_full_restore | 82 | 19/82 | 20/82 | 0/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |  ?\n\nTo solve:60,  v48:12,  v05:6,  c77:1,  v22:1,  ?\n\nI think:1,  v91:1 |
| remove_from_inline_middle_restore | 82 | 24/82 | 24/82 | 0/82 | 53/82 | 4.2 | -1.012 | newline:53, correct_prefix:24, space:4, word:1 |  ?\n\nTo solve:50,  v48:13,  v05:6,  v91:3,  48:3,  v22:2,  22:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_full_random | 82 | 73/82 | 71/82 | 2/82 | 0/82 | 1.1 | 2.225 | correct_prefix:73, space:9 |  v05:23,  v48:22,  v22:19,  v91:9,  22:5,  05:2,  48:1,  91:1 |
| remove_from_inline_full_reverse | 82 | 75/82 | 73/82 | 2/82 | 0/82 | 1.1 | 4.232 | correct_prefix:75, space:7 |  v05:25,  v48:21,  v22:20,  v91:9,  22:4,  48:2,  91:1 |
