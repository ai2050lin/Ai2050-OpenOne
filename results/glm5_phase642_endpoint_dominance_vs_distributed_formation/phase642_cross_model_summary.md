# Phase 642 Cross-Model Summary

目标：拆分 Phase 641 的强区间，比较 full/endpoint/leave-end/middle，并同时测试 to_original sufficiency 与 remove_from_inline necessity。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 17 / mode_rows: 1394
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L00_08': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'L08_16': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'L16_24': [16, 17, 18, 19, 20, 21, 22, 23, 24], 'L24_32': [24, 25, 26, 27, 28, 29, 30, 31, 32]}`
- variants: `['full', 'first', 'last', 'without_first', 'without_last', 'middle']`
- controls: `['restore', 'random', 'reverse']` / control_variants: `['full', 'last']`
- directions: `['to_original', 'remove_from_inline']`
- filtered: `{'not_target': 239, 'separator_len_mismatch': 0, 'empty_patch': 0, 'skipped_control_variant': 1088}`
- total_time_min: 1.73

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 0/17 | 1.2 | 1.272 | correct_prefix:14, space:3 |  v:14,  :3 |
| inline | 17 | 1/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |

### To Original Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | full | `0,1,2,3,4,5,6,7,8` | 17 | 3/17 | 14/17 | 3.1 | -0.853 | newline:14, correct_prefix:3 |
| L00_08 | first | `0` | 17 | 14/17 | 1/17 | 1.3 | 0.676 | correct_prefix:14, space:2, newline:1 |
| L00_08 | last | `8` | 17 | 3/17 | 14/17 | 3.1 | -0.853 | newline:14, correct_prefix:3 |
| L00_08 | without_first | `1,2,3,4,5,6,7,8` | 17 | 3/17 | 14/17 | 3.1 | -0.853 | newline:14, correct_prefix:3 |
| L00_08 | without_last | `0,1,2,3,4,5,6,7` | 17 | 2/17 | 15/17 | 3.4 | -0.831 | newline:15, correct_prefix:2 |
| L00_08 | middle | `1,2,3,4,5,6,7` | 17 | 2/17 | 15/17 | 3.4 | -0.831 | newline:15, correct_prefix:2 |
| L08_16 | full | `8,9,10,11,12,13,14,15,16` | 17 | 1/17 | 16/17 | 4.0 | -1.368 | newline:16, correct_prefix:1 |
| L08_16 | first | `8` | 17 | 3/17 | 14/17 | 3.1 | -0.853 | newline:14, correct_prefix:3 |
| L08_16 | last | `16` | 17 | 1/17 | 16/17 | 4.0 | -1.368 | newline:16, correct_prefix:1 |
| L08_16 | without_first | `9,10,11,12,13,14,15,16` | 17 | 1/17 | 16/17 | 4.0 | -1.368 | newline:16, correct_prefix:1 |
| L08_16 | without_last | `8,9,10,11,12,13,14,15` | 17 | 0/17 | 17/17 | 4.8 | -2.228 | newline:17 |
| L08_16 | middle | `9,10,11,12,13,14,15` | 17 | 0/17 | 17/17 | 4.8 | -2.228 | newline:17 |
| L16_24 | full | `16,17,18,19,20,21,22,23,24` | 17 | 0/17 | 17/17 | 4.9 | -2.301 | newline:17 |
| L16_24 | first | `16` | 17 | 1/17 | 16/17 | 4.0 | -1.368 | newline:16, correct_prefix:1 |
| L16_24 | last | `24` | 17 | 0/17 | 17/17 | 4.9 | -2.301 | newline:17 |
| L16_24 | without_first | `17,18,19,20,21,22,23,24` | 17 | 0/17 | 17/17 | 4.9 | -2.301 | newline:17 |
| L16_24 | without_last | `16,17,18,19,20,21,22,23` | 17 | 0/17 | 17/17 | 4.7 | -2.316 | newline:17 |
| L16_24 | middle | `17,18,19,20,21,22,23` | 17 | 0/17 | 17/17 | 4.7 | -2.316 | newline:17 |
| L24_32 | full | `24,25,26,27,28,29,30,31,32` | 17 | 1/17 | 13/17 | 4.8 | -1.537 | newline:13, space:3, correct_prefix:1 |
| L24_32 | first | `24` | 17 | 0/17 | 17/17 | 4.9 | -2.301 | newline:17 |
| L24_32 | last | `32` | 17 | 1/17 | 13/17 | 4.8 | -1.537 | newline:13, space:3, correct_prefix:1 |
| L24_32 | without_first | `25,26,27,28,29,30,31,32` | 17 | 1/17 | 13/17 | 4.8 | -1.537 | newline:13, space:3, correct_prefix:1 |
| L24_32 | without_last | `24,25,26,27,28,29,30,31` | 17 | 0/17 | 16/17 | 4.9 | -2.037 | newline:16, space:1 |
| L24_32 | middle | `25,26,27,28,29,30,31` | 17 | 0/17 | 16/17 | 4.9 | -2.037 | newline:16, space:1 |

### Remove From Inline Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | full | `0,1,2,3,4,5,6,7,8` | 17 | 6/17 | 0/17 | 2.6 | 0.294 | space:11, correct_prefix:6 |
| L00_08 | first | `0` | 17 | 4/17 | 4/17 | 3.9 | -0.662 | space:9, correct_prefix:4, newline:4 |
| L00_08 | last | `8` | 17 | 6/17 | 0/17 | 2.6 | 0.294 | space:11, correct_prefix:6 |
| L00_08 | without_first | `1,2,3,4,5,6,7,8` | 17 | 6/17 | 0/17 | 2.6 | 0.294 | space:11, correct_prefix:6 |
| L00_08 | without_last | `0,1,2,3,4,5,6,7` | 17 | 7/17 | 0/17 | 2.4 | 0.375 | space:10, correct_prefix:7 |
| L00_08 | middle | `1,2,3,4,5,6,7` | 17 | 7/17 | 0/17 | 2.4 | 0.375 | space:10, correct_prefix:7 |
| L08_16 | full | `8,9,10,11,12,13,14,15,16` | 17 | 9/17 | 1/17 | 2.5 | 0.162 | correct_prefix:9, space:7, newline:1 |
| L08_16 | first | `8` | 17 | 6/17 | 0/17 | 2.6 | 0.294 | space:11, correct_prefix:6 |
| L08_16 | last | `16` | 17 | 9/17 | 1/17 | 2.5 | 0.162 | correct_prefix:9, space:7, newline:1 |
| L08_16 | without_first | `9,10,11,12,13,14,15,16` | 17 | 9/17 | 1/17 | 2.5 | 0.162 | correct_prefix:9, space:7, newline:1 |
| L08_16 | without_last | `8,9,10,11,12,13,14,15` | 17 | 10/17 | 0/17 | 1.7 | 0.882 | correct_prefix:10, space:7 |
| L08_16 | middle | `9,10,11,12,13,14,15` | 17 | 10/17 | 0/17 | 1.7 | 0.882 | correct_prefix:10, space:7 |
| L16_24 | full | `16,17,18,19,20,21,22,23,24` | 17 | 13/17 | 0/17 | 1.3 | 1.184 | correct_prefix:13, space:4 |
| L16_24 | first | `16` | 17 | 9/17 | 1/17 | 2.5 | 0.162 | correct_prefix:9, space:7, newline:1 |
| L16_24 | last | `24` | 17 | 13/17 | 0/17 | 1.3 | 1.184 | correct_prefix:13, space:4 |
| L16_24 | without_first | `17,18,19,20,21,22,23,24` | 17 | 13/17 | 0/17 | 1.3 | 1.184 | correct_prefix:13, space:4 |
| L16_24 | without_last | `16,17,18,19,20,21,22,23` | 17 | 13/17 | 0/17 | 1.3 | 1.081 | correct_prefix:13, space:4 |
| L16_24 | middle | `17,18,19,20,21,22,23` | 17 | 13/17 | 0/17 | 1.3 | 1.081 | correct_prefix:13, space:4 |
| L24_32 | full | `24,25,26,27,28,29,30,31,32` | 17 | 13/17 | 1/17 | 1.6 | 0.743 | correct_prefix:13, space:3, newline:1 |
| L24_32 | first | `24` | 17 | 13/17 | 0/17 | 1.3 | 1.184 | correct_prefix:13, space:4 |
| L24_32 | last | `32` | 17 | 13/17 | 1/17 | 1.6 | 0.743 | correct_prefix:13, space:3, newline:1 |
| L24_32 | without_first | `25,26,27,28,29,30,31,32` | 17 | 13/17 | 1/17 | 1.6 | 0.743 | correct_prefix:13, space:3, newline:1 |
| L24_32 | without_last | `24,25,26,27,28,29,30,31` | 17 | 13/17 | 0/17 | 1.2 | 1.074 | correct_prefix:13, space:4 |
| L24_32 | middle | `25,26,27,28,29,30,31` | 17 | 13/17 | 0/17 | 1.2 | 1.074 | correct_prefix:13, space:4 |

### Random/Reverse Controls

| direction | interval | variant | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---|---|---|---|---:|---:|---:|---:|---:|
| remove_from_inline | L00_08 | full | random | 17 | 1/17 | 9/17 | 4.8 | -1.360 |
| remove_from_inline | L00_08 | full | reverse | 17 | 0/17 | 17/17 | 6.5 | -2.324 |
| remove_from_inline | L00_08 | last | random | 17 | 1/17 | 9/17 | 4.8 | -1.360 |
| remove_from_inline | L00_08 | last | reverse | 17 | 0/17 | 17/17 | 6.5 | -2.324 |
| remove_from_inline | L08_16 | full | random | 17 | 1/17 | 9/17 | 5.1 | -1.721 |
| remove_from_inline | L08_16 | full | reverse | 17 | 1/17 | 10/17 | 4.6 | -1.382 |
| remove_from_inline | L08_16 | last | random | 17 | 1/17 | 9/17 | 5.1 | -1.721 |
| remove_from_inline | L08_16 | last | reverse | 17 | 1/17 | 10/17 | 4.6 | -1.382 |
| remove_from_inline | L16_24 | full | random | 17 | 1/17 | 14/17 | 5.4 | -1.794 |
| remove_from_inline | L16_24 | full | reverse | 17 | 0/17 | 17/17 | 8.4 | -3.853 |
| remove_from_inline | L16_24 | last | random | 17 | 1/17 | 14/17 | 5.4 | -1.794 |
| remove_from_inline | L16_24 | last | reverse | 17 | 0/17 | 17/17 | 8.4 | -3.853 |
| remove_from_inline | L24_32 | full | random | 17 | 0/17 | 10/17 | 5.1 | -1.581 |
| remove_from_inline | L24_32 | full | reverse | 17 | 0/17 | 15/17 | 12.6 | -3.809 |
| remove_from_inline | L24_32 | last | random | 17 | 0/17 | 10/17 | 5.1 | -1.581 |
| remove_from_inline | L24_32 | last | reverse | 17 | 0/17 | 15/17 | 12.6 | -3.809 |
| to_original | L00_08 | full | random | 17 | 9/17 | 1/17 | 1.7 | 0.912 |
| to_original | L00_08 | full | reverse | 17 | 8/17 | 0/17 | 1.6 | 1.110 |
| to_original | L00_08 | last | random | 17 | 9/17 | 1/17 | 1.7 | 0.912 |
| to_original | L00_08 | last | reverse | 17 | 8/17 | 0/17 | 1.6 | 1.110 |
| to_original | L08_16 | full | random | 17 | 8/17 | 3/17 | 2.4 | 0.353 |
| to_original | L08_16 | full | reverse | 17 | 16/17 | 0/17 | 1.1 | 3.147 |
| to_original | L08_16 | last | random | 17 | 8/17 | 3/17 | 2.4 | 0.353 |
| to_original | L08_16 | last | reverse | 17 | 16/17 | 0/17 | 1.1 | 3.147 |
| to_original | L16_24 | full | random | 17 | 9/17 | 4/17 | 2.4 | 0.684 |
| to_original | L16_24 | full | reverse | 17 | 6/17 | 0/17 | 2.2 | 2.301 |
| to_original | L16_24 | last | random | 17 | 9/17 | 4/17 | 2.4 | 0.684 |
| to_original | L16_24 | last | reverse | 17 | 6/17 | 0/17 | 2.2 | 2.301 |
| to_original | L24_32 | full | random | 17 | 12/17 | 2/17 | 1.3 | 1.066 |
| to_original | L24_32 | full | reverse | 17 | 3/17 | 0/17 | 2.5 | 2.750 |
| to_original | L24_32 | last | random | 17 | 12/17 | 2/17 | 1.3 | 1.066 |
| to_original | L24_32 | last | reverse | 17 | 3/17 | 0/17 | 2.5 | 2.750 |

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 31 / mode_rows: 2542
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L00_08': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'L08_16': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'L16_24': [16, 17, 18, 19, 20, 21, 22, 23, 24], 'L24_32': [24, 25, 26, 27, 28, 29, 30, 31, 32]}`
- variants: `['full', 'first', 'last', 'without_first', 'without_last', 'middle']`
- controls: `['restore', 'random', 'reverse']` / control_variants: `['full', 'last']`
- directions: `['to_original', 'remove_from_inline']`
- filtered: `{'not_target': 225, 'separator_len_mismatch': 0, 'empty_patch': 0, 'skipped_control_variant': 1984}`
- total_time_min: 3.57

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 0/31 | 1.1 | 80.722 | correct_prefix:29, word:2 |  v:29,  c:2 |
| inline | 31 | 27/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |

### To Original Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | full | `0,1,2,3,4,5,6,7,8` | 31 | 29/31 | 0/31 | 1.1 | 80.696 | correct_prefix:29, explanation:1, word:1 |
| L00_08 | first | `0` | 31 | 30/31 | 0/31 | 1.1 | 77.688 | correct_prefix:30, word:1 |
| L00_08 | last | `8` | 31 | 29/31 | 0/31 | 1.1 | 80.696 | correct_prefix:29, explanation:1, word:1 |
| L00_08 | without_first | `1,2,3,4,5,6,7,8` | 31 | 29/31 | 0/31 | 1.1 | 80.696 | correct_prefix:29, explanation:1, word:1 |
| L00_08 | without_last | `0,1,2,3,4,5,6,7` | 31 | 30/31 | 0/31 | 1.1 | 71.613 | correct_prefix:30, explanation:1 |
| L00_08 | middle | `1,2,3,4,5,6,7` | 31 | 30/31 | 0/31 | 1.1 | 71.613 | correct_prefix:30, explanation:1 |
| L08_16 | full | `8,9,10,11,12,13,14,15,16` | 31 | 29/31 | 0/31 | 1.2 | 89.891 | correct_prefix:29, explanation:2 |
| L08_16 | first | `8` | 31 | 29/31 | 0/31 | 1.1 | 80.696 | correct_prefix:29, explanation:1, word:1 |
| L08_16 | last | `16` | 31 | 29/31 | 0/31 | 1.2 | 89.891 | correct_prefix:29, explanation:2 |
| L08_16 | without_first | `9,10,11,12,13,14,15,16` | 31 | 29/31 | 0/31 | 1.2 | 89.891 | correct_prefix:29, explanation:2 |
| L08_16 | without_last | `8,9,10,11,12,13,14,15` | 31 | 27/31 | 0/31 | 1.3 | 83.795 | correct_prefix:27, word:3, explanation:1 |
| L08_16 | middle | `9,10,11,12,13,14,15` | 31 | 27/31 | 0/31 | 1.3 | 83.795 | correct_prefix:27, word:3, explanation:1 |
| L16_24 | full | `16,17,18,19,20,21,22,23,24` | 31 | 27/31 | 0/31 | 1.3 | 95.966 | correct_prefix:27, explanation:4 |
| L16_24 | first | `16` | 31 | 29/31 | 0/31 | 1.2 | 89.891 | correct_prefix:29, explanation:2 |
| L16_24 | last | `24` | 31 | 27/31 | 0/31 | 1.3 | 95.966 | correct_prefix:27, explanation:4 |
| L16_24 | without_first | `17,18,19,20,21,22,23,24` | 31 | 27/31 | 0/31 | 1.3 | 95.966 | correct_prefix:27, explanation:4 |
| L16_24 | without_last | `16,17,18,19,20,21,22,23` | 31 | 27/31 | 0/31 | 1.3 | 92.955 | correct_prefix:27, explanation:4 |
| L16_24 | middle | `17,18,19,20,21,22,23` | 31 | 27/31 | 0/31 | 1.3 | 92.955 | correct_prefix:27, explanation:4 |
| L24_32 | full | `24,25,26,27,28,29,30,31,32` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |
| L24_32 | first | `24` | 31 | 27/31 | 0/31 | 1.3 | 95.966 | correct_prefix:27, explanation:4 |
| L24_32 | last | `32` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |
| L24_32 | without_first | `25,26,27,28,29,30,31,32` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |
| L24_32 | without_last | `24,25,26,27,28,29,30,31` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |
| L24_32 | middle | `25,26,27,28,29,30,31` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |

### Remove From Inline Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | full | `0,1,2,3,4,5,6,7,8` | 31 | 30/31 | 0/31 | 1.1 | 74.722 | correct_prefix:30, explanation:1 |
| L00_08 | first | `0` | 31 | 29/31 | 0/31 | 1.2 | 99.000 | correct_prefix:29, word:1, explanation:1 |
| L00_08 | last | `8` | 31 | 30/31 | 0/31 | 1.1 | 74.722 | correct_prefix:30, explanation:1 |
| L00_08 | without_first | `1,2,3,4,5,6,7,8` | 31 | 30/31 | 0/31 | 1.1 | 74.722 | correct_prefix:30, explanation:1 |
| L00_08 | without_last | `0,1,2,3,4,5,6,7` | 31 | 29/31 | 0/31 | 1.1 | 71.701 | correct_prefix:29, word:1, explanation:1 |
| L00_08 | middle | `1,2,3,4,5,6,7` | 31 | 29/31 | 0/31 | 1.1 | 71.701 | correct_prefix:29, word:1, explanation:1 |
| L08_16 | full | `8,9,10,11,12,13,14,15,16` | 31 | 31/31 | 0/31 | 1.0 | 68.671 | correct_prefix:31 |
| L08_16 | first | `8` | 31 | 30/31 | 0/31 | 1.1 | 74.722 | correct_prefix:30, explanation:1 |
| L08_16 | last | `16` | 31 | 31/31 | 0/31 | 1.0 | 68.671 | correct_prefix:31 |
| L08_16 | without_first | `9,10,11,12,13,14,15,16` | 31 | 31/31 | 0/31 | 1.0 | 68.671 | correct_prefix:31 |
| L08_16 | without_last | `8,9,10,11,12,13,14,15` | 31 | 31/31 | 0/31 | 1.0 | 65.602 | correct_prefix:31 |
| L08_16 | middle | `9,10,11,12,13,14,15` | 31 | 31/31 | 0/31 | 1.0 | 65.602 | correct_prefix:31 |
| L16_24 | full | `16,17,18,19,20,21,22,23,24` | 31 | 30/31 | 0/31 | 1.1 | 71.690 | correct_prefix:30, word:1 |
| L16_24 | first | `16` | 31 | 31/31 | 0/31 | 1.0 | 68.671 | correct_prefix:31 |
| L16_24 | last | `24` | 31 | 30/31 | 0/31 | 1.1 | 71.690 | correct_prefix:30, word:1 |
| L16_24 | without_first | `17,18,19,20,21,22,23,24` | 31 | 30/31 | 0/31 | 1.1 | 71.690 | correct_prefix:30, word:1 |
| L16_24 | without_last | `16,17,18,19,20,21,22,23` | 31 | 30/31 | 0/31 | 1.1 | 74.708 | correct_prefix:30, word:1 |
| L16_24 | middle | `17,18,19,20,21,22,23` | 31 | 30/31 | 0/31 | 1.1 | 74.708 | correct_prefix:30, word:1 |
| L24_32 | full | `24,25,26,27,28,29,30,31,32` | 31 | 30/31 | 0/31 | 1.1 | 19.429 | correct_prefix:30, word:1 |
| L24_32 | first | `24` | 31 | 30/31 | 0/31 | 1.1 | 71.690 | correct_prefix:30, word:1 |
| L24_32 | last | `32` | 31 | 30/31 | 0/31 | 1.1 | 19.429 | correct_prefix:30, word:1 |
| L24_32 | without_first | `25,26,27,28,29,30,31,32` | 31 | 30/31 | 0/31 | 1.1 | 19.429 | correct_prefix:30, word:1 |
| L24_32 | without_last | `24,25,26,27,28,29,30,31` | 31 | 30/31 | 0/31 | 1.1 | 16.319 | correct_prefix:30, word:1 |
| L24_32 | middle | `25,26,27,28,29,30,31` | 31 | 30/31 | 0/31 | 1.1 | 16.319 | correct_prefix:30, word:1 |

### Random/Reverse Controls

| direction | interval | variant | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---|---|---|---|---:|---:|---:|---:|---:|
| remove_from_inline | L00_08 | full | random | 31 | 28/31 | 0/31 | 1.2 | 83.740 |
| remove_from_inline | L00_08 | full | reverse | 31 | 28/31 | 0/31 | 1.3 | 77.732 |
| remove_from_inline | L00_08 | last | random | 31 | 28/31 | 0/31 | 1.2 | 83.740 |
| remove_from_inline | L00_08 | last | reverse | 31 | 28/31 | 0/31 | 1.3 | 77.732 |
| remove_from_inline | L08_16 | full | random | 31 | 26/31 | 0/31 | 1.3 | 77.541 |
| remove_from_inline | L08_16 | full | reverse | 31 | 21/31 | 0/31 | 2.1 | 34.291 |
| remove_from_inline | L08_16 | last | random | 31 | 26/31 | 0/31 | 1.3 | 77.541 |
| remove_from_inline | L08_16 | last | reverse | 31 | 21/31 | 0/31 | 2.1 | 34.291 |
| remove_from_inline | L16_24 | full | random | 31 | 27/31 | 0/31 | 1.3 | 68.358 |
| remove_from_inline | L16_24 | full | reverse | 31 | 23/31 | 0/31 | 1.9 | 77.515 |
| remove_from_inline | L16_24 | last | random | 31 | 27/31 | 0/31 | 1.3 | 68.358 |
| remove_from_inline | L16_24 | last | reverse | 31 | 23/31 | 0/31 | 1.9 | 77.515 |
| remove_from_inline | L24_32 | full | random | 31 | 28/31 | 0/31 | 1.3 | 77.643 |
| remove_from_inline | L24_32 | full | reverse | 31 | 20/31 | 0/31 | 2.3 | 99.000 |
| remove_from_inline | L24_32 | last | random | 31 | 28/31 | 0/31 | 1.3 | 77.643 |
| remove_from_inline | L24_32 | last | reverse | 31 | 20/31 | 0/31 | 2.3 | 99.000 |
| to_original | L00_08 | full | random | 31 | 27/31 | 0/31 | 1.2 | 83.762 |
| to_original | L00_08 | full | reverse | 31 | 25/31 | 0/31 | 1.2 | 89.855 |
| to_original | L00_08 | last | random | 31 | 27/31 | 0/31 | 1.2 | 83.762 |
| to_original | L00_08 | last | reverse | 31 | 25/31 | 0/31 | 1.2 | 89.855 |
| to_original | L08_16 | full | random | 31 | 26/31 | 0/31 | 1.3 | 77.607 |
| to_original | L08_16 | full | reverse | 31 | 19/31 | 0/31 | 1.5 | 99.000 |
| to_original | L08_16 | last | random | 31 | 26/31 | 0/31 | 1.3 | 77.607 |
| to_original | L08_16 | last | reverse | 31 | 19/31 | 0/31 | 1.5 | 99.000 |
| to_original | L16_24 | full | random | 31 | 29/31 | 0/31 | 1.1 | 80.737 |
| to_original | L16_24 | full | reverse | 31 | 27/31 | 0/31 | 1.3 | 65.473 |
| to_original | L16_24 | last | random | 31 | 29/31 | 0/31 | 1.1 | 80.737 |
| to_original | L16_24 | last | reverse | 31 | 27/31 | 0/31 | 1.3 | 65.473 |
| to_original | L24_32 | full | random | 31 | 29/31 | 0/31 | 1.1 | 71.584 |
| to_original | L24_32 | full | reverse | 31 | 29/31 | 0/31 | 1.1 | 40.883 |
| to_original | L24_32 | last | random | 31 | 29/31 | 0/31 | 1.1 | 71.584 |
| to_original | L24_32 | last | reverse | 31 | 29/31 | 0/31 | 1.1 | 40.883 |

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 82 / mode_rows: 6724
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L10_14': [10, 11, 12, 13, 14], 'L14_17': [14, 15, 16, 17], 'L17_20': [17, 18, 19, 20], 'L10_20': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}`
- variants: `['full', 'first', 'last', 'without_first', 'without_last', 'middle']`
- controls: `['restore', 'random', 'reverse']` / control_variants: `['full', 'last']`
- directions: `['to_original', 'remove_from_inline']`
- filtered: `{'not_target': 174, 'separator_len_mismatch': 0, 'empty_patch': 0, 'skipped_control_variant': 5248}`
- total_time_min: 6.10

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 82 | 20/82 | 57/82 | 9.4 | -1.704 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |
| inline | 82 | 75/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |  v:75,  :7 |

### To Original Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L10_14 | full | `10,11,12,13,14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L10_14 | first | `10` | 82 | 64/82 | 12/82 | 1.8 | 1.113 | correct_prefix:64, newline:12, explanation:5, word:1 |
| L10_14 | last | `14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L10_14 | without_first | `11,12,13,14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L10_14 | without_last | `10,11,12,13` | 82 | 73/82 | 5/82 | 1.3 | 1.457 | correct_prefix:73, newline:5, explanation:3, word:1 |
| L10_14 | middle | `11,12,13` | 82 | 73/82 | 5/82 | 1.3 | 1.457 | correct_prefix:73, newline:5, explanation:3, word:1 |
| L10_20 | full | `10,11,12,13,14,15,16,17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L10_20 | first | `10` | 82 | 64/82 | 12/82 | 1.8 | 1.113 | correct_prefix:64, newline:12, explanation:5, word:1 |
| L10_20 | last | `20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L10_20 | without_first | `11,12,13,14,15,16,17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L10_20 | without_last | `10,11,12,13,14,15,16,17,18,19` | 82 | 76/82 | 1/82 | 1.1 | 2.281 | correct_prefix:76, space:5, newline:1 |
| L10_20 | middle | `11,12,13,14,15,16,17,18,19` | 82 | 76/82 | 1/82 | 1.1 | 2.281 | correct_prefix:76, space:5, newline:1 |
| L14_17 | full | `14,15,16,17` | 82 | 76/82 | 1/82 | 1.2 | 1.986 | correct_prefix:76, space:3, word:1, explanation:1, newline:1 |
| L14_17 | first | `14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L14_17 | last | `17` | 82 | 76/82 | 1/82 | 1.2 | 1.986 | correct_prefix:76, space:3, word:1, explanation:1, newline:1 |
| L14_17 | without_first | `15,16,17` | 82 | 76/82 | 1/82 | 1.2 | 1.986 | correct_prefix:76, space:3, word:1, explanation:1, newline:1 |
| L14_17 | without_last | `14,15,16` | 82 | 76/82 | 2/82 | 1.2 | 1.664 | correct_prefix:76, space:2, newline:2, word:1, explanation:1 |
| L14_17 | middle | `15,16` | 82 | 76/82 | 2/82 | 1.2 | 1.664 | correct_prefix:76, space:2, newline:2, word:1, explanation:1 |
| L17_20 | full | `17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L17_20 | first | `17` | 82 | 76/82 | 1/82 | 1.2 | 1.986 | correct_prefix:76, space:3, word:1, explanation:1, newline:1 |
| L17_20 | last | `20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L17_20 | without_first | `18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L17_20 | without_last | `17,18,19` | 82 | 76/82 | 1/82 | 1.1 | 2.281 | correct_prefix:76, space:5, newline:1 |
| L17_20 | middle | `18,19` | 82 | 76/82 | 1/82 | 1.1 | 2.281 | correct_prefix:76, space:5, newline:1 |

### Remove From Inline Restore

| interval | variant | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---|---:|---:|---:|---:|---:|---|
| L10_14 | full | `10,11,12,13,14` | 82 | 31/82 | 15/82 | 2.3 | 0.195 | space:35, correct_prefix:31, newline:15, word:1 |
| L10_14 | first | `10` | 82 | 48/82 | 3/82 | 1.5 | 1.010 | correct_prefix:48, space:30, newline:3, word:1 |
| L10_14 | last | `14` | 82 | 31/82 | 15/82 | 2.3 | 0.195 | space:35, correct_prefix:31, newline:15, word:1 |
| L10_14 | without_first | `11,12,13,14` | 82 | 31/82 | 15/82 | 2.3 | 0.195 | space:35, correct_prefix:31, newline:15, word:1 |
| L10_14 | without_last | `10,11,12,13` | 82 | 31/82 | 12/82 | 2.1 | 0.377 | space:37, correct_prefix:31, newline:12, word:2 |
| L10_14 | middle | `11,12,13` | 82 | 31/82 | 12/82 | 2.1 | 0.377 | space:37, correct_prefix:31, newline:12, word:2 |
| L10_20 | full | `10,11,12,13,14,15,16,17,18,19,20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L10_20 | first | `10` | 82 | 48/82 | 3/82 | 1.5 | 1.010 | correct_prefix:48, space:30, newline:3, word:1 |
| L10_20 | last | `20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L10_20 | without_first | `11,12,13,14,15,16,17,18,19,20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L10_20 | without_last | `10,11,12,13,14,15,16,17,18,19` | 82 | 24/82 | 53/82 | 4.2 | -1.012 | newline:53, correct_prefix:24, space:4, word:1 |
| L10_20 | middle | `11,12,13,14,15,16,17,18,19` | 82 | 24/82 | 53/82 | 4.2 | -1.012 | newline:53, correct_prefix:24, space:4, word:1 |
| L14_17 | full | `14,15,16,17` | 82 | 25/82 | 32/82 | 2.9 | -0.408 | newline:32, correct_prefix:25, space:24, word:1 |
| L14_17 | first | `14` | 82 | 31/82 | 15/82 | 2.3 | 0.195 | space:35, correct_prefix:31, newline:15, word:1 |
| L14_17 | last | `17` | 82 | 25/82 | 32/82 | 2.9 | -0.408 | newline:32, correct_prefix:25, space:24, word:1 |
| L14_17 | without_first | `15,16,17` | 82 | 25/82 | 32/82 | 2.9 | -0.408 | newline:32, correct_prefix:25, space:24, word:1 |
| L14_17 | without_last | `14,15,16` | 82 | 25/82 | 23/82 | 2.8 | -0.157 | space:34, correct_prefix:25, newline:23 |
| L14_17 | middle | `15,16` | 82 | 25/82 | 23/82 | 2.8 | -0.157 | space:34, correct_prefix:25, newline:23 |
| L17_20 | full | `17,18,19,20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L17_20 | first | `17` | 82 | 25/82 | 32/82 | 2.9 | -0.408 | newline:32, correct_prefix:25, space:24, word:1 |
| L17_20 | last | `20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L17_20 | without_first | `18,19,20` | 82 | 19/82 | 62/82 | 6.2 | -1.503 | newline:62, correct_prefix:19, word:1 |
| L17_20 | without_last | `17,18,19` | 82 | 24/82 | 53/82 | 4.2 | -1.012 | newline:53, correct_prefix:24, space:4, word:1 |
| L17_20 | middle | `18,19` | 82 | 24/82 | 53/82 | 4.2 | -1.012 | newline:53, correct_prefix:24, space:4, word:1 |

### Random/Reverse Controls

| direction | interval | variant | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---|---|---|---|---:|---:|---:|---:|---:|
| remove_from_inline | L10_14 | full | random | 82 | 76/82 | 0/82 | 1.0 | 2.144 |
| remove_from_inline | L10_14 | full | reverse | 82 | 82/82 | 0/82 | 1.0 | 3.178 |
| remove_from_inline | L10_14 | last | random | 82 | 76/82 | 0/82 | 1.0 | 2.144 |
| remove_from_inline | L10_14 | last | reverse | 82 | 82/82 | 0/82 | 1.0 | 3.178 |
| remove_from_inline | L10_20 | full | random | 82 | 73/82 | 1/82 | 1.1 | 2.202 |
| remove_from_inline | L10_20 | full | reverse | 82 | 75/82 | 0/82 | 1.1 | 4.232 |
| remove_from_inline | L10_20 | last | random | 82 | 73/82 | 1/82 | 1.1 | 2.202 |
| remove_from_inline | L10_20 | last | reverse | 82 | 75/82 | 0/82 | 1.1 | 4.232 |
| remove_from_inline | L14_17 | full | random | 82 | 73/82 | 0/82 | 1.1 | 2.041 |
| remove_from_inline | L14_17 | full | reverse | 82 | 82/82 | 0/82 | 1.0 | 3.554 |
| remove_from_inline | L14_17 | last | random | 82 | 73/82 | 0/82 | 1.1 | 2.041 |
| remove_from_inline | L14_17 | last | reverse | 82 | 82/82 | 0/82 | 1.0 | 3.554 |
| remove_from_inline | L17_20 | full | random | 82 | 73/82 | 1/82 | 1.1 | 2.202 |
| remove_from_inline | L17_20 | full | reverse | 82 | 75/82 | 0/82 | 1.1 | 4.232 |
| remove_from_inline | L17_20 | last | random | 82 | 73/82 | 1/82 | 1.1 | 2.202 |
| remove_from_inline | L17_20 | last | reverse | 82 | 75/82 | 0/82 | 1.1 | 4.232 |
| to_original | L10_14 | full | random | 82 | 16/82 | 51/82 | 12.5 | -1.921 |
| to_original | L10_14 | full | reverse | 82 | 2/82 | 61/82 | 104.6 | -5.573 |
| to_original | L10_14 | last | random | 82 | 16/82 | 51/82 | 12.5 | -1.921 |
| to_original | L10_14 | last | reverse | 82 | 2/82 | 61/82 | 104.6 | -5.573 |
| to_original | L10_20 | full | random | 82 | 16/82 | 49/82 | 13.4 | -1.710 |
| to_original | L10_20 | full | reverse | 82 | 1/82 | 61/82 | 367.0 | -7.345 |
| to_original | L10_20 | last | random | 82 | 16/82 | 49/82 | 13.4 | -1.710 |
| to_original | L10_20 | last | reverse | 82 | 1/82 | 61/82 | 367.0 | -7.345 |
| to_original | L14_17 | full | random | 82 | 13/82 | 56/82 | 9.4 | -1.706 |
| to_original | L14_17 | full | reverse | 82 | 0/82 | 63/82 | 118.1 | -5.837 |
| to_original | L14_17 | last | random | 82 | 13/82 | 56/82 | 9.4 | -1.706 |
| to_original | L14_17 | last | reverse | 82 | 0/82 | 63/82 | 118.1 | -5.837 |
| to_original | L17_20 | full | random | 82 | 16/82 | 49/82 | 13.4 | -1.710 |
| to_original | L17_20 | full | reverse | 82 | 1/82 | 61/82 | 367.0 | -7.345 |
| to_original | L17_20 | last | random | 82 | 16/82 | 49/82 | 13.4 | -1.710 |
| to_original | L17_20 | last | reverse | 82 | 1/82 | 61/82 | 367.0 | -7.345 |
