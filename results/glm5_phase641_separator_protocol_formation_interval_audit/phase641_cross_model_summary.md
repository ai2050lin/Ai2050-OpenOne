# Phase 641 Cross-Model Summary

目标：把 inline separator 的 residual protocol trajectory 按层区间恢复到 original prompt，审计 protocol state 的形成/携带区间，并用 random/reverse 控制排除普通扰动解释。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 17 / mode_rows: 340
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L00_08': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'L08_16': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'L16_24': [16, 17, 18, 19, 20, 21, 22, 23, 24], 'L24_32': [24, 25, 26, 27, 28, 29, 30, 31, 32], 'L32_35': [32, 33, 34, 35], 'L24_35': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]}`
- controls: `['restore', 'random', 'reverse']`
- filtered: `{'not_target': 239, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 1.06

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 0/17 | 1.2 | 1.272 | correct_prefix:14, space:3 |  v:14,  :3 |
| inline | 17 | 1/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |

### restore

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 17 | 3/17 | 14/17 | 3.1 | -0.853 | newline:14, correct_prefix:3 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 17 | 1/17 | 16/17 | 4.0 | -1.368 | newline:16, correct_prefix:1 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 17 | 0/17 | 17/17 | 4.9 | -2.301 | newline:17 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 17 | 1/17 | 13/17 | 4.8 | -1.537 | newline:13, space:3, correct_prefix:1 |
| L24_35 | `24,25,26,27,28,29,30,31,32,33,34,35` | 17 | 1/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |
| L32_35 | `32,33,34,35` | 17 | 1/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |

### random

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 17 | 11/17 | 2/17 | 1.7 | 0.904 | correct_prefix:11, space:4, newline:2 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 17 | 9/17 | 0/17 | 1.5 | 1.846 | correct_prefix:9, space:8 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 17 | 9/17 | 7/17 | 2.0 | 0.316 | correct_prefix:9, newline:7, space:1 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 17 | 13/17 | 0/17 | 1.4 | 1.228 | correct_prefix:13, space:4 |
| L24_35 | `24,25,26,27,28,29,30,31,32,33,34,35` | 17 | 8/17 | 3/17 | 2.2 | 0.919 | correct_prefix:8, space:3, newline:3, word:3 |
| L32_35 | `32,33,34,35` | 17 | 8/17 | 3/17 | 2.2 | 0.919 | correct_prefix:8, space:3, newline:3, word:3 |

### reverse

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 17 | 8/17 | 0/17 | 1.6 | 1.110 | space:9, correct_prefix:8 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 17 | 16/17 | 0/17 | 1.1 | 3.147 | correct_prefix:16, space:1 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 17 | 6/17 | 0/17 | 2.2 | 2.301 | word:10, correct_prefix:6, explanation:1 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 17 | 3/17 | 0/17 | 2.5 | 2.750 | word:9, explanation:5, correct_prefix:3 |
| L24_35 | `24,25,26,27,28,29,30,31,32,33,34,35` | 17 | 4/17 | 0/17 | 3.0 | 3.272 | word:11, correct_prefix:4, explanation:2 |
| L32_35 | `32,33,34,35` | 17 | 4/17 | 0/17 | 3.0 | 3.272 | word:11, correct_prefix:4, explanation:2 |

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 31 / mode_rows: 620
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L00_08': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'L08_16': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'L16_24': [16, 17, 18, 19, 20, 21, 22, 23, 24], 'L24_32': [24, 25, 26, 27, 28, 29, 30, 31, 32], 'L32_39': [32, 33, 34, 35, 36, 37, 38, 39], 'L24_39': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]}`
- controls: `['restore', 'random', 'reverse']`
- filtered: `{'not_target': 225, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 1.89

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 0/31 | 1.1 | 80.722 | correct_prefix:29, word:2 |  v:29,  c:2 |
| inline | 31 | 27/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |

### restore

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 31 | 29/31 | 0/31 | 1.1 | 80.696 | correct_prefix:29, explanation:1, word:1 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 31 | 29/31 | 0/31 | 1.2 | 89.891 | correct_prefix:29, explanation:2 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 31 | 27/31 | 0/31 | 1.3 | 95.966 | correct_prefix:27, explanation:4 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 31 | 27/31 | 0/31 | 1.3 | 99.000 | correct_prefix:27, explanation:3, word:1 |
| L24_39 | `24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39` | 31 | 27/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |
| L32_39 | `32,33,34,35,36,37,38,39` | 31 | 27/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |

### random

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 31 | 28/31 | 0/31 | 1.1 | 77.715 | correct_prefix:28, word:3 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 31 | 26/31 | 0/31 | 1.3 | 89.867 | correct_prefix:26, word:4, explanation:1 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 31 | 29/31 | 0/31 | 1.1 | 77.659 | correct_prefix:29, word:2 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 31 | 28/31 | 0/31 | 1.2 | 83.783 | correct_prefix:28, word:3 |
| L24_39 | `24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39` | 31 | 26/31 | 0/31 | 1.4 | 77.622 | correct_prefix:26, word:5 |
| L32_39 | `32,33,34,35,36,37,38,39` | 31 | 26/31 | 0/31 | 1.4 | 77.622 | correct_prefix:26, word:5 |

### reverse

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 31 | 25/31 | 0/31 | 1.2 | 89.855 | correct_prefix:25, word:6 |
| L08_16 | `8,9,10,11,12,13,14,15,16` | 31 | 19/31 | 0/31 | 1.5 | 99.000 | correct_prefix:19, word:12 |
| L16_24 | `16,17,18,19,20,21,22,23,24` | 31 | 27/31 | 0/31 | 1.3 | 65.473 | correct_prefix:27, word:4 |
| L24_32 | `24,25,26,27,28,29,30,31,32` | 31 | 29/31 | 0/31 | 1.1 | 40.883 | correct_prefix:29, word:2 |
| L24_39 | `24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39` | 31 | 26/31 | 0/31 | 1.4 | 86.758 | correct_prefix:26, word:5 |
| L32_39 | `32,33,34,35,36,37,38,39` | 31 | 26/31 | 0/31 | 1.4 | 86.758 | correct_prefix:26, word:5 |

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 82 / mode_rows: 2870
- target_only: True / top_k: 20
- component: `layer_out`
- intervals: `{'L00_08': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'L08_12': [8, 9, 10, 11, 12], 'L12_14': [12, 13, 14], 'L14_17': [14, 15, 16, 17], 'L17_20': [17, 18, 19, 20], 'L20_23': [20, 21, 22, 23], 'L23_27': [23, 24, 25, 26, 27], 'L10_14': [10, 11, 12, 13, 14], 'L10_20': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'L14_20': [14, 15, 16, 17, 18, 19, 20], 'L14_27': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}`
- controls: `['restore', 'random', 'reverse']`
- filtered: `{'not_target': 174, 'separator_len_mismatch': 0, 'empty_patch': 0}`
- total_time_min: 3.32

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| inline | 82 | 75/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |  v:75,  :7 |
| original | 82 | 20/82 | 57/82 | 9.4 | -1.704 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |

### restore

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 82 | 60/82 | 12/82 | 2.3 | 1.042 | correct_prefix:60, newline:12, explanation:9, word:1 |
| L08_12 | `8,9,10,11,12` | 82 | 62/82 | 17/82 | 1.9 | 0.910 | correct_prefix:62, newline:17, explanation:2, word:1 |
| L10_14 | `10,11,12,13,14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L10_20 | `10,11,12,13,14,15,16,17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L12_14 | `12,13,14` | 82 | 76/82 | 2/82 | 1.2 | 1.540 | correct_prefix:76, explanation:3, newline:2, word:1 |
| L14_17 | `14,15,16,17` | 82 | 76/82 | 1/82 | 1.2 | 1.986 | correct_prefix:76, space:3, word:1, explanation:1, newline:1 |
| L14_20 | `14,15,16,17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L14_27 | `14,15,16,17,18,19,20,21,22,23,24,25,26,27` | 82 | 75/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |
| L17_20 | `17,18,19,20` | 82 | 77/82 | 0/82 | 1.1 | 2.419 | correct_prefix:77, space:5 |
| L20_23 | `20,21,22,23` | 82 | 72/82 | 0/82 | 1.1 | 2.534 | correct_prefix:72, space:9, word:1 |
| L23_27 | `23,24,25,26,27` | 82 | 75/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |

### random

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 82 | 25/82 | 38/82 | 8.8 | -1.220 | newline:38, correct_prefix:25, space:9, word:8, explanation:2 |
| L08_12 | `8,9,10,11,12` | 82 | 13/82 | 56/82 | 13.6 | -2.197 | newline:56, correct_prefix:13, word:5, space:5, explanation:3 |
| L10_14 | `10,11,12,13,14` | 82 | 21/82 | 49/82 | 12.3 | -1.854 | newline:49, correct_prefix:21, space:5, explanation:5, word:2 |
| L10_20 | `10,11,12,13,14,15,16,17,18,19,20` | 82 | 17/82 | 49/82 | 9.8 | -1.636 | newline:49, correct_prefix:17, word:8, space:8 |
| L12_14 | `12,13,14` | 82 | 21/82 | 49/82 | 12.3 | -1.854 | newline:49, correct_prefix:21, space:5, explanation:5, word:2 |
| L14_17 | `14,15,16,17` | 82 | 17/82 | 56/82 | 9.8 | -1.645 | newline:56, correct_prefix:17, word:4, space:3, explanation:2 |
| L14_20 | `14,15,16,17,18,19,20` | 82 | 17/82 | 49/82 | 9.8 | -1.636 | newline:49, correct_prefix:17, word:8, space:8 |
| L14_27 | `14,15,16,17,18,19,20,21,22,23,24,25,26,27` | 82 | 15/82 | 37/82 | 11.7 | -1.716 | newline:37, space:17, correct_prefix:15, explanation:7, word:6 |
| L17_20 | `17,18,19,20` | 82 | 17/82 | 49/82 | 9.8 | -1.636 | newline:49, correct_prefix:17, word:8, space:8 |
| L20_23 | `20,21,22,23` | 82 | 17/82 | 50/82 | 10.4 | -1.781 | newline:50, correct_prefix:17, word:8, space:5, explanation:2 |
| L23_27 | `23,24,25,26,27` | 82 | 15/82 | 37/82 | 11.7 | -1.716 | newline:37, space:17, correct_prefix:15, explanation:7, word:6 |

### reverse

| interval | layers | n | tok0 | newline_top0 | rank | prefix-newline | top0_category |
|---|---|---:|---:|---:|---:|---:|---|
| L00_08 | `0,1,2,3,4,5,6,7,8` | 82 | 3/82 | 39/82 | 26.2 | -2.861 | newline:39, word:33, space:6, correct_prefix:3, explanation:1 |
| L08_12 | `8,9,10,11,12` | 82 | 1/82 | 17/82 | 98.5 | -5.205 | space:60, newline:17, word:4, correct_prefix:1 |
| L10_14 | `10,11,12,13,14` | 82 | 2/82 | 61/82 | 104.6 | -5.573 | newline:61, word:13, explanation:4, correct_prefix:2, space:2 |
| L10_20 | `10,11,12,13,14,15,16,17,18,19,20` | 82 | 1/82 | 61/82 | 367.0 | -7.345 | newline:61, word:15, explanation:4, correct_prefix:1, space:1 |
| L12_14 | `12,13,14` | 82 | 2/82 | 61/82 | 104.6 | -5.573 | newline:61, word:13, explanation:4, correct_prefix:2, space:2 |
| L14_17 | `14,15,16,17` | 82 | 0/82 | 63/82 | 118.1 | -5.837 | newline:63, word:16, explanation:2, space:1 |
| L14_20 | `14,15,16,17,18,19,20` | 82 | 1/82 | 61/82 | 367.0 | -7.345 | newline:61, word:15, explanation:4, correct_prefix:1, space:1 |
| L14_27 | `14,15,16,17,18,19,20,21,22,23,24,25,26,27` | 82 | 3/82 | 35/82 | 708.1 | -5.733 | newline:35, explanation:33, word:11, correct_prefix:3 |
| L17_20 | `17,18,19,20` | 82 | 1/82 | 61/82 | 367.0 | -7.345 | newline:61, word:15, explanation:4, correct_prefix:1, space:1 |
| L20_23 | `20,21,22,23` | 82 | 3/82 | 71/82 | 764.5 | -8.648 | newline:71, explanation:7, correct_prefix:3, word:1 |
| L23_27 | `23,24,25,26,27` | 82 | 3/82 | 35/82 | 708.1 | -5.733 | newline:35, explanation:33, word:11, correct_prefix:3 |
