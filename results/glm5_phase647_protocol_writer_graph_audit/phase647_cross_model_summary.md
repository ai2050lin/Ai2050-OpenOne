# Phase 647 Cross-Model Summary

目标：把 Phase646 atlas 中的 value_short_answer_protocol 从 layer_out trajectory 继续拆成 attention / MLP / residual carry writer graph。

## qwen3

- raw_cases: 320 / target_seen: 26 / cases_written: 26 / mode_rows: 2444
- layers: `[17, 18, 19, 20]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']` / target_only: True
- filtered: `{'not_target': 294, 'separator_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 5.59

### Baselines

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| original | 26 |  | baseline |  |  |  |  | 19/26 | 23/26 | 0/26 | 1.2 | 1.216 | correct_prefix:23, space:3 |  v05:9,  v22:7,  v91:4,  v48:3,  22:2,  91:1 |
| inline | 26 |  | baseline |  |  |  |  | 0/26 | 1/26 | 15/26 | 4.8 | -1.433 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |

### Interval Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_interval_L17_20_layer_out_restore | 26 | to_original | interval | L17_20 |  | layer_out | restore | 1/26 | 1/26 | 24/26 | 3.7 | -1.380 | newline:24, space:1, correct_prefix:1 |  ?\n\nOkay,:22,  91:1,  22:1,  v22:1,  v91:1 |
| remove_from_inline_interval_L17_20_layer_out_restore | 26 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |  91:7,  05:5,  22:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| to_original_interval_L18_19_layer_out_restore | 26 | to_original | interval | L18_19 |  | layer_out | restore | 2/26 | 2/26 | 22/26 | 3.6 | -1.269 | newline:22, space:2, correct_prefix:2 |  ?\n\nOkay,:21,  91:1,  v48:1,  22:1,  v22:1,  v91:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 26 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| remove_from_inline_interval_L17_20_attn_out_restore | 26 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |  v05:5,  ?\n\nTo solve:4,  22:4,  91:4,  05:3,  ?\n\nOkay,:2,  v22:2,  48:1 |
| remove_from_inline_interval_L17_20_mlp_out_restore | 26 | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 26/26 | 26/26 | 0/26 | 1.0 | 3.341 | correct_prefix:26 |  v05:11,  v91:7,  v22:6,  v48:2 |
| remove_from_inline_interval_L18_19_attn_out_restore | 26 | remove_from_inline | interval | L18_19 |  | attn_out | restore | 10/26 | 12/26 | 1/26 | 2.2 | 0.510 | space:13, correct_prefix:12, newline:1 |  22:6,  v05:5,  05:4,  91:4,  v91:3,  v48:2,  v22:1,  \n\nOkay,:1 |
| remove_from_inline_interval_L18_19_mlp_out_restore | 26 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 0/26 | 0/26 | 26/26 | 3.7 | -1.413 | newline:26 |  ?\n\nOkay,:20,  ?\n\nTo solve:6 |
| to_original_interval_L17_20_attn_out_restore | 26 | to_original | interval | L17_20 |  | attn_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |  v05:11,  v22:6,  v91:4,  91:2,  22:1,  v?\nLet:1,  v48:1 |
| to_original_interval_L17_20_mlp_out_restore | 26 | to_original | interval | L17_20 |  | mlp_out | restore | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |  v05:10,  22:5,  v91:4,  91:2,  v48:2,  v22:2,  48:1 |
| to_original_interval_L18_19_attn_out_restore | 26 | to_original | interval | L18_19 |  | attn_out | restore | 21/26 | 24/26 | 0/26 | 1.1 | 2.697 | correct_prefix:24, space:2 |  v05:10,  v22:6,  v91:6,  v48:2,  91:1,  22:1 |
| to_original_interval_L18_19_mlp_out_restore | 26 | to_original | interval | L18_19 |  | mlp_out | restore | 16/26 | 18/26 | 2/26 | 1.3 | 1.087 | correct_prefix:18, space:6, newline:2 |  v05:6,  v91:6,  22:4,  v48:3,  v22:3,  91:2,  ?\n\nOkay,:2 |

### Best Sufficiency Single-Layer Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_interval_L18_19_attn_out_restore | 26 | to_original | interval | L18_19 |  | attn_out | restore | 21/26 | 24/26 | 0/26 | 1.1 | 2.697 | correct_prefix:24, space:2 |  v05:10,  v22:6,  v91:6,  v48:2,  91:1,  22:1 |
| to_original_L19_mlp_out_restore | 26 | to_original | single_layer |  | 19 | mlp_out | restore | 20/26 | 25/26 | 0/26 | 1.0 | 1.846 | correct_prefix:25, space:1 |  v05:9,  v22:7,  v91:4,  v48:3,  91:2,  22:1 |
| to_original_L20_attn_out_restore | 26 | to_original | single_layer |  | 20 | attn_out | restore | 20/26 | 24/26 | 0/26 | 1.1 | 1.582 | correct_prefix:24, space:2 |  v05:9,  v22:8,  v91:4,  v48:3,  22:1,  91:1 |
| to_original_interval_L17_20_attn_out_restore | 26 | to_original | interval | L17_20 |  | attn_out | restore | 20/26 | 23/26 | 0/26 | 1.2 | 4.764 | correct_prefix:23, space:3 |  v05:11,  v22:6,  v91:4,  91:2,  22:1,  v?\nLet:1,  v48:1 |
| to_original_interval_L17_20_mlp_out_restore | 26 | to_original | interval | L17_20 |  | mlp_out | restore | 17/26 | 18/26 | 0/26 | 1.3 | 3.072 | correct_prefix:18, space:8 |  v05:10,  22:5,  v91:4,  91:2,  v48:2,  v22:2,  48:1 |
| to_original_L19_attn_out_restore | 26 | to_original | single_layer |  | 19 | attn_out | restore | 17/26 | 20/26 | 2/26 | 1.3 | 0.851 | correct_prefix:20, space:4, newline:2 |  v05:9,  v22:5,  22:3,  v48:3,  v91:3,  ?\n\nOkay,:2,  91:1 |
| to_original_interval_L18_19_mlp_out_restore | 26 | to_original | interval | L18_19 |  | mlp_out | restore | 16/26 | 18/26 | 2/26 | 1.3 | 1.087 | correct_prefix:18, space:6, newline:2 |  v05:6,  v91:6,  22:4,  v48:3,  v22:3,  91:2,  ?\n\nOkay,:2 |
| to_original_L18_attn_out_restore | 26 | to_original | single_layer |  | 18 | attn_out | restore | 15/26 | 20/26 | 0/26 | 1.2 | 1.370 | correct_prefix:20, space:6 |  v05:8,  v22:5,  91:4,  22:3,  v48:3,  v91:2,  05:1 |
| to_original_L20_mlp_out_restore | 26 | to_original | single_layer |  | 20 | mlp_out | restore | 11/26 | 13/26 | 4/26 | 1.7 | 0.654 | correct_prefix:13, space:9, newline:4 |  v05:7,  22:5,  v48:3,  v22:3,  91:2,  \n\nOkay,:2,  05:2,  ?\n\nOkay,:1 |
| to_original_L17_mlp_out_restore | 26 | to_original | single_layer |  | 17 | mlp_out | restore | 9/26 | 10/26 | 0/26 | 1.7 | 1.317 | space:16, correct_prefix:10 |  22:7,  v05:5,  91:4,  05:4,  v48:2,  v91:2,  48:1,  v22:1 |
| to_original_L18_mlp_out_restore | 26 | to_original | single_layer |  | 18 | mlp_out | restore | 8/26 | 8/26 | 14/26 | 2.7 | -0.279 | newline:14, correct_prefix:8, space:4 |  ?\n\nOkay,:12,  v05:4,  22:3,  v48:2,  v22:2,  48:1,  91:1,  v91:1 |
| to_original_L17_attn_out_restore | 26 | to_original | single_layer |  | 17 | attn_out | restore | 5/26 | 6/26 | 14/26 | 3.1 | -0.389 | newline:14, correct_prefix:6, space:6 |  \n\nOkay,:12,  22:3,  v05:2,  91:2,  v22:2,  05:2,  v48:1,  ?\n\nOkay,:1 |
| to_original_L18_layer_out_restore | 26 | to_original | single_layer |  | 18 | layer_out | restore | 4/26 | 4/26 | 19/26 | 3.6 | -1.173 | newline:19, correct_prefix:4, space:3 |  ?\n\nOkay,:19,  91:2,  v05:1,  v48:1,  22:1,  v22:1,  v91:1 |
| to_original_L19_layer_input_restore | 26 | to_original | single_layer |  | 19 | layer_input | restore | 4/26 | 4/26 | 19/26 | 3.6 | -1.173 | newline:19, correct_prefix:4, space:3 |  ?\n\nOkay,:19,  91:2,  v05:1,  v48:1,  22:1,  v22:1,  v91:1 |
| to_original_L17_layer_out_restore | 26 | to_original | single_layer |  | 17 | layer_out | restore | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |  ?\n\nOkay,:19,  v22:2,  v05:1,  v48:1,  91:1,  22:1,  v91:1 |
| to_original_L18_layer_input_restore | 26 | to_original | single_layer |  | 18 | layer_input | restore | 4/26 | 4/26 | 20/26 | 3.5 | -1.202 | newline:20, correct_prefix:4, space:2 |  ?\n\nOkay,:19,  v22:2,  v05:1,  v48:1,  91:1,  22:1,  v91:1 |

### Best Necessity Single-Layer Remove

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_interval_L18_19_mlp_out_restore | 26 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 0/26 | 0/26 | 26/26 | 3.7 | -1.413 | newline:26 |  ?\n\nOkay,:20,  ?\n\nTo solve:6 |
| remove_from_inline_L19_mlp_out_restore | 26 | remove_from_inline | single_layer |  | 19 | mlp_out | restore | 0/26 | 0/26 | 26/26 | 5.1 | -1.788 | newline:26 |  ?\n\nOkay,:25,  ?\n\nTo solve:1 |
| remove_from_inline_L17_mlp_out_restore | 26 | remove_from_inline | single_layer |  | 17 | mlp_out | restore | 0/26 | 2/26 | 21/26 | 5.0 | -1.688 | newline:21, space:3, correct_prefix:2 |  ?\n\nOkay,:21,  22:2,  91:1,  48:1,  v22:1 |
| remove_from_inline_L20_attn_out_restore | 26 | remove_from_inline | single_layer |  | 20 | attn_out | restore | 0/26 | 0/26 | 14/26 | 5.3 | -1.606 | newline:14, space:12 |  ?\n\nOkay,:13,  91:5,  22:4,  05:2,  48:1,  ?\n\nTo solve:1 |
| remove_from_inline_L20_mlp_out_restore | 26 | remove_from_inline | single_layer |  | 20 | mlp_out | restore | 1/26 | 2/26 | 18/26 | 4.6 | -1.346 | newline:18, space:6, correct_prefix:2 |  ?\n\nOkay,:17,  22:3,  05:2,  91:2,  v48:1,  v22:1 |
| remove_from_inline_L18_mlp_out_restore | 26 | remove_from_inline | single_layer |  | 18 | mlp_out | restore | 1/26 | 2/26 | 11/26 | 4.2 | -0.861 | space:13, newline:11, correct_prefix:2 |  ?\n\nOkay,:10,  22:6,  91:5,  05:2,  v05:1,  48:1,  v22:1 |
| remove_from_inline_L18_attn_out_restore | 26 | remove_from_inline | single_layer |  | 18 | attn_out | restore | 2/26 | 1/26 | 16/26 | 4.5 | -1.269 | newline:16, space:9, correct_prefix:1 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  v05:1,  v91:1 |
| remove_from_inline_L19_attn_out_restore | 26 | remove_from_inline | single_layer |  | 19 | attn_out | restore | 3/26 | 2/26 | 19/26 | 4.8 | -1.471 | newline:19, space:5, correct_prefix:2 |  ?\n\nOkay,:15,  22:4,  05:2,  91:2,  v05:1,  v48:1,  v91:1 |
| remove_from_inline_L17_attn_out_restore | 26 | remove_from_inline | single_layer |  | 17 | attn_out | restore | 3/26 | 4/26 | 10/26 | 3.5 | -0.731 | space:12, newline:10, correct_prefix:4 |  ?\n\nOkay,:10,  22:5,  91:5,  05:2,  v48:2,  v05:1,  v22:1 |
| remove_from_inline_interval_L17_20_attn_out_restore | 26 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 7/26 | 7/26 | 7/26 | 3.6 | -0.736 | space:12, correct_prefix:7, newline:7 |  v05:5,  ?\n\nTo solve:4,  22:4,  91:4,  05:3,  ?\n\nOkay,:2,  v22:2,  48:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 26 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 7/26 | 9/26 | 0/26 | 2.7 | 0.212 | space:17, correct_prefix:9 |  22:6,  91:6,  05:5,  v05:4,  v48:2,  v22:2,  v91:1 |
| remove_from_inline_L18_layer_out_restore | 26 | remove_from_inline | single_layer |  | 18 | layer_out | restore | 8/26 | 11/26 | 2/26 | 2.6 | 0.019 | space:13, correct_prefix:11, newline:2 |  22:6,  v05:5,  05:4,  91:4,  v48:2,  ?\n\nOkay,:2,  v22:2,  v91:1 |
| remove_from_inline_L19_layer_input_restore | 26 | remove_from_inline | single_layer |  | 19 | layer_input | restore | 8/26 | 11/26 | 2/26 | 2.6 | 0.019 | space:13, correct_prefix:11, newline:2 |  22:6,  v05:5,  05:4,  91:4,  v48:2,  ?\n\nOkay,:2,  v22:2,  v91:1 |
| remove_from_inline_L17_layer_out_restore | 26 | remove_from_inline | single_layer |  | 17 | layer_out | restore | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |  05:5,  22:5,  91:5,  v05:4,  v22:3,  v48:2,  ?\n\nOkay,:1,  v91:1 |
| remove_from_inline_L18_layer_input_restore | 26 | remove_from_inline | single_layer |  | 18 | layer_input | restore | 8/26 | 10/26 | 1/26 | 2.6 | 0.111 | space:15, correct_prefix:10, newline:1 |  05:5,  22:5,  91:5,  v05:4,  v22:3,  v48:2,  ?\n\nOkay,:1,  v91:1 |
| remove_from_inline_interval_L17_20_layer_out_restore | 26 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 8/26 | 10/26 | 0/26 | 2.4 | 0.317 | space:16, correct_prefix:10 |  91:7,  05:5,  22:5,  v05:4,  v48:2,  v22:2,  v91:1 |

### Control Samples

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_L17_attn_out_random | 26 | remove_from_inline | single_layer |  | 17 | attn_out | random | 2/26 | 1/26 | 15/26 | 4.6 | -1.361 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:12,  22:5,  91:5,  05:2,  v05:1,  v48:1 |
| remove_from_inline_L17_attn_out_reverse | 26 | remove_from_inline | single_layer |  | 17 | attn_out | reverse | 0/26 | 0/26 | 19/26 | 5.8 | -1.918 | newline:19, space:7 |  ?\n\nOkay,:19,  22:4,  91:2,  05:1 |
| remove_from_inline_L17_mlp_out_random | 26 | remove_from_inline | single_layer |  | 17 | mlp_out | random | 0/26 | 0/26 | 20/26 | 5.1 | -1.577 | newline:20, space:6 |  ?\n\nOkay,:18,  91:3,  22:3,  05:1,  48:1 |
| remove_from_inline_L17_mlp_out_reverse | 26 | remove_from_inline | single_layer |  | 17 | mlp_out | reverse | 2/26 | 2/26 | 10/26 | 4.3 | -1.312 | space:14, newline:10, correct_prefix:2 |  ?\n\nOkay,:7,  22:5,  91:5,  05:4,  48:2,  v05:1,  ?\n\nTo solve:1,  v91:1 |
| remove_from_inline_L17_layer_out_random | 26 | remove_from_inline | single_layer |  | 17 | layer_out | random | 2/26 | 2/26 | 17/26 | 4.7 | -1.577 | newline:17, space:7, correct_prefix:2 |  ?\n\nOkay,:16,  22:3,  05:3,  91:2,  v05:1,  v48:1 |
| remove_from_inline_L17_layer_out_reverse | 26 | remove_from_inline | single_layer |  | 17 | layer_out | reverse | 0/26 | 0/26 | 17/26 | 5.7 | -1.880 | newline:17, space:9 |  ?\n\nOkay,:17,  91:4,  22:3,  05:2 |
| remove_from_inline_L18_attn_out_random | 26 | remove_from_inline | single_layer |  | 18 | attn_out | random | 0/26 | 1/26 | 15/26 | 4.8 | -1.428 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:14,  22:5,  91:4,  48:1,  05:1,  v22:1 |
| remove_from_inline_L18_attn_out_reverse | 26 | remove_from_inline | single_layer |  | 18 | attn_out | reverse | 2/26 | 2/26 | 13/26 | 4.6 | -1.351 | newline:13, space:11, correct_prefix:2 |  ?\n\nOkay,:11,  22:5,  91:5,  05:2,  v05:1,  v48:1,  v22:1 |
| remove_from_inline_L18_mlp_out_random | 26 | remove_from_inline | single_layer |  | 18 | mlp_out | random | 0/26 | 1/26 | 15/26 | 4.7 | -1.428 | newline:15, space:10, correct_prefix:1 |  ?\n\nOkay,:15,  91:4,  22:3,  05:2,  48:1,  v22:1 |
| remove_from_inline_L18_mlp_out_reverse | 26 | remove_from_inline | single_layer |  | 18 | mlp_out | reverse | 0/26 | 0/26 | 23/26 | 5.3 | -2.087 | newline:23, space:3 |  ?\n\nOkay,:22,  22:2,  91:1,  05:1 |
| remove_from_inline_L18_layer_out_random | 26 | remove_from_inline | single_layer |  | 18 | layer_out | random | 2/26 | 2/26 | 12/26 | 4.8 | -1.716 | newline:12, space:12, correct_prefix:2 |  ?\n\nOkay,:12,  22:5,  91:4,  05:2,  48:1,  v48:1,  v91:1 |
| remove_from_inline_L18_layer_out_reverse | 26 | remove_from_inline | single_layer |  | 18 | layer_out | reverse | 0/26 | 0/26 | 26/26 | 5.7 | -2.144 | newline:26 |  ?\n\nOkay,:25,  91:1 |
| remove_from_inline_L19_attn_out_random | 26 | remove_from_inline | single_layer |  | 19 | attn_out | random | 2/26 | 2/26 | 18/26 | 4.5 | -1.332 | newline:18, space:6, correct_prefix:2 |  ?\n\nOkay,:15,  22:5,  91:3,  v48:1,  05:1,  v91:1 |
| remove_from_inline_L19_attn_out_reverse | 26 | remove_from_inline | single_layer |  | 19 | attn_out | reverse | 0/26 | 1/26 | 12/26 | 4.5 | -1.337 | space:13, newline:12, correct_prefix:1 |  ?\n\nOkay,:12,  22:5,  91:5,  05:2,  48:1,  v22:1 |
| remove_from_inline_L19_mlp_out_random | 26 | remove_from_inline | single_layer |  | 19 | mlp_out | random | 2/26 | 2/26 | 16/26 | 4.8 | -1.486 | newline:16, space:8, correct_prefix:2 |  ?\n\nOkay,:14,  22:4,  05:2,  91:2,  v48:1,  ?\n\nTo solve:1,  v22:1,  v91:1 |
| remove_from_inline_L19_mlp_out_reverse | 26 | remove_from_inline | single_layer |  | 19 | mlp_out | reverse | 0/26 | 2/26 | 16/26 | 5.1 | -1.630 | newline:16, space:8, correct_prefix:2 |  ?\n\nOkay,:14,  91:4,  22:4,  05:2,  48:1,  v22:1 |
| remove_from_inline_L19_layer_out_random | 26 | remove_from_inline | single_layer |  | 19 | layer_out | random | 1/26 | 1/26 | 20/26 | 5.8 | -2.163 | newline:20, space:5, correct_prefix:1 |  ?\n\nOkay,:19,  91:4,  v05:1,  22:1,  05:1 |
| remove_from_inline_L19_layer_out_reverse | 26 | remove_from_inline | single_layer |  | 19 | layer_out | reverse | 0/26 | 0/26 | 25/26 | 5.8 | -2.361 | newline:25, space:1 |  ?\n\nOkay,:24,  ?\nQuestion::1,  91:1 |
| remove_from_inline_L20_attn_out_random | 26 | remove_from_inline | single_layer |  | 20 | attn_out | random | 0/26 | 1/26 | 16/26 | 4.7 | -1.428 | newline:16, space:9, correct_prefix:1 |  ?\n\nOkay,:14,  22:5,  91:4,  05:1,  48:1,  v22:1 |
| remove_from_inline_L20_attn_out_reverse | 26 | remove_from_inline | single_layer |  | 20 | attn_out | reverse | 2/26 | 2/26 | 19/26 | 4.4 | -1.341 | newline:19, space:5, correct_prefix:2 |  ?\n\nOkay,:17,  22:4,  v05:1,  05:1,  91:1,  v48:1,  v22:1 |

### Writer Notes

- Top sufficiency: to_original_interval_L18_19_attn_out_restore exact=21/26 newline=0/26; to_original_L19_mlp_out_restore exact=20/26 newline=0/26; to_original_L20_attn_out_restore exact=20/26 newline=0/26; to_original_interval_L17_20_attn_out_restore exact=20/26 newline=0/26; to_original_interval_L17_20_mlp_out_restore exact=17/26 newline=0/26
- Top necessity/remove: remove_from_inline_interval_L18_19_mlp_out_restore exact=0/26 newline=26/26; remove_from_inline_L19_mlp_out_restore exact=0/26 newline=26/26; remove_from_inline_L17_mlp_out_restore exact=0/26 newline=21/26; remove_from_inline_L20_attn_out_restore exact=0/26 newline=14/26; remove_from_inline_L20_mlp_out_restore exact=1/26 newline=18/26

## glm4

- raw_cases: 320 / target_seen: 36 / cases_written: 36 / mode_rows: 3384
- layers: `[17, 18, 19, 20]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']` / target_only: True
- filtered: `{'not_target': 284, 'separator_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 0}` / total_time_min: 8.73

### Baselines

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| original | 36 |  | baseline |  |  |  |  | 29/36 | 30/36 | 0/36 | 1.4 | 80.661 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:8,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| inline | 36 |  | baseline |  |  |  |  | 27/36 | 28/36 | 0/36 | 1.5 | 72.856 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:7,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2 |

### Interval Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_interval_L17_20_layer_out_restore | 36 | to_original | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| remove_from_inline_interval_L17_20_layer_out_restore | 36 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| to_original_interval_L18_19_layer_out_restore | 36 | to_original | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 36 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_interval_L17_20_attn_out_restore | 36 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |  v91:7,  v05:6,  v48:5,  c12:4,  o43:2,  o82:2,  o17:2,  c77:1 |
| remove_from_inline_interval_L17_20_mlp_out_restore | 36 | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |  c12:8,  c59:7,  c33:6,  v05:5,  v91:4,  c77:3,  v48:2,  Yes.\n\n:1 |
| remove_from_inline_interval_L18_19_attn_out_restore | 36 | remove_from_inline | interval | L18_19 |  | attn_out | restore | 30/36 | 33/36 | 0/36 | 1.3 | 75.506 | correct_prefix:33, explanation:3 |  v05:11,  v91:11,  v48:9,  v22:2,  Yes.\n:2,  Yes.\n\n:1 |
| remove_from_inline_interval_L18_19_mlp_out_restore | 36 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 29/36 | 31/36 | 0/36 | 1.3 | 32.868 | correct_prefix:31, word:3, explanation:2 |  v91:10,  v05:9,  v48:9,  v22:3,  c77:2,  c33:1,  Yes.\n\n:1,  Yes.\n:1 |
| to_original_interval_L17_20_attn_out_restore | 36 | to_original | interval | L17_20 |  | attn_out | restore | 1/36 | 2/36 | 0/36 | 28.7 | 99.000 | word:28, explanation:6, correct_prefix:2 |  c12:10,  c33:8,  c59:7,  The answer:4,  c77:3,  The given:1,  v22:1,  v05:1 |
| to_original_interval_L17_20_mlp_out_restore | 36 | to_original | interval | L17_20 |  | mlp_out | restore | 3/36 | 3/36 | 0/36 | 4.0 | 99.000 | word:32, correct_prefix:3, explanation:1 |  c12:10,  c33:7,  c59:6,  True.\n\n:5,  c77:3,  v05:2,  No.\n\n:1,  v48:1 |
| to_original_interval_L18_19_attn_out_restore | 36 | to_original | interval | L18_19 |  | attn_out | restore | 30/36 | 34/36 | 0/36 | 1.1 | 88.524 | correct_prefix:34, word:1, explanation:1 |  v91:13,  v05:10,  v48:9,  v22:2,  c77:1,  No,:1 |
| to_original_interval_L18_19_mlp_out_restore | 36 | to_original | interval | L18_19 |  | mlp_out | restore | 20/36 | 22/36 | 0/36 | 1.9 | 91.161 | correct_prefix:22, word:12, explanation:2 |  v91:9,  v05:8,  v48:5,  c12:4,  c33:3,  c59:3,  No.\n\n:2,  c77:2 |

### Best Sufficiency Single-Layer Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_interval_L18_19_attn_out_restore | 36 | to_original | interval | L18_19 |  | attn_out | restore | 30/36 | 34/36 | 0/36 | 1.1 | 88.524 | correct_prefix:34, word:1, explanation:1 |  v91:13,  v05:10,  v48:9,  v22:2,  c77:1,  No,:1 |
| to_original_L19_attn_out_restore | 36 | to_original | single_layer |  | 19 | attn_out | restore | 29/36 | 30/36 | 0/36 | 1.3 | 80.680 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:8,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| to_original_L18_attn_out_restore | 36 | to_original | single_layer |  | 18 | attn_out | restore | 29/36 | 30/36 | 0/36 | 1.4 | 85.951 | correct_prefix:30, word:5, explanation:1 |  v91:11,  v05:9,  v48:8,  c77:2,  v22:2,  c12:2,  c59:1,  Yes.\n\n:1 |
| to_original_L20_attn_out_restore | 36 | to_original | single_layer |  | 20 | attn_out | restore | 28/36 | 30/36 | 0/36 | 1.4 | 85.891 | correct_prefix:30, word:4, explanation:2 |  v91:11,  v05:9,  v48:8,  c12:3,  c77:2,  v22:1,  c59:1,  Yes.\n\n:1 |
| to_original_L18_mlp_out_restore | 36 | to_original | single_layer |  | 18 | mlp_out | restore | 28/36 | 29/36 | 0/36 | 1.5 | 80.688 | correct_prefix:29, word:6, explanation:1 |  v91:10,  v05:9,  v48:8,  c77:2,  v22:2,  c59:2,  c12:2,  Yes.\n\n:1 |
| to_original_L19_mlp_out_restore | 36 | to_original | single_layer |  | 19 | mlp_out | restore | 28/36 | 29/36 | 0/36 | 1.5 | 93.773 | correct_prefix:29, word:5, explanation:2 |  v91:10,  v05:9,  v48:8,  v22:2,  c59:2,  c12:2,  No.\n\n:1,  c77:1 |
| to_original_L17_layer_input_restore | 36 | to_original | single_layer |  | 17 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.6 | 88.558 | correct_prefix:30, explanation:4, word:2 |  v91:10,  v05:9,  v48:8,  v22:3,  No.\n\n:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_L19_layer_out_restore | 36 | to_original | single_layer |  | 19 | layer_out | restore | 28/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v05:10,  v91:10,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_L20_layer_input_restore | 36 | to_original | single_layer |  | 20 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v05:10,  v91:10,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_L20_layer_out_restore | 36 | to_original | single_layer |  | 20 | layer_out | restore | 28/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:9,  v48:8,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_L18_layer_out_restore | 36 | to_original | single_layer |  | 18 | layer_out | restore | 28/36 | 30/36 | 0/36 | 1.9 | 85.951 | correct_prefix:30, explanation:4, word:2 |  v05:9,  v91:9,  v48:9,  v22:2,  Yes.\n\n:2,  c12:2,  No.\n\n:1,  c59:1 |
| to_original_L19_layer_input_restore | 36 | to_original | single_layer |  | 19 | layer_input | restore | 28/36 | 30/36 | 0/36 | 1.9 | 85.951 | correct_prefix:30, explanation:4, word:2 |  v05:9,  v91:9,  v48:9,  v22:2,  Yes.\n\n:2,  c12:2,  No.\n\n:1,  c59:1 |
| to_original_L17_attn_out_restore | 36 | to_original | single_layer |  | 17 | attn_out | restore | 27/36 | 29/36 | 0/36 | 1.5 | 75.387 | correct_prefix:29, word:6, explanation:1 |  v91:10,  v05:9,  v48:7,  c77:2,  v22:2,  c59:2,  c12:2,  c33:1 |
| to_original_L20_mlp_out_restore | 36 | to_original | single_layer |  | 20 | mlp_out | restore | 27/36 | 29/36 | 0/36 | 1.5 | 88.543 | correct_prefix:29, word:6, explanation:1 |  v05:10,  v91:10,  v48:7,  c77:2,  v22:2,  c59:2,  c12:2,  Yes.\n\n:1 |
| to_original_interval_L18_19_layer_out_restore | 36 | to_original | interval | L18_19 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 78.064 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |
| to_original_interval_L17_20_layer_out_restore | 36 | to_original | interval | L17_20 |  | layer_out | restore | 27/36 | 30/36 | 0/36 | 1.8 | 91.141 | correct_prefix:30, explanation:3, word:3 |  v91:11,  v05:10,  v48:7,  No.\n\n:2,  v22:2,  c12:2,  Yes.\n\n:1,  c77:1 |

### Best Necessity Single-Layer Remove

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_interval_L17_20_mlp_out_restore | 36 | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 10/36 | 12/36 | 0/36 | 2.3 | 96.398 | word:23, correct_prefix:12, explanation:1 |  c12:8,  c59:7,  c33:6,  v05:5,  v91:4,  c77:3,  v48:2,  Yes.\n\n:1 |
| remove_from_inline_interval_L17_20_attn_out_restore | 36 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 17/36 | 20/36 | 0/36 | 4.6 | 96.381 | correct_prefix:20, word:15, explanation:1 |  v91:7,  v05:6,  v48:5,  c12:4,  o43:2,  o82:2,  o17:2,  c77:1 |
| remove_from_inline_L19_attn_out_restore | 36 | remove_from_inline | single_layer |  | 19 | attn_out | restore | 25/36 | 28/36 | 0/36 | 1.8 | 67.488 | correct_prefix:28, explanation:5, word:3 |  v91:11,  v05:9,  v48:5,  Yes.\n\n:5,  v22:3,  c12:2,  c77:1 |
| remove_from_inline_L17_attn_out_restore | 36 | remove_from_inline | single_layer |  | 17 | attn_out | restore | 28/36 | 30/36 | 0/36 | 1.5 | 78.140 | correct_prefix:30, explanation:4, word:2 |  v91:11,  v05:9,  v48:8,  v22:3,  Yes.\n\n:3,  c77:2 |
| remove_from_inline_interval_L18_19_mlp_out_restore | 36 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 29/36 | 31/36 | 0/36 | 1.3 | 32.868 | correct_prefix:31, word:3, explanation:2 |  v91:10,  v05:9,  v48:9,  v22:3,  c77:2,  c33:1,  Yes.\n\n:1,  Yes.\n:1 |
| remove_from_inline_L20_attn_out_restore | 36 | remove_from_inline | single_layer |  | 20 | attn_out | restore | 29/36 | 32/36 | 0/36 | 1.4 | 62.414 | correct_prefix:32, word:2, explanation:2 |  v91:11,  v05:9,  v48:9,  v22:3,  c77:2,  Yes.\n\n:2 |
| remove_from_inline_L20_mlp_out_restore | 36 | remove_from_inline | single_layer |  | 20 | mlp_out | restore | 29/36 | 31/36 | 0/36 | 1.4 | 56.968 | correct_prefix:31, explanation:3, word:2 |  v91:10,  v05:9,  v48:9,  v22:3,  c77:2,  Yes.\n\n:2,  c12:1 |
| remove_from_inline_L18_mlp_out_restore | 36 | remove_from_inline | single_layer |  | 18 | mlp_out | restore | 29/36 | 30/36 | 0/36 | 1.5 | 62.292 | correct_prefix:30, word:3, explanation:3 |  v91:10,  v05:9,  v48:9,  v22:3,  c77:2,  Yes.\n\n:2,  c12:1 |
| remove_from_inline_interval_L18_19_attn_out_restore | 36 | remove_from_inline | interval | L18_19 |  | attn_out | restore | 30/36 | 33/36 | 0/36 | 1.3 | 75.506 | correct_prefix:33, explanation:3 |  v05:11,  v91:11,  v48:9,  v22:2,  Yes.\n:2,  Yes.\n\n:1 |
| remove_from_inline_interval_L17_20_layer_out_restore | 36 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_L20_layer_out_restore | 36 | remove_from_inline | single_layer |  | 20 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.1 | 67.670 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 36 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_L17_layer_out_restore | 36 | remove_from_inline | single_layer |  | 17 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_L18_layer_input_restore | 36 | remove_from_inline | single_layer |  | 18 | layer_input | restore | 31/36 | 34/36 | 0/36 | 1.2 | 75.525 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_L19_layer_out_restore | 36 | remove_from_inline | single_layer |  | 19 | layer_out | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |
| remove_from_inline_L20_layer_input_restore | 36 | remove_from_inline | single_layer |  | 20 | layer_input | restore | 31/36 | 34/36 | 0/36 | 1.2 | 67.621 | correct_prefix:34, word:1, explanation:1 |  v91:12,  v05:10,  v48:9,  v22:3,  c77:1,  Yes.\n\n:1 |

### Control Samples

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_L17_attn_out_random | 36 | remove_from_inline | single_layer |  | 17 | attn_out | random | 26/36 | 28/36 | 0/36 | 1.6 | 75.393 | correct_prefix:28, word:4, explanation:4 |  v91:10,  v05:9,  v48:6,  Yes.\n\n:4,  v22:3,  c77:2,  c12:2 |
| remove_from_inline_L17_attn_out_reverse | 36 | remove_from_inline | single_layer |  | 17 | attn_out | reverse | 27/36 | 28/36 | 0/36 | 1.6 | 59.576 | correct_prefix:28, word:4, explanation:4 |  v05:10,  v91:10,  v48:6,  v22:3,  c77:2,  Yes.\n:2,  c12:2,  Yes.\n\n:1 |
| remove_from_inline_L17_mlp_out_random | 36 | remove_from_inline | single_layer |  | 17 | mlp_out | random | 27/36 | 28/36 | 0/36 | 1.6 | 72.823 | correct_prefix:28, word:4, explanation:4 |  v05:10,  v91:9,  v48:6,  Yes.\n\n:4,  v22:3,  c77:2,  c12:2 |
| remove_from_inline_L17_mlp_out_reverse | 36 | remove_from_inline | single_layer |  | 17 | mlp_out | reverse | 26/36 | 27/36 | 0/36 | 2.1 | 80.542 | correct_prefix:27, word:5, explanation:4 |  v05:10,  v91:9,  v48:5,  Yes.\n:4,  c77:2,  v22:2,  c12:2,  c59:1 |
| remove_from_inline_L17_layer_out_random | 36 | remove_from_inline | single_layer |  | 17 | layer_out | random | 24/36 | 26/36 | 0/36 | 1.9 | 64.719 | correct_prefix:26, word:6, explanation:4 |  v91:10,  v05:8,  v48:5,  v22:3,  c12:3,  Yes.\n\n:3,  c77:2,  c33:1 |
| remove_from_inline_L17_layer_out_reverse | 36 | remove_from_inline | single_layer |  | 17 | layer_out | reverse | 16/36 | 16/36 | 0/36 | 3.4 | 26.938 | correct_prefix:16, word:15, explanation:5 |  v05:7,  c33:6,  v91:5,  c12:4,  v48:3,  c59:3,  Yes.:3,  c77:2 |
| remove_from_inline_L18_attn_out_random | 36 | remove_from_inline | single_layer |  | 18 | attn_out | random | 29/36 | 30/36 | 0/36 | 1.6 | 64.957 | correct_prefix:30, word:3, explanation:3 |  v05:10,  v91:9,  v48:8,  v22:3,  Yes.\n\n:3,  c77:2,  c12:1 |
| remove_from_inline_L18_attn_out_reverse | 36 | remove_from_inline | single_layer |  | 18 | attn_out | reverse | 26/36 | 29/36 | 0/36 | 1.7 | 80.723 | correct_prefix:29, explanation:4, word:3 |  v91:10,  v05:9,  v48:6,  Yes.\n\n:4,  v22:3,  c77:2,  c12:2 |
| remove_from_inline_L18_mlp_out_random | 36 | remove_from_inline | single_layer |  | 18 | mlp_out | random | 27/36 | 29/36 | 0/36 | 1.6 | 64.896 | correct_prefix:29, explanation:4, word:3 |  v91:10,  v05:9,  v48:7,  v22:3,  Yes.\n\n:3,  c77:2,  Yes.\n:1,  c12:1 |
| remove_from_inline_L18_mlp_out_reverse | 36 | remove_from_inline | single_layer |  | 18 | mlp_out | reverse | 27/36 | 29/36 | 0/36 | 1.7 | 64.893 | correct_prefix:29, explanation:4, word:3 |  v91:10,  v05:9,  v48:7,  v22:3,  Yes.\n\n:3,  c77:2,  Yes.\n:1,  c12:1 |
| remove_from_inline_L18_layer_out_random | 36 | remove_from_inline | single_layer |  | 18 | layer_out | random | 28/36 | 29/36 | 0/36 | 1.7 | 70.048 | correct_prefix:29, explanation:4, word:3 |  v05:9,  v91:9,  v48:9,  c77:2,  c59:2,  v22:2,  Yes.\n\n:2,  Yes.\n:1 |
| remove_from_inline_L18_layer_out_reverse | 36 | remove_from_inline | single_layer |  | 18 | layer_out | reverse | 20/36 | 20/36 | 0/36 | 3.5 | 32.497 | correct_prefix:20, word:12, explanation:4 |  v05:7,  v91:6,  c33:5,  v48:5,  Yes.\n:3,  c12:3,  c77:2,  c59:2 |
| remove_from_inline_L19_attn_out_random | 36 | remove_from_inline | single_layer |  | 19 | attn_out | random | 26/36 | 29/36 | 0/36 | 1.5 | 70.235 | correct_prefix:29, word:4, explanation:3 |  v91:11,  v05:9,  v48:6,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2 |
| remove_from_inline_L19_attn_out_reverse | 36 | remove_from_inline | single_layer |  | 19 | attn_out | reverse | 26/36 | 29/36 | 0/36 | 1.5 | 72.880 | correct_prefix:29, word:4, explanation:3 |  v91:11,  v05:9,  v48:6,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2 |
| remove_from_inline_L19_mlp_out_random | 36 | remove_from_inline | single_layer |  | 19 | mlp_out | random | 28/36 | 30/36 | 0/36 | 1.6 | 67.515 | correct_prefix:30, word:3, explanation:3 |  v91:10,  v05:9,  v48:8,  v22:3,  Yes.\n\n:3,  c77:2,  c12:1 |
| remove_from_inline_L19_mlp_out_reverse | 36 | remove_from_inline | single_layer |  | 19 | mlp_out | reverse | 24/36 | 27/36 | 0/36 | 2.0 | 78.007 | correct_prefix:27, word:5, explanation:4 |  v91:11,  v05:8,  v48:5,  Yes.\n\n:4,  v22:3,  c77:2,  c12:2,  c33:1 |
| remove_from_inline_L19_layer_out_random | 36 | remove_from_inline | single_layer |  | 19 | layer_out | random | 25/36 | 27/36 | 0/36 | 1.8 | 64.776 | correct_prefix:27, word:5, explanation:4 |  v91:10,  v05:8,  v48:6,  v22:3,  c77:2,  c12:2,  c33:1,  c59:1 |
| remove_from_inline_L19_layer_out_reverse | 36 | remove_from_inline | single_layer |  | 19 | layer_out | reverse | 21/36 | 21/36 | 0/36 | 3.3 | 24.661 | correct_prefix:21, word:11, explanation:4 |  v05:7,  v91:7,  v48:5,  c33:4,  Yes.\n:3,  c12:3,  c77:2,  c59:2 |
| remove_from_inline_L20_attn_out_random | 36 | remove_from_inline | single_layer |  | 20 | attn_out | random | 25/36 | 28/36 | 0/36 | 1.6 | 64.935 | correct_prefix:28, word:4, explanation:4 |  v91:11,  v05:9,  v48:5,  v22:3,  Yes.\n\n:3,  c77:2,  c12:2,  Yes.\n:1 |
| remove_from_inline_L20_attn_out_reverse | 36 | remove_from_inline | single_layer |  | 20 | attn_out | reverse | 23/36 | 23/36 | 0/36 | 2.1 | 75.355 | correct_prefix:23, word:9, explanation:4 |  v05:9,  v48:6,  v91:6,  c33:3,  c77:2,  c59:2,  v22:2,  Yes.\n:2 |

### Writer Notes

- Top sufficiency: to_original_interval_L18_19_attn_out_restore exact=30/36 newline=0/36; to_original_L19_attn_out_restore exact=29/36 newline=0/36; to_original_L18_attn_out_restore exact=29/36 newline=0/36; to_original_L20_attn_out_restore exact=28/36 newline=0/36; to_original_L18_mlp_out_restore exact=28/36 newline=0/36
- Top necessity/remove: remove_from_inline_interval_L17_20_mlp_out_restore exact=10/36 newline=0/36; remove_from_inline_interval_L17_20_attn_out_restore exact=17/36 newline=0/36; remove_from_inline_L19_attn_out_restore exact=25/36 newline=0/36; remove_from_inline_L17_attn_out_restore exact=28/36 newline=0/36; remove_from_inline_interval_L18_19_mlp_out_restore exact=29/36 newline=0/36

## deepseek7b

- raw_cases: 320 / target_seen: 48 / cases_written: 48 / mode_rows: 4512
- layers: `[17, 18, 19, 20]` / components: `['layer_input', 'attn_out', 'mlp_out', 'layer_out']` / target_only: True
- filtered: `{'not_target': 88, 'separator_len_mismatch': 0, 'empty_patch': 0, 'case_cap': 1}` / total_time_min: 10.79

### Baselines

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| original | 48 |  | baseline |  |  |  |  | 12/48 | 12/48 | 34/48 | 6.9 | -1.400 | newline:34, correct_prefix:12, word:1, space:1 |  ?\n\nTo solve:26,  ?\n\nI think:8,  v48:7,  v05:4,  c77:1,  48:1,  v22:1 |
| inline | 48 |  | baseline |  |  |  |  | 45/48 | 47/48 | 0/48 | 1.0 | 2.387 | correct_prefix:47, space:1 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |

### Interval Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_interval_L17_20_layer_out_restore | 48 | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |  v48:14,  v05:13,  v22:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_interval_L17_20_layer_out_restore | 48 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |  ?\n\nTo solve:35,  v48:7,  v05:4,  c77:1,  v22:1 |
| to_original_interval_L18_19_layer_out_restore | 48 | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 48 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_interval_L17_20_attn_out_restore | 48 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |  c77:14,  c33:9,  c59:6,  c12:5,  v48:3,  v05:3,  v22:2,  o43:2 |
| remove_from_inline_interval_L17_20_mlp_out_restore | 48 | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |  ?\n\nTo solve:19,  v05:12,  v48:6,  v22:4,  v91:4,  ?\n\nI think:3 |
| remove_from_inline_interval_L18_19_attn_out_restore | 48 | remove_from_inline | interval | L18_19 |  | attn_out | restore | 48/48 | 48/48 | 0/48 | 1.0 | 2.234 | correct_prefix:48 |  v48:16,  v22:13,  v05:12,  v91:7 |
| remove_from_inline_interval_L18_19_mlp_out_restore | 48 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 34/48 | 34/48 | 7/48 | 1.6 | 0.655 | correct_prefix:34, newline:7, word:7 |  v48:12,  v05:10,  v22:6,  v91:6,  ?\n\nTo solve:3,  ?\n\nI think:2,  c77:2,  c12:2 |
| to_original_interval_L17_20_attn_out_restore | 48 | to_original | interval | L17_20 |  | attn_out | restore | 0/48 | 0/48 | 17/48 | 19.0 | -3.152 | word:31, newline:17 |  ?\n\nTo solve:13,  c77:12,  c59:8,  c12:6,  c33:5,  ?\n\nI think:2,  o95:1,  same as c:1 |
| to_original_interval_L17_20_mlp_out_restore | 48 | to_original | interval | L17_20 |  | mlp_out | restore | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |  v05:13,  v22:10,  v48:9,  v91:5,  ?\n\nTo solve:4,  c33:2,  c77:2,  o71:1 |
| to_original_interval_L18_19_attn_out_restore | 48 | to_original | interval | L18_19 |  | attn_out | restore | 13/48 | 12/48 | 7/48 | 2.8 | 0.082 | word:19, correct_prefix:12, space:10, newline:7 |  c77:10,  64:9,  v48:6,  c59:5,  v91:4,  ?\n\nI think:2,  22:2,  v05:2 |
| to_original_interval_L18_19_mlp_out_restore | 48 | to_original | interval | L18_19 |  | mlp_out | restore | 20/48 | 21/48 | 7/48 | 2.2 | 0.677 | correct_prefix:21, word:14, newline:7, space:6 |  v48:8,  v05:7,  c77:5,  v91:4,  c59:4,  ?\n\nTo solve:4,  ?\n\nI think:3,  c12:3 |

### Best Sufficiency Single-Layer Restore

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| to_original_L17_layer_input_restore | 48 | to_original | single_layer |  | 17 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 1.835 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  64:2 |
| to_original_L18_layer_out_restore | 48 | to_original | single_layer |  | 18 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.216 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| to_original_L19_layer_input_restore | 48 | to_original | single_layer |  | 19 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.216 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| to_original_L17_layer_out_restore | 48 | to_original | single_layer |  | 17 | layer_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:1,  64:1 |
| to_original_L18_layer_input_restore | 48 | to_original | single_layer |  | 18 | layer_input | restore | 46/48 | 46/48 | 0/48 | 1.0 | 2.159 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:1,  64:1 |
| to_original_L20_layer_out_restore | 48 | to_original | single_layer |  | 20 | layer_out | restore | 45/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_interval_L18_19_layer_out_restore | 48 | to_original | interval | L18_19 |  | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_L19_layer_out_restore | 48 | to_original | single_layer |  | 19 | layer_out | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_L20_layer_input_restore | 48 | to_original | single_layer |  | 20 | layer_input | restore | 45/48 | 45/48 | 0/48 | 1.1 | 2.452 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| to_original_interval_L17_20_layer_out_restore | 48 | to_original | interval | L17_20 |  | layer_out | restore | 43/48 | 46/48 | 0/48 | 1.0 | 2.574 | correct_prefix:46, space:2 |  v48:14,  v05:13,  v22:11,  v91:7,  22:2,  05:1 |
| to_original_interval_L17_20_mlp_out_restore | 48 | to_original | interval | L17_20 |  | mlp_out | restore | 33/48 | 38/48 | 4/48 | 1.3 | 1.695 | correct_prefix:38, word:6, newline:4 |  v05:13,  v22:10,  v48:9,  v91:5,  ?\n\nTo solve:4,  c33:2,  c77:2,  o71:1 |
| to_original_L18_mlp_out_restore | 48 | to_original | single_layer |  | 18 | mlp_out | restore | 23/48 | 22/48 | 23/48 | 3.1 | -0.121 | newline:23, correct_prefix:22, space:3 |  ?\n\nTo solve:15,  v48:9,  v05:7,  ?\n\nI think:5,  64:5,  v91:4,  v22:3 |
| to_original_L19_attn_out_restore | 48 | to_original | single_layer |  | 19 | attn_out | restore | 21/48 | 22/48 | 21/48 | 2.2 | 0.040 | correct_prefix:22, newline:21, space:5 |  ?\n\nTo solve:14,  v48:8,  v05:8,  ?\n\nI think:7,  v91:4,  64:4,  22:1,  48:1 |
| to_original_interval_L18_19_mlp_out_restore | 48 | to_original | interval | L18_19 |  | mlp_out | restore | 20/48 | 21/48 | 7/48 | 2.2 | 0.677 | correct_prefix:21, word:14, newline:7, space:6 |  v48:8,  v05:7,  c77:5,  v91:4,  c59:4,  ?\n\nTo solve:4,  ?\n\nI think:3,  c12:3 |
| to_original_L17_mlp_out_restore | 48 | to_original | single_layer |  | 17 | mlp_out | restore | 20/48 | 20/48 | 25/48 | 4.4 | -0.595 | newline:25, correct_prefix:20, word:1, space:1, explanation:1 |  ?\n\nTo solve:14,  ?\n\nI think:11,  v48:7,  v05:7,  v91:3,  v22:3,  c77:1,  64:1 |
| to_original_L19_mlp_out_restore | 48 | to_original | single_layer |  | 19 | mlp_out | restore | 18/48 | 19/48 | 18/48 | 2.7 | -0.107 | correct_prefix:19, newline:18, space:11 |  ?\n\nTo solve:16,  64:8,  v48:8,  v05:6,  v91:4,  ?\n\nI think:2,  22:1,  48:1 |

### Best Necessity Single-Layer Remove

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_interval_L17_20_attn_out_restore | 48 | remove_from_inline | interval | L17_20 |  | attn_out | restore | 8/48 | 8/48 | 0/48 | 2.4 | 0.552 | word:40, correct_prefix:8 |  c77:14,  c33:9,  c59:6,  c12:5,  v48:3,  v05:3,  v22:2,  o43:2 |
| remove_from_inline_interval_L17_20_layer_out_restore | 48 | remove_from_inline | interval | L17_20 |  | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |  ?\n\nTo solve:35,  v48:7,  v05:4,  c77:1,  v22:1 |
| remove_from_inline_L20_layer_out_restore | 48 | remove_from_inline | single_layer |  | 20 | layer_out | restore | 12/48 | 12/48 | 35/48 | 4.8 | -1.241 | newline:35, correct_prefix:12, word:1 |  ?\n\nTo solve:33,  v48:7,  v05:4,  ?\n\nI think:2,  c77:1,  v22:1 |
| remove_from_inline_interval_L18_19_layer_out_restore | 48 | remove_from_inline | interval | L18_19 |  | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_L19_layer_out_restore | 48 | remove_from_inline | single_layer |  | 19 | layer_out | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_L20_layer_input_restore | 48 | remove_from_inline | single_layer |  | 20 | layer_input | restore | 14/48 | 15/48 | 30/48 | 3.4 | -0.758 | newline:30, correct_prefix:15, space:2, word:1 |  ?\n\nTo solve:28,  v48:7,  v05:4,  48:3,  v91:2,  c77:1,  05:1,  ?\n\nI think:1 |
| remove_from_inline_L18_layer_out_restore | 48 | remove_from_inline | single_layer |  | 18 | layer_out | restore | 14/48 | 15/48 | 22/48 | 2.8 | -0.395 | newline:22, correct_prefix:15, space:10, word:1 |  ?\n\nTo solve:21,  v48:7,  22:6,  v05:4,  48:4,  v91:2,  c77:1,  05:1 |
| remove_from_inline_L19_layer_input_restore | 48 | remove_from_inline | single_layer |  | 19 | layer_input | restore | 14/48 | 15/48 | 22/48 | 2.8 | -0.395 | newline:22, correct_prefix:15, space:10, word:1 |  ?\n\nTo solve:21,  v48:7,  22:6,  v05:4,  48:4,  v91:2,  c77:1,  05:1 |
| remove_from_inline_L17_layer_out_restore | 48 | remove_from_inline | single_layer |  | 17 | layer_out | restore | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |  ?\n\nTo solve:16,  22:8,  v48:8,  v05:4,  05:3,  48:3,  v91:2,  91:1 |
| remove_from_inline_L18_layer_input_restore | 48 | remove_from_inline | single_layer |  | 18 | layer_input | restore | 15/48 | 16/48 | 18/48 | 2.5 | -0.152 | newline:18, correct_prefix:16, space:14 |  ?\n\nTo solve:16,  22:8,  v48:8,  v05:4,  05:3,  48:3,  v91:2,  91:1 |
| remove_from_inline_L17_layer_input_restore | 48 | remove_from_inline | single_layer |  | 17 | layer_input | restore | 15/48 | 16/48 | 15/48 | 2.4 | 0.103 | space:17, correct_prefix:16, newline:15 |  ?\n\nTo solve:11,  22:9,  v48:7,  48:5,  v05:4,  05:4,  v91:3,  c77:1 |
| remove_from_inline_interval_L17_20_mlp_out_restore | 48 | remove_from_inline | interval | L17_20 |  | mlp_out | restore | 22/48 | 26/48 | 22/48 | 1.8 | 0.120 | correct_prefix:26, newline:22 |  ?\n\nTo solve:19,  v05:12,  v48:6,  v22:4,  v91:4,  ?\n\nI think:3 |
| remove_from_inline_interval_L18_19_mlp_out_restore | 48 | remove_from_inline | interval | L18_19 |  | mlp_out | restore | 34/48 | 34/48 | 7/48 | 1.6 | 0.655 | correct_prefix:34, newline:7, word:7 |  v48:12,  v05:10,  v22:6,  v91:6,  ?\n\nTo solve:3,  ?\n\nI think:2,  c77:2,  c12:2 |
| remove_from_inline_L17_mlp_out_restore | 48 | remove_from_inline | single_layer |  | 17 | mlp_out | restore | 46/48 | 47/48 | 0/48 | 1.0 | 1.741 | correct_prefix:47, space:1 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| remove_from_inline_L19_mlp_out_restore | 48 | remove_from_inline | single_layer |  | 19 | mlp_out | restore | 46/48 | 47/48 | 0/48 | 1.0 | 1.930 | correct_prefix:47, space:1 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| remove_from_inline_L20_mlp_out_restore | 48 | remove_from_inline | single_layer |  | 20 | mlp_out | restore | 46/48 | 46/48 | 0/48 | 1.0 | 1.690 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |

### Control Samples

| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |
|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| remove_from_inline_L17_attn_out_random | 48 | remove_from_inline | single_layer |  | 17 | attn_out | random | 45/48 | 46/48 | 0/48 | 1.0 | 2.389 | correct_prefix:46, space:2 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_L17_attn_out_reverse | 48 | remove_from_inline | single_layer |  | 17 | attn_out | reverse | 45/48 | 45/48 | 0/48 | 1.0 | 2.624 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_L17_mlp_out_random | 48 | remove_from_inline | single_layer |  | 17 | mlp_out | random | 45/48 | 47/48 | 0/48 | 1.0 | 2.264 | correct_prefix:47, space:1 |  v48:16,  v05:12,  v22:10,  v91:7,  22:3 |
| remove_from_inline_L17_mlp_out_reverse | 48 | remove_from_inline | single_layer |  | 17 | mlp_out | reverse | 45/48 | 45/48 | 0/48 | 1.1 | 2.738 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_L17_layer_out_random | 48 | remove_from_inline | single_layer |  | 17 | layer_out | random | 43/48 | 46/48 | 0/48 | 1.0 | 2.259 | correct_prefix:46, space:2 |  v05:14,  v48:13,  v22:11,  v91:7,  22:2,  48:1 |
| remove_from_inline_L17_layer_out_reverse | 48 | remove_from_inline | single_layer |  | 17 | layer_out | reverse | 48/48 | 48/48 | 0/48 | 1.0 | 3.725 | correct_prefix:48 |  v48:16,  v22:13,  v05:12,  v91:7 |
| remove_from_inline_L18_attn_out_random | 48 | remove_from_inline | single_layer |  | 18 | attn_out | random | 47/48 | 47/48 | 0/48 | 1.0 | 2.294 | correct_prefix:47, space:1 |  v48:16,  v22:12,  v05:12,  v91:7,  22:1 |
| remove_from_inline_L18_attn_out_reverse | 48 | remove_from_inline | single_layer |  | 18 | attn_out | reverse | 38/48 | 40/48 | 0/48 | 1.1 | 1.594 | correct_prefix:40, space:8 |  v48:14,  v05:10,  v22:7,  v91:7,  22:6,  05:2,  48:2 |
| remove_from_inline_L18_mlp_out_random | 48 | remove_from_inline | single_layer |  | 18 | mlp_out | random | 43/48 | 44/48 | 0/48 | 1.0 | 2.277 | correct_prefix:44, space:4 |  v48:15,  v05:11,  v22:10,  v91:7,  22:3,  05:1,  48:1 |
| remove_from_inline_L18_mlp_out_reverse | 48 | remove_from_inline | single_layer |  | 18 | mlp_out | reverse | 41/48 | 42/48 | 0/48 | 1.1 | 2.591 | correct_prefix:42, space:6 |  v48:15,  v05:10,  v22:9,  v91:7,  22:4,  05:2,  48:1 |
| remove_from_inline_L18_layer_out_random | 48 | remove_from_inline | single_layer |  | 18 | layer_out | random | 46/48 | 47/48 | 0/48 | 1.0 | 2.314 | correct_prefix:47, space:1 |  v48:16,  v22:12,  v05:11,  v91:7,  22:1,  05:1 |
| remove_from_inline_L18_layer_out_reverse | 48 | remove_from_inline | single_layer |  | 18 | layer_out | reverse | 45/48 | 45/48 | 0/48 | 1.0 | 3.633 | correct_prefix:45, space:3 |  v48:16,  v05:12,  v22:10,  v91:7,  22:3 |
| remove_from_inline_L19_attn_out_random | 48 | remove_from_inline | single_layer |  | 19 | attn_out | random | 46/48 | 46/48 | 0/48 | 1.0 | 2.357 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| remove_from_inline_L19_attn_out_reverse | 48 | remove_from_inline | single_layer |  | 19 | attn_out | reverse | 44/48 | 44/48 | 0/48 | 1.1 | 2.680 | correct_prefix:44, space:4 |  v48:16,  v05:11,  v22:10,  v91:7,  22:3,  05:1 |
| remove_from_inline_L19_mlp_out_random | 48 | remove_from_inline | single_layer |  | 19 | mlp_out | random | 45/48 | 45/48 | 0/48 | 1.0 | 2.318 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_L19_mlp_out_reverse | 48 | remove_from_inline | single_layer |  | 19 | mlp_out | reverse | 43/48 | 44/48 | 0/48 | 1.1 | 2.711 | correct_prefix:44, space:4 |  v48:16,  v22:10,  v05:10,  v91:7,  22:3,  05:2 |
| remove_from_inline_L19_layer_out_random | 48 | remove_from_inline | single_layer |  | 19 | layer_out | random | 46/48 | 47/48 | 0/48 | 1.0 | 2.357 | correct_prefix:47, space:1 |  v48:15,  v22:12,  v05:12,  v91:7,  48:1,  22:1 |
| remove_from_inline_L19_layer_out_reverse | 48 | remove_from_inline | single_layer |  | 19 | layer_out | reverse | 46/48 | 46/48 | 0/48 | 1.0 | 3.880 | correct_prefix:46, space:2 |  v48:16,  v05:12,  v22:11,  v91:7,  22:2 |
| remove_from_inline_L20_attn_out_random | 48 | remove_from_inline | single_layer |  | 20 | attn_out | random | 45/48 | 45/48 | 0/48 | 1.0 | 2.391 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |
| remove_from_inline_L20_attn_out_reverse | 48 | remove_from_inline | single_layer |  | 20 | attn_out | reverse | 45/48 | 45/48 | 0/48 | 1.1 | 2.383 | correct_prefix:45, space:3 |  v48:16,  v22:11,  v05:11,  v91:7,  22:2,  05:1 |

### Writer Notes

- Top sufficiency: to_original_L17_layer_input_restore exact=46/48 newline=0/48; to_original_L18_layer_out_restore exact=46/48 newline=0/48; to_original_L19_layer_input_restore exact=46/48 newline=0/48; to_original_L17_layer_out_restore exact=46/48 newline=0/48; to_original_L18_layer_input_restore exact=46/48 newline=0/48
- Top necessity/remove: remove_from_inline_interval_L17_20_attn_out_restore exact=8/48 newline=0/48; remove_from_inline_interval_L17_20_layer_out_restore exact=12/48 newline=35/48; remove_from_inline_L20_layer_out_restore exact=12/48 newline=35/48; remove_from_inline_interval_L18_19_layer_out_restore exact=14/48 newline=30/48; remove_from_inline_L19_layer_out_restore exact=14/48 newline=30/48
