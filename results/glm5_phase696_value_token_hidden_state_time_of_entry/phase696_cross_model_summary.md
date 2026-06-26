# Phase 696 Value Token Hidden-State Time-of-Entry Audit

- generated: `2026-06-26 15:43:26`

| model | pairs | short_top1 | terse_top1 | short_rank | terse_rank |
|---|---:|---:|---:|---:|---:|
| deepseek7b | 72 | 0.000 | 1.000 | 167.69 | 1.00 |
| glm4 | 5 | 0.000 | 1.000 | 2.00 | 1.00 |
| qwen3 | 3 | 0.000 | 1.000 | 2.00 | 1.00 |

## Peak / First Positive Layers

### deepseek7b

| variant | group | first_pos_layer | first_pos_proj | peak_layer | peak_proj | final_layer | final_proj |
|---|---|---:|---:|---:|---:|---:|---:|
| short_only | target_value | 0 | 0.854 | 26 | 50.490 | 27 | 41.008 |
| short_only | relation | 0 | 0.416 | 26 | 50.863 | 27 | 32.153 |
| short_only | object_name | 0 | 0.662 | 26 | 49.830 | 27 | 48.854 |
| short_only | record_line | 0 | 0.618 | 26 | 51.776 | 27 | 27.891 |
| short_only | record_without_target_value | 0 | 0.574 | 26 | 52.333 | 27 | 25.134 |
| short_only | record_value_object_relation | 0 | 0.678 | 26 | 48.709 | 27 | 43.739 |
| short_only | instruction_line | 0 | 0.879 | 26 | 34.109 | 27 | 1.071 |
| short_only | answer_last | 0 | 0.184 | 26 | 37.825 | 27 | 25.644 |
| terse_no_explain | target_value | 0 | 0.854 | 26 | 50.500 | 27 | 41.337 |
| terse_no_explain | relation | 0 | 0.416 | 26 | 50.841 | 27 | 31.849 |
| terse_no_explain | object_name | 0 | 0.662 | 26 | 49.833 | 27 | 48.953 |
| terse_no_explain | record_line | 0 | 0.618 | 26 | 51.769 | 27 | 27.939 |
| terse_no_explain | record_without_target_value | 0 | 0.574 | 26 | 52.316 | 27 | 25.117 |
| terse_no_explain | record_value_object_relation | 0 | 0.678 | 26 | 48.711 | 27 | 43.808 |
| terse_no_explain | instruction_line | 0 | 0.773 | 26 | 29.753 | 27 | -5.320 |
| terse_no_explain | answer_last | 0 | 0.170 | 26 | 62.187 | 27 | 60.362 |

### glm4

| variant | group | first_pos_layer | first_pos_proj | peak_layer | peak_proj | final_layer | final_proj |
|---|---|---:|---:|---:|---:|---:|---:|
| short_only | target_value | 0 | 0.014 | 30 | 1.087 | 39 | 0.301 |
| short_only | relation | 0 | 0.002 | 39 | 7.023 | 39 | 7.023 |
| short_only | object_name | 0 | 0.002 | 39 | 0.234 | 39 | 0.234 |
| short_only | record_line | 0 | 0.004 | 23 | 0.870 | 39 | -1.213 |
| short_only | record_without_target_value | 0 | 0.004 | 23 | 0.901 | 39 | -1.285 |
| short_only | record_value_object_relation | 0 | 0.003 | 39 | 0.611 | 39 | 0.611 |
| short_only | instruction_line | 0 | 0.005 | 34 | 0.443 | 39 | -2.815 |
| short_only | answer_last | 9 | 0.003 | 34 | 2.640 | 39 | -3.271 |
| terse_no_explain | target_value | 0 | 0.014 | 30 | 1.087 | 39 | 0.301 |
| terse_no_explain | relation | 0 | 0.002 | 39 | 7.107 | 39 | 7.107 |
| terse_no_explain | object_name | 0 | 0.002 | 39 | 0.250 | 39 | 0.250 |
| terse_no_explain | record_line | 0 | 0.004 | 23 | 0.870 | 39 | -1.215 |
| terse_no_explain | record_without_target_value | 0 | 0.004 | 23 | 0.902 | 39 | -1.290 |
| terse_no_explain | record_value_object_relation | 0 | 0.003 | 39 | 0.604 | 39 | 0.604 |
| terse_no_explain | instruction_line | 0 | 0.003 | 23 | 0.292 | 39 | -3.724 |
| terse_no_explain | answer_last | 16 | 0.016 | 34 | 4.870 | 39 | -0.049 |

### qwen3

| variant | group | first_pos_layer | first_pos_proj | peak_layer | peak_proj | final_layer | final_proj |
|---|---|---:|---:|---:|---:|---:|---:|
| short_only | target_value | 2 | 0.021 | 34 | 28.703 | 35 | -2.434 |
| short_only | relation | 4 | 0.011 | 34 | 20.202 | 35 | 0.934 |
| short_only | object_name | 24 | 0.957 | 34 | 15.678 | 35 | 3.337 |
| short_only | record_line | 6 | 10.358 | 34 | 28.406 | 35 | 1.335 |
| short_only | record_without_target_value | 6 | 13.281 | 34 | 28.338 | 35 | 2.432 |
| short_only | record_value_object_relation | 18 | 0.077 | 34 | 20.253 | 35 | -0.932 |
| short_only | instruction_line | 20 | 0.465 | 34 | 7.896 | 35 | -7.295 |
| short_only | answer_last | 8 | 0.045 | 34 | 27.262 | 35 | 7.531 |
| terse_no_explain | target_value | 2 | 0.021 | 34 | 28.703 | 35 | -2.434 |
| terse_no_explain | relation | 4 | 0.011 | 34 | 20.032 | 35 | 0.752 |
| terse_no_explain | object_name | 24 | 0.957 | 34 | 15.726 | 35 | 3.373 |
| terse_no_explain | record_line | 6 | 10.358 | 34 | 28.451 | 35 | 1.393 |
| terse_no_explain | record_without_target_value | 6 | 13.281 | 34 | 28.387 | 35 | 2.505 |
| terse_no_explain | record_value_object_relation | 18 | 0.076 | 34 | 20.273 | 35 | -0.954 |
| terse_no_explain | instruction_line | 19 | 0.201 | 34 | 8.306 | 35 | -7.648 |
| terse_no_explain | answer_last | 9 | 0.088 | 34 | 36.193 | 35 | 14.904 |

