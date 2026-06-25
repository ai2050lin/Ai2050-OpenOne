# Phase 618 Cross Model Summary

Source group and pattern/content decomposition for top attention head paths.

## qwen3

rows=9, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 119}, layers=[27, 28, 29], heads={'27': 32, '28': 32, '29': 32}, top_heads=[11, 23, 6, 14, 5, 2], specs=48, compact=True, time_min=3.20

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `L29_H11_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 4/9 | +1.237 | +0.783 | -0.455 |
| 2 | `L29_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 4/9 | +1.210 | +0.778 | -0.432 |
| 3 | `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | False | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| 4 | `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | False | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| 5 | `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | False | 3 | 18 | 3/9 | +0.862 | +0.567 | -0.295 |
| 6 | `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | False | 3 | 18 | 3/9 | +0.848 | +0.549 | -0.299 |
| 7 | `L29_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 2/9 | +0.306 | +0.233 | -0.073 |
| 8 | `L29_H6_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 2/9 | +0.292 | +0.221 | -0.071 |
| 9 | `L27_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 2/9 | +0.028 | +0.010 | -0.018 |
| 10 | `L29_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 2/9 | +0.014 | +0.007 | -0.007 |
| 11 | `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | True | 3 | 18 | 1/9 | +0.049 | +0.041 | -0.008 |
| 12 | `L29_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/9 | +0.036 | +0.020 | -0.016 |
| 13 | `L27_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 1/9 | +0.028 | +0.015 | -0.013 |
| 14 | `L28_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 1/9 | +0.028 | +0.009 | -0.019 |
| 15 | `L29_H11_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 1/9 | +0.025 | +0.012 | -0.013 |
| 16 | `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | False | 3 | 18 | 1/9 | +0.014 | -0.002 | -0.016 |
| 17 | `L27_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| 18 | `L28_H11_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/9 | +0.011 | +0.008 | -0.003 |
| 19 | `L28_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/9 | +0.007 | +0.005 | -0.003 |
| 20 | `L27_H11_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/9 | +0.000 | -0.012 | -0.012 |
| 21 | `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | False | 3 | 18 | 1/9 | -0.000 | -0.009 | -0.009 |
| 22 | `L27_H23_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/9 | -0.000 | -0.007 | -0.007 |
| 23 | `L27_H6_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/9 | -0.014 | -0.016 | -0.002 |
| 24 | `L28_H11_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/9 | -0.014 | -0.023 | -0.009 |
| 25 | `L27_H23_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/9 | -0.014 | -0.020 | -0.006 |
| 26 | `top6_midlate_all_source_br_content` | all_source | br_content | False | 3 | 18 | 1/9 | -0.014 | -0.024 | -0.010 |
| 27 | `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | True | 3 | 18 | 1/9 | -0.017 | -0.025 | -0.008 |
| 28 | `L29_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/9 | -0.027 | -0.023 | +0.004 |
| 29 | `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | True | 3 | 18 | 0/9 | +0.042 | +0.005 | -0.038 |
| 30 | `L29_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 0/9 | +0.040 | +0.036 | -0.004 |
| 31 | `L28_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 0/9 | +0.036 | +0.019 | -0.018 |
| 32 | `top6_midlate_question_line_br_content` | question_line | br_content | True | 3 | 18 | 0/9 | +0.022 | +0.004 | -0.018 |
| 33 | `L29_H23_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 0/9 | +0.019 | +0.012 | -0.007 |
| 34 | `L29_H23_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 0/9 | +0.018 | +0.020 | +0.002 |
| 35 | `L27_H11_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 0/9 | +0.018 | -0.000 | -0.018 |
| 36 | `L28_H23_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 0/9 | +0.015 | +0.013 | -0.002 |

### top_path_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 3/9 | +0.862 | +0.567 | -0.295 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 3/9 | +0.848 | +0.549 | -0.299 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 1/9 | +0.014 | -0.002 | -0.016 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 1/9 | -0.000 | -0.009 | -0.009 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 1/9 | -0.014 | -0.024 | -0.010 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 0/9 | -0.028 | -0.030 | -0.002 |

### single_head_rr_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 4/9 | +1.237 | +0.783 | -0.455 |
| `L29_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 4/9 | +1.210 | +0.778 | -0.432 |
| `L29_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.306 | +0.233 | -0.073 |
| `L29_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 2/9 | +0.292 | +0.221 | -0.071 |
| `L27_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.028 | +0.010 | -0.018 |
| `L29_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.014 | +0.007 | -0.007 |
| `L27_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.028 | +0.015 | -0.013 |
| `L28_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.028 | +0.009 | -0.019 |
| `L27_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L27_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/9 | +0.000 | -0.012 | -0.012 |
| `L27_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.007 | -0.007 |
| `L27_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.016 | -0.002 |
| `L28_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.023 | -0.009 |
| `L27_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.020 | -0.006 |
| `L27_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.008 | -0.006 |
| `L29_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.012 | -0.002 |
| `L28_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.004 | -0.010 |
| `L28_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.009 | -0.005 |
| `L29_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.009 | -0.005 |
| `L27_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.008 | -0.008 |
| `L29_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.007 | -0.007 |
| `L27_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.006 | -0.006 |
| `L29_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.014 | -0.001 |
| `L28_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.016 | -0.002 |
| `L28_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | -0.028 | -0.026 | +0.002 |
| `L28_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.584 | -0.507 | +0.077 |
| `L28_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.584 | -0.506 | +0.078 |

### pattern_vs_content_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 4/9 | +1.237 | +0.783 | -0.455 |
| `L29_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 4/9 | +1.210 | +0.778 | -0.432 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 3/9 | +0.890 | +0.578 | -0.312 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 3/9 | +0.862 | +0.567 | -0.295 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 3/9 | +0.848 | +0.549 | -0.299 |
| `L29_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.306 | +0.233 | -0.073 |
| `L29_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 2/9 | +0.292 | +0.221 | -0.071 |
| `L27_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.028 | +0.010 | -0.018 |
| `L29_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 2/9 | +0.014 | +0.007 | -0.007 |
| `L27_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.028 | +0.015 | -0.013 |
| `L28_H11_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.028 | +0.009 | -0.019 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 1/9 | +0.014 | -0.002 | -0.016 |
| `L27_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L27_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/9 | +0.000 | -0.012 | -0.012 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 1/9 | -0.000 | -0.009 | -0.009 |
| `L27_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.007 | -0.007 |
| `L27_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.016 | -0.002 |
| `L28_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.023 | -0.009 |
| `L27_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/9 | -0.014 | -0.020 | -0.006 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 1/9 | -0.014 | -0.024 | -0.010 |
| `L27_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.008 | -0.006 |
| `L29_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.012 | -0.002 |
| `L28_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.004 | -0.010 |
| `L28_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.009 | -0.005 |
| `L29_H6_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.009 | -0.005 |
| `L27_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H11_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H23_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L29_H6_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L28_H6_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.008 | -0.008 |
| `L29_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.007 | -0.007 |
| `L27_H11_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.006 | -0.006 |
| `L29_H11_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.014 | -0.001 |
| `L28_H6_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.016 | -0.002 |
| `L28_H23_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/9 | -0.028 | -0.026 | +0.002 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 0/9 | -0.028 | -0.030 | -0.002 |
| `L28_H23_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/9 | -0.584 | -0.507 | +0.077 |
| `L28_H23_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.584 | -0.506 | +0.078 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 1/9 | +0.049 | +0.041 | -0.008 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 1/9 | -0.017 | -0.025 | -0.008 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 0/9 | +0.042 | +0.005 | -0.038 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 0/9 | +0.022 | +0.004 | -0.018 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 0/9 | +0.011 | +0.002 | -0.008 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 0/9 | +0.006 | -0.002 | -0.007 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/9 | -0.002 | -0.001 | +0.001 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 0/9 | -0.009 | +0.012 | +0.021 |

## glm4

rows=12, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 116}, layers=[32, 33, 34], heads={'32': 32, '33': 32, '34': 32}, top_heads=[12, 8, 4, 28, 6, 7], specs=48, compact=True, time_min=5.25

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | True | 3 | 18 | 2/12 | +0.004 | +0.001 | -0.003 |
| 2 | `L34_H8_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| 3 | `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | True | 3 | 18 | 1/12 | +0.011 | +0.032 | +0.020 |
| 4 | `L34_H4_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/12 | +0.010 | +0.007 | -0.004 |
| 5 | `L32_H12_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/12 | +0.005 | +0.002 | -0.003 |
| 6 | `L32_H4_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/12 | +0.005 | +0.007 | +0.002 |
| 7 | `top6_midlate_question_line_br_content` | question_line | br_content | True | 3 | 18 | 1/12 | +0.003 | +0.007 | +0.004 |
| 8 | `L32_H8_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/12 | +0.002 | +0.007 | +0.005 |
| 9 | `L33_H8_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 1/12 | -0.003 | +0.002 | +0.005 |
| 10 | `L34_H12_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/12 | -0.006 | -0.004 | +0.002 |
| 11 | `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | True | 3 | 18 | 1/12 | -0.009 | -0.001 | +0.008 |
| 12 | `L33_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/12 | -0.010 | -0.006 | +0.004 |
| 13 | `L33_H12_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 1/12 | -0.011 | -0.002 | +0.009 |
| 14 | `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | True | 3 | 18 | 1/12 | -0.015 | -0.009 | +0.005 |
| 15 | `L33_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 1/12 | -0.016 | -0.008 | +0.008 |
| 16 | `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | False | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| 17 | `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | False | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| 18 | `L33_H12_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/12 | -0.017 | -0.014 | +0.003 |
| 19 | `L32_H4_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/12 | -0.057 | -0.046 | +0.011 |
| 20 | `L34_H12_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| 21 | `L34_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 0/12 | +0.021 | +0.010 | -0.011 |
| 22 | `L34_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 0/12 | +0.021 | +0.008 | -0.013 |
| 23 | `L33_H4_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.006 | -0.004 |
| 24 | `L34_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| 25 | `L34_H12_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| 26 | `L33_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 0/12 | +0.006 | +0.008 | +0.001 |
| 27 | `L33_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| 28 | `L34_H4_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.001 | -0.004 |
| 29 | `L32_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 0/12 | +0.004 | +0.012 | +0.009 |
| 30 | `L32_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | True | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 31 | `L32_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 32 | `L32_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | True | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 33 | `L32_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 34 | `L32_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | True | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 35 | `L32_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| 36 | `L32_H8_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.000 | +0.004 | +0.004 |

### top_path_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/12 | -0.010 | -0.006 | +0.004 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 0/12 | -0.016 | -0.007 | +0.008 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 0/12 | -0.021 | -0.011 | +0.010 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 0/12 | -0.026 | -0.011 | +0.015 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 0/12 | -0.031 | -0.034 | -0.003 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.048 | -0.011 |

### single_head_rr_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L34_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| `L34_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | +0.010 | +0.007 | -0.004 |
| `L32_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.002 | -0.003 |
| `L32_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.007 | +0.002 |
| `L33_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/12 | -0.016 | -0.008 | +0.008 |
| `L32_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | -0.057 | -0.046 | +0.011 |
| `L34_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| `L34_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.021 | +0.010 | -0.011 |
| `L34_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.021 | +0.008 | -0.013 |
| `L33_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.006 | -0.004 |
| `L34_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| `L34_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| `L33_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `L34_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.001 | -0.004 |
| `L32_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.004 | +0.004 |
| `L33_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.000 | +0.001 | +0.001 |
| `L33_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | +0.001 | +0.007 |
| `L34_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L33_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | +0.000 | +0.011 |
| `L32_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.009 | +0.006 |
| `L33_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.011 | +0.010 |
| `L33_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.009 | +0.012 |
| `L33_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.015 | +0.011 |
| `L32_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.015 | +0.011 |
| `L32_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.031 | -0.020 | +0.011 |
| `L32_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.019 | +0.017 |
| `L32_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.068 | -0.050 | +0.018 |

### pattern_vs_content_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L34_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| `L34_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | +0.010 | +0.007 | -0.004 |
| `L32_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.002 | -0.003 |
| `L32_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.007 | +0.002 |
| `L33_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/12 | -0.016 | -0.008 | +0.008 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 1/12 | -0.016 | -0.033 | -0.017 |
| `L32_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/12 | -0.057 | -0.046 | +0.011 |
| `L34_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| `L34_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.021 | +0.010 | -0.011 |
| `L34_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.021 | +0.008 | -0.013 |
| `L33_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.006 | -0.004 |
| `L34_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| `L34_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.007 | -0.003 |
| `L33_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `L34_H4_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.001 | -0.004 |
| `L32_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.004 | +0.004 |
| `L33_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H12_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H4_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H8_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L33_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.000 | +0.001 | +0.001 |
| `L33_H12_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | +0.001 | +0.007 |
| `L34_H8_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L33_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | +0.000 | +0.011 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/12 | -0.010 | -0.006 | +0.004 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 0/12 | -0.016 | -0.007 | +0.008 |
| `L32_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.009 | +0.006 |
| `L33_H4_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.011 | +0.010 |
| `L33_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.009 | +0.012 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 0/12 | -0.021 | -0.011 | +0.010 |
| `L33_H8_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.015 | +0.011 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 0/12 | -0.026 | -0.011 | +0.015 |
| `L32_H12_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.015 | +0.011 |
| `L32_H12_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.031 | -0.020 | +0.011 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 0/12 | -0.031 | -0.034 | -0.003 |
| `L32_H8_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.019 | +0.017 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.048 | -0.011 |
| `L32_H4_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.068 | -0.050 | +0.018 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 2/12 | +0.004 | +0.001 | -0.003 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 1/12 | +0.011 | +0.032 | +0.020 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 1/12 | +0.003 | +0.007 | +0.004 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 1/12 | -0.009 | -0.001 | +0.008 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 1/12 | -0.015 | -0.009 | +0.005 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 0/12 | -0.000 | +0.004 | +0.004 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/12 | -0.012 | -0.007 | +0.005 |
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 0/12 | -0.024 | -0.013 | +0.011 |

## deepseek7b

rows=43, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 85}, layers=[20, 21, 22], heads={'20': 28, '21': 28, '22': 28}, top_heads=[3, 1, 7, 24, 25, 13], specs=48, compact=True, time_min=15.20

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | False | 3 | 18 | 41/43 | +2.624 | +1.477 | -1.147 |
| 2 | `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | False | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| 3 | `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | False | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| 4 | `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | False | 3 | 18 | 32/43 | +1.744 | +1.096 | -0.648 |
| 5 | `top6_midlate_all_source_br_content` | all_source | br_content | False | 3 | 18 | 14/43 | +0.873 | +0.681 | -0.192 |
| 6 | `L22_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 13/43 | +0.739 | +0.513 | -0.226 |
| 7 | `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | False | 3 | 18 | 13/43 | +0.731 | +0.562 | -0.169 |
| 8 | `L22_H1_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 13/43 | +0.730 | +0.510 | -0.220 |
| 9 | `top6_midlate_question_line_br_content` | question_line | br_content | False | 3 | 18 | 10/43 | +0.618 | +0.488 | -0.130 |
| 10 | `L22_H3_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 7/43 | +0.397 | +0.271 | -0.126 |
| 11 | `L22_H7_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 7/43 | +0.377 | +0.294 | -0.083 |
| 12 | `L22_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 6/43 | +0.398 | +0.275 | -0.123 |
| 13 | `L22_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 6/43 | +0.363 | +0.293 | -0.071 |
| 14 | `L20_H1_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 5/43 | +0.217 | +0.208 | -0.010 |
| 15 | `L20_H7_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 3/43 | -0.011 | -0.014 | -0.003 |
| 16 | `L20_H3_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 2/43 | +0.069 | +0.079 | +0.011 |
| 17 | `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | True | 3 | 18 | 2/43 | +0.014 | -0.028 | -0.042 |
| 18 | `L22_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 2/43 | +0.001 | -0.003 | -0.004 |
| 19 | `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | True | 3 | 18 | 2/43 | -0.010 | -0.022 | -0.012 |
| 20 | `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | True | 3 | 18 | 2/43 | -0.035 | -0.062 | -0.027 |
| 21 | `top6_midlate_question_line_br_content` | question_line | br_content | True | 3 | 18 | 2/43 | -0.047 | -0.063 | -0.016 |
| 22 | `top6_midlate_all_source_br_content` | all_source | br_content | True | 3 | 18 | 2/43 | -0.114 | -0.086 | +0.028 |
| 23 | `L20_H3_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/43 | +0.034 | +0.042 | +0.007 |
| 24 | `L20_H1_question_line_rr` | question_line | rr_pattern_content | True | 1 | 1 | 1/43 | +0.007 | -0.000 | -0.008 |
| 25 | `L20_H1_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/43 | +0.006 | +0.012 | +0.006 |
| 26 | `L21_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | False | 1 | 1 | 1/43 | -0.000 | -0.005 | -0.005 |
| 27 | `L22_H7_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/43 | -0.002 | -0.004 | -0.003 |
| 28 | `L20_H7_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/43 | -0.002 | +0.004 | +0.006 |
| 29 | `L21_H3_all_source_rr` | all_source | rr_pattern_content | False | 1 | 1 | 1/43 | -0.002 | -0.002 | -0.000 |
| 30 | `L22_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.002 | -0.010 | -0.008 |
| 31 | `L22_H1_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/43 | -0.004 | -0.017 | -0.014 |
| 32 | `L21_H7_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/43 | -0.006 | -0.005 | +0.001 |
| 33 | `L21_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.006 | -0.004 | +0.002 |
| 34 | `L21_H3_all_source_rr` | all_source | rr_pattern_content | True | 1 | 1 | 1/43 | -0.008 | -0.009 | -0.001 |
| 35 | `L21_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.009 | -0.006 | +0.003 |
| 36 | `L22_H7_question_line_rr` | question_line | rr_pattern_content | False | 1 | 1 | 1/43 | -0.009 | -0.002 | +0.007 |

### top_path_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 41/43 | +2.624 | +1.477 | -1.147 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 32/43 | +1.744 | +1.096 | -0.648 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 14/43 | +0.873 | +0.681 | -0.192 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 13/43 | +0.731 | +0.562 | -0.169 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 10/43 | +0.618 | +0.488 | -0.130 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/43 | -0.015 | -0.020 | -0.004 |

### single_head_rr_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L22_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 13/43 | +0.739 | +0.513 | -0.226 |
| `L22_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 13/43 | +0.730 | +0.510 | -0.220 |
| `L22_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 7/43 | +0.397 | +0.271 | -0.126 |
| `L22_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 7/43 | +0.377 | +0.294 | -0.083 |
| `L22_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 6/43 | +0.398 | +0.275 | -0.123 |
| `L22_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 6/43 | +0.363 | +0.293 | -0.071 |
| `L20_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 5/43 | +0.217 | +0.208 | -0.010 |
| `L20_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 2/43 | +0.069 | +0.079 | +0.011 |
| `L20_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | +0.034 | +0.042 | +0.007 |
| `L20_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | +0.006 | +0.012 | +0.006 |
| `L21_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.000 | -0.005 | -0.005 |
| `L20_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/43 | -0.002 | +0.004 | +0.006 |
| `L21_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/43 | -0.002 | -0.002 | -0.000 |
| `L22_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.009 | -0.002 | +0.007 |
| `L22_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.010 | -0.013 | -0.003 |
| `L20_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.010 | +0.002 |
| `L21_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.008 | +0.004 |
| `L20_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.011 | +0.005 |
| `L20_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.022 | -0.017 | +0.005 |
| `L21_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/43 | +0.006 | -0.000 | -0.006 |
| `L20_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.003 | -0.004 | -0.001 |
| `L21_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.006 | -0.002 |
| `L21_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.014 | -0.007 |
| `L21_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.002 | +0.005 |
| `L21_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.007 | +0.002 |
| `L20_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.016 | -0.006 |
| `L22_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.013 | -0.013 | +0.000 |

### pattern_vs_content_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 41/43 | +2.624 | +1.477 | -1.147 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 33/43 | +1.728 | +1.099 | -0.629 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 32/43 | +1.744 | +1.096 | -0.648 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 14/43 | +0.873 | +0.681 | -0.192 |
| `L22_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 13/43 | +0.739 | +0.513 | -0.226 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 13/43 | +0.731 | +0.562 | -0.169 |
| `L22_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 13/43 | +0.730 | +0.510 | -0.220 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 10/43 | +0.618 | +0.488 | -0.130 |
| `L22_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 7/43 | +0.397 | +0.271 | -0.126 |
| `L22_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 7/43 | +0.377 | +0.294 | -0.083 |
| `L22_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 6/43 | +0.398 | +0.275 | -0.123 |
| `L22_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 6/43 | +0.363 | +0.293 | -0.071 |
| `L20_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 5/43 | +0.217 | +0.208 | -0.010 |
| `L20_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 2/43 | +0.069 | +0.079 | +0.011 |
| `L20_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | +0.034 | +0.042 | +0.007 |
| `L20_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | +0.006 | +0.012 | +0.006 |
| `L21_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.000 | -0.005 | -0.005 |
| `L20_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/43 | -0.002 | +0.004 | +0.006 |
| `L21_H3_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 1/43 | -0.002 | -0.002 | -0.000 |
| `L22_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.009 | -0.002 | +0.007 |
| `L22_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.010 | -0.013 | -0.003 |
| `L20_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.010 | +0.002 |
| `L21_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.008 | +0.004 |
| `L20_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.011 | +0.005 |
| `L20_H1_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.022 | -0.017 | +0.005 |
| `L21_H1_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/43 | +0.006 | -0.000 | -0.006 |
| `L20_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H1_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H3_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L22_H7_final_object_category_line_rr` | final_object_category_line | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L21_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.003 | -0.004 | -0.001 |
| `L21_H3_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.006 | -0.002 |
| `L21_H3_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.014 | -0.007 |
| `L21_H7_value_rule_lines_rr` | value_rule_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.002 | +0.005 |
| `L21_H7_all_source_rr` | all_source | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.007 | +0.002 |
| `L20_H7_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.016 | -0.006 |
| `L22_H1_question_line_rr` | question_line | rr_pattern_content | 1 | 1 | 0/43 | -0.013 | -0.013 | +0.000 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 0/43 | -0.015 | -0.020 | -0.004 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_source_rr_pattern_content` | all_source | rr_pattern_content | 3 | 18 | 2/43 | +0.014 | -0.028 | -0.042 |
| `top6_midlate_question_line_rb_pattern` | question_line | rb_pattern | 3 | 18 | 2/43 | -0.010 | -0.022 | -0.012 |
| `top6_midlate_question_line_rr_pattern_content` | question_line | rr_pattern_content | 3 | 18 | 2/43 | -0.035 | -0.062 | -0.027 |
| `top6_midlate_question_line_br_content` | question_line | br_content | 3 | 18 | 2/43 | -0.047 | -0.063 | -0.016 |
| `top6_midlate_all_source_br_content` | all_source | br_content | 3 | 18 | 2/43 | -0.114 | -0.086 | +0.028 |
| `top6_midlate_final_object_category_line_br_content` | final_object_category_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rb_pattern` | final_object_category_line | rb_pattern | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_final_object_category_line_rr_pattern_content` | final_object_category_line | rr_pattern_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_br_content` | value_rule_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_value_rule_lines_rr_pattern_content` | value_rule_lines | rr_pattern_content | 3 | 18 | 0/43 | -0.046 | -0.044 | +0.002 |
| `top6_midlate_all_source_rb_pattern` | all_source | rb_pattern | 3 | 18 | 0/43 | -0.066 | -0.042 | +0.024 |
| `top6_midlate_value_rule_lines_rb_pattern` | value_rule_lines | rb_pattern | 3 | 18 | 0/43 | -0.094 | -0.094 | +0.000 |
