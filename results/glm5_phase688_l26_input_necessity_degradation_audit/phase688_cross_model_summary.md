# Phase 688 L26 Input Necessity and Degradation Audit

- generated: `2026-06-26 13:56:35`

| model | pairs | layers | strongest_drop | drop | patched_rank | rank_inc | pmv_inc | best_other |
|---|---:|---|---|---:|---:|---:|---:|---|
| deepseek7b | 72 | [26, 27] | cross_donor|same_relation_diff_value_short_replace|L27_layer_out | 1.000 | 1587.54 | 1586.54 | 12.520 | {'prose': 72} |
| glm4 | 5 | [38, 39] | cross_donor|same_family_diff_value_short_replace|L38_layer_out | 1.000 | 1829.00 | 1828.00 | 11.000 | {'continuation': 5} |
| qwen3 | 3 | [33, 34] | cross_donor|same_relation_diff_value_short_replace|L33_layer_input | 0.667 | 2.33 | 1.33 | 4.292 | {'continuation': 1, 'prose': 2} |

## Strongest Drop Conditions

### deepseek7b

| condition | drop | patched_top1 | patched_rank | rank_inc | patched_pmv | pmv_inc | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| cross_donor|same_relation_diff_value_short_replace|L27_layer_out | 1.000 | 0.000 | 1587.54 | 1586.54 | 10.634 | 12.520 | {'prose': 72} |
| cross_donor|unrelated_short_replace|L27_layer_out | 1.000 | 0.000 | 1387.62 | 1386.62 | 11.291 | 13.177 | {'prose': 72} |
| cross_donor|unrelated_short_replace|L26_layer_out | 1.000 | 0.000 | 1165.74 | 1164.74 | 10.675 | 12.561 | {'prose': 72} |
| cross_donor|unrelated_short_replace|L27_layer_input | 1.000 | 0.000 | 1165.74 | 1164.74 | 10.675 | 12.561 | {'prose': 72} |
| cross_donor|unrelated_short_replace|L26_layer_input | 1.000 | 0.000 | 1165.07 | 1164.07 | 10.262 | 12.148 | {'prose': 72} |
| cross_donor|same_relation_diff_value_short_replace|L26_layer_out | 1.000 | 0.000 | 932.35 | 931.35 | 9.460 | 11.346 | {'prose': 72} |
| cross_donor|same_relation_diff_value_short_replace|L27_layer_input | 1.000 | 0.000 | 932.35 | 931.35 | 9.460 | 11.346 | {'prose': 72} |
| cross_donor|same_relation_diff_value_short_replace|L26_layer_input | 1.000 | 0.000 | 729.68 | 728.68 | 8.543 | 10.430 | {'prose': 72} |
| cross_donor|same_family_diff_value_short_replace|L27_layer_out | 1.000 | 0.000 | 673.40 | 672.40 | 7.012 | 8.899 | {'prose': 72} |
| cross_donor|same_family_diff_value_short_replace|L26_layer_input | 1.000 | 0.000 | 380.67 | 379.67 | 7.891 | 9.777 | {'prose': 72} |
| cross_donor|same_family_diff_value_short_replace|L26_layer_out | 1.000 | 0.000 | 346.85 | 345.85 | 7.669 | 9.556 | {'prose': 72} |
| cross_donor|same_family_diff_value_short_replace|L27_layer_input | 1.000 | 0.000 | 346.85 | 345.85 | 7.669 | 9.556 | {'prose': 72} |
| same_case|same_case_remove_delta|L26_layer_out | 1.000 | 0.000 | 169.54 | 168.54 | 4.312 | 6.198 | {'prose': 72} |
| same_case|same_case_remove_delta|L27_layer_input | 1.000 | 0.000 | 169.54 | 168.54 | 4.312 | 6.198 | {'prose': 72} |
| same_case|same_case_replace_short|L26_layer_out | 1.000 | 0.000 | 168.12 | 167.12 | 4.321 | 6.207 | {'prose': 72} |
| same_case|same_case_replace_short|L27_layer_input | 1.000 | 0.000 | 168.12 | 167.12 | 4.321 | 6.207 | {'prose': 72} |
| same_case|same_case_remove_delta|L27_layer_out | 1.000 | 0.000 | 168.08 | 167.08 | 4.279 | 6.165 | {'prose': 72} |
| same_case|same_case_replace_short|L27_layer_out | 1.000 | 0.000 | 167.69 | 166.69 | 4.279 | 6.165 | {'prose': 72} |

### glm4

| condition | drop | patched_top1 | patched_rank | rank_inc | patched_pmv | pmv_inc | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| cross_donor|same_family_diff_value_short_replace|L38_layer_out | 1.000 | 0.000 | 1829.00 | 1828.00 | 7.250 | 11.000 | {'continuation': 5} |
| cross_donor|same_family_diff_value_short_replace|L39_layer_input | 1.000 | 0.000 | 1829.00 | 1828.00 | 7.250 | 11.000 | {'continuation': 5} |
| cross_donor|same_family_diff_value_short_replace|L39_layer_out | 1.000 | 0.000 | 1798.20 | 1797.20 | 7.245 | 10.995 | {'continuation': 5} |
| cross_donor|same_family_diff_value_short_replace|L38_layer_input | 1.000 | 0.000 | 1701.40 | 1700.40 | 7.157 | 10.907 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_short_replace|L39_layer_out | 1.000 | 0.000 | 1205.40 | 1204.40 | 6.362 | 10.113 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_short_replace|L38_layer_out | 1.000 | 0.000 | 1136.40 | 1135.40 | 6.341 | 10.091 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_short_replace|L39_layer_input | 1.000 | 0.000 | 1136.40 | 1135.40 | 6.341 | 10.091 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_short_replace|L38_layer_input | 1.000 | 0.000 | 1082.80 | 1081.80 | 6.283 | 10.033 | {'continuation': 5} |
| same_case|same_case_replace_short|L38_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | -2.200 | 1.550 | {'continuation': 5} |
| same_case|same_case_remove_delta|L38_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | -2.200 | 1.550 | {'continuation': 5} |
| same_case|same_case_replace_short|L38_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | -2.212 | 1.538 | {'continuation': 5} |
| same_case|same_case_remove_delta|L38_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | -2.212 | 1.538 | {'continuation': 5} |
| same_case|same_case_replace_short|L39_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | -2.212 | 1.538 | {'continuation': 5} |
| same_case|same_case_remove_delta|L39_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | -2.212 | 1.538 | {'continuation': 5} |
| same_case|same_case_replace_short|L39_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | -2.225 | 1.525 | {'continuation': 5} |
| same_case|same_case_remove_delta|L39_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | -2.225 | 1.525 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_remove_donor_delta|L38_layer_out | 0.400 | 0.600 | 1.60 | 0.60 | -3.994 | -0.244 | {'continuation': 5} |
| cross_donor|same_relation_diff_value_remove_donor_delta|L39_layer_input | 0.400 | 0.600 | 1.60 | 0.60 | -3.994 | -0.244 | {'continuation': 5} |

### qwen3

| condition | drop | patched_top1 | patched_rank | rank_inc | patched_pmv | pmv_inc | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| cross_donor|same_relation_diff_value_short_replace|L33_layer_input | 0.667 | 0.333 | 2.33 | 1.33 | 0.042 | 4.292 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_family_diff_value_short_replace|L33_layer_input | 0.667 | 0.333 | 2.33 | 1.33 | 0.042 | 4.292 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_replace_short|L33_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -0.917 | 3.333 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_replace_short|L34_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | -0.917 | 3.333 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_replace_short|L34_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -1.000 | 3.250 | {'continuation': 2, 'prose': 1} |
| same_case|same_case_remove_delta|L33_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | -1.042 | 3.208 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_remove_delta|L33_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -1.042 | 3.208 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_remove_delta|L34_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | -1.042 | 3.208 | {'continuation': 1, 'prose': 2} |
| same_case|same_case_remove_delta|L34_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -1.042 | 3.208 | {'continuation': 2, 'prose': 1} |
| same_case|same_case_replace_short|L33_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | -1.083 | 3.167 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_relation_diff_value_remove_donor_delta|L34_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -1.167 | 3.083 | {'continuation': 2, 'prose': 1} |
| cross_donor|same_family_diff_value_remove_donor_delta|L34_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -1.167 | 3.083 | {'continuation': 2, 'prose': 1} |
| cross_donor|same_relation_diff_value_short_replace|L33_layer_out | 0.333 | 0.667 | 2.33 | 1.33 | -0.083 | 4.167 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_relation_diff_value_short_replace|L34_layer_input | 0.333 | 0.667 | 2.33 | 1.33 | -0.083 | 4.167 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_family_diff_value_short_replace|L33_layer_out | 0.333 | 0.667 | 2.33 | 1.33 | -0.083 | 4.167 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_family_diff_value_short_replace|L34_layer_input | 0.333 | 0.667 | 2.33 | 1.33 | -0.083 | 4.167 | {'continuation': 1, 'prose': 2} |
| cross_donor|same_relation_diff_value_short_replace|L34_layer_out | 0.333 | 0.667 | 2.00 | 1.00 | -1.125 | 3.125 | {'continuation': 2, 'prose': 1} |
| cross_donor|same_family_diff_value_short_replace|L34_layer_out | 0.333 | 0.667 | 2.00 | 1.00 | -1.125 | 3.125 | {'continuation': 2, 'prose': 1} |

