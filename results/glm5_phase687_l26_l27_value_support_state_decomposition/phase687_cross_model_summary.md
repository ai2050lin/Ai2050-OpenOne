# Phase 687 L26/L27 Value-Support State Decomposition

- generated: `2026-06-26 13:49:12`

| model | pairs | layers | best_component | comp_repair | comp_rank | best_cross | cross_repair | cross_rank |
|---|---:|---|---|---:|---:|---|---:|---:|
| deepseek7b | 72 | [26, 27] | component|same_case_add_delta|L26_layer_input | 1.000 | 1.00 | cross_donor|same_value_replace|L26_layer_out | 1.000 | 1.00 |
| glm4 | 5 | [38, 39] | component|same_case_add_delta|L38_layer_input | 1.000 | 1.00 | cross_donor|same_relation_diff_value_add_delta|L38_layer_out | 0.800 | 1.20 |
| qwen3 | 3 | [33, 34] | component|same_case_add_delta|L33_layer_input | 1.000 | 1.00 | cross_donor|same_relation_diff_value_add_delta|L33_layer_out | 1.000 | 1.00 |

## Best Component Conditions

### deepseek7b

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| component|same_case_add_delta|L26_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | -2.030 |
| component|same_case_replace|L26_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | -2.016 |
| component|same_case_add_delta|L26_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | -1.975 |
| component|same_case_replace|L26_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | -1.972 |
| component|same_case_add_delta|L27_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | -1.975 |
| component|same_case_replace|L27_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | -1.972 |
| component|same_case_add_delta|L27_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | -1.884 |
| component|same_case_replace|L27_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | -1.886 |
| component|same_case_add_delta|L26_attn_out | 0.069 | 0.069 | 10.71 | 156.99 | 1.849 |
| component|same_case_replace|L26_attn_out | 0.069 | 0.069 | 10.74 | 156.96 | 1.847 |
| component|random_same_norm|L27_layer_out | 0.069 | 0.069 | 186.01 | -18.32 | 2.922 |
| component|same_case_add_delta|L27_attn_out | 0.042 | 0.042 | 51.39 | 116.31 | 3.275 |

### glm4

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| component|same_case_add_delta|L38_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.788 |
| component|same_case_replace|L38_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.788 |
| component|same_case_add_delta|L38_mlp_out | 1.000 | 1.000 | 1.00 | 1.00 | -2.688 |
| component|same_case_replace|L38_mlp_out | 1.000 | 1.000 | 1.00 | 1.00 | -2.675 |
| component|same_case_add_delta|L38_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 |
| component|same_case_replace|L38_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.737 |
| component|same_case_add_delta|L39_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 |
| component|same_case_replace|L39_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.737 |
| component|same_case_add_delta|L39_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 |
| component|same_case_replace|L39_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.750 |
| component|random_same_norm|L39_layer_out | 0.600 | 0.600 | 1.40 | 0.60 | -1.938 |
| component|random_same_norm|L38_mlp_out | 0.400 | 0.400 | 1.60 | 0.40 | -2.188 |

### qwen3

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| component|same_case_add_delta|L33_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.917 |
| component|same_case_replace|L33_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -4.000 |
| component|same_case_add_delta|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| component|same_case_replace|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.917 |
| component|same_case_add_delta|L34_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| component|same_case_replace|L34_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | -3.917 |
| component|same_case_add_delta|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| component|same_case_replace|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| component|random_same_norm|L34_layer_input | 0.333 | 0.333 | 1.67 | 0.33 | -1.250 |
| component|same_case_add_delta|L34_attn_out | 0.333 | 0.333 | 1.67 | 0.33 | -1.083 |
| component|same_case_replace|L34_attn_out | 0.333 | 0.333 | 1.67 | 0.33 | -1.083 |
| component|same_case_add_delta|L34_mlp_out | 0.333 | 0.333 | 1.67 | 0.33 | -1.292 |

## Best Cross-Donor Conditions

### deepseek7b

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| cross_donor|same_value_replace|L26_layer_out | 1.000 | 1.000 | 1.00 | 23.62 | -1.180 |
| cross_donor|same_value_replace|L27_layer_out | 1.000 | 1.000 | 1.00 | 23.62 | -1.148 |
| cross_donor|same_value_add_delta|L27_layer_out | 0.625 | 0.625 | 4.50 | 20.12 | -1.125 |
| cross_donor|same_family_diff_value_replace|L27_layer_out | 0.528 | 0.528 | 776.39 | -608.69 | 3.460 |
| cross_donor|same_value_add_delta|L26_layer_out | 0.500 | 0.500 | 2.25 | 22.38 | -0.641 |
| cross_donor|same_family_diff_value_add_delta|L26_layer_out | 0.361 | 0.361 | 14.83 | 152.86 | -0.185 |
| cross_donor|same_family_diff_value_replace|L26_layer_out | 0.264 | 0.264 | 448.33 | -280.64 | 2.852 |
| cross_donor|same_family_diff_value_add_delta|L27_layer_out | 0.181 | 0.181 | 28.79 | 138.90 | 0.692 |
| cross_donor|same_relation_diff_value_add_delta|L26_layer_out | 0.097 | 0.097 | 165.31 | 2.39 | 3.669 |
| cross_donor|unrelated_add_delta|L26_layer_out | 0.014 | 0.014 | 323.90 | -156.21 | 4.359 |
| cross_donor|same_relation_diff_value_add_delta|L27_layer_out | 0.000 | 0.000 | 402.18 | -234.49 | 5.025 |
| cross_donor|unrelated_add_delta|L27_layer_out | 0.000 | 0.000 | 617.75 | -450.06 | 5.657 |

### glm4

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| cross_donor|same_relation_diff_value_add_delta|L38_layer_out | 0.800 | 0.800 | 1.20 | 0.80 | -2.000 |
| cross_donor|same_relation_diff_value_add_delta|L39_layer_out | 0.800 | 0.800 | 1.20 | 0.80 | -1.325 |
| cross_donor|same_family_diff_value_add_delta|L38_layer_out | 0.600 | 0.600 | 1.40 | 0.60 | -2.669 |
| cross_donor|same_family_diff_value_add_delta|L39_layer_out | 0.600 | 0.600 | 1.40 | 0.60 | -1.944 |
| cross_donor|same_relation_diff_value_replace|L39_layer_out | 0.000 | 0.000 | 771.20 | -769.20 | 5.842 |
| cross_donor|same_family_diff_value_replace|L38_layer_out | 0.000 | 0.000 | 776.60 | -774.60 | 5.905 |
| cross_donor|same_relation_diff_value_replace|L38_layer_out | 0.000 | 0.000 | 779.60 | -777.60 | 5.820 |
| cross_donor|same_family_diff_value_replace|L39_layer_out | 0.000 | 0.000 | 944.40 | -942.40 | 6.039 |

### qwen3

| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |
|---|---:|---:|---:|---:|---:|
| cross_donor|same_relation_diff_value_add_delta|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.792 |
| cross_donor|same_relation_diff_value_replace|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -2.875 |
| cross_donor|same_relation_diff_value_add_delta|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| cross_donor|same_relation_diff_value_replace|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -4.042 |
| cross_donor|same_family_diff_value_add_delta|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.792 |
| cross_donor|same_family_diff_value_replace|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -2.875 |
| cross_donor|same_family_diff_value_add_delta|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -3.875 |
| cross_donor|same_family_diff_value_replace|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -4.042 |

