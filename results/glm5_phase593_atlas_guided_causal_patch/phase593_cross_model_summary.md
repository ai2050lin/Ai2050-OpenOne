# Phase593 Cross-Model Summary

Atlas-guided causal patch validation. This tests whether Phase592 Level-2 projection nodes become causal repair nodes.

## qwen3

- target_rows=5, alpha=1.0, nodes=[('prompt_last', 34), ('prompt_last', 33), ('prompt_last', 32), ('query_category', 32), ('query_category', 34), ('prompt_last', 30)]

| rank | node | mode | switch | margin_gain | specific_gain | common | correct_specific | positive_margin |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | prompt_last L30 | specific_norm_raw | 2/5 | +0.250 | +0.250 | -0.213 | +0.162 | 0.80 |
| 2 | prompt_last L32 | specific_norm_raw | 2/5 | +0.200 | +0.200 | +0.073 | +0.120 | 0.80 |
| 3 | prompt_last L34 | specific_norm_raw | 2/5 | +0.200 | +0.200 | +0.240 | +0.062 | 0.80 |
| 4 | prompt_last L33 | specific_norm_raw | 2/5 | +0.150 | +0.150 | -0.069 | +0.076 | 0.80 |
| 5 | prompt_last L32 | specific_only | 2/5 | +0.100 | +0.100 | +0.089 | +0.049 | 1.00 |
| 6 | prompt_last L34 | raw | 1/5 | +0.100 | +0.100 | +0.886 | +0.031 | 0.80 |
| 7 | prompt_last L33 | specific_only | 1/5 | +0.100 | +0.100 | +0.086 | +0.051 | 0.60 |
| 8 | prompt_last L34 | common_norm_raw | 1/5 | +0.075 | +0.075 | -0.333 | +0.012 | 0.80 |
| 9 | prompt_last L30 | specific_only | 1/5 | +0.075 | +0.075 | +0.018 | +0.050 | 0.80 |
| 10 | prompt_last L33 | raw | 1/5 | +0.050 | +0.050 | +0.799 | +0.018 | 0.80 |
| 11 | prompt_last L32 | common_only | 1/5 | +0.050 | +0.050 | +0.033 | +0.019 | 0.60 |
| 12 | query_category L34 | specific_only | 1/5 | +0.050 | +0.050 | -0.012 | +0.026 | 0.60 |

## glm4

- target_rows=4, alpha=1.0, nodes=[('prompt_last', 38), ('prompt_last', 39), ('prompt_last', 37), ('prompt_last', 36), ('prompt_last', 35), ('prompt_last', 34)]

| rank | node | mode | switch | margin_gain | specific_gain | common | correct_specific | positive_margin |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | prompt_last L34 | common_norm_raw | 0/4 | +0.016 | +0.016 | -9.331 | +0.004 | 0.50 |
| 2 | prompt_last L37 | raw | 0/4 | +0.016 | +0.016 | -0.597 | +0.004 | 0.50 |
| 3 | prompt_last L35 | common_norm_raw | 0/4 | +0.016 | +0.016 | -9.354 | -0.002 | 0.50 |
| 4 | prompt_last L38 | common_only | 0/4 | +0.000 | +0.000 | -0.400 | +0.004 | 0.50 |
| 5 | prompt_last L34 | specific_norm_raw | 0/4 | +0.000 | +0.000 | -0.203 | +0.031 | 0.75 |
| 6 | prompt_last L38 | specific_only | 0/4 | +0.000 | +0.000 | -0.001 | +0.004 | 0.50 |
| 7 | prompt_last L37 | common_only | 0/4 | +0.000 | +0.000 | -0.204 | +0.006 | 0.25 |
| 8 | prompt_last L38 | random_same_norm | 0/4 | +0.000 | +0.000 | -0.250 | +0.004 | 0.25 |
| 9 | prompt_last L39 | common_norm_raw | 0/4 | +0.000 | +0.000 | -29.404 | +0.000 | 0.00 |
| 10 | prompt_last L39 | common_only | 0/4 | +0.000 | +0.000 | -0.763 | +0.000 | 0.00 |
| 11 | prompt_last L39 | random_same_norm | 0/4 | +0.000 | +0.000 | +0.012 | +0.000 | 0.00 |
| 12 | prompt_last L39 | raw | 0/4 | +0.000 | +0.000 | -0.378 | +0.000 | 0.00 |

## deepseek7b

- target_rows=21, alpha=1.0, nodes=[('rule_value', 26), ('prompt_last', 26), ('rule_relation', 18), ('rule_relation', 20), ('query_relation', 16), ('query_relation', 19)]

| rank | node | mode | switch | margin_gain | specific_gain | common | correct_specific | positive_margin |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | rule_relation L20 | specific_only | 1/21 | -0.007 | -0.007 | -0.004 | +0.005 | 0.48 |
| 2 | rule_relation L18 | common_norm_raw | 1/21 | -0.018 | -0.018 | -0.039 | -0.002 | 0.52 |
| 3 | rule_relation L20 | common_only | 1/21 | -0.021 | -0.021 | +0.008 | -0.004 | 0.33 |
| 4 | rule_relation L18 | specific_norm_raw | 0/21 | +0.012 | +0.012 | -0.024 | +0.022 | 0.52 |
| 5 | rule_relation L20 | common_norm_raw | 0/21 | +0.012 | +0.012 | +0.019 | +0.011 | 0.52 |
| 6 | prompt_last L26 | random_same_norm | 0/21 | +0.012 | +0.012 | +0.104 | +0.006 | 0.71 |
| 7 | rule_relation L18 | raw | 0/21 | +0.006 | +0.006 | -0.019 | +0.016 | 0.62 |
| 8 | rule_relation L18 | specific_only | 0/21 | -0.002 | -0.002 | +0.006 | +0.001 | 0.67 |
| 9 | rule_value L26 | random_same_norm | 0/21 | -0.003 | -0.003 | +0.002 | +0.002 | 0.52 |
| 10 | rule_relation L18 | common_only | 0/21 | -0.006 | -0.006 | -0.003 | +0.022 | 0.43 |
| 11 | rule_value L26 | raw | 0/21 | -0.006 | -0.006 | +0.002 | -0.002 | 0.38 |
| 12 | rule_value L26 | specific_only | 0/21 | -0.009 | -0.009 | -0.006 | -0.002 | 0.43 |

## Mode Best

| model | mode | node | switch | margin_gain | common | correct_specific |
|---|---|---|---:|---:|---:|---:|
| qwen3 | raw | prompt_last L34 | 1/5 | +0.100 | +0.886 | +0.031 |
| qwen3 | specific_only | prompt_last L32 | 2/5 | +0.100 | +0.089 | +0.049 |
| qwen3 | specific_norm_raw | prompt_last L30 | 2/5 | +0.250 | -0.213 | +0.162 |
| qwen3 | common_only | prompt_last L32 | 1/5 | +0.050 | +0.033 | +0.019 |
| qwen3 | common_norm_raw | prompt_last L34 | 1/5 | +0.075 | -0.333 | +0.012 |
| qwen3 | random_same_norm | prompt_last L33 | 1/5 | +0.000 | -0.185 | -0.007 |
| glm4 | raw | prompt_last L37 | 0/4 | +0.016 | -0.597 | +0.004 |
| glm4 | specific_only | prompt_last L38 | 0/4 | +0.000 | -0.001 | +0.004 |
| glm4 | specific_norm_raw | prompt_last L34 | 0/4 | +0.000 | -0.203 | +0.031 |
| glm4 | common_only | prompt_last L38 | 0/4 | +0.000 | -0.400 | +0.004 |
| glm4 | common_norm_raw | prompt_last L34 | 0/4 | +0.016 | -9.331 | +0.004 |
| glm4 | random_same_norm | prompt_last L38 | 0/4 | +0.000 | -0.250 | +0.004 |
| deepseek7b | raw | rule_relation L18 | 0/21 | +0.006 | -0.019 | +0.016 |
| deepseek7b | specific_only | rule_relation L20 | 1/21 | -0.007 | -0.004 | +0.005 |
| deepseek7b | specific_norm_raw | rule_relation L18 | 0/21 | +0.012 | -0.024 | +0.022 |
| deepseek7b | common_only | rule_relation L20 | 1/21 | -0.021 | +0.008 | -0.004 |
| deepseek7b | common_norm_raw | rule_relation L18 | 1/21 | -0.018 | -0.039 | -0.002 |
| deepseek7b | random_same_norm | prompt_last L26 | 0/21 | +0.012 | +0.104 | +0.006 |

## Objective facts

- Qwen3 has limited target rows but specific_norm_raw prompt_last patches reach 2/5 switch.
- GLM4 has no switch and near-zero margin gains.
- DS7B has 21 target rows; no tested atlas node gives reliable positive margin or winner repair.
- DS7B observed 1/21 switches are not evidence of repair because their mean margin gains are negative and common/random controls can also switch.
- Phase592 projection nodes therefore remain candidates; they are not yet upgraded to robust causal repair nodes.
