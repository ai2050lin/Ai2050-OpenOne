# Phase590 Cross-Model Summary

Value winner multi-layer cumulative patch audit.

## qwen3

- cases=24, layers=[27, 34], alpha=1.0

| mode | position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |
|---|---|---|---:|---:|---:|---:|---:|
| repair_cumulative | prompt_last | all_both | 2 | 1/2 | +0.852 | +0.539 | +0.313 |
| repair_cumulative | prompt_last | residual_attn_both | 2 | 1/2 | +0.830 | +0.580 | +0.250 |
| repair_cumulative | prompt_last | residual_both | 2 | 1/2 | +0.777 | +0.590 | +0.188 |
| wrong_relation_cumulative | prompt_last | all_both | 2 | 1/2 | -0.406 | -0.656 | +0.250 |
| wrong_relation_cumulative | prompt_last | residual_attn_both | 2 | 1/2 | -0.048 | -0.235 | +0.188 |
| wrong_relation_cumulative | prompt_last | residual_mlp_both | 2 | 1/2 | -1.244 | -1.432 | +0.188 |
| random_cumulative | prompt_last | residual_both | 2 | 1/2 | -0.741 | -0.866 | +0.125 |
| random_cumulative | prompt_last | residual_mlp_both | 2 | 1/2 | -0.605 | -0.730 | +0.125 |
| random_cumulative | prompt_last | all_both | 2 | 1/2 | -0.676 | -0.801 | +0.125 |

## glm4

- cases=24, layers=[30, 38], alpha=1.0

| mode | position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |
|---|---|---|---:|---:|---:|---:|---:|
| repair_cumulative | prompt_last | residual_attn_both | 1 | 0/1 | -1.615 | -1.740 | +0.125 |
| repair_cumulative | prompt_last | residual_mlp_both | 1 | 0/1 | -1.972 | -2.097 | +0.125 |
| repair_cumulative | prompt_last | all_both | 1 | 0/1 | -2.018 | -2.143 | +0.125 |
| wrong_relation_cumulative | prompt_last | residual_mlp_both | 1 | 0/1 | -10.091 | -10.216 | +0.125 |
| wrong_relation_cumulative | prompt_last | residual_attn_both | 1 | 0/1 | -8.364 | -8.426 | +0.063 |
| wrong_relation_cumulative | prompt_last | residual_both | 1 | 0/1 | -8.308 | -8.371 | +0.062 |
| random_cumulative | prompt_last | mlp_both | 1 | 0/1 | -0.171 | -0.171 | +0.000 |
| random_cumulative | query_relation | residual_attn_both | 1 | 0/1 | +0.010 | +0.010 | +0.000 |
| random_cumulative | query_relation | all_both | 1 | 0/1 | +0.004 | +0.004 | +0.000 |

## deepseek7b

- cases=24, layers=[21, 26], alpha=1.0

| mode | position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |
|---|---|---|---:|---:|---:|---:|---:|
| repair_cumulative | query_relation | residual_attn_both | 9 | 0/9 | +0.056 | +0.020 | +0.035 |
| repair_cumulative | query_relation | all_both | 9 | 0/9 | +0.054 | +0.032 | +0.021 |
| repair_cumulative | query_relation | residual_mlp_both | 9 | 0/9 | +0.032 | +0.016 | +0.016 |
| wrong_relation_cumulative | prompt_last | residual_both | 9 | 0/9 | +2.530 | +2.452 | +0.078 |
| wrong_relation_cumulative | prompt_last | residual_attn_both | 9 | 0/9 | +2.334 | +2.298 | +0.036 |
| wrong_relation_cumulative | query_relation | mlp_both | 9 | 0/9 | +0.021 | +0.006 | +0.016 |
| random_cumulative | prompt_last | residual_both | 9 | 0/9 | -0.399 | -0.387 | -0.013 |
| random_cumulative | query_relation | mlp_both | 9 | 0/9 | -0.012 | +0.001 | -0.013 |
| random_cumulative | query_relation | residual_attn_both | 9 | 0/9 | +0.059 | +0.072 | -0.013 |

## DS7B repair detail

| position | combo | target | switch | correct_gain | top_wrong_gain | margin_gain |
|---|---|---:|---:|---:|---:|---:|
| prompt_last | all_both | 9 | 0/9 | +6.307 | +6.307 | +0.000 |
| prompt_last | attn_both | 9 | 0/9 | +2.061 | +2.060 | +0.000 |
| prompt_last | mlp_both | 9 | 0/9 | -0.190 | -0.150 | -0.040 |
| prompt_last | residual_attn_both | 9 | 0/9 | +6.559 | +6.551 | +0.007 |
| prompt_last | residual_both | 9 | 0/9 | +6.471 | +6.485 | -0.014 |
| prompt_last | residual_mlp_both | 9 | 0/9 | +6.172 | +6.220 | -0.048 |
| query_relation | all_both | 9 | 0/9 | +0.054 | +0.032 | +0.021 |
| query_relation | attn_both | 9 | 0/9 | +0.030 | +0.036 | -0.006 |
| query_relation | mlp_both | 9 | 0/9 | +0.028 | +0.063 | -0.035 |
| query_relation | residual_attn_both | 9 | 0/9 | +0.056 | +0.020 | +0.035 |
| query_relation | residual_both | 9 | 0/9 | +0.008 | -0.005 | +0.014 |
| query_relation | residual_mlp_both | 9 | 0/9 | +0.032 | +0.016 | +0.016 |

## Objective facts

- Qwen3 has limited target cases and shows 1/2 switch, but random and wrong-relation controls also switch.
- GLM4 has only 1 target case and no switch; residual cumulative patch mostly suppresses both correct and wrong candidates.
- DS7B has 9 target cases; prompt_last repair strongly raises correct and top-wrong together, but target switch remains 0/9.
- DS7B query_relation repair gives small positive margin gains, but still no winner switch.
