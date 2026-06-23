# Phase591 Cross-Model Summary

Value candidate internal ranking audit.

## qwen3

- cases=64, target_cases=5, layers=[27, 34], alpha=1.0

| mode | target | switch | common | correct_delta | old_top_wrong_delta | correct_specific | old_top_wrong_specific | margin_gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repair_prompt | 5 | 5/5 | -3.273 | +2.373 | -5.528 | +5.646 | -2.255 | +7.901 |
| patch_prompt_last_residual_attn | 5 | 2/5 | +0.907 | +0.938 | +0.763 | +0.032 | -0.143 | +0.175 |
| patch_query_relation_residual_attn | 5 | 0/5 | -0.106 | -0.094 | -0.118 | +0.013 | -0.012 | +0.025 |
| wrong_relation_prompt | 5 | 1/5 | -1.078 | -0.423 | -4.801 | +0.655 | -3.723 | +4.378 |
| random_prompt_last_residual_attn | 5 | 1/5 | -0.597 | -0.554 | -0.629 | +0.043 | -0.032 | +0.075 |
| random_query_relation_residual_attn | 5 | 1/5 | -0.066 | -0.024 | -0.074 | +0.042 | -0.008 | +0.050 |

Top-wrong labels on target cases:

- same_relation_other_category: 5
- value_prior_higher_than_correct: 4
- repeated_value: 2
- wrong_relation_any_category: 1

Top-wrong values on target cases:

- v22: 3
- v48: 1
- v91: 1

Mean top-wrong embedding cosine to correct: 0.857

## glm4

- cases=64, target_cases=4, layers=[30, 38], alpha=1.0

| mode | target | switch | common | correct_delta | old_top_wrong_delta | correct_specific | old_top_wrong_specific | margin_gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repair_prompt | 4 | 4/4 | -2.388 | +0.459 | -1.791 | +2.848 | +0.598 | +2.250 |
| patch_prompt_last_residual_attn | 4 | 0/4 | -1.730 | -1.734 | -1.734 | -0.004 | -0.004 | +0.000 |
| patch_query_relation_residual_attn | 4 | 0/4 | +0.006 | -0.029 | +0.018 | -0.035 | +0.012 | -0.047 |
| wrong_relation_prompt | 4 | 0/4 | -4.625 | -5.535 | -7.340 | -0.910 | -2.715 | +1.805 |
| random_prompt_last_residual_attn | 4 | 0/4 | -0.405 | -0.398 | -0.398 | +0.008 | +0.008 | +0.000 |
| random_query_relation_residual_attn | 4 | 0/4 | -0.001 | -0.022 | +0.025 | -0.021 | +0.025 | -0.047 |

Top-wrong labels on target cases:

- same_relation_other_category: 4
- repeated_value: 3
- same_category_wrong_relation: 1
- value_prior_higher_than_correct: 1
- wrong_relation_any_category: 1

Top-wrong values on target cases:

- v91: 2
- v22: 1
- v48: 1

Mean top-wrong embedding cosine to correct: 0.646

## deepseek7b

- cases=64, target_cases=21, layers=[21, 26], alpha=1.0

| mode | target | switch | common | correct_delta | old_top_wrong_delta | correct_specific | old_top_wrong_specific | margin_gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repair_prompt | 21 | 21/21 | +2.517 | +7.902 | +0.524 | +5.385 | -1.994 | +7.379 |
| patch_prompt_last_residual_attn | 21 | 0/21 | +6.110 | +6.146 | +6.104 | +0.036 | -0.006 | +0.042 |
| patch_query_relation_residual_attn | 21 | 0/21 | +0.026 | +0.024 | +0.023 | -0.002 | -0.003 | +0.001 |
| wrong_relation_prompt | 21 | 2/21 | +1.091 | -1.394 | +1.039 | -2.485 | -0.052 | -2.434 |
| random_prompt_last_residual_attn | 21 | 1/21 | -0.124 | -0.093 | -0.090 | +0.030 | +0.033 | -0.003 |
| random_query_relation_residual_attn | 21 | 0/21 | +0.057 | +0.026 | +0.061 | -0.031 | +0.004 | -0.035 |

Top-wrong labels on target cases:

- wrong_relation_any_category: 20
- same_relation_other_category: 19
- repeated_value: 18
- same_category_wrong_relation: 15
- value_prior_higher_than_correct: 3

Top-wrong values on target cases:

- v48: 14
- v05: 5
- v22: 2

Mean top-wrong embedding cosine to correct: 0.864

## Objective facts

- Prompt-level repair creates large candidate-specific support for the correct value in all three models.
- Hidden residual+attention patch creates mostly common candidate activation, especially in DS7B.
- DS7B patch_prompt_last_residual_attn: common +6.110, correct_specific +0.036, margin_gain +0.042, switch 0/21.
- DS7B repair_prompt: correct_specific +5.385, old_top_wrong_specific -1.994, margin_gain +7.379, switch 21/21.
- DS7B old top-wrong candidates usually have rule-level overlap: same_relation_other_category 19/21, wrong_relation_any_category 20/21, repeated_value 18/21.
