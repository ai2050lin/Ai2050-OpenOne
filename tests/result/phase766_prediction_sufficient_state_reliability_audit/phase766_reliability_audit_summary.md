# Phase 766 Prediction-Sufficient State Reliability Audit (confirm)

- Status: `complete`
- Input: Phase 765 commonsense confirm rows; no model was loaded.
- Purpose: compare target-top1 success vs failure states.

## Base Reliability By Relation

| model | relation | n | top1 | target rank | contrast rank |
|---|---|---:|---:|---:|---:|
| qwen3 | `category` | 36 | 0.861 | 1.389 | 149.361 |
| qwen3 | `edible` | 36 | 0.694 | 1.306 | 2.583 |
| qwen3 | `grows_on_tree` | 36 | 0.861 | 1.278 | 3.278 |
| glm4 | `category` | 36 | 0.806 | 1.444 | 10.750 |
| glm4 | `edible` | 36 | 0.500 | 1.667 | 1.972 |
| glm4 | `grows_on_tree` | 36 | 0.472 | 1.861 | 2.417 |
| deepseek7b | `category` | 36 | 0.361 | 5.000 | 184.722 |
| deepseek7b | `edible` | 36 | 0.111 | 7.028 | 9.722 |
| deepseek7b | `grows_on_tree` | 36 | 0.083 | 5.472 | 3.861 |

## Success Minus Failure By Source Group

| model | source | success n | failure n | target drop gap | attention gap | direct boost gap | route suppression gap |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `answer_prefix` | 174 | 42 | 0.024 | 0.004 | 0.000 | 0.009 |
| qwen3 | `instruction` | 174 | 42 | 0.049 | 0.002 | 0.111 | 0.066 |
| qwen3 | `object_tokens` | 174 | 42 | 0.019 | 0.005 | 0.015 | -0.012 |
| qwen3 | `question` | 174 | 42 | 0.032 | -0.007 | 0.002 | -0.024 |
| qwen3 | `relation_tokens` | 174 | 42 | 0.019 | -0.010 | -0.013 | -0.015 |
| glm4 | `answer_prefix` | 128 | 88 | -0.005 | -0.008 | 0.001 | 0.000 |
| glm4 | `instruction` | 128 | 88 | -0.011 | 0.020 | 0.007 | 0.004 |
| glm4 | `object_tokens` | 128 | 88 | -0.007 | -0.005 | 0.000 | 0.000 |
| glm4 | `question` | 128 | 88 | -0.007 | -0.008 | 0.003 | 0.001 |
| glm4 | `relation_tokens` | 128 | 88 | -0.012 | 0.002 | 0.000 | 0.000 |
| deepseek7b | `answer_prefix` | 40 | 176 | 0.018 | -0.007 | 0.000 | -0.006 |
| deepseek7b | `instruction` | 40 | 176 | 0.015 | 0.132 | -0.085 | 0.279 |
| deepseek7b | `object_tokens` | 40 | 176 | 0.058 | -0.039 | -0.090 | 0.084 |
| deepseek7b | `question` | 40 | 176 | 0.009 | -0.094 | -0.132 | 0.045 |
| deepseek7b | `relation_tokens` | 40 | 176 | 0.026 | -0.031 | -0.025 | -0.022 |

## Top Attention-Mass Gaps

| model | key | success n | failure n | gap | success | failure |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `{'obs_relation': 'edible', 'source_group': 'instruction'}` | 50 | 22 | -0.021 | 0.885 | 0.906 |
| qwen3 | `{'obs_relation': 'edible', 'source_group': 'question'}` | 50 | 22 | 0.014 | 0.090 | 0.077 |
| qwen3 | `{'obs_relation': 'edible', 'source_group': 'object_tokens'}` | 50 | 22 | 0.009 | 0.018 | 0.009 |
| qwen3 | `{'obs_relation': 'edible', 'source_group': 'answer_prefix'}` | 50 | 22 | 0.006 | 0.018 | 0.012 |
| qwen3 | `{'obs_relation': 'edible', 'source_group': 'relation_tokens'}` | 50 | 22 | -0.004 | 0.036 | 0.040 |
| glm4 | `{'obs_relation': 'grows_on_tree', 'source_group': 'question'}` | 34 | 38 | 0.080 | 0.254 | 0.173 |
| glm4 | `{'obs_relation': 'grows_on_tree', 'source_group': 'relation_tokens'}` | 34 | 38 | 0.080 | 0.140 | 0.059 |
| glm4 | `{'obs_relation': 'grows_on_tree', 'source_group': 'instruction'}` | 34 | 38 | -0.061 | 0.713 | 0.774 |
| glm4 | `{'obs_relation': 'grows_on_tree', 'source_group': 'answer_prefix'}` | 34 | 38 | -0.016 | 0.009 | 0.024 |
| glm4 | `{'obs_relation': 'edible', 'source_group': 'question'}` | 36 | 36 | -0.014 | 0.159 | 0.173 |
| glm4 | `{'obs_relation': 'edible', 'source_group': 'instruction'}` | 36 | 36 | 0.014 | 0.796 | 0.782 |
| glm4 | `{'obs_relation': 'category', 'source_group': 'instruction'}` | 58 | 14 | 0.011 | 0.881 | 0.870 |
| glm4 | `{'obs_relation': 'edible', 'source_group': 'object_tokens'}` | 36 | 36 | -0.008 | 0.022 | 0.030 |
| deepseek7b | `{'obs_relation': 'category', 'source_group': 'instruction'}` | 26 | 46 | 0.068 | 0.806 | 0.738 |
| deepseek7b | `{'obs_relation': 'category', 'source_group': 'question'}` | 26 | 46 | -0.047 | 0.143 | 0.190 |
| deepseek7b | `{'obs_relation': 'category', 'source_group': 'object_tokens'}` | 26 | 46 | 0.013 | 0.074 | 0.061 |
| deepseek7b | `{'obs_relation': 'category', 'source_group': 'relation_tokens'}` | 26 | 46 | -0.008 | 0.010 | 0.019 |
| deepseek7b | `{'obs_relation': 'category', 'source_group': 'answer_prefix'}` | 26 | 46 | -0.003 | 0.005 | 0.008 |

## Strict Interpretation

- This is an observational audit over Phase 765 rows, not a new intervention.
- If failures have lower attention/direct gaps on object or relation sources, the bottleneck is likely state formation.
- If failures have similar source effects but low target top1, the bottleneck is more likely readout threshold or candidate competition.
