# Phase 801 Target-Neutral Suppressor Causal Test (confirm)

- Status: `complete`
- Boundary: hidden-state route deltas are decomposed into raw, target-neutral, and target-only components.
- This tests whether blocker suppression survives removal of the direct target-readout direction.

## By Projection

| model | projection | rows | cases | target gain | blocker suppression | resolved | new rate | neutral score | token gain | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `target_neutral` | 12 | 6 | -0.542 | 0.747 | 0.229 | 0.392 | 0.068 | 0.000 | `{"neutral_suppression_with_residual_target_gain": 10, "weak_or_mixed": 2}` |
| qwen3 | `raw` | 12 | 6 | 2.852 | 0.691 | 0.728 | 0.041 | 0.125 | 0.000 | `{"raw_suppressor_like": 10, "weak_or_mixed": 2}` |
| qwen3 | `target_only` | 12 | 6 | 3.503 | -0.051 | 0.771 | 0.000 | 0.000 | 0.000 | `{"target_only_threshold_shift": 8, "weak_or_mixed": 4}` |
| glm4 | `target_neutral` | 12 | 6 | 0.174 | 0.418 | 0.338 | 0.193 | 0.097 | 0.000 | `{"neutral_suppression_with_residual_target_gain": 8, "target_neutral_suppressor_evidence": 1, "weak_or_mixed": 3}` |
| glm4 | `raw` | 12 | 6 | 1.146 | 0.419 | 0.570 | 0.068 | 0.104 | 0.000 | `{"raw_suppressor_like": 7, "weak_or_mixed": 5}` |
| glm4 | `target_only` | 12 | 6 | 0.979 | 0.002 | 0.479 | 0.019 | 0.000 | 0.000 | `{"target_only_threshold_shift": 8, "weak_or_mixed": 4}` |
| deepseek7b | `target_neutral` | 12 | 6 | 0.702 | -0.424 | 0.265 | 0.427 | 0.000 | 0.000 | `{"weak_or_mixed": 12}` |
| deepseek7b | `target_only` | 12 | 6 | 2.577 | -0.107 | 0.608 | 0.048 | 0.000 | 0.000 | `{"target_only_threshold_shift": 10, "weak_or_mixed": 2}` |
| deepseek7b | `raw` | 12 | 6 | 3.016 | -0.530 | 0.671 | 0.116 | 0.000 | 0.000 | `{"weak_or_mixed": 12}` |

## Top Target-Neutral Triplets

| model | case | route | raw target | neutral target | raw suppress | neutral suppress | neutral new | neutral score | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 3.750 | -0.062 | 1.117 | 1.242 | 0.374 | 0.287 | False |
| qwen3 | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 3.688 | 0.062 | 0.773 | 0.913 | 0.306 | 0.213 | False |
| qwen3 | `p765_0002_commonsense_statement_fruit:apple:category` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 0.375 | -0.500 | 0.994 | 1.036 | 0.383 | 0.055 | False |
| qwen3 | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 1.375 | -0.750 | 0.924 | 0.991 | 0.298 | 0.000 | False |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 4.188 | -1.094 | 0.939 | 0.943 | 0.378 | 0.000 | False |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 1.812 | -1.750 | 0.662 | 0.832 | 0.596 | 0.000 | False |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 4.062 | -0.906 | 1.057 | 0.824 | 0.371 | 0.000 | False |
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 1.812 | -1.125 | 0.679 | 0.782 | 0.477 | 0.000 | False |
| qwen3 | `p765_0058_commonsense_statement_object:chair:edible` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 2.688 | -0.969 | 0.732 | 0.716 | 0.576 | 0.000 | False |
| qwen3 | `p765_0056_commonsense_statement_object:chair:category` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 2.281 | 1.031 | 0.649 | 0.601 | 0.100 | 0.000 | False |
| qwen3 | `p765_0006_commonsense_statement_fruit:apple:grows_on_tree` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 1.688 | -0.750 | 0.041 | 0.111 | 0.470 | 0.000 | False |
| qwen3 | `p765_0002_commonsense_statement_fruit:apple:category` | `attn:L34+attn:L31+mlp:L34+mlp:L35` | 6.500 | 0.312 | -0.270 | -0.029 | 0.368 | 0.000 | False |
| glm4 | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.375 | -0.250 | 0.498 | 0.523 | 0.054 | 0.084 | True |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 0.688 | -0.109 | 0.757 | 0.753 | 0.239 | 0.169 | False |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | 0.344 | 0.784 | 0.790 | 0.242 | 0.156 | False |
| glm4 | `p765_0057_commonsense_question_object:chair:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 3.055 | 0.125 | 0.570 | 0.576 | 0.214 | 0.138 | False |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 0.422 | -0.172 | 0.384 | 0.374 | 0.231 | 0.043 | False |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 2.266 | 0.453 | 0.138 | 0.114 | 0.127 | 0.013 | False |
| glm4 | `p765_0024_commonsense_statement_animal:cat:grows_on_tree` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 1.016 | 0.391 | 0.029 | 0.040 | 0.074 | 0.006 | False |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 2.781 | 1.219 | 0.673 | 0.653 | 0.033 | 0.000 | False |
| glm4 | `p765_0057_commonsense_question_object:chair:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 1.492 | 0.836 | 0.496 | 0.498 | 0.112 | 0.000 | False |
| glm4 | `p765_0055_commonsense_question_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | -0.422 | -1.125 | 0.476 | 0.473 | 0.494 | 0.000 | False |
| glm4 | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L38+attn:L33+attn:L29+attn:L35` | 2.156 | 1.031 | 0.268 | 0.257 | 0.027 | 0.000 | False |
| glm4 | `p765_0056_commonsense_statement_object:chair:category` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | -0.078 | -0.656 | -0.041 | -0.039 | 0.464 | 0.000 | False |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 3.031 | -0.344 | 0.040 | 0.164 | 0.233 | 0.005 | False |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 7.176 | -2.949 | -0.543 | 0.218 | 0.855 | 0.000 | False |
| deepseek7b | `p765_0005_commonsense_question_fruit:apple:grows_on_tree` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 2.844 | 2.719 | 0.183 | 0.168 | 0.163 | 0.000 | False |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 2.688 | 2.500 | 0.150 | 0.167 | 0.206 | 0.000 | False |
| deepseek7b | `p765_0033_commonsense_question_animal:dog:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 2.766 | 0.906 | -0.192 | -0.133 | 0.458 | 0.000 | False |
| deepseek7b | `p765_0101_commonsense_question_abstract:time:grows_on_tree` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 1.531 | -0.344 | -0.283 | -0.212 | 0.635 | 0.000 | False |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 1.953 | -0.094 | -0.629 | -0.551 | 0.574 | 0.000 | False |
| deepseek7b | `p765_0033_commonsense_question_animal:dog:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 1.750 | 0.281 | -0.634 | -0.599 | 0.467 | 0.000 | False |
| deepseek7b | `p765_0101_commonsense_question_abstract:time:grows_on_tree` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 1.688 | 0.078 | -0.856 | -0.827 | 0.624 | 0.000 | False |
| deepseek7b | `p765_0052_commonsense_statement_plant:wheat:edible` | `attn:L26+attn:L27+mlp:L27+attn:L25` | 5.875 | 1.438 | -1.022 | -0.854 | 0.156 | 0.000 | False |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 2.922 | 0.938 | -1.157 | -1.100 | 0.501 | 0.000 | False |
| deepseek7b | `p765_0103_commonsense_question_abstract:justice:category` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 1.965 | 3.301 | -1.417 | -1.531 | 0.249 | 0.000 | False |
