# Phase 801 Target-Neutral Suppressor Causal Test (smoke)

- Status: `complete`
- Boundary: hidden-state route deltas are decomposed into raw, target-neutral, and target-only components.
- This tests whether blocker suppression survives removal of the direct target-readout direction.

## By Projection

| model | projection | rows | cases | target gain | blocker suppression | resolved | new rate | neutral score | token gain | labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `target_neutral` | 1 | 1 | -0.062 | 1.242 | 0.402 | 0.374 | 0.294 | 0.000 | `{"neutral_suppression_with_residual_target_gain": 1}` |
| qwen3 | `raw` | 1 | 1 | 3.750 | 1.117 | 0.888 | 0.048 | 0.199 | 0.000 | `{"raw_suppressor_like": 1}` |
| qwen3 | `target_only` | 1 | 1 | 3.938 | -0.143 | 0.872 | 0.000 | 0.000 | 0.000 | `{"target_only_threshold_shift": 1}` |
| glm4 | `target_neutral` | 1 | 1 | 0.344 | 0.790 | 0.481 | 0.242 | 0.214 | 0.000 | `{"neutral_suppression_with_residual_target_gain": 1}` |
| glm4 | `raw` | 1 | 1 | 0.000 | 0.784 | 0.390 | 0.301 | 0.214 | 0.000 | `{"weak_or_mixed": 1}` |
| glm4 | `target_only` | 1 | 1 | -0.375 | 0.005 | 0.000 | 0.224 | 0.000 | 0.000 | `{"weak_or_mixed": 1}` |
| deepseek7b | `target_neutral` | 1 | 1 | 0.938 | -1.100 | 0.114 | 0.501 | 0.000 | 0.000 | `{"weak_or_mixed": 1}` |
| deepseek7b | `target_only` | 1 | 1 | 2.203 | -0.064 | 0.724 | 0.000 | 0.000 | 0.000 | `{"target_only_threshold_shift": 1}` |
| deepseek7b | `raw` | 1 | 1 | 2.922 | -1.157 | 0.624 | 0.101 | 0.000 | 0.000 | `{"weak_or_mixed": 1}` |

## Top Target-Neutral Triplets

| model | case | route | raw target | neutral target | raw suppress | neutral suppress | neutral new | neutral score | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `p765_0041_commonsense_question_plant:oak:grows_on_tree` | `attn:L35+mlp:L35+mlp:L34+mlp:L33` | 3.750 | -0.062 | 1.117 | 1.242 | 0.374 | 0.287 | False |
| glm4 | `p765_0051_commonsense_question_plant:wheat:edible` | `mlp:L38+mlp:L39+mlp:L34+mlp:L27` | 0.000 | 0.344 | 0.784 | 0.790 | 0.242 | 0.156 | False |
| deepseek7b | `p765_0075_commonsense_question_tool:hammer:edible` | `mlp:L27+mlp:L26+mlp:L24+attn:L19` | 2.922 | 0.938 | -1.157 | -1.100 | 0.501 | 0.000 | False |
