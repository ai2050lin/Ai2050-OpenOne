# Phase 846 Geometry Boundary Equation Fitting (main)

- Source: Phase 845 gear interaction rows.
- Method: compare `additive_only` against `interaction_equation` on in-sample / object-holdout / prompt-holdout splits.
- Boundary: this is route-boundary prediction, not token closure.

## Model Summary

| model | source rows | predictions | split | predictor | n | MAE delta | RMSE delta | target acc | target F1 | gain F1 | mean interaction MAE gain |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 408 | 2400 | `in_sample` | `additive_only` | 400 | 0.9871 | 1.4740 | 0.4150 | 0.5806 | 0.3464 | 0.0319 |
| qwen3 | 408 | 2400 | `in_sample` | `interaction_equation` | 400 | 0.9553 | 1.4214 | 0.4150 | 0.5761 | 0.3446 | 0.0319 |
| qwen3 | 408 | 2400 | `object_holdout` | `additive_only` | 400 | 1.2455 | 1.8322 | 0.4050 | 0.4803 | 0.3896 | 0.0106 |
| qwen3 | 408 | 2400 | `object_holdout` | `interaction_equation` | 400 | 1.2349 | 1.8386 | 0.4100 | 0.4802 | 0.3882 | 0.0106 |
| qwen3 | 408 | 2400 | `prompt_holdout` | `additive_only` | 400 | 1.0280 | 1.5053 | 0.2675 | 0.2906 | 0.1119 | 0.0211 |
| qwen3 | 408 | 2400 | `prompt_holdout` | `interaction_equation` | 400 | 1.0070 | 1.4609 | 0.2675 | 0.2906 | 0.1119 | 0.0211 |
| glm4 | 408 | 2400 | `in_sample` | `additive_only` | 400 | 0.1143 | 0.2079 | 0.8550 | 0.9218 | 0.5714 | -0.0007 |
| glm4 | 408 | 2400 | `in_sample` | `interaction_equation` | 400 | 0.1150 | 0.2071 | 0.8550 | 0.9218 | 0.5714 | -0.0007 |
| glm4 | 408 | 2400 | `object_holdout` | `additive_only` | 400 | 0.1492 | 0.2596 | 0.7450 | 0.8539 | 0.5714 | 0.0013 |
| glm4 | 408 | 2400 | `object_holdout` | `interaction_equation` | 400 | 0.1479 | 0.2580 | 0.7450 | 0.8539 | 0.5714 | 0.0013 |
| glm4 | 408 | 2400 | `prompt_holdout` | `additive_only` | 400 | 0.1327 | 0.2397 | 0.8550 | 0.9218 | 0.5714 | -0.0018 |
| glm4 | 408 | 2400 | `prompt_holdout` | `interaction_equation` | 400 | 0.1346 | 0.2407 | 0.8550 | 0.9218 | 0.5714 | -0.0018 |
| deepseek7b | 408 | 2400 | `in_sample` | `additive_only` | 400 | 0.3158 | 0.4514 | 0.7575 | 0.8397 | 0.5340 | 0.0063 |
| deepseek7b | 408 | 2400 | `in_sample` | `interaction_equation` | 400 | 0.3095 | 0.4502 | 0.7600 | 0.8431 | 0.5540 | 0.0063 |
| deepseek7b | 408 | 2400 | `object_holdout` | `additive_only` | 400 | 0.4165 | 0.5916 | 0.5325 | 0.6690 | 0.3611 | 0.0093 |
| deepseek7b | 408 | 2400 | `object_holdout` | `interaction_equation` | 400 | 0.4071 | 0.5885 | 0.5325 | 0.6690 | 0.3611 | 0.0093 |
| deepseek7b | 408 | 2400 | `prompt_holdout` | `additive_only` | 400 | 0.3234 | 0.4638 | 0.5425 | 0.7024 | 0.4962 | 0.0029 |
| deepseek7b | 408 | 2400 | `prompt_holdout` | `interaction_equation` | 400 | 0.3205 | 0.4689 | 0.5525 | 0.7108 | 0.4962 | 0.0029 |

## Source Shapes

| model | objects | prompts | combo counts |
|---|---|---|---|
| qwen3 | `circle, rectangle, square, triangle` | `natural_category, natural_question` | `{"original": 8, "single": 96, "pair": 240, "triplet": 64}` |
| glm4 | `circle, rectangle, square, triangle` | `natural_category, natural_question` | `{"original": 8, "single": 96, "pair": 240, "triplet": 64}` |
| deepseek7b | `circle, rectangle, square, triangle` | `natural_category, natural_question` | `{"original": 8, "single": 96, "pair": 240, "triplet": 64}` |
