# Phase 846 Geometry Boundary Equation Fitting (confirm)

- Source: Phase 845 gear interaction rows.
- Method: compare `additive_only` against `interaction_equation` on in-sample / object-holdout / prompt-holdout splits.
- Boundary: this is route-boundary prediction, not token closure.

## Model Summary

| model | source rows | predictions | split | predictor | n | MAE delta | RMSE delta | target acc | target F1 | gain F1 | mean interaction MAE gain |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 765 | 4500 | `in_sample` | `additive_only` | 750 | 0.8237 | 1.2523 | 0.5867 | 0.5064 | 0.2837 | -0.0032 |
| qwen3 | 765 | 4500 | `in_sample` | `interaction_equation` | 750 | 0.8269 | 1.2376 | 0.5853 | 0.5056 | 0.2831 | -0.0032 |
| qwen3 | 765 | 4500 | `object_holdout` | `additive_only` | 750 | 0.9321 | 1.3976 | 0.5293 | 0.3225 | 0.1833 | -0.0153 |
| qwen3 | 765 | 4500 | `object_holdout` | `interaction_equation` | 750 | 0.9473 | 1.4055 | 0.5293 | 0.3250 | 0.1877 | -0.0153 |
| qwen3 | 765 | 4500 | `prompt_holdout` | `additive_only` | 750 | 0.9134 | 1.3456 | 0.3973 | 0.1929 | 0.0291 | -0.0075 |
| qwen3 | 765 | 4500 | `prompt_holdout` | `interaction_equation` | 750 | 0.9209 | 1.3565 | 0.4027 | 0.1942 | 0.0294 | -0.0075 |
| glm4 | 765 | 4500 | `in_sample` | `additive_only` | 750 | 0.1369 | 0.2122 | 0.7507 | 0.7792 | 0.2920 | -0.0008 |
| glm4 | 765 | 4500 | `in_sample` | `interaction_equation` | 750 | 0.1377 | 0.2117 | 0.7480 | 0.7774 | 0.2878 | -0.0008 |
| glm4 | 765 | 4500 | `object_holdout` | `additive_only` | 750 | 0.1623 | 0.2475 | 0.4800 | 0.5608 | 0.1544 | -0.0001 |
| glm4 | 765 | 4500 | `object_holdout` | `interaction_equation` | 750 | 0.1624 | 0.2474 | 0.4813 | 0.5614 | 0.1550 | -0.0001 |
| glm4 | 765 | 4500 | `prompt_holdout` | `additive_only` | 750 | 0.1554 | 0.2374 | 0.4067 | 0.5340 | 0.1250 | 0.0016 |
| glm4 | 765 | 4500 | `prompt_holdout` | `interaction_equation` | 750 | 0.1537 | 0.2352 | 0.4067 | 0.5350 | 0.1246 | 0.0016 |
| deepseek7b | 765 | 4500 | `in_sample` | `additive_only` | 750 | 0.2630 | 0.3904 | 0.7653 | 0.7419 | 0.3750 | 0.0027 |
| deepseek7b | 765 | 4500 | `in_sample` | `interaction_equation` | 750 | 0.2603 | 0.3899 | 0.7667 | 0.7438 | 0.3750 | 0.0027 |
| deepseek7b | 765 | 4500 | `object_holdout` | `additive_only` | 750 | 0.3201 | 0.4761 | 0.6627 | 0.6761 | 0.3295 | 0.0022 |
| deepseek7b | 765 | 4500 | `object_holdout` | `interaction_equation` | 750 | 0.3179 | 0.4757 | 0.6627 | 0.6761 | 0.3295 | 0.0022 |
| deepseek7b | 765 | 4500 | `prompt_holdout` | `additive_only` | 750 | 0.2696 | 0.3971 | 0.6493 | 0.6700 | 0.2299 | 0.0027 |
| deepseek7b | 765 | 4500 | `prompt_holdout` | `interaction_equation` | 750 | 0.2669 | 0.3974 | 0.6493 | 0.6700 | 0.2299 | 0.0027 |

## Source Shapes

| model | objects | prompts | combo counts |
|---|---|---|---|
| qwen3 | `circle, polygon, rectangle, square, triangle` | `natural_category, natural_question, object_only` | `{"original": 15, "single": 180, "pair": 450, "triplet": 120}` |
| glm4 | `circle, polygon, rectangle, square, triangle` | `natural_category, natural_question, object_only` | `{"original": 15, "single": 180, "pair": 450, "triplet": 120}` |
| deepseek7b | `circle, polygon, rectangle, square, triangle` | `natural_category, natural_question, object_only` | `{"original": 15, "single": 180, "pair": 450, "triplet": 120}` |
