# Phase 846 Geometry Boundary Equation Fitting (smoke)

- Source: Phase 845 gear interaction rows.
- Method: compare `additive_only` against `interaction_equation` on in-sample / object-holdout / prompt-holdout splits.
- Boundary: this is route-boundary prediction, not token closure.

## Model Summary

| model | source rows | predictions | split | predictor | n | MAE delta | RMSE delta | target acc | target F1 | gain F1 | mean interaction MAE gain |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 21 | 40 | `in_sample` | `additive_only` | 20 | 0.0906 | 0.1981 | 0.9000 | 0.5000 | 0.5000 | 0.0906 |
| qwen3 | 21 | 40 | `in_sample` | `interaction_equation` | 20 | 0.0000 | 0.0000 | 0.9000 | 0.5000 | 0.5000 | 0.0906 |
| glm4 | 21 | 40 | `in_sample` | `additive_only` | 20 | 0.0148 | 0.0229 | 1.0000 | 1.0000 | 0.0000 | 0.0148 |
| glm4 | 21 | 40 | `in_sample` | `interaction_equation` | 20 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0148 |
| deepseek7b | 21 | 40 | `in_sample` | `additive_only` | 20 | 0.0422 | 0.0629 | 0.8000 | 0.3333 | 0.3333 | 0.0422 |
| deepseek7b | 21 | 40 | `in_sample` | `interaction_equation` | 20 | 0.0000 | 0.0000 | 0.8500 | 0.5714 | 0.5714 | 0.0422 |

## Source Shapes

| model | objects | prompts | combo counts |
|---|---|---|---|
| qwen3 | `triangle` | `natural_question` | `{"original": 1, "single": 8, "pair": 12}` |
| glm4 | `triangle` | `natural_question` | `{"original": 1, "single": 8, "pair": 12}` |
| deepseek7b | `triangle` | `natural_question` | `{"original": 1, "single": 8, "pair": 12}` |
