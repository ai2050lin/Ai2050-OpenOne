# Phase 767 Commonsense Failure-Type Top-k Audit (smoke)

- Status: `complete`
- Test: Phase 765 commonsense prompts, logits-only top-k audit.
- Quantization: `off`; dtype: `bfloat16`.

## Overall Reliability

| model | cases | top1 | in top-k | target rank | allowed rank | margin | clean n | rank<=2 n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 12 | 0.917 | 1.000 | 1.417 | 1.083 | 2.458 | 11 | 11 |
| glm4 | 12 | 0.500 | 1.000 | 1.667 | 1.333 | 0.115 | 6 | 11 |
| deepseek7b | 12 | 0.250 | 1.000 | 6.083 | 1.833 | -2.073 | 3 | 3 |

## Failure-Type Counts

| model | failure type | n | top1 | in top-k | rank | allowed rank | margin |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `known_contrast_competition` | 1 | 0.000 | 1.000 | 6.000 | 2.000 | -6.500 |
| qwen3 | `success_top1` | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 3.273 |
| glm4 | `known_contrast_competition` | 1 | 0.000 | 1.000 | 4.000 | 2.000 | -3.375 |
| glm4 | `readout_threshold_miss` | 5 | 0.000 | 1.000 | 2.000 | 1.600 | -0.688 |
| glm4 | `success_top1` | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.365 |
| deepseek7b | `allowed_value_candidate_competition` | 2 | 0.000 | 1.000 | 9.000 | 2.000 | -3.031 |
| deepseek7b | `known_contrast_competition` | 7 | 0.000 | 1.000 | 7.429 | 2.143 | -3.071 |
| deepseek7b | `success_top1` | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.896 |

## Relation By Failure Type

| model | relation | failure type | n | top1 | in top-k | rank | allowed rank |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `category` | `success_top1` | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen3 | `edible` | `success_top1` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen3 | `grows_on_tree` | `known_contrast_competition` | 1 | 0.000 | 1.000 | 6.000 | 2.000 |
| qwen3 | `grows_on_tree` | `success_top1` | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| glm4 | `category` | `success_top1` | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| glm4 | `edible` | `readout_threshold_miss` | 2 | 0.000 | 1.000 | 2.000 | 2.000 |
| glm4 | `grows_on_tree` | `known_contrast_competition` | 1 | 0.000 | 1.000 | 4.000 | 2.000 |
| glm4 | `grows_on_tree` | `readout_threshold_miss` | 3 | 0.000 | 1.000 | 2.000 | 1.333 |
| glm4 | `grows_on_tree` | `success_top1` | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| deepseek7b | `category` | `known_contrast_competition` | 2 | 0.000 | 1.000 | 5.000 | 2.500 |
| deepseek7b | `category` | `success_top1` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| deepseek7b | `edible` | `known_contrast_competition` | 2 | 0.000 | 1.000 | 11.000 | 2.000 |
| deepseek7b | `grows_on_tree` | `allowed_value_candidate_competition` | 2 | 0.000 | 1.000 | 9.000 | 2.000 |
| deepseek7b | `grows_on_tree` | `known_contrast_competition` | 3 | 0.000 | 1.000 | 6.667 | 2.000 |

## Strict Interpretation

- `success_top1` is the clean subset proxy for prediction-sufficient state.
- `readout_threshold_miss` means the target was rank 2: close to closure, but not closed.
- `allowed_value_candidate_competition` means the allowed value set favored another candidate.
- `knowledge_or_state_formation_miss` means the target did not appear in top-k and cannot be used as a reliable mechanism sample.
- `format_protocol_miss` can be identified only when the top token is visibly format-like; broader protocol failures need generation traces.
