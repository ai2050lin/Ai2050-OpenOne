# Phase 767 Commonsense Failure-Type Top-k Audit (main)

- Status: `complete`
- Test: Phase 765 commonsense prompts, logits-only top-k audit.
- Quantization: `off`; dtype: `bfloat16`.

## Overall Reliability

| model | cases | semantic top1 | exact top1 | in top-k | semantic rank | exact rank | allowed rank | margin | clean n | rank<=2 n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 108 | 0.824 | 0.824 | 1.000 | 1.315 | 1.315 | 1.185 | 2.936 | 89 | 102 |
| glm4 | 108 | 0.750 | 0.611 | 1.000 | 1.380 | 1.630 | 1.287 | 1.237 | 81 | 99 |
| deepseek7b | 108 | 0.352 | 0.176 | 1.000 | 3.806 | 6.407 | 1.667 | -0.764 | 38 | 55 |

## Failure-Type Counts

| model | failure type | n | semantic top1 | exact top1 | in top-k | semantic rank | exact rank | allowed rank | margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `allowed_value_candidate_competition` | 4 | 0.000 | 0.000 | 1.000 | 4.500 | 4.500 | 2.250 | -2.719 |
| qwen3 | `known_contrast_competition` | 2 | 0.000 | 0.000 | 1.000 | 4.500 | 4.500 | 2.000 | -7.000 |
| qwen3 | `readout_threshold_miss` | 13 | 0.000 | 0.000 | 1.000 | 2.000 | 2.000 | 2.000 | -2.385 |
| qwen3 | `success_top1` | 89 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.191 |
| glm4 | `candidate_competition_other` | 1 | 0.000 | 0.000 | 1.000 | 3.000 | 3.000 | 1.000 | -0.250 |
| glm4 | `known_contrast_competition` | 8 | 0.000 | 0.000 | 1.000 | 3.625 | 4.000 | 2.750 | -1.820 |
| glm4 | `readout_threshold_miss` | 18 | 0.000 | 0.000 | 1.000 | 2.000 | 2.222 | 1.944 | -0.545 |
| glm4 | `success_top1` | 81 | 1.000 | 0.815 | 1.000 | 1.000 | 1.247 | 1.000 | 1.953 |
| deepseek7b | `allowed_value_candidate_competition` | 7 | 0.000 | 0.000 | 1.000 | 11.714 | 13.714 | 2.286 | -2.705 |
| deepseek7b | `candidate_competition_other` | 1 | 0.000 | 0.000 | 1.000 | 3.000 | 3.000 | 1.000 | -0.438 |
| deepseek7b | `format_protocol_miss` | 4 | 0.000 | 0.000 | 1.000 | 6.000 | 8.750 | 1.500 | -1.609 |
| deepseek7b | `known_contrast_competition` | 41 | 0.000 | 0.000 | 1.000 | 5.610 | 8.244 | 2.146 | -2.599 |
| deepseek7b | `readout_threshold_miss` | 17 | 0.000 | 0.000 | 1.000 | 2.000 | 6.118 | 1.824 | -1.335 |
| deepseek7b | `success_top1` | 38 | 1.000 | 0.500 | 1.000 | 1.000 | 3.053 | 1.000 | 1.910 |

## Relation By Failure Type

| model | relation | failure type | n | top1 | in top-k | rank | allowed rank |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `category` | `allowed_value_candidate_competition` | 4 | 0.000 | 1.000 | 4.500 | 2.250 |
| qwen3 | `category` | `readout_threshold_miss` | 1 | 0.000 | 1.000 | 2.000 | 2.000 |
| qwen3 | `category` | `success_top1` | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen3 | `edible` | `readout_threshold_miss` | 10 | 0.000 | 1.000 | 2.000 | 2.000 |
| qwen3 | `edible` | `success_top1` | 26 | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen3 | `grows_on_tree` | `known_contrast_competition` | 2 | 0.000 | 1.000 | 4.500 | 2.000 |
| qwen3 | `grows_on_tree` | `readout_threshold_miss` | 2 | 0.000 | 1.000 | 2.000 | 2.000 |
| qwen3 | `grows_on_tree` | `success_top1` | 32 | 1.000 | 1.000 | 1.000 | 1.000 |
| glm4 | `category` | `candidate_competition_other` | 1 | 0.000 | 1.000 | 3.000 | 1.000 |
| glm4 | `category` | `known_contrast_competition` | 2 | 0.000 | 1.000 | 5.500 | 5.000 |
| glm4 | `category` | `readout_threshold_miss` | 3 | 0.000 | 1.000 | 2.000 | 1.667 |
| glm4 | `category` | `success_top1` | 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| glm4 | `edible` | `known_contrast_competition` | 3 | 0.000 | 1.000 | 3.000 | 2.000 |
| glm4 | `edible` | `readout_threshold_miss` | 11 | 0.000 | 1.000 | 2.000 | 2.000 |
| glm4 | `edible` | `success_top1` | 22 | 1.000 | 1.000 | 1.000 | 1.000 |
| glm4 | `grows_on_tree` | `known_contrast_competition` | 3 | 0.000 | 1.000 | 3.000 | 2.000 |
| glm4 | `grows_on_tree` | `readout_threshold_miss` | 4 | 0.000 | 1.000 | 2.000 | 2.000 |
| glm4 | `grows_on_tree` | `success_top1` | 29 | 1.000 | 1.000 | 1.000 | 1.000 |
| deepseek7b | `category` | `allowed_value_candidate_competition` | 7 | 0.000 | 1.000 | 11.714 | 2.286 |
| deepseek7b | `category` | `candidate_competition_other` | 1 | 0.000 | 1.000 | 3.000 | 1.000 |
| deepseek7b | `category` | `format_protocol_miss` | 4 | 0.000 | 1.000 | 6.000 | 1.500 |
| deepseek7b | `category` | `known_contrast_competition` | 7 | 0.000 | 1.000 | 9.857 | 2.857 |
| deepseek7b | `category` | `readout_threshold_miss` | 4 | 0.000 | 1.000 | 2.000 | 1.250 |
| deepseek7b | `category` | `success_top1` | 13 | 1.000 | 1.000 | 1.000 | 1.000 |
| deepseek7b | `edible` | `known_contrast_competition` | 18 | 0.000 | 1.000 | 6.000 | 2.000 |
| deepseek7b | `edible` | `readout_threshold_miss` | 3 | 0.000 | 1.000 | 2.000 | 2.000 |
| deepseek7b | `edible` | `success_top1` | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| deepseek7b | `grows_on_tree` | `known_contrast_competition` | 16 | 0.000 | 1.000 | 3.312 | 2.000 |
| deepseek7b | `grows_on_tree` | `readout_threshold_miss` | 10 | 0.000 | 1.000 | 2.000 | 2.000 |
| deepseek7b | `grows_on_tree` | `success_top1` | 10 | 1.000 | 1.000 | 1.000 | 1.000 |

## Strict Interpretation

- `semantic top1` merges simple lexical aliases such as `yes/Yes/YES`; `exact top1` is the stricter first-token match.
- `success_top1` is the semantic clean subset proxy for prediction-sufficient state.
- `readout_threshold_miss` means the target was rank 2: close to closure, but not closed.
- `allowed_value_candidate_competition` means the allowed value set favored another candidate.
- `knowledge_or_state_formation_miss` means the target did not appear in top-k and cannot be used as a reliable mechanism sample.
- `format_protocol_miss` can be identified only when the top token is visibly format-like; broader protocol failures need generation traces.
