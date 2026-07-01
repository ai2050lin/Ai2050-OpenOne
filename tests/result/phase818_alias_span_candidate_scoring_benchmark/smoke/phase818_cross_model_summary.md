# Phase 818 Alias Span Candidate Scoring Benchmark (smoke)

- Boundary: target answer is evaluated as an alias class, while near-miss, wrong, and generic spans remain explicit competitors.

## Model Summary

| model | rows | exact score | alias score | exact rollout | alias rollout | exact full | alias full | near cleared | wrong cleared | generic cleared | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"alias_score_and_rollout_closed": 4}` |
| glm4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"alias_score_and_rollout_closed": 4}` |
| deepseek7b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"alias_score_and_rollout_closed": 4}` |

## Prompt Variant Summary

| model | prompt | n | alias score | alias rollout | alias full | generation classes | labels |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | 4 | 4 | 4 | 4 | `{"target_alias": 4}` | `{"alias_score_and_rollout_closed": 4}` |
| glm4 | exact_choices | 4 | 4 | 4 | 4 | `{"target_alias": 4}` | `{"alias_score_and_rollout_closed": 4}` |
| deepseek7b | exact_choices | 4 | 4 | 4 | 4 | `{"target_alias": 4}` | `{"alias_score_and_rollout_closed": 4}` |

## First Failure Rows

| model | prompt | case | target | generated | gen class | best alias | best non-alias | margin | label |
|---|---|---|---|---|---|---|---|---:|---|
