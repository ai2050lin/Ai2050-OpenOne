# Phase 817 Alias Aware Answer Span Audit (smoke)

- Source: Phase 816 saved rollout rows; no new model forward pass.
- Boundary: tests whether exact phrase rollout undercounts semantically acceptable multi-token answers.

## Model Summary

| model | rows | exact rollout | alias rollout | exact full | alias full | rollout gain | full gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 4 | 4 | 4 | 4 | 4 | 0 | 0 | `{"exact_span_and_rollout_closed": 4}` |
| glm4 | 4 | 4 | 4 | 4 | 4 | 0 | 0 | `{"exact_span_and_rollout_closed": 4}` |
| deepseek7b | 4 | 4 | 4 | 4 | 4 | 0 | 0 | `{"exact_span_and_rollout_closed": 4}` |

## Prompt Variant Summary

| model | prompt | n | exact rollout | alias rollout | exact full | alias full | labels |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | exact_choices | 4 | 4 | 4 | 4 | 4 | `{"exact_span_and_rollout_closed": 4}` |
| glm4 | exact_choices | 4 | 4 | 4 | 4 | 4 | `{"exact_span_and_rollout_closed": 4}` |
| deepseek7b | exact_choices | 4 | 4 | 4 | 4 | 4 | `{"exact_span_and_rollout_closed": 4}` |

## Rescued Rows

| model | prompt | case | target | generated | alias | label |
|---|---|---|---|---|---|---|
