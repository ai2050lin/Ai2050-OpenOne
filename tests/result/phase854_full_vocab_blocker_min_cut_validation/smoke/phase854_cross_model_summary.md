# Phase 854 Full-Vocabulary Blocker Field and Min-Cut Validation (smoke)

- Source: Phase 853 strong / control interaction rows.
- Boundary: first-step full-vocabulary blocker audit; no rollout closure claim.

## Cross-Model Summary

| model | source rows | full rows | class closure | strict closure | improved blockers | worsened blockers | min-cut necessary | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | `{"weak_or_no_min_cut_effect": 8}` |
| glm4 | 2 | 2 | 1 | 0 | 1 | 0 | 2 | `{"necessary_blocker_reducer": 2, "weak_or_no_min_cut_effect": 2}` |
| deepseek7b | 4 | 4 | 3 | 0 | 2 | 1 | 1 | `{"single_sufficient_partial_reducer": 2, "weak_or_no_min_cut_effect": 5, "necessary_blocker_reducer": 1}` |

## Condition Means

| model | condition | n | answer class closure | strict closure | mean class blockers | mean class rank | mean class-object logit |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 4 | 4 | 0 | 0.0000 | 1.0000 | 5.5938 |
| qwen3 | `full_combo` | 4 | 4 | 0 | 0.0000 | 1.0000 | 3.8750 |
| qwen3 | `candidate_only` | 8 | 5 | 0 | 0.3750 | 1.3750 | 4.3125 |
| qwen3 | `without_candidate` | 8 | 7 | 0 | 0.1250 | 1.1250 | 5.7344 |
| glm4 | `original` | 2 | 1 | 0 | 6.0000 | 7.0000 | 0.6562 |
| glm4 | `full_combo` | 2 | 1 | 0 | 1.5000 | 2.5000 | 1.9688 |
| glm4 | `candidate_only` | 4 | 2 | 0 | 3.7500 | 4.7500 | 1.2422 |
| glm4 | `without_candidate` | 4 | 2 | 0 | 3.2500 | 4.2500 | 1.4219 |
| deepseek7b | `original` | 4 | 1 | 0 | 0.7500 | 1.7500 | 0.7969 |
| deepseek7b | `full_combo` | 4 | 3 | 0 | 1.2500 | 2.2500 | 0.1406 |
| deepseek7b | `candidate_only` | 8 | 5 | 0 | 0.3750 | 1.3750 | 0.7500 |
| deepseek7b | `without_candidate` | 8 | 5 | 0 | 1.3750 | 2.3750 | 0.1250 |
