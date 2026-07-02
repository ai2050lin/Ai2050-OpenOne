# Phase 854 Full-Vocabulary Blocker Field and Min-Cut Validation (main)

- Source: Phase 853 strong / control interaction rows.
- Boundary: first-step full-vocabulary blocker audit; no rollout closure claim.

## Cross-Model Summary

| model | source rows | full rows | class closure | strict closure | improved blockers | worsened blockers | min-cut necessary | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 24 | 24 | 19 | 0 | 4 | 5 | 3 | `{"weak_or_no_min_cut_effect": 38, "single_sufficient_partial_reducer": 2, "necessary_blocker_reducer": 3, "candidate_harmful_or_antagonistic": 5}` |
| glm4 | 7 | 7 | 6 | 0 | 1 | 0 | 2 | `{"necessary_blocker_reducer": 2, "weak_or_no_min_cut_effect": 12}` |
| deepseek7b | 14 | 14 | 11 | 0 | 5 | 3 | 2 | `{"single_sufficient_partial_reducer": 6, "weak_or_no_min_cut_effect": 19, "necessary_blocker_reducer": 2, "candidate_harmful_or_antagonistic": 1}` |

## Condition Means

| model | condition | n | answer class closure | strict closure | mean class blockers | mean class rank | mean class-object logit |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 24 | 20 | 0 | 0.1667 | 1.1667 | 5.3125 |
| qwen3 | `full_combo` | 24 | 19 | 0 | 0.6667 | 1.6667 | 2.2552 |
| qwen3 | `candidate_only` | 48 | 29 | 0 | 0.7708 | 1.7708 | 3.7240 |
| qwen3 | `without_candidate` | 48 | 32 | 0 | 0.6667 | 1.6667 | 4.3359 |
| glm4 | `original` | 7 | 6 | 0 | 1.7143 | 2.7143 | 1.6161 |
| glm4 | `full_combo` | 7 | 6 | 0 | 0.4286 | 1.4286 | 2.5268 |
| glm4 | `candidate_only` | 14 | 12 | 0 | 1.0714 | 2.0714 | 1.9844 |
| glm4 | `without_candidate` | 14 | 12 | 0 | 0.9286 | 1.9286 | 2.1429 |
| deepseek7b | `original` | 14 | 6 | 0 | 0.5714 | 1.5714 | 0.6071 |
| deepseek7b | `full_combo` | 14 | 11 | 0 | 1.0714 | 2.0714 | 0.0402 |
| deepseek7b | `candidate_only` | 28 | 20 | 0 | 0.4286 | 1.4286 | 0.4911 |
| deepseek7b | `without_candidate` | 28 | 21 | 0 | 0.9643 | 1.9643 | 0.1339 |
