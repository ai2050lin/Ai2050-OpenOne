# Phase 854 Full-Vocabulary Blocker Field and Min-Cut Validation (confirm)

- Source: Phase 853 strong / control interaction rows.
- Boundary: first-step full-vocabulary blocker audit; no rollout closure claim.

## Cross-Model Summary

| model | source rows | full rows | class closure | strict closure | improved blockers | worsened blockers | min-cut necessary | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 60 | 60 | 42 | 0 | 11 | 18 | 10 | `{"weak_or_no_min_cut_effect": 123, "single_sufficient_partial_reducer": 2, "necessary_blocker_reducer": 10, "candidate_harmful_or_antagonistic": 22}` |
| glm4 | 13 | 13 | 12 | 0 | 1 | 0 | 2 | `{"necessary_blocker_reducer": 2, "single_sufficient_partial_reducer": 1, "weak_or_no_min_cut_effect": 28}` |
| deepseek7b | 20 | 20 | 17 | 0 | 5 | 3 | 2 | `{"weak_or_no_min_cut_effect": 38, "single_sufficient_partial_reducer": 7, "necessary_blocker_reducer": 2, "candidate_harmful_or_antagonistic": 3}` |

## Condition Means

| model | condition | n | answer class closure | strict closure | mean class blockers | mean class rank | mean class-object logit |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 60 | 45 | 0 | 0.2500 | 1.2500 | 5.7875 |
| qwen3 | `full_combo` | 60 | 42 | 0 | 0.8333 | 1.8333 | 3.6031 |
| qwen3 | `candidate_only` | 157 | 105 | 0 | 0.6242 | 1.6242 | 4.9642 |
| qwen3 | `without_candidate` | 157 | 102 | 0 | 0.7707 | 1.7707 | 4.4689 |
| glm4 | `original` | 13 | 12 | 0 | 0.9231 | 1.9231 | 1.7933 |
| glm4 | `full_combo` | 13 | 12 | 0 | 0.2308 | 1.2308 | 2.3942 |
| glm4 | `candidate_only` | 31 | 28 | 0 | 0.7419 | 1.7419 | 1.9889 |
| glm4 | `without_candidate` | 31 | 28 | 0 | 0.5161 | 1.5161 | 2.0645 |
| deepseek7b | `original` | 20 | 12 | 0 | 0.4000 | 1.4000 | 0.4250 |
| deepseek7b | `full_combo` | 20 | 17 | 0 | 0.7500 | 1.7500 | 0.0281 |
| deepseek7b | `candidate_only` | 50 | 38 | 0 | 0.4800 | 1.4800 | 0.2750 |
| deepseek7b | `without_candidate` | 50 | 43 | 0 | 0.5400 | 1.5400 | 0.1675 |
