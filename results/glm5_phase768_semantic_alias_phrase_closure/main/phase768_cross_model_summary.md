# Phase 768 Semantic-Alias Phrase Closure (main)

- Status: `complete`
- Test: phrase likelihood over allowed values plus short greedy generation.
- Input subset labels: Phase 767 semantic/exact closure rows.

## By Subset

| model | subset | n | semantic top1 | exact top1 | phrase top1 | generation match | phrase rank | phrase margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `all` | 108 | 0.824 | 0.824 | 0.806 | 0.824 | 1.204 | 4.193 |
| qwen3 | `exact_clean` | 89 | 1.000 | 1.000 | 0.978 | 1.000 | 1.022 | 5.709 |
| qwen3 | `semantic_clean` | 89 | 1.000 | 1.000 | 0.978 | 1.000 | 1.022 | 5.709 |
| qwen3 | `semantic_only` | 0 | null | null | null | null | null | null |
| qwen3 | `semantic_fail` | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 2.053 | -2.908 |
| qwen3 | `rank_le2` | 102 | 0.873 | 0.873 | 0.853 | 0.873 | 1.147 | 4.683 |
| glm4 | `all` | 108 | 0.750 | 0.611 | 0.750 | 0.750 | 1.306 | 1.518 |
| glm4 | `exact_clean` | 66 | 1.000 | 1.000 | 0.970 | 1.000 | 1.030 | 2.510 |
| glm4 | `semantic_clean` | 81 | 1.000 | 0.815 | 0.975 | 1.000 | 1.025 | 2.307 |
| glm4 | `semantic_only` | 15 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.413 |
| glm4 | `semantic_fail` | 27 | 0.000 | 0.000 | 0.074 | 0.000 | 2.148 | -0.850 |
| glm4 | `rank_le2` | 99 | 0.818 | 0.667 | 0.808 | 0.818 | 1.192 | 1.800 |
| deepseek7b | `all` | 108 | 0.352 | 0.176 | 0.380 | 0.352 | 1.667 | 0.034 |
| deepseek7b | `exact_clean` | 19 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4.477 |
| deepseek7b | `semantic_clean` | 38 | 1.000 | 0.500 | 0.974 | 1.000 | 1.026 | 3.350 |
| deepseek7b | `semantic_only` | 19 | 1.000 | 0.000 | 0.947 | 1.000 | 1.053 | 2.224 |
| deepseek7b | `semantic_fail` | 70 | 0.000 | 0.000 | 0.057 | 0.000 | 2.014 | -1.767 |
| deepseek7b | `rank_le2` | 55 | 0.691 | 0.345 | 0.709 | 0.691 | 1.291 | 2.082 |

## Relation And Subset

| model | relation | subset | n | phrase top1 | generation match | phrase rank | phrase margin |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `category` | `semantic_clean` | 31 | 1.000 | 1.000 | 1.000 | 7.649 |
| qwen3 | `category` | `semantic_fail` | 5 | 0.000 | 0.000 | 2.200 | -2.300 |
| qwen3 | `edible` | `semantic_clean` | 26 | 0.962 | 1.000 | 1.038 | 4.293 |
| qwen3 | `edible` | `semantic_fail` | 10 | 0.000 | 0.000 | 2.000 | -2.862 |
| qwen3 | `grows_on_tree` | `semantic_clean` | 32 | 0.969 | 1.000 | 1.031 | 4.980 |
| qwen3 | `grows_on_tree` | `semantic_fail` | 4 | 0.000 | 0.000 | 2.000 | -3.781 |
| glm4 | `category` | `semantic_clean` | 30 | 0.967 | 1.000 | 1.033 | 3.665 |
| glm4 | `category` | `semantic_fail` | 6 | 0.333 | 0.000 | 2.667 | -0.719 |
| glm4 | `edible` | `semantic_clean` | 22 | 1.000 | 1.000 | 1.000 | 1.972 |
| glm4 | `edible` | `semantic_fail` | 14 | 0.000 | 0.000 | 2.000 | -0.670 |
| glm4 | `grows_on_tree` | `semantic_clean` | 29 | 0.966 | 1.000 | 1.034 | 1.157 |
| glm4 | `grows_on_tree` | `semantic_fail` | 7 | 0.000 | 0.000 | 2.000 | -1.321 |
| deepseek7b | `category` | `semantic_clean` | 13 | 1.000 | 1.000 | 1.000 | 3.909 |
| deepseek7b | `category` | `semantic_fail` | 23 | 0.174 | 0.000 | 2.043 | -1.136 |
| deepseek7b | `edible` | `semantic_clean` | 15 | 1.000 | 1.000 | 1.000 | 3.662 |
| deepseek7b | `edible` | `semantic_fail` | 21 | 0.000 | 0.000 | 2.000 | -2.545 |
| deepseek7b | `grows_on_tree` | `semantic_clean` | 10 | 0.900 | 1.000 | 1.100 | 2.156 |
| deepseek7b | `grows_on_tree` | `semantic_fail` | 26 | 0.000 | 0.000 | 2.000 | -1.697 |

## Strict Interpretation

- Phrase top1 tests whether the best full allowed-value phrase is the target value.
- Generation match is stricter: greedy continuation must start with the target semantic value.
- If semantic-clean cases lose phrase top1, first-token semantic closure is not sufficient for phrase closure.
- If semantic-only cases keep phrase top1, lexical capitalization is likely a surface realization issue.
