# Phase 768 Semantic-Alias Phrase Closure (smoke)

- Status: `complete`
- Test: phrase likelihood over allowed values plus short greedy generation.
- Input subset labels: Phase 767 semantic/exact closure rows.

## By Subset

| model | subset | n | semantic top1 | exact top1 | phrase top1 | generation match | phrase rank | phrase margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `all` | 12 | 0.917 | 0.917 | 0.833 | 0.000 | 1.167 | 3.927 |
| qwen3 | `exact_clean` | 11 | 1.000 | 1.000 | 0.909 | 0.000 | 1.091 | 4.875 |
| qwen3 | `semantic_clean` | 11 | 1.000 | 1.000 | 0.909 | 0.000 | 1.091 | 4.875 |
| qwen3 | `semantic_only` | 0 | null | null | null | null | null | null |
| qwen3 | `semantic_fail` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 2.000 | -6.500 |
| qwen3 | `rank_le2` | 11 | 1.000 | 1.000 | 0.909 | 0.000 | 1.091 | 4.875 |
| glm4 | `all` | 12 | 0.667 | 0.500 | 0.667 | 0.167 | 1.333 | 1.047 |
| glm4 | `exact_clean` | 6 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 2.677 |
| glm4 | `semantic_clean` | 8 | 1.000 | 0.750 | 1.000 | 0.250 | 1.000 | 2.297 |
| glm4 | `semantic_only` | 2 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.156 |
| glm4 | `semantic_fail` | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 2.000 | -1.453 |
| glm4 | `rank_le2` | 11 | 0.727 | 0.545 | 0.727 | 0.182 | 1.273 | 1.443 |
| deepseek7b | `all` | 12 | 0.333 | 0.250 | 0.333 | 0.083 | 1.750 | -0.411 |
| deepseek7b | `exact_clean` | 3 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 3.812 |
| deepseek7b | `semantic_clean` | 4 | 1.000 | 0.750 | 1.000 | 0.250 | 1.000 | 3.062 |
| deepseek7b | `semantic_only` | 1 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.812 |
| deepseek7b | `semantic_fail` | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 2.125 | -2.148 |
| deepseek7b | `rank_le2` | 5 | 0.800 | 0.600 | 0.800 | 0.200 | 1.200 | 2.412 |

## Relation And Subset

| model | relation | subset | n | phrase top1 | generation match | phrase rank | phrase margin |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `category` | `semantic_clean` | 5 | 1.000 | 0.000 | 1.000 | 7.400 |
| qwen3 | `edible` | `semantic_clean` | 2 | 0.500 | 0.000 | 1.500 | 0.750 |
| qwen3 | `grows_on_tree` | `semantic_clean` | 4 | 1.000 | 0.000 | 1.000 | 3.781 |
| qwen3 | `grows_on_tree` | `semantic_fail` | 1 | 0.000 | 0.000 | 2.000 | -6.500 |
| glm4 | `category` | `semantic_clean` | 5 | 1.000 | 0.000 | 1.000 | 3.163 |
| glm4 | `edible` | `semantic_fail` | 2 | 0.000 | 0.000 | 2.000 | -0.875 |
| glm4 | `grows_on_tree` | `semantic_clean` | 3 | 1.000 | 0.667 | 1.000 | 0.854 |
| glm4 | `grows_on_tree` | `semantic_fail` | 2 | 0.000 | 0.000 | 2.000 | -2.031 |
| deepseek7b | `category` | `semantic_clean` | 3 | 1.000 | 0.000 | 1.000 | 3.812 |
| deepseek7b | `category` | `semantic_fail` | 2 | 0.000 | 0.000 | 2.500 | -0.812 |
| deepseek7b | `edible` | `semantic_fail` | 2 | 0.000 | 0.000 | 2.000 | -3.125 |
| deepseek7b | `grows_on_tree` | `semantic_clean` | 1 | 1.000 | 1.000 | 1.000 | 0.812 |
| deepseek7b | `grows_on_tree` | `semantic_fail` | 4 | 0.000 | 0.000 | 2.000 | -2.328 |

## Strict Interpretation

- Phrase top1 tests whether the best full allowed-value phrase is the target value.
- Generation match is stricter: greedy continuation must start with the target semantic value.
- If semantic-clean cases lose phrase top1, first-token semantic closure is not sufficient for phrase closure.
- If semantic-only cases keep phrase top1, lexical capitalization is likely a surface realization issue.
