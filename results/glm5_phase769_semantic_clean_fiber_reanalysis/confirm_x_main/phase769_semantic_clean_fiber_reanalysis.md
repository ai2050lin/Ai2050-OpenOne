# Phase 769 Semantic Clean Fiber Reanalysis (confirm / main)

- Status: `complete`
- Input: Phase 765 causal-fiber effect rows filtered by Phase 767 semantic/exact subsets.
- This is an offline reanalysis; no model was loaded.

## Subset Summary

| model | subset | cases | effect rows | mean sep | mean NN | object gap | domain gap |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `all` | 108 | 1080 | 0.605 | 0.583 | 0.088 | 0.585 |
| qwen3 | `exact_clean` | 89 | 890 | 0.471 | 0.500 | 0.084 | 0.437 |
| qwen3 | `semantic_clean` | 89 | 890 | 0.471 | 0.500 | 0.084 | 0.437 |
| qwen3 | `semantic_only` | 0 | 0 | null | null | null | null |
| qwen3 | `semantic_fail` | 19 | 190 | 0.868 | 0.436 | -0.237 | 1.126 |
| qwen3 | `rank_le2` | 102 | 1020 | 0.609 | 0.556 | 0.034 | 0.571 |
| glm4 | `all` | 108 | 1080 | 0.182 | 0.250 | 0.124 | 0.037 |
| glm4 | `exact_clean` | 66 | 660 | 0.163 | 0.149 | 0.482 | 0.085 |
| glm4 | `semantic_clean` | 81 | 810 | 0.258 | 0.167 | 0.389 | 0.307 |
| glm4 | `semantic_only` | 15 | 150 | 0.647 | 0.364 | null | null |
| glm4 | `semantic_fail` | 27 | 270 | 0.448 | 0.300 | 0.556 | 0.264 |
| glm4 | `rank_le2` | 99 | 990 | 0.104 | 0.194 | 0.153 | 0.043 |
| deepseek7b | `all` | 108 | 1080 | 0.378 | 0.667 | 0.071 | 0.126 |
| deepseek7b | `exact_clean` | 19 | 190 | 0.814 | 0.633 | 0.125 | -0.084 |
| deepseek7b | `semantic_clean` | 38 | 380 | 0.228 | 0.300 | 0.097 | 0.124 |
| deepseek7b | `semantic_only` | 19 | 190 | -0.061 | 0.231 | null | null |
| deepseek7b | `semantic_fail` | 70 | 700 | 0.144 | 0.281 | 0.129 | 0.062 |
| deepseek7b | `rank_le2` | 55 | 550 | 0.310 | 0.418 | 0.109 | 0.001 |

## Strict Interpretation

- If `semantic_clean` improves separation or stability over `all`, failed states were polluting the mechanism graph.
- If `semantic_only` resembles `exact_clean`, lexical realization is likely a surface output issue.
- Small subsets, especially qwen3 `semantic_only`, should not be over-interpreted.
