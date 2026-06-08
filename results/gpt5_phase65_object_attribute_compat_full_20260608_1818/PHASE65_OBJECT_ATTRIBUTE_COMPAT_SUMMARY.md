# Phase65 Object-Attribute Compatibility Decomposition Summary

## qwen3

layers=[4, 8, 12, 16, 20], pairs=144

| layer | full | neutral_ideal | L1_FULL | L2cf_FULL | OBJcf_FULL |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 3 | 1 | 1 | 1 |
| 8 | 2 | 3 | 0 | 0 | 2 |
| 12 | 1 | 3 | 0 | 0 | 1 |
| 16 | 2 | 3 | 0 | 0 | 2 |
| 20 | 0 | 1 | 0 | 0 | 0 |

## glm4

layers=[4, 10, 20, 30], pairs=144

| layer | full | neutral_ideal | L1_FULL | L2cf_FULL | OBJcf_FULL |
|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 0 | 0 | 0 | 0 |
| 10 | 3 | 0 | 2 | 1 | 0 |
| 20 | 1 | 1 | 0 | 0 | 1 |
| 30 | 2 | 1 | 0 | 0 | 2 |

## deepseek7b

layers=[4, 8, 12, 16, 20], pairs=144

| layer | full | neutral_ideal | L1_FULL | L2cf_FULL | OBJcf_FULL |
|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 4 | 3 | 4 | 5 |
| 8 | 6 | 3 | 2 | 3 | 1 |
| 12 | 11 | 2 | 6 | 2 | 3 |
| 16 | 10 | 1 | 3 | 3 | 4 |
| 20 | 7 | 5 | 2 | 1 | 4 |

