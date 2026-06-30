# Phase 803 Semantic New-Blocker Source Localization (smoke)

- Status: `complete`
- Boundary: tracks the same semantic new blockers released at alpha=0 across target-direction doses.
- A lower new-blocker rate is not counted as true suppression unless matched semantic blocker logits drop.

## Matched Semantic New Blockers By Alpha

| model | alpha | rows | cases | target gain | target gain vs a0 | old suppress | new rate | matched delta vs a0 | true suppress vs a0 | still above | label counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 1 | 1 | -0.062 | 0.000 | 1.242 | 0.374 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| qwen3 | 0.750 | 1 | 1 | 2.812 | 2.875 | 1.143 | 0.121 | 0.174 | -0.174 | 0.031 | `{"threshold_cover_not_true_suppression": 1}` |
| qwen3 | 1.000 | 1 | 1 | 3.750 | 3.812 | 1.117 | 0.048 | 0.246 | -0.246 | 0.031 | `{"threshold_cover_not_true_suppression": 1}` |
| glm4 | 0.000 | 1 | 1 | 0.344 | 0.000 | 0.790 | 0.242 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| glm4 | 0.750 | 1 | 1 | 0.094 | -0.250 | 0.784 | 0.281 | 0.019 | -0.019 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| glm4 | 1.000 | 1 | 1 | 0.000 | -0.344 | 0.784 | 0.301 | 0.022 | -0.022 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| deepseek7b | 0.000 | 1 | 1 | 0.938 | 0.000 | -1.100 | 0.501 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| deepseek7b | 0.750 | 1 | 1 | 2.422 | 1.484 | -1.139 | 0.160 | 0.030 | -0.030 | 1.000 | `{"semantic_new_blockers_persist": 1}` |
| deepseek7b | 1.000 | 1 | 1 | 2.922 | 1.984 | -1.157 | 0.101 | 0.036 | -0.036 | 1.000 | `{"semantic_new_blockers_persist": 1}` |

## Top Component Sources

| model | component | rows | cases | semantic new | overlap | jaccard | gap | logit delta | release score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `mlp:L35` | 1 | 1 | 32.000 | 10.000 | 0.185 | 0.605 | 0.854 | 19.375 |
| qwen3 | `mlp:L34` | 1 | 1 | 5.000 | 3.000 | 0.088 | 0.250 | 0.562 | 1.250 |
| qwen3 | `attn:L35` | 1 | 1 | 2.000 | 2.000 | 0.062 | 0.656 | 1.156 | 1.312 |
| qwen3 | `mlp:L33` | 1 | 1 | 3.000 | 2.000 | 0.061 | 0.417 | 1.354 | 1.250 |
| glm4 | `mlp:L38` | 1 | 1 | 15.000 | 15.000 | 0.714 | 0.356 | 1.081 | 5.344 |
| glm4 | `mlp:L39` | 1 | 1 | 11.000 | 7.000 | 0.280 | 0.372 | 0.911 | 4.094 |
| glm4 | `mlp:L27` | 1 | 1 | 10.000 | 4.000 | 0.148 | 0.169 | 0.138 | 1.688 |
| glm4 | `mlp:L34` | 1 | 1 | 5.000 | 3.000 | 0.130 | 0.175 | 0.219 | 0.875 |
| deepseek7b | `mlp:L27` | 1 | 1 | 32.000 | 20.000 | 0.455 | 1.645 | 3.246 | 52.625 |
| deepseek7b | `attn:L19` | 1 | 1 | 24.000 | 17.000 | 0.436 | 0.581 | 2.908 | 13.938 |
| deepseek7b | `mlp:L26` | 1 | 1 | 32.000 | 13.000 | 0.255 | 0.317 | 1.405 | 10.141 |
| deepseek7b | `mlp:L24` | 1 | 1 | 25.000 | 7.000 | 0.140 | 0.201 | 0.768 | 5.031 |
