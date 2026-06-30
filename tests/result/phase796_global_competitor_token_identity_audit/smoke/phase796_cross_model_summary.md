# Phase 796 Global Competitor Suppression and Token Identity Closure Audit (smoke)

- Status: `complete`
- Goal: audit the real top-k vocabulary competitors that block token top1 closure.
- Boundary: target-vs-contrast improvement is not sufficient; target must beat every non-target token.

## Competitor Class Counts After Intervention

| model | class | count |
|---|---|---:|
| qwen3 | `designated_contrast` | 20 |
| glm4 | `designated_contrast` | 10 |
| glm4 | `candidate_list_or_case_value` | 2 |
| deepseek7b | `designated_contrast` | 12 |
| deepseek7b | `candidate_list_or_case_value` | 8 |

## Top Minus Matched Global Specificity

| model | ladder | subspace | source group | top global delta | matched global delta | gap | top target gain | top suppress rate | top token gain |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.281 | -0.188 | 2.469 | 1.594 | 1.000 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 4.594 | 4.594 | 0.000 | 2.969 | 1.000 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 5.031 | 5.031 | 0.000 | 2.469 | 1.000 | 0.000 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.594 | 4.094 | -2.500 | 1.469 | 0.500 | 0.000 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 1.719 | 4.438 | -2.719 | 1.406 | 0.500 | 0.000 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 1.250 | -0.188 | 1.438 | 1.125 | 1.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | 1.344 | 1.344 | 0.000 | 2.719 | 0.000 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 1.219 | 1.219 | 0.000 | 0.594 | 1.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | 1.344 | 1.344 | 0.000 | 2.719 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 1.469 | 1.469 | 0.000 | -0.219 | 1.000 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 1.094 | 1.250 | -0.156 | 0.469 | 1.000 | 0.000 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 1.266 | -0.023 | 1.289 | 1.266 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.672 | 1.578 | 0.094 | 1.547 | 1.000 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 1.297 | 1.281 | 0.016 | 1.922 | 1.000 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.031 | 2.031 | 0.000 | 2.719 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.125 | 2.375 | -0.250 | 2.562 | 0.000 | 0.000 |

## Top Global Effects

| model | selection | ladder | subspace | source group | cases | global delta | target gain | suppress rate | cross rate | token gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 5.031 | 2.469 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 5.031 | 2.469 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 4.594 | 2.969 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 4.594 | 2.594 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 4.438 | 3.750 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 4.094 | 3.469 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 2.281 | 1.594 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.719 | 1.406 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.594 | 1.469 | 0.500 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.188 | -0.500 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 1.250 | 1.125 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.250 | 0.688 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.219 | 0.594 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.219 | 0.594 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.094 | 0.469 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 1 | 1.469 | -0.219 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 1 | 1.469 | -0.219 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 1.344 | 2.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 1.344 | 2.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 1.344 | 2.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 1.344 | 2.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.188 | -0.250 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 2.375 | 2.719 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.672 | 1.547 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.578 | 2.172 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 2.125 | 2.562 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 2.031 | 2.719 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 2.031 | 2.719 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.297 | 1.922 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.281 | 1.969 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 1.266 | 1.266 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.023 | 0.008 | 0.500 | 0.000 | 0.000 |
