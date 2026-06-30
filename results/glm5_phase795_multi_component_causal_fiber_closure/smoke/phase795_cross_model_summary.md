# Phase 795 Multi-Component Causal Fiber Closure (smoke)

- Status: `complete`
- Intervention ladder: O only, K/V source, K/V + O, route answer, K/V + O + route.
- Goal: test whether multi-component assembly crosses token or phrase closure.
- Strict interpretation: margin/rank gains alone are not full generation closure.

## Top Minus Matched Ladder Specificity

| model | ladder | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.281 | -0.188 | 2.469 | 0.000 | 0.000 | 1.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 4.594 | 4.594 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 5.031 | 5.031 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.594 | 4.094 | -2.500 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 1.719 | 4.438 | -2.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 1.250 | -0.188 | 1.438 | 0.000 | 0.000 | 1.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | 1.344 | 1.344 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 1.219 | 1.219 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | 1.344 | 1.344 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 1.656 | 1.656 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 1.094 | 1.250 | -0.156 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 1.266 | -0.023 | 1.289 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 2.391 | 2.375 | 0.016 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.344 | 2.344 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.438 | 2.750 | -0.312 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.672 | 2.391 | -0.719 | 0.000 | 0.000 | 0.000 |

## Top Ladder Effects

| model | selection | ladder | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 5.031 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 5.031 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 4.594 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 4.594 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 4.438 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 4.094 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 2.281 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.719 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.594 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.188 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 1.344 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 1.344 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 1.344 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 1.344 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 1.250 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.250 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.219 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 1.219 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.094 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 1 | 1.656 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 1 | 1.656 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.188 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 2.750 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 2.438 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 1 | 2.391 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 1 | 2.391 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 1 | 2.375 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 2.344 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 2.344 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 1 | 1.672 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `all_pre_answer` | 1 | 1.266 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `positive` | `all_pre_answer` | 1 | -0.023 | 1.000 | 0.000 | 0.000 |
