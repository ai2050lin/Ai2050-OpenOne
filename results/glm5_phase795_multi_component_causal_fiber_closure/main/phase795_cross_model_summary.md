# Phase 795 Multi-Component Causal Fiber Closure (main)

- Status: `complete`
- Intervention ladder: O only, K/V source, K/V + O, route answer, K/V + O + route.
- Goal: test whether multi-component assembly crosses token or phrase closure.
- Strict interpretation: margin/rank gains alone are not full generation closure.

## Top Minus Matched Ladder Specificity

| model | ladder | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.406 | -0.062 | 2.469 | 0.000 | 0.167 | 0.500 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.552 | -0.167 | 1.719 | 0.000 | 0.167 | 0.333 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 2.125 | 0.633 | 1.492 | 0.000 | 0.167 | 0.500 |
| qwen3 | `kv_source` | `negative` | `all_pre_answer` | 0.594 | -0.341 | 0.935 | 0.000 | 0.167 | 0.333 |
| qwen3 | `o_only` | `negative` | `all_pre_answer` | 1.526 | 0.766 | 0.760 | 0.000 | 0.167 | 0.167 |
| qwen3 | `kv_o_route` | `negative` | `all_pre_answer` | 5.599 | 5.438 | 0.161 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 5.578 | 5.422 | 0.156 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o` | `negative` | `all_pre_answer` | 0.979 | 0.909 | 0.070 | 0.000 | 0.167 | 0.167 |
| qwen3 | `route_answer` | `negative` | `all_pre_answer` | 5.630 | 5.630 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 5.630 | 5.630 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `o_only` | `negative` | `all_pre_answer` | 0.495 | -0.010 | 0.505 | 0.000 | 0.000 | 0.667 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 0.547 | -0.016 | 0.562 | 0.000 | 0.000 | 0.333 |
| glm4 | `kv_o_route` | `negative` | `all_pre_answer` | 0.776 | 0.776 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | 0.776 | 0.776 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `negative` | `all_pre_answer` | 0.682 | 0.682 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 0.682 | 0.682 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `negative` | `all_pre_answer` | 0.807 | 0.807 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | 0.807 | 0.807 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 0.677 | 0.677 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 0.521 | 0.547 | -0.026 | 0.000 | 0.000 | -0.333 |
| glm4 | `kv_o` | `negative` | `all_pre_answer` | 0.573 | 0.635 | -0.062 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `o_only` | `negative` | `all_pre_answer` | 0.657 | 0.089 | 0.568 | 0.000 | 0.000 | 0.500 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 0.602 | 0.103 | 0.499 | 0.000 | 0.000 | 0.167 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 2.010 | 1.573 | 0.437 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_source` | `negative` | `all_pre_answer` | 1.724 | 1.405 | 0.319 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.474 | 1.392 | 0.083 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `all_pre_answer` | 2.201 | 2.129 | 0.072 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.152 | 2.139 | 0.013 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `all_pre_answer` | 2.107 | 2.107 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.107 | 2.107 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `all_pre_answer` | 1.265 | 1.367 | -0.102 | 0.000 | 0.000 | 0.000 |

## Top Ladder Effects

| model | selection | ladder | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 3 | 5.630 | 1.000 | 0.000 | 0.167 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 3 | 5.630 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 3 | 5.630 | 1.000 | 0.000 | 0.167 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 3 | 5.630 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 5.599 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 5.578 | 1.000 | 0.000 | 0.167 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 5.438 | 1.000 | 0.000 | 0.167 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 5.422 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 3 | 2.406 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 3 | 2.125 | 0.833 | 0.000 | 0.167 |
| qwen3 | `top` | `o_only` | `negative` | `all_pre_answer` | 3 | 1.526 | 0.833 | 0.000 | 0.167 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 3 | 1.552 | 0.667 | 0.000 | 0.167 |
| qwen3 | `top` | `kv_o` | `negative` | `all_pre_answer` | 3 | 0.979 | 0.833 | 0.000 | 0.167 |
| qwen3 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 3 | 0.909 | 0.667 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `negative` | `all_pre_answer` | 3 | 0.766 | 0.667 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `negative` | `all_pre_answer` | 3 | 0.594 | 0.833 | 0.000 | 0.167 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 3 | 0.633 | 0.333 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `positive` | `all_pre_answer` | 3 | -0.062 | 0.500 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 3 | -0.167 | 0.333 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 3 | -0.341 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 3 | 0.807 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 3 | 0.807 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 3 | 0.807 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 3 | 0.807 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 0.776 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 0.776 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 0.776 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 0.776 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 3 | 0.682 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 3 | 0.682 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `negative` | `all_pre_answer` | 3 | 0.682 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 3 | 0.682 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 3 | 0.677 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 3 | 0.677 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 3 | 0.547 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 3 | 0.635 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `all_pre_answer` | 3 | 0.495 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `negative` | `all_pre_answer` | 3 | 0.573 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 3 | 0.547 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 3 | 0.521 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `negative` | `all_pre_answer` | 3 | -0.010 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `positive` | `all_pre_answer` | 3 | -0.016 | 0.333 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 2.201 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 2.152 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 2.139 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 2.129 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 3 | 2.107 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 3 | 2.107 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 3 | 2.107 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 3 | 2.107 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 3 | 2.010 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `negative` | `all_pre_answer` | 3 | 1.724 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 3 | 1.573 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 3 | 1.474 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `negative` | `all_pre_answer` | 3 | 1.405 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 3 | 1.392 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `negative` | `all_pre_answer` | 3 | 1.367 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `all_pre_answer` | 3 | 1.265 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `all_pre_answer` | 3 | 0.657 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `all_pre_answer` | 3 | 0.602 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `positive` | `all_pre_answer` | 3 | 0.103 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `negative` | `all_pre_answer` | 3 | 0.089 | 0.500 | 0.000 | 0.000 |
