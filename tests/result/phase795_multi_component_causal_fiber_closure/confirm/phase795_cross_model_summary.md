# Phase 795 Multi-Component Causal Fiber Closure (confirm)

- Status: `complete`
- Intervention ladder: O only, K/V source, K/V + O, route answer, K/V + O + route.
- Goal: test whether multi-component assembly crosses token or phrase closure.
- Strict interpretation: margin/rank gains alone are not full generation closure.

## Top Minus Matched Ladder Specificity

| model | ladder | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.222 | -0.356 | 2.578 | 0.000 | 0.100 | 0.500 |
| qwen3 | `o_only` | `positive` | `instruction` | 2.222 | -0.356 | 2.578 | 0.000 | 0.100 | 0.500 |
| qwen3 | `kv_o` | `positive` | `instruction` | 2.222 | -0.453 | 2.675 | 0.000 | 0.100 | 0.400 |
| qwen3 | `kv_o` | `negative` | `instruction` | 1.731 | 0.047 | 1.684 | 0.000 | 0.100 | 0.000 |
| qwen3 | `o_only` | `negative` | `all_pre_answer` | 1.716 | 0.144 | 1.572 | 0.000 | 0.100 | 0.000 |
| qwen3 | `o_only` | `negative` | `instruction` | 1.716 | 0.144 | 1.572 | 0.000 | 0.100 | 0.000 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.920 | 0.409 | 1.511 | 0.000 | -0.100 | 0.000 |
| qwen3 | `kv_source` | `negative` | `all_pre_answer` | 1.125 | 0.147 | 0.978 | 0.000 | -0.100 | 0.100 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 2.456 | 1.855 | 0.602 | 0.000 | 0.000 | 0.200 |
| qwen3 | `kv_o` | `negative` | `all_pre_answer` | 1.670 | 1.373 | 0.297 | 0.000 | 0.000 | 0.200 |
| qwen3 | `kv_source` | `negative` | `instruction` | 0.031 | -0.200 | 0.231 | 0.000 | 0.000 | -0.100 |
| qwen3 | `kv_source` | `positive` | `instruction` | 0.028 | -0.194 | 0.222 | 0.000 | 0.000 | -0.100 |
| qwen3 | `kv_o_route` | `negative` | `all_pre_answer` | 5.412 | 5.328 | 0.084 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 5.403 | 5.334 | 0.069 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o_route` | `negative` | `instruction` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `instruction` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `negative` | `all_pre_answer` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `negative` | `instruction` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3 | `route_answer` | `positive` | `instruction` | 5.472 | 5.472 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 0.736 | -0.102 | 0.838 | 0.000 | 0.000 | 1.000 |
| glm4 | `o_only` | `positive` | `instruction` | 0.736 | -0.102 | 0.838 | 0.000 | 0.000 | 1.000 |
| glm4 | `kv_o` | `negative` | `instruction` | 0.730 | -0.070 | 0.800 | 0.000 | 0.000 | 1.000 |
| glm4 | `kv_o` | `positive` | `instruction` | 0.723 | -0.056 | 0.780 | 0.000 | 0.000 | 1.000 |
| glm4 | `o_only` | `negative` | `all_pre_answer` | 0.736 | -0.087 | 0.823 | 0.000 | 0.000 | 0.800 |
| glm4 | `o_only` | `negative` | `instruction` | 0.736 | -0.087 | 0.823 | 0.000 | 0.000 | 0.800 |
| glm4 | `kv_o` | `negative` | `all_pre_answer` | 0.798 | 0.792 | 0.006 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 0.811 | 0.805 | 0.006 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `negative` | `all_pre_answer` | 0.952 | 0.952 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `negative` | `instruction` | 0.939 | 0.939 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | 0.952 | 0.952 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `instruction` | 0.939 | 0.939 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `negative` | `all_pre_answer` | 0.895 | 0.895 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `negative` | `instruction` | -0.022 | -0.022 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 0.895 | 0.895 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `kv_source` | `positive` | `instruction` | -0.022 | -0.022 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `negative` | `all_pre_answer` | 0.964 | 0.964 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `negative` | `instruction` | 0.964 | 0.964 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | 0.964 | 0.964 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `instruction` | 0.964 | 0.964 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 0.892 | 0.892 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `instruction` | 0.892 | 0.892 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 2.223 | 1.286 | 0.937 | 0.000 | 0.000 | 0.200 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 0.892 | 0.080 | 0.812 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `o_only` | `positive` | `instruction` | 0.892 | 0.080 | 0.812 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `o_only` | `negative` | `all_pre_answer` | 0.854 | 0.049 | 0.804 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `o_only` | `negative` | `instruction` | 0.854 | 0.049 | 0.804 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `kv_o` | `positive` | `instruction` | 1.300 | 0.759 | 0.542 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `kv_source` | `positive` | `instruction` | 1.467 | 0.822 | 0.645 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_source` | `negative` | `all_pre_answer` | 2.052 | 1.533 | 0.519 | 0.000 | 0.000 | 0.100 |
| deepseek7b | `kv_source` | `negative` | `instruction` | 1.485 | 1.039 | 0.446 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `instruction` | 1.260 | 0.916 | 0.344 | 0.000 | 0.000 | 0.200 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.573 | 1.337 | 0.236 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `all_pre_answer` | 1.543 | 1.489 | 0.054 | 0.000 | 0.000 | 0.100 |
| deepseek7b | `kv_o_route` | `positive` | `instruction` | 2.362 | 2.359 | 0.003 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `instruction` | 2.359 | 2.359 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `all_pre_answer` | 2.331 | 2.331 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `instruction` | 2.331 | 2.331 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.331 | 2.331 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `instruction` | 2.331 | 2.331 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.349 | 2.373 | -0.023 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `all_pre_answer` | 2.370 | 2.415 | -0.045 | 0.000 | 0.000 | 0.000 |

## Top Ladder Effects

| model | selection | ladder | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o_route` | `positive` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `route_answer` | `positive` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `route_answer` | `negative` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o_route` | `negative` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `route_answer` | `negative` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `instruction` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 5 | 5.472 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 5.412 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 5.403 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 5.334 | 1.000 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 5.328 | 1.000 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 5 | 2.456 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `o_only` | `positive` | `instruction` | 5 | 2.222 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o` | `positive` | `instruction` | 5 | 2.222 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 5 | 2.222 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 5 | 1.920 | 0.700 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o` | `negative` | `instruction` | 5 | 1.731 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `o_only` | `negative` | `instruction` | 5 | 1.716 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `o_only` | `negative` | `all_pre_answer` | 5 | 1.716 | 0.900 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 5 | 1.855 | 0.700 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_o` | `negative` | `all_pre_answer` | 5 | 1.670 | 0.900 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 5 | 1.373 | 0.700 | 0.000 | 0.100 |
| qwen3 | `top` | `kv_source` | `negative` | `all_pre_answer` | 5 | 1.125 | 0.800 | 0.000 | 0.100 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 5 | 0.409 | 0.700 | 0.000 | 0.200 |
| qwen3 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 5 | 0.147 | 0.700 | 0.000 | 0.200 |
| glm4 | `top` | `route_answer` | `positive` | `instruction` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `instruction` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `instruction` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `instruction` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 5 | 0.964 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 0.952 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 0.952 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 0.952 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 0.952 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `instruction` | 5 | 0.939 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `instruction` | 5 | 0.939 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `instruction` | 5 | 0.939 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `instruction` | 5 | 0.939 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 5 | 0.895 | 0.800 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 5 | 0.895 | 0.800 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `negative` | `all_pre_answer` | 5 | 0.895 | 0.800 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 5 | 0.895 | 0.800 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `instruction` | 5 | 0.892 | 0.800 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 5 | 0.892 | 0.800 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `instruction` | 5 | 0.892 | 0.800 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 5 | 0.892 | 0.800 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `instruction` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `instruction` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `all_pre_answer` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 5 | 0.811 | 0.800 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `negative` | `instruction` | 5 | 0.730 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 2.415 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 2.373 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 5 | 2.370 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `instruction` | 5 | 2.362 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `instruction` | 5 | 2.359 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `instruction` | 5 | 2.359 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `instruction` | 5 | 2.359 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 5 | 2.349 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `instruction` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `instruction` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `instruction` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `instruction` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 5 | 2.331 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 5 | 2.223 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `negative` | `all_pre_answer` | 5 | 2.052 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 5 | 1.573 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `all_pre_answer` | 5 | 1.543 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `negative` | `all_pre_answer` | 5 | 1.533 | 0.800 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `negative` | `all_pre_answer` | 5 | 1.489 | 0.800 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 5 | 1.337 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `instruction` | 5 | 1.300 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `instruction` | 5 | 1.260 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `instruction` | 5 | 1.467 | 0.600 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `negative` | `instruction` | 5 | 1.485 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 5 | 1.286 | 0.700 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `instruction` | 5 | 0.854 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `all_pre_answer` | 5 | 0.854 | 0.900 | 0.000 | 0.000 |
