# Phase 796 Global Competitor Suppression and Token Identity Closure Audit (main)

- Status: `complete`
- Goal: audit the real top-k vocabulary competitors that block token top1 closure.
- Boundary: target-vs-contrast improvement is not sufficient; target must beat every non-target token.

## Competitor Class Counts After Intervention

| model | class | count |
|---|---|---:|
| qwen3 | `designated_contrast` | 86 |
| qwen3 | `candidate_list_or_case_value` | 56 |
| qwen3 | `whitespace_or_newline` | 18 |
| glm4 | `designated_contrast` | 51 |
| glm4 | `candidate_list_or_case_value` | 36 |
| glm4 | `echo_token` | 1 |
| deepseek7b | `designated_contrast` | 78 |
| deepseek7b | `candidate_list_or_case_value` | 42 |
| deepseek7b | `echo_token` | 40 |

## Top Minus Matched Global Specificity

| model | ladder | subspace | source group | top global delta | matched global delta | gap | top target gain | top suppress rate | top token gain |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.258 | 0.102 | 2.156 | 1.289 | 0.875 | 0.000 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 2.039 | 0.920 | 1.119 | 1.008 | 0.875 | 0.000 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.695 | 0.352 | 1.344 | 1.523 | 0.500 | 0.000 |
| qwen3 | `o_only` | `negative` | `all_pre_answer` | 1.629 | 0.652 | 0.977 | 0.879 | 0.875 | 0.000 |
| qwen3 | `kv_source` | `negative` | `all_pre_answer` | 1.008 | 0.205 | 0.803 | 1.070 | 0.375 | 0.000 |
| qwen3 | `kv_o` | `negative` | `all_pre_answer` | 1.211 | 1.049 | 0.162 | 0.539 | 0.750 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 2.504 | 2.418 | 0.086 | 2.848 | 0.875 | 0.000 |
| qwen3 | `kv_o_route` | `negative` | `all_pre_answer` | 2.520 | 2.469 | 0.051 | 2.816 | 0.875 | 0.000 |
| qwen3 | `route_answer` | `negative` | `all_pre_answer` | 2.629 | 2.629 | 0.000 | 2.785 | 0.875 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 2.629 | 2.629 | 0.000 | 2.785 | 0.875 | 0.000 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 0.258 | -0.031 | 0.289 | 0.398 | 0.500 | 0.000 |
| glm4 | `o_only` | `negative` | `all_pre_answer` | 0.215 | -0.035 | 0.250 | 0.418 | 0.500 | 0.000 |
| glm4 | `kv_o_route` | `negative` | `all_pre_answer` | -0.242 | -0.242 | 0.000 | 1.508 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | -0.242 | -0.242 | 0.000 | 1.508 | 0.000 | 0.000 |
| glm4 | `kv_source` | `negative` | `all_pre_answer` | 0.340 | 0.340 | 0.000 | 0.309 | 0.500 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 0.340 | 0.340 | 0.000 | 0.309 | 0.500 | 0.000 |
| glm4 | `route_answer` | `negative` | `all_pre_answer` | -0.180 | -0.180 | 0.000 | 1.539 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | -0.180 | -0.180 | 0.000 | 1.539 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 0.363 | 0.363 | 0.000 | 0.129 | 0.500 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 0.121 | 0.242 | -0.121 | 0.230 | 0.500 | 0.000 |
| glm4 | `kv_o` | `negative` | `all_pre_answer` | 0.172 | 0.316 | -0.145 | 0.281 | 0.500 | 0.000 |
| deepseek7b | `o_only` | `negative` | `all_pre_answer` | 1.250 | 0.145 | 1.105 | 1.375 | 0.375 | 0.000 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 1.155 | 0.085 | 1.070 | 1.194 | 0.375 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.539 | 0.997 | 0.542 | 1.719 | 0.500 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `all_pre_answer` | 1.465 | 1.045 | 0.420 | 1.644 | 0.500 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 1.295 | 1.074 | 0.221 | 1.623 | 0.875 | 0.000 |
| deepseek7b | `kv_source` | `negative` | `all_pre_answer` | 1.192 | 1.050 | 0.142 | 1.489 | 0.750 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `all_pre_answer` | 2.664 | 2.614 | 0.050 | 3.523 | 0.500 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.601 | 2.591 | 0.010 | 3.405 | 0.500 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `all_pre_answer` | 2.658 | 2.658 | 0.000 | 3.557 | 0.375 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.658 | 2.658 | 0.000 | 3.557 | 0.375 | 0.000 |

## Top Global Effects

| model | selection | ladder | subspace | source group | cases | global delta | target gain | suppress rate | cross rate | token gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 4 | 2.629 | 2.785 | 0.875 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 4 | 2.629 | 2.785 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 4 | 2.629 | 2.785 | 0.875 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 4 | 2.629 | 2.785 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | 2.520 | 2.816 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | 2.504 | 2.848 | 0.875 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | 2.469 | 2.844 | 0.875 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | 2.418 | 2.824 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 4 | 2.258 | 1.289 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 4 | 2.039 | 1.008 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `negative` | `all_pre_answer` | 4 | 1.629 | 0.879 | 0.875 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 4 | 1.695 | 1.523 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `negative` | `all_pre_answer` | 4 | 1.211 | 0.539 | 0.750 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 4 | 1.049 | 0.533 | 0.625 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `negative` | `all_pre_answer` | 4 | 1.008 | 1.070 | 0.375 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 4 | 0.920 | 0.670 | 0.500 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `negative` | `all_pre_answer` | 4 | 0.652 | 0.480 | 0.750 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 4 | 0.352 | 0.430 | 0.375 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 4 | 0.205 | 0.268 | 0.375 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_only` | `positive` | `all_pre_answer` | 4 | 0.102 | 0.102 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 4 | 0.363 | 0.129 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 4 | 0.363 | 0.129 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 4 | 0.340 | 0.309 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 4 | 0.340 | 0.309 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `negative` | `all_pre_answer` | 4 | 0.340 | 0.309 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 4 | 0.340 | 0.309 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 4 | 0.316 | 0.363 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 4 | 0.242 | 0.320 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 4 | 0.258 | 0.398 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `all_pre_answer` | 4 | 0.215 | 0.418 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `negative` | `all_pre_answer` | 4 | 0.172 | 0.281 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 4 | 0.121 | 0.230 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `positive` | `all_pre_answer` | 4 | -0.031 | 0.031 | 0.250 | 0.000 | 0.000 |
| glm4 | `matched` | `o_only` | `negative` | `all_pre_answer` | 4 | -0.035 | -0.035 | 0.250 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 4 | -0.180 | 1.539 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 4 | -0.180 | 1.539 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 4 | -0.180 | 1.539 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 4 | -0.180 | 1.539 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | -0.242 | 1.508 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | -0.242 | 1.508 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | -0.242 | 1.508 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | -0.242 | 1.508 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | 2.614 | 3.435 | 0.625 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 4 | 2.664 | 3.523 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | 2.601 | 3.405 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 4 | 2.591 | 3.427 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 4 | 2.658 | 3.557 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 4 | 2.658 | 3.557 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 4 | 2.658 | 3.557 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 4 | 2.658 | 3.557 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 4 | 1.539 | 1.719 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 4 | 1.295 | 1.623 | 0.875 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `all_pre_answer` | 4 | 1.465 | 1.644 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `negative` | `all_pre_answer` | 4 | 1.192 | 1.489 | 0.750 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `all_pre_answer` | 4 | 1.250 | 1.375 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 4 | 1.074 | 1.652 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `all_pre_answer` | 4 | 1.155 | 1.194 | 0.375 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `negative` | `all_pre_answer` | 4 | 1.050 | 1.417 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `negative` | `all_pre_answer` | 4 | 1.045 | 1.420 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 4 | 0.997 | 1.560 | 0.250 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `negative` | `all_pre_answer` | 4 | 0.145 | 0.028 | 0.750 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_only` | `positive` | `all_pre_answer` | 4 | 0.085 | 0.132 | 0.375 | 0.000 | 0.000 |
