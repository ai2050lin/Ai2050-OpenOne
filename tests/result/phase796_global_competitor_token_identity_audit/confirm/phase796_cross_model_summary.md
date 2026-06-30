# Phase 796 Global Competitor Suppression and Token Identity Closure Audit (confirm)

- Status: `complete`
- Goal: audit the real top-k vocabulary competitors that block token top1 closure.
- Boundary: target-vs-contrast improvement is not sufficient; target must beat every non-target token.

## Competitor Class Counts After Intervention

| model | class | count |
|---|---|---:|
| qwen3 | `designated_contrast` | 270 |
| qwen3 | `candidate_list_or_case_value` | 162 |
| qwen3 | `whitespace_or_newline` | 48 |
| glm4 | `designated_contrast` | 166 |
| glm4 | `candidate_list_or_case_value` | 96 |
| glm4 | `echo_token` | 2 |
| deepseek7b | `designated_contrast` | 256 |
| deepseek7b | `candidate_list_or_case_value` | 144 |
| deepseek7b | `echo_token` | 80 |

## Top Minus Matched Global Specificity

| model | ladder | subspace | source group | top global delta | matched global delta | gap | top target gain | top suppress rate | top token gain |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `kv_o` | `positive` | `instruction` | 2.195 | -0.346 | 2.542 | 1.435 | 0.833 | 0.000 |
| qwen3 | `o_only` | `positive` | `all_pre_answer` | 2.190 | -0.260 | 2.451 | 1.440 | 0.833 | 0.000 |
| qwen3 | `o_only` | `positive` | `instruction` | 2.190 | -0.260 | 2.451 | 1.440 | 0.833 | 0.000 |
| qwen3 | `kv_o` | `negative` | `instruction` | 1.750 | 0.076 | 1.674 | 1.135 | 0.833 | 0.000 |
| qwen3 | `o_only` | `negative` | `all_pre_answer` | 1.747 | 0.135 | 1.612 | 1.122 | 0.833 | 0.000 |
| qwen3 | `o_only` | `negative` | `instruction` | 1.747 | 0.135 | 1.612 | 1.122 | 0.833 | 0.000 |
| qwen3 | `kv_source` | `positive` | `all_pre_answer` | 1.928 | 0.565 | 1.363 | 1.793 | 0.417 | 0.000 |
| qwen3 | `kv_source` | `negative` | `all_pre_answer` | 1.453 | 0.336 | 1.117 | 1.651 | 0.250 | 0.000 |
| qwen3 | `kv_o` | `positive` | `all_pre_answer` | 2.344 | 1.618 | 0.725 | 1.948 | 0.667 | 0.000 |
| qwen3 | `kv_o` | `negative` | `all_pre_answer` | 1.897 | 1.270 | 0.628 | 1.803 | 0.500 | 0.000 |
| qwen3 | `kv_source` | `positive` | `instruction` | 0.023 | -0.193 | 0.216 | 0.023 | 0.083 | 0.000 |
| qwen3 | `kv_source` | `negative` | `instruction` | 0.010 | -0.188 | 0.198 | 0.021 | 0.000 | 0.000 |
| qwen3 | `kv_o_route` | `negative` | `all_pre_answer` | 2.438 | 2.383 | 0.055 | 2.958 | 0.833 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `all_pre_answer` | 2.424 | 2.383 | 0.042 | 2.977 | 0.833 | 0.000 |
| qwen3 | `kv_o_route` | `positive` | `instruction` | 2.560 | 2.560 | 0.000 | 2.852 | 0.833 | 0.000 |
| qwen3 | `route_answer` | `negative` | `all_pre_answer` | 2.560 | 2.560 | 0.000 | 2.852 | 0.833 | 0.000 |
| qwen3 | `route_answer` | `negative` | `instruction` | 2.560 | 2.560 | 0.000 | 2.852 | 0.833 | 0.000 |
| qwen3 | `route_answer` | `positive` | `all_pre_answer` | 2.560 | 2.560 | 0.000 | 2.852 | 0.833 | 0.000 |
| qwen3 | `route_answer` | `positive` | `instruction` | 2.560 | 2.560 | 0.000 | 2.852 | 0.833 | 0.000 |
| qwen3 | `kv_o_route` | `negative` | `instruction` | 2.560 | 2.565 | -0.005 | 2.852 | 0.833 | 0.000 |
| glm4 | `o_only` | `positive` | `all_pre_answer` | 0.441 | -0.100 | 0.542 | 0.681 | 0.500 | 0.000 |
| glm4 | `o_only` | `positive` | `instruction` | 0.441 | -0.100 | 0.542 | 0.681 | 0.500 | 0.000 |
| glm4 | `o_only` | `negative` | `all_pre_answer` | 0.447 | -0.083 | 0.530 | 0.665 | 0.500 | 0.000 |
| glm4 | `o_only` | `negative` | `instruction` | 0.447 | -0.083 | 0.530 | 0.665 | 0.500 | 0.000 |
| glm4 | `kv_o` | `negative` | `instruction` | 0.434 | -0.066 | 0.500 | 0.663 | 0.500 | 0.000 |
| glm4 | `kv_o` | `positive` | `instruction` | 0.418 | -0.057 | 0.475 | 0.668 | 0.500 | 0.000 |
| glm4 | `kv_o_route` | `negative` | `all_pre_answer` | 0.004 | 0.004 | 0.000 | 1.889 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `negative` | `instruction` | 0.025 | 0.025 | 0.000 | 1.889 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `all_pre_answer` | 0.004 | 0.004 | 0.000 | 1.889 | 0.000 | 0.000 |
| glm4 | `kv_o_route` | `positive` | `instruction` | 0.025 | 0.025 | 0.000 | 1.889 | 0.000 | 0.000 |
| glm4 | `kv_source` | `negative` | `all_pre_answer` | 0.548 | 0.548 | 0.000 | 0.548 | 0.500 | 0.000 |
| glm4 | `kv_source` | `negative` | `instruction` | -0.026 | -0.026 | 0.000 | -0.005 | 0.167 | 0.000 |
| glm4 | `kv_source` | `positive` | `all_pre_answer` | 0.548 | 0.548 | 0.000 | 0.548 | 0.500 | 0.000 |
| glm4 | `kv_source` | `positive` | `instruction` | -0.026 | -0.026 | 0.000 | -0.005 | 0.167 | 0.000 |
| glm4 | `route_answer` | `negative` | `all_pre_answer` | 0.030 | 0.030 | 0.000 | 1.895 | 0.000 | 0.000 |
| glm4 | `route_answer` | `negative` | `instruction` | 0.030 | 0.030 | 0.000 | 1.895 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `all_pre_answer` | 0.030 | 0.030 | 0.000 | 1.895 | 0.000 | 0.000 |
| glm4 | `route_answer` | `positive` | `instruction` | 0.030 | 0.030 | 0.000 | 1.895 | 0.000 | 0.000 |
| glm4 | `route_answer` | `route` | `all_pre_answer` | 0.647 | 0.647 | 0.000 | 0.397 | 0.667 | 0.000 |
| glm4 | `route_answer` | `route` | `instruction` | 0.647 | 0.647 | 0.000 | 0.397 | 0.667 | 0.000 |
| glm4 | `kv_o` | `positive` | `all_pre_answer` | 0.410 | 0.473 | -0.062 | 0.546 | 0.500 | 0.000 |
| glm4 | `kv_o` | `negative` | `all_pre_answer` | 0.402 | 0.486 | -0.083 | 0.559 | 0.500 | 0.000 |
| deepseek7b | `o_only` | `positive` | `all_pre_answer` | 1.248 | 0.051 | 1.198 | 1.238 | 0.500 | 0.000 |
| deepseek7b | `o_only` | `positive` | `instruction` | 1.248 | 0.051 | 1.198 | 1.238 | 0.500 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `instruction` | 1.581 | 0.533 | 1.048 | 1.555 | 0.667 | 0.000 |
| deepseek7b | `o_only` | `negative` | `all_pre_answer` | 1.193 | 0.062 | 1.131 | 1.287 | 0.417 | 0.000 |
| deepseek7b | `o_only` | `negative` | `instruction` | 1.193 | 0.062 | 1.131 | 1.287 | 0.417 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `instruction` | 1.511 | 0.708 | 0.802 | 1.558 | 0.583 | 0.000 |
| deepseek7b | `kv_o` | `positive` | `all_pre_answer` | 1.472 | 0.971 | 0.502 | 1.780 | 0.583 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `instruction` | 0.986 | 0.542 | 0.444 | 1.147 | 0.417 | 0.000 |
| deepseek7b | `kv_source` | `positive` | `all_pre_answer` | 1.131 | 0.921 | 0.210 | 1.803 | 0.917 | 0.000 |
| deepseek7b | `kv_o` | `negative` | `all_pre_answer` | 1.363 | 1.085 | 0.278 | 1.655 | 0.583 | 0.000 |
| deepseek7b | `kv_source` | `negative` | `instruction` | 0.988 | 0.723 | 0.265 | 1.133 | 0.333 | 0.000 |
| deepseek7b | `kv_source` | `negative` | `all_pre_answer` | 1.125 | 0.991 | 0.134 | 1.562 | 0.917 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `all_pre_answer` | 2.455 | 2.436 | 0.020 | 2.856 | 0.667 | 0.000 |
| deepseek7b | `kv_o_route` | `negative` | `instruction` | 2.452 | 2.439 | 0.013 | 3.035 | 0.583 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `all_pre_answer` | 2.427 | 2.427 | 0.000 | 3.016 | 0.583 | 0.000 |
| deepseek7b | `route_answer` | `negative` | `instruction` | 2.427 | 2.427 | 0.000 | 3.016 | 0.583 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `all_pre_answer` | 2.427 | 2.427 | 0.000 | 3.016 | 0.583 | 0.000 |
| deepseek7b | `route_answer` | `positive` | `instruction` | 2.427 | 2.427 | 0.000 | 3.016 | 0.583 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `instruction` | 2.436 | 2.439 | -0.003 | 3.004 | 0.583 | 0.000 |
| deepseek7b | `kv_o_route` | `positive` | `all_pre_answer` | 2.344 | 2.400 | -0.056 | 2.824 | 0.667 | 0.000 |

## Top Global Effects

| model | selection | ladder | subspace | source group | cases | global delta | target gain | suppress rate | cross rate | token gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `matched` | `kv_o_route` | `negative` | `instruction` | 6 | 2.565 | 2.857 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `positive` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `instruction` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2.560 | 2.852 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2.438 | 2.958 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2.424 | 2.977 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2.383 | 3.049 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2.383 | 3.029 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `positive` | `instruction` | 6 | 2.195 | 1.435 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `positive` | `instruction` | 6 | 2.190 | 1.440 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `positive` | `all_pre_answer` | 6 | 2.190 | 1.440 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `positive` | `all_pre_answer` | 6 | 2.344 | 1.948 | 0.667 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `negative` | `instruction` | 6 | 1.750 | 1.135 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `negative` | `instruction` | 6 | 1.747 | 1.122 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `o_only` | `negative` | `all_pre_answer` | 6 | 1.747 | 1.122 | 0.833 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_o` | `negative` | `all_pre_answer` | 6 | 1.897 | 1.803 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `positive` | `all_pre_answer` | 6 | 1.928 | 1.793 | 0.417 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 6 | 1.618 | 1.712 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `kv_source` | `negative` | `all_pre_answer` | 6 | 1.453 | 1.651 | 0.250 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 6 | 1.270 | 1.405 | 0.500 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 6 | 0.565 | 1.742 | 0.417 | 0.000 | 0.000 |
| qwen3 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 6 | 0.336 | 1.503 | 0.333 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `instruction` | 6 | 0.647 | 0.397 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 6 | 0.647 | 0.397 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `instruction` | 6 | 0.647 | 0.397 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 6 | 0.647 | 0.397 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `positive` | `all_pre_answer` | 6 | 0.548 | 0.548 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `positive` | `all_pre_answer` | 6 | 0.548 | 0.548 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_source` | `negative` | `all_pre_answer` | 6 | 0.548 | 0.548 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_source` | `negative` | `all_pre_answer` | 6 | 0.548 | 0.548 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `negative` | `all_pre_answer` | 6 | 0.486 | 0.527 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `instruction` | 6 | 0.447 | 0.665 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `negative` | `all_pre_answer` | 6 | 0.447 | 0.665 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `instruction` | 6 | 0.441 | 0.681 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `o_only` | `positive` | `all_pre_answer` | 6 | 0.441 | 0.681 | 0.500 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o` | `positive` | `all_pre_answer` | 6 | 0.473 | 0.546 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `negative` | `instruction` | 6 | 0.434 | 0.663 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `instruction` | 6 | 0.418 | 0.668 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `positive` | `all_pre_answer` | 6 | 0.410 | 0.546 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o` | `negative` | `all_pre_answer` | 6 | 0.402 | 0.559 | 0.500 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `instruction` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `instruction` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `instruction` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `instruction` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 0.030 | 1.895 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `instruction` | 6 | 0.025 | 1.889 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `instruction` | 6 | 0.025 | 1.889 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `instruction` | 6 | 0.025 | 1.889 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `instruction` | 6 | 0.025 | 1.889 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2.455 | 2.856 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2.400 | 2.843 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `instruction` | 6 | 2.452 | 3.035 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `instruction` | 6 | 2.439 | 2.980 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `instruction` | 6 | 2.439 | 2.975 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `instruction` | 6 | 2.436 | 3.004 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2.436 | 2.883 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `instruction` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `instruction` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `instruction` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `instruction` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2.427 | 3.016 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2.344 | 2.824 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `instruction` | 6 | 1.581 | 1.555 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `positive` | `all_pre_answer` | 6 | 1.472 | 1.780 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `instruction` | 6 | 1.511 | 1.558 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_o` | `negative` | `all_pre_answer` | 6 | 1.363 | 1.655 | 0.583 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `positive` | `all_pre_answer` | 6 | 1.131 | 1.803 | 0.917 | 0.000 | 0.000 |
| deepseek7b | `top` | `kv_source` | `negative` | `all_pre_answer` | 6 | 1.125 | 1.562 | 0.917 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `instruction` | 6 | 1.248 | 1.238 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `positive` | `all_pre_answer` | 6 | 1.248 | 1.238 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `instruction` | 6 | 1.193 | 1.287 | 0.417 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_only` | `negative` | `all_pre_answer` | 6 | 1.193 | 1.287 | 0.417 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `negative` | `all_pre_answer` | 6 | 1.085 | 1.486 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `negative` | `all_pre_answer` | 6 | 0.991 | 1.480 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_o` | `positive` | `all_pre_answer` | 6 | 0.971 | 1.398 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `kv_source` | `positive` | `all_pre_answer` | 6 | 0.921 | 1.389 | 0.583 | 0.000 | 0.000 |
