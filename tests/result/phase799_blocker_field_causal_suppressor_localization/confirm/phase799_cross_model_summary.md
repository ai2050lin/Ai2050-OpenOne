# Phase 799 Blocker-Field Causal Suppressor Localization (confirm)

- Status: `complete`
- Boundary: scores candidate fibers by target gain, identity-anchor improvement, baseline blocker suppression, and new-blocker penalty.
- This phase gives suppressor candidates, not final token closure.

## By Model

| model | rows | cases | target gain | blocker suppression | target-relative lift | new blocker rate | resolved rate | anchor gap | token gain | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 192 | 6 | 2.890 | 0.662 | 3.552 | 0.041 | 0.728 | 2.439 | 0.000 | 1.265 |
| glm4 | 120 | 6 | 1.593 | 0.444 | 2.037 | 0.045 | 0.684 | 0.147 | 0.000 | 0.479 |
| deepseek7b | 192 | 6 | 2.970 | -0.522 | 2.448 | 0.123 | 0.664 | 2.097 | 0.000 | 0.821 |

## Top Suppressor Candidates

| model | component | selection | ladder | source group | rows | target gain | blocker suppression | target-relative lift | new rate | resolved rate | anchor gap | score |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `instruction` | 6 | 2.984 | 0.491 | 3.475 | 0.022 | 0.716 | 2.839 | 1.394 |
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `instruction` | 6 | 2.974 | 0.493 | 3.467 | 0.022 | 0.716 | 2.828 | 1.391 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `instruction` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `all_pre_answer` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `instruction` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `all_pre_answer` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `instruction` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `all_pre_answer` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `instruction` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `all_pre_answer` | 6 | 2.974 | 0.494 | 3.468 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `instruction` | 6 | 2.974 | 0.493 | 3.467 | 0.022 | 0.717 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `instruction` | 6 | 2.974 | 0.491 | 3.465 | 0.022 | 0.716 | 2.828 | 1.390 |
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `all_pre_answer` | 6 | 3.370 | 0.228 | 3.598 | 0.016 | 0.710 | 2.474 | 1.380 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `all_pre_answer` | 6 | 3.188 | 0.294 | 3.482 | 0.027 | 0.715 | 2.583 | 1.367 |
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `all_pre_answer` | 6 | 3.328 | 0.221 | 3.549 | 0.026 | 0.711 | 2.474 | 1.361 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `all_pre_answer` | 6 | 3.224 | 0.291 | 3.515 | 0.025 | 0.713 | 2.557 | 1.351 |
| qwen3 | `attn:L35` | `top` | `route_answer` | `instruction` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `top` | `kv_o_route` | `instruction` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `top` | `route_answer` | `all_pre_answer` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `top` | `kv_o_route` | `all_pre_answer` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `matched` | `route_answer` | `instruction` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `matched` | `kv_o_route` | `instruction` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `matched` | `route_answer` | `all_pre_answer` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| qwen3 | `attn:L35` | `matched` | `kv_o_route` | `all_pre_answer` | 6 | 2.729 | 0.889 | 3.618 | 0.060 | 0.740 | 2.125 | 1.145 |
| glm4 | `attn:L33` | `top` | `route_answer` | `instruction` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `top` | `route_answer` | `all_pre_answer` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `instruction` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `all_pre_answer` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `top` | `route_answer` | `instruction` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `top` | `route_answer` | `all_pre_answer` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `instruction` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `all_pre_answer` | 6 | 1.895 | 0.465 | 2.360 | 0.028 | 0.764 | 0.030 | 0.545 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `all_pre_answer` | 6 | 1.889 | 0.454 | 2.344 | 0.034 | 0.752 | 0.004 | 0.545 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `all_pre_answer` | 6 | 1.889 | 0.454 | 2.344 | 0.034 | 0.752 | 0.004 | 0.545 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `all_pre_answer` | 6 | 1.889 | 0.454 | 2.344 | 0.034 | 0.752 | 0.004 | 0.545 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `all_pre_answer` | 6 | 1.889 | 0.454 | 2.344 | 0.034 | 0.752 | 0.004 | 0.545 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `instruction` | 6 | 1.889 | 0.462 | 2.351 | 0.028 | 0.763 | 0.025 | 0.542 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `instruction` | 6 | 1.889 | 0.462 | 2.351 | 0.028 | 0.763 | 0.025 | 0.542 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `instruction` | 6 | 1.889 | 0.462 | 2.351 | 0.028 | 0.763 | 0.025 | 0.542 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `instruction` | 6 | 1.889 | 0.462 | 2.351 | 0.028 | 0.763 | 0.025 | 0.542 |
| glm4 | `route_only:L38` | `top` | `route_answer` | `instruction` | 6 | 0.397 | 0.373 | 0.771 | 0.109 | 0.376 | 0.647 | 0.216 |
| glm4 | `route_only:L38` | `top` | `route_answer` | `all_pre_answer` | 6 | 0.397 | 0.373 | 0.771 | 0.109 | 0.376 | 0.647 | 0.216 |
| glm4 | `route_only:L38` | `matched` | `route_answer` | `instruction` | 6 | 0.397 | 0.373 | 0.771 | 0.109 | 0.376 | 0.647 | 0.216 |
| glm4 | `route_only:L38` | `matched` | `route_answer` | `all_pre_answer` | 6 | 0.397 | 0.373 | 0.771 | 0.109 | 0.376 | 0.647 | 0.216 |
| deepseek7b | `attn:L26` | `top` | `route_answer` | `instruction` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `top` | `kv_o_route` | `instruction` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `top` | `route_answer` | `all_pre_answer` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `matched` | `route_answer` | `instruction` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `matched` | `kv_o_route` | `instruction` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `matched` | `route_answer` | `all_pre_answer` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L26` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 5.875 | -1.022 | 4.853 | 0.000 | 0.942 | 2.500 | 2.496 |
| deepseek7b | `attn:L23` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 3.406 | 0.021 | 3.428 | 0.000 | 0.905 | 2.281 | 1.467 |
| deepseek7b | `attn:L23` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 3.531 | 0.005 | 3.536 | 0.000 | 0.911 | 2.031 | 1.452 |
| deepseek7b | `attn:L23` | `top` | `kv_o_route` | `instruction` | 1 | 3.531 | 0.066 | 3.597 | 0.000 | 0.905 | 2.031 | 1.447 |
| deepseek7b | `attn:L23` | `top` | `route_answer` | `instruction` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L23` | `top` | `route_answer` | `all_pre_answer` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L23` | `matched` | `route_answer` | `instruction` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L23` | `matched` | `route_answer` | `all_pre_answer` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L26` | `top` | `route_answer` | `instruction` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `top` | `kv_o_route` | `instruction` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `top` | `route_answer` | `all_pre_answer` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `top` | `kv_o_route` | `all_pre_answer` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `matched` | `route_answer` | `instruction` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `matched` | `kv_o_route` | `instruction` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `matched` | `route_answer` | `all_pre_answer` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L26` | `matched` | `kv_o_route` | `all_pre_answer` | 4 | 3.152 | -0.561 | 2.591 | 0.018 | 0.703 | 2.262 | 1.148 |
| deepseek7b | `attn:L23` | `matched` | `kv_o_route` | `instruction` | 1 | 2.781 | 0.047 | 2.829 | 0.000 | 0.832 | 1.906 | 1.068 |
