# Phase 799 Blocker-Field Causal Suppressor Localization (main)

- Status: `complete`
- Boundary: scores candidate fibers by target gain, identity-anchor improvement, baseline blocker suppression, and new-blocker penalty.
- This phase gives suppressor candidates, not final token closure.

## By Model

| model | rows | cases | target gain | blocker suppression | target-relative lift | new blocker rate | resolved rate | anchor gap | token gain | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 48 | 3 | 3.153 | 0.820 | 3.973 | 0.032 | 0.812 | 2.314 | 0.000 | 1.099 |
| glm4 | 30 | 3 | 1.573 | 0.497 | 2.070 | 0.053 | 0.712 | 0.227 | 0.000 | 0.543 |
| deepseek7b | 48 | 3 | 3.271 | -0.580 | 2.691 | 0.134 | 0.686 | 2.028 | 0.000 | 0.816 |

## Top Suppressor Candidates

| model | component | selection | ladder | source group | rows | target gain | blocker suppression | target-relative lift | new rate | resolved rate | anchor gap | score |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attn:L35` | `top` | `route_answer` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `matched` | `route_answer` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `top` | `route_answer` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `matched` | `route_answer` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L35` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 3.500 | 0.969 | 4.469 | 0.041 | 0.854 | 2.542 | 1.255 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `all_pre_answer` | 3 | 2.760 | 0.756 | 3.516 | 0.023 | 0.770 | 2.260 | 0.979 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `all_pre_answer` | 3 | 2.760 | 0.756 | 3.516 | 0.023 | 0.770 | 2.260 | 0.979 |
| qwen3 | `attn:L34` | `top` | `route_answer` | `all_pre_answer` | 3 | 2.760 | 0.756 | 3.516 | 0.023 | 0.770 | 2.260 | 0.979 |
| qwen3 | `attn:L34` | `matched` | `route_answer` | `all_pre_answer` | 3 | 2.760 | 0.756 | 3.516 | 0.023 | 0.770 | 2.260 | 0.979 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 2.823 | 0.670 | 3.493 | 0.024 | 0.775 | 2.031 | 0.940 |
| qwen3 | `attn:L34` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 2.906 | 0.598 | 3.504 | 0.023 | 0.776 | 1.990 | 0.915 |
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 2.875 | 0.541 | 3.416 | 0.023 | 0.765 | 1.875 | 0.890 |
| qwen3 | `attn:L34` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 2.802 | 0.546 | 3.348 | 0.026 | 0.756 | 1.760 | 0.877 |
| glm4 | `attn:L33` | `top` | `route_answer` | `all_pre_answer` | 3 | 1.911 | 0.523 | 2.434 | 0.022 | 0.804 | 0.141 | 0.634 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `all_pre_answer` | 3 | 1.911 | 0.523 | 2.434 | 0.022 | 0.804 | 0.141 | 0.634 |
| glm4 | `attn:L33` | `top` | `route_answer` | `all_pre_answer` | 3 | 1.911 | 0.523 | 2.434 | 0.022 | 0.804 | 0.141 | 0.634 |
| glm4 | `attn:L33` | `matched` | `route_answer` | `all_pre_answer` | 3 | 1.911 | 0.523 | 2.434 | 0.022 | 0.804 | 0.141 | 0.634 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 1.922 | 0.504 | 2.426 | 0.029 | 0.797 | 0.130 | 0.634 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 1.922 | 0.504 | 2.426 | 0.029 | 0.797 | 0.130 | 0.634 |
| glm4 | `attn:L33` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 1.922 | 0.504 | 2.426 | 0.029 | 0.797 | 0.130 | 0.634 |
| glm4 | `attn:L33` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 1.922 | 0.504 | 2.426 | 0.029 | 0.797 | 0.130 | 0.634 |
| glm4 | `route_only:L38` | `top` | `route_answer` | `all_pre_answer` | 3 | 0.198 | 0.429 | 0.627 | 0.159 | 0.356 | 0.594 | 0.180 |
| glm4 | `route_only:L38` | `matched` | `route_answer` | `all_pre_answer` | 3 | 0.198 | 0.429 | 0.627 | 0.159 | 0.356 | 0.594 | 0.180 |
| deepseek7b | `attn:L23` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 3.469 | 0.002 | 3.470 | 0.000 | 0.911 | 2.219 | 1.484 |
| deepseek7b | `attn:L23` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 3.344 | -0.001 | 3.343 | 0.000 | 0.883 | 1.844 | 1.268 |
| deepseek7b | `attn:L23` | `top` | `route_answer` | `all_pre_answer` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L23` | `matched` | `route_answer` | `all_pre_answer` | 1 | 3.031 | 0.040 | 3.071 | 0.000 | 0.877 | 1.906 | 1.204 |
| deepseek7b | `attn:L27` | `top` | `route_answer` | `all_pre_answer` | 1 | 7.176 | -0.543 | 6.633 | 0.011 | 0.992 | -0.293 | 1.153 |
| deepseek7b | `attn:L27` | `top` | `kv_o_route` | `all_pre_answer` | 1 | 7.176 | -0.543 | 6.633 | 0.011 | 0.992 | -0.293 | 1.153 |
| deepseek7b | `attn:L27` | `matched` | `route_answer` | `all_pre_answer` | 1 | 7.176 | -0.543 | 6.633 | 0.011 | 0.992 | -0.293 | 1.153 |
| deepseek7b | `attn:L27` | `matched` | `kv_o_route` | `all_pre_answer` | 1 | 7.176 | -0.543 | 6.633 | 0.011 | 0.992 | -0.293 | 1.153 |
| deepseek7b | `attn:L26` | `top` | `route_answer` | `all_pre_answer` | 2 | 2.492 | -0.295 | 2.197 | 0.011 | 0.699 | 2.211 | 0.865 |
| deepseek7b | `attn:L26` | `top` | `kv_o_route` | `all_pre_answer` | 2 | 2.492 | -0.295 | 2.197 | 0.011 | 0.699 | 2.211 | 0.865 |
| deepseek7b | `attn:L26` | `matched` | `route_answer` | `all_pre_answer` | 2 | 2.492 | -0.295 | 2.197 | 0.011 | 0.699 | 2.211 | 0.865 |
| deepseek7b | `attn:L26` | `matched` | `kv_o_route` | `all_pre_answer` | 2 | 2.492 | -0.295 | 2.197 | 0.011 | 0.699 | 2.211 | 0.865 |
| deepseek7b | `attn:L27` | `top` | `route_answer` | `all_pre_answer` | 2 | 4.564 | -0.586 | 3.978 | 0.016 | 0.756 | 1.111 | 0.839 |
| deepseek7b | `attn:L27` | `top` | `kv_o_route` | `all_pre_answer` | 2 | 4.564 | -0.586 | 3.978 | 0.016 | 0.756 | 1.111 | 0.839 |
| deepseek7b | `attn:L27` | `matched` | `route_answer` | `all_pre_answer` | 2 | 4.564 | -0.586 | 3.978 | 0.016 | 0.756 | 1.111 | 0.839 |
| deepseek7b | `attn:L27` | `matched` | `kv_o_route` | `all_pre_answer` | 2 | 4.564 | -0.586 | 3.978 | 0.016 | 0.756 | 1.111 | 0.839 |
| deepseek7b | `attn:L19` | `top` | `route_answer` | `all_pre_answer` | 3 | 2.577 | -0.797 | 1.780 | 0.248 | 0.591 | 2.629 | 0.679 |
| deepseek7b | `attn:L19` | `matched` | `route_answer` | `all_pre_answer` | 3 | 2.577 | -0.797 | 1.780 | 0.248 | 0.591 | 2.629 | 0.679 |
| deepseek7b | `attn:L19` | `top` | `route_answer` | `all_pre_answer` | 3 | 2.577 | -0.797 | 1.780 | 0.248 | 0.591 | 2.629 | 0.679 |
| deepseek7b | `attn:L19` | `matched` | `route_answer` | `all_pre_answer` | 3 | 2.577 | -0.797 | 1.780 | 0.248 | 0.591 | 2.629 | 0.679 |
| deepseek7b | `attn:L19` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 2.405 | -0.761 | 1.643 | 0.263 | 0.565 | 2.671 | 0.648 |
| deepseek7b | `attn:L19` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 2.355 | -0.770 | 1.585 | 0.264 | 0.561 | 2.715 | 0.640 |
| deepseek7b | `attn:L19` | `matched` | `kv_o_route` | `all_pre_answer` | 3 | 2.355 | -0.770 | 1.585 | 0.264 | 0.561 | 2.715 | 0.640 |
| deepseek7b | `attn:L19` | `top` | `kv_o_route` | `all_pre_answer` | 3 | 2.236 | -0.742 | 1.494 | 0.278 | 0.544 | 2.741 | 0.619 |
