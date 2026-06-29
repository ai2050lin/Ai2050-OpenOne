# Phase 748 Natural Route Suppressor Matrix (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: whole-component donor-recipient deltas measured against route-level max logits.

| model | component | n | donor top1 | target boost | route suppression | route coverage | margin gain | selected prob gain | effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L32:mlp_out | 16 | 0.000 | -0.047 | 2.961 | 3.88 | 0.525 | 0.000 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | 16 | 0.000 | 2.547 | 2.531 | 2.19 | 2.950 | 0.067 | `global_suppressor_margin_candidate` |
| qwen3 | L30:attn_out | 16 | 0.000 | 1.332 | 1.484 | 2.12 | 1.519 | 0.005 | `global_suppressor_margin_candidate` |
| qwen3 | L34:mlp_out | 16 | 0.000 | -1.695 | 1.445 | 2.31 | -1.774 | -0.001 | `harmful_or_competitor_support` |
| qwen3 | L28:attn_out | 16 | 0.000 | 0.531 | 1.281 | 3.62 | 0.793 | 0.002 | `global_suppressor_margin_candidate` |
| qwen3 | L32:attn_out | 16 | 0.000 | 3.066 | 1.180 | 1.94 | 3.105 | 0.011 | `global_suppressor_margin_candidate` |
| qwen3 | L31:mlp_out | 16 | 0.000 | 0.668 | 1.148 | 2.06 | 0.754 | 0.001 | `global_suppressor_margin_candidate` |
| qwen3 | L30:mlp_out | 16 | 0.000 | 0.535 | 0.977 | 2.50 | 0.597 | 0.000 | `global_suppressor_margin_candidate` |
| qwen3 | L28:mlp_out | 16 | 0.000 | 0.590 | 0.906 | 1.56 | 0.548 | 0.001 | `route_specific_suppressor_candidate` |
| qwen3 | L35:attn_out | 16 | 0.000 | 2.344 | 0.219 | 0.44 | 1.324 | 0.003 | `booster_candidate` |
| qwen3 | L35:mlp_out | 16 | 0.000 | 0.121 | 0.125 | 0.44 | -1.505 | -0.001 | `harmful_or_competitor_support` |
| glm4 | L34:attn_out | 16 | 0.875 | 1.504 | 0.525 | 2.50 | 1.447 | 0.171 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | 16 | 0.625 | 0.543 | 0.611 | 2.81 | 0.552 | 0.058 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:mlp_out | 16 | 0.625 | -0.035 | 0.482 | 2.12 | -0.024 | -0.000 | `small_or_no_effect` |
| glm4 | L36:attn_out | 16 | 0.562 | 0.309 | 0.307 | 2.31 | 0.316 | 0.029 | `small_or_no_effect` |
| glm4 | L34:mlp_out | 16 | 0.500 | 0.012 | 0.803 | 3.19 | 0.135 | 0.008 | `small_or_no_effect` |
| glm4 | L39:attn_out | 16 | 0.500 | 0.238 | 0.285 | 1.56 | 0.152 | 0.018 | `small_or_no_effect` |
| glm4 | L37:attn_out | 16 | 0.500 | 0.305 | 0.143 | 1.69 | 0.283 | 0.027 | `small_or_no_effect` |
| glm4 | L38:attn_out | 16 | 0.500 | 0.199 | 0.045 | 0.25 | 0.076 | 0.009 | `small_or_no_effect` |
| glm4 | L36:mlp_out | 16 | 0.312 | -0.441 | 1.332 | 3.12 | -0.443 | -0.110 | `global_suppressor_maintenance_candidate` |
| deepseek7b | L23:attn_out | 16 | 0.250 | 1.762 | 1.637 | 2.88 | 2.042 | 0.134 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | 16 | 0.188 | 1.531 | 2.809 | 2.69 | 2.040 | 0.141 | `global_suppressor_margin_candidate` |
| deepseek7b | L24:attn_out | 16 | 0.125 | 0.832 | 1.211 | 2.69 | 1.024 | 0.052 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:mlp_out | 16 | 0.062 | 0.090 | 1.449 | 3.00 | 0.323 | 0.019 | `global_suppressor_margin_candidate` |
| deepseek7b | L26:mlp_out | 16 | 0.000 | 0.586 | 2.215 | 3.44 | 0.995 | 0.021 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | 16 | 0.000 | 0.414 | 2.055 | 3.12 | 0.732 | 0.030 | `global_suppressor_margin_candidate` |
| deepseek7b | L24:mlp_out | 16 | 0.000 | 0.699 | 1.406 | 2.88 | 0.911 | 0.039 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:mlp_out | 16 | 0.000 | -0.027 | 1.266 | 2.94 | 0.167 | 0.006 | `small_or_no_effect` |
| deepseek7b | L25:attn_out | 16 | 0.000 | 0.535 | 1.180 | 2.62 | 0.687 | 0.015 | `global_suppressor_margin_candidate` |

## Route-Specific Matrix Slices

| model | route | component | n | suppression | positive rate | margin gain | donor top1 | effect counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L28:attn_out | 16 | 0.461 | 1.000 | 0.992 | 0.000 | `{"global_suppressor_margin_candidate": 15, "small_or_no_effect": 1}` |
| qwen3 | recipient_answer | L33:attn_out | 16 | 1.789 | 0.938 | 4.336 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 11, "route_specific_suppressor_candidate": 4}` |
| qwen3 | recipient_answer | L30:attn_out | 16 | 0.906 | 0.875 | 2.238 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 1}` |
| qwen3 | recipient_answer | L32:mlp_out | 16 | 0.586 | 0.750 | 0.539 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 6}` |
| qwen3 | recipient_answer | L28:mlp_out | 16 | 0.445 | 0.750 | 1.035 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 5, "harmful_or_competitor_support": 2, "route_specific_suppressor_candidate": 5, "small_or_no_effect": 2}` |
| qwen3 | recipient_answer | L30:mlp_out | 16 | 0.156 | 0.688 | 0.691 | 0.000 | `{"global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 6}` |
| qwen3 | recipient_answer | L31:mlp_out | 16 | 0.289 | 0.625 | 0.957 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 3, "small_or_no_effect": 3}` |
| qwen3 | recipient_answer | L32:attn_out | 16 | 0.211 | 0.562 | 3.277 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 12, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L35:attn_out | 16 | -0.469 | 0.250 | 1.875 | 0.000 | `{"booster_candidate": 12, "global_suppressor_margin_candidate": 2, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L34:mlp_out | 16 | -0.938 | 0.250 | -2.633 | 0.000 | `{"global_suppressor_margin_candidate": 2, "harmful_or_competitor_support": 12, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L35:mlp_out | 16 | -2.844 | 0.000 | -2.723 | 0.000 | `{"booster_candidate": 4, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 9, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 1}` |
| qwen3 | punctuation_or_stop | L32:mlp_out | 16 | 0.570 | 0.875 | 0.523 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 6}` |
| qwen3 | punctuation_or_stop | L34:mlp_out | 16 | 0.430 | 0.750 | -1.266 | 0.000 | `{"global_suppressor_margin_candidate": 2, "harmful_or_competitor_support": 12, "route_specific_suppressor_candidate": 2}` |
| qwen3 | punctuation_or_stop | L28:attn_out | 16 | 0.094 | 0.562 | 0.625 | 0.000 | `{"global_suppressor_margin_candidate": 15, "small_or_no_effect": 1}` |
| qwen3 | punctuation_or_stop | L32:attn_out | 16 | 0.094 | 0.500 | 3.160 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 12, "route_specific_suppressor_candidate": 2}` |
| qwen3 | punctuation_or_stop | L30:attn_out | 16 | 0.023 | 0.500 | 1.355 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 1}` |
| qwen3 | punctuation_or_stop | L30:mlp_out | 16 | -0.023 | 0.438 | 0.512 | 0.000 | `{"global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 6}` |
| qwen3 | punctuation_or_stop | L31:mlp_out | 16 | -0.008 | 0.312 | 0.660 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 3, "small_or_no_effect": 3}` |
| qwen3 | punctuation_or_stop | L33:attn_out | 16 | -0.164 | 0.188 | 2.383 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 11, "route_specific_suppressor_candidate": 4}` |
| qwen3 | punctuation_or_stop | L35:mlp_out | 16 | -0.523 | 0.188 | -0.402 | 0.000 | `{"booster_candidate": 4, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 9, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 1}` |
| qwen3 | punctuation_or_stop | L28:mlp_out | 16 | -0.320 | 0.125 | 0.270 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 5, "harmful_or_competitor_support": 2, "route_specific_suppressor_candidate": 5, "small_or_no_effect": 2}` |
| qwen3 | punctuation_or_stop | L35:attn_out | 16 | -1.047 | 0.125 | 1.297 | 0.000 | `{"booster_candidate": 12, "global_suppressor_margin_candidate": 2, "route_specific_suppressor_candidate": 2}` |
| qwen3 | other_vocab | L32:mlp_out | 16 | 0.516 | 1.000 | 0.469 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 6}` |
| qwen3 | other_vocab | L28:attn_out | 16 | 0.188 | 0.750 | 0.719 | 0.000 | `{"global_suppressor_margin_candidate": 15, "small_or_no_effect": 1}` |
| glm4 | recipient_answer | L34:mlp_out | 14 | 0.297 | 1.000 | 0.310 | 0.571 | `{"global_suppressor_closure_candidate": 2, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 4, "mixed_boost_global_suppressor_maintenance_candidate": 1, "small_or_no_effect": 3}` |
| glm4 | recipient_answer | L35:mlp_out | 14 | 0.125 | 0.857 | 0.129 | 0.714 | `{"global_suppressor_maintenance_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 3, "small_or_no_effect": 5}` |
| glm4 | recipient_answer | L35:attn_out | 14 | 0.205 | 0.714 | 0.603 | 0.714 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 6, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| glm4 | recipient_answer | L36:attn_out | 14 | 0.054 | 0.571 | 0.286 | 0.643 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 2, "global_suppressor_margin_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 3, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | recipient_answer | L36:mlp_out | 14 | -0.790 | 0.571 | -1.312 | 0.357 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 4, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 2}` |
| glm4 | recipient_answer | L34:attn_out | 14 | 0.062 | 0.429 | 1.384 | 1.000 | `{"booster_candidate": 4, "booster_maintenance_candidate": 1, "global_suppressor_maintenance_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 4, "mixed_boost_global_suppressor_maintenance_candidate": 4}` |
| glm4 | recipient_answer | L39:attn_out | 14 | -0.018 | 0.429 | 0.228 | 0.571 | `{"booster_candidate": 2, "global_suppressor_closure_candidate": 2, "small_or_no_effect": 10}` |
| glm4 | recipient_answer | L37:attn_out | 14 | -0.009 | 0.143 | 0.321 | 0.571 | `{"booster_candidate": 3, "global_suppressor_maintenance_candidate": 3, "small_or_no_effect": 8}` |
| glm4 | recipient_answer | L38:attn_out | 14 | -0.062 | 0.071 | 0.147 | 0.571 | `{"small_or_no_effect": 14}` |
| glm4 | punctuation_or_stop | L36:attn_out | 14 | 0.156 | 0.786 | 0.455 | 0.643 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 2, "global_suppressor_margin_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 3, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | punctuation_or_stop | L36:mlp_out | 14 | 0.116 | 0.714 | -0.482 | 0.357 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 4, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 2}` |
| glm4 | punctuation_or_stop | L35:attn_out | 14 | 0.087 | 0.714 | 0.681 | 0.714 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 6, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| glm4 | punctuation_or_stop | L34:mlp_out | 14 | 0.100 | 0.571 | -0.029 | 0.571 | `{"global_suppressor_closure_candidate": 2, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 1, "small_or_no_effect": 5}` |
| glm4 | punctuation_or_stop | L37:attn_out | 14 | 0.038 | 0.429 | 0.355 | 0.571 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 3, "small_or_no_effect": 9}` |
| glm4 | punctuation_or_stop | L34:attn_out | 14 | 0.036 | 0.429 | 1.451 | 0.857 | `{"booster_candidate": 4, "booster_maintenance_candidate": 1, "global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_closure_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 4}` |
| glm4 | punctuation_or_stop | L35:mlp_out | 14 | -0.058 | 0.286 | -0.098 | 0.714 | `{"global_suppressor_maintenance_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 3, "small_or_no_effect": 5}` |
| glm4 | punctuation_or_stop | L39:attn_out | 14 | -0.062 | 0.286 | 0.183 | 0.571 | `{"booster_candidate": 2, "global_suppressor_closure_candidate": 2, "small_or_no_effect": 10}` |
| glm4 | punctuation_or_stop | L38:attn_out | 14 | -0.067 | 0.214 | 0.134 | 0.571 | `{"small_or_no_effect": 14}` |
| glm4 | other_vocab | L34:attn_out | 16 | 0.105 | 0.688 | 1.609 | 0.875 | `{"booster_candidate": 4, "booster_maintenance_candidate": 1, "global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_closure_candidate": 4, "mixed_boost_global_suppressor_maintenance_candidate": 4}` |
| glm4 | other_vocab | L35:attn_out | 16 | 0.055 | 0.688 | 0.598 | 0.625 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 4, "mixed_boost_global_suppressor_maintenance_candidate": 6, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| glm4 | other_vocab | L34:mlp_out | 16 | 0.074 | 0.625 | 0.086 | 0.500 | `{"global_suppressor_closure_candidate": 2, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 4, "mixed_boost_global_suppressor_maintenance_candidate": 1, "small_or_no_effect": 5}` |
| glm4 | other_vocab | L36:attn_out | 16 | 0.043 | 0.500 | 0.352 | 0.562 | `{"booster_candidate": 4, "global_suppressor_maintenance_candidate": 2, "global_suppressor_margin_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 3, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | other_vocab | L36:mlp_out | 16 | 0.145 | 0.438 | -0.297 | 0.312 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 4, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | other_vocab | L37:attn_out | 16 | 0.008 | 0.375 | 0.312 | 0.500 | `{"booster_candidate": 3, "global_suppressor_maintenance_candidate": 3, "small_or_no_effect": 10}` |
| deepseek7b | recipient_answer | L25:mlp_out | 6 | 1.208 | 1.000 | 1.229 | 0.000 | `{"global_suppressor_margin_candidate": 5, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L24:mlp_out | 6 | 0.833 | 1.000 | 1.375 | 0.000 | `{"global_suppressor_margin_candidate": 5, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L22:attn_out | 6 | 3.062 | 0.833 | 4.073 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L23:attn_out | 6 | 1.458 | 0.833 | 2.969 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L26:mlp_out | 6 | 0.823 | 0.833 | 1.802 | 0.000 | `{"global_suppressor_margin_candidate": 4, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L22:mlp_out | 6 | 0.396 | 0.833 | 0.125 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 1, "small_or_no_effect": 4}` |
| deepseek7b | recipient_answer | L23:mlp_out | 6 | 0.396 | 0.833 | 0.375 | 0.167 | `{"global_suppressor_margin_candidate": 3, "harmful_or_competitor_support": 1, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L25:attn_out | 6 | 0.667 | 0.667 | 1.135 | 0.000 | `{"global_suppressor_margin_candidate": 5, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L24:attn_out | 6 | 0.531 | 0.667 | 1.406 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 3, "route_specific_suppressor_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L23:mlp_out | 16 | 0.531 | 0.875 | 0.621 | 0.062 | `{"global_suppressor_margin_candidate": 7, "harmful_or_competitor_support": 1, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 6}` |
| deepseek7b | punctuation_or_stop | L25:mlp_out | 16 | 0.504 | 0.875 | 0.918 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 4, "small_or_no_effect": 3}` |
| deepseek7b | punctuation_or_stop | L22:mlp_out | 16 | 0.324 | 0.812 | 0.297 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 4, "harmful_or_competitor_support": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 8}` |
| deepseek7b | punctuation_or_stop | L24:mlp_out | 16 | 0.312 | 0.812 | 1.012 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 12, "small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L25:attn_out | 16 | 0.195 | 0.812 | 0.730 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 10, "harmful_or_competitor_support": 2, "small_or_no_effect": 2}` |
| deepseek7b | punctuation_or_stop | L22:attn_out | 16 | 0.379 | 0.750 | 1.910 | 0.188 | `{"global_suppressor_margin_candidate": 11, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 2}` |
| deepseek7b | punctuation_or_stop | L26:mlp_out | 16 | 0.312 | 0.750 | 0.898 | 0.000 | `{"global_suppressor_margin_candidate": 12, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 3}` |
| deepseek7b | punctuation_or_stop | L24:attn_out | 16 | 0.203 | 0.750 | 1.035 | 0.125 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 10, "mixed_boost_global_suppressor_closure_candidate": 2, "route_specific_suppressor_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L23:attn_out | 16 | 0.125 | 0.625 | 1.887 | 0.250 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 11, "mixed_boost_global_suppressor_closure_candidate": 3, "route_specific_suppressor_candidate": 1}` |
| deepseek7b | other_vocab | L26:mlp_out | 16 | 0.469 | 0.875 | 1.055 | 0.000 | `{"global_suppressor_margin_candidate": 12, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 3}` |
| deepseek7b | other_vocab | L24:mlp_out | 16 | 0.199 | 0.750 | 0.898 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 12, "small_or_no_effect": 1}` |
| deepseek7b | other_vocab | L25:mlp_out | 16 | 0.281 | 0.625 | 0.695 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 4, "small_or_no_effect": 3}` |
| deepseek7b | other_vocab | L23:attn_out | 16 | 0.234 | 0.625 | 1.996 | 0.250 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 11, "mixed_boost_global_suppressor_closure_candidate": 3, "route_specific_suppressor_candidate": 1}` |
| deepseek7b | other_vocab | L25:attn_out | 16 | 0.164 | 0.625 | 0.699 | 0.000 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 10, "harmful_or_competitor_support": 2, "small_or_no_effect": 2}` |
| deepseek7b | other_vocab | L23:mlp_out | 16 | 0.137 | 0.562 | 0.227 | 0.062 | `{"global_suppressor_margin_candidate": 7, "harmful_or_competitor_support": 1, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 6}` |

## Strict Interpretation

- `target boost` measures constructive force toward the donor answer.
- `route suppression` measures selective force against measured route maxima; positive values are suppressor evidence.
- `route coverage` estimates whether the component is route-specific or broad/global over measured top-k routes.
- This is still whole-component donor-recipient delta evidence; it does not yet prove a natural neuron-level suppressor or training-origin suppressor.

Atlas graph: nodes=30 edges=27
