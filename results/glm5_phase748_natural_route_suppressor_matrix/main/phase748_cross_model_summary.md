# Phase 748 Natural Route Suppressor Matrix (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: whole-component donor-recipient deltas measured against route-level max logits.

| model | component | n | donor top1 | target boost | route suppression | route coverage | margin gain | selected prob gain | effect |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L32:attn_out | 12 | 0.167 | 2.536 | 1.521 | 1.83 | 2.583 | 0.089 | `global_suppressor_margin_candidate` |
| qwen3 | L32:mlp_out | 12 | 0.000 | -0.047 | 3.094 | 3.75 | 0.509 | 0.007 | `global_suppressor_margin_candidate` |
| qwen3 | L33:attn_out | 12 | 0.000 | 2.948 | 2.844 | 2.75 | 3.404 | 0.069 | `global_suppressor_margin_candidate` |
| qwen3 | L34:mlp_out | 12 | 0.000 | -1.479 | 2.177 | 3.08 | -1.294 | -0.008 | `harmful_or_competitor_support` |
| qwen3 | L30:attn_out | 12 | 0.000 | 1.349 | 1.208 | 1.83 | 1.435 | 0.011 | `global_suppressor_margin_candidate` |
| qwen3 | L28:attn_out | 12 | 0.000 | 0.406 | 1.042 | 3.08 | 0.564 | 0.010 | `global_suppressor_margin_candidate` |
| qwen3 | L31:mlp_out | 12 | 0.000 | 0.693 | 1.042 | 2.00 | 0.702 | 0.010 | `global_suppressor_margin_candidate` |
| qwen3 | L30:mlp_out | 12 | 0.000 | 0.438 | 0.906 | 2.33 | 0.469 | 0.001 | `global_suppressor_margin_candidate` |
| qwen3 | L28:mlp_out | 12 | 0.000 | 0.620 | 0.875 | 1.50 | 0.572 | 0.005 | `route_specific_suppressor_candidate` |
| qwen3 | L35:attn_out | 12 | 0.000 | 2.609 | 0.062 | 0.17 | 1.468 | 0.013 | `booster_candidate` |
| qwen3 | L35:mlp_out | 12 | 0.000 | 0.193 | 0.000 | 0.00 | -1.642 | -0.002 | `harmful_or_competitor_support` |
| glm4 | L34:attn_out | 12 | 0.833 | 1.896 | 0.526 | 2.17 | 1.711 | 0.177 | `mixed_boost_global_suppressor_maintenance_candidate` |
| glm4 | L35:attn_out | 12 | 0.667 | 0.552 | 0.497 | 2.67 | 0.543 | 0.054 | `small_or_no_effect` |
| glm4 | L37:attn_out | 12 | 0.667 | 0.359 | 0.135 | 1.83 | 0.325 | 0.032 | `small_or_no_effect` |
| glm4 | L36:attn_out | 12 | 0.583 | 0.292 | 0.242 | 2.08 | 0.280 | 0.024 | `small_or_no_effect` |
| glm4 | L34:mlp_out | 12 | 0.500 | -0.161 | 0.826 | 2.67 | -0.035 | -0.009 | `small_or_no_effect` |
| glm4 | L35:mlp_out | 12 | 0.500 | 0.000 | 0.375 | 1.92 | -0.009 | 0.002 | `small_or_no_effect` |
| glm4 | L39:attn_out | 12 | 0.500 | 0.260 | 0.224 | 1.00 | 0.147 | 0.016 | `small_or_no_effect` |
| glm4 | L38:attn_out | 12 | 0.417 | 0.193 | 0.008 | 0.08 | 0.055 | 0.004 | `small_or_no_effect` |
| glm4 | L36:mlp_out | 12 | 0.333 | -0.344 | 1.070 | 3.08 | -0.375 | -0.099 | `global_suppressor_maintenance_candidate` |
| deepseek7b | L23:attn_out | 12 | 0.250 | 1.620 | 2.057 | 3.17 | 1.929 | 0.115 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:attn_out | 12 | 0.167 | 1.208 | 3.005 | 2.83 | 1.701 | 0.093 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:mlp_out | 12 | 0.167 | 0.354 | 2.193 | 3.33 | 0.722 | 0.037 | `global_suppressor_margin_candidate` |
| deepseek7b | L26:mlp_out | 12 | 0.167 | 0.505 | 2.099 | 3.08 | 0.764 | 0.021 | `global_suppressor_margin_candidate` |
| deepseek7b | L24:attn_out | 12 | 0.167 | 0.964 | 1.839 | 3.25 | 1.234 | 0.082 | `global_suppressor_margin_candidate` |
| deepseek7b | L24:mlp_out | 12 | 0.167 | 0.771 | 1.734 | 3.00 | 1.022 | 0.054 | `global_suppressor_margin_candidate` |
| deepseek7b | L23:mlp_out | 12 | 0.083 | 0.156 | 1.516 | 3.08 | 0.407 | 0.011 | `global_suppressor_margin_candidate` |
| deepseek7b | L25:attn_out | 12 | 0.083 | 0.339 | 1.250 | 2.83 | 0.457 | -0.008 | `global_suppressor_margin_candidate` |
| deepseek7b | L22:mlp_out | 12 | 0.083 | -0.047 | 0.953 | 2.75 | 0.104 | 0.000 | `small_or_no_effect` |

## Route-Specific Matrix Slices

| model | route | component | n | suppression | positive rate | margin gain | donor top1 | effect counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | recipient_answer | L33:attn_out | 12 | 1.594 | 1.000 | 4.542 | 0.000 | `{"global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L32:mlp_out | 12 | 0.646 | 0.833 | 0.599 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 2}` |
| qwen3 | recipient_answer | L28:attn_out | 12 | 0.229 | 0.833 | 0.635 | 0.000 | `{"global_suppressor_margin_candidate": 10, "small_or_no_effect": 2}` |
| qwen3 | recipient_answer | L30:attn_out | 12 | 0.625 | 0.667 | 1.974 | 0.000 | `{"booster_candidate": 4, "global_suppressor_margin_candidate": 5, "route_specific_suppressor_candidate": 3}` |
| qwen3 | recipient_answer | L28:mlp_out | 12 | 0.292 | 0.583 | 0.911 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 3, "route_specific_suppressor_candidate": 5, "small_or_no_effect": 3}` |
| qwen3 | recipient_answer | L32:attn_out | 12 | 0.260 | 0.500 | 2.797 | 0.167 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 2, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L31:mlp_out | 12 | 0.125 | 0.417 | 0.818 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 5, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 2}` |
| qwen3 | recipient_answer | L30:mlp_out | 12 | -0.094 | 0.417 | 0.344 | 0.000 | `{"global_suppressor_margin_candidate": 6, "route_specific_suppressor_candidate": 4, "small_or_no_effect": 2}` |
| qwen3 | recipient_answer | L34:mlp_out | 12 | -0.625 | 0.333 | -2.104 | 0.000 | `{"global_suppressor_margin_candidate": 4, "harmful_or_competitor_support": 8}` |
| qwen3 | recipient_answer | L35:attn_out | 12 | -0.792 | 0.167 | 1.818 | 0.000 | `{"booster_candidate": 10, "route_specific_suppressor_candidate": 2}` |
| qwen3 | recipient_answer | L35:mlp_out | 12 | -2.781 | 0.000 | -2.589 | 0.000 | `{"booster_candidate": 4, "harmful_or_competitor_support": 8}` |
| qwen3 | punctuation_or_stop | L34:mlp_out | 12 | 0.490 | 0.833 | -0.990 | 0.000 | `{"global_suppressor_margin_candidate": 4, "harmful_or_competitor_support": 8}` |
| qwen3 | punctuation_or_stop | L32:mlp_out | 12 | 0.271 | 0.583 | 0.224 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 2}` |
| qwen3 | punctuation_or_stop | L32:attn_out | 12 | 0.062 | 0.500 | 2.599 | 0.167 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 2, "route_specific_suppressor_candidate": 2}` |
| qwen3 | punctuation_or_stop | L30:mlp_out | 12 | 0.031 | 0.500 | 0.469 | 0.000 | `{"global_suppressor_margin_candidate": 6, "route_specific_suppressor_candidate": 4, "small_or_no_effect": 2}` |
| qwen3 | punctuation_or_stop | L28:attn_out | 12 | -0.021 | 0.417 | 0.385 | 0.000 | `{"global_suppressor_margin_candidate": 10, "small_or_no_effect": 2}` |
| qwen3 | punctuation_or_stop | L33:attn_out | 12 | -0.156 | 0.333 | 2.792 | 0.000 | `{"global_suppressor_margin_candidate": 10, "route_specific_suppressor_candidate": 2}` |
| qwen3 | punctuation_or_stop | L31:mlp_out | 12 | -0.073 | 0.250 | 0.620 | 0.000 | `{"booster_candidate": 3, "global_suppressor_margin_candidate": 5, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 2}` |
| qwen3 | punctuation_or_stop | L30:attn_out | 12 | -0.083 | 0.250 | 1.266 | 0.000 | `{"booster_candidate": 4, "global_suppressor_margin_candidate": 5, "route_specific_suppressor_candidate": 3}` |
| qwen3 | punctuation_or_stop | L28:mlp_out | 12 | -0.396 | 0.000 | 0.224 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 3, "route_specific_suppressor_candidate": 5, "small_or_no_effect": 3}` |
| qwen3 | punctuation_or_stop | L35:mlp_out | 12 | -0.990 | 0.000 | -0.797 | 0.000 | `{"booster_candidate": 4, "harmful_or_competitor_support": 8}` |
| qwen3 | punctuation_or_stop | L35:attn_out | 12 | -1.271 | 0.000 | 1.339 | 0.000 | `{"booster_candidate": 10, "route_specific_suppressor_candidate": 2}` |
| qwen3 | other_vocab | L32:mlp_out | 11 | 0.568 | 1.000 | 0.562 | 0.000 | `{"global_suppressor_margin_candidate": 8, "route_specific_suppressor_candidate": 2, "small_or_no_effect": 1}` |
| qwen3 | other_vocab | L34:mlp_out | 11 | 0.659 | 0.909 | -0.580 | 0.000 | `{"global_suppressor_margin_candidate": 4, "harmful_or_competitor_support": 7}` |
| glm4 | recipient_answer | L34:mlp_out | 10 | 0.172 | 0.800 | -0.022 | 0.600 | `{"global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | recipient_answer | L35:mlp_out | 10 | 0.087 | 0.800 | 0.150 | 0.600 | `{"booster_maintenance_candidate": 2, "global_suppressor_maintenance_candidate": 2, "harmful_or_competitor_support": 2, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| glm4 | recipient_answer | L36:attn_out | 10 | 0.069 | 0.800 | 0.250 | 0.700 | `{"global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 7}` |
| glm4 | recipient_answer | L34:attn_out | 10 | 0.131 | 0.600 | 1.850 | 1.000 | `{"booster_candidate": 2, "booster_maintenance_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 5}` |
| glm4 | recipient_answer | L35:attn_out | 10 | 0.078 | 0.600 | 0.428 | 0.800 | `{"booster_candidate": 2, "booster_maintenance_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | recipient_answer | L36:mlp_out | 10 | -0.688 | 0.600 | -1.125 | 0.400 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 6, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 2}` |
| glm4 | recipient_answer | L37:attn_out | 10 | -0.006 | 0.200 | 0.400 | 0.800 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 6}` |
| glm4 | recipient_answer | L39:attn_out | 10 | -0.037 | 0.200 | 0.237 | 0.600 | `{"booster_candidate": 2, "small_or_no_effect": 8}` |
| glm4 | recipient_answer | L38:attn_out | 10 | -0.069 | 0.100 | 0.138 | 0.500 | `{"small_or_no_effect": 10}` |
| glm4 | punctuation_or_stop | L34:mlp_out | 8 | 0.184 | 0.750 | 0.129 | 0.250 | `{"global_suppressor_maintenance_candidate": 2, "global_suppressor_margin_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | punctuation_or_stop | L35:attn_out | 8 | 0.086 | 0.750 | 0.711 | 0.500 | `{"booster_candidate": 2, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 2}` |
| glm4 | punctuation_or_stop | L36:attn_out | 8 | 0.102 | 0.625 | 0.461 | 0.375 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 1, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 3}` |
| glm4 | punctuation_or_stop | L37:attn_out | 8 | 0.055 | 0.625 | 0.578 | 0.500 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | punctuation_or_stop | L36:mlp_out | 8 | 0.090 | 0.500 | -0.277 | 0.250 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 2, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 2, "small_or_no_effect": 2}` |
| glm4 | punctuation_or_stop | L34:attn_out | 8 | 0.043 | 0.375 | 2.105 | 0.750 | `{"booster_candidate": 2, "booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_closure_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 1}` |
| glm4 | punctuation_or_stop | L35:mlp_out | 8 | -0.035 | 0.250 | -0.059 | 0.500 | `{"global_suppressor_maintenance_candidate": 2, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 4}` |
| glm4 | punctuation_or_stop | L39:attn_out | 8 | -0.059 | 0.250 | 0.207 | 0.250 | `{"booster_candidate": 2, "small_or_no_effect": 6}` |
| glm4 | punctuation_or_stop | L38:attn_out | 8 | -0.148 | 0.000 | 0.055 | 0.250 | `{"small_or_no_effect": 8}` |
| glm4 | other_vocab | L36:mlp_out | 12 | 0.193 | 0.750 | -0.151 | 0.333 | `{"booster_candidate": 1, "global_suppressor_maintenance_candidate": 6, "global_suppressor_margin_candidate": 1, "harmful_or_competitor_support": 2, "small_or_no_effect": 2}` |
| glm4 | other_vocab | L34:mlp_out | 12 | 0.068 | 0.667 | -0.094 | 0.500 | `{"global_suppressor_maintenance_candidate": 4, "global_suppressor_margin_candidate": 2, "small_or_no_effect": 6}` |
| glm4 | other_vocab | L34:attn_out | 12 | 0.068 | 0.583 | 1.964 | 0.833 | `{"booster_candidate": 2, "booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_closure_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 5}` |
| glm4 | other_vocab | L35:attn_out | 12 | 0.062 | 0.583 | 0.615 | 0.667 | `{"booster_candidate": 2, "booster_maintenance_candidate": 2, "global_suppressor_margin_candidate": 2, "mixed_boost_global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 4}` |
| glm4 | other_vocab | L37:attn_out | 12 | 0.005 | 0.417 | 0.365 | 0.667 | `{"booster_candidate": 2, "global_suppressor_maintenance_candidate": 2, "small_or_no_effect": 8}` |
| glm4 | other_vocab | L35:mlp_out | 12 | 0.036 | 0.333 | 0.036 | 0.500 | `{"booster_maintenance_candidate": 2, "global_suppressor_maintenance_candidate": 2, "harmful_or_competitor_support": 2, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 4}` |
| deepseek7b | recipient_answer | L23:attn_out | 6 | 1.698 | 1.000 | 3.375 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L24:attn_out | 6 | 1.625 | 1.000 | 2.729 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L25:mlp_out | 6 | 1.271 | 1.000 | 1.427 | 0.167 | `{"global_suppressor_margin_candidate": 4, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | recipient_answer | L24:mlp_out | 6 | 0.906 | 1.000 | 1.865 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L22:attn_out | 6 | 2.562 | 0.833 | 3.604 | 0.167 | `{"global_suppressor_margin_candidate": 5, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | recipient_answer | L26:mlp_out | 6 | 0.781 | 0.833 | 1.458 | 0.167 | `{"global_suppressor_margin_candidate": 4, "small_or_no_effect": 2}` |
| deepseek7b | recipient_answer | L23:mlp_out | 6 | 0.427 | 0.833 | 0.479 | 0.000 | `{"global_suppressor_margin_candidate": 4, "small_or_no_effect": 2}` |
| deepseek7b | recipient_answer | L25:attn_out | 6 | -0.104 | 0.500 | 0.146 | 0.000 | `{"global_suppressor_margin_candidate": 3, "harmful_or_competitor_support": 3}` |
| deepseek7b | recipient_answer | L22:mlp_out | 6 | -0.052 | 0.333 | -0.135 | 0.000 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 1, "small_or_no_effect": 4}` |
| deepseek7b | punctuation_or_stop | L23:mlp_out | 12 | 0.510 | 0.917 | 0.667 | 0.083 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 7, "small_or_no_effect": 4}` |
| deepseek7b | punctuation_or_stop | L25:mlp_out | 12 | 0.469 | 0.833 | 0.823 | 0.167 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 7, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| deepseek7b | punctuation_or_stop | L22:attn_out | 12 | 0.401 | 0.833 | 1.609 | 0.167 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 9, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L24:attn_out | 12 | 0.214 | 0.833 | 1.177 | 0.167 | `{"booster_candidate": 1, "booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 9, "mixed_boost_global_suppressor_closure_candidate": 1}` |
| deepseek7b | punctuation_or_stop | L25:attn_out | 12 | 0.141 | 0.833 | 0.479 | 0.083 | `{"global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 6, "harmful_or_competitor_support": 3, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L22:mlp_out | 12 | 0.333 | 0.750 | 0.286 | 0.083 | `{"booster_candidate": 1, "global_suppressor_margin_candidate": 3, "harmful_or_competitor_support": 1, "small_or_no_effect": 7}` |
| deepseek7b | punctuation_or_stop | L24:mlp_out | 12 | 0.339 | 0.667 | 1.109 | 0.167 | `{"global_suppressor_margin_candidate": 10, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | punctuation_or_stop | L26:mlp_out | 12 | 0.271 | 0.667 | 0.776 | 0.167 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 7, "small_or_no_effect": 4}` |
| deepseek7b | punctuation_or_stop | L23:attn_out | 12 | 0.125 | 0.583 | 1.745 | 0.250 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 9, "mixed_boost_global_suppressor_closure_candidate": 2}` |
| deepseek7b | other_vocab | L26:mlp_out | 12 | 0.359 | 0.833 | 0.865 | 0.167 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 7, "small_or_no_effect": 4}` |
| deepseek7b | other_vocab | L24:mlp_out | 12 | 0.219 | 0.833 | 0.990 | 0.167 | `{"global_suppressor_margin_candidate": 10, "mixed_boost_global_suppressor_closure_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | other_vocab | L25:mlp_out | 12 | 0.359 | 0.667 | 0.714 | 0.167 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 7, "mixed_boost_global_suppressor_closure_candidate": 1, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 2}` |
| deepseek7b | other_vocab | L25:attn_out | 12 | 0.219 | 0.667 | 0.557 | 0.083 | `{"global_suppressor_maintenance_candidate": 1, "global_suppressor_margin_candidate": 6, "harmful_or_competitor_support": 3, "route_specific_suppressor_candidate": 1, "small_or_no_effect": 1}` |
| deepseek7b | other_vocab | L23:attn_out | 12 | 0.188 | 0.667 | 1.807 | 0.250 | `{"booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 9, "mixed_boost_global_suppressor_closure_candidate": 2}` |
| deepseek7b | other_vocab | L24:attn_out | 12 | 0.130 | 0.583 | 1.094 | 0.167 | `{"booster_candidate": 1, "booster_maintenance_candidate": 1, "global_suppressor_margin_candidate": 9, "mixed_boost_global_suppressor_closure_candidate": 1}` |

## Strict Interpretation

- `target boost` measures constructive force toward the donor answer.
- `route suppression` measures selective force against measured route maxima; positive values are suppressor evidence.
- `route coverage` estimates whether the component is route-specific or broad/global over measured top-k routes.
- This is still whole-component donor-recipient delta evidence; it does not yet prove a natural neuron-level suppressor or training-origin suppressor.

Atlas graph: nodes=30 edges=27
