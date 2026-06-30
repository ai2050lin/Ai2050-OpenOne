# Phase 793 Q/K/V/O Independent Causal Decomposition (main)

- Status: `complete`
- Intervention: independent zero-ablation of q_proj/k_proj/v_proj/o_proj paths.
- Q/O are answer-position head interventions; K/V are source-position kv-head interventions.
- This tests necessity-like path effects and token closure gate, not full generation closure.

## Top Minus Matched Specificity

| model | op | subspace | source group | top drop | matched drop | drop gap | top1 loss gap | rank worse gap |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 2.406 | -0.574 | 2.980 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `positive` | `candidate_tokens` | 1.680 | 0.516 | 1.164 | 0.000 | 0.750 |
| qwen3 | `k_source_zero` | `positive` | `candidate_tokens` | 1.766 | 0.547 | 1.219 | 0.000 | 0.500 |
| qwen3 | `k_source_zero` | `negative` | `candidate_tokens` | 1.250 | 0.406 | 0.844 | 0.000 | 0.500 |
| qwen3 | `k_source_zero` | `negative` | `all_pre_answer` | 1.352 | 0.488 | 0.863 | 0.000 | 0.375 |
| qwen3 | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 1.414 | 0.332 | 1.082 | 0.000 | -0.250 |
| qwen3 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.035 | -0.035 | 1.070 | 0.000 | -0.125 |
| qwen3 | `v_source_zero` | `negative` | `candidate_tokens` | 1.227 | 0.398 | 0.828 | 0.000 | 0.250 |
| qwen3 | `v_source_zero` | `positive` | `all_pre_answer` | 0.500 | -0.344 | 0.844 | 0.000 | -0.125 |
| qwen3 | `k_source_zero` | `positive` | `all_pre_answer` | 1.309 | 0.699 | 0.609 | 0.000 | 0.250 |
| qwen3 | `k_source_zero` | `positive` | `target_value_tokens` | 0.844 | 0.305 | 0.539 | 0.000 | 0.250 |
| qwen3 | `v_source_zero` | `negative` | `all_pre_answer` | 0.254 | -0.320 | 0.574 | 0.000 | -0.250 |
| qwen3 | `v_source_zero` | `positive` | `target_value_tokens` | 1.000 | 0.461 | 0.539 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `negative` | `target_value_tokens` | 0.594 | 0.352 | 0.242 | 0.000 | 0.250 |
| qwen3 | `v_source_zero` | `positive` | `instruction` | -0.215 | -0.477 | 0.262 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `negative` | `instruction` | -0.164 | -0.406 | 0.242 | 0.000 | -0.125 |
| qwen3 | `k_source_zero` | `negative` | `target_value_tokens` | 0.539 | 0.398 | 0.141 | 0.000 | 0.250 |
| qwen3 | `k_source_zero` | `negative` | `instruction` | -0.637 | 0.082 | -0.719 | 0.000 | -0.375 |
| qwen3 | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 0.199 | 1.059 | -0.859 | 0.000 | -0.500 |
| qwen3 | `k_source_zero` | `positive` | `instruction` | -1.074 | 0.250 | -1.324 | 0.000 | -0.500 |
| glm4 | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 0.875 | 0.062 | 0.812 | 0.000 | 0.500 |
| glm4 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 0.852 | 0.047 | 0.805 | 0.000 | 0.500 |
| glm4 | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 0.820 | -0.008 | 0.828 | 0.000 | 0.250 |
| glm4 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 0.828 | 0.055 | 0.773 | 0.000 | 0.250 |
| glm4 | `k_source_zero` | `negative` | `all_pre_answer` | 0.945 | 0.945 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `candidate_tokens` | 0.781 | 0.781 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `instruction` | 0.164 | 0.164 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `target_value_tokens` | 0.469 | 0.469 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `all_pre_answer` | 0.945 | 0.945 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `candidate_tokens` | 0.781 | 0.781 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `instruction` | 0.164 | 0.164 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `target_value_tokens` | 0.469 | 0.469 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `all_pre_answer` | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `candidate_tokens` | 0.758 | 0.758 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `instruction` | 0.039 | 0.039 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `target_value_tokens` | 0.648 | 0.648 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `all_pre_answer` | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `candidate_tokens` | 0.758 | 0.758 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `instruction` | 0.039 | 0.039 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `target_value_tokens` | 0.648 | 0.648 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 1.166 | 0.102 | 1.064 | 0.000 | 0.250 |
| deepseek7b | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 1.125 | 0.062 | 1.063 | 0.000 | 0.250 |
| deepseek7b | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1.092 | 0.200 | 0.892 | 0.000 | -0.125 |
| deepseek7b | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.127 | 0.345 | 0.782 | 0.000 | -0.250 |
| deepseek7b | `v_source_zero` | `negative` | `all_pre_answer` | 1.330 | 0.872 | 0.458 | 0.000 | 0.125 |
| deepseek7b | `k_source_zero` | `positive` | `all_pre_answer` | 1.045 | 0.613 | 0.433 | 0.000 | 0.125 |
| deepseek7b | `v_source_zero` | `positive` | `all_pre_answer` | 1.427 | 0.955 | 0.473 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `negative` | `instruction` | 1.137 | 0.825 | 0.312 | 0.000 | 0.375 |
| deepseek7b | `v_source_zero` | `positive` | `instruction` | 1.243 | 0.954 | 0.289 | 0.000 | -0.125 |
| deepseek7b | `k_source_zero` | `negative` | `all_pre_answer` | 0.730 | 0.523 | 0.207 | 0.000 | 0.250 |
| deepseek7b | `v_source_zero` | `negative` | `candidate_tokens` | 0.281 | 0.051 | 0.230 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `negative` | `candidate_tokens` | 0.203 | 0.055 | 0.148 | 0.000 | 0.250 |
| deepseek7b | `v_source_zero` | `negative` | `target_value_tokens` | 0.125 | 0.000 | 0.125 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `target_value_tokens` | 0.297 | 0.172 | 0.125 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `positive` | `instruction` | 1.251 | 1.163 | 0.088 | 0.000 | 0.250 |
| deepseek7b | `v_source_zero` | `positive` | `candidate_tokens` | 0.312 | 0.219 | 0.094 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `negative` | `target_value_tokens` | 0.117 | 0.062 | 0.055 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `positive` | `candidate_tokens` | 0.266 | 0.219 | 0.047 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `negative` | `instruction` | 1.087 | 1.100 | -0.013 | 0.000 | 0.250 |
| deepseek7b | `k_source_zero` | `positive` | `target_value_tokens` | 0.266 | 0.289 | -0.023 | 0.000 | 0.000 |

## Top Operation Effects

| model | selection | op | subspace | source group | cases | margin drop | target drop | rank worse | top1 loss |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 4 | 2.406 | 0.641 | 0.750 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 4 | 1.766 | 0.812 | 1.000 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 4 | 1.680 | 0.742 | 1.000 | 0.000 |
| qwen3 | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 4 | 1.414 | -0.117 | 0.625 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `candidate_tokens` | 4 | 1.250 | 0.516 | 0.750 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `negative` | `candidate_tokens` | 4 | 1.227 | 0.477 | 0.750 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 1.352 | -0.039 | 0.500 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `target_value_tokens` | 4 | 1.000 | 0.484 | 1.000 | 0.000 |
| qwen3 | `matched` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 4 | 1.059 | 0.934 | 0.750 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `target_value_tokens` | 4 | 0.844 | 0.391 | 1.000 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 1.309 | -0.254 | 0.250 | 0.000 |
| qwen3 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 4 | 1.035 | -0.020 | 0.375 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `negative` | `target_value_tokens` | 4 | 0.594 | 0.297 | 0.750 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `target_value_tokens` | 4 | 0.539 | 0.195 | 0.750 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `target_value_tokens` | 4 | 0.461 | 0.258 | 1.000 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 4 | 0.547 | 0.391 | 0.500 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 4 | 0.500 | 0.422 | 0.625 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 0.699 | -1.395 | 0.000 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 4 | 0.516 | 0.375 | 0.250 | 0.000 |
| qwen3 | `matched` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 4 | 0.332 | 1.145 | 0.875 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `negative` | `target_value_tokens` | 4 | 0.398 | 0.273 | 0.500 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `negative` | `candidate_tokens` | 4 | 0.398 | 0.305 | 0.500 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 0.488 | -1.371 | 0.125 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `target_value_tokens` | 4 | 0.305 | 0.133 | 0.750 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 0.945 | 0.492 | 0.750 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 0.945 | 0.492 | 0.750 | 0.000 |
| glm4 | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 0.945 | 0.492 | 0.750 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 0.945 | 0.492 | 0.750 | 0.000 |
| glm4 | `top` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 4 | 0.875 | 0.375 | 0.750 | 0.000 |
| glm4 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 4 | 0.852 | 0.352 | 0.750 | 0.000 |
| glm4 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 4 | 0.828 | 0.359 | 0.750 | 0.000 |
| glm4 | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 4 | 0.820 | 0.367 | 0.750 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 4 | 0.781 | 0.375 | 0.750 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 4 | 0.781 | 0.375 | 0.750 | 0.000 |
| glm4 | `top` | `k_source_zero` | `negative` | `candidate_tokens` | 4 | 0.781 | 0.375 | 0.750 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `negative` | `candidate_tokens` | 4 | 0.781 | 0.375 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 4 | 0.758 | 0.352 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 4 | 0.758 | 0.352 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `candidate_tokens` | 4 | 0.758 | 0.352 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `candidate_tokens` | 4 | 0.758 | 0.352 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 4 | 0.750 | 0.297 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 4 | 0.750 | 0.297 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `all_pre_answer` | 4 | 0.750 | 0.297 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `all_pre_answer` | 4 | 0.750 | 0.297 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `target_value_tokens` | 4 | 0.648 | 0.305 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `target_value_tokens` | 4 | 0.648 | 0.305 | 0.750 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `target_value_tokens` | 4 | 0.648 | 0.305 | 0.750 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `target_value_tokens` | 4 | 0.648 | 0.305 | 0.750 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 4 | 1.427 | 0.531 | 0.750 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `negative` | `all_pre_answer` | 4 | 1.330 | 1.070 | 0.750 | 0.000 |
| deepseek7b | `top` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 4 | 1.166 | 0.885 | 0.875 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `negative` | `instruction` | 4 | 1.137 | 0.762 | 0.875 | 0.000 |
| deepseek7b | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 4 | 1.125 | 0.949 | 0.875 | 0.000 |
| deepseek7b | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 4 | 1.127 | 0.649 | 0.750 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `instruction` | 4 | 1.251 | 0.745 | 0.500 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `instruction` | 4 | 1.243 | 0.468 | 0.500 | 0.000 |
| deepseek7b | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 4 | 1.092 | 0.590 | 0.625 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 1.045 | 0.563 | 0.625 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 4 | 0.955 | 0.589 | 0.750 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `negative` | `instruction` | 4 | 1.087 | 0.683 | 0.500 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `instruction` | 4 | 0.954 | 0.464 | 0.625 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `instruction` | 4 | 1.163 | 0.399 | 0.250 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `negative` | `all_pre_answer` | 4 | 0.872 | 0.889 | 0.625 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `negative` | `instruction` | 4 | 1.100 | 0.282 | 0.250 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `negative` | `instruction` | 4 | 0.825 | 0.499 | 0.500 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 0.730 | 0.738 | 0.625 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 4 | 0.613 | 0.271 | 0.500 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 4 | 0.523 | 0.345 | 0.375 | 0.000 |
| deepseek7b | `matched` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 4 | 0.345 | 0.345 | 1.000 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 4 | 0.312 | 0.656 | 1.000 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `target_value_tokens` | 4 | 0.297 | 0.734 | 1.000 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `target_value_tokens` | 4 | 0.289 | 0.570 | 1.000 | 0.000 |
