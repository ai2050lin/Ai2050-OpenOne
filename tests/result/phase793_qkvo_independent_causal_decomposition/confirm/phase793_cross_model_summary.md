# Phase 793 Q/K/V/O Independent Causal Decomposition (confirm)

- Status: `complete`
- Intervention: independent zero-ablation of q_proj/k_proj/v_proj/o_proj paths.
- Q/O are answer-position head interventions; K/V are source-position kv-head interventions.
- This tests necessity-like path effects and token closure gate, not full generation closure.

## Top Minus Matched Specificity

| model | op | subspace | source group | top drop | matched drop | drop gap | top1 loss gap | rank worse gap |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 2.328 | -0.628 | 2.956 | 0.000 | 0.333 |
| qwen3 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.396 | 0.042 | 1.354 | 0.000 | 0.250 |
| qwen3 | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 1.474 | 0.065 | 1.409 | 0.000 | -0.083 |
| qwen3 | `v_source_zero` | `positive` | `candidate_tokens` | 1.839 | 0.958 | 0.880 | 0.000 | 0.167 |
| qwen3 | `k_source_zero` | `positive` | `candidate_tokens` | 1.885 | 1.016 | 0.870 | 0.000 | 0.167 |
| qwen3 | `k_source_zero` | `negative` | `candidate_tokens` | 1.370 | 0.609 | 0.760 | 0.000 | 0.167 |
| qwen3 | `v_source_zero` | `negative` | `candidate_tokens` | 1.333 | 0.562 | 0.771 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `positive` | `all_pre_answer` | 0.529 | -0.190 | 0.719 | 0.000 | -0.167 |
| qwen3 | `k_source_zero` | `positive` | `all_pre_answer` | 1.346 | 0.773 | 0.573 | 0.000 | 0.250 |
| qwen3 | `v_source_zero` | `positive` | `target_value_tokens` | 1.094 | 0.547 | 0.547 | 0.000 | 0.167 |
| qwen3 | `v_source_zero` | `negative` | `all_pre_answer` | 0.263 | -0.367 | 0.630 | 0.000 | -0.083 |
| qwen3 | `k_source_zero` | `negative` | `all_pre_answer` | 1.214 | 0.701 | 0.513 | 0.000 | 0.167 |
| qwen3 | `k_source_zero` | `positive` | `target_value_tokens` | 0.932 | 0.448 | 0.484 | 0.000 | 0.167 |
| qwen3 | `v_source_zero` | `negative` | `target_value_tokens` | 0.734 | 0.339 | 0.396 | 0.000 | -0.167 |
| qwen3 | `k_source_zero` | `negative` | `target_value_tokens` | 0.578 | 0.302 | 0.276 | 0.000 | -0.167 |
| qwen3 | `v_source_zero` | `positive` | `instruction` | -0.247 | -0.461 | 0.214 | 0.000 | -0.167 |
| qwen3 | `v_source_zero` | `positive` | `question` | -0.089 | -0.245 | 0.156 | 0.000 | 0.083 |
| qwen3 | `v_source_zero` | `negative` | `instruction` | -0.294 | -0.461 | 0.167 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `negative` | `question` | -0.065 | -0.229 | 0.164 | 0.000 | -0.083 |
| qwen3 | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 0.716 | 0.604 | 0.112 | 0.000 | 0.083 |
| glm4 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 0.995 | -0.078 | 1.073 | 0.000 | 0.833 |
| glm4 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.010 | 0.000 | 1.010 | 0.000 | 0.833 |
| glm4 | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 1.005 | -0.073 | 1.078 | 0.000 | 0.667 |
| glm4 | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 1.016 | -0.016 | 1.031 | 0.000 | 0.667 |
| glm4 | `k_source_zero` | `negative` | `all_pre_answer` | 1.083 | 1.083 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `candidate_tokens` | 0.958 | 0.958 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `instruction` | 0.068 | 0.068 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `question` | 0.042 | 0.042 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `negative` | `target_value_tokens` | 0.453 | 0.453 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `all_pre_answer` | 1.083 | 1.083 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `candidate_tokens` | 0.958 | 0.958 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `instruction` | 0.068 | 0.068 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `question` | 0.042 | 0.042 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `target_value_tokens` | 0.453 | 0.453 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `all_pre_answer` | 0.927 | 0.927 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `candidate_tokens` | 0.938 | 0.938 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `instruction` | 0.005 | 0.005 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `question` | 0.078 | 0.078 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `negative` | `target_value_tokens` | 0.719 | 0.719 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `all_pre_answer` | 0.927 | 0.927 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1.315 | 0.086 | 1.229 | 0.000 | 0.083 |
| deepseek7b | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 1.117 | 0.200 | 0.917 | 0.000 | 0.167 |
| deepseek7b | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.301 | 0.271 | 1.030 | 0.000 | -0.250 |
| deepseek7b | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 1.178 | 0.349 | 0.829 | 0.000 | -0.083 |
| deepseek7b | `v_source_zero` | `negative` | `all_pre_answer` | 1.519 | 1.131 | 0.388 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `all_pre_answer` | 1.420 | 1.060 | 0.361 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `instruction` | 1.240 | 0.918 | 0.322 | 0.000 | 0.083 |
| deepseek7b | `v_source_zero` | `negative` | `instruction` | 1.283 | 0.995 | 0.288 | 0.000 | 0.083 |
| deepseek7b | `k_source_zero` | `negative` | `instruction` | 1.829 | 1.653 | 0.176 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `positive` | `instruction` | 1.434 | 1.321 | 0.113 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `negative` | `question` | 0.041 | -0.016 | 0.057 | 0.000 | 0.083 |
| deepseek7b | `k_source_zero` | `positive` | `all_pre_answer` | 1.191 | 1.150 | 0.041 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `question` | 0.007 | -0.025 | 0.033 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `negative` | `target_value_tokens` | 0.159 | 0.159 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `negative` | `candidate_tokens` | 0.398 | 0.409 | -0.010 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `candidate_tokens` | 0.464 | 0.479 | -0.016 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `negative` | `question` | -0.215 | -0.192 | -0.023 | 0.000 | 0.250 |
| deepseek7b | `v_source_zero` | `positive` | `target_value_tokens` | 0.273 | 0.299 | -0.026 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `negative` | `all_pre_answer` | 1.212 | 1.253 | -0.041 | 0.000 | 0.083 |
| deepseek7b | `k_source_zero` | `negative` | `candidate_tokens` | 0.505 | 0.557 | -0.052 | 0.000 | 0.000 |

## Top Operation Effects

| model | selection | op | subspace | source group | cases | margin drop | target drop | rank worse | top1 loss |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 6 | 2.328 | 0.714 | 0.917 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 6 | 1.885 | 0.833 | 1.000 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 6 | 1.839 | 0.797 | 1.000 | 0.000 |
| qwen3 | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 6 | 1.474 | 0.224 | 0.833 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 1.370 | 0.609 | 0.833 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `negative` | `candidate_tokens` | 6 | 1.333 | 0.604 | 0.833 | 0.000 |
| qwen3 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 6 | 1.396 | 0.521 | 0.583 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `target_value_tokens` | 6 | 1.094 | 0.562 | 1.000 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 1.346 | -0.487 | 0.417 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `target_value_tokens` | 6 | 0.932 | 0.464 | 1.000 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 6 | 1.016 | 0.661 | 0.833 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 6 | 0.958 | 0.604 | 0.833 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 1.214 | -0.505 | 0.333 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `negative` | `target_value_tokens` | 6 | 0.734 | 0.370 | 0.833 | 0.000 |
| qwen3 | `top` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 6 | 0.716 | 0.060 | 0.500 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `negative` | `target_value_tokens` | 6 | 0.578 | 0.286 | 0.833 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `negative` | `candidate_tokens` | 6 | 0.562 | 0.302 | 0.833 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 0.609 | 0.349 | 0.667 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `target_value_tokens` | 6 | 0.547 | 0.318 | 0.833 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 6 | 0.529 | 0.497 | 0.833 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 0.773 | -1.732 | 0.167 | 0.000 |
| qwen3 | `matched` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 6 | 0.604 | -0.370 | 0.417 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `target_value_tokens` | 6 | 0.448 | 0.271 | 0.833 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 0.701 | -1.951 | 0.167 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 1.083 | 0.573 | 0.833 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 1.083 | 0.573 | 0.833 | 0.000 |
| glm4 | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 1.083 | 0.573 | 0.833 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 1.083 | 0.573 | 0.833 | 0.000 |
| glm4 | `top` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 6 | 1.016 | 0.505 | 0.833 | 0.000 |
| glm4 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 6 | 1.010 | 0.500 | 0.833 | 0.000 |
| glm4 | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 6 | 1.005 | 0.495 | 0.833 | 0.000 |
| glm4 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 6 | 0.995 | 0.505 | 0.833 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 6 | 0.958 | 0.500 | 0.833 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 6 | 0.958 | 0.500 | 0.833 | 0.000 |
| glm4 | `top` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 0.958 | 0.500 | 0.833 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 0.958 | 0.500 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 6 | 0.938 | 0.469 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 6 | 0.938 | 0.469 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `candidate_tokens` | 6 | 0.938 | 0.469 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `candidate_tokens` | 6 | 0.938 | 0.469 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 6 | 0.927 | 0.396 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 6 | 0.927 | 0.396 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `all_pre_answer` | 6 | 0.927 | 0.396 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `all_pre_answer` | 6 | 0.927 | 0.396 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `target_value_tokens` | 6 | 0.719 | 0.354 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `target_value_tokens` | 6 | 0.719 | 0.354 | 0.833 | 0.000 |
| glm4 | `top` | `v_source_zero` | `negative` | `target_value_tokens` | 6 | 0.719 | 0.354 | 0.833 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `negative` | `target_value_tokens` | 6 | 0.719 | 0.354 | 0.833 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `negative` | `instruction` | 6 | 1.829 | 0.809 | 0.667 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `negative` | `all_pre_answer` | 6 | 1.519 | 1.531 | 0.833 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `negative` | `instruction` | 6 | 1.653 | 0.477 | 0.667 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 6 | 1.420 | 1.135 | 0.833 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `negative` | `instruction` | 6 | 1.283 | 0.803 | 0.833 | 0.000 |
| deepseek7b | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 6 | 1.315 | 0.947 | 0.750 | 0.000 |
| deepseek7b | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 6 | 1.301 | 0.750 | 0.750 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `instruction` | 6 | 1.240 | 0.721 | 0.833 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `instruction` | 6 | 1.434 | 0.805 | 0.500 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 1.212 | 0.859 | 0.750 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `negative` | `all_pre_answer` | 6 | 1.253 | 0.885 | 0.667 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 1.191 | 0.947 | 0.750 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `negative` | `all_pre_answer` | 6 | 1.131 | 1.419 | 0.833 | 0.000 |
| deepseek7b | `top` | `q_answer_zero` | `negative` | `q_answer_zero:answer_position` | 6 | 1.178 | 0.735 | 0.750 | 0.000 |
| deepseek7b | `top` | `o_answer_zero` | `negative` | `o_answer_zero:answer_position` | 6 | 1.117 | 1.181 | 0.833 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 6 | 1.150 | 0.636 | 0.750 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `instruction` | 6 | 1.321 | 0.437 | 0.500 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 6 | 1.060 | 0.863 | 0.833 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `negative` | `instruction` | 6 | 0.995 | 0.658 | 0.750 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `instruction` | 6 | 0.918 | 0.472 | 0.750 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 0.557 | 0.391 | 1.000 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `negative` | `candidate_tokens` | 6 | 0.505 | 0.464 | 1.000 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 6 | 0.484 | 0.328 | 1.000 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 6 | 0.479 | 0.635 | 1.000 | 0.000 |
