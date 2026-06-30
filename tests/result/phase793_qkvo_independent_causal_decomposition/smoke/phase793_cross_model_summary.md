# Phase 793 Q/K/V/O Independent Causal Decomposition (smoke)

- Status: `complete`
- Intervention: independent zero-ablation of q_proj/k_proj/v_proj/o_proj paths.
- Q/O are answer-position head interventions; K/V are source-position kv-head interventions.
- This tests necessity-like path effects and token closure gate, not full generation closure.

## Top Minus Matched Specificity

| model | op | subspace | source group | top drop | matched drop | drop gap | top1 loss gap | rank worse gap |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 2.156 | -0.219 | 2.375 | 0.000 | 1.000 |
| qwen3 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.500 | 0.188 | 1.312 | 0.000 | 0.500 |
| qwen3 | `k_source_zero` | `positive` | `all_pre_answer` | 1.969 | 0.938 | 1.031 | 0.000 | 0.500 |
| qwen3 | `k_source_zero` | `positive` | `candidate_tokens` | 2.875 | 2.625 | 0.250 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `positive` | `candidate_tokens` | 2.875 | 2.875 | 0.000 | 0.000 | 0.000 |
| qwen3 | `v_source_zero` | `positive` | `all_pre_answer` | 1.250 | 1.500 | -0.250 | 0.000 | -0.500 |
| glm4 | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1.500 | -0.250 | 1.750 | 0.000 | 0.000 |
| glm4 | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.312 | -0.062 | 1.375 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `all_pre_answer` | 1.375 | 1.375 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_zero` | `positive` | `candidate_tokens` | 1.312 | 1.312 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `all_pre_answer` | 1.250 | 1.250 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_zero` | `positive` | `candidate_tokens` | 1.250 | 1.250 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1.398 | 0.328 | 1.070 | 0.000 | 0.000 |
| deepseek7b | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1.281 | 0.422 | 0.859 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `all_pre_answer` | 2.164 | 1.664 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `positive` | `all_pre_answer` | 1.445 | 1.117 | 0.328 | 0.000 | 0.000 |
| deepseek7b | `k_source_zero` | `positive` | `candidate_tokens` | 0.562 | 0.656 | -0.094 | 0.000 | 0.000 |
| deepseek7b | `v_source_zero` | `positive` | `candidate_tokens` | 0.562 | 0.719 | -0.156 | 0.000 | 0.000 |

## Top Operation Effects

| model | selection | op | subspace | source group | cases | margin drop | target drop | rank worse | top1 loss |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 2.875 | 1.375 | 1.000 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 2.875 | 1.500 | 1.000 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 2.875 | 1.250 | 1.000 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 2.625 | 1.125 | 1.000 | 0.000 |
| qwen3 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | 2.156 | 0.531 | 1.000 | 0.000 |
| qwen3 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | 1.500 | 0.312 | 1.000 | 0.000 |
| qwen3 | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 1.500 | 1.000 | 1.000 | 0.000 |
| qwen3 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 1.969 | 0.156 | 0.500 | 0.000 |
| qwen3 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 1.250 | 0.750 | 0.500 | 0.000 |
| qwen3 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 0.938 | -0.562 | 0.000 | 0.000 |
| qwen3 | `matched` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | 0.188 | 0.312 | 0.500 | 0.000 |
| qwen3 | `matched` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | -0.219 | -0.594 | 0.000 | 0.000 |
| glm4 | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | 1.500 | 0.938 | 0.000 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 1.375 | 0.750 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 1.375 | 0.750 | 0.000 | 0.000 |
| glm4 | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | 1.312 | 0.750 | 0.000 | 0.000 |
| glm4 | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 1.312 | 0.750 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 1.312 | 0.750 | 0.000 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 1.250 | 0.688 | 0.000 | 0.000 |
| glm4 | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 1.250 | 0.562 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 1.250 | 0.688 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 1.250 | 0.562 | 0.000 | 0.000 |
| glm4 | `matched` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | -0.062 | -0.125 | 0.000 | 0.000 |
| glm4 | `matched` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | -0.250 | -0.250 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 2.164 | 1.039 | 1.000 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `all_pre_answer` | 1 | 1.664 | 1.164 | 1.000 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 1.445 | 1.320 | 1.000 | 0.000 |
| deepseek7b | `top` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | 1.398 | 1.117 | 1.000 | 0.000 |
| deepseek7b | `top` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | 1.281 | 0.969 | 1.000 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `all_pre_answer` | 1 | 1.117 | 1.305 | 1.000 | 0.000 |
| deepseek7b | `matched` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 0.719 | 0.594 | 1.000 | 0.000 |
| deepseek7b | `matched` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 0.656 | 0.531 | 1.000 | 0.000 |
| deepseek7b | `top` | `k_source_zero` | `positive` | `candidate_tokens` | 1 | 0.562 | 0.562 | 1.000 | 0.000 |
| deepseek7b | `top` | `v_source_zero` | `positive` | `candidate_tokens` | 1 | 0.562 | 0.438 | 1.000 | 0.000 |
| deepseek7b | `matched` | `q_answer_zero` | `positive` | `q_answer_zero:answer_position` | 1 | 0.422 | 0.047 | 1.000 | 0.000 |
| deepseek7b | `matched` | `o_answer_zero` | `positive` | `o_answer_zero:answer_position` | 1 | 0.328 | -0.078 | 1.000 | 0.000 |
