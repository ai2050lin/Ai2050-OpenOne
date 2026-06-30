# Phase 794 Q/K/V/O Replacement Closure Validation (main)

- Status: `complete`
- Intervention: donor-to-recipient replacement of q_proj/k_proj/v_proj/o_proj path slices.
- Q/O are answer-position replacements; K/V are paired source-group replacements.
- This tests sufficiency-like closure pressure. Strong closure requires token or phrase gain.

## Top Minus Matched Replacement Specificity

| model | op | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 2.406 | -0.062 | 2.469 | 0.000 | 0.167 | 0.500 |
| qwen3 | `v_source_replace` | `negative` | `all_pre_answer` | 0.807 | -1.168 | 1.975 | 0.000 | 0.167 | 0.167 |
| qwen3 | `v_source_replace` | `positive` | `all_pre_answer` | 0.914 | -0.680 | 1.594 | 0.000 | 0.167 | 0.000 |
| qwen3 | `k_source_replace` | `positive` | `all_pre_answer` | 0.503 | -0.557 | 1.060 | 0.000 | 0.000 | 0.167 |
| qwen3 | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 1.526 | 0.766 | 0.760 | 0.000 | 0.167 | 0.167 |
| qwen3 | `k_source_replace` | `negative` | `all_pre_answer` | -0.331 | -0.898 | 0.568 | 0.000 | 0.000 | 0.500 |
| qwen3 | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 0.401 | 0.083 | 0.318 | 0.000 | 0.000 | 0.667 |
| qwen3 | `v_source_replace` | `positive` | `instruction` | -0.005 | -0.292 | 0.286 | 0.000 | 0.000 | -0.333 |
| qwen3 | `v_source_replace` | `negative` | `instruction` | -0.005 | -0.286 | 0.281 | 0.000 | 0.000 | -0.333 |
| qwen3 | `k_source_replace` | `positive` | `instruction` | 0.036 | 0.010 | 0.026 | 0.000 | 0.000 | -0.167 |
| qwen3 | `k_source_replace` | `negative` | `instruction` | 0.026 | 0.005 | 0.021 | 0.000 | 0.000 | -0.167 |
| qwen3 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.182 | 0.193 | -0.010 | 0.000 | 0.000 | 0.167 |
| glm4 | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 0.495 | -0.010 | 0.505 | 0.000 | 0.000 | 0.667 |
| glm4 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 0.547 | -0.016 | 0.562 | 0.000 | 0.000 | 0.333 |
| glm4 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | -0.036 | -0.047 | 0.010 | 0.000 | 0.000 | -0.333 |
| glm4 | `k_source_replace` | `negative` | `all_pre_answer` | 0.036 | 0.036 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `negative` | `instruction` | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `positive` | `all_pre_answer` | 0.036 | 0.036 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `positive` | `instruction` | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `negative` | `all_pre_answer` | -0.125 | -0.125 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `negative` | `instruction` | -0.021 | -0.021 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `positive` | `all_pre_answer` | -0.125 | -0.125 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `positive` | `instruction` | -0.021 | -0.021 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | -0.036 | -0.026 | -0.010 | 0.000 | 0.000 | -0.333 |
| deepseek7b | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 0.657 | 0.089 | 0.568 | 0.000 | 0.000 | 0.500 |
| deepseek7b | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 0.602 | 0.103 | 0.499 | 0.000 | 0.000 | 0.167 |
| deepseek7b | `k_source_replace` | `negative` | `all_pre_answer` | -0.027 | -0.275 | 0.247 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_replace` | `negative` | `instruction` | 0.355 | 0.221 | 0.135 | 0.000 | 0.000 | -0.167 |
| deepseek7b | `v_source_replace` | `positive` | `instruction` | 0.328 | 0.212 | 0.117 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `k_source_replace` | `positive` | `all_pre_answer` | -0.172 | -0.288 | 0.116 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.039 | -0.018 | 0.057 | 0.000 | 0.000 | 0.333 |
| deepseek7b | `k_source_replace` | `positive` | `instruction` | 0.077 | 0.039 | 0.038 | 0.000 | 0.000 | 0.167 |
| deepseek7b | `k_source_replace` | `negative` | `instruction` | 0.051 | 0.039 | 0.012 | 0.000 | 0.000 | 0.167 |
| deepseek7b | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 0.068 | 0.057 | 0.011 | 0.000 | 0.000 | -0.167 |
| deepseek7b | `v_source_replace` | `negative` | `all_pre_answer` | 0.608 | 0.767 | -0.159 | 0.000 | 0.000 | -0.333 |
| deepseek7b | `v_source_replace` | `positive` | `all_pre_answer` | 0.535 | 0.849 | -0.314 | 0.000 | 0.000 | 0.000 |

## Top Replacement Effects

| model | selection | op | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | 2.406 | 1.000 | 0.000 | 0.167 |
| qwen3 | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | 1.526 | 0.833 | 0.000 | 0.167 |
| qwen3 | `top` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | 0.914 | 0.833 | 0.000 | 0.167 |
| qwen3 | `top` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | 0.807 | 1.000 | 0.000 | 0.167 |
| qwen3 | `matched` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | 0.766 | 0.667 | 0.000 | null |
| qwen3 | `top` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | 0.503 | 0.667 | 0.000 | 0.000 |
| qwen3 | `top` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | 0.401 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | 0.182 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | 0.193 | 0.667 | 0.000 | null |
| qwen3 | `matched` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | 0.083 | 0.333 | 0.000 | null |
| qwen3 | `top` | `k_source_replace` | `positive` | `instruction` | 3 | 0.036 | 0.333 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `negative` | `instruction` | 3 | 0.026 | 0.333 | 0.000 | 0.000 |
| qwen3 | `matched` | `k_source_replace` | `positive` | `instruction` | 3 | 0.010 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `k_source_replace` | `negative` | `instruction` | 3 | 0.005 | 0.500 | 0.000 | null |
| qwen3 | `top` | `v_source_replace` | `positive` | `instruction` | 3 | -0.005 | 0.167 | 0.000 | 0.000 |
| qwen3 | `top` | `v_source_replace` | `negative` | `instruction` | 3 | -0.005 | 0.167 | 0.000 | 0.000 |
| qwen3 | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | -0.062 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `v_source_replace` | `negative` | `instruction` | 3 | -0.286 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `v_source_replace` | `positive` | `instruction` | 3 | -0.292 | 0.500 | 0.000 | null |
| qwen3 | `top` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | -0.331 | 0.833 | 0.000 | 0.000 |
| qwen3 | `matched` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | -0.557 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | -0.680 | 0.833 | 0.000 | null |
| qwen3 | `matched` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | -0.898 | 0.333 | 0.000 | null |
| qwen3 | `matched` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | -1.168 | 0.833 | 0.000 | null |
| glm4 | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | 0.495 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | 0.547 | 0.667 | 0.000 | 0.000 |
| glm4 | `top` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | 0.036 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | 0.036 | 0.667 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | 0.036 | 0.667 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | 0.036 | 0.667 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `positive` | `instruction` | 3 | 0.010 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `positive` | `instruction` | 3 | 0.010 | 0.333 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `negative` | `instruction` | 3 | 0.010 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `negative` | `instruction` | 3 | 0.010 | 0.333 | 0.000 | null |
| glm4 | `matched` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | -0.010 | 0.333 | 0.000 | null |
| glm4 | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | -0.016 | 0.333 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `positive` | `instruction` | 3 | -0.021 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `positive` | `instruction` | 3 | -0.021 | 0.333 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `negative` | `instruction` | 3 | -0.021 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `negative` | `instruction` | 3 | -0.021 | 0.333 | 0.000 | null |
| glm4 | `matched` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | -0.026 | 0.333 | 0.000 | null |
| glm4 | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | -0.036 | 0.000 | 0.000 | 0.000 |
| glm4 | `top` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | -0.036 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | -0.047 | 0.333 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | -0.125 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | -0.125 | 0.333 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | -0.125 | 0.333 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | -0.125 | 0.333 | 0.000 | null |
| deepseek7b | `matched` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | 0.849 | 0.667 | 0.000 | null |
| deepseek7b | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | 0.657 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | 0.767 | 0.500 | 0.000 | null |
| deepseek7b | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | 0.602 | 0.833 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `positive` | `all_pre_answer` | 3 | 0.535 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `negative` | `all_pre_answer` | 3 | 0.608 | 0.167 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `positive` | `instruction` | 3 | 0.328 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `negative` | `instruction` | 3 | 0.355 | 0.333 | 0.000 | 0.000 |
| deepseek7b | `matched` | `v_source_replace` | `negative` | `instruction` | 3 | 0.221 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `v_source_replace` | `positive` | `instruction` | 3 | 0.212 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 3 | 0.103 | 0.667 | 0.000 | null |
| deepseek7b | `matched` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 3 | 0.089 | 0.500 | 0.000 | null |
| deepseek7b | `top` | `k_source_replace` | `positive` | `instruction` | 3 | 0.077 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `top` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | 0.068 | 0.333 | 0.000 | 0.000 |
| deepseek7b | `matched` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 3 | 0.057 | 0.500 | 0.000 | null |
| deepseek7b | `top` | `k_source_replace` | `negative` | `instruction` | 3 | 0.051 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | 0.039 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `matched` | `k_source_replace` | `positive` | `instruction` | 3 | 0.039 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `k_source_replace` | `negative` | `instruction` | 3 | 0.039 | 0.333 | 0.000 | null |
| deepseek7b | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 3 | -0.018 | 0.333 | 0.000 | null |
| deepseek7b | `top` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | -0.027 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | -0.172 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `k_source_replace` | `negative` | `all_pre_answer` | 3 | -0.275 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `k_source_replace` | `positive` | `all_pre_answer` | 3 | -0.288 | 0.500 | 0.000 | null |
