# Phase 794 Q/K/V/O Replacement Closure Validation (confirm)

- Status: `complete`
- Intervention: donor-to-recipient replacement of q_proj/k_proj/v_proj/o_proj path slices.
- Q/O are answer-position replacements; K/V are paired source-group replacements.
- This tests sufficiency-like closure pressure. Strong closure requires token or phrase gain.

## Top Minus Matched Replacement Specificity

| model | op | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 2.222 | -0.356 | 2.578 | 0.000 | 0.100 | 0.500 |
| qwen3 | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 1.716 | 0.144 | 1.572 | 0.000 | 0.100 | 0.000 |
| qwen3 | `k_source_replace` | `positive` | `all_pre_answer` | 0.928 | -0.583 | 1.511 | 0.000 | 0.000 | 0.000 |
| qwen3 | `k_source_replace` | `negative` | `all_pre_answer` | 0.311 | -0.770 | 1.081 | 0.000 | 0.000 | -0.100 |
| qwen3 | `v_source_replace` | `positive` | `all_pre_answer` | 0.733 | -0.278 | 1.011 | 0.000 | 0.000 | 0.000 |
| qwen3 | `v_source_replace` | `negative` | `all_pre_answer` | 0.572 | -0.434 | 1.006 | 0.000 | 0.000 | 0.000 |
| qwen3 | `v_source_replace` | `negative` | `instruction` | -0.009 | -0.341 | 0.331 | 0.000 | 0.000 | -0.200 |
| qwen3 | `v_source_replace` | `positive` | `instruction` | 0.000 | -0.322 | 0.322 | 0.000 | 0.000 | -0.200 |
| qwen3 | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 0.228 | 0.062 | 0.166 | 0.000 | 0.000 | 0.100 |
| qwen3 | `k_source_replace` | `positive` | `question` | -0.062 | -0.113 | 0.050 | 0.000 | 0.000 | -0.300 |
| qwen3 | `k_source_replace` | `negative` | `question` | -0.091 | -0.100 | 0.009 | 0.000 | 0.000 | -0.200 |
| qwen3 | `k_source_replace` | `negative` | `instruction` | 0.016 | 0.019 | -0.003 | 0.000 | 0.000 | 0.100 |
| qwen3 | `k_source_replace` | `positive` | `instruction` | -0.003 | 0.025 | -0.028 | 0.000 | 0.000 | 0.000 |
| qwen3 | `v_source_replace` | `positive` | `question` | 0.053 | 0.081 | -0.028 | 0.000 | 0.000 | 0.300 |
| qwen3 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.150 | 0.178 | -0.028 | 0.000 | 0.000 | -0.100 |
| qwen3 | `v_source_replace` | `negative` | `question` | 0.006 | 0.087 | -0.081 | 0.000 | 0.000 | 0.300 |
| glm4 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 0.736 | -0.102 | 0.838 | 0.000 | 0.000 | 1.000 |
| glm4 | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 0.736 | -0.087 | 0.823 | 0.000 | 0.000 | 0.800 |
| glm4 | `k_source_replace` | `negative` | `all_pre_answer` | -0.006 | -0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `negative` | `instruction` | 0.002 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `negative` | `question` | -0.009 | -0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `positive` | `all_pre_answer` | -0.006 | -0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `positive` | `instruction` | 0.002 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `k_source_replace` | `positive` | `question` | -0.009 | -0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `negative` | `all_pre_answer` | -0.089 | -0.089 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `negative` | `instruction` | -0.027 | -0.027 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `negative` | `question` | 0.006 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `positive` | `all_pre_answer` | -0.089 | -0.089 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `positive` | `instruction` | -0.027 | -0.027 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `v_source_replace` | `positive` | `question` | 0.006 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | -0.036 | -0.027 | -0.009 | 0.000 | 0.000 | 0.000 |
| glm4 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | -0.045 | -0.028 | -0.017 | 0.000 | 0.000 | -0.200 |
| deepseek7b | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 0.892 | 0.080 | 0.812 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 0.854 | 0.049 | 0.804 | 0.000 | 0.000 | 0.300 |
| deepseek7b | `v_source_replace` | `positive` | `all_pre_answer` | 0.976 | 0.605 | 0.371 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_replace` | `positive` | `instruction` | 0.348 | 0.207 | 0.141 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_replace` | `negative` | `instruction` | 0.350 | 0.241 | 0.109 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `k_source_replace` | `positive` | `all_pre_answer` | -0.188 | -0.270 | 0.082 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 0.043 | 0.011 | 0.032 | 0.000 | 0.000 | 0.200 |
| deepseek7b | `v_source_replace` | `negative` | `all_pre_answer` | 0.652 | 0.620 | 0.032 | 0.000 | 0.000 | -0.100 |
| deepseek7b | `k_source_replace` | `negative` | `instruction` | 0.109 | 0.079 | 0.029 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `k_source_replace` | `positive` | `instruction` | 0.083 | 0.059 | 0.024 | 0.000 | 0.000 | 0.100 |
| deepseek7b | `k_source_replace` | `negative` | `question` | 0.032 | 0.010 | 0.022 | 0.000 | 0.000 | 0.100 |
| deepseek7b | `k_source_replace` | `positive` | `question` | 0.030 | 0.014 | 0.016 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_replace` | `positive` | `question` | 0.076 | 0.079 | -0.003 | 0.000 | 0.000 | -0.200 |
| deepseek7b | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.005 | 0.009 | -0.004 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `v_source_replace` | `negative` | `question` | 0.050 | 0.067 | -0.017 | 0.000 | 0.000 | -0.100 |
| deepseek7b | `k_source_replace` | `negative` | `all_pre_answer` | -0.295 | -0.248 | -0.047 | 0.000 | 0.000 | 0.000 |

## Top Replacement Effects

| model | selection | op | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 5 | 2.222 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 5 | 1.716 | 0.900 | 0.000 | 0.100 |
| qwen3 | `top` | `v_source_replace` | `positive` | `all_pre_answer` | 5 | 0.733 | 0.900 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `positive` | `all_pre_answer` | 5 | 0.928 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `v_source_replace` | `negative` | `all_pre_answer` | 5 | 0.572 | 0.900 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `negative` | `all_pre_answer` | 5 | 0.311 | 0.500 | 0.000 | 0.000 |
| qwen3 | `top` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 5 | 0.228 | 0.600 | 0.000 | 0.000 |
| qwen3 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 5 | 0.178 | 0.800 | 0.000 | null |
| qwen3 | `matched` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 5 | 0.144 | 0.900 | 0.000 | null |
| qwen3 | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 5 | 0.150 | 0.700 | 0.000 | 0.000 |
| qwen3 | `matched` | `v_source_replace` | `negative` | `question` | 5 | 0.087 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `v_source_replace` | `positive` | `question` | 5 | 0.081 | 0.500 | 0.000 | null |
| qwen3 | `top` | `v_source_replace` | `positive` | `question` | 5 | 0.053 | 0.800 | 0.000 | 0.000 |
| qwen3 | `matched` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 5 | 0.062 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `k_source_replace` | `positive` | `instruction` | 5 | 0.025 | 0.400 | 0.000 | null |
| qwen3 | `matched` | `k_source_replace` | `negative` | `instruction` | 5 | 0.019 | 0.300 | 0.000 | null |
| qwen3 | `top` | `k_source_replace` | `negative` | `instruction` | 5 | 0.016 | 0.400 | 0.000 | 0.000 |
| qwen3 | `top` | `v_source_replace` | `negative` | `question` | 5 | 0.006 | 0.800 | 0.000 | 0.000 |
| qwen3 | `top` | `v_source_replace` | `positive` | `instruction` | 5 | 0.000 | 0.400 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `positive` | `instruction` | 5 | -0.003 | 0.400 | 0.000 | 0.000 |
| qwen3 | `top` | `v_source_replace` | `negative` | `instruction` | 5 | -0.009 | 0.300 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `positive` | `question` | 5 | -0.062 | 0.300 | 0.000 | 0.000 |
| qwen3 | `top` | `k_source_replace` | `negative` | `question` | 5 | -0.091 | 0.300 | 0.000 | 0.000 |
| qwen3 | `matched` | `k_source_replace` | `negative` | `question` | 5 | -0.100 | 0.500 | 0.000 | null |
| glm4 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 5 | 0.736 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `v_source_replace` | `positive` | `question` | 5 | 0.006 | 0.600 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `positive` | `question` | 5 | 0.006 | 0.600 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `negative` | `question` | 5 | 0.006 | 0.600 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `negative` | `question` | 5 | 0.006 | 0.600 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `positive` | `instruction` | 5 | 0.002 | 0.200 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `positive` | `instruction` | 5 | 0.002 | 0.200 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `negative` | `instruction` | 5 | 0.002 | 0.200 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `negative` | `instruction` | 5 | 0.002 | 0.200 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `positive` | `all_pre_answer` | 5 | -0.006 | 0.600 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `positive` | `all_pre_answer` | 5 | -0.006 | 0.600 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `negative` | `all_pre_answer` | 5 | -0.006 | 0.600 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `negative` | `all_pre_answer` | 5 | -0.006 | 0.600 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `positive` | `question` | 5 | -0.009 | 0.200 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `positive` | `question` | 5 | -0.009 | 0.200 | 0.000 | null |
| glm4 | `top` | `k_source_replace` | `negative` | `question` | 5 | -0.009 | 0.200 | 0.000 | 0.000 |
| glm4 | `matched` | `k_source_replace` | `negative` | `question` | 5 | -0.009 | 0.200 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `positive` | `instruction` | 5 | -0.027 | 0.400 | 0.000 | 0.000 |
| glm4 | `matched` | `v_source_replace` | `positive` | `instruction` | 5 | -0.027 | 0.400 | 0.000 | null |
| glm4 | `top` | `v_source_replace` | `negative` | `instruction` | 5 | -0.027 | 0.400 | 0.000 | 0.000 |
| glm4 | `matched` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 5 | -0.027 | 0.000 | 0.000 | null |
| glm4 | `matched` | `v_source_replace` | `negative` | `instruction` | 5 | -0.027 | 0.400 | 0.000 | null |
| glm4 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 5 | -0.028 | 0.200 | 0.000 | null |
| deepseek7b | `top` | `v_source_replace` | `positive` | `all_pre_answer` | 5 | 0.976 | 0.700 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 5 | 0.854 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 5 | 0.892 | 0.800 | 0.000 | 0.000 |
| deepseek7b | `matched` | `v_source_replace` | `positive` | `all_pre_answer` | 5 | 0.605 | 0.700 | 0.000 | null |
| deepseek7b | `matched` | `v_source_replace` | `negative` | `all_pre_answer` | 5 | 0.620 | 0.600 | 0.000 | null |
| deepseek7b | `top` | `v_source_replace` | `negative` | `all_pre_answer` | 5 | 0.652 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `positive` | `instruction` | 5 | 0.348 | 0.600 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `negative` | `instruction` | 5 | 0.350 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `v_source_replace` | `negative` | `instruction` | 5 | 0.241 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `v_source_replace` | `positive` | `instruction` | 5 | 0.207 | 0.600 | 0.000 | null |
| deepseek7b | `top` | `k_source_replace` | `negative` | `instruction` | 5 | 0.109 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `matched` | `v_source_replace` | `positive` | `question` | 5 | 0.079 | 0.900 | 0.000 | null |
| deepseek7b | `top` | `k_source_replace` | `positive` | `instruction` | 5 | 0.083 | 0.700 | 0.000 | 0.000 |
| deepseek7b | `top` | `v_source_replace` | `positive` | `question` | 5 | 0.076 | 0.700 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 5 | 0.080 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `k_source_replace` | `negative` | `instruction` | 5 | 0.079 | 0.500 | 0.000 | null |
| deepseek7b | `matched` | `v_source_replace` | `negative` | `question` | 5 | 0.067 | 0.700 | 0.000 | null |
| deepseek7b | `matched` | `k_source_replace` | `positive` | `instruction` | 5 | 0.059 | 0.600 | 0.000 | null |
| deepseek7b | `top` | `v_source_replace` | `negative` | `question` | 5 | 0.050 | 0.600 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_answer_replace` | `negative` | `o_answer_replace:answer_position` | 5 | 0.049 | 0.600 | 0.000 | null |
| deepseek7b | `top` | `q_answer_replace` | `negative` | `q_answer_replace:answer_position` | 5 | 0.043 | 0.500 | 0.000 | 0.000 |
| deepseek7b | `top` | `k_source_replace` | `negative` | `question` | 5 | 0.032 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `top` | `k_source_replace` | `positive` | `question` | 5 | 0.030 | 0.900 | 0.000 | 0.000 |
| deepseek7b | `matched` | `k_source_replace` | `positive` | `question` | 5 | 0.014 | 0.900 | 0.000 | null |
