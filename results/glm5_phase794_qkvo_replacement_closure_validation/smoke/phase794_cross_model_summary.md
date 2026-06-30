# Phase 794 Q/K/V/O Replacement Closure Validation (smoke)

- Status: `complete`
- Intervention: donor-to-recipient replacement of q_proj/k_proj/v_proj/o_proj path slices.
- Q/O are answer-position replacements; K/V are paired source-group replacements.
- This tests sufficiency-like closure pressure. Strong closure requires token or phrase gain.

## Top Minus Matched Replacement Specificity

| model | op | subspace | source group | top delta | matched delta | gap | token gain gap | phrase gain gap | rank improve gap |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 2.281 | -0.188 | 2.469 | 0.000 | 0.000 | 1.000 |
| qwen3 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.125 | -0.031 | 0.156 | 0.000 | 0.000 | 0.500 |
| glm4 | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1.250 | -0.188 | 1.438 | 0.000 | 0.000 | 1.000 |
| glm4 | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 0.062 | -0.031 | 0.094 | 0.000 | 0.000 | 1.000 |
| deepseek7b | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1.266 | -0.023 | 1.289 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | -0.008 | 0.008 | -0.016 | 0.000 | 0.000 | -0.500 |

## Top Replacement Effects

| model | selection | op | subspace | source group | cases | delta margin | rank improve | token gain | phrase gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | 2.281 | 1.000 | 0.000 | 0.000 |
| qwen3 | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | 0.125 | 1.000 | 0.000 | 0.000 |
| qwen3 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | -0.031 | 0.500 | 0.000 | null |
| qwen3 | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | -0.188 | 0.000 | 0.000 | null |
| glm4 | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | 1.250 | 1.000 | 0.000 | 0.000 |
| glm4 | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | 0.062 | 1.000 | 0.000 | 0.000 |
| glm4 | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | -0.031 | 0.000 | 0.000 | null |
| glm4 | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | -0.188 | 0.000 | 0.000 | null |
| deepseek7b | `top` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | 1.266 | 1.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | 0.008 | 0.500 | 0.000 | null |
| deepseek7b | `top` | `q_answer_replace` | `positive` | `q_answer_replace:answer_position` | 1 | -0.008 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `matched` | `o_answer_replace` | `positive` | `o_answer_replace:answer_position` | 1 | -0.023 | 1.000 | 0.000 | null |
