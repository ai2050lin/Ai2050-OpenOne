# Phase 791 Upstream Q/K/V and Source-Token Causal Fiber Trace (smoke)

- Status: `complete`
- Test: donor source-token group contribution removal for Phase 788 matched-control attention source units.
- Q/K path is represented by attention mass; V path by source value contribution; O path by projected contribution.
- This is path-level audit, not full Q/K causal patch or generation closure.

## Cross-Model Path Summary

| model | selection | subspace | source group | cases | attn mass | v norm | direct margin | margin drop | top1 loss |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `positive` | `all_pre_answer` | 1 | 0.685 | 203.127 | 3.443 | 2.406 | 0.000 |
| qwen3 | `matched` | `positive` | `all_pre_answer` | 1 | 0.602 | 23.287 | -0.524 | 0.156 | 0.000 |
| qwen3 | `matched` | `positive` | `answer_prefix` | 1 | 0.183 | 19.702 | -0.283 | 0.125 | 0.000 |
| qwen3 | `matched` | `positive` | `relation_tokens` | 1 | 0.004 | 0.300 | 0.005 | 0.031 | 0.000 |
| qwen3 | `matched` | `positive` | `object_tokens` | 1 | 0.001 | 0.091 | -0.000 | 0.000 | 0.000 |
| qwen3 | `top` | `positive` | `relation_tokens` | 1 | 0.009 | 0.571 | -0.000 | 0.000 | 0.000 |
| qwen3 | `top` | `positive` | `object_tokens` | 1 | 0.015 | 4.145 | 0.028 | -0.031 | 0.000 |
| qwen3 | `top` | `positive` | `answer_prefix` | 1 | 0.323 | 31.709 | -0.244 | -0.188 | 0.000 |
| glm4 | `top` | `positive` | `all_pre_answer` | 1 | 0.990 | 87.261 | 1.696 | 1.531 | 0.000 |
| glm4 | `matched` | `positive` | `answer_prefix` | 1 | 0.004 | 0.798 | -0.003 | 0.000 | 0.000 |
| glm4 | `matched` | `positive` | `object_tokens` | 1 | 0.005 | 0.830 | 0.000 | 0.000 | 0.000 |
| glm4 | `matched` | `positive` | `relation_tokens` | 1 | 0.001 | 0.257 | 0.001 | 0.000 | 0.000 |
| glm4 | `top` | `positive` | `answer_prefix` | 1 | 0.001 | 0.152 | -0.000 | 0.000 | 0.000 |
| glm4 | `top` | `positive` | `object_tokens` | 1 | 0.008 | 1.481 | -0.002 | 0.000 | 0.000 |
| glm4 | `top` | `positive` | `relation_tokens` | 1 | 0.006 | 0.916 | 0.001 | 0.000 | 0.000 |
| glm4 | `matched` | `positive` | `all_pre_answer` | 1 | 0.968 | 28.463 | -0.245 | -0.188 | 0.000 |
| deepseek7b | `top` | `positive` | `all_pre_answer` | 1 | 0.728 | 29.547 | 0.380 | 1.086 | 0.000 |
| deepseek7b | `matched` | `positive` | `all_pre_answer` | 1 | 0.958 | 33.665 | 2.373 | 0.516 | 0.000 |
| deepseek7b | `matched` | `positive` | `object_tokens` | 1 | 0.186 | 24.468 | -0.206 | 0.062 | 0.000 |
| deepseek7b | `top` | `positive` | `answer_prefix` | 1 | 0.096 | 5.810 | 0.037 | 0.047 | 0.000 |
| deepseek7b | `top` | `positive` | `object_tokens` | 1 | 0.023 | 1.976 | -0.086 | 0.031 | 0.000 |
| deepseek7b | `matched` | `positive` | `relation_tokens` | 1 | 0.118 | 14.907 | -0.020 | 0.016 | 0.000 |
| deepseek7b | `matched` | `positive` | `answer_prefix` | 1 | 0.025 | 0.739 | 0.007 | 0.000 | 0.000 |
| deepseek7b | `top` | `positive` | `relation_tokens` | 1 | 0.009 | 1.312 | 0.046 | -0.094 | 0.000 |

## Top Minus Matched Path Specificity

| model | subspace | source group | top mass | matched mass | mass gap | top drop | matched drop | drop gap | direct gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `positive` | `all_pre_answer` | 0.685 | 0.602 | 0.083 | 2.406 | 0.156 | 2.250 | 3.967 |
| qwen3 | `positive` | `object_tokens` | 0.015 | 0.001 | 0.014 | -0.031 | 0.000 | -0.031 | 0.028 |
| qwen3 | `positive` | `relation_tokens` | 0.009 | 0.004 | 0.005 | 0.000 | 0.031 | -0.031 | -0.005 |
| qwen3 | `positive` | `answer_prefix` | 0.323 | 0.183 | 0.140 | -0.188 | 0.125 | -0.312 | 0.040 |
| glm4 | `positive` | `all_pre_answer` | 0.990 | 0.968 | 0.022 | 1.531 | -0.188 | 1.719 | 1.941 |
| glm4 | `positive` | `answer_prefix` | 0.001 | 0.004 | -0.003 | 0.000 | 0.000 | 0.000 | 0.003 |
| glm4 | `positive` | `object_tokens` | 0.008 | 0.005 | 0.003 | 0.000 | 0.000 | 0.000 | -0.002 |
| glm4 | `positive` | `relation_tokens` | 0.006 | 0.001 | 0.004 | 0.000 | 0.000 | 0.000 | 0.001 |
| deepseek7b | `positive` | `all_pre_answer` | 0.728 | 0.958 | -0.230 | 1.086 | 0.516 | 0.570 | -1.993 |
| deepseek7b | `positive` | `answer_prefix` | 0.096 | 0.025 | 0.072 | 0.047 | 0.000 | 0.047 | 0.030 |
| deepseek7b | `positive` | `object_tokens` | 0.023 | 0.186 | -0.163 | 0.031 | 0.062 | -0.031 | 0.119 |
| deepseek7b | `positive` | `relation_tokens` | 0.009 | 0.118 | -0.109 | -0.094 | 0.016 | -0.109 | 0.065 |

## Boundary

- Attention mass is a Q/K proxy, not an independent Q/K patch.
- The intervention removes source-group value contribution at donor prompt answer site.
- Positive margin drop means this source path supported the target-vs-contrast margin in donor context.
