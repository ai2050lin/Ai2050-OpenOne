# Phase 791 Upstream Q/K/V and Source-Token Causal Fiber Trace (main)

- Status: `complete`
- Test: donor source-token group contribution removal for Phase 788 matched-control attention source units.
- Q/K path is represented by attention mass; V path by source value contribution; O path by projected contribution.
- This is path-level audit, not full Q/K causal patch or generation closure.

## Cross-Model Path Summary

| model | selection | subspace | source group | cases | attn mass | v norm | direct margin | margin drop | top1 loss |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `positive` | `candidate_tokens` | 4 | 0.115 | 27.592 | 4.659 | 2.172 | 0.000 |
| qwen3 | `top` | `negative` | `candidate_tokens` | 4 | 0.109 | 26.522 | 2.785 | 1.625 | 0.000 |
| qwen3 | `top` | `positive` | `all_pre_answer` | 4 | 0.649 | 81.619 | 0.995 | 1.055 | 0.000 |
| qwen3 | `top` | `positive` | `target_value_tokens` | 4 | 0.037 | 18.292 | 2.293 | 1.375 | 0.000 |
| qwen3 | `top` | `negative` | `all_pre_answer` | 4 | 0.634 | 80.949 | 0.106 | 0.734 | 0.000 |
| qwen3 | `top` | `negative` | `target_value_tokens` | 4 | 0.034 | 17.781 | 0.774 | 0.812 | 0.000 |
| qwen3 | `matched` | `negative` | `candidate_tokens` | 4 | 0.086 | 13.073 | 1.206 | 0.375 | 0.000 |
| qwen3 | `matched` | `negative` | `target_value_tokens` | 4 | 0.025 | 8.600 | 1.073 | 0.391 | 0.000 |
| qwen3 | `matched` | `positive` | `answer_prefix` | 4 | 0.065 | 13.731 | -0.017 | 0.035 | 0.000 |
| qwen3 | `top` | `positive` | `object_tokens` | 4 | 0.010 | 3.979 | -0.011 | 0.008 | 0.000 |
| qwen3 | `top` | `negative` | `relation_tokens` | 4 | 0.015 | 2.330 | 0.027 | -0.020 | 0.000 |
| qwen3 | `top` | `positive` | `relation_tokens` | 4 | 0.015 | 2.199 | 0.009 | -0.023 | 0.000 |
| qwen3 | `matched` | `negative` | `answer_prefix` | 4 | 0.080 | 17.464 | -0.277 | -0.027 | 0.000 |
| qwen3 | `top` | `negative` | `object_tokens` | 4 | 0.009 | 3.844 | -0.043 | -0.027 | 0.000 |
| qwen3 | `matched` | `negative` | `object_tokens` | 4 | 0.020 | 6.609 | 0.020 | -0.035 | 0.000 |
| qwen3 | `matched` | `positive` | `object_tokens` | 4 | 0.020 | 6.480 | -0.016 | -0.039 | 0.000 |
| qwen3 | `matched` | `positive` | `relation_tokens` | 4 | 0.036 | 5.581 | -0.049 | -0.043 | 0.000 |
| qwen3 | `matched` | `negative` | `relation_tokens` | 4 | 0.036 | 5.587 | -0.064 | -0.082 | 0.000 |
| qwen3 | `top` | `positive` | `answer_prefix` | 4 | 0.166 | 31.011 | -0.522 | -0.117 | 0.000 |
| qwen3 | `top` | `negative` | `answer_prefix` | 4 | 0.158 | 30.478 | -0.463 | -0.125 | 0.000 |
| glm4 | `top` | `negative` | `all_pre_answer` | 4 | 0.972 | 95.403 | 0.229 | 0.852 | 0.000 |
| glm4 | `top` | `positive` | `all_pre_answer` | 4 | 0.972 | 95.135 | 0.233 | 0.852 | 0.000 |
| glm4 | `top` | `positive` | `candidate_tokens` | 4 | 0.471 | 92.068 | 0.260 | 0.859 | 0.000 |
| glm4 | `top` | `negative` | `candidate_tokens` | 4 | 0.468 | 91.763 | 0.262 | 0.836 | 0.000 |
| glm4 | `top` | `negative` | `target_value_tokens` | 4 | 0.183 | 66.393 | 0.059 | 0.688 | 0.000 |
| glm4 | `top` | `positive` | `target_value_tokens` | 4 | 0.186 | 66.683 | 0.056 | 0.680 | 0.000 |
| glm4 | `top` | `negative` | `relation_tokens` | 4 | 0.048 | 10.600 | 0.001 | 0.039 | 0.000 |
| glm4 | `top` | `positive` | `relation_tokens` | 4 | 0.048 | 11.195 | 0.000 | 0.023 | 0.000 |
| glm4 | `matched` | `negative` | `object_tokens` | 4 | 0.014 | 5.458 | -0.003 | 0.016 | 0.000 |
| glm4 | `matched` | `positive` | `object_tokens` | 4 | 0.011 | 4.797 | 0.000 | 0.016 | 0.000 |
| glm4 | `matched` | `negative` | `relation_tokens` | 4 | 0.035 | 7.142 | -0.005 | 0.008 | 0.000 |
| glm4 | `top` | `negative` | `object_tokens` | 4 | 0.012 | 4.863 | 0.001 | 0.008 | 0.000 |
| glm4 | `top` | `negative` | `answer_prefix` | 4 | 0.001 | 0.303 | -0.001 | 0.008 | 0.000 |
| glm4 | `matched` | `positive` | `relation_tokens` | 4 | 0.041 | 11.049 | -0.014 | 0.000 | 0.000 |
| glm4 | `top` | `positive` | `answer_prefix` | 4 | 0.001 | 0.365 | -0.001 | 0.000 | 0.000 |
| glm4 | `top` | `positive` | `object_tokens` | 4 | 0.013 | 5.052 | 0.001 | -0.008 | 0.000 |
| glm4 | `matched` | `negative` | `answer_prefix` | 4 | 0.001 | 0.579 | 0.000 | -0.023 | 0.000 |
| glm4 | `matched` | `positive` | `answer_prefix` | 4 | 0.001 | 0.355 | 0.000 | -0.023 | 0.000 |
| glm4 | `matched` | `positive` | `target_value_tokens` | 4 | 0.046 | 12.306 | -0.056 | -0.031 | 0.000 |
| glm4 | `matched` | `negative` | `target_value_tokens` | 4 | 0.047 | 12.872 | -0.066 | -0.055 | 0.000 |
| deepseek7b | `top` | `negative` | `all_pre_answer` | 4 | 0.831 | 46.373 | 0.943 | 1.210 | 0.000 |
| deepseek7b | `top` | `positive` | `all_pre_answer` | 4 | 0.799 | 44.002 | 1.526 | 1.170 | 0.000 |
| deepseek7b | `top` | `positive` | `candidate_tokens` | 4 | 0.272 | 40.503 | 3.474 | 0.242 | 0.000 |
| deepseek7b | `top` | `positive` | `target_value_tokens` | 4 | 0.072 | 22.664 | 2.810 | 0.266 | 0.000 |
| deepseek7b | `top` | `negative` | `target_value_tokens` | 4 | 0.115 | 25.844 | 1.099 | 0.242 | 0.000 |
| deepseek7b | `top` | `negative` | `candidate_tokens` | 4 | 0.337 | 44.246 | 1.356 | 0.164 | 0.000 |
| deepseek7b | `matched` | `negative` | `candidate_tokens` | 4 | 0.150 | 12.032 | 0.665 | 0.117 | 0.000 |
| deepseek7b | `top` | `negative` | `object_tokens` | 4 | 0.062 | 17.196 | 0.253 | 0.080 | 0.000 |
| deepseek7b | `matched` | `negative` | `target_value_tokens` | 4 | 0.024 | 4.906 | 0.103 | 0.047 | 0.000 |
| deepseek7b | `matched` | `negative` | `all_pre_answer` | 4 | 0.822 | 24.452 | 0.245 | 0.018 | 0.000 |
| deepseek7b | `matched` | `negative` | `answer_prefix` | 4 | 0.081 | 7.085 | 0.035 | 0.018 | 0.000 |
| deepseek7b | `matched` | `positive` | `candidate_tokens` | 4 | 0.140 | 12.116 | 0.490 | 0.008 | 0.000 |
| deepseek7b | `matched` | `positive` | `target_value_tokens` | 4 | 0.026 | 4.845 | -0.045 | 0.008 | 0.000 |
| deepseek7b | `top` | `positive` | `answer_prefix` | 4 | 0.045 | 5.323 | 0.084 | -0.004 | 0.000 |
| deepseek7b | `top` | `negative` | `answer_prefix` | 4 | 0.028 | 3.247 | 0.038 | -0.004 | 0.000 |
| deepseek7b | `top` | `positive` | `object_tokens` | 4 | 0.073 | 20.145 | -0.204 | -0.023 | 0.000 |
| deepseek7b | `matched` | `positive` | `answer_prefix` | 4 | 0.068 | 6.425 | 0.055 | -0.061 | 0.000 |
| deepseek7b | `matched` | `positive` | `object_tokens` | 4 | 0.041 | 9.196 | -0.208 | -0.061 | 0.000 |
| deepseek7b | `matched` | `positive` | `all_pre_answer` | 4 | 0.806 | 22.998 | -0.021 | -0.066 | 0.000 |
| deepseek7b | `matched` | `positive` | `relation_tokens` | 4 | 0.057 | 6.856 | -0.084 | -0.066 | 0.000 |

## Top Minus Matched Path Specificity

| model | subspace | source group | top mass | matched mass | mass gap | top drop | matched drop | drop gap | direct gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `positive` | `candidate_tokens` | 0.115 | 0.084 | 0.032 | 2.172 | -0.172 | 2.344 | 5.145 |
| qwen3 | `positive` | `target_value_tokens` | 0.037 | 0.022 | 0.015 | 1.375 | -0.164 | 1.539 | 2.738 |
| qwen3 | `positive` | `all_pre_answer` | 0.649 | 0.778 | -0.128 | 1.055 | -0.316 | 1.371 | 1.272 |
| qwen3 | `negative` | `candidate_tokens` | 0.109 | 0.086 | 0.023 | 1.625 | 0.375 | 1.250 | 1.579 |
| qwen3 | `negative` | `all_pre_answer` | 0.634 | 0.795 | -0.161 | 0.734 | -0.168 | 0.902 | -0.217 |
| qwen3 | `negative` | `target_value_tokens` | 0.034 | 0.025 | 0.009 | 0.812 | 0.391 | 0.422 | -0.299 |
| qwen3 | `negative` | `relation_tokens` | 0.015 | 0.036 | -0.021 | -0.020 | -0.082 | 0.062 | 0.091 |
| qwen3 | `positive` | `object_tokens` | 0.010 | 0.020 | -0.010 | 0.008 | -0.039 | 0.047 | 0.005 |
| qwen3 | `positive` | `relation_tokens` | 0.015 | 0.036 | -0.021 | -0.023 | -0.043 | 0.020 | 0.057 |
| qwen3 | `negative` | `object_tokens` | 0.009 | 0.020 | -0.011 | -0.027 | -0.035 | 0.008 | -0.063 |
| qwen3 | `negative` | `answer_prefix` | 0.158 | 0.080 | 0.078 | -0.125 | -0.027 | -0.098 | -0.186 |
| qwen3 | `positive` | `answer_prefix` | 0.166 | 0.065 | 0.101 | -0.117 | 0.035 | -0.152 | -0.505 |
| glm4 | `positive` | `candidate_tokens` | 0.471 | 0.287 | 0.185 | 0.859 | -0.102 | 0.961 | 0.362 |
| glm4 | `negative` | `candidate_tokens` | 0.468 | 0.275 | 0.192 | 0.836 | -0.109 | 0.945 | 0.381 |
| glm4 | `negative` | `all_pre_answer` | 0.972 | 0.969 | 0.003 | 0.852 | -0.086 | 0.938 | 0.400 |
| glm4 | `positive` | `all_pre_answer` | 0.972 | 0.972 | 0.000 | 0.852 | -0.078 | 0.930 | 0.400 |
| glm4 | `negative` | `target_value_tokens` | 0.183 | 0.047 | 0.136 | 0.688 | -0.055 | 0.742 | 0.125 |
| glm4 | `positive` | `target_value_tokens` | 0.186 | 0.046 | 0.140 | 0.680 | -0.031 | 0.711 | 0.112 |
| glm4 | `negative` | `relation_tokens` | 0.048 | 0.035 | 0.013 | 0.039 | 0.008 | 0.031 | 0.006 |
| glm4 | `negative` | `answer_prefix` | 0.001 | 0.001 | -0.001 | 0.008 | -0.023 | 0.031 | -0.001 |
| glm4 | `positive` | `relation_tokens` | 0.048 | 0.041 | 0.007 | 0.023 | 0.000 | 0.023 | 0.014 |
| glm4 | `positive` | `answer_prefix` | 0.001 | 0.001 | 0.000 | 0.000 | -0.023 | 0.023 | -0.001 |
| glm4 | `negative` | `object_tokens` | 0.012 | 0.014 | -0.003 | 0.008 | 0.016 | -0.008 | 0.004 |
| glm4 | `positive` | `object_tokens` | 0.013 | 0.011 | 0.002 | -0.008 | 0.016 | -0.023 | 0.001 |
| deepseek7b | `positive` | `all_pre_answer` | 0.799 | 0.806 | -0.007 | 1.170 | -0.066 | 1.236 | 1.547 |
| deepseek7b | `negative` | `all_pre_answer` | 0.831 | 0.822 | 0.009 | 1.210 | 0.018 | 1.192 | 0.698 |
| deepseek7b | `positive` | `target_value_tokens` | 0.072 | 0.026 | 0.047 | 0.266 | 0.008 | 0.258 | 2.855 |
| deepseek7b | `positive` | `candidate_tokens` | 0.272 | 0.140 | 0.132 | 0.242 | 0.008 | 0.234 | 2.984 |
| deepseek7b | `negative` | `target_value_tokens` | 0.115 | 0.024 | 0.090 | 0.242 | 0.047 | 0.195 | 0.996 |
| deepseek7b | `negative` | `object_tokens` | 0.062 | 0.046 | 0.016 | 0.080 | -0.112 | 0.192 | 0.442 |
| deepseek7b | `positive` | `answer_prefix` | 0.045 | 0.068 | -0.023 | -0.004 | -0.061 | 0.057 | 0.029 |
| deepseek7b | `negative` | `candidate_tokens` | 0.337 | 0.150 | 0.187 | 0.164 | 0.117 | 0.047 | 0.691 |
| deepseek7b | `positive` | `object_tokens` | 0.073 | 0.041 | 0.032 | -0.023 | -0.061 | 0.037 | 0.004 |
| deepseek7b | `positive` | `relation_tokens` | 0.053 | 0.057 | -0.005 | -0.068 | -0.066 | -0.002 | -0.170 |
| deepseek7b | `negative` | `relation_tokens` | 0.060 | 0.059 | 0.000 | -0.080 | -0.067 | -0.013 | -0.239 |
| deepseek7b | `negative` | `answer_prefix` | 0.028 | 0.081 | -0.053 | -0.004 | 0.018 | -0.022 | 0.003 |

## Boundary

- Attention mass is a Q/K proxy, not an independent Q/K patch.
- The intervention removes source-group value contribution at donor prompt answer site.
- Positive margin drop means this source path supported the target-vs-contrast margin in donor context.
