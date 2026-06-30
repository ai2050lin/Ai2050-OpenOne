# Phase 791 Upstream Q/K/V and Source-Token Causal Fiber Trace (confirm)

- Status: `complete`
- Test: donor source-token group contribution removal for Phase 788 matched-control attention source units.
- Q/K path is represented by attention mass; V path by source value contribution; O path by projected contribution.
- This is path-level audit, not full Q/K causal patch or generation closure.

## Cross-Model Path Summary

| model | selection | subspace | source group | cases | attn mass | v norm | direct margin | margin drop | top1 loss |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `positive` | `candidate_tokens` | 6 | 0.130 | 29.242 | 5.737 | 2.354 | 0.000 |
| qwen3 | `top` | `positive` | `all_pre_answer` | 6 | 0.647 | 97.452 | 1.464 | 1.260 | 0.000 |
| qwen3 | `top` | `negative` | `candidate_tokens` | 6 | 0.117 | 27.742 | 3.721 | 1.771 | 0.000 |
| qwen3 | `top` | `positive` | `target_value_tokens` | 6 | 0.039 | 18.843 | 3.147 | 1.458 | 0.000 |
| qwen3 | `top` | `negative` | `all_pre_answer` | 6 | 0.631 | 97.054 | 0.464 | 0.917 | 0.000 |
| qwen3 | `top` | `negative` | `target_value_tokens` | 6 | 0.034 | 17.973 | 1.569 | 0.917 | 0.000 |
| qwen3 | `matched` | `negative` | `target_value_tokens` | 6 | 0.026 | 9.362 | 1.147 | 0.406 | 0.000 |
| qwen3 | `matched` | `negative` | `candidate_tokens` | 6 | 0.093 | 14.096 | 1.345 | 0.375 | 0.000 |
| qwen3 | `matched` | `positive` | `answer_prefix` | 6 | 0.078 | 16.658 | -0.036 | 0.044 | 0.000 |
| qwen3 | `matched` | `negative` | `answer_prefix` | 6 | 0.083 | 18.695 | -0.241 | 0.013 | 0.000 |
| qwen3 | `top` | `positive` | `object_tokens` | 6 | 0.010 | 4.253 | 0.034 | 0.005 | 0.000 |
| qwen3 | `top` | `negative` | `relation_tokens` | 6 | 0.013 | 1.920 | 0.018 | -0.003 | 0.000 |
| qwen3 | `matched` | `positive` | `object_tokens` | 6 | 0.018 | 5.341 | -0.025 | -0.005 | 0.000 |
| qwen3 | `top` | `positive` | `relation_tokens` | 6 | 0.013 | 1.897 | 0.004 | -0.005 | 0.000 |
| qwen3 | `matched` | `negative` | `object_tokens` | 6 | 0.019 | 5.911 | 0.036 | -0.008 | 0.000 |
| qwen3 | `top` | `negative` | `object_tokens` | 6 | 0.009 | 3.943 | -0.025 | -0.013 | 0.000 |
| qwen3 | `top` | `positive` | `question` | 6 | 0.102 | 12.038 | -0.077 | -0.013 | 0.000 |
| qwen3 | `top` | `negative` | `question` | 6 | 0.097 | 11.557 | -0.127 | -0.029 | 0.000 |
| qwen3 | `matched` | `positive` | `relation_tokens` | 6 | 0.033 | 5.220 | -0.041 | -0.034 | 0.000 |
| qwen3 | `top` | `negative` | `instruction` | 6 | 0.304 | 6.814 | -0.065 | -0.047 | 0.000 |
| glm4 | `top` | `positive` | `all_pre_answer` | 6 | 0.968 | 90.469 | 0.788 | 1.026 | 0.000 |
| glm4 | `top` | `negative` | `all_pre_answer` | 6 | 0.966 | 90.621 | 0.776 | 1.026 | 0.000 |
| glm4 | `top` | `positive` | `candidate_tokens` | 6 | 0.437 | 87.670 | 0.807 | 1.010 | 0.000 |
| glm4 | `top` | `negative` | `candidate_tokens` | 6 | 0.431 | 87.294 | 0.801 | 0.995 | 0.000 |
| glm4 | `top` | `negative` | `target_value_tokens` | 6 | 0.172 | 60.825 | 0.434 | 0.755 | 0.000 |
| glm4 | `top` | `positive` | `target_value_tokens` | 6 | 0.176 | 61.067 | 0.439 | 0.750 | 0.000 |
| glm4 | `top` | `negative` | `relation_tokens` | 6 | 0.050 | 11.263 | 0.005 | 0.036 | 0.000 |
| glm4 | `top` | `positive` | `question` | 6 | 0.109 | 15.109 | -0.001 | 0.026 | 0.000 |
| glm4 | `top` | `negative` | `object_tokens` | 6 | 0.014 | 5.726 | -0.002 | 0.021 | 0.000 |
| glm4 | `matched` | `negative` | `question` | 6 | 0.115 | 11.853 | -0.023 | 0.016 | 0.000 |
| glm4 | `top` | `negative` | `question` | 6 | 0.113 | 15.720 | -0.004 | 0.016 | 0.000 |
| glm4 | `top` | `positive` | `relation_tokens` | 6 | 0.051 | 11.697 | 0.005 | 0.016 | 0.000 |
| glm4 | `matched` | `positive` | `object_tokens` | 6 | 0.020 | 7.863 | -0.004 | 0.016 | 0.000 |
| glm4 | `matched` | `negative` | `relation_tokens` | 6 | 0.037 | 7.366 | -0.004 | 0.010 | 0.000 |
| glm4 | `matched` | `negative` | `instruction` | 6 | 0.611 | 9.135 | -0.022 | 0.005 | 0.000 |
| glm4 | `top` | `negative` | `instruction` | 6 | 0.421 | 10.132 | -0.018 | 0.005 | 0.000 |
| glm4 | `matched` | `positive` | `question` | 6 | 0.122 | 14.861 | -0.031 | 0.005 | 0.000 |
| glm4 | `matched` | `negative` | `object_tokens` | 6 | 0.019 | 7.078 | -0.003 | 0.005 | 0.000 |
| glm4 | `top` | `negative` | `answer_prefix` | 6 | 0.002 | 0.940 | -0.002 | 0.005 | 0.000 |
| glm4 | `top` | `positive` | `answer_prefix` | 6 | 0.002 | 0.983 | -0.002 | 0.000 | 0.000 |
| deepseek7b | `top` | `positive` | `all_pre_answer` | 6 | 0.782 | 42.459 | 2.198 | 1.329 | 0.000 |
| deepseek7b | `top` | `negative` | `all_pre_answer` | 6 | 0.804 | 43.644 | 1.761 | 1.229 | 0.000 |
| deepseek7b | `top` | `positive` | `instruction` | 6 | 0.387 | 11.673 | 0.401 | 1.148 | 0.000 |
| deepseek7b | `top` | `negative` | `instruction` | 6 | 0.393 | 11.974 | 0.419 | 0.973 | 0.000 |
| deepseek7b | `top` | `positive` | `candidate_tokens` | 6 | 0.238 | 36.583 | 4.190 | 0.378 | 0.000 |
| deepseek7b | `top` | `negative` | `candidate_tokens` | 6 | 0.284 | 39.097 | 2.793 | 0.346 | 0.000 |
| deepseek7b | `top` | `negative` | `target_value_tokens` | 6 | 0.090 | 20.464 | 1.628 | 0.294 | 0.000 |
| deepseek7b | `top` | `positive` | `target_value_tokens` | 6 | 0.062 | 18.344 | 2.766 | 0.299 | 0.000 |
| deepseek7b | `matched` | `negative` | `all_pre_answer` | 6 | 0.821 | 23.013 | 0.484 | 0.139 | 0.000 |
| deepseek7b | `matched` | `negative` | `candidate_tokens` | 6 | 0.141 | 11.091 | 0.739 | 0.128 | 0.000 |
| deepseek7b | `matched` | `positive` | `candidate_tokens` | 6 | 0.137 | 11.202 | 0.646 | 0.128 | 0.000 |
| deepseek7b | `matched` | `negative` | `instruction` | 6 | 0.430 | 7.848 | 0.080 | 0.037 | 0.000 |
| deepseek7b | `matched` | `negative` | `target_value_tokens` | 6 | 0.021 | 4.219 | 0.084 | 0.044 | 0.000 |
| deepseek7b | `top` | `positive` | `answer_prefix` | 6 | 0.049 | 5.503 | 0.148 | 0.034 | 0.000 |
| deepseek7b | `matched` | `positive` | `instruction` | 6 | 0.440 | 8.469 | -0.040 | 0.023 | 0.000 |
| deepseek7b | `top` | `negative` | `object_tokens` | 6 | 0.070 | 18.928 | -0.003 | 0.030 | 0.000 |
| deepseek7b | `matched` | `negative` | `answer_prefix` | 6 | 0.075 | 7.228 | 0.049 | 0.023 | 0.000 |
| deepseek7b | `matched` | `positive` | `target_value_tokens` | 6 | 0.022 | 4.182 | -0.012 | 0.021 | 0.000 |
| deepseek7b | `top` | `negative` | `answer_prefix` | 6 | 0.030 | 3.100 | 0.067 | 0.013 | 0.000 |
| deepseek7b | `top` | `positive` | `object_tokens` | 6 | 0.076 | 20.711 | -0.285 | 0.012 | 0.000 |

## Top Minus Matched Path Specificity

| model | subspace | source group | top mass | matched mass | mass gap | top drop | matched drop | drop gap | direct gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `positive` | `candidate_tokens` | 0.130 | 0.083 | 0.047 | 2.354 | -0.167 | 2.521 | 6.286 |
| qwen3 | `positive` | `target_value_tokens` | 0.039 | 0.021 | 0.018 | 1.458 | -0.109 | 1.568 | 3.577 |
| qwen3 | `positive` | `all_pre_answer` | 0.647 | 0.771 | -0.124 | 1.260 | -0.263 | 1.523 | 1.807 |
| qwen3 | `negative` | `candidate_tokens` | 0.117 | 0.093 | 0.024 | 1.771 | 0.375 | 1.396 | 2.375 |
| qwen3 | `negative` | `all_pre_answer` | 0.631 | 0.789 | -0.158 | 0.917 | -0.049 | 0.966 | 0.001 |
| qwen3 | `negative` | `target_value_tokens` | 0.034 | 0.026 | 0.008 | 0.917 | 0.406 | 0.510 | 0.422 |
| qwen3 | `negative` | `instruction` | 0.304 | 0.518 | -0.213 | -0.047 | -0.250 | 0.203 | -0.123 |
| qwen3 | `positive` | `instruction` | 0.308 | 0.515 | -0.206 | -0.086 | -0.195 | 0.109 | -0.098 |
| qwen3 | `negative` | `relation_tokens` | 0.013 | 0.033 | -0.020 | -0.003 | -0.070 | 0.068 | 0.071 |
| qwen3 | `positive` | `question` | 0.102 | 0.137 | -0.035 | -0.013 | -0.052 | 0.039 | -0.070 |
| qwen3 | `negative` | `question` | 0.097 | 0.142 | -0.044 | -0.029 | -0.057 | 0.029 | -0.163 |
| qwen3 | `positive` | `relation_tokens` | 0.013 | 0.033 | -0.019 | -0.005 | -0.034 | 0.029 | 0.045 |
| qwen3 | `positive` | `object_tokens` | 0.010 | 0.018 | -0.007 | 0.005 | -0.005 | 0.010 | 0.058 |
| qwen3 | `negative` | `object_tokens` | 0.009 | 0.019 | -0.010 | -0.013 | -0.008 | -0.005 | -0.061 |
| qwen3 | `negative` | `answer_prefix` | 0.171 | 0.083 | 0.088 | -0.120 | 0.013 | -0.133 | -0.253 |
| qwen3 | `positive` | `answer_prefix` | 0.172 | 0.078 | 0.095 | -0.120 | 0.044 | -0.164 | -0.530 |
| glm4 | `negative` | `candidate_tokens` | 0.431 | 0.239 | 0.192 | 0.995 | -0.125 | 1.120 | 0.951 |
| glm4 | `positive` | `candidate_tokens` | 0.437 | 0.245 | 0.192 | 1.010 | -0.073 | 1.083 | 0.909 |
| glm4 | `negative` | `all_pre_answer` | 0.966 | 0.967 | -0.001 | 1.026 | -0.089 | 1.115 | 0.971 |
| glm4 | `positive` | `all_pre_answer` | 0.968 | 0.967 | 0.001 | 1.026 | -0.083 | 1.109 | 0.945 |
| glm4 | `negative` | `target_value_tokens` | 0.172 | 0.047 | 0.125 | 0.755 | -0.057 | 0.812 | 0.509 |
| glm4 | `positive` | `target_value_tokens` | 0.176 | 0.047 | 0.129 | 0.750 | -0.021 | 0.771 | 0.484 |
| glm4 | `negative` | `relation_tokens` | 0.050 | 0.037 | 0.013 | 0.036 | 0.010 | 0.026 | 0.009 |
| glm4 | `positive` | `relation_tokens` | 0.051 | 0.040 | 0.011 | 0.016 | -0.005 | 0.021 | 0.015 |
| glm4 | `positive` | `question` | 0.109 | 0.122 | -0.014 | 0.026 | 0.005 | 0.021 | 0.030 |
| glm4 | `positive` | `answer_prefix` | 0.002 | 0.001 | 0.001 | 0.000 | -0.016 | 0.016 | -0.002 |
| glm4 | `negative` | `answer_prefix` | 0.002 | 0.001 | 0.001 | 0.005 | -0.010 | 0.016 | -0.002 |
| glm4 | `negative` | `object_tokens` | 0.014 | 0.019 | -0.005 | 0.021 | 0.005 | 0.016 | 0.001 |
| glm4 | `positive` | `instruction` | 0.420 | 0.599 | -0.179 | 0.000 | -0.005 | 0.005 | 0.009 |
| glm4 | `negative` | `instruction` | 0.421 | 0.611 | -0.191 | 0.005 | 0.005 | 0.000 | 0.004 |
| glm4 | `negative` | `question` | 0.113 | 0.115 | -0.002 | 0.016 | 0.016 | 0.000 | 0.019 |
| glm4 | `positive` | `object_tokens` | 0.011 | 0.020 | -0.009 | 0.000 | 0.016 | -0.016 | 0.004 |
| deepseek7b | `positive` | `all_pre_answer` | 0.782 | 0.813 | -0.031 | 1.329 | -0.072 | 1.401 | 2.077 |
| deepseek7b | `positive` | `instruction` | 0.387 | 0.440 | -0.053 | 1.148 | 0.023 | 1.124 | 0.441 |
| deepseek7b | `negative` | `all_pre_answer` | 0.804 | 0.821 | -0.017 | 1.229 | 0.139 | 1.090 | 1.278 |
| deepseek7b | `negative` | `instruction` | 0.393 | 0.430 | -0.037 | 0.973 | 0.037 | 0.936 | 0.339 |
| deepseek7b | `positive` | `target_value_tokens` | 0.062 | 0.022 | 0.040 | 0.299 | 0.021 | 0.279 | 2.778 |
| deepseek7b | `positive` | `candidate_tokens` | 0.238 | 0.137 | 0.101 | 0.378 | 0.128 | 0.250 | 3.544 |
| deepseek7b | `negative` | `target_value_tokens` | 0.090 | 0.021 | 0.070 | 0.294 | 0.044 | 0.250 | 1.544 |
| deepseek7b | `negative` | `candidate_tokens` | 0.284 | 0.141 | 0.143 | 0.346 | 0.128 | 0.219 | 2.054 |
| deepseek7b | `positive` | `question` | 0.228 | 0.235 | -0.007 | -0.001 | -0.122 | 0.121 | -0.309 |
| deepseek7b | `negative` | `object_tokens` | 0.070 | 0.039 | 0.031 | 0.030 | -0.081 | 0.111 | 0.198 |
| deepseek7b | `positive` | `answer_prefix` | 0.049 | 0.070 | -0.021 | 0.034 | -0.044 | 0.078 | 0.173 |
| deepseek7b | `positive` | `object_tokens` | 0.076 | 0.036 | 0.040 | 0.012 | -0.061 | 0.073 | -0.069 |
| deepseek7b | `negative` | `relation_tokens` | 0.049 | 0.052 | -0.003 | -0.011 | -0.038 | 0.027 | -0.202 |
| deepseek7b | `positive` | `relation_tokens` | 0.045 | 0.051 | -0.006 | -0.026 | -0.042 | 0.016 | -0.154 |
| deepseek7b | `negative` | `question` | 0.227 | 0.240 | -0.013 | -0.010 | -0.013 | 0.003 | -0.199 |
| deepseek7b | `negative` | `answer_prefix` | 0.030 | 0.075 | -0.045 | 0.013 | 0.023 | -0.010 | 0.017 |

## Boundary

- Attention mass is a Q/K proxy, not an independent Q/K patch.
- The intervention removes source-group value contribution at donor prompt answer site.
- Positive margin drop means this source path supported the target-vs-contrast margin in donor context.
