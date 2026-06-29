# Phase 751 Natural Attention Head Mechanism Backtrace (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source attention mass plus causal source V/O contribution removal.

| model | context | site | subunit | source | n | attn mass | source target contrib | source route supp contrib | remove target drop | route release | coverage | margin drop | top1 loss | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | target_record_line | 1 | 0.260 | 6.881 | 0.722 | 0.375 | 0.250 | 1.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_record_line | 1 | 0.260 | 6.881 | 0.722 | 0.375 | 0.250 | 1.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | records_all | 1 | 0.293 | 6.923 | 0.737 | 0.250 | 0.625 | 2.00 | 0.344 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | records_all | 1 | 0.293 | 6.923 | 0.737 | 0.250 | 0.625 | 2.00 | 0.344 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | target_value_tokens | 1 | 0.247 | 6.867 | 0.722 | 0.250 | 0.500 | 2.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_value_tokens | 1 | 0.247 | 6.867 | 0.722 | 0.250 | 0.500 | 2.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | records_other | 1 | 0.039 | 0.070 | 0.027 | 0.125 | 0.000 | 0.00 | 0.062 | 0.000 | `small_or_unclear` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | records_all | 1 | 0.247 | 7.180 | 1.042 | 0.000 | 1.125 | 4.00 | 0.281 | 0.000 | `route_suppressor_content` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | target_value_tokens | 1 | 0.188 | 7.053 | 0.980 | 0.000 | 0.875 | 3.00 | 0.219 | 0.000 | `route_suppressor_content` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH2 | target_record_line | 1 | 0.209 | 7.110 | 1.016 | 0.000 | 0.875 | 2.00 | 0.188 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH2 | records_other | 1 | 0.024 | 0.069 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | records_all | 1 | 0.161 | 0.615 | 0.016 | 0.000 | 0.000 | 0.00 | -0.125 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H0 | records_all | 1 | 0.161 | 0.615 | 0.016 | 0.000 | 0.000 | 0.00 | -0.125 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH2 | target_value_tokens | 1 | 0.164 | 1.832 | 0.092 | -0.125 | 0.250 | 2.00 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH2 | records_all | 1 | 0.199 | 1.971 | 0.079 | -0.125 | 0.000 | 0.00 | -0.125 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | target_value_tokens | 1 | 0.129 | 0.544 | 0.028 | -0.125 | 0.000 | 0.00 | -0.125 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H0 | target_value_tokens | 1 | 0.129 | 0.544 | 0.028 | -0.125 | 0.000 | 0.00 | -0.125 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:topH1 | records_other | 1 | 0.033 | 0.042 | 0.025 | -0.125 | 0.000 | 0.00 | -0.125 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | target_value_tokens | 1 | 0.009 | -0.002 | 0.004 | 0.062 | 0.125 | 2.00 | 0.094 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | records_all | 1 | 0.625 | 0.036 | 0.153 | 0.062 | 0.062 | 1.00 | 0.078 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | records_other | 1 | 0.483 | 0.043 | 0.139 | 0.062 | 0.062 | 1.00 | 0.078 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH1 | records_other | 1 | 0.180 | 0.001 | 0.051 | 0.000 | 0.375 | 4.00 | 0.094 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | records_other | 1 | 0.180 | 0.001 | 0.051 | 0.000 | 0.375 | 4.00 | 0.094 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | target_record_line | 1 | 0.142 | -0.007 | 0.020 | 0.000 | 0.125 | 2.00 | 0.031 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH2 | records_other | 1 | 0.303 | 0.004 | 0.071 | 0.000 | 0.125 | 2.00 | -0.047 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:topH2 | records_other | 1 | 0.123 | 0.034 | 0.013 | 0.000 | 0.062 | 1.00 | 0.031 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | records_all | 1 | 0.304 | -0.003 | 0.005 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | records_other | 1 | 0.158 | 0.001 | 0.003 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | target_record_line | 1 | 0.146 | -0.004 | 0.002 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH1 | target_value_tokens | 1 | 0.144 | -0.024 | 0.041 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | target_value_tokens | 1 | 0.144 | -0.024 | 0.041 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:topH1 | records_other | 1 | 0.053 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H2 | records_other | 1 | 0.053 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | target_value_tokens | 1 | 0.011 | -0.003 | 0.003 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:topH1 | records_all | 1 | 0.555 | -0.015 | 0.144 | -0.062 | 0.312 | 3.00 | 0.016 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H2 | records_all | 1 | 0.555 | -0.015 | 0.144 | -0.062 | 0.312 | 3.00 | 0.016 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | target_record_line | 1 | 0.333 | 1.036 | 0.508 | 0.750 | 0.000 | 0.00 | 0.646 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | target_value_tokens | 1 | 0.316 | 1.045 | 0.516 | 0.500 | 0.625 | 3.00 | 0.708 | 0.000 | `mixed_target_support_and_route_guard` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | records_all | 1 | 0.558 | 0.926 | 0.577 | 0.500 | 0.500 | 3.00 | 0.667 | 0.000 | `mixed_target_support_and_route_guard` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | records_all | 1 | 0.419 | 0.172 | 0.072 | 0.250 | 0.125 | 1.00 | 0.266 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH1 | target_record_line | 1 | 0.476 | 0.286 | 0.088 | 0.125 | 0.375 | 3.00 | 0.250 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H7 | target_record_line | 1 | 0.476 | 0.286 | 0.088 | 0.125 | 0.375 | 3.00 | 0.250 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH2 | target_record_line | 1 | 0.248 | 0.269 | 0.096 | 0.125 | 0.250 | 3.00 | 0.208 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | target_record_line | 1 | 0.095 | 0.452 | 0.054 | 0.125 | 0.188 | 2.00 | 0.172 | 0.000 | `readout_target_aligned_observational` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH2 | records_all | 1 | 0.311 | 0.228 | 0.097 | 0.000 | 0.938 | 3.00 | 0.312 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH1 | records_all | 1 | 0.574 | 0.246 | 0.075 | 0.000 | 0.688 | 3.00 | 0.229 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H7 | records_all | 1 | 0.574 | 0.246 | 0.075 | 0.000 | 0.688 | 3.00 | 0.229 | 0.000 | `route_suppressor_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH2 | target_record_line | 1 | 0.082 | 0.152 | 0.065 | 0.000 | 0.438 | 4.00 | 0.109 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH1 | target_value_tokens | 1 | 0.042 | 0.136 | 0.032 | 0.000 | 0.438 | 4.00 | 0.109 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H7 | target_value_tokens | 1 | 0.042 | 0.136 | 0.032 | 0.000 | 0.438 | 4.00 | 0.109 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH2 | target_value_tokens | 1 | 0.022 | 0.132 | 0.037 | 0.000 | 0.250 | 3.00 | 0.062 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH1 | target_value_tokens | 1 | 0.091 | 0.261 | 0.083 | 0.000 | 0.125 | 2.00 | 0.042 | 0.000 | `readout_target_aligned_observational` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H7 | target_value_tokens | 1 | 0.091 | 0.261 | 0.083 | 0.000 | 0.125 | 2.00 | 0.042 | 0.000 | `readout_target_aligned_observational` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH2 | target_value_tokens | 1 | 0.050 | 0.240 | 0.090 | 0.000 | 0.125 | 2.00 | 0.042 | 0.000 | `readout_target_aligned_observational` |

## Strict Interpretation

- QK/pattern evidence is attention mass only.
- V/O/content evidence is causal source contribution removal before o_proj.
- A source group with high attention but weak removal effect is not a causal content source.
- This is head/source-path evidence, not neuron-level evidence.
