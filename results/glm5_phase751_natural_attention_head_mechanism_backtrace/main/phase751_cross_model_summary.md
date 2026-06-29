# Phase 751 Natural Attention Head Mechanism Backtrace (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source attention mass plus causal source V/O contribution removal.

| model | context | site | subunit | source | n | attn mass | source target contrib | source route supp contrib | remove target drop | route release | coverage | margin drop | top1 loss | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | records_all | 1 | 0.379 | 13.785 | 0.076 | 0.875 | 0.000 | 0.00 | 0.250 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | target_record_line | 1 | 0.313 | 13.557 | 0.070 | 0.875 | 0.000 | 0.00 | 0.250 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | target_value_tokens | 1 | 0.266 | 12.925 | 0.106 | 0.750 | 0.000 | 0.00 | 0.188 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | records_all | 2 | 0.550 | 13.235 | 0.265 | 0.625 | 0.000 | 0.00 | 0.109 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | target_value_tokens | 2 | 0.519 | 13.205 | 0.269 | 0.562 | 0.000 | 0.00 | 0.078 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | target_record_line | 2 | 0.522 | 13.207 | 0.269 | 0.438 | 0.062 | 0.50 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | target_value_tokens | 4 | 0.124 | 6.150 | 0.522 | 0.406 | 0.250 | 1.50 | 0.234 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_record_line | 1 | 0.260 | 6.881 | 0.722 | 0.375 | 0.250 | 1.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | records_all | 4 | 0.254 | 6.483 | 0.758 | 0.344 | 0.344 | 1.50 | 0.164 | 0.000 | `qk_pattern_visible_content_weak` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | target_record_line | 4 | 0.162 | 6.256 | 0.562 | 0.344 | 0.250 | 1.25 | 0.180 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | records_all | 4 | 0.316 | 6.771 | 0.137 | 0.312 | 0.031 | 0.25 | 0.031 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH2 | target_value_tokens | 4 | 0.143 | 4.875 | 0.318 | 0.281 | 0.188 | 1.50 | 0.156 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | records_all | 1 | 0.293 | 6.923 | 0.737 | 0.250 | 0.625 | 2.00 | 0.344 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_value_tokens | 1 | 0.247 | 6.867 | 0.722 | 0.250 | 0.500 | 2.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH2 | records_all | 4 | 0.292 | 5.064 | 0.363 | 0.250 | 0.250 | 1.50 | 0.055 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:H30 | target_value_tokens | 1 | 0.307 | 11.239 | 0.740 | 0.250 | 0.000 | 0.00 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:topH1 | target_value_tokens | 4 | 0.292 | 6.738 | 0.141 | 0.250 | 0.000 | 0.00 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH2 | target_record_line | 4 | 0.184 | 4.996 | 0.328 | 0.219 | 0.188 | 1.25 | 0.062 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | records_all | 1 | 0.622 | 0.986 | 0.036 | 0.250 | 0.000 | 0.00 | 0.172 | 0.000 | `target_support_content` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | target_record_line | 1 | 0.514 | 0.980 | 0.035 | 0.250 | 0.000 | 0.00 | 0.172 | 0.000 | `target_support_content` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | target_value_tokens | 1 | 0.459 | 0.962 | 0.036 | 0.250 | 0.000 | 0.00 | 0.172 | 0.000 | `target_support_content` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH2 | records_all | 4 | 0.250 | 0.375 | 0.029 | 0.188 | 0.031 | 0.50 | 0.120 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH2 | target_value_tokens | 4 | 0.146 | 0.368 | 0.014 | 0.172 | 0.000 | 0.00 | 0.116 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH2 | target_record_line | 4 | 0.179 | 0.372 | 0.022 | 0.141 | 0.016 | 0.25 | 0.104 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H7 | relation_tokens | 1 | 0.002 | 0.000 | 0.000 | 0.125 | 0.000 | 0.00 | 0.094 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | target_value_tokens | 4 | 0.209 | 0.388 | 0.033 | 0.078 | 0.094 | 1.25 | 0.077 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH1 | target_record_line | 4 | 0.135 | 0.246 | 0.009 | 0.078 | 0.016 | 0.25 | 0.049 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | records_all | 4 | 0.605 | 0.406 | 0.116 | 0.062 | 0.125 | 1.50 | 0.060 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | target_record_line | 4 | 0.267 | 0.389 | 0.039 | 0.062 | 0.078 | 1.25 | 0.059 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H9 | object_tokens | 1 | 0.575 | 0.016 | 0.000 | 0.062 | 0.062 | 1.00 | -0.031 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH1 | records_all | 4 | 0.195 | 0.249 | 0.012 | 0.062 | 0.031 | 0.50 | 0.047 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:topH1 | target_value_tokens | 4 | 0.115 | 0.241 | 0.009 | 0.062 | 0.000 | 0.00 | 0.038 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | instruction | 1 | 0.252 | 0.001 | 0.006 | 0.062 | 0.000 | 0.00 | 0.031 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H29 | records_all | 1 | 0.233 | -0.002 | 0.001 | 0.062 | 0.000 | 0.00 | 0.031 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H29 | records_other | 1 | 0.148 | -0.001 | 0.002 | 0.062 | 0.000 | 0.00 | 0.016 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | target_record_line | 1 | 0.050 | 0.000 | 0.000 | 0.062 | 0.000 | 0.00 | 0.016 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | target_record_line | 1 | 0.817 | 0.281 | 0.320 | 1.250 | 0.000 | 0.00 | 0.922 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | records_all | 1 | 0.926 | 0.323 | 0.343 | 1.000 | 0.062 | 1.00 | 0.922 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | target_value_tokens | 1 | 0.598 | 0.294 | 0.362 | 1.000 | 0.000 | 0.00 | 0.641 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | records_all | 4 | 0.409 | 2.271 | 0.726 | 0.984 | 0.234 | 1.00 | 0.793 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | records_all | 1 | 0.909 | 1.336 | 0.187 | 0.875 | 0.000 | 0.00 | 0.516 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | target_record_line | 1 | 0.844 | 1.393 | 0.165 | 0.875 | 0.000 | 0.00 | 0.500 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | target_record_line | 4 | 0.244 | 1.926 | 0.388 | 0.812 | 0.203 | 1.00 | 0.613 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH4 | records_all | 4 | 0.441 | 1.517 | 0.432 | 0.750 | 0.469 | 2.50 | 0.707 | 0.500 | `route_suppressor_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | target_value_tokens | 1 | 0.691 | 1.434 | 0.162 | 0.625 | 0.125 | 1.00 | 0.406 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | target_value_tokens | 4 | 0.173 | 1.913 | 0.361 | 0.594 | 0.250 | 1.75 | 0.539 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | records_all | 4 | 0.594 | 1.308 | 0.518 | 0.594 | 0.062 | 0.50 | 0.422 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH4 | target_value_tokens | 4 | 0.168 | 1.576 | 0.098 | 0.594 | 0.047 | 0.75 | 0.464 | 0.500 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH1 | target_record_line | 4 | 0.385 | 1.000 | 0.285 | 0.531 | 0.234 | 1.50 | 0.469 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | target_record_line | 4 | 0.267 | 1.367 | 0.244 | 0.531 | 0.094 | 1.00 | 0.387 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | target_record_line | 4 | 0.342 | 1.346 | 0.283 | 0.531 | 0.000 | 0.00 | 0.438 | 0.250 | `target_support_content` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:topH4 | target_record_line | 4 | 0.199 | 3.857 | 0.389 | 0.516 | 0.219 | 1.00 | 0.467 | 0.250 | `target_support_content` |
| deepseek7b | natural_donor | L23:attn_out | L23:attn_out:topH4 | target_record_line | 4 | 0.166 | 3.012 | 0.642 | 0.516 | 0.062 | 0.50 | 0.340 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:H11 | target_value_tokens | 4 | 0.406 | 3.231 | 0.684 | 0.500 | 0.234 | 1.25 | 0.452 | 0.250 | `target_support_content` |

## Strict Interpretation

- QK/pattern evidence is attention mass only.
- V/O/content evidence is causal source contribution removal before o_proj.
- A source group with high attention but weak removal effect is not a causal content source.
- This is head/source-path evidence, not neuron-level evidence.
