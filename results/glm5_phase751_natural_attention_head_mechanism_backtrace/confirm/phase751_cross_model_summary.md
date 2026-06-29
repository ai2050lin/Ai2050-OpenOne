# Phase 751 Natural Attention Head Mechanism Backtrace (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source attention mass plus causal source V/O contribution removal.

| model | context | site | subunit | source | n | attn mass | source target contrib | source route supp contrib | remove target drop | route release | coverage | margin drop | top1 loss | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:H15 | target_record_line | 1 | 0.307 | 10.700 | 0.494 | 1.125 | 0.125 | 1.00 | 1.125 | 0.000 | `target_support_content` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:H15 | target_value_tokens | 1 | 0.301 | 10.706 | 0.497 | 1.125 | 0.125 | 1.00 | 1.125 | 0.000 | `target_support_content` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:H15 | records_all | 1 | 0.354 | 10.661 | 0.543 | 1.125 | 0.000 | 0.00 | 1.042 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | records_all | 1 | 0.379 | 13.785 | 0.076 | 0.875 | 0.000 | 0.00 | 0.438 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | target_record_line | 1 | 0.313 | 13.557 | 0.070 | 0.875 | 0.000 | 0.00 | 0.406 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:H23 | target_value_tokens | 1 | 0.266 | 12.925 | 0.106 | 0.750 | 0.000 | 0.00 | 0.312 | 0.000 | `target_support_content` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | records_all | 2 | 0.550 | 13.235 | 0.265 | 0.625 | 0.000 | 0.00 | 0.109 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:topH4 | target_record_line | 6 | 0.183 | 9.563 | 1.140 | 0.583 | 0.625 | 2.17 | 0.671 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | target_value_tokens | 2 | 0.519 | 13.205 | 0.269 | 0.562 | 0.000 | 0.00 | 0.078 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:topH4 | target_value_tokens | 6 | 0.149 | 9.298 | 1.121 | 0.458 | 0.750 | 1.83 | 0.588 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:topH4 | records_all | 6 | 0.252 | 9.687 | 1.156 | 0.438 | 0.771 | 2.17 | 0.591 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_recipient | L32:attn_out | L32:attn_out:H11 | target_record_line | 2 | 0.522 | 13.207 | 0.269 | 0.438 | 0.062 | 0.50 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_record_line | 1 | 0.260 | 6.881 | 0.722 | 0.375 | 0.250 | 1.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | target_value_tokens | 6 | 0.115 | 6.296 | 0.713 | 0.312 | 0.292 | 1.50 | 0.238 | 0.000 | `readout_target_aligned_observational` |
| qwen3 | natural_recipient | L33:attn_out | L33:attn_out:topH4 | records_all | 6 | 0.234 | 6.751 | 0.792 | 0.271 | 0.312 | 1.33 | 0.158 | 0.000 | `inverse_or_compensatory_content` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | records_all | 1 | 0.293 | 6.923 | 0.737 | 0.250 | 0.625 | 2.00 | 0.344 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L32:attn_out | L32:attn_out:H0 | target_value_tokens | 1 | 0.247 | 6.867 | 0.722 | 0.250 | 0.500 | 2.00 | 0.281 | 0.000 | `mixed_target_support_and_route_guard` |
| qwen3 | natural_donor | L33:attn_out | L33:attn_out:H30 | target_value_tokens | 1 | 0.307 | 11.239 | 0.740 | 0.250 | 0.000 | 0.00 | 0.000 | 0.000 | `readout_target_aligned_observational` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | records_all | 1 | 0.622 | 0.986 | 0.106 | 0.250 | 0.000 | 0.00 | 0.175 | 0.000 | `target_support_content` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | target_record_line | 1 | 0.514 | 0.980 | 0.106 | 0.250 | 0.000 | 0.00 | 0.175 | 0.000 | `target_support_content` |
| glm4 | natural_recipient | L35:attn_out | L35:attn_out:H29 | target_value_tokens | 1 | 0.459 | 0.962 | 0.109 | 0.250 | 0.000 | 0.00 | 0.175 | 0.000 | `target_support_content` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H7 | relation_tokens | 1 | 0.002 | 0.000 | 0.000 | 0.125 | 0.000 | 0.00 | 0.104 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | target_record_line | 2 | 0.154 | -0.007 | 0.013 | 0.094 | 0.031 | 0.50 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | instruction | 2 | 0.198 | 0.001 | 0.005 | 0.094 | 0.000 | 0.00 | 0.050 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | target_value_tokens | 6 | 0.147 | 0.233 | 0.077 | 0.073 | 0.062 | 0.50 | 0.047 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | records_all | 6 | 0.402 | 0.237 | 0.085 | 0.062 | 0.062 | 0.67 | 0.040 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | records_all | 2 | 0.532 | 0.009 | 0.028 | 0.062 | 0.031 | 0.50 | -0.113 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | object_tokens | 2 | 0.421 | 0.009 | 0.053 | 0.062 | 0.031 | 0.50 | -0.156 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | target_value_tokens | 2 | 0.016 | -0.006 | 0.011 | 0.062 | 0.000 | 0.00 | 0.031 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H29 | records_all | 1 | 0.233 | -0.002 | 0.001 | 0.062 | 0.000 | 0.00 | 0.025 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L35:attn_out | L35:attn_out:H29 | records_other | 1 | 0.148 | -0.001 | 0.002 | 0.062 | 0.000 | 0.00 | 0.013 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H8 | instruction | 1 | 0.092 | 0.000 | 0.001 | 0.062 | 0.000 | 0.00 | 0.000 | 0.000 | `small_or_unclear` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H9 | records_other | 2 | 0.378 | 0.016 | 0.030 | 0.062 | 0.000 | 0.00 | -0.075 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | records_all | 6 | 0.579 | 0.273 | 0.130 | 0.052 | 0.104 | 1.17 | 0.050 | 0.000 | `qk_pattern_visible_content_weak` |
| glm4 | natural_donor | L34:attn_out | L34:attn_out:H4 | target_record_line | 6 | 0.247 | 0.235 | 0.081 | 0.052 | 0.073 | 0.67 | 0.028 | 0.000 | `small_or_unclear` |
| glm4 | natural_recipient | L34:attn_out | L34:attn_out:H4 | target_value_tokens | 6 | 0.155 | 0.256 | 0.047 | 0.052 | 0.062 | 0.83 | 0.052 | 0.000 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | target_record_line | 1 | 0.817 | 0.281 | 0.320 | 1.250 | 0.000 | 0.00 | 0.922 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | records_all | 6 | 0.427 | 1.918 | 0.922 | 1.031 | 0.177 | 1.00 | 0.803 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | records_all | 1 | 0.926 | 0.323 | 0.343 | 1.000 | 0.062 | 1.00 | 0.922 | 0.000 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H24 | target_value_tokens | 1 | 0.598 | 0.294 | 0.362 | 1.000 | 0.000 | 0.00 | 0.641 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | records_all | 1 | 0.909 | 1.336 | 0.187 | 0.875 | 0.000 | 0.00 | 0.516 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | target_record_line | 1 | 0.844 | 1.393 | 0.165 | 0.875 | 0.000 | 0.00 | 0.500 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH4 | records_all | 6 | 0.425 | 1.319 | 0.399 | 0.833 | 0.365 | 2.00 | 0.692 | 0.333 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | target_record_line | 6 | 0.270 | 1.684 | 0.669 | 0.833 | 0.135 | 0.67 | 0.603 | 0.167 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | target_record_line | 6 | 0.384 | 1.602 | 0.368 | 0.708 | 0.188 | 0.50 | 0.693 | 0.167 | `target_support_content` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:H6 | relation_tokens | 1 | 0.600 | 3.754 | 0.267 | 0.688 | 0.125 | 1.00 | 0.487 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | target_value_tokens | 6 | 0.367 | 1.602 | 0.371 | 0.635 | 0.427 | 2.00 | 0.705 | 0.167 | `target_support_content` |
| deepseek7b | natural_recipient | L23:attn_out | L23:attn_out:H6 | question | 1 | 0.690 | 3.794 | 0.341 | 0.625 | 0.125 | 1.00 | 0.425 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H24 | target_value_tokens | 1 | 0.691 | 1.434 | 0.162 | 0.625 | 0.125 | 1.00 | 0.406 | 0.000 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:H1 | records_all | 6 | 0.701 | 1.435 | 0.617 | 0.615 | 0.562 | 2.00 | 0.672 | 0.000 | `mixed_target_support_and_route_guard` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | records_all | 6 | 0.623 | 1.612 | 0.453 | 0.604 | 0.208 | 0.83 | 0.508 | 0.167 | `target_support_content` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:topH4 | target_value_tokens | 6 | 0.159 | 1.473 | 0.564 | 0.583 | 0.208 | 1.33 | 0.501 | 0.167 | `small_or_unclear` |
| deepseek7b | natural_donor | L22:attn_out | L22:attn_out:H1 | target_record_line | 6 | 0.377 | 1.656 | 0.274 | 0.583 | 0.177 | 1.00 | 0.465 | 0.167 | `target_support_content` |
| deepseek7b | natural_recipient | L22:attn_out | L22:attn_out:topH4 | target_record_line | 6 | 0.225 | 1.276 | 0.116 | 0.552 | 0.302 | 1.67 | 0.466 | 0.333 | `target_support_content` |

## Strict Interpretation

- QK/pattern evidence is attention mass only.
- V/O/content evidence is causal source contribution removal before o_proj.
- A source group with high attention but weak removal effect is not a causal content source.
- This is head/source-path evidence, not neuron-level evidence.
