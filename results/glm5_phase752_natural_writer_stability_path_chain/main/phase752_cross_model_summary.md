# Phase 752 Natural Writer Stability and Path Chain Validation (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed head/source contribution removal across expanded object-relation-answer cases.

| model | context | site | head | source | n | relations | support rate | mean drop | route guard rate | mean release | top1 loss | final delta | stability |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L33:attn_out | 23 | target_record_line | 8 | 5 | 0.500 | 0.281 | 0.000 | 0.016 | 0.000 | 4.014 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L33:attn_out | 23 | records_all | 8 | 5 | 0.375 | 0.219 | 0.000 | 0.016 | 0.000 | 5.260 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L32:attn_out | 11 | records_all | 8 | 5 | 0.375 | 0.203 | 0.375 | 0.125 | 0.000 | 5.471 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 15 | records_all | 8 | 5 | 0.375 | 0.109 | 0.375 | 0.234 | 0.000 | 4.410 | `route_guard_without_stable_target_support` |
| qwen3 | natural_donor | L32:attn_out | 0 | target_value_tokens | 8 | 5 | 0.375 | 0.078 | 0.250 | 0.172 | 0.000 | 4.106 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 23 | target_value_tokens | 8 | 5 | 0.250 | 0.219 | 0.000 | 0.016 | 0.000 | 3.905 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L32:attn_out | 11 | target_value_tokens | 8 | 5 | 0.250 | 0.141 | 0.500 | 0.172 | 0.000 | 5.076 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 15 | target_value_tokens | 8 | 5 | 0.250 | 0.125 | 0.375 | 0.156 | 0.000 | 3.686 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L32:attn_out | 0 | target_record_line | 8 | 5 | 0.250 | 0.109 | 0.375 | 0.125 | 0.000 | 4.184 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 15 | target_record_line | 8 | 5 | 0.250 | 0.094 | 0.625 | 0.281 | 0.000 | 3.921 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 23 | records_all | 8 | 5 | 0.250 | 0.078 | 0.375 | 0.125 | 0.000 | 4.273 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L32:attn_out | 0 | records_all | 8 | 5 | 0.250 | 0.062 | 0.375 | 0.203 | 0.000 | 5.264 | `route_guard_without_stable_target_support` |
| qwen3 | natural_donor | L32:attn_out | 11 | target_value_tokens | 8 | 5 | 0.250 | 0.047 | 0.125 | 0.047 | 0.000 | 2.719 | `weak_or_unstable` |
| qwen3 | natural_donor | L32:attn_out | 11 | target_record_line | 8 | 5 | 0.250 | 0.047 | 0.000 | 0.016 | 0.000 | 2.759 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | object_tokens | 8 | 5 | 0.250 | 0.016 | 0.125 | 0.109 | 0.000 | 2.660 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | target_value_tokens | 8 | 5 | 0.250 | 0.016 | 0.125 | 0.047 | 0.000 | 2.460 | `weak_or_unstable` |
| qwen3 | natural_recipient | L32:attn_out | 11 | target_record_line | 8 | 5 | 0.125 | 0.125 | 0.375 | 0.125 | 0.000 | 5.076 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 23 | target_record_line | 8 | 5 | 0.125 | 0.094 | 0.250 | 0.094 | 0.000 | 2.333 | `relation_conditioned_writer` |
| glm4 | natural_recipient | L34:attn_out | 4 | records_all | 8 | 5 | 0.250 | 0.117 | 0.125 | 0.109 | 0.000 | 6.147 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_value_tokens | 8 | 5 | 0.250 | 0.109 | 0.125 | 0.125 | 0.000 | 4.278 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_record_line | 8 | 5 | 0.250 | 0.094 | 0.125 | 0.141 | 0.000 | 4.362 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | records_all | 8 | 5 | 0.125 | 0.094 | 0.000 | 0.023 | 0.000 | 3.470 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_value_tokens | 8 | 5 | 0.125 | 0.078 | 0.000 | 0.008 | 0.000 | 1.521 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_record_line | 8 | 5 | 0.125 | 0.062 | 0.000 | 0.023 | 0.000 | 1.601 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | records_all | 8 | 5 | 0.125 | 0.055 | 0.000 | 0.008 | 0.000 | 2.720 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | target_record_line | 8 | 5 | 0.000 | 0.078 | 0.000 | 0.008 | 0.000 | 1.898 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | target_value_tokens | 8 | 5 | 0.000 | 0.039 | 0.000 | 0.039 | 0.000 | 1.623 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | records_all | 8 | 5 | 0.000 | 0.039 | 0.000 | 0.023 | 0.000 | 1.660 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | relation_tokens | 8 | 5 | 0.000 | 0.031 | 0.000 | 0.023 | 0.000 | 1.742 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | object_tokens | 8 | 5 | 0.000 | 0.031 | 0.000 | 0.008 | 0.000 | 1.285 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | object_tokens | 8 | 5 | 0.000 | 0.023 | 0.000 | 0.016 | 0.000 | 1.243 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | relation_tokens | 8 | 5 | 0.000 | 0.016 | 0.000 | 0.055 | 0.000 | 1.190 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | object_tokens | 8 | 5 | 0.000 | 0.016 | 0.000 | 0.031 | 0.000 | 2.534 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | relation_tokens | 8 | 5 | 0.000 | 0.016 | 0.000 | 0.023 | 0.000 | 1.028 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 9 | target_record_line | 8 | 5 | 0.000 | 0.016 | 0.000 | 0.016 | 0.000 | 1.902 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | target_record_line | 8 | 5 | 0.000 | 0.008 | 0.125 | 0.031 | 0.000 | 1.141 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | records_all | 8 | 5 | 0.875 | 0.695 | 0.125 | 0.109 | 0.000 | 10.239 | `stable_target_writer` |
| deepseek7b | natural_donor | L22:attn_out | 24 | records_all | 8 | 5 | 0.875 | 0.555 | 0.000 | 0.031 | 0.000 | 9.733 | `stable_target_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | records_all | 8 | 5 | 0.875 | 0.422 | 0.375 | 0.266 | 0.000 | 12.667 | `stable_mixed_writer_guard` |
| deepseek7b | natural_donor | L22:attn_out | 1 | records_all | 8 | 5 | 0.750 | 0.578 | 0.000 | 0.031 | 0.000 | 8.538 | `stable_target_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_record_line | 8 | 5 | 0.750 | 0.547 | 0.125 | 0.219 | 0.000 | 9.058 | `stable_mixed_writer_guard` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_record_line | 8 | 5 | 0.750 | 0.414 | 0.000 | 0.039 | 0.000 | 6.297 | `stable_target_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_value_tokens | 8 | 5 | 0.750 | 0.367 | 0.250 | 0.180 | 0.000 | 5.908 | `stable_mixed_writer_guard` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_record_line | 8 | 5 | 0.625 | 0.500 | 0.000 | 0.016 | 0.000 | 8.191 | `stable_target_writer` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_value_tokens | 8 | 5 | 0.625 | 0.344 | 0.250 | 0.125 | 0.000 | 6.788 | `relation_conditioned_writer` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_record_line | 8 | 5 | 0.500 | 0.414 | 0.125 | 0.078 | 0.000 | 5.553 | `stable_target_writer` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_value_tokens | 8 | 5 | 0.500 | 0.406 | 0.250 | 0.148 | 0.000 | 5.970 | `stable_mixed_writer_guard` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_value_tokens | 8 | 5 | 0.500 | 0.234 | 0.250 | 0.203 | 0.000 | 5.807 | `relation_conditioned_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | records_all | 8 | 5 | 0.500 | 0.211 | 0.375 | 0.273 | 0.000 | 6.644 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | target_record_line | 8 | 5 | 0.500 | 0.195 | 0.375 | 0.195 | 0.000 | 5.357 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | target_value_tokens | 8 | 5 | 0.500 | 0.164 | 0.250 | 0.125 | 0.000 | 4.782 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L23:attn_out | 6 | object_tokens | 8 | 5 | 0.500 | 0.164 | 0.250 | 0.102 | 0.000 | 5.236 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 7 | records_all | 8 | 5 | 0.375 | 0.148 | 0.375 | 0.273 | 0.000 | 5.518 | `relation_conditioned_writer` |
| deepseek7b | natural_donor | L22:attn_out | 7 | target_value_tokens | 8 | 5 | 0.375 | 0.148 | 0.250 | 0.172 | 0.000 | 4.224 | `weak_or_unstable` |

## Strict Interpretation

- Stable target drop supports fixed writer-path necessity.
- Route release supports guard/suppressor participation.
- Downstream hidden delta only says the perturbation propagates; it does not prove a complete chain closure.
- Source groups are still external token-span labels.
