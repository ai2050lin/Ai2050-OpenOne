# Phase 752 Natural Writer Stability and Path Chain Validation (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed head/source contribution removal across expanded object-relation-answer cases.

| model | context | site | head | source | n | relations | support rate | mean drop | route guard rate | mean release | top1 loss | final delta | stability |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_donor | L32:attn_out | 11 | target_record_line | 12 | 6 | 0.333 | 0.062 | 0.250 | 0.125 | 0.000 | 5.809 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L32:attn_out | 0 | target_value_tokens | 12 | 6 | 0.333 | 0.042 | 0.500 | 0.240 | 0.000 | 4.018 | `route_guard_without_stable_target_support` |
| qwen3 | natural_donor | L33:attn_out | 23 | target_record_line | 12 | 6 | 0.250 | 0.188 | 0.250 | 0.125 | 0.000 | 3.289 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L33:attn_out | 23 | records_all | 12 | 6 | 0.250 | 0.156 | 0.333 | 0.115 | 0.000 | 4.641 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 15 | records_all | 12 | 6 | 0.250 | 0.146 | 0.333 | 0.177 | 0.000 | 5.268 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 23 | records_all | 12 | 6 | 0.250 | 0.146 | 0.250 | 0.094 | 0.083 | 4.714 | `answer_value_specific_writer` |
| qwen3 | natural_recipient | L33:attn_out | 15 | target_record_line | 12 | 6 | 0.250 | 0.125 | 0.417 | 0.260 | 0.000 | 4.773 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 15 | target_value_tokens | 12 | 6 | 0.250 | 0.125 | 0.500 | 0.250 | 0.083 | 4.655 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 23 | target_value_tokens | 12 | 6 | 0.250 | 0.115 | 0.333 | 0.115 | 0.083 | 3.226 | `answer_value_specific_writer` |
| qwen3 | natural_donor | L32:attn_out | 11 | target_value_tokens | 12 | 6 | 0.250 | 0.042 | 0.250 | 0.104 | 0.000 | 5.654 | `relation_conditioned_writer` |
| qwen3 | natural_donor | L33:attn_out | 15 | target_value_tokens | 12 | 6 | 0.250 | 0.021 | 0.417 | 0.219 | 0.000 | 3.556 | `answer_value_specific_writer` |
| qwen3 | natural_donor | L32:attn_out | 0 | target_record_line | 12 | 6 | 0.250 | 0.010 | 0.500 | 0.198 | 0.000 | 4.240 | `weak_or_unstable` |
| qwen3 | natural_donor | L32:attn_out | 0 | records_all | 12 | 6 | 0.250 | 0.000 | 0.333 | 0.219 | 0.000 | 5.192 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | target_record_line | 12 | 6 | 0.250 | -0.010 | 0.417 | 0.292 | 0.000 | 3.847 | `answer_value_specific_writer` |
| qwen3 | natural_donor | L33:attn_out | 23 | target_value_tokens | 12 | 6 | 0.167 | 0.167 | 0.333 | 0.135 | 0.000 | 3.257 | `relation_conditioned_writer` |
| qwen3 | natural_recipient | L33:attn_out | 23 | target_record_line | 12 | 6 | 0.167 | 0.125 | 0.167 | 0.073 | 0.083 | 3.328 | `answer_value_specific_writer` |
| qwen3 | natural_recipient | L32:attn_out | 11 | records_all | 12 | 6 | 0.167 | 0.031 | 0.500 | 0.208 | 0.083 | 6.076 | `route_guard_without_stable_target_support` |
| qwen3 | natural_donor | L33:attn_out | 23 | relation_tokens | 12 | 6 | 0.167 | -0.010 | 0.333 | 0.167 | 0.000 | 1.880 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | records_all | 12 | 6 | 0.167 | 0.094 | 0.000 | 0.021 | 0.000 | 2.775 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | target_record_line | 12 | 6 | 0.167 | 0.073 | 0.000 | 0.005 | 0.000 | 2.117 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | target_value_tokens | 12 | 6 | 0.167 | 0.062 | 0.000 | 0.016 | 0.000 | 2.090 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | target_value_tokens | 12 | 6 | 0.167 | 0.047 | 0.083 | 0.052 | 0.000 | 2.404 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | records_all | 12 | 6 | 0.167 | 0.047 | 0.000 | 0.052 | 0.000 | 2.986 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | target_record_line | 12 | 6 | 0.167 | 0.047 | 0.083 | 0.049 | 0.000 | 2.543 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | records_all | 12 | 6 | 0.083 | 0.047 | 0.083 | 0.057 | 0.000 | 3.735 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | records_all | 12 | 6 | 0.083 | 0.047 | 0.083 | 0.042 | 0.000 | 3.448 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | records_all | 12 | 6 | 0.083 | 0.042 | 0.000 | 0.005 | 0.000 | 2.870 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_value_tokens | 12 | 6 | 0.083 | 0.031 | 0.000 | 0.047 | 0.000 | 1.901 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_record_line | 12 | 6 | 0.083 | 0.031 | 0.000 | 0.031 | 0.000 | 1.786 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_value_tokens | 12 | 6 | 0.083 | 0.031 | 0.000 | 0.000 | 0.000 | 1.646 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_record_line | 12 | 6 | 0.083 | 0.021 | 0.167 | 0.094 | 0.000 | 2.008 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | target_record_line | 12 | 6 | 0.000 | 0.042 | 0.000 | 0.031 | 0.000 | 1.979 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | target_value_tokens | 12 | 6 | 0.000 | 0.031 | 0.083 | 0.036 | 0.000 | 1.486 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 9 | target_value_tokens | 12 | 6 | 0.000 | 0.021 | 0.000 | 0.044 | 0.000 | 1.605 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 9 | relation_tokens | 12 | 6 | 0.000 | 0.021 | 0.000 | 0.016 | 0.000 | 1.442 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | object_tokens | 12 | 6 | 0.000 | 0.016 | 0.000 | 0.021 | 0.000 | 1.253 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | records_all | 12 | 6 | 0.917 | 0.688 | 0.500 | 0.516 | 0.167 | 14.240 | `stable_mixed_writer_guard` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_record_line | 12 | 6 | 0.833 | 0.594 | 0.417 | 0.484 | 0.167 | 12.786 | `stable_mixed_writer_guard` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | records_all | 12 | 6 | 0.750 | 0.474 | 0.667 | 0.620 | 0.167 | 11.413 | `stable_mixed_writer_guard` |
| deepseek7b | natural_donor | L22:attn_out | 24 | records_all | 12 | 6 | 0.750 | 0.401 | 0.167 | 0.104 | 0.167 | 9.938 | `stable_target_writer` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_record_line | 12 | 6 | 0.750 | 0.396 | 0.167 | 0.099 | 0.250 | 8.967 | `stable_target_writer` |
| deepseek7b | natural_donor | L22:attn_out | 1 | records_all | 12 | 6 | 0.750 | 0.375 | 0.250 | 0.214 | 0.167 | 8.322 | `stable_mixed_writer_guard` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_record_line | 12 | 6 | 0.750 | 0.339 | 0.333 | 0.135 | 0.167 | 7.331 | `answer_value_specific_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_value_tokens | 12 | 6 | 0.667 | 0.438 | 0.500 | 0.573 | 0.083 | 8.941 | `stable_mixed_writer_guard` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_record_line | 12 | 6 | 0.583 | 0.417 | 0.500 | 0.557 | 0.167 | 9.275 | `stable_mixed_writer_guard` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_value_tokens | 12 | 6 | 0.583 | 0.339 | 0.417 | 0.172 | 0.083 | 6.763 | `relation_conditioned_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_value_tokens | 12 | 6 | 0.583 | 0.297 | 0.583 | 0.458 | 0.167 | 8.486 | `relation_conditioned_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | records_all | 12 | 6 | 0.583 | 0.208 | 0.667 | 0.422 | 0.083 | 7.144 | `relation_conditioned_writer` |
| deepseek7b | natural_donor | L23:attn_out | 6 | records_all | 12 | 6 | 0.500 | 0.135 | 0.083 | 0.042 | 0.083 | 5.636 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_value_tokens | 12 | 6 | 0.417 | 0.214 | 0.417 | 0.318 | 0.083 | 7.173 | `relation_conditioned_writer` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | target_record_line | 12 | 6 | 0.417 | 0.203 | 0.583 | 0.339 | 0.083 | 6.620 | `relation_conditioned_writer` |
| deepseek7b | natural_recipient | L23:attn_out | 6 | object_tokens | 12 | 6 | 0.417 | 0.099 | 0.167 | 0.083 | 0.167 | 6.305 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 7 | target_value_tokens | 12 | 6 | 0.333 | 0.109 | 0.250 | 0.193 | 0.083 | 5.074 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 7 | records_all | 12 | 6 | 0.250 | 0.099 | 0.750 | 0.391 | 0.083 | 6.161 | `route_guard_without_stable_target_support` |

## Strict Interpretation

- Stable target drop supports fixed writer-path necessity.
- Route release supports guard/suppressor participation.
- Downstream hidden delta only says the perturbation propagates; it does not prove a complete chain closure.
- Source groups are still external token-span labels.
