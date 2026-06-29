# Phase 752 Natural Writer Stability and Path Chain Validation (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: fixed head/source contribution removal across expanded object-relation-answer cases.

| model | context | site | head | source | n | relations | support rate | mean drop | route guard rate | mean release | top1 loss | final delta | stability |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | natural_recipient | L33:attn_out | 15 | target_record_line | 1 | 1 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 8.338 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 15 | records_all | 1 | 1 | 1.000 | 0.375 | 0.000 | 0.125 | 0.000 | 8.489 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 15 | target_value_tokens | 1 | 1 | 1.000 | 0.375 | 0.000 | 0.000 | 0.000 | 8.044 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | records_all | 1 | 1 | 0.000 | 0.125 | 0.000 | 0.125 | 0.000 | 4.515 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 3.377 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 23 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 2.506 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 23 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 2.284 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 23 | records_all | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 3.687 | `weak_or_unstable` |
| qwen3 | natural_donor | L33:attn_out | 15 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.915 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 23 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.353 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 23 | target_record_line | 1 | 1 | 0.000 | -0.125 | 0.000 | 0.000 | 0.000 | 1.310 | `weak_or_unstable` |
| qwen3 | natural_recipient | L33:attn_out | 23 | records_all | 1 | 1 | 0.000 | -0.250 | 0.000 | 0.125 | 0.000 | 2.352 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | records_all | 1 | 1 | 0.000 | 0.125 | 0.000 | 0.000 | 0.000 | 1.490 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_value_tokens | 1 | 1 | 0.000 | 0.062 | 0.000 | 0.125 | 0.000 | 1.327 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | records_all | 1 | 1 | 0.000 | 0.062 | 0.000 | 0.062 | 0.000 | 5.747 | `weak_or_unstable` |
| glm4 | natural_recipient | L34:attn_out | 4 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 1.554 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | 1.034 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | 1.351 | `weak_or_unstable` |
| glm4 | natural_donor | L35:attn_out | 29 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.252 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | target_record_line | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.030 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.955 | `weak_or_unstable` |
| glm4 | natural_donor | L34:attn_out | 4 | records_all | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.370 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | target_value_tokens | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.973 | `weak_or_unstable` |
| glm4 | natural_recipient | L35:attn_out | 29 | records_all | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.614 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_record_line | 1 | 1 | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 | 8.519 | `weak_or_unstable` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | records_all | 1 | 1 | 1.000 | 0.625 | 1.000 | 0.375 | 0.000 | 11.167 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | target_value_tokens | 1 | 1 | 1.000 | 0.500 | 1.000 | 0.625 | 0.000 | 9.155 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_recipient | L22:attn_out | 1 | records_all | 1 | 1 | 1.000 | 0.500 | 1.000 | 0.500 | 0.000 | 10.294 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_record_line | 1 | 1 | 1.000 | 0.375 | 1.000 | 0.938 | 0.000 | 9.854 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_value_tokens | 1 | 1 | 1.000 | 0.375 | 0.000 | 0.000 | 0.000 | 5.356 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 1 | records_all | 1 | 1 | 1.000 | 0.250 | 0.000 | 0.125 | 0.000 | 5.882 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 24 | target_record_line | 1 | 1 | 1.000 | 0.250 | 0.000 | 0.000 | 0.000 | 6.350 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 24 | records_all | 1 | 1 | 1.000 | 0.250 | 0.000 | 0.000 | 0.000 | 5.799 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_record_line | 1 | 1 | 0.000 | 0.125 | 0.000 | 0.188 | 0.000 | 3.354 | `weak_or_unstable` |
| deepseek7b | natural_donor | L22:attn_out | 1 | target_value_tokens | 1 | 1 | 0.000 | -0.125 | 1.000 | 0.688 | 0.000 | 5.208 | `route_guard_without_stable_target_support` |
| deepseek7b | natural_recipient | L22:attn_out | 24 | target_value_tokens | 1 | 1 | 0.000 | -0.250 | 1.000 | 0.688 | 0.000 | 4.398 | `route_guard_without_stable_target_support` |

## Strict Interpretation

- Stable target drop supports fixed writer-path necessity.
- Route release supports guard/suppressor participation.
- Downstream hidden delta only says the perturbation propagates; it does not prove a complete chain closure.
- Source groups are still external token-span labels.
