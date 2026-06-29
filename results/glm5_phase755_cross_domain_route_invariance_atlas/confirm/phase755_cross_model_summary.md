# Phase 755 Cross-Domain Route Invariance Atlas (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Scope: fruit / animal / plant / object / tool / abstract.
- Evidence: natural route class profile + fixed head/source contribution removal.

## Route Profile

| model | route observations | mean pairwise domain JS | strongest shared top classes |
|---|---:|---:|---|
| qwen3 | 58 | 0.0433 | `{'donor_answer': 45, 'format_or_schema': 13}` |
| glm4 | 58 | 0.0970 | `{'donor_answer': 58}` |
| deepseek7b | 58 | 0.0542 | `{'donor_answer': 34, 'format_or_schema': 18, 'echo_object_or_relation': 3, 'punctuation_or_stop': 2, 'other_vocab': 1}` |

## Top Cross-Domain Writer / Guard Candidates

| model | site | head | source | n | domains | support rate | mean drop | guard rate | mean release | guess |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out | 23 | records_all | 58 | 6 | 0.224 | 0.110 | 0.241 | 0.108 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 23 | target_record_line | 58 | 6 | 0.155 | 0.086 | 0.190 | 0.097 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 23 | target_value_tokens | 58 | 6 | 0.155 | 0.080 | 0.190 | 0.097 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | records_all | 58 | 6 | 0.121 | 0.045 | 0.310 | 0.127 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | target_value_tokens | 58 | 6 | 0.069 | 0.026 | 0.259 | 0.108 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | target_record_line | 58 | 6 | 0.069 | 0.004 | 0.328 | 0.159 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | records_all | 58 | 6 | 0.069 | 0.000 | 0.414 | 0.192 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | target_value_tokens | 58 | 6 | 0.052 | -0.022 | 0.431 | 0.192 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | target_record_line | 58 | 6 | 0.034 | 0.028 | 0.293 | 0.136 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | records_all | 58 | 6 | 0.034 | 0.023 | 0.069 | 0.055 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_value_tokens | 58 | 6 | 0.034 | 0.014 | 0.052 | 0.051 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_record_line | 58 | 6 | 0.034 | 0.014 | 0.052 | 0.050 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | target_value_tokens | 58 | 6 | 0.034 | -0.011 | 0.121 | 0.096 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | target_record_line | 58 | 6 | 0.034 | -0.013 | 0.103 | 0.089 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | records_all | 58 | 6 | 0.034 | -0.015 | 0.138 | 0.114 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | target_record_line | 58 | 6 | 0.000 | 0.012 | 0.017 | 0.042 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | target_value_tokens | 58 | 6 | 0.000 | 0.010 | 0.000 | 0.043 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | records_all | 58 | 6 | 0.000 | 0.002 | 0.034 | 0.061 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 24 | records_all | 58 | 6 | 0.862 | 0.528 | 0.310 | 0.200 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 1 | records_all | 58 | 6 | 0.810 | 0.554 | 0.328 | 0.189 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 24 | target_record_line | 58 | 6 | 0.810 | 0.448 | 0.328 | 0.223 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 1 | target_record_line | 58 | 6 | 0.672 | 0.435 | 0.379 | 0.204 | `cross_domain_mixed_writer_guard` |
| deepseek7b | L22:attn_out | 1 | target_value_tokens | 58 | 6 | 0.621 | 0.409 | 0.345 | 0.228 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 24 | target_value_tokens | 58 | 6 | 0.534 | 0.268 | 0.379 | 0.218 | `cross_domain_route_guard_candidate` |
| deepseek7b | L22:attn_out | 7 | records_all | 58 | 6 | 0.431 | 0.188 | 0.586 | 0.458 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 7 | target_value_tokens | 58 | 6 | 0.328 | 0.166 | 0.448 | 0.288 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 7 | target_record_line | 58 | 6 | 0.310 | 0.131 | 0.517 | 0.393 | `domain_specific_or_weak` |

## Strict Interpretation

- Low JS across domains supports route-profile similarity, not a discovered invariant by itself.
- Fixed source removal effects support path-level necessity, not a full neuron graph.
- If a candidate is strong only in DS7B, it is a model-local atlas result until replicated.
