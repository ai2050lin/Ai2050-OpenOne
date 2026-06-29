# Phase 755 Cross-Domain Route Invariance Atlas (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Scope: fruit / animal / plant / object / tool / abstract.
- Evidence: natural route class profile + fixed head/source contribution removal.

## Route Profile

| model | route observations | mean pairwise domain JS | strongest shared top classes |
|---|---:|---:|---|
| qwen3 | 30 | 0.0409 | `{'donor_answer': 22, 'format_or_schema': 8}` |
| glm4 | 30 | 0.1300 | `{'donor_answer': 30}` |
| deepseek7b | 30 | 0.0738 | `{'donor_answer': 16, 'format_or_schema': 10, 'punctuation_or_stop': 2, 'echo_object_or_relation': 2}` |

## Top Cross-Domain Writer / Guard Candidates

| model | site | head | source | n | domains | support rate | mean drop | guard rate | mean release | guess |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out | 23 | records_all | 30 | 6 | 0.200 | 0.100 | 0.300 | 0.125 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 23 | target_value_tokens | 30 | 6 | 0.200 | 0.079 | 0.200 | 0.108 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 23 | target_record_line | 30 | 6 | 0.167 | 0.075 | 0.167 | 0.100 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | records_all | 30 | 6 | 0.167 | 0.050 | 0.400 | 0.154 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | target_record_line | 30 | 6 | 0.100 | 0.037 | 0.300 | 0.150 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | target_value_tokens | 30 | 6 | 0.067 | 0.025 | 0.300 | 0.125 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | records_all | 30 | 6 | 0.067 | 0.017 | 0.400 | 0.167 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | target_value_tokens | 30 | 6 | 0.067 | 0.004 | 0.467 | 0.183 | `domain_specific_or_weak` |
| qwen3 | L32:attn_out | 11 | target_record_line | 30 | 6 | 0.033 | 0.021 | 0.333 | 0.154 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | records_all | 30 | 6 | 0.067 | 0.050 | 0.033 | 0.033 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_record_line | 30 | 6 | 0.067 | 0.031 | 0.033 | 0.029 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_value_tokens | 30 | 6 | 0.067 | 0.027 | 0.033 | 0.031 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | records_all | 30 | 6 | 0.067 | 0.006 | 0.100 | 0.087 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | target_record_line | 30 | 6 | 0.067 | 0.004 | 0.100 | 0.078 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 4 | target_value_tokens | 30 | 6 | 0.067 | 0.002 | 0.100 | 0.073 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | target_value_tokens | 30 | 6 | 0.000 | 0.019 | 0.000 | 0.032 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | target_record_line | 30 | 6 | 0.000 | 0.019 | 0.000 | 0.026 | `domain_specific_or_weak` |
| glm4 | L34:attn_out | 9 | records_all | 30 | 6 | 0.000 | 0.015 | 0.000 | 0.044 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 1 | records_all | 30 | 6 | 0.867 | 0.537 | 0.333 | 0.190 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 24 | records_all | 30 | 6 | 0.867 | 0.506 | 0.233 | 0.108 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 24 | target_record_line | 30 | 6 | 0.767 | 0.412 | 0.200 | 0.156 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 1 | target_record_line | 30 | 6 | 0.700 | 0.419 | 0.300 | 0.163 | `cross_domain_writer_candidate` |
| deepseek7b | L22:attn_out | 1 | target_value_tokens | 30 | 6 | 0.600 | 0.373 | 0.367 | 0.223 | `cross_domain_mixed_writer_guard` |
| deepseek7b | L22:attn_out | 24 | target_value_tokens | 30 | 6 | 0.567 | 0.287 | 0.267 | 0.117 | `multi_domain_but_not_global` |
| deepseek7b | L22:attn_out | 7 | records_all | 30 | 6 | 0.333 | 0.092 | 0.667 | 0.371 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 7 | target_value_tokens | 30 | 6 | 0.267 | 0.131 | 0.400 | 0.208 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 7 | target_record_line | 30 | 6 | 0.167 | 0.069 | 0.567 | 0.375 | `domain_specific_or_weak` |

## Strict Interpretation

- Low JS across domains supports route-profile similarity, not a discovered invariant by itself.
- Fixed source removal effects support path-level necessity, not a full neuron graph.
- If a candidate is strong only in DS7B, it is a model-local atlas result until replicated.
