# Phase 755 Cross-Domain Route Invariance Atlas (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Scope: fruit / animal / plant / object / tool / abstract.
- Evidence: natural route class profile + fixed head/source contribution removal.

## Route Profile

| model | route observations | mean pairwise domain JS | strongest shared top classes |
|---|---:|---:|---|
| qwen3 | 2 | 0.3031 | `{'donor_answer': 2}` |
| glm4 | 2 | 0.5313 | `{'donor_answer': 2}` |
| deepseek7b | 2 | 0.1857 | `{'punctuation_or_stop': 1, 'echo_object_or_relation': 1}` |

## Top Cross-Domain Writer / Guard Candidates

| model | site | head | source | n | domains | support rate | mean drop | guard rate | mean release | guess |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out | 15 | target_record_line | 2 | 2 | 0.500 | 0.250 | 0.000 | 0.000 | `domain_specific_or_weak` |
| qwen3 | L33:attn_out | 15 | target_value_tokens | 2 | 2 | 0.500 | 0.188 | 0.000 | 0.000 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_value_tokens | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.094 | `domain_specific_or_weak` |
| glm4 | L35:attn_out | 29 | target_record_line | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.062 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 24 | target_record_line | 2 | 2 | 0.500 | 0.188 | 0.000 | 0.000 | `domain_specific_or_weak` |
| deepseek7b | L22:attn_out | 24 | target_value_tokens | 2 | 2 | 0.500 | 0.125 | 0.500 | 0.219 | `domain_specific_or_weak` |

## Strict Interpretation

- Low JS across domains supports route-profile similarity, not a discovered invariant by itself.
- Fixed source removal effects support path-level necessity, not a full neuron graph.
- If a candidate is strong only in DS7B, it is a model-local atlas result until replicated.
