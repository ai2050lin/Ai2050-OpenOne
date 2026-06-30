# Phase 776 Readout-Bridge Competition Audit (smoke)

- Status: `complete`
- Test: classify open-vocabulary top-k competitors that beat the semantic target.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.

## Prompt Observation Summary

| model | variant | rows | cases | base top1 | pool top1 | latent hit | base rank | pool rank | top1 gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 3146.000 | 4.000 | 9.375 |
| qwen3 | `with_candidate_list` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | `constrained_free_prompt` | 2 | 2 | 0.000 | 0.500 | 0.500 | 1268.500 | 3.500 | 8.062 |
| glm4 | `without_candidate_list` | 1 | 1 | 0.000 | 1.000 | 1.000 | 201.000 | 1.000 | 6.656 |
| glm4 | `with_candidate_list` | 1 | 1 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| glm4 | `constrained_free_prompt` | 1 | 1 | 0.000 | 1.000 | 1.000 | 166.000 | 1.000 | 6.125 |
| deepseek7b | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 4591.000 | 1.500 | 8.703 |
| deepseek7b | `with_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 4.000 | 1.500 | 2.188 |
| deepseek7b | `constrained_free_prompt` | 2 | 2 | 0.000 | 0.500 | 0.500 | 2587.000 | 1.500 | 8.103 |

## Latent-Hit Top1 Competitor Classes

| model | variant | class | rows | cases | mean gap above target |
|---|---|---|---:|---:|---:|
| qwen3 | `without_candidate_list` | `boolean_value` | 1 | 1 | 7.875 |
| qwen3 | `constrained_free_prompt` | `boolean_value` | 1 | 1 | 4.625 |
| glm4 | `without_candidate_list` | `lexical_capitalized` | 1 | 1 | 6.656 |
| glm4 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 6.125 |
| deepseek7b | `constrained_free_prompt` | `boolean_value` | 1 | 1 | 4.750 |
| deepseek7b | `without_candidate_list` | `boolean_value` | 1 | 1 | 4.438 |
| deepseek7b | `with_candidate_list` | `boolean_value` | 1 | 1 | 2.750 |

## All Top1 Competitor Classes

| model | variant | class | rows | cases | mean gap above target |
|---|---|---|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 11.500 |
| qwen3 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 10.875 |
| qwen3 | `without_candidate_list` | `boolean_value` | 1 | 1 | 7.875 |
| qwen3 | `constrained_free_prompt` | `boolean_value` | 1 | 1 | 4.625 |
| glm4 | `without_candidate_list` | `lexical_capitalized` | 1 | 1 | 6.656 |
| glm4 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 6.125 |
| deepseek7b | `without_candidate_list` | `format_or_explanation_word` | 1 | 1 | 12.969 |
| deepseek7b | `constrained_free_prompt` | `format_or_explanation_word` | 1 | 1 | 11.456 |
| deepseek7b | `constrained_free_prompt` | `boolean_value` | 1 | 1 | 4.750 |
| deepseek7b | `without_candidate_list` | `boolean_value` | 1 | 1 | 4.438 |
| deepseek7b | `with_candidate_list` | `boolean_value` | 1 | 1 | 2.750 |
| deepseek7b | `with_candidate_list` | `lexical_capitalized` | 1 | 1 | 1.625 |

## Strict Interpretation

- This audit names the open-vocabulary competitor classes that beat the target.
- It does not prove the competitor class is causally suppressing the target.
- It separates readout competition from semantic value-pool selection.
