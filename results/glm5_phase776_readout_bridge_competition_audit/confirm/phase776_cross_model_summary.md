# Phase 776 Readout-Bridge Competition Audit (confirm)

- Status: `complete`
- Test: classify open-vocabulary top-k competitors that beat the semantic target.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.

## Prompt Observation Summary

| model | variant | rows | cases | base top1 | pool top1 | latent hit | base rank | pool rank | top1 gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 7 | 7 | 0.000 | 0.857 | 0.857 | 902.857 | 1.857 | 5.732 |
| qwen3 | `with_candidate_list` | 7 | 7 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| qwen3 | `constrained_free_prompt` | 7 | 7 | 0.000 | 0.857 | 0.857 | 364.143 | 1.714 | 4.232 |
| glm4 | `without_candidate_list` | 8 | 8 | 0.000 | 1.000 | 1.000 | 193.500 | 1.000 | 2.693 |
| glm4 | `with_candidate_list` | 8 | 8 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| glm4 | `constrained_free_prompt` | 8 | 8 | 0.000 | 0.625 | 0.625 | 171.750 | 1.375 | 2.863 |
| deepseek7b | `without_candidate_list` | 8 | 8 | 0.125 | 0.625 | 0.500 | 1186.375 | 1.500 | 5.391 |
| deepseek7b | `with_candidate_list` | 8 | 8 | 0.375 | 0.750 | 0.375 | 2.875 | 1.250 | 1.570 |
| deepseek7b | `constrained_free_prompt` | 8 | 8 | 0.125 | 0.750 | 0.625 | 675.000 | 1.250 | 5.225 |

## Latent-Hit Top1 Competitor Classes

| model | variant | class | rows | cases | mean gap above target |
|---|---|---|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | `case_variant_target_value` | 6 | 6 | 3.021 |
| qwen3 | `without_candidate_list` | `case_variant_target_value` | 5 | 5 | 5.500 |
| qwen3 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 1.750 |
| glm4 | `without_candidate_list` | `case_variant_target_value` | 6 | 6 | 1.229 |
| glm4 | `constrained_free_prompt` | `case_variant_target_value` | 4 | 4 | 1.516 |
| glm4 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 7.516 |
| glm4 | `without_candidate_list` | `lexical_capitalized` | 1 | 1 | 6.656 |
| glm4 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 6.125 |
| deepseek7b | `constrained_free_prompt` | `case_variant_target_value` | 4 | 4 | 5.016 |
| deepseek7b | `without_candidate_list` | `case_variant_target_value` | 4 | 4 | 5.000 |
| deepseek7b | `with_candidate_list` | `case_variant_target_value` | 3 | 3 | 2.625 |
| deepseek7b | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 4.531 |

## All Top1 Competitor Classes

| model | variant | class | rows | cases | mean gap above target |
|---|---|---|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | `case_variant_target_value` | 6 | 6 | 3.021 |
| qwen3 | `without_candidate_list` | `case_variant_target_value` | 5 | 5 | 5.500 |
| qwen3 | `without_candidate_list` | `whitespace_or_empty` | 2 | 2 | 6.312 |
| qwen3 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 11.500 |
| glm4 | `constrained_free_prompt` | `case_variant_target_value` | 6 | 6 | 1.677 |
| glm4 | `without_candidate_list` | `case_variant_target_value` | 6 | 6 | 1.229 |
| glm4 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 7.516 |
| glm4 | `constrained_free_prompt` | `whitespace_or_empty` | 1 | 1 | 6.719 |
| glm4 | `without_candidate_list` | `lexical_capitalized` | 1 | 1 | 6.656 |
| glm4 | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 6.125 |
| deepseek7b | `without_candidate_list` | `case_variant_target_value` | 5 | 5 | 5.188 |
| deepseek7b | `constrained_free_prompt` | `case_variant_target_value` | 5 | 5 | 5.162 |
| deepseek7b | `with_candidate_list` | `case_variant_target_value` | 5 | 5 | 2.513 |
| deepseek7b | `without_candidate_list` | `format_or_explanation_word` | 1 | 1 | 12.969 |
| deepseek7b | `constrained_free_prompt` | `format_or_explanation_word` | 1 | 1 | 11.456 |
| deepseek7b | `constrained_free_prompt` | `lexical_capitalized` | 1 | 1 | 4.531 |
| deepseek7b | `without_candidate_list` | `lexical_capitalized` | 1 | 1 | 4.219 |

## Strict Interpretation

- This audit names the open-vocabulary competitor classes that beat the target.
- It does not prove the competitor class is causally suppressing the target.
- It separates readout competition from semantic value-pool selection.
