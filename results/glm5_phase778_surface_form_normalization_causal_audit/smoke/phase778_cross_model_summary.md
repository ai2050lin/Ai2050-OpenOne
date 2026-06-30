# Phase 778 Surface-Form Normalization Causal Audit (smoke)

- Status: `complete`
- Test: prompt-level surface-form interventions on Phase 776 case-variant strict failures.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.

## Prompt Observation Summary

| model | variant | rows | cases | strict open | semantic-equiv open | surface gain | pool top1 | case-variant top1 | hard readout after equiv | base rank | top1 gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 4.000 | 4.812 |
| qwen3 | `with_candidate_list` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `token_identity_contract` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `lowercase_short_value` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `lowercase_no_punctuation` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `constrained_free_prompt` | 2 | 2 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.500 | 2.938 |
| glm4 | `without_candidate_list` | 2 | 2 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 1.469 |
| glm4 | `with_candidate_list` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| glm4 | `token_identity_contract` | 2 | 2 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 4.000 | 1.031 |
| glm4 | `lowercase_short_value` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| glm4 | `lowercase_no_punctuation` | 2 | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| glm4 | `constrained_free_prompt` | 2 | 2 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 3.500 | 1.844 |
| deepseek7b | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 4593.500 | 9.234 |
| deepseek7b | `with_candidate_list` | 2 | 2 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 3.000 | 1.938 |
| deepseek7b | `token_identity_contract` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 385.000 | 4.969 |
| deepseek7b | `lowercase_short_value` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 2044.000 | 5.062 |
| deepseek7b | `lowercase_no_punctuation` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 1677.000 | 6.770 |
| deepseek7b | `constrained_free_prompt` | 2 | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 2585.000 | 8.728 |

## Top1 Competitor Classes

| model | variant | class | rows | cases | strict open | semantic-equiv open | mean gap above target |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | `case_variant_target_value` | 2 | 2 | 0.000 | 1.000 | 2.938 |
| qwen3 | `lowercase_short_value` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| qwen3 | `lowercase_no_punctuation` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| qwen3 | `token_identity_contract` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| qwen3 | `with_candidate_list` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| qwen3 | `without_candidate_list` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 7.875 |
| qwen3 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 0.000 | 0.000 | 1.750 |
| glm4 | `constrained_free_prompt` | `case_variant_target_value` | 2 | 2 | 0.000 | 1.000 | 1.844 |
| glm4 | `without_candidate_list` | `case_variant_target_value` | 2 | 2 | 0.000 | 1.000 | 1.469 |
| glm4 | `token_identity_contract` | `case_variant_target_value` | 2 | 2 | 0.000 | 1.000 | 1.031 |
| glm4 | `lowercase_short_value` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| glm4 | `lowercase_no_punctuation` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| glm4 | `with_candidate_list` | `target_value` | 2 | 2 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `with_candidate_list` | `case_variant_target_value` | 2 | 2 | 0.000 | 1.000 | 1.938 |
| deepseek7b | `without_candidate_list` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 12.969 |
| deepseek7b | `constrained_free_prompt` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 11.456 |
| deepseek7b | `lowercase_no_punctuation` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 10.852 |
| deepseek7b | `lowercase_short_value` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 9.438 |
| deepseek7b | `token_identity_contract` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 9.125 |
| deepseek7b | `constrained_free_prompt` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 6.000 |
| deepseek7b | `without_candidate_list` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 5.500 |
| deepseek7b | `lowercase_no_punctuation` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 2.688 |
| deepseek7b | `token_identity_contract` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 0.812 |
| deepseek7b | `lowercase_short_value` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 0.688 |

## Strict Interpretation

- This is a prompt-level causal audit, not a component-level localization.
- If lowercase/token-identity instructions repair strict open closure, the missing layer is at least partly surface-form normalization.
- If candidate_list still dominates, candidate_list supplies more than a generic lowercase instruction.
