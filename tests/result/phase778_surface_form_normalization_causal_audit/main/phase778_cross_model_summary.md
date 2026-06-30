# Phase 778 Surface-Form Normalization Causal Audit (main)

- Status: `complete`
- Test: prompt-level surface-form interventions on Phase 776 case-variant strict failures.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.

## Prompt Observation Summary

| model | variant | rows | cases | strict open | semantic-equiv open | surface gain | pool top1 | case-variant top1 | hard readout after equiv | base rank | top1 gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 6 | 6 | 0.000 | 0.833 | 0.833 | 1.000 | 0.833 | 0.167 | 5.333 | 4.875 |
| qwen3 | `with_candidate_list` | 6 | 6 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `token_identity_contract` | 6 | 6 | 0.667 | 0.667 | 0.000 | 1.000 | 0.000 | 0.333 | 2.167 | 1.000 |
| qwen3 | `lowercase_short_value` | 6 | 6 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `lowercase_no_punctuation` | 6 | 6 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| qwen3 | `constrained_free_prompt` | 6 | 6 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.500 | 3.021 |
| glm4 | `without_candidate_list` | 6 | 6 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.500 | 1.229 |
| glm4 | `with_candidate_list` | 6 | 6 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| glm4 | `token_identity_contract` | 6 | 6 | 0.167 | 0.667 | 0.500 | 0.667 | 0.500 | 0.333 | 3.667 | 1.052 |
| glm4 | `lowercase_short_value` | 6 | 6 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 | 0.000 | 2.167 | 0.354 |
| glm4 | `lowercase_no_punctuation` | 6 | 6 | 0.667 | 0.667 | 0.000 | 1.000 | 0.000 | 0.333 | 1.667 | 0.354 |
| glm4 | `constrained_free_prompt` | 6 | 6 | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 | 0.000 | 4.833 | 1.677 |
| deepseek7b | `without_candidate_list` | 6 | 6 | 0.000 | 0.833 | 0.833 | 0.667 | 0.833 | 0.000 | 1544.500 | 6.484 |
| deepseek7b | `with_candidate_list` | 6 | 6 | 0.167 | 1.000 | 0.833 | 0.667 | 0.833 | 0.000 | 3.500 | 2.094 |
| deepseek7b | `token_identity_contract` | 6 | 6 | 0.167 | 0.667 | 0.500 | 0.667 | 0.667 | 0.000 | 132.667 | 2.875 |
| deepseek7b | `lowercase_short_value` | 6 | 6 | 0.500 | 0.667 | 0.167 | 0.667 | 0.167 | 0.000 | 682.833 | 1.958 |
| deepseek7b | `lowercase_no_punctuation` | 6 | 6 | 0.167 | 0.667 | 0.500 | 0.500 | 0.500 | 0.000 | 569.167 | 3.902 |
| deepseek7b | `constrained_free_prompt` | 6 | 6 | 0.000 | 0.833 | 0.833 | 0.667 | 0.833 | 0.000 | 870.500 | 6.211 |

## Top1 Competitor Classes

| model | variant | class | rows | cases | strict open | semantic-equiv open | mean gap above target |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | `case_variant_target_value` | 6 | 6 | 0.000 | 1.000 | 3.021 |
| qwen3 | `lowercase_short_value` | `target_value` | 6 | 6 | 1.000 | 1.000 | 0.000 |
| qwen3 | `lowercase_no_punctuation` | `target_value` | 6 | 6 | 1.000 | 1.000 | 0.000 |
| qwen3 | `with_candidate_list` | `target_value` | 6 | 6 | 1.000 | 1.000 | 0.000 |
| qwen3 | `without_candidate_list` | `case_variant_target_value` | 5 | 5 | 0.000 | 1.000 | 5.500 |
| qwen3 | `token_identity_contract` | `target_value` | 4 | 4 | 1.000 | 1.000 | 0.000 |
| qwen3 | `token_identity_contract` | `punctuation` | 1 | 1 | 0.000 | 0.000 | 5.250 |
| qwen3 | `without_candidate_list` | `whitespace_or_empty` | 1 | 1 | 0.000 | 0.000 | 1.750 |
| qwen3 | `token_identity_contract` | `lexical_word` | 1 | 1 | 0.000 | 0.000 | 0.750 |
| glm4 | `constrained_free_prompt` | `case_variant_target_value` | 6 | 6 | 0.000 | 1.000 | 1.677 |
| glm4 | `without_candidate_list` | `case_variant_target_value` | 6 | 6 | 0.000 | 1.000 | 1.229 |
| glm4 | `with_candidate_list` | `target_value` | 6 | 6 | 1.000 | 1.000 | 0.000 |
| glm4 | `lowercase_no_punctuation` | `target_value` | 4 | 4 | 1.000 | 1.000 | 0.000 |
| glm4 | `token_identity_contract` | `case_variant_target_value` | 3 | 3 | 0.000 | 1.000 | 1.292 |
| glm4 | `lowercase_short_value` | `case_variant_target_value` | 3 | 3 | 0.000 | 1.000 | 0.708 |
| glm4 | `lowercase_short_value` | `target_value` | 3 | 3 | 1.000 | 1.000 | 0.000 |
| glm4 | `token_identity_contract` | `lexical_word` | 2 | 2 | 0.000 | 0.000 | 1.219 |
| glm4 | `lowercase_no_punctuation` | `lexical_word` | 2 | 2 | 0.000 | 0.000 | 1.062 |
| glm4 | `token_identity_contract` | `target_value` | 1 | 1 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `without_candidate_list` | `case_variant_target_value` | 5 | 5 | 0.000 | 1.000 | 5.188 |
| deepseek7b | `constrained_free_prompt` | `case_variant_target_value` | 5 | 5 | 0.000 | 1.000 | 5.162 |
| deepseek7b | `with_candidate_list` | `case_variant_target_value` | 5 | 5 | 0.000 | 1.000 | 2.513 |
| deepseek7b | `token_identity_contract` | `case_variant_target_value` | 4 | 4 | 0.250 | 1.000 | 1.094 |
| deepseek7b | `lowercase_no_punctuation` | `case_variant_target_value` | 3 | 3 | 0.000 | 1.000 | 2.812 |
| deepseek7b | `lowercase_short_value` | `target_value` | 3 | 3 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `without_candidate_list` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 12.969 |
| deepseek7b | `constrained_free_prompt` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 11.456 |
| deepseek7b | `lowercase_no_punctuation` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 10.852 |
| deepseek7b | `lowercase_short_value` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 9.438 |
| deepseek7b | `token_identity_contract` | `format_or_explanation_word` | 1 | 1 | 0.000 | 0.000 | 9.125 |
| deepseek7b | `lowercase_no_punctuation` | `case_variant_contrast_value` | 1 | 1 | 0.000 | 0.000 | 4.125 |
| deepseek7b | `token_identity_contract` | `case_variant_contrast_value` | 1 | 1 | 0.000 | 0.000 | 3.750 |
| deepseek7b | `lowercase_short_value` | `contrast_value` | 1 | 1 | 0.000 | 0.000 | 1.625 |
| deepseek7b | `lowercase_short_value` | `case_variant_target_value` | 1 | 1 | 0.000 | 1.000 | 0.688 |
| deepseek7b | `lowercase_no_punctuation` | `target_value` | 1 | 1 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `with_candidate_list` | `target_value` | 1 | 1 | 1.000 | 1.000 | 0.000 |

## Strict Interpretation

- This is a prompt-level causal audit, not a component-level localization.
- If lowercase/token-identity instructions repair strict open closure, the missing layer is at least partly surface-form normalization.
- If candidate_list still dominates, candidate_list supplies more than a generic lowercase instruction.
