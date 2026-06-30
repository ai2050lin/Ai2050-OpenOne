# Phase 780 Surface-Form Component Candidate Localization (smoke)

- Status: `complete`
- Test: direct-logit attribution over attention/MLP outputs for surface-form repair prompts.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.
- Strict interpretation: candidate localization, not final causal proof.

## Prompt Outcome Summary

| model | variant | rows | cases | strict open | semantic-equiv open | pool top1 | margin target-case | target rank | top1 classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 1.000 | -4.562 | 4.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list` | 2 | 2 | 1.000 | 1.000 | 1.000 | 5.125 | 1.000 | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value` | 2 | 2 | 1.000 | 1.000 | 1.000 | 6.750 | 1.000 | `{"target_value": 2}` |
| glm4 | `without_candidate_list` | 2 | 2 | 0.000 | 1.000 | 1.000 | -1.469 | 2.000 | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list` | 2 | 2 | 1.000 | 1.000 | 1.000 | 1.656 | 1.000 | `{"target_value": 2}` |
| glm4 | `lowercase_short_value` | 2 | 2 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | `{"target_value": 2}` |
| deepseek7b | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | -5.281 | 4593.500 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list` | 2 | 2 | 0.000 | 1.000 | 0.500 | -1.938 | 3.000 | `{"case_variant_target_value": 2}` |
| deepseek7b | `lowercase_short_value` | 2 | 2 | 0.000 | 0.500 | 0.500 | -0.727 | 2044.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |

## Top Component Delta Candidates

| model | compare | kind | layer | rows | cases | repair rate | actual margin delta | direct target-case delta | positive rate | align rate | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list` | `attn` | 34 | 2 | 2 | 1.000 | 9.688 | 7.039 | 1.000 | 1.000 | 7.039 |
| qwen3 | `lowercase_short_value` | `mlp` | 35 | 2 | 2 | 1.000 | 11.312 | 6.792 | 1.000 | 1.000 | 6.792 |
| qwen3 | `with_candidate_list` | `mlp` | 34 | 2 | 2 | 1.000 | 9.688 | 6.489 | 1.000 | 1.000 | 6.489 |
| qwen3 | `with_candidate_list` | `mlp` | 35 | 2 | 2 | 1.000 | 9.688 | 6.164 | 1.000 | 1.000 | 6.164 |
| qwen3 | `lowercase_short_value` | `attn` | 35 | 2 | 2 | 1.000 | 11.312 | 6.089 | 1.000 | 1.000 | 6.089 |
| qwen3 | `lowercase_short_value` | `mlp` | 34 | 2 | 2 | 1.000 | 11.312 | 4.703 | 1.000 | 1.000 | 4.703 |
| qwen3 | `with_candidate_list` | `attn` | 31 | 2 | 2 | 1.000 | 9.688 | 4.292 | 1.000 | 1.000 | 4.292 |
| qwen3 | `lowercase_short_value` | `mlp` | 33 | 2 | 2 | 1.000 | 11.312 | 3.399 | 1.000 | 1.000 | 3.399 |
| qwen3 | `with_candidate_list` | `attn` | 35 | 2 | 2 | 1.000 | 9.688 | 3.055 | 1.000 | 1.000 | 3.055 |
| qwen3 | `with_candidate_list` | `mlp` | 33 | 2 | 2 | 1.000 | 9.688 | 1.652 | 1.000 | 1.000 | 1.652 |
| qwen3 | `with_candidate_list` | `attn` | 32 | 2 | 2 | 1.000 | 9.688 | 1.631 | 1.000 | 1.000 | 1.631 |
| qwen3 | `lowercase_short_value` | `mlp` | 30 | 2 | 2 | 1.000 | 11.312 | 1.550 | 1.000 | 1.000 | 1.550 |
| qwen3 | `lowercase_short_value` | `mlp` | 28 | 2 | 2 | 1.000 | 11.312 | 1.147 | 1.000 | 1.000 | 1.147 |
| qwen3 | `with_candidate_list` | `attn` | 33 | 2 | 2 | 1.000 | 9.688 | 1.069 | 1.000 | 1.000 | 1.069 |
| qwen3 | `lowercase_short_value` | `mlp` | 26 | 2 | 2 | 1.000 | 11.312 | 0.789 | 1.000 | 1.000 | 0.789 |
| qwen3 | `with_candidate_list` | `attn` | 30 | 2 | 2 | 1.000 | 9.688 | 0.787 | 1.000 | 1.000 | 0.787 |
| glm4 | `with_candidate_list` | `attn` | 33 | 2 | 2 | 1.000 | 3.125 | 1.741 | 1.000 | 1.000 | 1.741 |
| glm4 | `with_candidate_list` | `mlp` | 38 | 2 | 2 | 1.000 | 3.125 | 1.208 | 1.000 | 1.000 | 1.208 |
| glm4 | `lowercase_short_value` | `mlp` | 38 | 2 | 2 | 1.000 | 1.969 | 1.089 | 1.000 | 1.000 | 1.089 |
| glm4 | `lowercase_short_value` | `mlp` | 39 | 2 | 2 | 1.000 | 1.969 | 0.774 | 1.000 | 1.000 | 0.774 |
| glm4 | `with_candidate_list` | `attn` | 29 | 2 | 2 | 1.000 | 3.125 | 0.583 | 1.000 | 1.000 | 0.583 |
| glm4 | `with_candidate_list` | `attn` | 35 | 2 | 2 | 1.000 | 3.125 | 0.266 | 1.000 | 1.000 | 0.266 |
| glm4 | `with_candidate_list` | `mlp` | 34 | 2 | 2 | 1.000 | 3.125 | 0.248 | 1.000 | 1.000 | 0.248 |
| glm4 | `with_candidate_list` | `mlp` | 37 | 2 | 2 | 1.000 | 3.125 | 0.242 | 1.000 | 1.000 | 0.242 |
| glm4 | `with_candidate_list` | `attn` | 34 | 2 | 2 | 1.000 | 3.125 | 0.223 | 1.000 | 1.000 | 0.223 |
| glm4 | `with_candidate_list` | `attn` | 37 | 2 | 2 | 1.000 | 3.125 | 0.216 | 1.000 | 1.000 | 0.216 |
| glm4 | `lowercase_short_value` | `mlp` | 34 | 2 | 2 | 1.000 | 1.969 | 0.165 | 1.000 | 1.000 | 0.165 |
| glm4 | `with_candidate_list` | `mlp` | 36 | 2 | 2 | 1.000 | 3.125 | 0.161 | 1.000 | 1.000 | 0.161 |
| glm4 | `lowercase_short_value` | `mlp` | 27 | 2 | 2 | 1.000 | 1.969 | 0.128 | 1.000 | 1.000 | 0.128 |
| glm4 | `lowercase_short_value` | `attn` | 37 | 2 | 2 | 1.000 | 1.969 | 0.114 | 1.000 | 1.000 | 0.114 |
| glm4 | `lowercase_short_value` | `mlp` | 26 | 2 | 2 | 1.000 | 1.969 | 0.107 | 1.000 | 1.000 | 0.107 |
| glm4 | `with_candidate_list` | `attn` | 28 | 2 | 2 | 1.000 | 3.125 | 0.106 | 1.000 | 1.000 | 0.106 |
| deepseek7b | `lowercase_short_value` | `mlp` | 27 | 2 | 2 | 0.000 | 4.555 | 17.532 | 1.000 | 1.000 | 17.532 |
| deepseek7b | `lowercase_short_value` | `mlp` | 26 | 2 | 2 | 0.000 | 4.555 | 8.326 | 1.000 | 1.000 | 8.326 |
| deepseek7b | `with_candidate_list` | `attn` | 26 | 2 | 2 | 0.000 | 3.344 | 7.775 | 1.000 | 1.000 | 7.775 |
| deepseek7b | `with_candidate_list` | `attn` | 25 | 2 | 2 | 0.000 | 3.344 | 1.373 | 1.000 | 1.000 | 1.373 |
| deepseek7b | `with_candidate_list` | `attn` | 27 | 2 | 2 | 0.000 | 3.344 | 1.249 | 1.000 | 1.000 | 1.249 |
| deepseek7b | `with_candidate_list` | `mlp` | 20 | 2 | 2 | 0.000 | 3.344 | 0.941 | 1.000 | 1.000 | 0.941 |
| deepseek7b | `lowercase_short_value` | `mlp` | 21 | 2 | 2 | 0.000 | 4.555 | 0.895 | 1.000 | 1.000 | 0.895 |
| deepseek7b | `with_candidate_list` | `attn` | 20 | 2 | 2 | 0.000 | 3.344 | 0.893 | 1.000 | 1.000 | 0.893 |
| deepseek7b | `lowercase_short_value` | `mlp` | 18 | 2 | 2 | 0.000 | 4.555 | 0.579 | 1.000 | 1.000 | 0.579 |
| deepseek7b | `with_candidate_list` | `attn` | 22 | 2 | 2 | 0.000 | 3.344 | 0.558 | 1.000 | 1.000 | 0.558 |
| deepseek7b | `with_candidate_list` | `mlp` | 25 | 2 | 2 | 0.000 | 3.344 | 1.769 | 0.500 | 0.500 | 0.442 |
| deepseek7b | `lowercase_short_value` | `attn` | 17 | 2 | 2 | 0.000 | 4.555 | 0.216 | 1.000 | 1.000 | 0.216 |
| deepseek7b | `with_candidate_list` | `mlp` | 21 | 2 | 2 | 0.000 | 3.344 | 0.765 | 0.500 | 0.500 | 0.191 |
| deepseek7b | `lowercase_short_value` | `mlp` | 10 | 2 | 2 | 0.000 | 4.555 | 0.188 | 1.000 | 1.000 | 0.188 |
| deepseek7b | `with_candidate_list` | `mlp` | 19 | 2 | 2 | 0.000 | 3.344 | 0.740 | 0.500 | 0.500 | 0.185 |
| deepseek7b | `with_candidate_list` | `mlp` | 23 | 2 | 2 | 0.000 | 3.344 | 0.713 | 0.500 | 0.500 | 0.178 |

## By Component Kind

| model | compare | kind | rows | cases | repair rate | direct target-case delta | positive rate | align rate | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list` | `attn` | 72 | 2 | 1.000 | 0.538 | 0.639 | 0.639 | 0.220 |
| qwen3 | `lowercase_short_value` | `mlp` | 72 | 2 | 1.000 | 0.570 | 0.611 | 0.611 | 0.213 |
| qwen3 | `lowercase_short_value` | `attn` | 72 | 2 | 1.000 | 0.283 | 0.667 | 0.667 | 0.126 |
| qwen3 | `with_candidate_list` | `mlp` | 72 | 2 | 1.000 | 0.293 | 0.458 | 0.458 | 0.062 |
| glm4 | `with_candidate_list` | `attn` | 80 | 2 | 1.000 | 0.089 | 0.637 | 0.637 | 0.036 |
| glm4 | `lowercase_short_value` | `mlp` | 80 | 2 | 1.000 | 0.061 | 0.500 | 0.500 | 0.015 |
| glm4 | `with_candidate_list` | `mlp` | 80 | 2 | 1.000 | 0.029 | 0.562 | 0.562 | 0.009 |
| glm4 | `lowercase_short_value` | `attn` | 80 | 2 | 1.000 | 0.012 | 0.675 | 0.675 | 0.005 |
| deepseek7b | `lowercase_short_value` | `mlp` | 56 | 2 | 0.000 | 0.982 | 0.589 | 0.589 | 0.341 |
| deepseek7b | `with_candidate_list` | `attn` | 56 | 2 | 0.000 | 0.379 | 0.554 | 0.554 | 0.116 |
| deepseek7b | `with_candidate_list` | `mlp` | 56 | 2 | 0.000 | 0.147 | 0.625 | 0.625 | 0.058 |
| deepseek7b | `lowercase_short_value` | `attn` | 56 | 2 | 0.000 | -0.162 | 0.411 | 0.411 | -0.027 |

## Strict Interpretation

- This phase uses direct-logit attribution on captured component outputs.
- Positive candidate layers are not yet proven necessary or sufficient.
- The next step must patch or ablate top candidate components while measuring C_pool, C_surface, and C_token.
