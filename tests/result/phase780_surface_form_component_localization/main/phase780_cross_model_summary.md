# Phase 780 Surface-Form Component Candidate Localization (main)

- Status: `complete`
- Test: direct-logit attribution over attention/MLP outputs for surface-form repair prompts.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.
- Strict interpretation: candidate localization, not final causal proof.

## Prompt Outcome Summary

| model | variant | rows | cases | strict open | semantic-equiv open | pool top1 | margin target-case | target rank | top1 classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `without_candidate_list` | 6 | 6 | 0.000 | 0.833 | 1.000 | -4.792 | 5.333 | `{"case_variant_target_value": 5, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list` | 6 | 6 | 1.000 | 1.000 | 1.000 | 6.302 | 1.000 | `{"target_value": 6}` |
| qwen3 | `lowercase_short_value` | 6 | 6 | 1.000 | 1.000 | 1.000 | 6.667 | 1.000 | `{"target_value": 6}` |
| glm4 | `without_candidate_list` | 6 | 6 | 0.000 | 1.000 | 1.000 | -1.229 | 3.500 | `{"case_variant_target_value": 6}` |
| glm4 | `with_candidate_list` | 6 | 6 | 1.000 | 1.000 | 1.000 | 1.771 | 1.000 | `{"target_value": 6}` |
| glm4 | `lowercase_short_value` | 6 | 6 | 0.500 | 1.000 | 1.000 | -0.052 | 2.167 | `{"case_variant_target_value": 3, "target_value": 3}` |
| deepseek7b | `without_candidate_list` | 6 | 6 | 0.000 | 0.833 | 0.667 | -5.167 | 1544.500 | `{"case_variant_target_value": 5, "format_or_explanation_word": 1}` |
| deepseek7b | `with_candidate_list` | 6 | 6 | 0.167 | 1.000 | 0.667 | -1.948 | 3.500 | `{"case_variant_target_value": 5, "target_value": 1}` |
| deepseek7b | `lowercase_short_value` | 6 | 6 | 0.500 | 0.667 | 0.667 | 0.216 | 682.833 | `{"case_variant_contrast_value": 1, "case_variant_target_value": 1, "format_or_explanation_word": 1, "target_value": 3}` |

## Top Component Delta Candidates

| model | compare | kind | layer | rows | cases | repair rate | actual margin delta | direct target-case delta | positive rate | align rate | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list` | `attn` | 34 | 6 | 6 | 1.000 | 11.094 | 7.250 | 1.000 | 1.000 | 7.250 |
| qwen3 | `lowercase_short_value` | `attn` | 35 | 6 | 6 | 1.000 | 11.458 | 6.546 | 1.000 | 1.000 | 6.546 |
| qwen3 | `with_candidate_list` | `attn` | 31 | 6 | 6 | 1.000 | 11.094 | 6.199 | 1.000 | 1.000 | 6.199 |
| qwen3 | `lowercase_short_value` | `mlp` | 35 | 6 | 6 | 1.000 | 11.458 | 6.047 | 1.000 | 1.000 | 6.047 |
| qwen3 | `with_candidate_list` | `mlp` | 34 | 6 | 6 | 1.000 | 11.094 | 5.527 | 1.000 | 1.000 | 5.527 |
| qwen3 | `with_candidate_list` | `mlp` | 35 | 6 | 6 | 1.000 | 11.094 | 5.217 | 1.000 | 1.000 | 5.217 |
| qwen3 | `lowercase_short_value` | `mlp` | 34 | 6 | 6 | 1.000 | 11.458 | 5.161 | 1.000 | 1.000 | 5.161 |
| qwen3 | `lowercase_short_value` | `mlp` | 33 | 6 | 6 | 1.000 | 11.458 | 3.562 | 1.000 | 1.000 | 3.562 |
| qwen3 | `with_candidate_list` | `attn` | 35 | 6 | 6 | 1.000 | 11.094 | 3.413 | 1.000 | 1.000 | 3.413 |
| qwen3 | `with_candidate_list` | `attn` | 32 | 6 | 6 | 1.000 | 11.094 | 1.948 | 1.000 | 1.000 | 1.948 |
| qwen3 | `with_candidate_list` | `attn` | 33 | 6 | 6 | 1.000 | 11.094 | 1.596 | 0.833 | 0.833 | 1.109 |
| qwen3 | `lowercase_short_value` | `mlp` | 32 | 6 | 6 | 1.000 | 11.458 | 1.099 | 1.000 | 1.000 | 1.099 |
| qwen3 | `with_candidate_list` | `attn` | 30 | 6 | 6 | 1.000 | 11.094 | 0.840 | 1.000 | 1.000 | 0.840 |
| qwen3 | `lowercase_short_value` | `mlp` | 26 | 6 | 6 | 1.000 | 11.458 | 0.783 | 1.000 | 1.000 | 0.783 |
| qwen3 | `lowercase_short_value` | `attn` | 31 | 6 | 6 | 1.000 | 11.458 | 0.748 | 1.000 | 1.000 | 0.748 |
| qwen3 | `with_candidate_list` | `mlp` | 33 | 6 | 6 | 1.000 | 11.094 | 1.029 | 0.833 | 0.833 | 0.714 |
| glm4 | `with_candidate_list` | `mlp` | 38 | 6 | 6 | 1.000 | 3.000 | 1.520 | 1.000 | 1.000 | 1.520 |
| glm4 | `with_candidate_list` | `attn` | 33 | 6 | 6 | 1.000 | 3.000 | 1.341 | 1.000 | 1.000 | 1.341 |
| glm4 | `lowercase_short_value` | `mlp` | 38 | 6 | 6 | 0.500 | 1.177 | 0.616 | 0.833 | 0.833 | 0.428 |
| glm4 | `with_candidate_list` | `attn` | 29 | 6 | 6 | 1.000 | 3.000 | 0.408 | 1.000 | 1.000 | 0.408 |
| glm4 | `lowercase_short_value` | `mlp` | 39 | 6 | 6 | 0.500 | 1.177 | 0.570 | 0.833 | 0.833 | 0.396 |
| glm4 | `with_candidate_list` | `attn` | 35 | 6 | 6 | 1.000 | 3.000 | 0.150 | 1.000 | 1.000 | 0.150 |
| glm4 | `with_candidate_list` | `attn` | 32 | 6 | 6 | 1.000 | 3.000 | 0.124 | 1.000 | 1.000 | 0.124 |
| glm4 | `with_candidate_list` | `mlp` | 34 | 6 | 6 | 1.000 | 3.000 | 0.161 | 0.833 | 0.833 | 0.112 |
| glm4 | `with_candidate_list` | `mlp` | 32 | 6 | 6 | 1.000 | 3.000 | 0.140 | 0.833 | 0.833 | 0.097 |
| glm4 | `with_candidate_list` | `attn` | 39 | 6 | 6 | 1.000 | 3.000 | 0.129 | 0.833 | 0.833 | 0.090 |
| glm4 | `lowercase_short_value` | `mlp` | 34 | 6 | 6 | 0.500 | 1.177 | 0.104 | 0.833 | 0.833 | 0.072 |
| glm4 | `lowercase_short_value` | `mlp` | 27 | 6 | 6 | 0.500 | 1.177 | 0.094 | 0.667 | 1.000 | 0.063 |
| glm4 | `with_candidate_list` | `mlp` | 30 | 6 | 6 | 1.000 | 3.000 | 0.070 | 0.833 | 0.833 | 0.049 |
| glm4 | `lowercase_short_value` | `mlp` | 36 | 6 | 6 | 0.500 | 1.177 | 0.062 | 0.833 | 0.833 | 0.043 |
| glm4 | `with_candidate_list` | `attn` | 38 | 6 | 6 | 1.000 | 3.000 | 0.061 | 0.833 | 0.833 | 0.043 |
| glm4 | `with_candidate_list` | `mlp` | 31 | 6 | 6 | 1.000 | 3.000 | 0.061 | 0.833 | 0.833 | 0.043 |
| deepseek7b | `lowercase_short_value` | `mlp` | 27 | 6 | 6 | 0.500 | 5.383 | 17.339 | 1.000 | 1.000 | 17.339 |
| deepseek7b | `lowercase_short_value` | `mlp` | 26 | 6 | 6 | 0.500 | 5.383 | 8.681 | 1.000 | 1.000 | 8.681 |
| deepseek7b | `with_candidate_list` | `attn` | 26 | 6 | 6 | 0.167 | 3.219 | 7.761 | 1.000 | 1.000 | 7.761 |
| deepseek7b | `with_candidate_list` | `attn` | 27 | 6 | 6 | 0.167 | 3.219 | 3.609 | 1.000 | 1.000 | 3.609 |
| deepseek7b | `with_candidate_list` | `mlp` | 27 | 6 | 6 | 0.167 | 3.219 | 3.818 | 0.833 | 0.833 | 2.652 |
| deepseek7b | `with_candidate_list` | `attn` | 25 | 6 | 6 | 0.167 | 3.219 | 1.347 | 1.000 | 1.000 | 1.347 |
| deepseek7b | `lowercase_short_value` | `mlp` | 24 | 6 | 6 | 0.500 | 5.383 | 1.805 | 0.833 | 0.833 | 1.254 |
| deepseek7b | `with_candidate_list` | `attn` | 23 | 6 | 6 | 0.167 | 3.219 | 1.436 | 0.833 | 0.833 | 0.997 |
| deepseek7b | `with_candidate_list` | `attn` | 22 | 6 | 6 | 0.167 | 3.219 | 0.866 | 1.000 | 1.000 | 0.866 |
| deepseek7b | `lowercase_short_value` | `attn` | 19 | 6 | 6 | 0.500 | 5.383 | 1.103 | 0.833 | 0.833 | 0.766 |
| deepseek7b | `with_candidate_list` | `attn` | 20 | 6 | 6 | 0.167 | 3.219 | 0.653 | 1.000 | 1.000 | 0.653 |
| deepseek7b | `lowercase_short_value` | `mlp` | 22 | 6 | 6 | 0.500 | 5.383 | 0.918 | 0.833 | 0.833 | 0.638 |
| deepseek7b | `with_candidate_list` | `mlp` | 20 | 6 | 6 | 0.167 | 3.219 | 0.852 | 0.833 | 0.833 | 0.592 |
| deepseek7b | `lowercase_short_value` | `mlp` | 21 | 6 | 6 | 0.500 | 5.383 | 0.737 | 0.833 | 0.833 | 0.512 |
| deepseek7b | `lowercase_short_value` | `mlp` | 16 | 6 | 6 | 0.500 | 5.383 | 0.482 | 0.833 | 0.833 | 0.335 |
| deepseek7b | `with_candidate_list` | `attn` | 18 | 6 | 6 | 0.167 | 3.219 | 0.374 | 0.833 | 0.833 | 0.260 |

## By Component Kind

| model | compare | kind | rows | cases | repair rate | direct target-case delta | positive rate | align rate | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value` | `mlp` | 216 | 6 | 1.000 | 0.573 | 0.671 | 0.671 | 0.258 |
| qwen3 | `with_candidate_list` | `attn` | 216 | 6 | 1.000 | 0.613 | 0.625 | 0.625 | 0.239 |
| qwen3 | `lowercase_short_value` | `attn` | 216 | 6 | 1.000 | 0.270 | 0.611 | 0.611 | 0.101 |
| qwen3 | `with_candidate_list` | `mlp` | 216 | 6 | 1.000 | 0.304 | 0.556 | 0.556 | 0.094 |
| glm4 | `with_candidate_list` | `attn` | 240 | 6 | 1.000 | 0.058 | 0.521 | 0.521 | 0.016 |
| glm4 | `lowercase_short_value` | `mlp` | 240 | 6 | 0.500 | 0.042 | 0.554 | 0.537 | 0.012 |
| glm4 | `with_candidate_list` | `mlp` | 240 | 6 | 1.000 | 0.030 | 0.588 | 0.588 | 0.010 |
| glm4 | `lowercase_short_value` | `attn` | 240 | 6 | 0.500 | 0.003 | 0.496 | 0.604 | 0.001 |
| deepseek7b | `lowercase_short_value` | `mlp` | 168 | 6 | 0.500 | 1.112 | 0.637 | 0.637 | 0.451 |
| deepseek7b | `with_candidate_list` | `attn` | 168 | 6 | 0.167 | 0.537 | 0.613 | 0.613 | 0.202 |
| deepseek7b | `with_candidate_list` | `mlp` | 168 | 6 | 0.167 | 0.095 | 0.548 | 0.548 | 0.028 |
| deepseek7b | `lowercase_short_value` | `attn` | 168 | 6 | 0.500 | -0.038 | 0.446 | 0.446 | -0.008 |

## Strict Interpretation

- This phase uses direct-logit attribution on captured component outputs.
- Positive candidate layers are not yet proven necessary or sufficient.
- The next step must patch or ablate top candidate components while measuring C_pool, C_surface, and C_token.
