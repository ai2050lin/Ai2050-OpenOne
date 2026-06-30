# Phase 781 Surface-Form Candidate Causal Patch and Ablation (smoke)

- Status: `complete`
- Test: patch baseline component from donor repair prompt; replace/zero donor component for necessity.
- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.
- Strict interpretation: block-level causal evidence, not head/neuron-level proof.

## Candidates

| model | candidate | compare | kind | layer | phase780 score |
|---|---|---|---|---:|---:|
| qwen3 | `with_candidate_list:attn:L34` | `with_candidate_list` | `attn` | 34 | 7.250 |
| qwen3 | `lowercase_short_value:attn:L35` | `lowercase_short_value` | `attn` | 35 | 6.546 |
| glm4 | `with_candidate_list:mlp:L38` | `with_candidate_list` | `mlp` | 38 | 1.520 |
| glm4 | `with_candidate_list:attn:L33` | `with_candidate_list` | `attn` | 33 | 1.341 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `lowercase_short_value` | `mlp` | 27 | 17.339 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `lowercase_short_value` | `mlp` | 26 | 8.681 |

## Intervention Summary

| model | candidate | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `lowercase_short_value:attn:L35` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:attn:L35` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 6.750 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:attn:L35` | `patch_baseline_from_donor` | 2 | 0.000 | 0.500 | 1.000 | -2.250 | 2.312 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `lowercase_short_value:attn:L35` | `replace_donor_with_baseline` | 2 | 1.000 | 1.000 | 1.000 | 4.625 | -2.125 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `lowercase_short_value:attn:L35` | `zero_donor_component` | 2 | 1.000 | 1.000 | 1.000 | 4.812 | -1.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:attn:L34` | `normal_baseline` | 2 | 0.000 | 0.500 | 1.000 | -4.562 | null | null | null | null | null | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| qwen3 | `with_candidate_list:attn:L34` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 5.125 | null | null | null | null | null | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:attn:L34` | `patch_baseline_from_donor` | 2 | 0.000 | 1.000 | 1.000 | -2.625 | 1.938 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| qwen3 | `with_candidate_list:attn:L34` | `replace_donor_with_baseline` | 2 | 1.000 | 1.000 | 1.000 | 2.938 | -2.188 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| qwen3 | `with_candidate_list:attn:L34` | `zero_donor_component` | 2 | 0.500 | 1.000 | 1.000 | 2.781 | -2.344 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:attn:L33` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:attn:L33` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 1.656 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `with_candidate_list:attn:L33` | `patch_baseline_from_donor` | 2 | 0.500 | 1.000 | 1.000 | -0.500 | 0.969 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:attn:L33` | `replace_donor_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | 0.469 | -1.188 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:attn:L33` | `zero_donor_component` | 2 | 0.500 | 1.000 | 1.000 | 0.469 | -1.188 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:mlp:L38` | `normal_baseline` | 2 | 0.000 | 1.000 | 1.000 | -1.469 | null | null | null | null | null | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:mlp:L38` | `normal_donor` | 2 | 1.000 | 1.000 | 1.000 | 1.656 | null | null | null | null | null | `{"target_value": 2}` |
| glm4 | `with_candidate_list:mlp:L38` | `patch_baseline_from_donor` | 2 | 0.000 | 1.000 | 1.000 | -0.781 | 0.688 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 2}` |
| glm4 | `with_candidate_list:mlp:L38` | `replace_donor_with_baseline` | 2 | 0.500 | 1.000 | 1.000 | 0.875 | -0.781 | 0.000 | 0.500 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "target_value": 1}` |
| glm4 | `with_candidate_list:mlp:L38` | `zero_donor_component` | 2 | 1.000 | 1.000 | 1.000 | 2.688 | 1.031 | 0.000 | 0.000 | 0.000 | 0.000 | `{"target_value": 2}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `normal_donor` | 2 | 0.000 | 0.500 | 0.500 | -0.727 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `patch_baseline_from_donor` | 2 | 0.000 | 0.500 | 0.500 | -3.391 | 1.891 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `replace_donor_with_baseline` | 2 | 0.000 | 0.500 | 0.500 | -2.416 | -1.689 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L26` | `zero_donor_component` | 2 | 0.000 | 0.500 | 0.500 | -0.922 | -0.195 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `normal_baseline` | 2 | 0.000 | 0.500 | 0.500 | -5.281 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `normal_donor` | 2 | 0.000 | 0.500 | 0.500 | -0.727 | null | null | null | null | null | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `patch_baseline_from_donor` | 2 | 0.000 | 0.500 | 0.500 | -2.648 | 2.633 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `replace_donor_with_baseline` | 2 | 0.000 | 0.500 | 0.500 | -3.274 | -2.548 | 0.000 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "whitespace_or_empty": 1}` |
| deepseek7b | `lowercase_short_value:mlp:L27` | `zero_donor_component` | 2 | 0.500 | 0.500 | 0.500 | 0.160 | 0.887 | 0.500 | 0.000 | 0.000 | 0.000 | `{"case_variant_target_value": 1, "format_or_explanation_word": 1}` |

## Top Sufficiency Candidates

| model | candidate | intervention | cases | delta margin | strict gain | score |
|---|---|---|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:attn:L35` | `patch_baseline_from_donor` | 2 | 2.312 | 0.000 | 0.000 |
| qwen3 | `with_candidate_list:attn:L34` | `patch_baseline_from_donor` | 2 | 1.938 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:attn:L33` | `patch_baseline_from_donor` | 2 | 0.969 | 0.500 | 0.484 |
| glm4 | `with_candidate_list:mlp:L38` | `patch_baseline_from_donor` | 2 | 0.688 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `patch_baseline_from_donor` | 2 | 2.633 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `patch_baseline_from_donor` | 2 | 1.891 | 0.000 | 0.000 |

## Top Necessity Candidates

| model | candidate | intervention | cases | delta margin | strict loss | semantic loss | score |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:attn:L34` | `zero_donor_component` | 2 | -2.344 | 0.500 | 0.000 | 1.172 |
| qwen3 | `with_candidate_list:attn:L34` | `replace_donor_with_baseline` | 2 | -2.188 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:attn:L35` | `replace_donor_with_baseline` | 2 | -2.125 | 0.000 | 0.000 | 0.000 |
| qwen3 | `lowercase_short_value:attn:L35` | `zero_donor_component` | 2 | -1.938 | 0.000 | 0.000 | 0.000 |
| glm4 | `with_candidate_list:attn:L33` | `replace_donor_with_baseline` | 2 | -1.188 | 0.500 | 0.000 | 0.594 |
| glm4 | `with_candidate_list:attn:L33` | `zero_donor_component` | 2 | -1.188 | 0.500 | 0.000 | 0.594 |
| glm4 | `with_candidate_list:mlp:L38` | `replace_donor_with_baseline` | 2 | -0.781 | 0.500 | 0.000 | 0.391 |
| glm4 | `with_candidate_list:mlp:L38` | `zero_donor_component` | 2 | 1.031 | 0.000 | 0.000 | -0.000 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `replace_donor_with_baseline` | 2 | -2.548 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `replace_donor_with_baseline` | 2 | -1.689 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L26` | `zero_donor_component` | 2 | -0.195 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `lowercase_short_value:mlp:L27` | `zero_donor_component` | 2 | 0.887 | 0.000 | 0.000 | -0.000 |

## Strict Interpretation

- `patch_baseline_from_donor` tests sufficiency at block granularity.
- `replace_donor_with_baseline` and `zero_donor_component` test necessity at block granularity.
- Weak single-block patch is still compatible with a distributed multi-component route.
