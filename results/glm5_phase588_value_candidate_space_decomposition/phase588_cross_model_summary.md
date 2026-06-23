# Phase 588 Value Candidate Space Decomposition Summary

Confirm setting: 32 value cases per model. Components are tested at prompt_last/query_relation and two late layers.

| model | target cases | diagnostic key | switch | correct gain | top-wrong gain | margin gain |
|---|---:|---|---:|---:|---:|---:|
| qwen3 | 4 | prompt_last|L27|full_delta | 2/4 (50.0%) | 0.588 | 0.400 | 0.188 |
| glm4 | 3 | query_relation|L38|full_delta | 0/3 (0.0%) | 0.021 | 0.021 | 0.000 |
| deepseek7b | 12 | prompt_last|L21|full_delta | 0/12 (0.0%) | 4.782 | 4.718 | 0.063 |

## DS7B Component Audit

| component | switch | correct gain | top-wrong gain | margin gain |
|---|---:|---:|---:|---:|
| prompt_last|L21|full_delta | 0/12 (0.0%) | 4.782 | 4.718 | 0.063 |
| prompt_last|L21|remove_common | 0/12 (0.0%) | 4.782 | 4.766 | 0.017 |
| prompt_last|L21|common_only | 0/12 (0.0%) | -0.078 | -0.076 | -0.003 |
| prompt_last|L21|remove_contrast | 0/12 (0.0%) | 4.782 | 4.718 | 0.063 |
| prompt_last|L21|contrast_only | 0/12 (0.0%) | 0.000 | 0.000 | 0.000 |
| prompt_last|L21|suppress_top_wrong | 0/12 (0.0%) | -13.780 | -13.807 | 0.027 |
| prompt_last|L21|boost_minus_suppress | 0/12 (0.0%) | 0.000 | 0.000 | 0.000 |

## Objective Facts

- DS7B full repair delta again raises correct and top-wrong together: +4.782 vs +4.718, switch 0/12.
- Removing the simple unembedding contrast changes nothing on DS7B, so the harmful shared activation is not captured by W(correct)-W(top_wrong).
- Common-only at DS7B L26 reproduces most of the shared gain: correct +4.318, top-wrong +4.313, margin +0.005.
- Simple suppress_top_wrong lowers correct and wrong together and does not improve winner switch.
- Phase588 therefore does not yet produce a controllable suppression patch. It shows the current unembedding-based decomposition is too crude.
