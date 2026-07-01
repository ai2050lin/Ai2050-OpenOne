# Phase 816 Multi Token Answer Span Rollout Closure (smoke)

- Boundary: target phrase must be multi-token; closure is tested by teacher-forced span score and greedy rollout.

## Model Summary

| model | rows | cases | multi-token rows | span-score | rollout | full | contrast cleared | generic cleared | strict step top1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"span_score_and_rollout_closed": 4}` |
| glm4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"span_score_and_rollout_closed": 4}` |
| deepseek7b | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | `{"span_score_and_rollout_closed": 4}` |

## Best Rows

| model | variant | case | target | best target | best non-target | span-score | rollout | full | margin | generated | label |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.467 | `edible fruit` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.936 | `eating utensil` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.901 | `living thing` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.697 | `medical worker` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` musical instrument`/distractor | 1 | 1 | 1 | 3.376 | `edible fruit` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` warm color`/distractor | 1 | 1 | 1 | 2.750 | `medical worker` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` body organ`/distractor | 1 | 1 | 1 | 2.617 | `eating utensil` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 2.125 | `living thing` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.642 | `edible fruit` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.887 | `medical worker` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` body organ`/distractor | 1 | 1 | 1 | 1.387 | `eating utensil` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 0.781 | `living thing` | `span_score_and_rollout_closed` |
