# Phase 876 Nonclean Route Causal Validation (validation)

- Source: Phase 865 gear candidates with new Phase 876 objects and route-pressure prompts.
- Boundary: causal validation input for Phase 870/872/874/875; not closure.

| model | status | candidates | domains | source-clean -> validation-clean stats |
|---|---|---:|---|---|
| qwen3 | complete | 4 | `['material']` | `{'n': 4, 'tp': 0, 'fp': 3, 'fn': 0, 'tn': 1, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.25, 'source_clean_count': 3, 'holdout_clean_count': 0}` |
| glm4 | no_phase865_candidates | 0 | `[]` | `{}` |
| deepseek7b | complete | 8 | `['animal', 'color']` | `{'n': 8, 'tp': 0, 'fp': 3, 'fn': 0, 'tn': 5, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.625, 'source_clean_count': 3, 'holdout_clean_count': 0}` |

- Source purity counts: `{'clean_mixed_answer_blocker_route': 6, 'inactive_or_weak': 1, 'harmful_or_unstable': 2, 'object_side_effect_risk': 2, 'clean_answer_lift_route': 1}`
- Overall stats: `{'n': 12, 'tp': 0, 'fp': 6, 'fn': 0, 'tn': 6, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.5, 'source_clean_count': 6, 'holdout_clean_count': 0}`
