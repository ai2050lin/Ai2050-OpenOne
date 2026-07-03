# Phase 871 Field Admissibility External Validation (validation)

- Source: Phase 865 candidates, new validation objects and prompts.
- Boundary: external validation data collection; FieldAdmissible scoring is run separately.

| model | status | candidates | domains | source-clean -> holdout-clean stats |
|---|---|---:|---|---|
| qwen3 | complete | 4 | `['material']` | `{'n': 4, 'tp': 0, 'fp': 3, 'fn': 0, 'tn': 1, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.25, 'source_clean_count': 3, 'holdout_clean_count': 0}` |
| glm4 | no_phase865_candidates | 0 | `[]` | `{}` |
| deepseek7b | complete | 8 | `['animal', 'color']` | `{'n': 8, 'tp': 0, 'fp': 3, 'fn': 0, 'tn': 5, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.625, 'source_clean_count': 3, 'holdout_clean_count': 0}` |
