# Phase 869 Blocker Field Conditional Gate Audit (holdout)

- Source: Phase 867 original rows + Phase 868 transfer taxonomy.
- Boundary: blocker field description, not closure.

| model_domain | field profile | transfer status | failure reasons | mean blockers | margin class-object | object rank | top10 roles/context |
|---|---|---|---|---:|---:|---:|---|
| `deepseek7b:animal` | `['semantic_other_pressure']` | `{'source_clean_failed': 3, 'stable_nonclean': 1}` | `{'original_blocker_not_negative': 4, 'no_clear_gain': 2, 'clear_loss': 1, 'answer_not_lifted': 1, 'blocker_not_reduced': 1, 'format_or_other_side_effect': 1}` | 5.1667 | 3.2604 | 38.6667 | `{'other_blocker': 1.0833333333333333, 'format_punct': 0.8333333333333334, 'protocol_word': 0.75, 'format_space': 0.5, 'object_echo': 0.16666666666666666}` |
| `deepseek7b:color` | `['format_pressure', 'semantic_other_pressure']` | `{'stable_nonclean': 2, 'emergent_clean': 2}` | `{'original_blocker_not_negative': 1, 'no_clear_gain': 1, 'clear_loss': 1, 'answer_not_lifted': 1, 'blocker_not_reduced': 1, 'format_or_other_side_effect': 1}` | 17.5000 | 1.9375 | 37.3333 | `{'format_punct': 1.5, 'other_blocker': 1.4166666666666667, 'protocol_word': 0.6666666666666666, 'format_space': 0.5833333333333334, 'object_echo': 0.5}` |
| `qwen3:material` | `['high_blocker_count', 'object_above_class', 'format_pressure', 'object_echo_pressure', 'semantic_other_pressure']` | `{'source_clean_failed': 3, 'stable_nonclean': 1}` | `{'original_blocker_not_negative': 3, 'no_clear_gain': 3, 'answer_not_lifted': 1, 'blocker_not_reduced': 1}` | 58.3333 | -1.6979 | 14.0833 | `{'format_punct': 2.9166666666666665, 'format_space': 2.3333333333333335, 'other_blocker': 1.1666666666666667, 'object_echo': 1.0, 'protocol_word': 0.5833333333333334}` |
