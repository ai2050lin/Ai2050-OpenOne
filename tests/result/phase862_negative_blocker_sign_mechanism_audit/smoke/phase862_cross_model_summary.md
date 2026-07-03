# Phase 862 Negative-Blocker Sign Mechanism Audit (smoke)

- Source: Phase 861 Level 6 high-confidence signatures.
- Boundary: sign-mechanism audit, not gear search and not language closure.

## Cross-Model Summary

| model | status | rows | domains | clear modes by domain | interpretation |
|---|---|---:|---|---|---|
| qwen3 | complete | 7 | `['material']` | `{'material': []}` | `{'material': 'mode_specific_or_unresolved'}` |
| glm4 | no_level6_target | 0 | `[]` | `{}` | `{}` |
| deepseek7b | complete | 14 | `['animal', 'color']` | `{'animal': ['flip', 'zero'], 'color': []}` | `{'animal': 'shared_blocker_weakening', 'color': 'mode_specific_or_unresolved'}` |

## Full-Set Effects

| model | domain | mode | clear gain/loss | blocker reduction | original blocker delta | answer delta | object delta | weaken? | answer lift? |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | material | `flip` | 0/0 | 14.0000 | -0.2500 | 2.3750 | -0.8750 | False | False |
| qwen3 | material | `zero` | 0/0 | 9.0000 | -0.0875 | 1.3750 | -0.3750 | False | False |
| deepseek7b | animal | `flip` | 1/0 | 1.0000 | -0.0625 | 4.2500 | -0.5625 | True | True |
| deepseek7b | animal | `zero` | 1/0 | 1.0000 | -0.0625 | 2.1875 | -0.1875 | True | True |
| deepseek7b | color | `flip` | 0/0 | 1.0000 | 0.0625 | 7.6875 | 0.1875 | False | False |
| deepseek7b | color | `zero` | 0/0 | 1.0000 | 0.0625 | 3.1875 | 0.1250 | False | False |
