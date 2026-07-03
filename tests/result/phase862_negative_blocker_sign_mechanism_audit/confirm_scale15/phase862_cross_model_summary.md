# Phase 862 Negative-Blocker Sign Mechanism Audit (confirm_scale15)

- Source: Phase 861 Level 6 high-confidence signatures.
- Boundary: sign-mechanism audit, not gear search and not language closure.

## Cross-Model Summary

| model | status | rows | domains | clear modes by domain | interpretation |
|---|---|---:|---|---|---|
| qwen3 | complete | 30 | `['material']` | `{'material': []}` | `{'material': 'mode_specific_or_unresolved'}` |
| glm4 | no_level6_target | 0 | `[]` | `{}` | `{}` |
| deepseek7b | complete | 60 | `['animal', 'color']` | `{'animal': [], 'color': []}` | `{'animal': 'mode_specific_or_unresolved', 'color': 'mode_specific_or_unresolved'}` |

## Full-Set Effects

| model | domain | mode | clear gain/loss | blocker reduction | original blocker delta | answer delta | object delta | weaken? | answer lift? |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | material | `scale_up` | 0/0 | -1.3333 | 0.0104 | -0.2000 | 0.0417 | False | False |
| deepseek7b | animal | `scale_up` | 0/2 | -1.4000 | 0.0458 | -0.5750 | 0.0333 | False | False |
| deepseek7b | color | `scale_up` | 0/1 | -0.8000 | 0.0074 | -0.7250 | -0.1688 | False | False |
