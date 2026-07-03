# Phase 862 Negative-Blocker Sign Mechanism Audit (main)

- Source: Phase 861 Level 6 high-confidence signatures.
- Boundary: sign-mechanism audit, not gear search and not language closure.

## Cross-Model Summary

| model | status | rows | domains | clear modes by domain | interpretation |
|---|---|---:|---|---|---|
| qwen3 | complete | 195 | `['material']` | `{'material': ['flip', 'half', 'zero']}` | `{'material': 'shared_blocker_weakening'}` |
| glm4 | no_level6_target | 0 | `[]` | `{}` | `{}` |
| deepseek7b | complete | 390 | `['animal', 'color']` | `{'animal': ['flip', 'half', 'zero'], 'color': ['flip', 'half', 'zero']}` | `{'animal': 'shared_blocker_weakening', 'color': 'mode_specific_or_unresolved'}` |

## Full-Set Effects

| model | domain | mode | clear gain/loss | blocker reduction | original blocker delta | answer delta | object delta | weaken? | answer lift? |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | material | `flip` | 5/0 | 2.4000 | -0.1984 | 0.7583 | -0.3000 | True | True |
| qwen3 | material | `half` | 2/0 | 0.9333 | -0.0807 | 0.1833 | -0.0750 | True | True |
| qwen3 | material | `scale_up` | 0/0 | -2.4667 | 0.0286 | -0.4292 | 0.1083 | False | False |
| qwen3 | material | `zero` | 2/0 | 1.7333 | -0.0927 | 0.4083 | -0.1417 | True | True |
| deepseek7b | animal | `flip` | 10/0 | 1.2667 | -0.2382 | 2.9750 | -0.1417 | True | True |
| deepseek7b | animal | `half` | 5/0 | 0.6000 | -0.0618 | 0.6292 | -0.0250 | True | True |
| deepseek7b | animal | `scale_up` | 0/3 | -2.5333 | 0.0792 | -0.9854 | 0.0292 | False | False |
| deepseek7b | animal | `zero` | 6/0 | 0.8667 | -0.1083 | 1.3208 | -0.0437 | True | True |
| deepseek7b | color | `flip` | 5/0 | 1.1333 | -0.0006 | 4.9250 | 1.2458 | True | True |
| deepseek7b | color | `half` | 2/0 | 0.6667 | 0.0037 | 0.8833 | 0.2188 | False | True |
| deepseek7b | color | `scale_up` | 0/2 | -2.0667 | -0.0196 | -1.2542 | -0.2917 | False | False |
| deepseek7b | color | `zero` | 5/0 | 1.1333 | 0.0045 | 1.9792 | 0.4833 | False | True |
