# Phase 859 Domain Gear Holdout and Sign Calibration Audit (smoke)

- Source: Phase 858 confirm top domain gears.
- Boundary: holdout/sign calibration, not language closure.

## Cross-Model Summary

| model | rows | best positive domains | best clear domains | alternate clear domains | control clear domains | shared probe clear domains |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 12 | 3 | 1 | 1 | 0 | 0 |
| glm4 | 12 | 2 | 0 | 0 | 0 | 0 |
| deepseek7b | 12 | 2 | 0 | 0 | 0 | 0 |

## Best Holdout Effects

| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `geometry` | `positive_supporter` | `zero` | `L29C1532` | 1 | 6.2812 | 1/0 | 1/0 | 1/0 | 0/0 | 1.0000 | 0.8750 |
| qwen3 | `animal` | `negative_blocker` | `flip` | `L27C811+L32C5164` | 1 | 0.0187 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.1250 |
| qwen3 | `tool` | `negative_blocker` | `flip` | `L29C7451+L31C8854` | 1 | 0.0094 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0625 |
| glm4 | `geometry` | `negative_blocker` | `flip` | `L32C4909` | 1 | 0.0187 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.1250 |
| glm4 | `animal` | `negative_blocker` | `flip` | `L29C4696+L29C13502` | 1 | 0.0094 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0625 |
| glm4 | `tool` | `negative_blocker` | `flip` | `L31C13504+L32C12804` | 1 | -0.0047 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.0312 |
| deepseek7b | `animal` | `negative_blocker` | `flip` | `L27C16651+L24C3875` | 1 | 0.5437 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 3.6250 |
| deepseek7b | `tool` | `negative_blocker` | `flip` | `L27C15841` | 1 | 0.5250 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 3.5000 |
| deepseek7b | `geometry` | `negative_blocker` | `flip` | `L24C18863` | 1 | 0.0000 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0000 |
