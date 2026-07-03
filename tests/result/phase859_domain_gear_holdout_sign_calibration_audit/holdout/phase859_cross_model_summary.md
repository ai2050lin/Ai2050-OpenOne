# Phase 859 Domain Gear Holdout and Sign Calibration Audit (holdout)

- Source: Phase 858 confirm top domain gears.
- Boundary: holdout/sign calibration, not language closure.

## Cross-Model Summary

| model | rows | best positive domains | best clear domains | alternate clear domains | control clear domains | shared probe clear domains |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 128 | 6 | 2 | 1 | 0 | 0 |
| glm4 | 160 | 7 | 1 | 1 | 0 | 0 |
| deepseek7b | 128 | 6 | 2 | 2 | 0 | 0 |

## Best Holdout Effects

| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `material` | `negative_blocker` | `flip` | `L31C4800+L31C2257` | 4 | 6.7031 | 1/0 | 1/0 | 1/0 | 0/0 | 3.7500 | 0.9375 |
| qwen3 | `geometry` | `positive_supporter` | `zero` | `L29C1532` | 4 | 5.9695 | 1/0 | 1/0 | 1/0 | 0/0 | 0.2500 | -0.4531 |
| qwen3 | `color` | `negative_blocker` | `flip` | `L29C3701+L32C4478` | 4 | 0.1266 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.8438 |
| qwen3 | `object` | `negative_blocker` | `flip` | `L30C5438+L31C409` | 4 | 0.0727 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | -0.0156 |
| qwen3 | `tool` | `negative_blocker` | `flip` | `L29C7451+L31C8854` | 4 | 0.0727 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.4844 |
| qwen3 | `animal` | `negative_blocker` | `flip` | `L27C811+L32C5164` | 4 | 0.0187 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.1250 |
| qwen3 | `plant` | `negative_blocker` | `flip` | `L30C2274` | 4 | -0.0070 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.0469 |
| qwen3 | `abstract` | `negative_blocker` | `flip` | `L30C8725` | 4 | -0.2648 | 0/0 | 0/0 | 0/0 | 0/0 | -1.7500 | -0.0156 |
| glm4 | `color` | `negative_blocker` | `flip` | `L30C7088+L30C11128` | 4 | 7.0844 | 1/0 | 1/0 | 1/0 | 1/0 | 0.2500 | 0.3125 |
| glm4 | `object` | `positive_supporter` | `zero` | `L29C1214+L30C6115` | 4 | 0.0844 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 0.0625 |
| glm4 | `abstract` | `negative_blocker` | `zero` | `L29C2300` | 4 | 0.0387 | 0/0 | 0/0 | 0/0 | 0/0 | 0.2500 | 0.0078 |
| glm4 | `plant` | `negative_blocker` | `flip` | `L32C8466+L26C1162` | 4 | 0.0070 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0469 |
| glm4 | `animal` | `negative_blocker` | `flip` | `L29C4696+L29C13502` | 4 | 0.0035 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0234 |
| glm4 | `geometry` | `negative_blocker` | `flip` | `L32C4909` | 4 | 0.0023 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0156 |
| glm4 | `material` | `negative_blocker` | `flip` | `L32C8466` | 4 | 0.0012 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0078 |
| glm4 | `tool` | `negative_blocker` | `flip` | `L31C13504+L32C12804` | 4 | -0.0492 | 0/0 | 0/0 | 0/0 | 0/0 | -0.2500 | -0.0781 |
| deepseek7b | `color` | `negative_blocker` | `flip` | `L27C15369+L26C8587` | 4 | 12.9270 | 2/0 | 2/0 | 2/0 | 0/0 | 1.0000 | 5.1797 |
| deepseek7b | `animal` | `negative_blocker` | `flip` | `L27C16651+L24C3875` | 4 | 6.7711 | 1/0 | 1/0 | 1/0 | 0/0 | 2.7500 | 2.3906 |
| deepseek7b | `tool` | `negative_blocker` | `flip` | `L27C15841` | 4 | 0.5473 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 3.6484 |
| deepseek7b | `object` | `negative_blocker` | `zero` | `L24C15056` | 4 | 0.1523 | 0/0 | 0/0 | 0/0 | 0/0 | 1.0000 | 0.0156 |
| deepseek7b | `material` | `negative_blocker` | `flip` | `L26C11106` | 4 | 0.0352 | 0/0 | 0/0 | 0/0 | 0/0 | 0.2500 | -0.0156 |
| deepseek7b | `geometry` | `negative_blocker` | `flip` | `L24C18863` | 4 | 0.0023 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0156 |
| deepseek7b | `plant` | `negative_blocker` | `zero` | `L27C1106` | 4 | -0.0035 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.0234 |
| deepseek7b | `abstract` | `positive_supporter` | `zero` | `L27C14495+L27C3218` | 4 | -0.0352 | 0/0 | 0/0 | 0/0 | 0/0 | -0.2500 | 0.0156 |
