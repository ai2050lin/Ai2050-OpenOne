# Phase 858 Cross-Domain Independent Gear Discovery and Isomorphism Audit (smoke)

- Source: independent domain-local class-vs-object readout support scan.
- Boundary: gear atlas discovery and isomorphism audit, not language closure.

## Cross-Model Summary

| model | domains | candidates | rows | positive domains | clear-gain domains | shared best gears | shared best layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 3 | 432 | 21 | 3 | 1 | 0 | 2 |
| glm4 | 3 | 432 | 21 | 3 | 0 | 0 | 2 |
| deepseek7b | 3 | 144 | 21 | 3 | 0 | 0 | 1 |

## Best Domain Effects

| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `animal` | `negative_blocker` | `flip` | `L30C1469+L30C1867` | 1 | 6.9000 | 1/0 | 1/0 | 1/0 | 0/0 | 2.0000 | 4.0000 |
| qwen3 | `geometry` | `negative_blocker` | `flip` | `L30C2848+L28C4038` | 1 | 0.4688 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 3.1250 |
| qwen3 | `tool` | `negative_blocker` | `flip` | `L28C4765+L28C1208` | 1 | 0.4219 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 2.8125 |
| glm4 | `animal` | `negative_blocker` | `flip` | `L30C2432+L28C7104` | 1 | 0.2344 | 0/0 | 0/0 | 0/0 | 0/0 | 1.0000 | 0.5625 |
| glm4 | `geometry` | `negative_blocker` | `flip` | `L28C6279+L30C6115` | 1 | 0.0656 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.4375 |
| glm4 | `tool` | `negative_blocker` | `flip` | `L28C10241+L28C13510` | 1 | 0.1031 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.6875 |
| deepseek7b | `animal` | `negative_blocker` | `flip` | `L26C3270+L26C13069` | 1 | 0.0281 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.1875 |
| deepseek7b | `geometry` | `negative_blocker` | `flip` | `L26C13399+L26C17883` | 1 | 0.1875 | 0/0 | 0/0 | 0/0 | 0/0 | 1.0000 | 0.2500 |
| deepseek7b | `tool` | `negative_blocker` | `zero` | `L26C8121` | 1 | 0.0328 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.2188 |

## Isomorphism Notes

### qwen3

- shared best gears: `{}`
- shared best layers: `{"L30": ["animal", "geometry"], "L28": ["geometry", "tool"]}`

### glm4

- shared best gears: `{}`
- shared best layers: `{"L30": ["animal", "geometry"], "L28": ["animal", "geometry", "tool"]}`

### deepseek7b

- shared best gears: `{}`
- shared best layers: `{"L26": ["animal", "geometry", "tool"]}`

