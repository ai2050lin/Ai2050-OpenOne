# Phase 858 Cross-Domain Independent Gear Discovery and Isomorphism Audit (main)

- Source: independent domain-local class-vs-object readout support scan.
- Boundary: gear atlas discovery and isomorphism audit, not language closure.

## Cross-Model Summary

| model | domains | candidates | rows | positive domains | clear-gain domains | shared best gears | shared best layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 3339 | 108 | 6 | 1 | 0 | 3 |
| glm4 | 6 | 3270 | 108 | 6 | 0 | 0 | 3 |
| deepseek7b | 6 | 1439 | 108 | 6 | 1 | 0 | 2 |

## Best Domain Effects

| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `abstract` | `negative_blocker` | `zero` | `L30C8725+L30C2992` | 2 | 0.5016 | 0/0 | 0/0 | 0/0 | 0/0 | 3.0000 | 0.3438 |
| qwen3 | `animal` | `negative_blocker` | `flip` | `L32C9411+L30C1469` | 2 | 0.3094 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 1.5625 |
| qwen3 | `color` | `negative_blocker` | `flip` | `L32C4478+L30C8643` | 2 | 0.1781 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.1875 |
| qwen3 | `geometry` | `negative_blocker` | `flip` | `L30C2848+L28C4038` | 2 | 0.4219 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 2.8125 |
| qwen3 | `material` | `negative_blocker` | `flip` | `L30C7770+L28C5020` | 2 | 8.1281 | 1/0 | 1/0 | 1/0 | 0/0 | 10.5000 | 3.6875 |
| qwen3 | `tool` | `negative_blocker` | `flip` | `L32C8830` | 2 | 0.1687 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.1250 |
| glm4 | `abstract` | `negative_blocker` | `flip` | `L24C5154+L30C11788` | 2 | 0.7172 | 0/0 | 0/0 | 0/0 | 0/0 | 4.5000 | 0.2812 |
| glm4 | `animal` | `negative_blocker` | `flip` | `L30C2432+L28C7104` | 2 | 0.1570 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 0.5469 |
| glm4 | `color` | `negative_blocker` | `flip` | `L30C7088+L26C4566` | 2 | 0.1031 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.6875 |
| glm4 | `geometry` | `negative_blocker` | `flip` | `L32C4909+L28C6279` | 2 | 0.0422 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.2812 |
| glm4 | `material` | `negative_blocker` | `flip` | `L32C8466+L32C5188` | 2 | 0.3750 | 0/0 | 0/0 | 0/0 | 0/0 | 1.5000 | 1.0000 |
| glm4 | `tool` | `negative_blocker` | `flip` | `L32C12804+L28C10241` | 2 | 0.0961 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.6406 |
| deepseek7b | `abstract` | `negative_blocker` | `flip` | `L24C1599` | 2 | 1.6609 | 1/0 | 0/0 | 0/0 | 0/0 | 3.5000 | 0.9062 |
| deepseek7b | `animal` | `negative_blocker` | `flip` | `L24C3875+L26C8629` | 2 | 0.3094 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 1.5625 |
| deepseek7b | `color` | `negative_blocker` | `flip` | `L26C8587` | 2 | 6.1547 | 1/0 | 1/0 | 1/0 | 0/0 | 0.5000 | 0.5312 |
| deepseek7b | `geometry` | `negative_blocker` | `flip` | `L24C18863` | 2 | 0.1781 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 0.6875 |
| deepseek7b | `material` | `negative_blocker` | `flip` | `L26C985` | 2 | 0.9891 | 0/0 | 0/0 | 0/0 | 0/0 | 6.0000 | 0.5938 |
| deepseek7b | `tool` | `negative_blocker` | `flip` | `L24C779+L24C1878` | 2 | 0.1875 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.2500 |

## Isomorphism Notes

### qwen3

- shared best gears: `{}`
- shared best layers: `{"L30": ["abstract", "animal", "color", "geometry", "material"], "L28": ["geometry", "material"], "L32": ["animal", "color", "tool"]}`

### glm4

- shared best gears: `{}`
- shared best layers: `{"L30": ["abstract", "animal", "color"], "L32": ["geometry", "material", "tool"], "L28": ["animal", "geometry", "tool"]}`

### deepseek7b

- shared best gears: `{}`
- shared best layers: `{"L26": ["animal", "color", "material"], "L24": ["abstract", "animal", "geometry", "tool"]}`

