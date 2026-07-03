# Phase 858 Cross-Domain Independent Gear Discovery and Isomorphism Audit (confirm)

- Source: independent domain-local class-vs-object readout support scan.
- Boundary: gear atlas discovery and isomorphism audit, not language closure.

## Cross-Model Summary

| model | domains | candidates | rows | positive domains | clear-gain domains | shared best gears | shared best layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 14120 | 288 | 8 | 2 | 0 | 4 |
| glm4 | 8 | 13931 | 288 | 7 | 3 | 1 | 3 |
| deepseek7b | 8 | 5258 | 288 | 6 | 4 | 0 | 3 |

## Best Domain Effects

| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `abstract` | `negative_blocker` | `flip` | `L30C8725` | 4 | 0.0023 | 0/0 | 0/0 | 0/0 | 0/0 | -0.2500 | 0.2656 |
| qwen3 | `animal` | `negative_blocker` | `flip` | `L27C811+L32C5164` | 4 | 18.7688 | 3/0 | 3/0 | 3/0 | 0/0 | 1.2500 | 3.8750 |
| qwen3 | `color` | `negative_blocker` | `flip` | `L29C3701+L32C4478` | 4 | 0.2766 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.8438 |
| qwen3 | `geometry` | `positive_supporter` | `zero` | `L29C1532` | 4 | 12.0516 | 2/0 | 2/0 | 2/0 | 0/0 | 0.5000 | -0.1562 |
| qwen3 | `material` | `negative_blocker` | `flip` | `L31C4800+L31C2257` | 4 | 0.9000 | 0/0 | 0/0 | 0/0 | 0/0 | 4.5000 | 1.5000 |
| qwen3 | `object` | `negative_blocker` | `flip` | `L30C5438+L31C409` | 4 | 5.8143 | 0/0 | 0/0 | 0/0 | 0/0 | 31.5000 | 7.2617 |
| qwen3 | `plant` | `negative_blocker` | `flip` | `L30C2274` | 4 | 0.0961 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.6406 |
| qwen3 | `tool` | `negative_blocker` | `flip` | `L29C7451+L31C8854` | 4 | 0.2133 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.4219 |
| glm4 | `abstract` | `negative_blocker` | `zero` | `L29C2300` | 4 | -0.0410 | 0/0 | 0/0 | 0/0 | 0/0 | -0.2500 | -0.0234 |
| glm4 | `animal` | `negative_blocker` | `flip` | `L29C4696+L29C13502` | 4 | 0.2379 | 0/0 | 0/0 | 0/0 | 0/0 | 0.5000 | 1.0859 |
| glm4 | `color` | `negative_blocker` | `flip` | `L30C7088+L30C11128` | 4 | 0.0574 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.3828 |
| glm4 | `geometry` | `negative_blocker` | `flip` | `L32C4909` | 4 | 0.0094 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0625 |
| glm4 | `material` | `negative_blocker` | `flip` | `L32C8466` | 8 | 12.0914 | 2/0 | 2/0 | 2/0 | 0/0 | 0.5000 | 0.1094 |
| glm4 | `object` | `positive_supporter` | `zero` | `L29C1214+L30C6115` | 4 | 6.0598 | 1/0 | 1/0 | 1/0 | 0/0 | 0.5000 | -0.1016 |
| glm4 | `plant` | `negative_blocker` | `flip` | `L32C8466+L26C1162` | 4 | 12.0551 | 2/0 | 2/0 | 2/0 | 0/0 | 0.7500 | -0.3828 |
| glm4 | `tool` | `negative_blocker` | `flip` | `L31C13504+L32C12804` | 4 | 0.1781 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 1.1875 |
| deepseek7b | `abstract` | `positive_supporter` | `zero` | `L27C14495+L27C3218` | 4 | 0.2133 | 0/0 | 0/0 | 0/0 | 0/0 | 2.0000 | -0.5781 |
| deepseek7b | `animal` | `negative_blocker` | `flip` | `L27C16651+L24C3875` | 4 | 24.7746 | 4/0 | 4/0 | 4/0 | 0/0 | 1.2500 | 3.9141 |
| deepseek7b | `color` | `negative_blocker` | `flip` | `L27C15369+L26C8587` | 4 | 8.1156 | 2/0 | 1/0 | 1/0 | 0/0 | 0.5000 | 6.9375 |
| deepseek7b | `geometry` | `negative_blocker` | `flip` | `L24C18863` | 4 | 0.0891 | 0/0 | 0/0 | 0/0 | 0/0 | 0.2500 | 0.3438 |
| deepseek7b | `material` | `negative_blocker` | `flip` | `L26C11106` | 4 | 6.1055 | 1/0 | 1/0 | 1/0 | 0/0 | 0.2500 | 0.4531 |
| deepseek7b | `object` | `negative_blocker` | `zero` | `L24C15056` | 4 | 0.0000 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | 0.0000 |
| deepseek7b | `plant` | `negative_blocker` | `zero` | `L27C1106` | 4 | -0.0480 | 0/0 | 0/0 | 0/0 | 0/0 | -0.2500 | -0.0703 |
| deepseek7b | `tool` | `negative_blocker` | `flip` | `L27C15841` | 4 | 6.6568 | 1/0 | 1/0 | 1/0 | 0/0 | 0.5000 | 3.8789 |

## Isomorphism Notes

### qwen3

- shared best gears: `{}`
- shared best layers: `{"L32": ["animal", "color"], "L29": ["color", "geometry", "tool"], "L30": ["abstract", "object", "plant"], "L31": ["material", "object", "tool"]}`

### glm4

- shared best gears: `{"L32C8466": ["material", "plant"]}`
- shared best layers: `{"L32": ["geometry", "material", "plant", "tool"], "L29": ["abstract", "animal", "object"], "L30": ["color", "object"]}`

### deepseek7b

- shared best gears: `{}`
- shared best layers: `{"L27": ["abstract", "animal", "color", "plant", "tool"], "L24": ["animal", "geometry", "object"], "L26": ["color", "material"]}`

