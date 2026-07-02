# Phase 845 Geometry Gear Interaction Edge Atlas (smoke)

- Search: pair/triplet interaction residuals over Phase 844 top geometry gears.
- Boundary: interaction-edge atlas probe; not token closure.

## Model Summary

| model | gears | specs | rows | original target | target | lost | gained | synergy | antagonistic | additive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 21 | 21 | 0 | 3 | 0 | 3 | 1 | 0 | 11 |
| glm4 | 4 | 21 | 21 | 1 | 21 | 0 | 0 | 0 | 0 | 12 |
| deepseek7b | 4 | 21 | 21 | 0 | 5 | 0 | 5 | 0 | 0 | 12 |

## Top Gears

| model | rank | layer | channel | score | neg ratio |
|---|---:|---:|---:|---:|---:|
| qwen3 | 1 | 29 | 1532 | 16.0686 | 0.0000 |
| qwen3 | 2 | 30 | 2848 | 8.0151 | 1.0000 |
| qwen3 | 3 | 30 | 1349 | 6.3512 | 0.9231 |
| qwen3 | 4 | 27 | 2767 | 4.4762 | 1.0000 |
| glm4 | 1 | 28 | 2777 | 0.7090 | 0.0000 |
| glm4 | 2 | 30 | 6115 | 0.6769 | 0.0000 |
| glm4 | 3 | 26 | 6031 | 0.6353 | 0.0000 |
| glm4 | 4 | 28 | 8036 | 0.6229 | 0.1333 |
| deepseek7b | 1 | 27 | 15791 | 22.1959 | 1.0000 |
| deepseek7b | 2 | 27 | 1106 | 21.9991 | 1.0000 |
| deepseek7b | 3 | 27 | 15305 | 18.9811 | 1.0000 |
| deepseek7b | 4 | 25 | 4036 | 16.3187 | 0.0000 |

## Combo Type Summary

| model | combo | n | target | lost | gained | mean delta | mean residual | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `original` | 1 | 0 | 0 | 0 | 0.0000 | NA | `{"broad_near_miss": 1}` |
| qwen3 | `pair` | 12 | 2 | 0 | 2 | 0.1719 | 0.1094 | `{"broad_near_miss": 10, "target_equivalent": 2}` |
| qwen3 | `single` | 8 | 1 | 0 | 1 | 0.0312 | NA | `{"broad_near_miss": 7, "target_equivalent": 1}` |
| glm4 | `original` | 1 | 1 | 0 | 0 | 0.0000 | NA | `{"target_equivalent": 1}` |
| glm4 | `pair` | 12 | 12 | 0 | 0 | -0.1289 | -0.0117 | `{"target_equivalent": 12}` |
| glm4 | `single` | 8 | 8 | 0 | 0 | -0.0586 | NA | `{"target_equivalent": 8}` |
| deepseek7b | `original` | 1 | 0 | 0 | 0 | 0.0000 | NA | `{"unknown_other": 1}` |
| deepseek7b | `pair` | 12 | 4 | 0 | 4 | 0.1406 | 0.0703 | `{"target_equivalent": 4, "unknown_other": 8}` |
| deepseek7b | `single` | 8 | 1 | 0 | 1 | 0.0352 | NA | `{"target_equivalent": 1, "unknown_other": 7}` |

## Interaction Class Summary

| model | class | n | target | lost | gained | mean residual | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `additive` | 11 | 2 | 0 | 2 | 0.0455 | `{"broad_near_miss": 9, "target_equivalent": 2}` |
| qwen3 | `synergy` | 1 | 0 | 0 | 0 | 0.8125 | `{"broad_near_miss": 1}` |
| glm4 | `additive` | 12 | 12 | 0 | 0 | -0.0117 | `{"target_equivalent": 12}` |
| deepseek7b | `additive` | 12 | 4 | 0 | 4 | 0.0703 | `{"target_equivalent": 4, "unknown_other": 8}` |

## Edge Summary

| model | type | mode | combo | n | synergy | antagonistic | additive | gained | lost | mean residual | mean abs residual |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `pair` | `flip` | `L30C2848+L27C2767` | 1 | 1 | 0 | 0 | 0 | 0 | 0.8125 | 0.8125 |
| qwen3 | `pair` | `flip` | `L30C1349+L27C2767` | 1 | 0 | 0 | 1 | 0 | 0 | -0.1875 | 0.1875 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C2848` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C2848` | 1 | 0 | 0 | 1 | 1 | 0 | 0.1250 | 0.1250 |
| qwen3 | `pair` | `zero` | `L29C1532+L27C2767` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| qwen3 | `pair` | `flip` | `L29C1532+L27C2767` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| qwen3 | `pair` | `zero` | `L30C2848+L27C2767` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C1349` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0625 | 0.0625 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C1349` | 1 | 0 | 0 | 1 | 1 | 0 | 0.0625 | 0.0625 |
| qwen3 | `pair` | `zero` | `L30C1349+L27C2767` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0625 | 0.0625 |
| qwen3 | `pair` | `zero` | `L30C2848+L30C1349` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0000 | 0.0000 |
| qwen3 | `pair` | `flip` | `L30C2848+L30C1349` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0000 | 0.0000 |
| glm4 | `pair` | `zero` | `L28C2777+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0469 | 0.0469 |
| glm4 | `pair` | `zero` | `L30C6115+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0469 | 0.0469 |
| glm4 | `pair` | `zero` | `L26C6031+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0469 | 0.0469 |
| glm4 | `pair` | `flip` | `L28C2777+L30C6115` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0312 | 0.0312 |
| glm4 | `pair` | `zero` | `L30C6115+L26C6031` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0312 | 0.0312 |
| glm4 | `pair` | `flip` | `L26C6031+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0312 | 0.0312 |
| glm4 | `pair` | `zero` | `L28C2777+L30C6115` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0156 | 0.0156 |
| glm4 | `pair` | `zero` | `L28C2777+L26C6031` | 1 | 0 | 0 | 1 | 0 | 0 | -0.0156 | 0.0156 |
| glm4 | `pair` | `flip` | `L28C2777+L26C6031` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0156 | 0.0156 |
| glm4 | `pair` | `flip` | `L28C2777+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0156 | 0.0156 |
| glm4 | `pair` | `flip` | `L30C6115+L26C6031` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0000 | 0.0000 |
| glm4 | `pair` | `flip` | `L30C6115+L28C8036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0000 | 0.0000 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C15305` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| deepseek7b | `pair` | `zero` | `L27C15305+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| deepseek7b | `pair` | `flip` | `L27C15305+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.1250 | 0.1250 |
| deepseek7b | `pair` | `flip` | `L27C1106+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0938 | 0.0938 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C1106` | 1 | 0 | 0 | 1 | 1 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `zero` | `L27C15791+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `flip` | `L27C15791+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `zero` | `L27C1106+L27C15305` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `flip` | `L27C1106+L27C15305` | 1 | 0 | 0 | 1 | 1 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `zero` | `L27C1106+L25C4036` | 1 | 0 | 0 | 1 | 0 | 0 | 0.0625 | 0.0625 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C1106` | 1 | 0 | 0 | 1 | 1 | 0 | 0.0000 | 0.0000 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C15305` | 1 | 0 | 0 | 1 | 1 | 0 | 0.0000 | 0.0000 |
