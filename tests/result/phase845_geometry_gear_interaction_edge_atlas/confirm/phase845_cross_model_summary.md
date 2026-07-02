# Phase 845 Geometry Gear Interaction Edge Atlas (confirm)

- Search: pair/triplet interaction residuals over Phase 844 top geometry gears.
- Boundary: interaction-edge atlas probe; not token closure.

## Model Summary

| model | gears | specs | rows | original target | target | lost | gained | synergy | antagonistic | additive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 51 | 765 | 2 | 185 | 0 | 83 | 19 | 15 | 536 |
| glm4 | 6 | 51 | 765 | 8 | 400 | 28 | 20 | 0 | 0 | 570 |
| deepseek7b | 6 | 51 | 765 | 6 | 378 | 1 | 73 | 3 | 0 | 567 |

## Top Gears

| model | rank | layer | channel | score | neg ratio |
|---|---:|---:|---:|---:|---:|
| qwen3 | 1 | 29 | 1532 | 16.0686 | 0.0000 |
| qwen3 | 2 | 30 | 2848 | 8.0151 | 1.0000 |
| qwen3 | 3 | 30 | 1349 | 6.3512 | 0.9231 |
| qwen3 | 4 | 27 | 2767 | 4.4762 | 1.0000 |
| qwen3 | 5 | 29 | 4588 | 3.7315 | 1.0000 |
| qwen3 | 6 | 30 | 5558 | 3.7105 | 1.0000 |
| glm4 | 1 | 28 | 2777 | 0.7090 | 0.0000 |
| glm4 | 2 | 30 | 6115 | 0.6769 | 0.0000 |
| glm4 | 3 | 26 | 6031 | 0.6353 | 0.0000 |
| glm4 | 4 | 28 | 8036 | 0.6229 | 0.1333 |
| glm4 | 5 | 29 | 10031 | 0.5797 | 0.0000 |
| glm4 | 6 | 27 | 10905 | 0.5430 | 1.0000 |
| deepseek7b | 1 | 27 | 15791 | 22.1959 | 1.0000 |
| deepseek7b | 2 | 27 | 1106 | 21.9991 | 1.0000 |
| deepseek7b | 3 | 27 | 15305 | 18.9811 | 1.0000 |
| deepseek7b | 4 | 25 | 4036 | 16.3187 | 0.0000 |
| deepseek7b | 5 | 27 | 13360 | 15.4794 | 0.0000 |
| deepseek7b | 6 | 27 | 2295 | 14.8089 | 1.0000 |

## Combo Type Summary

| model | combo | n | target | lost | gained | mean delta | mean residual | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `original` | 15 | 2 | 0 | 0 | 0.0000 | NA | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 2, "unknown_other": 11}` |
| qwen3 | `pair` | 450 | 107 | 0 | 47 | -0.5638 | -0.0119 | `{"broad_near_miss": 26, "object_echo": 35, "target_equivalent": 107, "unknown_other": 282}` |
| qwen3 | `single` | 180 | 35 | 0 | 11 | -0.2760 | NA | `{"broad_near_miss": 11, "object_echo": 13, "target_equivalent": 35, "unknown_other": 121}` |
| qwen3 | `triplet` | 120 | 41 | 0 | 25 | -1.4049 | -0.0477 | `{"broad_near_miss": 5, "object_echo": 11, "target_equivalent": 41, "unknown_other": 63}` |
| glm4 | `original` | 15 | 8 | 0 | 0 | 0.0000 | NA | `{"object_echo": 1, "target_equivalent": 8, "unknown_other": 6}` |
| glm4 | `pair` | 450 | 234 | 18 | 12 | 0.0925 | 0.0060 | `{"object_echo": 30, "target_equivalent": 234, "unknown_other": 186}` |
| glm4 | `single` | 180 | 95 | 6 | 5 | 0.0433 | NA | `{"object_echo": 12, "target_equivalent": 95, "unknown_other": 73}` |
| glm4 | `triplet` | 120 | 63 | 4 | 3 | 0.1075 | 0.0043 | `{"object_echo": 8, "target_equivalent": 63, "unknown_other": 49}` |
| deepseek7b | `original` | 15 | 6 | 0 | 0 | 0.0000 | NA | `{"object_echo": 2, "target_equivalent": 6, "unknown_other": 7}` |
| deepseek7b | `pair` | 450 | 223 | 1 | 44 | 0.0831 | 0.0163 | `{"broad_near_miss": 3, "object_echo": 66, "target_equivalent": 223, "unknown_other": 158}` |
| deepseek7b | `single` | 180 | 80 | 0 | 8 | 0.0334 | NA | `{"object_echo": 25, "target_equivalent": 80, "unknown_other": 75}` |
| deepseek7b | `triplet` | 120 | 69 | 0 | 21 | 0.1105 | 0.0311 | `{"broad_near_miss": 1, "object_echo": 18, "target_equivalent": 69, "unknown_other": 32}` |

## Interaction Class Summary

| model | class | n | target | lost | gained | mean residual | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `additive` | 536 | 129 | 0 | 61 | 0.0147 | `{"broad_near_miss": 29, "object_echo": 44, "target_equivalent": 129, "unknown_other": 334}` |
| qwen3 | `antagonistic` | 15 | 9 | 0 | 3 | -2.2406 | `{"object_echo": 2, "target_equivalent": 9, "unknown_other": 4}` |
| qwen3 | `synergy` | 19 | 10 | 0 | 8 | 0.7722 | `{"broad_near_miss": 2, "target_equivalent": 10, "unknown_other": 7}` |
| glm4 | `additive` | 570 | 297 | 22 | 15 | 0.0056 | `{"object_echo": 38, "target_equivalent": 297, "unknown_other": 235}` |
| deepseek7b | `additive` | 567 | 289 | 1 | 62 | 0.0166 | `{"broad_near_miss": 4, "object_echo": 84, "target_equivalent": 289, "unknown_other": 190}` |
| deepseek7b | `synergy` | 3 | 3 | 0 | 3 | 0.5417 | `{"target_equivalent": 3}` |

## Edge Summary

| model | type | mode | combo | n | synergy | antagonistic | additive | gained | lost | mean residual | mean abs residual |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L27C2767` | 15 | 2 | 6 | 7 | 3 | 0 | -0.8771 | 1.2437 |
| qwen3 | `pair` | `flip` | `L29C1532+L27C2767` | 15 | 1 | 6 | 8 | 2 | 0 | -0.9781 | 1.1865 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C1349` | 15 | 2 | 1 | 12 | 3 | 0 | 0.0448 | 0.2156 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L30C1349` | 15 | 2 | 1 | 12 | 4 | 0 | 0.0688 | 0.2146 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L29C4588` | 15 | 2 | 0 | 13 | 4 | 0 | 0.1344 | 0.1615 |
| qwen3 | `pair` | `flip` | `L29C1532+L29C4588` | 15 | 2 | 0 | 13 | 4 | 0 | 0.1135 | 0.1323 |
| qwen3 | `pair` | `flip` | `L30C2848+L27C2767` | 15 | 2 | 0 | 13 | 2 | 0 | 0.0698 | 0.1323 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L27C2767` | 15 | 1 | 0 | 14 | 3 | 0 | 0.0167 | 0.1875 |
| qwen3 | `pair` | `flip` | `L27C2767+L30C5558` | 15 | 1 | 0 | 14 | 2 | 0 | 0.1094 | 0.1427 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L30C5558` | 15 | 1 | 0 | 14 | 4 | 0 | 0.1146 | 0.1229 |
| qwen3 | `pair` | `flip` | `L30C1349+L27C2767` | 15 | 0 | 1 | 14 | 2 | 0 | -0.0865 | 0.1198 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C5558` | 15 | 1 | 0 | 14 | 4 | 0 | 0.0833 | 0.1125 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L30C5558` | 15 | 1 | 0 | 14 | 3 | 0 | 0.0583 | 0.0958 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C5558` | 15 | 1 | 0 | 14 | 2 | 0 | 0.0365 | 0.0677 |
| qwen3 | `pair` | `zero` | `L29C1532+L27C2767` | 15 | 0 | 0 | 15 | 3 | 0 | -0.0250 | 0.1667 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L30C1349` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0854 | 0.1062 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C1349` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0677 | 0.0844 |
| qwen3 | `pair` | `flip` | `L30C1349+L30C5558` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0104 | 0.0813 |
| qwen3 | `pair` | `flip` | `L27C2767+L29C4588` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0500 | 0.0750 |
| qwen3 | `pair` | `zero` | `L29C1532+L29C4588` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0552 | 0.0677 |
| qwen3 | `pair` | `zero` | `L27C2767+L30C5558` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0344 | 0.0677 |
| qwen3 | `pair` | `flip` | `L29C4588+L30C5558` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0250 | 0.0667 |
| qwen3 | `pair` | `zero` | `L30C1349+L30C5558` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0000 | 0.0583 |
| qwen3 | `pair` | `flip` | `L30C2848+L29C4588` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0302 | 0.0573 |
| qwen3 | `pair` | `zero` | `L29C4588+L30C5558` | 15 | 0 | 0 | 15 | 0 | 0 | -0.0021 | 0.0563 |
| qwen3 | `pair` | `zero` | `L27C2767+L29C4588` | 15 | 0 | 0 | 15 | 2 | 0 | -0.0010 | 0.0552 |
| qwen3 | `pair` | `zero` | `L30C2848+L27C2767` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0094 | 0.0531 |
| qwen3 | `pair` | `zero` | `L30C1349+L27C2767` | 15 | 0 | 0 | 15 | 2 | 0 | -0.0281 | 0.0531 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C2848` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0094 | 0.0510 |
| qwen3 | `pair` | `flip` | `L30C2848+L30C5558` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0104 | 0.0479 |
| glm4 | `pair` | `flip` | `L28C8036+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0660 | 0.0821 |
| glm4 | `pair` | `flip` | `L26C6031+L28C8036` | 15 | 0 | 0 | 15 | 1 | 1 | -0.0284 | 0.0633 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0070 | 0.0622 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L27C10905` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0049 | 0.0586 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L26C6031` | 15 | 0 | 0 | 15 | 1 | 1 | 0.0046 | 0.0572 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0104 | 0.0542 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L27C10905` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0070 | 0.0539 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0077 | 0.0530 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0141 | 0.0528 |
| glm4 | `pair` | `flip` | `L30C6115+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | -0.0006 | 0.0510 |
| glm4 | `pair` | `flip` | `L28C2777+L26C6031` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0018 | 0.0497 |
| glm4 | `pair` | `zero` | `L28C8036+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0202 | 0.0465 |
| glm4 | `pair` | `zero` | `L26C6031+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0180 | 0.0456 |
| glm4 | `pair` | `flip` | `L28C2777+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0104 | 0.0437 |
| glm4 | `pair` | `zero` | `L30C6115+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0029 | 0.0435 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L26C6031` | 15 | 0 | 0 | 15 | 0 | 0 | -0.0008 | 0.0435 |
| glm4 | `pair` | `flip` | `L28C2777+L27C10905` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0008 | 0.0383 |
| glm4 | `pair` | `zero` | `L28C2777+L26C6031` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0130 | 0.0380 |
| glm4 | `pair` | `flip` | `L26C6031+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | -0.0005 | 0.0380 |
| glm4 | `pair` | `zero` | `L28C2777+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | -0.0109 | 0.0375 |
| glm4 | `pair` | `zero` | `L28C2777+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | -0.0042 | 0.0367 |
| glm4 | `pair` | `flip` | `L30C6115+L27C10905` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0145 | 0.0363 |
| glm4 | `pair` | `flip` | `L28C2777+L30C6115` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0202 | 0.0358 |
| glm4 | `pair` | `zero` | `L30C6115+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0160 | 0.0350 |
| glm4 | `pair` | `flip` | `L28C2777+L29C10031` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0057 | 0.0349 |
| glm4 | `pair` | `flip` | `L30C6115+L26C6031` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0044 | 0.0346 |
| glm4 | `pair` | `zero` | `L28C8036+L27C10905` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0138 | 0.0336 |
| glm4 | `pair` | `zero` | `L28C2777+L27C10905` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0125 | 0.0323 |
| glm4 | `pair` | `flip` | `L30C6115+L28C8036` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0100 | 0.0314 |
| glm4 | `pair` | `zero` | `L28C2777+L30C6115` | 15 | 0 | 0 | 15 | 0 | 1 | 0.0076 | 0.0310 |
| deepseek7b | `pair` | `flip` | `L27C13360+L27C2295` | 15 | 1 | 0 | 14 | 2 | 0 | 0.0479 | 0.0667 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C2295` | 15 | 1 | 0 | 14 | 2 | 0 | 0.0365 | 0.0635 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C2295` | 15 | 1 | 0 | 14 | 2 | 0 | 0.0458 | 0.0521 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L25C4036` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0333 | 0.0625 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C13360` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0375 | 0.0583 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C13360` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0344 | 0.0531 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C2295` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0333 | 0.0521 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C15305` | 15 | 0 | 0 | 15 | 4 | 0 | 0.0208 | 0.0458 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C15305` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0240 | 0.0448 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C13360` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0271 | 0.0417 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L25C4036` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0292 | 0.0417 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C2295` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0250 | 0.0396 |
| deepseek7b | `pair` | `zero` | `L27C15305+L25C4036` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0146 | 0.0396 |
| deepseek7b | `pair` | `zero` | `L27C15791+L25C4036` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0115 | 0.0385 |
| deepseek7b | `pair` | `flip` | `L27C15305+L27C2295` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0240 | 0.0385 |
| deepseek7b | `pair` | `zero` | `L25C4036+L27C13360` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0156 | 0.0385 |
| deepseek7b | `pair` | `zero` | `L27C1106+L25C4036` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0187 | 0.0375 |
| deepseek7b | `pair` | `flip` | `L27C15305+L25C4036` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0021 | 0.0354 |
| deepseek7b | `pair` | `flip` | `L27C1106+L27C13360` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0135 | 0.0344 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C1106` | 15 | 0 | 0 | 15 | 3 | 0 | 0.0187 | 0.0333 |
| deepseek7b | `pair` | `flip` | `L27C1106+L27C15305` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0125 | 0.0333 |
| deepseek7b | `pair` | `zero` | `L27C15305+L27C13360` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0250 | 0.0333 |
| deepseek7b | `pair` | `zero` | `L27C13360+L27C2295` | 15 | 0 | 0 | 15 | 1 | 0 | 0.0229 | 0.0333 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C15305` | 15 | 0 | 0 | 15 | 4 | 0 | 0.0115 | 0.0323 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C1106` | 15 | 0 | 0 | 15 | 4 | 0 | 0.0187 | 0.0312 |
| deepseek7b | `pair` | `flip` | `L25C4036+L27C13360` | 15 | 0 | 0 | 15 | 1 | 1 | -0.0063 | 0.0312 |
| deepseek7b | `pair` | `zero` | `L25C4036+L27C2295` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0063 | 0.0312 |
| deepseek7b | `pair` | `zero` | `L27C1106+L27C2295` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0177 | 0.0281 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C13360` | 15 | 0 | 0 | 15 | 2 | 0 | 0.0167 | 0.0271 |
| deepseek7b | `pair` | `flip` | `L27C1106+L25C4036` | 15 | 0 | 0 | 15 | 0 | 0 | 0.0177 | 0.0260 |
