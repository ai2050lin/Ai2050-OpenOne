# Phase 845 Geometry Gear Interaction Edge Atlas (main)

- Search: pair/triplet interaction residuals over Phase 844 top geometry gears.
- Boundary: interaction-edge atlas probe; not token closure.

## Model Summary

| model | gears | specs | rows | original target | target | lost | gained | synergy | antagonistic | additive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 51 | 408 | 2 | 164 | 0 | 62 | 17 | 15 | 272 |
| glm4 | 6 | 51 | 408 | 7 | 349 | 28 | 20 | 0 | 0 | 304 |
| deepseek7b | 6 | 51 | 408 | 4 | 269 | 1 | 66 | 3 | 0 | 301 |

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
| qwen3 | `original` | 8 | 2 | 0 | 0 | 0.0000 | NA | `{"broad_near_miss": 1, "target_equivalent": 2, "unknown_other": 5}` |
| qwen3 | `pair` | 240 | 97 | 0 | 37 | -0.7196 | -0.0298 | `{"broad_near_miss": 26, "object_echo": 5, "target_equivalent": 97, "unknown_other": 112}` |
| qwen3 | `single` | 96 | 32 | 0 | 8 | -0.3449 | NA | `{"broad_near_miss": 11, "object_echo": 1, "target_equivalent": 32, "unknown_other": 52}` |
| qwen3 | `triplet` | 64 | 33 | 0 | 17 | -1.6626 | -0.1221 | `{"broad_near_miss": 5, "object_echo": 3, "target_equivalent": 33, "unknown_other": 23}` |
| glm4 | `original` | 8 | 7 | 0 | 0 | 0.0000 | NA | `{"target_equivalent": 7, "unknown_other": 1}` |
| glm4 | `pair` | 240 | 204 | 18 | 12 | 0.0402 | 0.0094 | `{"target_equivalent": 204, "unknown_other": 36}` |
| glm4 | `single` | 96 | 83 | 6 | 5 | 0.0154 | NA | `{"target_equivalent": 83, "unknown_other": 13}` |
| glm4 | `triplet` | 64 | 55 | 4 | 3 | 0.0632 | 0.0052 | `{"target_equivalent": 55, "unknown_other": 9}` |
| deepseek7b | `original` | 8 | 4 | 0 | 0 | 0.0000 | NA | `{"target_equivalent": 4, "unknown_other": 4}` |
| deepseek7b | `pair` | 240 | 160 | 1 | 41 | 0.0755 | 0.0279 | `{"broad_near_miss": 3, "object_echo": 6, "target_equivalent": 160, "unknown_other": 71}` |
| deepseek7b | `single` | 96 | 55 | 0 | 7 | 0.0238 | NA | `{"object_echo": 1, "target_equivalent": 55, "unknown_other": 40}` |
| deepseek7b | `triplet` | 64 | 50 | 0 | 18 | 0.0825 | 0.0491 | `{"broad_near_miss": 1, "object_echo": 2, "target_equivalent": 50, "unknown_other": 11}` |

## Interaction Class Summary

| model | class | n | target | lost | gained | mean residual | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `additive` | 272 | 111 | 0 | 43 | 0.0185 | `{"broad_near_miss": 29, "object_echo": 6, "target_equivalent": 111, "unknown_other": 126}` |
| qwen3 | `antagonistic` | 15 | 9 | 0 | 3 | -2.2406 | `{"object_echo": 2, "target_equivalent": 9, "unknown_other": 4}` |
| qwen3 | `synergy` | 17 | 10 | 0 | 8 | 0.8006 | `{"broad_near_miss": 2, "target_equivalent": 10, "unknown_other": 5}` |
| glm4 | `additive` | 304 | 259 | 22 | 15 | 0.0085 | `{"target_equivalent": 259, "unknown_other": 45}` |
| deepseek7b | `additive` | 301 | 207 | 1 | 56 | 0.0273 | `{"broad_near_miss": 4, "object_echo": 8, "target_equivalent": 207, "unknown_other": 82}` |
| deepseek7b | `synergy` | 3 | 3 | 0 | 3 | 0.5417 | `{"target_equivalent": 3}` |

## Edge Summary

| model | type | mode | combo | n | synergy | antagonistic | additive | gained | lost | mean residual | mean abs residual |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L27C2767` | 8 | 1 | 6 | 1 | 2 | 0 | -1.8008 | 2.1289 |
| qwen3 | `pair` | `flip` | `L29C1532+L27C2767` | 8 | 0 | 6 | 2 | 1 | 0 | -2.0098 | 2.0410 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C1349` | 8 | 2 | 1 | 5 | 2 | 0 | 0.0996 | 0.3730 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L30C1349` | 8 | 2 | 1 | 5 | 3 | 0 | 0.1523 | 0.3633 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L29C4588` | 8 | 2 | 0 | 6 | 3 | 0 | 0.2246 | 0.2520 |
| qwen3 | `pair` | `flip` | `L30C2848+L27C2767` | 8 | 2 | 0 | 6 | 2 | 0 | 0.1660 | 0.2129 |
| qwen3 | `pair` | `flip` | `L29C1532+L29C4588` | 8 | 2 | 0 | 6 | 3 | 0 | 0.1738 | 0.2090 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L27C2767` | 8 | 1 | 0 | 7 | 2 | 0 | -0.0195 | 0.2695 |
| qwen3 | `pair` | `flip` | `L27C2767+L30C5558` | 8 | 1 | 0 | 7 | 2 | 0 | 0.1816 | 0.2285 |
| qwen3 | `triplet` | `flip` | `L29C1532+L30C2848+L30C5558` | 8 | 1 | 0 | 7 | 3 | 0 | 0.1953 | 0.1953 |
| qwen3 | `pair` | `flip` | `L30C1349+L27C2767` | 8 | 0 | 1 | 7 | 2 | 0 | -0.1230 | 0.1855 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C5558` | 8 | 1 | 0 | 7 | 3 | 0 | 0.1406 | 0.1641 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L30C5558` | 8 | 1 | 0 | 7 | 2 | 0 | 0.1055 | 0.1367 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C5558` | 8 | 1 | 0 | 7 | 2 | 0 | 0.0645 | 0.0996 |
| qwen3 | `pair` | `zero` | `L29C1532+L27C2767` | 8 | 0 | 0 | 8 | 2 | 0 | -0.0938 | 0.2188 |
| qwen3 | `triplet` | `zero` | `L29C1532+L30C2848+L30C1349` | 8 | 0 | 0 | 8 | 1 | 0 | 0.1406 | 0.1406 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C1349` | 8 | 0 | 0 | 8 | 1 | 0 | 0.1191 | 0.1191 |
| qwen3 | `pair` | `zero` | `L27C2767+L30C5558` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0723 | 0.1035 |
| qwen3 | `pair` | `flip` | `L30C1349+L30C5558` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0391 | 0.1016 |
| qwen3 | `pair` | `flip` | `L27C2767+L29C4588` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0703 | 0.0938 |
| qwen3 | `pair` | `zero` | `L29C1532+L29C4588` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0723 | 0.0801 |
| qwen3 | `pair` | `flip` | `L30C2848+L29C4588` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0449 | 0.0801 |
| qwen3 | `pair` | `zero` | `L30C2848+L27C2767` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0293 | 0.0723 |
| qwen3 | `pair` | `zero` | `L30C1349+L30C5558` | 8 | 0 | 0 | 8 | 1 | 0 | -0.0039 | 0.0664 |
| qwen3 | `pair` | `flip` | `L29C4588+L30C5558` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0117 | 0.0664 |
| qwen3 | `pair` | `flip` | `L30C2848+L30C5558` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0312 | 0.0625 |
| qwen3 | `pair` | `zero` | `L29C1532+L30C2848` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0293 | 0.0605 |
| qwen3 | `pair` | `zero` | `L27C2767+L29C4588` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0176 | 0.0527 |
| qwen3 | `pair` | `flip` | `L29C1532+L30C2848` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0469 | 0.0469 |
| qwen3 | `pair` | `zero` | `L30C1349+L27C2767` | 8 | 0 | 0 | 8 | 2 | 0 | -0.0449 | 0.0449 |
| glm4 | `pair` | `flip` | `L28C8036+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0729 | 0.0847 |
| glm4 | `pair` | `flip` | `L26C6031+L28C8036` | 8 | 0 | 0 | 8 | 1 | 1 | -0.0327 | 0.0728 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | -0.0098 | 0.0664 |
| glm4 | `pair` | `zero` | `L26C6031+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | -0.0254 | 0.0596 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0110 | 0.0579 |
| glm4 | `pair` | `flip` | `L30C6115+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0003 | 0.0570 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L27C10905` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0049 | 0.0518 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | -0.0122 | 0.0513 |
| glm4 | `triplet` | `zero` | `L28C2777+L30C6115+L26C6031` | 8 | 0 | 0 | 8 | 0 | 0 | -0.0083 | 0.0503 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0166 | 0.0498 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L26C6031` | 8 | 0 | 0 | 8 | 1 | 1 | 0.0291 | 0.0476 |
| glm4 | `pair` | `zero` | `L30C6115+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | -0.0146 | 0.0420 |
| glm4 | `pair` | `zero` | `L28C8036+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0110 | 0.0417 |
| glm4 | `pair` | `flip` | `L28C8036+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0298 | 0.0376 |
| glm4 | `triplet` | `flip` | `L28C2777+L30C6115+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0104 | 0.0370 |
| glm4 | `pair` | `flip` | `L30C6115+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0193 | 0.0369 |
| glm4 | `pair` | `flip` | `L30C6115+L27C10905` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0251 | 0.0369 |
| glm4 | `pair` | `flip` | `L26C6031+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | -0.0054 | 0.0365 |
| glm4 | `pair` | `zero` | `L28C2777+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | -0.0107 | 0.0356 |
| glm4 | `pair` | `zero` | `L28C8036+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0244 | 0.0342 |
| glm4 | `pair` | `flip` | `L28C2777+L26C6031` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0317 | 0.0337 |
| glm4 | `pair` | `flip` | `L30C6115+L26C6031` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0103 | 0.0337 |
| glm4 | `pair` | `flip` | `L28C2777+L30C6115` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0110 | 0.0334 |
| glm4 | `pair` | `zero` | `L28C2777+L28C8036` | 8 | 0 | 0 | 8 | 0 | 1 | -0.0127 | 0.0332 |
| glm4 | `pair` | `zero` | `L28C2777+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0269 | 0.0327 |
| glm4 | `pair` | `zero` | `L30C6115+L26C6031` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0029 | 0.0322 |
| glm4 | `pair` | `zero` | `L30C6115+L29C10031` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0032 | 0.0320 |
| glm4 | `pair` | `zero` | `L30C6115+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0210 | 0.0308 |
| glm4 | `pair` | `flip` | `L26C6031+L27C10905` | 8 | 0 | 0 | 8 | 0 | 1 | 0.0171 | 0.0308 |
| glm4 | `pair` | `zero` | `L29C10031+L27C10905` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0208 | 0.0300 |
| deepseek7b | `pair` | `flip` | `L27C13360+L27C2295` | 8 | 1 | 0 | 7 | 2 | 0 | 0.0938 | 0.1172 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C2295` | 8 | 1 | 0 | 7 | 2 | 0 | 0.0664 | 0.0977 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C2295` | 8 | 1 | 0 | 7 | 2 | 0 | 0.0742 | 0.0859 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C13360` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0742 | 0.0977 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L25C4036` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0508 | 0.0898 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C2295` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0508 | 0.0820 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C13360` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0508 | 0.0742 |
| deepseek7b | `triplet` | `zero` | `L27C15791+L27C1106+L27C15305` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0312 | 0.0703 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C13360` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0488 | 0.0645 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L27C15305` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0254 | 0.0645 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C2295` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0527 | 0.0605 |
| deepseek7b | `pair` | `zero` | `L27C15305+L25C4036` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0273 | 0.0586 |
| deepseek7b | `pair` | `flip` | `L27C15305+L27C2295` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0371 | 0.0566 |
| deepseek7b | `pair` | `zero` | `L27C15791+L25C4036` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0234 | 0.0547 |
| deepseek7b | `pair` | `zero` | `L27C1106+L25C4036` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0312 | 0.0547 |
| deepseek7b | `pair` | `flip` | `L27C1106+L27C13360` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0371 | 0.0527 |
| deepseek7b | `triplet` | `flip` | `L27C15791+L27C1106+L25C4036` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0430 | 0.0508 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C1106` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0254 | 0.0488 |
| deepseek7b | `pair` | `zero` | `L27C13360+L27C2295` | 8 | 0 | 0 | 8 | 1 | 0 | 0.0332 | 0.0488 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C1106` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0312 | 0.0469 |
| deepseek7b | `pair` | `flip` | `L27C1106+L27C15305` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0195 | 0.0469 |
| deepseek7b | `pair` | `zero` | `L27C15305+L27C13360` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0312 | 0.0469 |
| deepseek7b | `pair` | `zero` | `L25C4036+L27C13360` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0312 | 0.0469 |
| deepseek7b | `pair` | `flip` | `L27C15305+L27C13360` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0059 | 0.0449 |
| deepseek7b | `pair` | `flip` | `L27C15305+L25C4036` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0039 | 0.0430 |
| deepseek7b | `pair` | `zero` | `L25C4036+L27C2295` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0195 | 0.0430 |
| deepseek7b | `pair` | `flip` | `L27C15791+L27C15305` | 8 | 0 | 0 | 8 | 3 | 0 | 0.0020 | 0.0410 |
| deepseek7b | `pair` | `flip` | `L27C1106+L25C4036` | 8 | 0 | 0 | 8 | 0 | 0 | 0.0312 | 0.0391 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C13360` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0293 | 0.0371 |
| deepseek7b | `pair` | `zero` | `L27C15791+L27C15305` | 8 | 0 | 0 | 8 | 2 | 0 | 0.0273 | 0.0352 |
