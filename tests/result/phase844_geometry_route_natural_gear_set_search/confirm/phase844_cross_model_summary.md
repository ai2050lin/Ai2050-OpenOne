# Phase 844 Geometry Route Natural Gear Set Search (confirm)

- Search: natural MLP down-input channel activation x readout-coupling over geometry cases.
- Boundary: gear-set atlas probe; not global closure.

## Model Summary

| model | gears | rows | cases | original target | target | lost | gained |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 12 | 195 | 5 | 2 | 62 | 0 | 36 |
| glm4 | 12 | 195 | 5 | 8 | 104 | 6 | 6 |
| deepseek7b | 12 | 195 | 5 | 6 | 87 | 11 | 20 |

## Top Gears

| model | rank | layer | channel | hits | mean act | neg ratio | mean abs support | gear score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 29 | 1532 | 15 | 17.9667 | 0.0000 | 5.7955 | 16.0686 |
| qwen3 | 2 | 30 | 2848 | 3 | -10.6875 | 1.0000 | 5.7817 | 8.0151 |
| qwen3 | 3 | 30 | 1349 | 13 | -6.6629 | 0.9231 | 2.5029 | 6.3512 |
| qwen3 | 4 | 27 | 2767 | 14 | -17.4464 | 1.0000 | 1.6529 | 4.4762 |
| qwen3 | 5 | 29 | 4588 | 3 | -6.9271 | 1.0000 | 2.6917 | 3.7315 |
| qwen3 | 6 | 30 | 5558 | 3 | -5.1029 | 1.0000 | 2.6765 | 3.7105 |
| qwen3 | 7 | 27 | 1561 | 15 | -6.8682 | 1.0000 | 1.2672 | 3.5135 |
| qwen3 | 8 | 30 | 2121 | 3 | -4.1771 | 1.0000 | 1.8069 | 2.5049 |
| qwen3 | 9 | 30 | 8818 | 3 | 6.5990 | 0.0000 | 1.7893 | 2.4805 |
| qwen3 | 10 | 30 | 3376 | 3 | -8.5208 | 1.0000 | 1.1319 | 1.5692 |
| qwen3 | 11 | 27 | 7219 | 3 | 4.4740 | 0.0000 | 1.1076 | 1.5354 |
| qwen3 | 12 | 28 | 4231 | 14 | -3.2578 | 1.0000 | 0.5338 | 1.4454 |
| glm4 | 1 | 28 | 2777 | 3 | 3.0599 | 0.0000 | 0.5114 | 0.7090 |
| glm4 | 2 | 30 | 6115 | 15 | 4.3490 | 0.0000 | 0.2442 | 0.6769 |
| glm4 | 3 | 26 | 6031 | 3 | 3.0495 | 0.0000 | 0.4583 | 0.6353 |
| glm4 | 4 | 28 | 8036 | 15 | 2.7167 | 0.1333 | 0.2407 | 0.6229 |
| glm4 | 5 | 29 | 10031 | 14 | 4.1777 | 0.0000 | 0.2141 | 0.5797 |
| glm4 | 6 | 27 | 10905 | 3 | -2.1484 | 1.0000 | 0.3917 | 0.5430 |
| glm4 | 7 | 27 | 7041 | 12 | 3.9455 | 0.0000 | 0.1825 | 0.4681 |
| glm4 | 8 | 29 | 8345 | 3 | 3.3958 | 0.0000 | 0.3035 | 0.4208 |
| glm4 | 9 | 30 | 10283 | 14 | 6.9509 | 0.0000 | 0.1514 | 0.4100 |
| glm4 | 10 | 25 | 7711 | 15 | -2.3385 | 1.0000 | 0.1360 | 0.3770 |
| glm4 | 11 | 29 | 5532 | 15 | -2.7620 | 1.0000 | 0.1347 | 0.3733 |
| glm4 | 12 | 30 | 3411 | 11 | 1.5874 | 0.0000 | 0.1143 | 0.2840 |
| deepseek7b | 1 | 27 | 15791 | 15 | -128.3000 | 1.0000 | 8.0055 | 22.1959 |
| deepseek7b | 2 | 27 | 1106 | 15 | -54.8000 | 1.0000 | 7.9345 | 21.9991 |
| deepseek7b | 3 | 27 | 15305 | 15 | -91.6167 | 1.0000 | 6.8460 | 18.9811 |
| deepseek7b | 4 | 25 | 4036 | 3 | 19.1667 | 0.0000 | 11.7715 | 16.3187 |
| deepseek7b | 5 | 27 | 13360 | 15 | 83.0667 | 0.0000 | 5.5830 | 15.4794 |
| deepseek7b | 6 | 27 | 2295 | 15 | -45.7667 | 1.0000 | 5.3412 | 14.8089 |
| deepseek7b | 7 | 24 | 3099 | 15 | 40.6500 | 0.0000 | 5.3367 | 14.7965 |
| deepseek7b | 8 | 24 | 77 | 15 | 40.6167 | 0.0000 | 5.2969 | 14.6862 |
| deepseek7b | 9 | 25 | 11187 | 15 | -49.4667 | 1.0000 | 5.1892 | 14.3876 |
| deepseek7b | 10 | 27 | 2699 | 2 | 20.0938 | 0.0000 | 10.8701 | 11.9420 |
| deepseek7b | 11 | 26 | 17524 | 3 | -23.9375 | 1.0000 | 8.0222 | 11.1211 |
| deepseek7b | 12 | 27 | 18866 | 15 | 86.8333 | 0.0000 | 3.9783 | 11.0303 |

## Subset Summary

| model | subset | n | target | lost | gained | object_echo | unknown | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `original` | 15 | 2 | 0 | 0 | 1 | 11 | 2.0271 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 2, "unknown_other": 11}` |
| qwen3 | `top12_flip` | 15 | 6 | 0 | 4 | 1 | 7 | -0.0141 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 6, "unknown_other": 7}` |
| qwen3 | `top12_half` | 15 | 5 | 0 | 3 | 1 | 8 | 1.5708 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top12_zero` | 15 | 5 | 0 | 3 | 1 | 8 | 1.1615 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top1_flip` | 15 | 5 | 0 | 3 | 2 | 8 | 0.2396 | `{"object_echo": 2, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top1_half` | 15 | 4 | 0 | 2 | 1 | 9 | 1.3990 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 4, "unknown_other": 9}` |
| qwen3 | `top1_zero` | 15 | 4 | 0 | 2 | 1 | 9 | 0.7833 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 4, "unknown_other": 9}` |
| qwen3 | `top4_flip` | 15 | 5 | 0 | 3 | 2 | 7 | -1.2875 | `{"broad_near_miss": 1, "object_echo": 2, "target_equivalent": 5, "unknown_other": 7}` |
| qwen3 | `top4_half` | 15 | 5 | 0 | 3 | 1 | 8 | 1.2771 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top4_zero` | 15 | 5 | 0 | 3 | 1 | 8 | 0.5198 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top8_flip` | 15 | 6 | 0 | 4 | 1 | 7 | -0.4448 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 6, "unknown_other": 7}` |
| qwen3 | `top8_half` | 15 | 5 | 0 | 3 | 1 | 8 | 1.5042 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| qwen3 | `top8_zero` | 15 | 5 | 0 | 3 | 1 | 8 | 1.0427 | `{"broad_near_miss": 1, "object_echo": 1, "target_equivalent": 5, "unknown_other": 8}` |
| glm4 | `original` | 15 | 8 | 0 | 0 | 1 | 6 | 0.9255 | `{"object_echo": 1, "target_equivalent": 8, "unknown_other": 6}` |
| glm4 | `top12_flip` | 15 | 9 | 0 | 1 | 1 | 5 | 1.4297 | `{"object_echo": 1, "target_equivalent": 9, "unknown_other": 5}` |
| glm4 | `top12_half` | 15 | 9 | 0 | 1 | 1 | 5 | 1.0943 | `{"object_echo": 1, "target_equivalent": 9, "unknown_other": 5}` |
| glm4 | `top12_zero` | 15 | 9 | 0 | 1 | 1 | 5 | 1.2293 | `{"object_echo": 1, "target_equivalent": 9, "unknown_other": 5}` |
| glm4 | `top1_flip` | 15 | 7 | 1 | 0 | 1 | 7 | 0.9404 | `{"object_echo": 1, "target_equivalent": 7, "unknown_other": 7}` |
| glm4 | `top1_half` | 15 | 7 | 1 | 0 | 1 | 7 | 0.9440 | `{"object_echo": 1, "target_equivalent": 7, "unknown_other": 7}` |
| glm4 | `top1_zero` | 15 | 7 | 1 | 0 | 1 | 7 | 0.9437 | `{"object_echo": 1, "target_equivalent": 7, "unknown_other": 7}` |
| glm4 | `top4_flip` | 15 | 7 | 1 | 0 | 1 | 7 | 0.8883 | `{"object_echo": 1, "target_equivalent": 7, "unknown_other": 7}` |
| glm4 | `top4_half` | 15 | 8 | 0 | 0 | 1 | 6 | 0.9302 | `{"object_echo": 1, "target_equivalent": 8, "unknown_other": 6}` |
| glm4 | `top4_zero` | 15 | 7 | 1 | 0 | 1 | 7 | 0.9115 | `{"object_echo": 1, "target_equivalent": 7, "unknown_other": 7}` |
| glm4 | `top8_flip` | 15 | 8 | 1 | 1 | 1 | 6 | 1.6795 | `{"object_echo": 1, "target_equivalent": 8, "unknown_other": 6}` |
| glm4 | `top8_half` | 15 | 9 | 0 | 1 | 1 | 5 | 1.1083 | `{"object_echo": 1, "target_equivalent": 9, "unknown_other": 5}` |
| glm4 | `top8_zero` | 15 | 9 | 0 | 1 | 1 | 5 | 1.2957 | `{"object_echo": 1, "target_equivalent": 9, "unknown_other": 5}` |
| deepseek7b | `original` | 15 | 6 | 0 | 0 | 2 | 7 | -0.8812 | `{"object_echo": 2, "target_equivalent": 6, "unknown_other": 7}` |
| deepseek7b | `top12_flip` | 15 | 4 | 3 | 1 | 4 | 7 | -1.2612 | `{"object_echo": 4, "target_equivalent": 4, "unknown_other": 7}` |
| deepseek7b | `top12_half` | 15 | 6 | 1 | 1 | 3 | 6 | -1.0531 | `{"object_echo": 3, "target_equivalent": 6, "unknown_other": 6}` |
| deepseek7b | `top12_zero` | 15 | 5 | 2 | 1 | 3 | 7 | -1.1583 | `{"object_echo": 3, "target_equivalent": 5, "unknown_other": 7}` |
| deepseek7b | `top1_flip` | 15 | 10 | 0 | 4 | 2 | 3 | -0.8594 | `{"object_echo": 2, "target_equivalent": 10, "unknown_other": 3}` |
| deepseek7b | `top1_half` | 15 | 7 | 0 | 1 | 2 | 6 | -0.8823 | `{"object_echo": 2, "target_equivalent": 7, "unknown_other": 6}` |
| deepseek7b | `top1_zero` | 15 | 7 | 0 | 1 | 2 | 6 | -0.8760 | `{"object_echo": 2, "target_equivalent": 7, "unknown_other": 6}` |
| deepseek7b | `top4_flip` | 15 | 9 | 0 | 3 | 3 | 3 | -0.8844 | `{"object_echo": 3, "target_equivalent": 9, "unknown_other": 3}` |
| deepseek7b | `top4_half` | 15 | 7 | 0 | 1 | 2 | 6 | -0.8938 | `{"object_echo": 2, "target_equivalent": 7, "unknown_other": 6}` |
| deepseek7b | `top4_zero` | 15 | 10 | 0 | 4 | 2 | 3 | -0.9010 | `{"object_echo": 2, "target_equivalent": 10, "unknown_other": 3}` |
| deepseek7b | `top8_flip` | 15 | 4 | 3 | 1 | 4 | 7 | -0.5771 | `{"object_echo": 4, "target_equivalent": 4, "unknown_other": 7}` |
| deepseek7b | `top8_half` | 15 | 6 | 1 | 1 | 2 | 7 | -0.8521 | `{"object_echo": 2, "target_equivalent": 6, "unknown_other": 7}` |
| deepseek7b | `top8_zero` | 15 | 6 | 1 | 1 | 3 | 6 | -0.7625 | `{"object_echo": 3, "target_equivalent": 6, "unknown_other": 6}` |

## Object Summary

| model | object | n | target | lost | gained | object_echo | unknown | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `circle` | 39 | 15 | 0 | 2 | 2 | 22 | `{"object_echo": 2, "target_equivalent": 15, "unknown_other": 22}` |
| qwen3 | `polygon` | 39 | 12 | 0 | 12 | 0 | 27 | `{"target_equivalent": 12, "unknown_other": 27}` |
| qwen3 | `rectangle` | 39 | 9 | 0 | 9 | 0 | 30 | `{"target_equivalent": 9, "unknown_other": 30}` |
| qwen3 | `square` | 39 | 13 | 0 | 0 | 13 | 13 | `{"object_echo": 13, "target_equivalent": 13, "unknown_other": 13}` |
| qwen3 | `triangle` | 39 | 13 | 0 | 13 | 0 | 14 | `{"broad_near_miss": 12, "target_equivalent": 13, "unknown_other": 14}` |
| glm4 | `circle` | 39 | 26 | 0 | 0 | 0 | 13 | `{"target_equivalent": 26, "unknown_other": 13}` |
| glm4 | `polygon` | 39 | 13 | 0 | 0 | 0 | 26 | `{"target_equivalent": 13, "unknown_other": 26}` |
| glm4 | `rectangle` | 39 | 14 | 5 | 6 | 0 | 25 | `{"target_equivalent": 14, "unknown_other": 25}` |
| glm4 | `square` | 39 | 26 | 0 | 0 | 13 | 0 | `{"object_echo": 13, "target_equivalent": 26}` |
| glm4 | `triangle` | 39 | 25 | 1 | 0 | 0 | 14 | `{"target_equivalent": 25, "unknown_other": 14}` |
| deepseek7b | `circle` | 39 | 11 | 6 | 4 | 1 | 27 | `{"object_echo": 1, "target_equivalent": 11, "unknown_other": 27}` |
| deepseek7b | `polygon` | 39 | 29 | 0 | 3 | 0 | 10 | `{"target_equivalent": 29, "unknown_other": 10}` |
| deepseek7b | `rectangle` | 39 | 11 | 2 | 0 | 13 | 15 | `{"object_echo": 13, "target_equivalent": 11, "unknown_other": 15}` |
| deepseek7b | `square` | 39 | 15 | 3 | 5 | 20 | 4 | `{"object_echo": 20, "target_equivalent": 15, "unknown_other": 4}` |
| deepseek7b | `triangle` | 39 | 21 | 0 | 8 | 0 | 18 | `{"target_equivalent": 21, "unknown_other": 18}` |

## Top Rows

| model | object | prompt | subset | mode | gears | class | output | orig class | lost | gained | target-object | top tokens |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---|
| qwen3 | `circle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | -5.3750 | `[" Shapes", " Geometry", " ", " Circle", " Math"]` |
| qwen3 | `rectangle` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.9688 | `[" shape", " Shape", " quadr", " shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.9688 | `[" shape", " Shape", " quadr", " shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.7812 | `[" shape", " Shape", " quadr", " ", " shapes"]` |
| qwen3 | `rectangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.1562 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.0938 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 3.7500 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `triangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 2.5625 | `[" polygon", " triangle", " ", " type", " Polygon"]` |
| qwen3 | `triangle` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.4375 | `[" shape", " polygon", " Shape", " Shapes", " "]` |
| qwen3 | `triangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.3750 | `[" shape", " polygon", " Shape", " Shapes", " "]` |
| qwen3 | `triangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 2.1875 | `[" polygon", " ", " triangle", " shape", " Polygon"]` |
| qwen3 | `triangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.1875 | `[" shape", " polygon", " Shape", " ", " Shapes"]` |
| qwen3 | `triangle` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 2.1875 | `[" polygon", " shape", " ", " Shape", " Shapes"]` |
| qwen3 | `polygon` | `natural_category` | `top1_half` | `half` | 1 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | 2.1875 | `[" Shapes", " shapes", " ", " geometric", " type"]` |
| qwen3 | `polygon` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 2.1875 | `[" shapes", " Shapes", " geometric", " shape", " "]` |
| qwen3 | `triangle` | `natural_category` | `top1_half` | `half` | 1 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 2.1250 | `[" polygon", " ", " shape", " triangle", " Shape"]` |
| qwen3 | `triangle` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.1250 | `[" shape", " polygon", " ", " Shape", " Shapes"]` |
| qwen3 | `polygon` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 2.1250 | `[" shapes", " Shapes", " ", " geometric", " shape"]` |
| qwen3 | `triangle` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 2.0625 | `[" polygon", " shape", " ", " Shape", " Shapes"]` |
| qwen3 | `polygon` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | 2.0625 | `[" Shapes", " shapes", " ", " geometry", " shape"]` |
| qwen3 | `circle` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | -2.0000 | `[" Shapes", " Geometry", " ", " Shape", " Math"]` |
| qwen3 | `rectangle` | `natural_category` | `top12_flip` | `flip` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.9688 | `[" shape", " Shape", " shapes", " Shapes", " __"]` |
| qwen3 | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.9062 | `[" shape", " Shape", " shapes", " Shapes", " __"]` |
| qwen3 | `triangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.5625 | `[" shape", " Shape", " polygon", " Shapes", " "]` |
| qwen3 | `triangle` | `natural_category` | `top12_flip` | `flip` | 12 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.3750 | `[" shape", " Shape", " Shapes", " polygon", " "]` |
| qwen3 | `triangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.3125 | `[" shape", " Shape", " polygon", " Shapes", " "]` |
| qwen3 | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Polygon | `broad_near_miss` | 0 | 1 | 1.1875 | `[" Polygon", " Triangle", " Geometry", " Poly", " Tri"]` |
| qwen3 | `polygon` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 1.1250 | `[" shapes", " Shapes", " shape", " type", " "]` |
| qwen3 | `polygon` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 1.1250 | `[" shapes", " Shapes", " shape", " type", " "]` |
| qwen3 | `polygon` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 1.0625 | `[" shapes", " Shapes", " type", " ", " types"]` |
| qwen3 | `rectangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.0000 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `polygon` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shapes | `unknown_other` | 0 | 1 | 0.7500 | `[" shapes", " Shapes", " shape", " ", " type"]` |
| qwen3 | `polygon` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Polygons | `unknown_other` | 0 | 1 | 0.0000 | `[" Poly", " Polygon", " Shapes", " ", " Geometry"]` |
| qwen3 | `polygon` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.0000 | `[" Polygon", " Shapes", " Poly", " Mathematics", " Geometry"]` |
| qwen3 | `polygon` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | 0.0000 | `[" Shapes", " Polygon", " Math", " Geometry", " Mathematics"]` |
| qwen3 | `polygon` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Shapes | `unknown_other` | 0 | 1 | 0.0000 | `[" Shapes", " Polygon", " Math", " Geometry", " Mathematics"]` |
| qwen3 | `circle` | `natural_question` | `top4_flip` | `flip` | 4 | `object_echo` | Circle | `unknown_other` | 0 | 0 | -8.1875 | `[" Circle", " Shapes", " Circ", " ", " circle"]` |
| qwen3 | `square` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 8.0938 | `[" shape", " Shape", " Shapes", " shapes", " quadr"]` |
| qwen3 | `square` | `natural_category` | `top1_half` | `half` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.2969 | `[" shape", " Shape", " Shapes", " shapes", " quadr"]` |
| qwen3 | `square` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.2188 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.1562 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.0000 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 6.2188 | `[" shape", " Shape", " quadr", " Shapes", " shapes"]` |
| qwen3 | `square` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 6.2188 | `[" shape", " quadr", " Shape", " Shapes", " shapes"]` |
| qwen3 | `square` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 6.0781 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 6.0156 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `circle` | `object_only` | `top1_flip` | `flip` | 1 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -5.8125 | `[" \"", " A", " \n\n", " Circle", " \n"]` |
| qwen3 | `circle` | `object_only` | `top4_flip` | `flip` | 4 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -5.7812 | `[" \"", " A", " \n\n", " Circle", " \n"]` |
| qwen3 | `square` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 5.6719 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.5938 | `[" Geometry", " Quadr", " Ge", " Shapes", " Polygon"]` |
| qwen3 | `rectangle` | `natural_category` | `original` | `original` | 0 | `unknown_other` | quadrilateral | `unknown_other` | 0 | 0 | 5.4688 | `[" quadr", " shape", " ", " Shape", " Shapes"]` |
| qwen3 | `circle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | -5.4375 | `[" Shape", " Shapes", " shape", " shapes", " "]` |
| qwen3 | `square` | `natural_question` | `top8_zero` | `zero` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.3125 | `[" Geometry", " Shapes", " Polygon", " Shape", " Ge"]` |
| qwen3 | `square` | `natural_question` | `top12_zero` | `zero` | 12 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.3125 | `[" Geometry", " Shapes", " Polygon", " Shape", " Ge"]` |
| qwen3 | `square` | `natural_question` | `top12_half` | `half` | 12 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.2812 | `[" Geometry", " Shapes", " Polygon", " Ge", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top8_half` | `half` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.2188 | `[" Geometry", " Shapes", " Polygon", " Ge", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 5.1875 | `[" Quadr", " Polygon", " quadr", " Shapes", " Poly"]` |
| qwen3 | `circle` | `object_only` | `top1_zero` | `zero` | 1 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -5.0938 | `[" \"", " A", " \n\n", " Circle", " a"]` |
| qwen3 | `circle` | `object_only` | `top4_zero` | `zero` | 4 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -5.0625 | `[" \"", " A", " \n\n", " Circle", " a"]` |
| qwen3 | `square` | `natural_question` | `top1_half` | `half` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.0000 | `[" Geometry", " Quadr", " Polygon", " Shapes", " Ge"]` |
| qwen3 | `circle` | `natural_question` | `top1_flip` | `flip` | 1 | `object_echo` | Circle | `unknown_other` | 0 | 0 | -5.0000 | `[" Circle", " Shapes", " ", " Circ", " Geometry"]` |
| qwen3 | `square` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 4.9375 | `[" Quadr", " Geometry", " Polygon", " quadr", " Shapes"]` |
| qwen3 | `circle` | `object_only` | `top1_half` | `half` | 1 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.8438 | `[" \"", " A", " \n\n", " Circle", " \n"]` |
| qwen3 | `circle` | `object_only` | `top4_half` | `half` | 4 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.8438 | `[" \"", " A", " \n\n", " Circle", " a"]` |
| qwen3 | `circle` | `object_only` | `top8_flip` | `flip` | 8 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.8125 | `[" \"", " \n\n", " A", " \n", " a"]` |
| qwen3 | `rectangle` | `natural_category` | `top1_half` | `half` | 1 | `unknown_other` | quadrilateral | `unknown_other` | 0 | 0 | 4.6562 | `[" quadr", " shape", " ", " Shape", " Shapes"]` |
| qwen3 | `circle` | `object_only` | `top8_zero` | `zero` | 8 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.6562 | `[" \"", " A", " \n\n", " \n", " a"]` |
| qwen3 | `circle` | `object_only` | `top8_half` | `half` | 8 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.6250 | `[" \"", " A", " \n\n", " \n", " a"]` |
| qwen3 | `circle` | `object_only` | `top12_flip` | `flip` | 12 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.6250 | `[" \"", " \n\n", " A", " \n", " a"]` |
| qwen3 | `circle` | `object_only` | `original` | `original` | 0 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.5625 | `[" \"", " A", " \n\n", " Circle", " a"]` |
| qwen3 | `square` | `natural_question` | `top4_half` | `half` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 4.5000 | `[" Geometry", " Shapes", " Polygon", " Ge", " Quadr"]` |
| qwen3 | `circle` | `object_only` | `top12_half` | `half` | 12 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.5000 | `[" \"", " A", " \n\n", " Circle", " \n"]` |
| qwen3 | `circle` | `object_only` | `top12_zero` | `zero` | 12 | `unknown_other` | A closed curve with all | `unknown_other` | 0 | 0 | -4.4062 | `[" \"", " A", " \n\n", " \n", " a"]` |
| qwen3 | `rectangle` | `natural_category` | `top1_flip` | `flip` | 1 | `unknown_other` | quadrilateral | `unknown_other` | 0 | 0 | 4.0000 | `[" quadr", " ", " shape", " Quadr", " four"]` |
| qwen3 | `rectangle` | `natural_category` | `top1_zero` | `zero` | 1 | `unknown_other` | quadrilateral | `unknown_other` | 0 | 0 | 3.8750 | `[" quadr", " shape", " ", " Shape", " shapes"]` |
| qwen3 | `square` | `natural_question` | `top8_flip` | `flip` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.5625 | `[" Geometry", " Shapes", " geometry", " ", " \n"]` |
| qwen3 | `circle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | -3.5625 | `[" Shape", " Shapes", " shapes", " shape", " "]` |
| qwen3 | `square` | `natural_question` | `top12_flip` | `flip` | 12 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.5312 | `[" Geometry", " geometry", " Shapes", " ", " \n"]` |
| qwen3 | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.5000 | `[" Geometry", " Shapes", " Shape", " Polygon", " Ge"]` |
| qwen3 | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.2500 | `[" Geometry", " Quadr", " Shapes", " ", " Ge"]` |
| glm4 | `triangle` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | geometric figure | `target_equivalent` | 1 | 0 | 3.2676 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `rectangle` | `natural_category` | `top1_half` | `half` | 1 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.3516 | `[" quadr", " geometric", " polygon", " shape", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top1_zero` | `zero` | 1 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.3320 | `[" quadr", " geometric", " polygon", " shape", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top1_flip` | `flip` | 1 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.2773 | `[" quadr", " geometric", " shape", " polygon", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top4_zero` | `zero` | 4 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.2148 | `[" quadr", " geometric", " polygon", " shape", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top4_flip` | `flip` | 4 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.0742 | `[" quadr", " geometric", " paralle", " shape", " polygon"]` |
| glm4 | `rectangle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `unknown_other` | 0 | 1 | 3.8525 | `[" Shape", " Ge", " ", " geometric", " shape"]` |
| glm4 | `rectangle` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Shape | `unknown_other` | 0 | 1 | 3.8418 | `[" Shape", " Two", " Ge", " ", " geometric"]` |
| glm4 | `rectangle` | `natural_question` | `top12_zero` | `zero` | 12 | `target_equivalent` | Shape | `unknown_other` | 0 | 1 | 3.1035 | `[" Shape", " Ge", " Two", " ", " Quadr"]` |
| glm4 | `rectangle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Shape | `unknown_other` | 0 | 1 | 2.9492 | `[" Shape", " Ge", " Quadr", " Two", " "]` |
| glm4 | `rectangle` | `natural_question` | `top12_half` | `half` | 12 | `target_equivalent` | Geometric Shape | `unknown_other` | 0 | 1 | 2.6328 | `[" Ge", " Shape", " Quadr", " Two", " "]` |
| glm4 | `rectangle` | `natural_question` | `top8_half` | `half` | 8 | `target_equivalent` | Geometric Shape | `unknown_other` | 0 | 1 | 2.6055 | `[" Ge", " Quadr", " Shape", " Two", " "]` |
| glm4 | `triangle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.8828 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `triangle` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.5508 | `[" Shape", " Ge", " Mathematical", " Two", " shape"]` |
| glm4 | `square` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.4712 | `[" Shape", " shape", " Ge", " geometric", " "]` |
| glm4 | `square` | `natural_question` | `top12_zero` | `zero` | 12 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.0664 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `polygon` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 3.0625 | `[" Ge", " Shape", " Geometry", " Mathematical", " geometric"]` |
| glm4 | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 3.0176 | `[" geometric", " shape", " quadr", " polygon", " paralle"]` |
| glm4 | `square` | `natural_question` | `top12_half` | `half` | 12 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.9805 | `[" Ge", " Shape", " geometric", " shape", " "]` |
| glm4 | `triangle` | `natural_category` | `top12_flip` | `flip` | 12 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 2.9375 | `[" shape", " geometric", " polygon", " Shape", " polygons"]` |
| glm4 | `square` | `natural_category` | `top12_flip` | `flip` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.9062 | `[" geometric", " shape", " __", " Shape", " ("]` |
| glm4 | `square` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.8320 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.8203 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_question` | `original` | `original` | 0 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.8164 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.8164 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_question` | `top1_half` | `half` | 1 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.8125 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `rectangle` | `natural_category` | `top12_flip` | `flip` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.7617 | `[" geometric", " shape", " Shape", " __", " "]` |
| glm4 | `square` | `natural_question` | `top8_half` | `half` | 8 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.7578 | `[" Ge", " Shape", " geometric", " shape", " A"]` |
| glm4 | `square` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.7500 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `square` | `natural_question` | `top4_half` | `half` | 4 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.6875 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `triangle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.6797 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `square` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.6284 | `[" geometric", " shape", " __", " polygon", " Shape"]` |
| glm4 | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.6250 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `polygon` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.6250 | `[" Ge", " Shape", " Mathematical", " Geometry", " Math"]` |
| glm4 | `triangle` | `natural_question` | `top12_zero` | `zero` | 12 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.5859 | `[" Shape", " Ge", " geometric", " shape", " Geometry"]` |
| glm4 | `square` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.4961 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.4609 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `square` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.4180 | `[" geometric", " shape", " Shape", " __", " polygon"]` |
| glm4 | `rectangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.4004 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `triangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.3984 | `[" polygon", " geometric", " shape", " quadr", " polygons"]` |
| glm4 | `rectangle` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3594 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3398 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3320 | `[" geometric", " quadr", " polygon", " shape", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3281 | `[" geometric", " quadr", " shape", " polygon", " __"]` |
| glm4 | `square` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3203 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `rectangle` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3203 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `polygon` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.3203 | `[" Ge", " Shape", " Plane", " Geometry", " Two"]` |
| glm4 | `square` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2969 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `square` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2852 | `[" geometric", " shape", " polygon", " quadr", " __"]` |
| glm4 | `square` | `natural_category` | `top1_half` | `half` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2812 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `square` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2773 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `square` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2695 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `triangle` | `natural_category` | `top12_zero` | `zero` | 12 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.2578 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `polygon` | `natural_question` | `top12_zero` | `zero` | 12 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.2500 | `[" Ge", " Shape", " Plane", " Mathematical", " Geometry"]` |
| glm4 | `polygon` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | geometric figure | `unknown_other` | 0 | 0 | 2.2461 | `[" geometric", " shape", " geometry", " Shape", " "]` |
| glm4 | `square` | `natural_category` | `top4_half` | `half` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2422 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `rectangle` | `natural_question` | `top1_half` | `half` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.2422 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `triangle` | `natural_category` | `top8_half` | `half` | 8 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.2344 | `[" polygon", " geometric", " shape", " quadr", " polygons"]` |
| glm4 | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.2266 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `rectangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.2109 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `rectangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.1797 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `square` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.1758 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `square` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.1562 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `rectangle` | `natural_question` | `top4_half` | `half` | 4 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.1484 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `triangle` | `natural_category` | `top12_half` | `half` | 12 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.1328 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `rectangle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.1016 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `polygon` | `natural_category` | `top12_flip` | `flip` | 12 | `unknown_other` | geometric figure | `unknown_other` | 0 | 0 | 2.0898 | `[" geometric", " shape", " geometry", " ", " Shape"]` |
| glm4 | `triangle` | `natural_category` | `top1_half` | `half` | 1 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.0391 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `triangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.0234 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `triangle` | `natural_question` | `top8_half` | `half` | 8 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.0156 | `[" Ge", " Shape", " geometric", " shape", " Geometry"]` |
| glm4 | `triangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.0156 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `triangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.0156 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `polygon` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.0156 | `[" Ge", " Shape", " Plane", " Two", " Quadr"]` |
| glm4 | `triangle` | `object_only` | `top12_flip` | `flip` | 12 | `unknown_other` | A triangle is a polygon | `unknown_other` | 0 | 0 | -1.9805 | `[" \"", " A", " a", " three", " The"]` |
| glm4 | `triangle` | `object_only` | `top4_flip` | `flip` | 4 | `unknown_other` | A triangle is a polygon | `unknown_other` | 0 | 0 | -1.9766 | `[" \"", " A", " three", " a", " equ"]` |
| glm4 | `triangle` | `natural_question` | `top12_half` | `half` | 12 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 1.9688 | `[" Ge", " Shape", " geometric", " shape", " Geometry"]` |
| glm4 | `triangle` | `object_only` | `top4_zero` | `zero` | 4 | `unknown_other` | A triangle is a polygon | `unknown_other` | 0 | 0 | -1.9570 | `[" \"", " A", " three", " a", " equ"]` |
| glm4 | `triangle` | `object_only` | `top4_half` | `half` | 4 | `unknown_other` | A triangle is a polygon | `unknown_other` | 0 | 0 | -1.9492 | `[" \"", " A", " three", " a", " equ"]` |
| glm4 | `polygon` | `natural_question` | `top12_half` | `half` | 12 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 1.9375 | `[" Ge", " Shape", " Plane", " Geometry", " Two"]` |
| glm4 | `rectangle` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 1.9219 | `[" Quadr", " Ge", " Shape", " quadr", " Two"]` |
| deepseek7b | `circle` | `natural_category` | `top12_zero` | `zero` | 12 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.8750 | `[" ?\n", " ?\n\n", " __", " Geometry", " shape"]` |
| deepseek7b | `circle` | `natural_category` | `top12_flip` | `flip` | 12 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.7852 | `[" ?\n", " __", " circle", " Geometry", " ?\n\n"]` |
| deepseek7b | `circle` | `natural_category` | `top8_zero` | `zero` | 8 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.4688 | `[" ?\n", " ?\n\n", " __", " shape", " ["]` |
| deepseek7b | `circle` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.3750 | `[" ?\n", " __", " ?\n\n", " circle", " ["]` |
| deepseek7b | `circle` | `natural_category` | `top12_half` | `half` | 12 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.2500 | `[" ?\n", " shape", " ?\n\n", " geometry", " __"]` |
| deepseek7b | `circle` | `natural_category` | `top8_half` | `half` | 8 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -2.0469 | `[" ?\n", " shape", " ?\n\n", " __", " ["]` |
| deepseek7b | `rectangle` | `natural_category` | `top12_flip` | `flip` | 12 | `unknown_other` | __________ | `target_equivalent` | 1 | 0 | 1.0000 | `[" __", " geometry", " quadr", " Geometry", " ?\n"]` |
| deepseek7b | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | __________ | `target_equivalent` | 1 | 0 | 0.9219 | `[" __", " ?\n", " geometry", " polygon", " quadr"]` |
| deepseek7b | `square` | `natural_category` | `top12_flip` | `flip` | 12 | `object_echo` | square | `target_equivalent` | 1 | 0 | -0.8906 | `[" square", " geometry", " Geometry", " ?\n", " polygon"]` |
| deepseek7b | `square` | `natural_category` | `top12_zero` | `zero` | 12 | `unknown_other` | geometry | `target_equivalent` | 1 | 0 | 0.3438 | `[" geometry", " shape", " ?\n", " Geometry", " ?\n\n"]` |
| deepseek7b | `square` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | geometry | `target_equivalent` | 1 | 0 | 0.0781 | `[" geometry", " ?\n", " polygon", " __", " square"]` |
| deepseek7b | `circle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -2.5000 | `[" shape", " ", " A", " circle", " ["]` |
| deepseek7b | `circle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -2.5000 | `[" shape", " ", " Circle", " A", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -2.2812 | `[" shape", " ", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -2.2812 | `[" shape", " ", " A", " [", " circle"]` |
| deepseek7b | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0938 | `[" Polygon", " polygon", " ", " geometry", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `top12_half` | `half` | 12 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0938 | `[" Polygon", " geometry", " polygon", " ", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0000 | `[" Polygon", " ", " geometry", " [", " polygon"]` |
| deepseek7b | `triangle` | `natural_question` | `top12_zero` | `zero` | 12 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.9375 | `[" Polygon", " polygon", " geometry", " triangle", " "]` |
| deepseek7b | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.8125 | `[" Polygon", " polygon", " geometry", " ", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `top12_flip` | `flip` | 12 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | -0.7812 | `[" polygon", " Polygon", " triangle", " geometry", " Triangle"]` |
| deepseek7b | `triangle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.5625 | `[" Polygon", " polygon", " triangle", " ", " ["]` |
| deepseek7b | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.3125 | `[" Polygon", " polygon", " shape", " ", " square"]` |
| deepseek7b | `square` | `natural_question` | `top1_half` | `half` | 1 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.2188 | `[" Polygon", " ", " polygon", " shape", " square"]` |
| deepseek7b | `square` | `natural_question` | `top8_half` | `half` | 8 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.1875 | `[" Polygon", " polygon", " square", " ", " quadr"]` |
| deepseek7b | `triangle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | -0.0938 | `[" polygon", " triangle", " Polygon", " [", " Triangle"]` |
| deepseek7b | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.0938 | `[" Polygon", " square", " ", " polygon", " shape"]` |
| deepseek7b | `square` | `natural_question` | `top4_half` | `half` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.0000 | `[" Polygon", " ", " polygon", " square", " shape"]` |
| deepseek7b | `polygon` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 0.0000 | `[" geometry", " shape", " ?\n\n", " ?\n", " Geometry"]` |
| deepseek7b | `polygon` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 0.0000 | `[" geometry", " shape", " ?\n\n", " ?\n", " Geometry"]` |
| deepseek7b | `polygon` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 0.0000 | `[" shape", " geometry", " ?\n\n", " ?\n", " Geometry"]` |
| deepseek7b | `square` | `object_only` | `top12_flip` | `flip` | 12 | `object_echo` | square | `object_echo` | 0 | 0 | -6.4844 | `[" square", " Square", " Squ", " squared", " four"]` |
| deepseek7b | `square` | `natural_question` | `top12_flip` | `flip` | 12 | `object_echo` | square | `unknown_other` | 0 | 0 | -4.2188 | `[" square", " Square", " quadr", " squares", " Squ"]` |
| deepseek7b | `square` | `object_only` | `top12_zero` | `zero` | 12 | `object_echo` | square | `object_echo` | 0 | 0 | -4.1875 | `[" square", " Square", " four", " a", " \""]` |
| deepseek7b | `rectangle` | `object_only` | `original` | `original` | 0 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.2188 | `[" rectangle", " \"", " Rectangle", " A", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `top1_half` | `half` | 1 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.2188 | `[" rectangle", " \"", " Rectangle", " A", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `top1_zero` | `zero` | 1 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.1250 | `[" rectangle", " \"", " Rectangle", " four", " A"]` |
| deepseek7b | `rectangle` | `object_only` | `top4_half` | `half` | 4 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.1250 | `[" rectangle", " Rectangle", " \"", " The", " A"]` |
| deepseek7b | `circle` | `object_only` | `original` | `original` | 0 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0625 | `[" a", " A", " \"", " The", " Round"]` |
| deepseek7b | `circle` | `object_only` | `top12_half` | `half` | 12 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0625 | `[" A", " \"", " a", " circle", " The"]` |
| deepseek7b | `circle` | `object_only` | `top1_zero` | `zero` | 1 | `unknown_other` | A round shape with all points | `unknown_other` | 0 | 0 | -3.0469 | `[" A", " a", " \"", " Round", " The"]` |
| deepseek7b | `circle` | `object_only` | `top1_half` | `half` | 1 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0312 | `[" a", " A", " \"", " The", " circle"]` |
| deepseek7b | `circle` | `object_only` | `top1_flip` | `flip` | 1 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0156 | `[" a", " A", " \"", " Round", " circle"]` |
| deepseek7b | `circle` | `object_only` | `top4_half` | `half` | 4 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0156 | `[" a", " A", " \"", " The", " Round"]` |
| deepseek7b | `rectangle` | `object_only` | `top1_flip` | `flip` | 1 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.0000 | `[" rectangle", " \"", " four", " Rectangle", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `top4_zero` | `zero` | 4 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.0000 | `[" rectangle", " \"", " Rectangle", " four", " The"]` |
| deepseek7b | `circle` | `object_only` | `top4_zero` | `zero` | 4 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -3.0000 | `[" A", " \"", " a", " The", " Round"]` |
| deepseek7b | `rectangle` | `object_only` | `top12_half` | `half` | 12 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.9688 | `[" rectangle", " Rectangle", " four", " \"", " A"]` |
| deepseek7b | `square` | `object_only` | `top12_half` | `half` | 12 | `object_echo` | square | `object_echo` | 0 | 0 | -2.8906 | `[" square", " four", " Square", " a", " A"]` |
| deepseek7b | `circle` | `object_only` | `top4_flip` | `flip` | 4 | `unknown_other` | a round shape with no | `unknown_other` | 0 | 0 | -2.8594 | `[" \"", " A", " a", " Round", " The"]` |
| deepseek7b | `circle` | `object_only` | `top12_zero` | `zero` | 12 | `unknown_other` | a round shape with no | `unknown_other` | 0 | 0 | -2.8594 | `[" \"", " A", " a", " circle", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `top8_half` | `half` | 8 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.8438 | `[" rectangle", " Rectangle", " \"", " four", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `top4_flip` | `flip` | 4 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.7500 | `[" rectangle", " four", " \"", " Rectangle", " The"]` |
| deepseek7b | `square` | `object_only` | `top8_flip` | `flip` | 8 | `object_echo` | square | `object_echo` | 0 | 0 | -2.7031 | `[" square", " four", " a", " Square", " \""]` |
| deepseek7b | `circle` | `object_only` | `top8_half` | `half` | 8 | `unknown_other` | a round shape with no beginning | `unknown_other` | 0 | 0 | -2.7031 | `[" A", " \"", " a", " The", " circle"]` |
| deepseek7b | `rectangle` | `object_only` | `top12_zero` | `zero` | 12 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.6562 | `[" rectangle", " Rectangle", " four", " \"", " The"]` |
| deepseek7b | `circle` | `natural_question` | `top12_half` | `half` | 12 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.5625 | `[" ", " A", " shape", " circle", " ["]` |
| deepseek7b | `rectangle` | `object_only` | `top8_zero` | `zero` | 8 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.4375 | `[" rectangle", " four", " \"", " Rectangle", " The"]` |
| deepseek7b | `circle` | `object_only` | `top8_zero` | `zero` | 8 | `unknown_other` | a round shape with no | `unknown_other` | 0 | 0 | -2.4062 | `[" \"", " A", " a", " The", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top8_half` | `half` | 8 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.3750 | `[" ", " A", " shape", " [", " circle"]` |
| deepseek7b | `square` | `object_only` | `top4_flip` | `flip` | 4 | `object_echo` | square | `object_echo` | 0 | 0 | -2.2812 | `[" square", " four", " a", " Square", " \""]` |
| deepseek7b | `square` | `object_only` | `top8_zero` | `zero` | 8 | `object_echo` | square | `object_echo` | 0 | 0 | -2.2188 | `[" square", " four", " a", " Square", " \""]` |
| deepseek7b | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1875 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `rectangle` | `natural_question` | `top1_half` | `half` | 1 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1875 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `rectangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1562 | `[" square", " A", " ", " Rectangle", " shape"]` |
| deepseek7b | `circle` | `natural_question` | `top1_half` | `half` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.1562 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `rectangle` | `natural_question` | `top4_half` | `half` | 4 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1250 | `[" square", " A", " ", " Rectangle", " shape"]` |
| deepseek7b | `circle` | `natural_question` | `top4_half` | `half` | 4 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.1250 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -2.0938 | `[" shape", " ?\n", " ?\n\n", " Shape", " __"]` |
| deepseek7b | `square` | `object_only` | `top4_zero` | `zero` | 4 | `object_echo` | square | `object_echo` | 0 | 0 | -2.0625 | `[" square", " four", " a", " Square", " \""]` |
| deepseek7b | `rectangle` | `object_only` | `top12_flip` | `flip` | 12 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -2.0156 | `[" rectangle", " four", " Rectangle", " \"", " The"]` |
| deepseek7b | `square` | `object_only` | `top8_half` | `half` | 8 | `object_echo` | square | `object_echo` | 0 | 0 | -2.0000 | `[" square", " four", " a", " Square", " A"]` |
| deepseek7b | `circle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0000 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top12_zero` | `zero` | 12 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0000 | `[" ", " A", " circle", " Circle", " ["]` |
| deepseek7b | `rectangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | square | `unknown_other` | 0 | 0 | -1.9688 | `[" square", " shape", " A", " ", " Rectangle"]` |
| deepseek7b | `rectangle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | square | `unknown_other` | 0 | 0 | -1.9688 | `[" square", " A", " Rectangle", " ", " shape"]` |
| deepseek7b | `circle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.9688 | `[" shape", " ?\n", " ?\n\n", " __", " geometry"]` |
| deepseek7b | `square` | `object_only` | `top4_half` | `half` | 4 | `object_echo` | square | `object_echo` | 0 | 0 | -1.8906 | `[" square", " four", " a", " Square", " A"]` |
| deepseek7b | `circle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.8906 | `[" shape", " ?\n", " ?\n\n", " geometry", " __"]` |
| deepseek7b | `circle` | `object_only` | `top12_flip` | `flip` | 12 | `unknown_other` | a round shape with no | `unknown_other` | 0 | 0 | -1.8203 | `[" \"", " A", " The", " a", " closed"]` |
