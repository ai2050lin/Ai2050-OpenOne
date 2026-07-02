# Phase 844 Geometry Route Natural Gear Set Search (main)

- Search: natural MLP down-input channel activation x readout-coupling over geometry cases.
- Boundary: gear-set atlas probe; not global closure.

## Model Summary

| model | gears | rows | cases | original target | target | lost | gained |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 56 | 4 | 2 | 25 | 0 | 11 |
| glm4 | 8 | 56 | 4 | 7 | 48 | 4 | 3 |
| deepseek7b | 8 | 56 | 4 | 4 | 38 | 0 | 10 |

## Top Gears

| model | rank | layer | channel | hits | mean act | neg ratio | mean abs support | gear score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 27 | 2767 | 8 | -22.9062 | 1.0000 | 2.1805 | 4.7910 |
| qwen3 | 2 | 27 | 1561 | 8 | -9.0234 | 1.0000 | 1.7600 | 3.8672 |
| qwen3 | 3 | 28 | 4231 | 8 | -4.4180 | 1.0000 | 0.7467 | 1.6406 |
| qwen3 | 4 | 27 | 7219 | 2 | 5.8047 | 0.0000 | 1.4370 | 1.5787 |
| qwen3 | 5 | 28 | 2872 | 8 | 16.5000 | 0.0000 | 0.6629 | 1.4565 |
| qwen3 | 6 | 27 | 6061 | 7 | 4.8131 | 0.0000 | 0.5976 | 1.2426 |
| qwen3 | 7 | 28 | 7316 | 8 | 5.6191 | 0.0000 | 0.5653 | 1.2421 |
| qwen3 | 8 | 28 | 5220 | 8 | -6.9473 | 1.0000 | 0.5099 | 1.1205 |
| glm4 | 1 | 28 | 8036 | 8 | 3.3867 | 0.0000 | 0.2948 | 0.6478 |
| glm4 | 2 | 27 | 7041 | 8 | 3.4846 | 0.0000 | 0.1581 | 0.3475 |
| glm4 | 3 | 26 | 6031 | 2 | 1.8867 | 0.0000 | 0.2835 | 0.3115 |
| glm4 | 4 | 28 | 2777 | 2 | 1.5117 | 0.0000 | 0.2527 | 0.2776 |
| glm4 | 5 | 27 | 10905 | 2 | -1.3789 | 1.0000 | 0.2514 | 0.2762 |
| glm4 | 6 | 27 | 13523 | 8 | 7.7344 | 0.0000 | 0.1119 | 0.2458 |
| glm4 | 7 | 28 | 6279 | 8 | 3.7461 | 0.0000 | 0.1033 | 0.2269 |
| glm4 | 8 | 26 | 13347 | 8 | 1.6113 | 0.0000 | 0.0924 | 0.2029 |
| deepseek7b | 1 | 27 | 1106 | 8 | -54.5625 | 1.0000 | 7.3943 | 16.2470 |
| deepseek7b | 2 | 27 | 15791 | 8 | -116.4375 | 1.0000 | 6.7375 | 14.8039 |
| deepseek7b | 3 | 27 | 2295 | 8 | -50.0312 | 1.0000 | 6.1684 | 13.5533 |
| deepseek7b | 4 | 27 | 15305 | 8 | -83.8438 | 1.0000 | 5.8138 | 12.7743 |
| deepseek7b | 5 | 27 | 13360 | 8 | 85.0625 | 0.0000 | 5.4758 | 12.0316 |
| deepseek7b | 6 | 27 | 18866 | 8 | 77.7500 | 0.0000 | 3.7269 | 8.1888 |
| deepseek7b | 7 | 26 | 13399 | 8 | -25.8750 | 1.0000 | 3.5553 | 7.8118 |
| deepseek7b | 8 | 26 | 10821 | 8 | 7.9609 | 0.0000 | 3.4327 | 7.5424 |

## Subset Summary

| model | subset | n | target | lost | gained | object_echo | unknown | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `original` | 8 | 2 | 0 | 0 | 0 | 5 | 3.8867 | `{"broad_near_miss": 1, "target_equivalent": 2, "unknown_other": 5}` |
| qwen3 | `top1_flip` | 8 | 4 | 0 | 2 | 0 | 3 | 1.6621 | `{"broad_near_miss": 1, "target_equivalent": 4, "unknown_other": 3}` |
| qwen3 | `top1_zero` | 8 | 4 | 0 | 2 | 0 | 3 | 2.8691 | `{"broad_near_miss": 1, "target_equivalent": 4, "unknown_other": 3}` |
| qwen3 | `top4_flip` | 8 | 4 | 0 | 2 | 0 | 3 | 2.8008 | `{"broad_near_miss": 1, "target_equivalent": 4, "unknown_other": 3}` |
| qwen3 | `top4_zero` | 8 | 4 | 0 | 2 | 0 | 3 | 3.4082 | `{"broad_near_miss": 1, "target_equivalent": 4, "unknown_other": 3}` |
| qwen3 | `top8_flip` | 8 | 3 | 0 | 1 | 0 | 3 | 0.4883 | `{"broad_near_miss": 2, "target_equivalent": 3, "unknown_other": 3}` |
| qwen3 | `top8_zero` | 8 | 4 | 0 | 2 | 0 | 3 | 2.1602 | `{"broad_near_miss": 1, "target_equivalent": 4, "unknown_other": 3}` |
| glm4 | `original` | 8 | 7 | 0 | 0 | 0 | 1 | 1.6611 | `{"target_equivalent": 7, "unknown_other": 1}` |
| glm4 | `top1_flip` | 8 | 8 | 0 | 1 | 0 | 0 | 1.3179 | `{"target_equivalent": 8}` |
| glm4 | `top1_zero` | 8 | 6 | 1 | 0 | 0 | 2 | 1.4878 | `{"target_equivalent": 6, "unknown_other": 2}` |
| glm4 | `top4_flip` | 8 | 6 | 1 | 0 | 0 | 2 | 1.7803 | `{"target_equivalent": 6, "unknown_other": 2}` |
| glm4 | `top4_zero` | 8 | 7 | 0 | 0 | 0 | 1 | 1.7036 | `{"target_equivalent": 7, "unknown_other": 1}` |
| glm4 | `top8_flip` | 8 | 7 | 1 | 1 | 0 | 1 | 1.9371 | `{"target_equivalent": 7, "unknown_other": 1}` |
| glm4 | `top8_zero` | 8 | 7 | 1 | 1 | 0 | 1 | 1.7505 | `{"target_equivalent": 7, "unknown_other": 1}` |
| deepseek7b | `original` | 8 | 4 | 0 | 0 | 0 | 4 | -0.5234 | `{"target_equivalent": 4, "unknown_other": 4}` |
| deepseek7b | `top1_flip` | 8 | 4 | 0 | 0 | 0 | 4 | -0.5000 | `{"target_equivalent": 4, "unknown_other": 4}` |
| deepseek7b | `top1_zero` | 8 | 4 | 0 | 0 | 0 | 4 | -0.5156 | `{"target_equivalent": 4, "unknown_other": 4}` |
| deepseek7b | `top4_flip` | 8 | 7 | 0 | 3 | 0 | 1 | 0.0859 | `{"target_equivalent": 7, "unknown_other": 1}` |
| deepseek7b | `top4_zero` | 8 | 7 | 0 | 3 | 0 | 1 | -0.2734 | `{"target_equivalent": 7, "unknown_other": 1}` |
| deepseek7b | `top8_flip` | 8 | 6 | 0 | 2 | 0 | 1 | -0.7568 | `{"broad_near_miss": 1, "target_equivalent": 6, "unknown_other": 1}` |
| deepseek7b | `top8_zero` | 8 | 6 | 0 | 2 | 0 | 1 | -0.6895 | `{"broad_near_miss": 1, "target_equivalent": 6, "unknown_other": 1}` |

## Object Summary

| model | object | n | target | lost | gained | object_echo | unknown | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `circle` | 14 | 7 | 0 | 0 | 0 | 7 | `{"target_equivalent": 7, "unknown_other": 7}` |
| qwen3 | `rectangle` | 14 | 6 | 0 | 6 | 0 | 8 | `{"target_equivalent": 6, "unknown_other": 8}` |
| qwen3 | `square` | 14 | 7 | 0 | 0 | 0 | 7 | `{"target_equivalent": 7, "unknown_other": 7}` |
| qwen3 | `triangle` | 14 | 5 | 0 | 5 | 0 | 1 | `{"broad_near_miss": 8, "target_equivalent": 5, "unknown_other": 1}` |
| glm4 | `circle` | 14 | 14 | 0 | 0 | 0 | 0 | `{"target_equivalent": 14}` |
| glm4 | `rectangle` | 14 | 9 | 1 | 3 | 0 | 5 | `{"target_equivalent": 9, "unknown_other": 5}` |
| glm4 | `square` | 14 | 14 | 0 | 0 | 0 | 0 | `{"target_equivalent": 14}` |
| glm4 | `triangle` | 14 | 11 | 3 | 0 | 0 | 3 | `{"target_equivalent": 11, "unknown_other": 3}` |
| deepseek7b | `circle` | 14 | 11 | 0 | 4 | 0 | 3 | `{"target_equivalent": 11, "unknown_other": 3}` |
| deepseek7b | `rectangle` | 14 | 7 | 0 | 0 | 0 | 7 | `{"target_equivalent": 7, "unknown_other": 7}` |
| deepseek7b | `square` | 14 | 11 | 0 | 4 | 0 | 3 | `{"target_equivalent": 11, "unknown_other": 3}` |
| deepseek7b | `triangle` | 14 | 9 | 0 | 2 | 0 | 3 | `{"broad_near_miss": 2, "target_equivalent": 9, "unknown_other": 3}` |

## Top Rows

| model | object | prompt | subset | mode | gears | class | output | orig class | lost | gained | target-object | top tokens |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---|
| qwen3 | `rectangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.4375 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 4.1562 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 3.4688 | `[" shape", " Shape", " shapes", " Shapes", " __"]` |
| qwen3 | `rectangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.8750 | `[" shape", " Shape", " quadr", " shapes", " __"]` |
| qwen3 | `triangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.5625 | `[" shape", " Shape", " geometric", " polygon", " Shapes"]` |
| qwen3 | `rectangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.4375 | `[" shape", " Shape", " Shapes", " shapes", " __"]` |
| qwen3 | `triangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 2.0000 | `[" shape", " Shape", " geometric", " geometry", " Shapes"]` |
| qwen3 | `triangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 1.8750 | `[" shape", " Shape", " polygon", " geometric", " "]` |
| qwen3 | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 0.9375 | `[" shape", " Shape", " __", " geometry", " ?\n"]` |
| qwen3 | `triangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 0.8125 | `[" polygon", " shape", " triangle", " geometry", " Geometry"]` |
| qwen3 | `triangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | 0.6250 | `[" shape", " Shape", " Shapes", " geometric", " geometry"]` |
| qwen3 | `square` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 8.0938 | `[" shape", " Shape", " Shapes", " shapes", " quadr"]` |
| qwen3 | `square` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.4531 | `[" shape", " Shape", " shapes", " Shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 7.3594 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 5.9062 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `square` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 5.7500 | `[" shape", " Shape", " Shapes", " shapes", " __"]` |
| qwen3 | `square` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.5938 | `[" Geometry", " Quadr", " Ge", " Shapes", " Polygon"]` |
| qwen3 | `rectangle` | `natural_category` | `original` | `original` | 0 | `unknown_other` | quadrilateral | `unknown_other` | 0 | 0 | 5.4688 | `[" quadr", " shape", " ", " Shape", " Shapes"]` |
| qwen3 | `square` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 5.4219 | `[" shape", " Shape", " Shapes", " shapes", " __"]` |
| qwen3 | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 5.0625 | `[" Geometry", " Shapes", " geometry", " Ge", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 4.6875 | `[" Geometry", " Shapes", " geometry", " Ge", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 4.6250 | `[" Geometry", " geometry", " Ge", " Shapes", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.8750 | `[" Geometry", " geometry", " Shapes", " Ge", " Shape"]` |
| qwen3 | `square` | `natural_question` | `top8_zero` | `zero` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.6250 | `[" Geometry", " geometry", " Ge", " Shapes", " Polygon"]` |
| qwen3 | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 3.2500 | `[" Geometry", " Quadr", " Shapes", " ", " Ge"]` |
| qwen3 | `circle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 3.1875 | `[" shape", " Shape", " shapes", " Shapes", " geometric"]` |
| qwen3 | `circle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 3.1250 | `[" shape", " Shape", " Shapes", " shapes", " "]` |
| qwen3 | `circle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 3.1250 | `[" shape", " Shape", " Shapes", " shapes", " geometric"]` |
| qwen3 | `triangle` | `natural_category` | `original` | `original` | 0 | `unknown_other` | 2D shapes | `unknown_other` | 0 | 0 | 2.9375 | `[" ", " polygon", " shape", " geometric", " Shape"]` |
| qwen3 | `square` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 2.7812 | `[" shape", " Shape", " Shapes", " __", " type"]` |
| qwen3 | `circle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 2.5000 | `[" Shape", " shape", " geometric", " Shapes", " Geometry"]` |
| qwen3 | `circle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 2.4375 | `[" shape", " Shape", " Shapes", " shapes", " geometric"]` |
| qwen3 | `circle` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 2.1875 | `[" Geometry", " Shapes", " Shape", " Ge", " geometry"]` |
| qwen3 | `circle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 2.0000 | `[" Geometry", " Shapes", " Shape", " geometry", " Ge"]` |
| qwen3 | `square` | `natural_question` | `top8_flip` | `flip` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.9375 | `[" Geometry", " geometry", " Ge", " Square", " "]` |
| qwen3 | `rectangle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.9375 | `[" Geometry", " Shapes", " geometry", " Ge", " Shape"]` |
| qwen3 | `circle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.6875 | `[" Geometry", " Shapes", " Shape", " geometry", " "]` |
| qwen3 | `circle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.6875 | `[" shape", " Shape", " Shapes", " shapes", " geometric"]` |
| qwen3 | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | -1.5000 | `[" Geometry", " geometry", " Triangle", " Shapes", " Ge"]` |
| qwen3 | `rectangle` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.5000 | `[" Geometry", " geometry", " Shapes", " Ge", " Shape"]` |
| qwen3 | `rectangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.3750 | `[" Geometry", " Shapes", " geometry", " Ge", " "]` |
| qwen3 | `circle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 1.2500 | `[" geometric", " Geometry", " geometry", " Shape", " shape"]` |
| qwen3 | `triangle` | `natural_question` | `top8_flip` | `flip` | 8 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | -1.1875 | `[" Geometry", " geometry", "Geometry", " Triangle", " Ge"]` |
| qwen3 | `circle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 1.0000 | `[" Geometry", " Shapes", " Shape", " geometry", " Ge"]` |
| qwen3 | `triangle` | `natural_question` | `original` | `original` | 0 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.9375 | `[" Geometry", " Polygon", " ", " Triangle", " Shapes"]` |
| qwen3 | `triangle` | `natural_category` | `top8_flip` | `flip` | 8 | `broad_near_miss` | Geometry | `unknown_other` | 0 | 0 | -0.9375 | `[" Geometry", " geometry", " type", " triangle", " geometric"]` |
| qwen3 | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.6875 | `[" Geometry", " geometry", " Polygon", " Shapes", " Ge"]` |
| qwen3 | `circle` | `natural_question` | `top8_zero` | `zero` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 0.6875 | `[" Geometry", " Shapes", " geometry", " Ge", " Shape"]` |
| qwen3 | `rectangle` | `natural_question` | `top8_zero` | `zero` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 0.6250 | `[" Geometry", " geometry", " Shapes", " Ge", " "]` |
| qwen3 | `rectangle` | `natural_question` | `top8_flip` | `flip` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | -0.6250 | `[" Geometry", " geometry", " Ge", "Geometry", " Shapes"]` |
| qwen3 | `rectangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 0.3750 | `[" Geometry", " geometry", " Shapes", " Ge", " Shape"]` |
| qwen3 | `circle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | 0.3750 | `[" Geometry", " Shapes", " Shape", " geometry", " Ge"]` |
| qwen3 | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | -0.3125 | `[" Geometry", " geometry", " Shapes", " Ge", "Geometry"]` |
| qwen3 | `triangle` | `natural_question` | `top8_zero` | `zero` | 8 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.2500 | `[" Geometry", " geometry", " Polygon", " Triangle", " Ge"]` |
| qwen3 | `circle` | `natural_question` | `top8_flip` | `flip` | 8 | `unknown_other` | Geometry | `unknown_other` | 0 | 0 | -0.2500 | `[" Geometry", " geometry", " Ge", " Shapes", " Circle"]` |
| qwen3 | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.0625 | `[" Geometry", " Polygon", " geometry", " Triangle", " Shapes"]` |
| glm4 | `triangle` | `natural_category` | `top8_flip` | `flip` | 8 | `unknown_other` | geometric figure | `target_equivalent` | 1 | 0 | 2.8252 | `[" geometric", " shape", " three", " polygon", " plane"]` |
| glm4 | `triangle` | `natural_category` | `top4_flip` | `flip` | 4 | `unknown_other` | geometric figure | `target_equivalent` | 1 | 0 | 2.5117 | `[" geometric", " polygon", " shape", " three", " Shape"]` |
| glm4 | `rectangle` | `natural_category` | `top1_zero` | `zero` | 1 | `unknown_other` | quadrilateral | `target_equivalent` | 1 | 0 | 2.1992 | `[" quadr", " geometric", " polygon", " shape", " paralle"]` |
| glm4 | `triangle` | `natural_category` | `top8_zero` | `zero` | 8 | `unknown_other` | geometric figure | `target_equivalent` | 1 | 0 | 2.1328 | `[" geometric", " polygon", " shape", " quadr", " three"]` |
| glm4 | `rectangle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Geometric Shape | `unknown_other` | 0 | 1 | 2.0234 | `[" Ge", " Quadr", " Shape", " Two", " "]` |
| glm4 | `rectangle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Geometric Shape | `unknown_other` | 0 | 1 | 1.9922 | `[" Ge", " Shape", " Two", " ", " Quadr"]` |
| glm4 | `rectangle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Geometric Shape | `unknown_other` | 0 | 1 | 1.8750 | `[" Ge", " Quadr", " Shape", " Two", " quadr"]` |
| glm4 | `triangle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.8398 | `[" Shape", " Plane", " Ge", " Geometry", " Mathematical"]` |
| glm4 | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.0000 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `square` | `natural_question` | `original` | `original` | 0 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.8164 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `triangle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.7734 | `[" Shape", " Ge", " geometric", " Plane", " "]` |
| glm4 | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.6562 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.6172 | `[" Ge", " Shape", " geometric", " shape", " "]` |
| glm4 | `square` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.6133 | `[" Ge", " Shape", " geometric", " shape", " A"]` |
| glm4 | `square` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.5664 | `[" Ge", " Shape", " geometric", " shape", " Quadr"]` |
| glm4 | `square` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.5195 | `[" Shape", " geometric", " ", " Ge", " A"]` |
| glm4 | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Geometric shape | `target_equivalent` | 0 | 0 | 2.3359 | `[" Ge", " Shape", " geometric", " shape", " A"]` |
| glm4 | `rectangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3320 | `[" geometric", " quadr", " polygon", " shape", " paralle"]` |
| glm4 | `square` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.3203 | `[" geometric", " shape", " polygon", " quadr", " Shape"]` |
| glm4 | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 2.2500 | `[" Ge", " Shape", " geometric", " shape", " "]` |
| glm4 | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.2266 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `square` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.2031 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `rectangle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.1719 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `square` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.1680 | `[" geometric", " shape", " polygon", " quadr", " __"]` |
| glm4 | `triangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.1562 | `[" polygon", " geometric", " shape", " quadr", " polygons"]` |
| glm4 | `rectangle` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.1172 | `[" Quadr", " Ge", " Shape", " Two", " "]` |
| glm4 | `square` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.1133 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `square` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.0977 | `[" geometric", " shape", " polygon", " Shape", " __"]` |
| glm4 | `rectangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.0977 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `square` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.0742 | `[" geometric", " shape", " polygon", " __", " Shape"]` |
| glm4 | `rectangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | Quadrilateral | `unknown_other` | 0 | 0 | 2.0469 | `[" Quadr", " Ge", " Shape", " Two", " quadr"]` |
| glm4 | `triangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 2.0156 | `[" polygon", " geometric", " shape", " polygons", " quadr"]` |
| glm4 | `rectangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 2.0156 | `[" geometric", " quadr", " paralle", " polygon", " shape"]` |
| glm4 | `rectangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 1.7930 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `rectangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 1.7422 | `[" geometric", " quadr", " shape", " polygon", " paralle"]` |
| glm4 | `triangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.7188 | `[" polygon", " geometric", " shape", " quadr", " polygons"]` |
| glm4 | `square` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 1.6875 | `[" geometric", " shape", " __", " ______", " Shape"]` |
| glm4 | `triangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.5312 | `[" polygon", " geometric", " shape", " quadr", " triangle"]` |
| glm4 | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 1.4375 | `[" geometric", " quadr", " shape", " polygon", " __"]` |
| glm4 | `triangle` | `natural_question` | `original` | `original` | 0 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 1.3906 | `[" Ge", " Shape", " geometric", " shape", " Geometry"]` |
| glm4 | `circle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 1.2969 | `[" Shape", " Ge", " shape", " Plane", " "]` |
| glm4 | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 1.1875 | `[" Ge", " Shape", " geometric", " shape", " Geometry"]` |
| glm4 | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 0.9375 | `[" Ge", " geometric", " Shape", " Geometry", " shape"]` |
| glm4 | `circle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.8125 | `[" Shape", " shape", " Ge", " ", " A"]` |
| glm4 | `circle` | `natural_question` | `original` | `original` | 0 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.5781 | `[" Shape", " shape", " closed", " Ge", " "]` |
| glm4 | `circle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.5156 | `[" Shape", " shape", " Ge", " closed", " Plane"]` |
| glm4 | `circle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.5000 | `[" Shape", " shape", " Ge", " Plane", " "]` |
| glm4 | `circle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.4219 | `[" geometric", " shape", " shapes", " circle", " geometry"]` |
| glm4 | `circle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.4219 | `[" geometric", " shape", " shapes", " circle", " "]` |
| glm4 | `circle` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.4062 | `[" Shape", " shape", " Ge", " closed", " A"]` |
| glm4 | `circle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.4062 | `[" geometric", " shape", " shapes", " geometry", " "]` |
| glm4 | `circle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.3906 | `[" geometric", " shape", " shapes", " circle", " geometry"]` |
| glm4 | `circle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.3906 | `[" geometric", " shape", " shapes", " circle", " "]` |
| glm4 | `circle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 0.1719 | `[" Shape", " shape", " Ge", " closed", " A"]` |
| glm4 | `circle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.1719 | `[" geometric", " shape", " geometry", " ", " shapes"]` |
| glm4 | `circle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | -0.1016 | `[" geometric", " shape", " geometry", " ", " shapes"]` |
| deepseek7b | `circle` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -2.1562 | `[" shape", " ", " A", " circle", " Circle"]` |
| deepseek7b | `circle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -1.8750 | `[" shape", " ", " A", " circle", " ["]` |
| deepseek7b | `circle` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -1.4062 | `[" shape", " circle", " shapes", " ", " Circle"]` |
| deepseek7b | `square` | `natural_question` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -1.0938 | `[" shape", " Polygon", " polygon", " geometry", " "]` |
| deepseek7b | `circle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `unknown_other` | 0 | 1 | -0.9062 | `[" shape", " ", " A", " Circle", " circle"]` |
| deepseek7b | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.5938 | `[" Polygon", " ", " geometry", " polygon", " ["]` |
| deepseek7b | `square` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.5938 | `[" Polygon", " polygon", " shape", " ", " square"]` |
| deepseek7b | `square` | `natural_question` | `top8_zero` | `zero` | 8 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.5000 | `[" Polygon", " shape", " ", " polygon", " square"]` |
| deepseek7b | `square` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -0.2500 | `[" Polygon", " polygon", " ", " shape", " square"]` |
| deepseek7b | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.0000 | `[" Polygon", " polygon", " geometry", " ", " ["]` |
| deepseek7b | `triangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 2.2812 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `circle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -2.2578 | `[" shape", " ?\n", " ?\n\n", " geometry", " __"]` |
| deepseek7b | `rectangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1875 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `rectangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1875 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `rectangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1250 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `circle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -2.0156 | `[" shape", " ?\n", " ?\n\n", " geometry", " __"]` |
| deepseek7b | `circle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0000 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0000 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.9688 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `triangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.8750 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `circle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.8125 | `[" shape", " ?\n", " ?\n\n", " __", " geometry"]` |
| deepseek7b | `rectangle` | `natural_question` | `top8_zero` | `zero` | 8 | `unknown_other` | square | `unknown_other` | 0 | 0 | -1.7812 | `[" square", " Rectangle", " paralle", " shape", " A"]` |
| deepseek7b | `circle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.7500 | `[" shape", " ?\n", " ?\n\n", " __", " geometry"]` |
| deepseek7b | `circle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.6875 | `[" shape", " ?\n", " ?\n\n", " [", " geometry"]` |
| deepseek7b | `circle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.6875 | `[" ?\n", " shape", " ?\n\n", " geometry", " ["]` |
| deepseek7b | `circle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.6562 | `[" shape", " ?\n", " ?\n\n", " geometry", " ["]` |
| deepseek7b | `triangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5000 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `triangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5000 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `triangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.4688 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `rectangle` | `natural_question` | `top8_flip` | `flip` | 8 | `unknown_other` | parallelogram | `unknown_other` | 0 | 0 | -1.4062 | `[" paralle", " square", " Rectangle", " shape", " rectangle"]` |
| deepseek7b | `triangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.2188 | `[" geometry", " shape", " Geometry", " ?\n", " polygon"]` |
| deepseek7b | `rectangle` | `natural_question` | `top4_zero` | `zero` | 4 | `unknown_other` | square | `unknown_other` | 0 | 0 | -1.2188 | `[" square", " Rectangle", " ", " A", " shape"]` |
| deepseek7b | `rectangle` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.2188 | `[" shape", " geometry", " ?\n", " __", " Shape"]` |
| deepseek7b | `triangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | `[" ", " [", " geometry", " Polygon", " A"]` |
| deepseek7b | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | `[" ", " Polygon", " [", " geometry", " polygon"]` |
| deepseek7b | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1562 | `[" ", " Polygon", " [", " geometry", " polygon"]` |
| deepseek7b | `square` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.0938 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.0938 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.0625 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `triangle` | `natural_question` | `top8_zero` | `zero` | 8 | `broad_near_miss` | geometry | `unknown_other` | 0 | 0 | -1.0000 | `[" geometry", " Polygon", " ", " [", " polygon"]` |
| deepseek7b | `triangle` | `natural_question` | `top8_flip` | `flip` | 8 | `broad_near_miss` | geometry | `unknown_other` | 0 | 0 | -0.9062 | `[" geometry", " Polygon", " ", " triangle", " polygon"]` |
| deepseek7b | `square` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.9062 | `[" shape", " geometry", " ?\n", " Geometry", " ?\n\n"]` |
| deepseek7b | `triangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 0.7500 | `[" geometry", " shape", " Geometry", " ?\n", " ?\n\n"]` |
| deepseek7b | `square` | `natural_category` | `top4_flip` | `flip` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.7188 | `[" shape", " geometry", " ?\n", " Shape", " Geometry"]` |
| deepseek7b | `rectangle` | `natural_category` | `top4_zero` | `zero` | 4 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.7188 | `[" shape", " geometry", " ?\n", " ?\n\n", " Shape"]` |
| deepseek7b | `square` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.6250 | `[" shape", " geometry", " ?\n", " Geometry", " ?\n\n"]` |
| deepseek7b | `square` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | 0.2188 | `[" ", " Polygon", " polygon", " shape", " square"]` |
| deepseek7b | `square` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | 0.2188 | `[" ", " Polygon", " polygon", " shape", " square"]` |
| deepseek7b | `rectangle` | `natural_question` | `top4_flip` | `flip` | 4 | `unknown_other` | square | `unknown_other` | 0 | 0 | -0.2188 | `[" square", " shape", " Rectangle", " A", " "]` |
| deepseek7b | `square` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.2031 | `[" shape", " geometry", " ?\n", " Geometry", " shapes"]` |
| deepseek7b | `square` | `natural_question` | `original` | `original` | 0 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | 0.1562 | `[" ", " Polygon", " polygon", " shape", " square"]` |
| deepseek7b | `rectangle` | `natural_category` | `top1_zero` | `zero` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1562 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `rectangle` | `natural_category` | `original` | `original` | 0 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1250 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `rectangle` | `natural_category` | `top1_flip` | `flip` | 1 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1250 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `rectangle` | `natural_category` | `top8_zero` | `zero` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.0938 | `[" shape", " geometry", " ?\n", " shapes", " ?\n\n"]` |
| deepseek7b | `rectangle` | `natural_category` | `top8_flip` | `flip` | 8 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.0625 | `[" shape", " geometry", " shapes", " ?\n", " __"]` |
