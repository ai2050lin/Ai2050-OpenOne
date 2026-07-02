# Phase 844 Geometry Route Natural Gear Set Search (smoke)

- Search: natural MLP down-input channel activation x readout-coupling over geometry cases.
- Boundary: gear-set atlas probe; not global closure.

## Model Summary

| model | gears | rows | cases | original target | target | lost | gained |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| glm4 | 4 | 5 | 1 | 1 | 4 | 1 | 0 |
| deepseek7b | 4 | 5 | 1 | 0 | 2 | 0 | 2 |

## Top Gears

| model | rank | layer | channel | hits | mean act | neg ratio | mean abs support | gear score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 27 | 7219 | 1 | 9.3125 | 0.0000 | 2.3054 | 1.5980 |
| qwen3 | 2 | 27 | 1561 | 1 | -9.6875 | 1.0000 | 2.0863 | 1.4461 |
| qwen3 | 3 | 27 | 2767 | 1 | -20.0000 | 1.0000 | 1.5933 | 1.1044 |
| qwen3 | 4 | 27 | 6061 | 1 | 4.4688 | 0.0000 | 0.4502 | 0.3121 |
| glm4 | 1 | 27 | 7041 | 1 | 8.6875 | 0.0000 | 0.5677 | 0.3935 |
| glm4 | 2 | 27 | 13523 | 1 | 9.5000 | 0.0000 | 0.2573 | 0.1783 |
| glm4 | 3 | 27 | 8310 | 1 | 4.3750 | 0.0000 | 0.0977 | 0.0677 |
| glm4 | 4 | 27 | 2870 | 1 | -1.7422 | 1.0000 | 0.0601 | 0.0417 |
| deepseek7b | 1 | 27 | 1106 | 1 | -62.2500 | 1.0000 | 6.5518 | 4.5414 |
| deepseek7b | 2 | 27 | 15791 | 1 | -124.5000 | 1.0000 | 6.2084 | 4.3033 |
| deepseek7b | 3 | 27 | 13360 | 1 | 90.0000 | 0.0000 | 4.0961 | 2.8392 |
| deepseek7b | 4 | 27 | 18866 | 1 | 68.0000 | 0.0000 | 3.6456 | 2.5269 |

## Subset Summary

| model | subset | n | target | lost | gained | object_echo | unknown | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `original` | 1 | 0 | 0 | 0 | 0 | 0 | 0.9375 | `{"broad_near_miss": 1}` |
| qwen3 | `top1_flip` | 1 | 0 | 0 | 0 | 0 | 0 | 1.8750 | `{"broad_near_miss": 1}` |
| qwen3 | `top1_zero` | 1 | 0 | 0 | 0 | 0 | 0 | 1.3750 | `{"broad_near_miss": 1}` |
| qwen3 | `top4_flip` | 1 | 0 | 0 | 0 | 0 | 0 | 0.0000 | `{"broad_near_miss": 1}` |
| qwen3 | `top4_zero` | 1 | 0 | 0 | 0 | 0 | 0 | 0.6250 | `{"broad_near_miss": 1}` |
| glm4 | `original` | 1 | 1 | 0 | 0 | 0 | 0 | 1.3906 | `{"target_equivalent": 1}` |
| glm4 | `top1_flip` | 1 | 1 | 0 | 0 | 0 | 0 | 3.3359 | `{"target_equivalent": 1}` |
| glm4 | `top1_zero` | 1 | 1 | 0 | 0 | 0 | 0 | 2.4531 | `{"target_equivalent": 1}` |
| glm4 | `top4_flip` | 1 | 0 | 1 | 0 | 0 | 0 | 3.7344 | `{"broad_near_miss": 1}` |
| glm4 | `top4_zero` | 1 | 1 | 0 | 0 | 0 | 0 | 2.7578 | `{"target_equivalent": 1}` |
| deepseek7b | `original` | 1 | 0 | 0 | 0 | 0 | 1 | -1.1875 | `{"unknown_other": 1}` |
| deepseek7b | `top1_flip` | 1 | 0 | 0 | 0 | 0 | 1 | -1.1562 | `{"unknown_other": 1}` |
| deepseek7b | `top1_zero` | 1 | 0 | 0 | 0 | 0 | 1 | -1.1875 | `{"unknown_other": 1}` |
| deepseek7b | `top4_flip` | 1 | 1 | 0 | 1 | 0 | 0 | -1.2500 | `{"target_equivalent": 1}` |
| deepseek7b | `top4_zero` | 1 | 1 | 0 | 1 | 0 | 0 | -1.2812 | `{"target_equivalent": 1}` |

## Object Summary

| model | object | n | target | lost | gained | object_echo | unknown | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `triangle` | 5 | 0 | 0 | 0 | 0 | 0 | `{"broad_near_miss": 5}` |
| glm4 | `triangle` | 5 | 4 | 1 | 0 | 0 | 0 | `{"broad_near_miss": 1, "target_equivalent": 4}` |
| deepseek7b | `triangle` | 5 | 2 | 0 | 2 | 0 | 3 | `{"target_equivalent": 2, "unknown_other": 3}` |

## Top Rows

| model | object | prompt | subset | mode | gears | class | output | orig class | lost | gained | target-object | top tokens |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---|
| qwen3 | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 1.8750 | `[" Geometry", " Polygon", " ", " Shapes", " geometry"]` |
| qwen3 | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 1.3750 | `[" Geometry", " Polygon", " ", " Shapes", " geometry"]` |
| qwen3 | `triangle` | `natural_question` | `original` | `original` | 0 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.9375 | `[" Geometry", " Polygon", " ", " Triangle", " Shapes"]` |
| qwen3 | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.6250 | `[" Geometry", " geometry", " Shapes", " Polygon", " Ge"]` |
| qwen3 | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `broad_near_miss` | Geometry | `broad_near_miss` | 0 | 0 | 0.0000 | `[" Geometry", " geometry", " Shapes", " Ge", "Geometry"]` |
| glm4 | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `broad_near_miss` | Geometry | `target_equivalent` | 1 | 0 | 3.7344 | `[" Geometry", " Mathematical", " Math", " Plane", " Ge"]` |
| glm4 | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 3.3359 | `[" Shape", " Ge", " shape", " geometric", " Plane"]` |
| glm4 | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.7578 | `[" Shape", " Ge", " Geometry", " geometric", " Plane"]` |
| glm4 | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `target_equivalent` | Shape | `target_equivalent` | 0 | 0 | 2.4531 | `[" Shape", " Ge", " geometric", " shape", " "]` |
| glm4 | `triangle` | `natural_question` | `original` | `original` | 0 | `target_equivalent` | Geometric Shape | `target_equivalent` | 0 | 0 | 1.3906 | `[" Ge", " Shape", " geometric", " shape", " Geometry"]` |
| deepseek7b | `triangle` | `natural_question` | `top4_zero` | `zero` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.2812 | `[" Polygon", " geometry", " polygon", " ", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `top4_flip` | `flip` | 4 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.2500 | `[" Polygon", " polygon", " geometry", " triangle", " "]` |
| deepseek7b | `triangle` | `natural_question` | `original` | `original` | 0 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | `[" ", " [", " geometry", " Polygon", " A"]` |
| deepseek7b | `triangle` | `natural_question` | `top1_zero` | `zero` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | `[" ", " Polygon", " [", " geometry", " polygon"]` |
| deepseek7b | `triangle` | `natural_question` | `top1_flip` | `flip` | 1 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1562 | `[" ", " Polygon", " [", " geometry", " polygon"]` |
