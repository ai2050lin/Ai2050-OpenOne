# Phase 843 Core Channel Natural Route Validation (main)

- Source: Phase 842 core channel candidate.
- Boundary: natural activation + first-step channel edit; not global closure.

## Model Summary

| model | skipped | rows | cases | original target | target | lost vs original | gained vs original |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 16 | 1 | 2 | 10 | 0 | 2 |

## Mode Summary

| model | mode | n | target | lost | gained | object_echo | unknown | mean act | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip` | 4 | 3 | 0 | 1 | 0 | 1 | -16.9434 | 1.0703 | `{"target_equivalent": 3, "unknown_other": 1}` |
| deepseek7b | `half` | 4 | 2 | 0 | 0 | 0 | 2 | -16.9434 | 0.9062 | `{"target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `original` | 4 | 2 | 0 | 0 | 0 | 2 | -16.9434 | 0.8828 | `{"target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `zero` | 4 | 3 | 0 | 1 | 0 | 1 | -16.9434 | 0.9844 | `{"target_equivalent": 3, "unknown_other": 1}` |

## Object Summary

| model | object | n | target | lost | gained | mean act | mean abs act | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `triangle` | 16 | 10 | 0 | 2 | -16.9434 | 16.9434 | `{"target_equivalent": 10, "unknown_other": 6}` |

## Top Rows

| model | object | prompt | mode | act | class | output | orig class | lost | gained | target-object | target rank | object rank | top tokens |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| deepseek7b | `triangle` | `natural_question` | `zero` | -18.6250 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0625 | 176 | 89 | `[" Polygon", " polygon", " ", " triangle", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `flip` | -18.6250 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | -0.9375 | 130 | 73 | `[" polygon", " Polygon", " polygons", " triangle", " "]` |
| deepseek7b | `triangle` | `exact_choices` | `original` | -1.7734 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 4.2500 | 58 | 518 | `[" geometric", " The", " __", " ?\n\n", " Ge"]` |
| deepseek7b | `triangle` | `exact_choices` | `zero` | -1.7734 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 4.2188 | 58 | 503 | `[" geometric", " The", " __", " ?\n\n", " Ge"]` |
| deepseek7b | `triangle` | `exact_choices` | `half` | -1.7734 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 4.2188 | 58 | 503 | `[" geometric", " The", " __", " ?\n\n", " Ge"]` |
| deepseek7b | `triangle` | `exact_choices` | `flip` | -1.7734 | `target_equivalent` | geometric shape | `target_equivalent` | 0 | 0 | 4.1875 | 58 | 495 | `[" geometric", " The", " __", " ?\n\n", " Ge"]` |
| deepseek7b | `triangle` | `natural_category` | `flip` | -26.6250 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.8125 | 187 | 554 | `[" polygon", " geometry", " Polygon", " shape", " polygons"]` |
| deepseek7b | `triangle` | `natural_category` | `zero` | -26.6250 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.6875 | 265 | 757 | `[" polygon", " geometry", " shape", " Geometry", " ?\n"]` |
| deepseek7b | `triangle` | `natural_category` | `half` | -26.6250 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5938 | 330 | 903 | `[" geometry", " polygon", " shape", " Geometry", " ?\n"]` |
| deepseek7b | `triangle` | `natural_category` | `original` | -26.6250 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5000 | 420 | 1093 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `triangle` | `natural_question` | `original` | -18.6250 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | 245 | 121 | `[" ", " [", " geometry", " Polygon", " A"]` |
| deepseek7b | `triangle` | `natural_question` | `half` | -18.6250 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | 216 | 97 | `[" ", " Polygon", " polygon", " [", " geometry"]` |
| deepseek7b | `triangle` | `object_only` | `original` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -1.0312 | 1431 | 747 | `[" three", " \"", " a", " triangle", " acute"]` |
| deepseek7b | `triangle` | `object_only` | `half` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -1.0000 | 1206 | 666 | `[" three", " \"", " triangle", " a", " acute"]` |
| deepseek7b | `triangle` | `object_only` | `zero` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -0.9062 | 1022 | 590 | `[" three", " triangle", " \"", " a", " polygon"]` |
| deepseek7b | `triangle` | `object_only` | `flip` | -20.7500 | `unknown_other` | triangle is a polygon with three sides | `unknown_other` | 0 | 0 | -0.7812 | 740 | 435 | `[" triangle", " polygon", " three", " \"", " a"]` |
