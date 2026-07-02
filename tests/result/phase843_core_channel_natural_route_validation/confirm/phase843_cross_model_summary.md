# Phase 843 Core Channel Natural Route Validation (confirm)

- Source: Phase 842 core channel candidate.
- Boundary: natural activation + first-step channel edit; not global closure.

## Model Summary

| model | skipped | rows | cases | original target | target | lost vs original | gained vs original |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 60 | 5 | 6 | 28 | 1 | 5 |

## Mode Summary

| model | mode | n | target | lost | gained | object_echo | unknown | mean act | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip` | 15 | 8 | 0 | 2 | 3 | 4 | -17.8292 | -0.6854 | `{"object_echo": 3, "target_equivalent": 8, "unknown_other": 4}` |
| deepseek7b | `half` | 15 | 7 | 0 | 1 | 2 | 6 | -17.8292 | -0.8427 | `{"object_echo": 2, "target_equivalent": 7, "unknown_other": 6}` |
| deepseek7b | `original` | 15 | 6 | 0 | 0 | 2 | 7 | -17.8292 | -0.8812 | `{"object_echo": 2, "target_equivalent": 6, "unknown_other": 7}` |
| deepseek7b | `zero` | 15 | 7 | 1 | 2 | 3 | 5 | -17.8292 | -0.7833 | `{"object_echo": 3, "target_equivalent": 7, "unknown_other": 5}` |

## Object Summary

| model | object | n | target | lost | gained | mean act | mean abs act | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `circle` | 12 | 3 | 1 | 0 | -10.3542 | 10.3542 | `{"target_equivalent": 3, "unknown_other": 9}` |
| deepseek7b | `polygon` | 12 | 8 | 0 | 0 | -23.1667 | 23.1667 | `{"target_equivalent": 8, "unknown_other": 4}` |
| deepseek7b | `rectangle` | 12 | 4 | 0 | 0 | -19.4583 | 19.4583 | `{"object_echo": 6, "target_equivalent": 4, "unknown_other": 2}` |
| deepseek7b | `square` | 12 | 7 | 0 | 3 | -14.1667 | 14.1667 | `{"object_echo": 4, "target_equivalent": 7, "unknown_other": 1}` |
| deepseek7b | `triangle` | 12 | 6 | 0 | 2 | -22.0000 | 22.0000 | `{"target_equivalent": 6, "unknown_other": 6}` |

## Top Rows

| model | object | prompt | mode | act | class | output | orig class | lost | gained | target-object | target rank | object rank | top tokens |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| deepseek7b | `circle` | `natural_category` | `zero` | -10.6250 | `unknown_other` | ? | `target_equivalent` | 1 | 0 | -1.8125 | 2490 | 677 | `[" ?\n", " shape", " ?\n\n", " geometry", " ["]` |
| deepseek7b | `square` | `natural_question` | `flip` | -20.3750 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | 1.2500 | 142 | 292 | `[" polygon", " Polygon", " polygons", " ", " shape"]` |
| deepseek7b | `triangle` | `natural_question` | `zero` | -18.6250 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0625 | 176 | 89 | `[" Polygon", " polygon", " ", " triangle", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `flip` | -18.6250 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | -0.9375 | 130 | 73 | `[" polygon", " Polygon", " polygons", " triangle", " "]` |
| deepseek7b | `square` | `natural_question` | `zero` | -20.3750 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.6562 | 193 | 296 | `[" Polygon", " polygon", " ", " polygons", " shape"]` |
| deepseek7b | `square` | `natural_question` | `half` | -20.3750 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | 0.4375 | 223 | 303 | `[" Polygon", " polygon", " ", " shape", " square"]` |
| deepseek7b | `rectangle` | `object_only` | `original` | -19.6250 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.2188 | 4173 | 567 | `[" rectangle", " \"", " Rectangle", " A", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `half` | -19.6250 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.2188 | 3643 | 485 | `[" rectangle", " Rectangle", " \"", " A", " The"]` |
| deepseek7b | `rectangle` | `object_only` | `zero` | -19.6250 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.1562 | 3121 | 423 | `[" rectangle", " Rectangle", " \"", " The", " A"]` |
| deepseek7b | `rectangle` | `object_only` | `flip` | -19.6250 | `object_echo` | rectangle | `object_echo` | 0 | 0 | -3.1250 | 2365 | 322 | `[" rectangle", " Rectangle", " \"", " The", " A"]` |
| deepseek7b | `circle` | `object_only` | `original` | -9.6875 | `unknown_other` | a round shape with no beginning or end | `unknown_other` | 0 | 0 | -3.0625 | 9284 | 1463 | `[" a", " A", " \"", " The", " Round"]` |
| deepseek7b | `circle` | `object_only` | `half` | -9.6875 | `unknown_other` | a round shape with no beginning or end | `unknown_other` | 0 | 0 | -2.9844 | 8590 | 1435 | `[" a", " A", " \"", " The", " circle"]` |
| deepseek7b | `circle` | `object_only` | `zero` | -9.6875 | `unknown_other` | a round shape with no beginning or end | `unknown_other` | 0 | 0 | -2.9062 | 7961 | 1397 | `[" a", " A", " \"", " The", " Round"]` |
| deepseek7b | `circle` | `object_only` | `flip` | -9.6875 | `unknown_other` | a round shape with no beginning or end | `unknown_other` | 0 | 0 | -2.7500 | 6614 | 1314 | `[" a", " A", " \"", " The", " circle"]` |
| deepseek7b | `circle` | `natural_question` | `flip` | -10.7500 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.3125 | 778 | 158 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `rectangle` | `natural_question` | `original` | -20.5000 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1875 | 532 | 126 | `[" square", " ", " A", " Rectangle", " shape"]` |
| deepseek7b | `circle` | `natural_question` | `zero` | -10.7500 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.1875 | 770 | 178 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `rectangle` | `natural_question` | `half` | -20.5000 | `unknown_other` | square | `unknown_other` | 0 | 0 | -2.1562 | 441 | 110 | `[" square", " Rectangle", " ", " A", " shape"]` |
| deepseek7b | `circle` | `natural_question` | `half` | -10.7500 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0938 | 754 | 185 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `rectangle` | `natural_question` | `zero` | -20.5000 | `object_echo` | Rectangle | `unknown_other` | 0 | 0 | -2.0625 | 342 | 95 | `[" Rectangle", " square", " ", " A", " rectangle"]` |
| deepseek7b | `rectangle` | `natural_question` | `flip` | -20.5000 | `object_echo` | Rectangle | `unknown_other` | 0 | 0 | -2.0625 | 241 | 71 | `[" Rectangle", " rectangle", " polygon", " Polygon", " square"]` |
| deepseek7b | `circle` | `natural_question` | `original` | -10.7500 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -2.0000 | 732 | 189 | `[" ", " shape", " A", " [", " circle"]` |
| deepseek7b | `circle` | `natural_category` | `flip` | -10.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.9062 | 2484 | 640 | `[" shape", " ?\n", " ?\n\n", " [", " geometry"]` |
| deepseek7b | `triangle` | `natural_category` | `flip` | -26.6250 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.8125 | 187 | 554 | `[" polygon", " geometry", " Polygon", " shape", " polygons"]` |
| deepseek7b | `circle` | `natural_category` | `half` | -10.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.7500 | 2474 | 703 | `[" shape", " ?\n", " ?\n\n", " [", " geometry"]` |
| deepseek7b | `square` | `object_only` | `original` | -9.5000 | `object_echo` | square | `object_echo` | 0 | 0 | -1.7188 | 4716 | 1623 | `[" square", " four", " a", " A", " "]` |
| deepseek7b | `triangle` | `natural_category` | `zero` | -26.6250 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 1.6875 | 265 | 757 | `[" polygon", " geometry", " shape", " Geometry", " ?\n"]` |
| deepseek7b | `circle` | `natural_category` | `original` | -10.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | -1.6875 | 2430 | 715 | `[" shape", " ?\n", " ?\n\n", " [", " geometry"]` |
| deepseek7b | `square` | `natural_category` | `flip` | -12.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.6250 | 873 | 2752 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `object_only` | `half` | -9.5000 | `object_echo` | square | `object_echo` | 0 | 0 | -1.6250 | 4354 | 1592 | `[" square", " four", " a", " A", " "]` |
| deepseek7b | `triangle` | `natural_category` | `half` | -26.6250 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5938 | 330 | 903 | `[" geometry", " polygon", " shape", " Geometry", " ?\n"]` |
| deepseek7b | `square` | `object_only` | `zero` | -9.5000 | `object_echo` | square | `object_echo` | 0 | 0 | -1.5312 | 4080 | 1588 | `[" square", " four", " a", " A", " "]` |
| deepseek7b | `triangle` | `natural_category` | `original` | -26.6250 | `target_equivalent` | geometry shape | `target_equivalent` | 0 | 0 | 1.5000 | 420 | 1093 | `[" geometry", " shape", " polygon", " Geometry", " ?\n"]` |
| deepseek7b | `square` | `natural_category` | `zero` | -12.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.3438 | 1101 | 2859 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `object_only` | `flip` | -9.5000 | `object_echo` | square | `object_echo` | 0 | 0 | -1.3438 | 3575 | 1568 | `[" square", " four", " a", " A", " "]` |
| deepseek7b | `triangle` | `natural_question` | `original` | -18.6250 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | 245 | 121 | `[" ", " [", " geometry", " Polygon", " A"]` |
| deepseek7b | `triangle` | `natural_question` | `half` | -18.6250 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | 216 | 97 | `[" ", " Polygon", " polygon", " [", " geometry"]` |
| deepseek7b | `square` | `natural_category` | `half` | -12.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.1875 | 1208 | 2845 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `natural_category` | `original` | -12.6250 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 1.0938 | 1338 | 2932 | `[" shape", " geometry", " ?\n", " ?\n\n", " Geometry"]` |
| deepseek7b | `triangle` | `object_only` | `original` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -1.0312 | 1431 | 747 | `[" three", " \"", " a", " triangle", " acute"]` |
| deepseek7b | `triangle` | `object_only` | `half` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -1.0000 | 1206 | 666 | `[" three", " \"", " triangle", " a", " acute"]` |
| deepseek7b | `triangle` | `object_only` | `zero` | -20.7500 | `unknown_other` | three-sided figure | `unknown_other` | 0 | 0 | -0.9062 | 1022 | 590 | `[" three", " triangle", " \"", " a", " polygon"]` |
| deepseek7b | `triangle` | `object_only` | `flip` | -20.7500 | `unknown_other` | triangle is a polygon with three sides | `unknown_other` | 0 | 0 | -0.7812 | 740 | 435 | `[" triangle", " polygon", " three", " \"", " a"]` |
| deepseek7b | `rectangle` | `natural_category` | `flip` | -18.2500 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.2500 | 607 | 713 | `[" shape", " ?\n", " geometry", " polygon", " ?\n\n"]` |
| deepseek7b | `rectangle` | `natural_category` | `zero` | -18.2500 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1875 | 823 | 923 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `square` | `natural_question` | `original` | -20.3750 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | 0.1562 | 271 | 310 | `[" ", " Polygon", " polygon", " shape", " square"]` |
| deepseek7b | `rectangle` | `natural_category` | `half` | -18.2500 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1562 | 959 | 1066 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `rectangle` | `natural_category` | `original` | -18.2500 | `target_equivalent` | shape | `target_equivalent` | 0 | 0 | 0.1250 | 1109 | 1231 | `[" shape", " ?\n", " geometry", " ?\n\n", " Geometry"]` |
| deepseek7b | `polygon` | `object_only` | `original` | -27.0000 | `target_equivalent` | polygon is a shape with straight sides | `target_equivalent` | 0 | 0 | 0.0000 | 931 | 931 | `[" polygon", " a", " A", " closed", " \""]` |
| deepseek7b | `polygon` | `object_only` | `zero` | -27.0000 | `target_equivalent` | polygon is a shape with straight sides | `target_equivalent` | 0 | 0 | 0.0000 | 626 | 626 | `[" polygon", " a", " closed", " A", " Polygon"]` |
| deepseek7b | `polygon` | `object_only` | `flip` | -27.0000 | `target_equivalent` | polygon is a shape with straight sides | `target_equivalent` | 0 | 0 | 0.0000 | 400 | 400 | `[" polygon", " Polygon", " a", " closed", " A"]` |
| deepseek7b | `polygon` | `object_only` | `half` | -27.0000 | `target_equivalent` | polygon is a shape with straight sides | `target_equivalent` | 0 | 0 | 0.0000 | 763 | 763 | `[" polygon", " a", " closed", " A", " \""]` |
| deepseek7b | `polygon` | `natural_question` | `original` | -21.5000 | `target_equivalent` | Polygon | `target_equivalent` | 0 | 0 | 0.0000 | 173 | 173 | `[" Polygon", " polygon", " ", " A", " polygons"]` |
| deepseek7b | `polygon` | `natural_question` | `zero` | -21.5000 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 0.0000 | 127 | 127 | `[" polygon", " Polygon", " polygons", " ", " A"]` |
| deepseek7b | `polygon` | `natural_question` | `flip` | -21.5000 | `target_equivalent` | polygon | `target_equivalent` | 0 | 0 | 0.0000 | 86 | 86 | `[" polygon", " Polygon", " polygons", " ", " Poly"]` |
| deepseek7b | `polygon` | `natural_question` | `half` | -21.5000 | `target_equivalent` | Polygon | `target_equivalent` | 0 | 0 | 0.0000 | 148 | 148 | `[" Polygon", " polygon", " ", " polygons", " A"]` |
| deepseek7b | `polygon` | `natural_category` | `original` | -21.0000 | `unknown_other` | geometry | `unknown_other` | 0 | 0 | 0.0000 | 1475 | 1475 | `[" geometry", " shape", " ?\n\n", " ?\n", " Geometry"]` |
| deepseek7b | `polygon` | `natural_category` | `zero` | -21.0000 | `unknown_other` | geometry | `unknown_other` | 0 | 0 | 0.0000 | 1011 | 1011 | `[" geometry", " shape", " ?\n\n", " ?\n", " Geometry"]` |
| deepseek7b | `polygon` | `natural_category` | `flip` | -21.0000 | `unknown_other` | geometry | `unknown_other` | 0 | 0 | 0.0000 | 729 | 729 | `[" geometry", " shape", " polygon", " ?\n\n", " ?\n"]` |
| deepseek7b | `polygon` | `natural_category` | `half` | -21.0000 | `unknown_other` | geometry | `unknown_other` | 0 | 0 | 0.0000 | 1217 | 1217 | `[" geometry", " shape", " ?\n\n", " ?\n", " Geometry"]` |
