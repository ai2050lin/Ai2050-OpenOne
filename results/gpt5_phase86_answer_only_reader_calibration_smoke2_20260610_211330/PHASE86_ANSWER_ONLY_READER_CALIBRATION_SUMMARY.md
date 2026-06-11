# Phase86 Answer-Only Reader Calibration Summary

## qwen3

items=4, rows=8, max_new_tokens=8, relations=['is_a', 'part_of']

### By template

| rank | key | n | exact | prefix | contains | word_subset | family_overlap | coverage | precision | short | format_violation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | answer_only_plain | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.5000 | 1.0000 | 0.2500 |
| 2 | question_value | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.5000 | 1.0000 | 0.5000 |

### By relation

| rank | key | n | exact | prefix | contains | word_subset | family_overlap | coverage | precision | short | format_violation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | is_a | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.2500 |
| 2 | part_of | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |

### Top template relation

| rank | key | n | exact | prefix | contains | word_subset | family_overlap | coverage | precision | short | format_violation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | question_value:is_a | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| 2 | answer_only_plain:is_a | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 0.5000 |
| 3 | answer_only_plain:part_of | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 4 | question_value:part_of | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## Cross Model Template Ranking

| rank | key | n | exact | prefix | contains | word_subset | family_overlap | coverage | precision | short | format_violation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | answer_only_plain | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.5000 | 1.0000 | 0.2500 |
| 2 | question_value | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 | 0.5000 | 1.0000 | 0.5000 |
