# Phase87 Reader Stack Calibration Summary

## qwen3

items=4, rows=20, relations=['is_a', 'part_of']

### By reader

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | closed | 4 | 0.7500 | 2.2246 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | choice | 12 | 0.0000 | 0.0000 | 0.6667 | 0.9167 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| 3 | open | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |

### By reader template

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | closed:fullseq_candidate | 4 | 0.7500 | 2.2246 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | choice:choice_plain | 12 | 0.0000 | 0.0000 | 0.6667 | 0.9167 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| 3 | open:open_fill_blank | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |

### By choice order

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | choice_plain:target_last | 4 | 0.0000 | 0.0000 | 0.7500 | 1.0000 | 0.7500 | 0.0000 | 0.0000 | 0.0000 |
| 2 | choice_plain:rotating | 4 | 0.0000 | 0.0000 | 0.7500 | 1.0000 | 0.7500 | 0.0000 | 0.0000 | 0.0000 |
| 3 | choice_plain:target_first | 4 | 0.0000 | 0.0000 | 0.5000 | 0.7500 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

### By relation

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | part_of | 10 | 1.0000 | 0.5688 | 0.8333 | 0.8333 | 0.8333 | 0.0000 | 0.0000 | 0.0000 |
| 2 | is_a | 10 | 0.5000 | 3.8803 | 0.5000 | 1.0000 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |

## Cross Model By Reader

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | closed | 4 | 0.7500 | 2.2246 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | choice | 12 | 0.0000 | 0.0000 | 0.6667 | 0.9167 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| 3 | open | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |

## Cross Model By Reader Template

| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_valid | target_letter_rate | open_subset | open_family | open_format_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | closed:fullseq_candidate | 4 | 0.7500 | 2.2246 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | choice:choice_plain | 12 | 0.0000 | 0.0000 | 0.6667 | 0.9167 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| 3 | open:open_fill_blank | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2500 |
