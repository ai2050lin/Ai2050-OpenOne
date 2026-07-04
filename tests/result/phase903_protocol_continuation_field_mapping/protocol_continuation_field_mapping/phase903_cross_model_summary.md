# Phase 903 protocol continuation field mapping

## Overall

- models: qwen3, glm4, deepseek7b
- component_next_top_changed: 417
- component_protocol_logit_reduced: 2309
- component_protocol_logit_reduced_strong: 873
- component_protocol_rank1_removed: 66
- component_rows: 4504
- component_stop_rank_improved: 738
- selected_answer_drift_rows: 68
- state_rows: 68

## State Priors

| model | rows | answer top categories | answer protocol categories | protocol top1 | stop top1 | stop top10 | median rank delta |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | 18 | `{"newline": 11, "period": 7}` | `{"newline": 18}` | 11 | 7 | 18 | -5.0 |
| glm4 | 17 | `{"newline": 16, "other": 1}` | `{"newline": 17}` | 16 | 0 | 12 | -38.0 |
| deepseek7b | 33 | `{"comma": 20, "newline": 13}` | `{"comma": 18, "newline": 15}` | 33 | 0 | 31 | -2.0 |

## Component Summaries

| model | component rows | reduced | strong reduced | rank1 removed | top changed | stop improved | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 1296 | 575 | 147 | 1 | 101 | 140 | protocol_field_has_layer_component_sources |
| glm4 | 1360 | 711 | 173 | 13 | 36 | 307 | protocol_field_has_layer_component_sources |
| deepseek7b | 1848 | 1023 | 553 | 52 | 280 | 291 | protocol_field_has_layer_component_sources |

## Top Components

| model | layer | kind | category | rows | strong | removed | mean delta | top changed | stop improved |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | 27 | attention | comma | 18 | 18 | 11 | -8.663194444444445 | 11 | 0 |
| deepseek7b | 0 | attention | comma | 18 | 18 | 6 | -8.48611111111111 | 10 | 0 |
| qwen3 | 35 | mlp | newline | 18 | 18 | 0 | -12.159722222222221 | 4 | 0 |
| qwen3 | 35 | attention | newline | 18 | 18 | 0 | -2.2222222222222223 | 1 | 3 |
| deepseek7b | 25 | attention | comma | 18 | 18 | 0 | -1.3645833333333333 | 0 | 1 |
| deepseek7b | 16 | mlp | comma | 18 | 17 | 4 | -1.2013888888888888 | 4 | 0 |
| glm4 | 39 | mlp | newline | 17 | 17 | 0 | -2.2683823529411766 | 0 | 2 |
| deepseek7b | 23 | attention | comma | 18 | 17 | 0 | -1.3055555555555556 | 0 | 0 |
| deepseek7b | 17 | mlp | comma | 18 | 16 | 0 | -0.9027777777777778 | 0 | 6 |
| deepseek7b | 16 | attention | comma | 18 | 16 | 0 | -0.6909722222222222 | 0 | 3 |
| deepseek7b | 27 | attention | newline | 15 | 15 | 6 | -7.689583333333333 | 9 | 0 |
| deepseek7b | 0 | attention | newline | 15 | 15 | 5 | -8.816666666666666 | 10 | 0 |
| deepseek7b | 25 | attention | newline | 15 | 15 | 1 | -1.1791666666666667 | 4 | 1 |
| deepseek7b | 27 | mlp | newline | 15 | 15 | 0 | -3.4541666666666666 | 5 | 9 |
| deepseek7b | 20 | mlp | newline | 15 | 14 | 0 | -0.8875 | 6 | 4 |
| deepseek7b | 27 | mlp | comma | 18 | 14 | 0 | -0.8263888888888888 | 0 | 2 |
| deepseek7b | 9 | mlp | comma | 18 | 14 | 0 | -0.6805555555555556 | 0 | 1 |
| deepseek7b | 23 | mlp | comma | 18 | 13 | 0 | -1.2152777777777777 | 0 | 0 |
| qwen3 | 34 | attention | newline | 18 | 13 | 0 | -0.7430555555555556 | 2 | 8 |
| deepseek7b | 10 | mlp | comma | 18 | 13 | 0 | -0.6076388888888888 | 0 | 4 |
| deepseek7b | 18 | mlp | newline | 15 | 12 | 0 | -0.8958333333333334 | 4 | 6 |
| deepseek7b | 22 | mlp | newline | 15 | 12 | 0 | -0.7166666666666667 | 4 | 7 |
| glm4 | 30 | mlp | newline | 17 | 12 | 0 | -0.5036764705882353 | 0 | 2 |
| deepseek7b | 23 | mlp | newline | 15 | 11 | 0 | -0.6416666666666667 | 5 | 6 |
| glm4 | 31 | attention | newline | 17 | 11 | 0 | -0.5919117647058824 | 0 | 0 |
| deepseek7b | 20 | mlp | comma | 18 | 11 | 0 | -0.5243055555555556 | 0 | 4 |
| glm4 | 4 | mlp | newline | 17 | 11 | 0 | -0.4338235294117647 | 0 | 3 |
| glm4 | 22 | attention | newline | 17 | 10 | 4 | -0.3860294117647059 | 4 | 5 |
| qwen3 | 32 | attention | newline | 18 | 10 | 0 | -0.4027777777777778 | 3 | 6 |
| glm4 | 0 | mlp | newline | 17 | 9 | 0 | -0.7242647058823529 | 1 | 8 |
| glm4 | 38 | attention | newline | 17 | 8 | 6 | -0.9522058823529411 | 7 | 2 |
| glm4 | 22 | mlp | newline | 17 | 7 | 2 | -0.5110294117647058 | 2 | 4 |
| glm4 | 0 | attention | newline | 17 | 7 | 1 | -0.6360294117647058 | 2 | 5 |
| glm4 | 5 | mlp | newline | 17 | 7 | 0 | -0.4411764705882353 | 0 | 2 |
| glm4 | 21 | mlp | newline | 17 | 7 | 0 | -0.40808823529411764 | 0 | 2 |
| glm4 | 25 | mlp | newline | 17 | 7 | 0 | -0.4007352941176471 | 1 | 11 |
| qwen3 | 25 | mlp | newline | 18 | 7 | 0 | -0.3888888888888889 | 2 | 6 |
| qwen3 | 21 | mlp | newline | 18 | 7 | 0 | -0.375 | 1 | 1 |
| glm4 | 27 | mlp | newline | 17 | 6 | 0 | -0.40808823529411764 | 0 | 2 |
| glm4 | 17 | mlp | newline | 17 | 6 | 0 | -0.3639705882352941 | 0 | 2 |

## Substitution Graph

```json
{
  "comma->list_word": 4,
  "comma->newline": 48,
  "comma->other": 21,
  "newline->comma": 73,
  "newline->newline": 195,
  "newline->other": 45,
  "other->newline": 20,
  "other->other": 2,
  "period->newline": 6,
  "period->other": 1,
  "period->period": 2
}
```
