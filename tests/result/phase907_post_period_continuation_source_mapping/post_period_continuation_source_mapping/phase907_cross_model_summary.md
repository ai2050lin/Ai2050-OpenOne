# Phase 907 post-period continuation source mapping

## Overall

- models: qwen3, glm4, deepseek7b
- eos_rank_improved: 2105
- eos_rank_improved_100: 1903
- eos_rank_improved_1000: 1306
- next_category_changed: 942
- next_top_changed: 1291
- patched_eos_top1: 0
- patched_eos_top10: 13
- patched_eos_top50: 17
- protocol_rank1_removed: 258
- rows: 4504

## Model Summaries

| model | rows | eos improved | eos improved 1000 | eos top50 | next category changed | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 1296 | 606 | 244 | 0 | 262 | post_period_component_improves_eos_but_not_near |
| glm4 | 1360 | 757 | 464 | 17 | 189 | post_period_component_can_make_eos_near |
| deepseek7b | 1848 | 742 | 598 | 0 | 491 | post_period_component_improves_eos_but_not_near |

## Top Components

| model | layer | kind | base cat | rows | eos top50 | eos improved 1000 | next cat changed | mean eos rank delta |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| glm4 | 0 | attention | field_word | 8 | 8 | 8 | 8 | -18323.375 |
| glm4 | 0 | attention | other | 9 | 7 | 7 | 8 | -22756.444444444445 |
| glm4 | 38 | mlp | other | 9 | 2 | 7 | 3 | -15921.222222222223 |
| deepseek7b | 23 | mlp | other | 22 | 0 | 21 | 7 | -6409.590909090909 |
| deepseek7b | 27 | mlp | other | 22 | 0 | 21 | 5 | -17870.136363636364 |
| deepseek7b | 20 | mlp | other | 22 | 0 | 21 | 12 | -8789.136363636364 |
| deepseek7b | 24 | mlp | other | 22 | 0 | 21 | 10 | -9481.454545454546 |
| deepseek7b | 19 | mlp | other | 22 | 0 | 19 | 8 | -5717.454545454545 |
| deepseek7b | 25 | mlp | other | 22 | 0 | 19 | 5 | -6605.181818181818 |
| deepseek7b | 1 | attention | other | 22 | 0 | 18 | 22 | -8384.318181818182 |
| deepseek7b | 26 | mlp | other | 22 | 0 | 17 | 1 | -14.0 |
| deepseek7b | 9 | attention | other | 22 | 0 | 15 | 4 | -2263.6363636363635 |
| deepseek7b | 18 | attention | other | 22 | 0 | 15 | 2 | -3820.2272727272725 |
| deepseek7b | 22 | mlp | other | 22 | 0 | 15 | 13 | -951.3181818181819 |
| deepseek7b | 8 | mlp | other | 22 | 0 | 13 | 1 | -1523.3636363636363 |
| deepseek7b | 6 | attention | other | 22 | 0 | 13 | 7 | -366.22727272727275 |
| deepseek7b | 0 | attention | other | 22 | 0 | 11 | 4 | 13344.0 |
| deepseek7b | 15 | attention | other | 22 | 0 | 11 | 2 | -186.9090909090909 |
| deepseek7b | 8 | attention | other | 22 | 0 | 10 | 2 | -1288.909090909091 |
| deepseek7b | 2 | mlp | other | 22 | 0 | 10 | 6 | -298.45454545454544 |
| deepseek7b | 13 | attention | other | 22 | 0 | 8 | 5 | 1153.8636363636363 |
| glm4 | 37 | attention | field_word | 8 | 0 | 8 | 2 | -6332.25 |
| glm4 | 39 | attention | field_word | 8 | 0 | 8 | 0 | -6834.0 |
| glm4 | 36 | mlp | field_word | 8 | 0 | 8 | 0 | -4249.75 |
| glm4 | 36 | attention | field_word | 8 | 0 | 8 | 0 | -4065.375 |
| glm4 | 35 | attention | field_word | 8 | 0 | 8 | 0 | -3846.25 |
| deepseek7b | 27 | attention | other | 22 | 0 | 7 | 1 | 9789.863636363636 |
| qwen3 | 34 | mlp | explanation | 9 | 0 | 7 | 9 | -28784.444444444445 |
| qwen3 | 25 | mlp | explanation | 9 | 0 | 7 | 3 | -9584.111111111111 |
| qwen3 | 14 | attention | explanation | 9 | 0 | 7 | 2 | -5852.444444444444 |
