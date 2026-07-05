# Phase 927 blocker-class split response surface audit

## Overall

- models: qwen3, glm4, deepseek7b
- article_a_best_coord_is_base: 8
- article_a_new_margin_closure_vs_surface_base: 2
- article_a_new_strict_vs_surface_base: 0
- article_a_new_top1_vs_surface_base: 2
- article_a_rows: 864
- article_a_selected_seeds: 18
- article_a_surfaces: 36
- article_a_top1: 97
- article_a_with_closure_coord: 6
- phase926_rows: 1440
- phase926_selected_seeds: 30
- phase926_surfaces: 60
- punctuation_period_best_coord_is_base: 0
- punctuation_period_new_margin_closure_vs_surface_base: 0
- punctuation_period_new_strict_vs_surface_base: 0
- punctuation_period_new_top1_vs_surface_base: 0
- punctuation_period_rows: 576
- punctuation_period_selected_seeds: 12
- punctuation_period_surfaces: 24
- punctuation_period_top1: 0
- punctuation_period_with_closure_coord: 0

## Evidence

- blocker_class_split_confirmed_article_closes_punctuation_moves_only: 1
- no_blocker_class_data: 2

## Class Summaries

| model | class | seeds | rows | surfaces | best base | top1 | new top1 | new margin | new strict | closure surfaces | best alpha | best protocol |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4 | article_a | 18 | 864 | 36 | 8 | 97 | 2 | 2 | 0 | 6 | {'1.25': 16, '1.0': 10, '1.375': 4, '1.125': 4, '0.75': 2} | {'1.0': 11, '1.1': 2, '0.9': 17, '0.85': 6} |
| glm4 | punctuation_period | 12 | 576 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | {'0.875': 9, '1.375': 1, '1.25': 8, '1.125': 2, '1.0': 4} | {'0.85': 7, '1.1': 16, '0.9': 1} |

## New Closure Rows

| model | class | state | case | domain | group | l39 | alpha | protocol | base margin | margin | strict |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | article_a | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.4 | p856_009_animal_fish | animal | band32_support_64 | 1.375 | 1.375 | 0.85 | -0.0625 | 0.0625 | False |
| glm4 | article_a | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.4 | p856_009_animal_fish | animal | band32_support_64 | 1.375 | 1.375 | 0.9 | -0.0625 | 0.0625 | False |
