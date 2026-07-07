# Phase 236 Pattern Family Behavior Benchmark

models: qwen3, glm4, deepseek7b
case_rows: 132
observation_rows: 1056
mean_behavior_score: 0.6462
pattern_match_rate: 0.6288

## Model Summary

| model | cases | mean score | match rate | drift types |
| --- | ---: | ---: | ---: | --- |
| qwen3 | 44 | 0.7307 | 0.6591 | {'none': 29, 'wrong_or_missing_target': 8, 'over_generation': 7} |
| glm4 | 44 | 0.6580 | 0.6136 | {'none': 27, 'wrong_or_missing_target': 11, 'over_generation': 6} |
| deepseek7b | 44 | 0.5500 | 0.6136 | {'none': 27, 'wrong_or_missing_target': 11, 'over_generation': 6} |
