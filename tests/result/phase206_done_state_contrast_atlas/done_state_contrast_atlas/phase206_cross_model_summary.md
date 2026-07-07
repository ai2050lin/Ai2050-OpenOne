# Phase 206 done-state contrast atlas

## Trajectory Contrast
| model | stop rule | mode | protocol | rows | model stop | task stop | period | continued | external stop | avg steps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | none | natural | plain | 84 | 0 | 7 | 36 | 29 | 0 | 8.00 |
| qwen3 | none | natural | short_answer | 84 | 0 | 0 | 56 | 56 | 0 | 8.00 |
| qwen3 | none | natural | stop_explicit | 84 | 0 | 2 | 68 | 66 | 0 | 8.00 |
| qwen3 | none | post_answer | plain | 84 | 0 | 1 | 51 | 50 | 0 | 8.00 |
| qwen3 | none | post_answer | short_answer | 84 | 0 | 0 | 78 | 78 | 0 | 8.00 |
| qwen3 | none | post_answer | stop_explicit | 84 | 0 | 2 | 84 | 82 | 0 | 8.00 |
| qwen3 | period | natural | plain | 84 | 0 | 37 | 37 | 0 | 37 | 6.20 |
| qwen3 | period | natural | short_answer | 84 | 0 | 56 | 56 | 0 | 56 | 4.55 |
| qwen3 | period | natural | stop_explicit | 84 | 0 | 68 | 68 | 0 | 68 | 4.19 |
| qwen3 | period | post_answer | plain | 84 | 0 | 51 | 51 | 0 | 51 | 4.46 |
| qwen3 | period | post_answer | short_answer | 84 | 0 | 78 | 78 | 0 | 78 | 1.71 |
| qwen3 | period | post_answer | stop_explicit | 84 | 0 | 84 | 84 | 0 | 84 | 1.65 |
| glm4 | none | natural | plain | 67 | 0 | 0 | 13 | 13 | 0 | 8.00 |
| glm4 | none | natural | short_answer | 67 | 0 | 1 | 63 | 62 | 0 | 8.00 |
| glm4 | none | natural | stop_explicit | 67 | 0 | 10 | 32 | 22 | 0 | 8.00 |
| glm4 | none | post_answer | plain | 67 | 0 | 0 | 5 | 5 | 0 | 8.00 |
| glm4 | none | post_answer | short_answer | 67 | 0 | 0 | 54 | 54 | 0 | 8.00 |
| glm4 | none | post_answer | stop_explicit | 67 | 0 | 0 | 67 | 67 | 0 | 8.00 |
| glm4 | period | natural | plain | 67 | 0 | 14 | 14 | 0 | 14 | 7.22 |
| glm4 | period | natural | short_answer | 67 | 0 | 62 | 62 | 0 | 62 | 5.40 |
| glm4 | period | natural | stop_explicit | 67 | 0 | 21 | 21 | 0 | 21 | 6.93 |
| glm4 | period | post_answer | plain | 67 | 0 | 9 | 9 | 0 | 9 | 7.58 |
| glm4 | period | post_answer | short_answer | 67 | 0 | 60 | 60 | 0 | 60 | 3.87 |
| glm4 | period | post_answer | stop_explicit | 67 | 0 | 67 | 67 | 0 | 67 | 1.48 |
| deepseek7b | none | natural | plain | 12 | 0 | 0 | 6 | 6 | 0 | 8.00 |
| deepseek7b | none | natural | short_answer | 12 | 0 | 0 | 12 | 12 | 0 | 8.00 |
| deepseek7b | none | natural | stop_explicit | 12 | 0 | 0 | 12 | 12 | 0 | 8.00 |
| deepseek7b | none | post_answer | plain | 12 | 0 | 1 | 9 | 8 | 0 | 8.00 |
| deepseek7b | none | post_answer | short_answer | 12 | 0 | 0 | 12 | 12 | 0 | 8.00 |
| deepseek7b | none | post_answer | stop_explicit | 12 | 0 | 0 | 10 | 10 | 0 | 8.00 |
| deepseek7b | period | natural | plain | 12 | 0 | 6 | 6 | 0 | 6 | 6.25 |
| deepseek7b | period | natural | short_answer | 12 | 0 | 12 | 12 | 0 | 12 | 2.83 |
| deepseek7b | period | natural | stop_explicit | 12 | 0 | 12 | 12 | 0 | 12 | 3.50 |
| deepseek7b | period | post_answer | plain | 12 | 0 | 9 | 9 | 0 | 9 | 6.33 |
| deepseek7b | period | post_answer | short_answer | 12 | 0 | 12 | 12 | 0 | 12 | 3.83 |
| deepseek7b | period | post_answer | stop_explicit | 12 | 0 | 10 | 10 | 0 | 10 | 4.33 |

## Forced Context Delta
| model | transition | protocol | rows | eos rank delta | prose rank delta | stop margin delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | after_answer->forced_eos | plain | 32 | -90991.25 | 4.94 | 3.81 |
| qwen3 | after_answer->forced_eos | short_answer | 32 | -82610.62 | 2.50 | 9.70 |
| qwen3 | after_answer->forced_eos | stop_explicit | 32 | -53150.38 | 10.56 | 10.93 |
| qwen3 | after_answer->forced_period | plain | 32 | -4876.25 | -3.69 | -13.64 |
| qwen3 | after_answer->forced_period | short_answer | 32 | -32599.25 | -9.31 | -18.27 |
| qwen3 | after_answer->forced_period | stop_explicit | 32 | -29609.81 | -8.81 | -19.61 |
| qwen3 | forced_period->forced_eos | plain | 32 | -86115.00 | 8.62 | 17.45 |
| qwen3 | forced_period->forced_eos | short_answer | 32 | -50011.38 | 11.81 | 27.98 |
| qwen3 | forced_period->forced_eos | stop_explicit | 32 | -23540.56 | 19.38 | 30.53 |
| glm4 | after_answer->forced_eos | plain | 32 | -2005.06 | -1.00 | -3.42 |
| glm4 | after_answer->forced_eos | short_answer | 32 | 4356.94 | -4.94 | -4.92 |
| glm4 | after_answer->forced_eos | stop_explicit | 32 | 3021.50 | -7.25 | -4.96 |
| glm4 | after_answer->forced_period | plain | 32 | -5596.62 | -3.25 | -5.41 |
| glm4 | after_answer->forced_period | short_answer | 32 | -1627.06 | -6.94 | -6.79 |
| glm4 | after_answer->forced_period | stop_explicit | 32 | -2943.00 | -9.25 | -5.22 |
| glm4 | forced_period->forced_eos | plain | 32 | 3591.56 | 2.25 | 1.99 |
| glm4 | forced_period->forced_eos | short_answer | 32 | 5984.00 | 2.00 | 1.88 |
| glm4 | forced_period->forced_eos | stop_explicit | 32 | 5964.50 | 2.00 | 0.25 |
| deepseek7b | after_answer->forced_eos | plain | 24 | 120208.67 | -0.50 | -6.05 |
| deepseek7b | after_answer->forced_eos | short_answer | 24 | 55022.67 | -5.00 | -7.02 |
| deepseek7b | after_answer->forced_eos | stop_explicit | 24 | 52310.50 | -4.83 | -7.27 |
| deepseek7b | after_answer->forced_period | plain | 24 | -1634.50 | 1.42 | 1.80 |
| deepseek7b | after_answer->forced_period | short_answer | 24 | -1172.33 | -4.50 | -6.70 |
| deepseek7b | after_answer->forced_period | stop_explicit | 24 | -1513.33 | -4.50 | -7.22 |
| deepseek7b | forced_period->forced_eos | plain | 24 | 121843.17 | -1.92 | -7.85 |
| deepseek7b | forced_period->forced_eos | short_answer | 24 | 56195.00 | -0.50 | -0.32 |
| deepseek7b | forced_period->forced_eos | stop_explicit | 24 | 53823.83 | -0.33 | -0.05 |

## State Summary
| model | state | stop rule | label | rows | eos rank | prose rank | stop margin |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | after_answer | forced_context | None | 96 | 75597.58 | 8.48 | 6.64 |
| qwen3 | after_continue1 | none | fail | 361 | 115744.13 | 102.53 | -0.44 |
| qwen3 | after_continue2 | none | fail | 350 | 112330.73 | 37.15 | -6.66 |
| qwen3 | after_period | none | fail | 361 | 49110.39 | 3.53 | -11.35 |
| qwen3 | after_period | none | success | 12 | 53431.58 | 4.67 | -7.35 |
| qwen3 | after_period | period | success | 374 | 49492.77 | 3.68 | -11.21 |
| qwen3 | before_answer | forced_context | None | 96 | 93321.38 | 3.25 | -3.59 |
| qwen3 | forced_eos | forced_context | forced_eos_success_proxy | 96 | 13.50 | 14.48 | 14.79 |
| qwen3 | forced_period | forced_context | None | 96 | 53235.81 | 1.21 | -10.53 |
| glm4 | after_answer | forced_context | None | 96 | 4322.29 | 7.73 | -0.65 |
| glm4 | after_continue1 | none | fail | 223 | 9600.62 | 49.53 | -1.71 |
| glm4 | after_continue2 | none | fail | 212 | 6050.64 | 20.81 | -2.31 |
| glm4 | after_period | none | fail | 223 | 4579.63 | 17.62 | -0.93 |
| glm4 | after_period | none | success | 11 | 17853.91 | 2.45 | -4.92 |
| glm4 | after_period | period | success | 233 | 10944.76 | 45.63 | -1.74 |
| glm4 | before_answer | forced_context | None | 96 | 6142.50 | 801.62 | 1.60 |
| glm4 | forced_eos | forced_context | forced_eos_success_proxy | 96 | 6113.42 | 3.33 | -5.08 |
| glm4 | forced_period | forced_context | None | 96 | 933.40 | 1.25 | -6.46 |
| deepseek7b | after_answer | forced_context | None | 72 | 2097.67 | 6.69 | -1.41 |
| deepseek7b | after_continue1 | none | fail | 60 | 8614.40 | 51.38 | -4.66 |
| deepseek7b | after_continue2 | none | fail | 57 | 4686.84 | 3.42 | -7.24 |
| deepseek7b | after_period | none | fail | 60 | 824.25 | 2.42 | -7.33 |
| deepseek7b | after_period | none | success | 1 | 1092.00 | 11.00 | -8.12 |
| deepseek7b | after_period | period | success | 61 | 827.38 | 2.59 | -7.34 |
| deepseek7b | before_answer | forced_context | None | 72 | 10168.44 | 248.06 | 1.66 |
| deepseek7b | forced_eos | forced_context | forced_eos_success_proxy | 72 | 77944.94 | 3.25 | -8.19 |
| deepseek7b | forced_period | forced_context | None | 72 | 657.61 | 4.17 | -5.45 |
