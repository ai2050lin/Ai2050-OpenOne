# Phase 899 domain axis rollout and protocol/object echo audit

## Overall

- models: qwen3, glm4, deepseek7b
- base_rows: 77
- component_rows: 42
- rollout_rows: 196
- selected_conditions: 77
- source_candidate_rows: 77
- sources: 13

## Relation summaries

| relation | rows | clear no protocol | class no echo no protocol | answer class | object echo | protocol drift | bad transition | labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base_original | 77 | 0 | 0 | 0 | 2 | 75 | 77 | {'format_or_empty': 5, 'object_echo': 2, 'other': 70} |
| component_single | 42 | 0 | 0 | 10 | 0 | 42 | 42 | {'answer_alias': 8, 'format_or_empty': 2, 'other': 30, 'strict_canonical': 2} |
| source_candidate_pair | 21 | 0 | 0 | 17 | 0 | 21 | 21 | {'answer_alias': 12, 'other': 4, 'strict_canonical': 5} |
| source_candidate_single | 56 | 0 | 0 | 51 | 1 | 56 | 56 | {'answer_alias': 13, 'object_echo': 1, 'other': 4, 'strict_canonical': 38} |

## Source summaries

| model | source | domain | subset | selected | source clean | source class-clean | source echo | source drift | gain | loss | labels |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | phase897_single_candidate | animal | L27C16651 | 24 | 0 | 0 | 0 | 24 | 0 | 0 | {'answer_alias': 2, 'other': 1, 'strict_canonical': 21} |
| glm4 | phase897_single_candidate | animal | L35C8824 | 12 | 0 | 0 | 1 | 12 | 0 | 0 | {'object_echo': 1, 'strict_canonical': 11} |
| qwen3 | phase897_single_candidate | material | L31C2257 | 9 | 0 | 0 | 0 | 9 | 0 | 0 | {'answer_alias': 1, 'other': 2, 'strict_canonical': 6} |
| deepseek7b | phase897_pair_candidate | geometry | L27C15791+L27C15305 | 7 | 0 | 0 | 0 | 7 | 0 | 0 | {'answer_alias': 6, 'other': 1} |
| qwen3 | phase897_single_candidate | geometry | L31C2414 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | {'answer_alias': 6} |
| qwen3 | phase897_pair_candidate | geometry | L31C3531+L35C935 | 5 | 0 | 0 | 0 | 5 | 0 | 0 | {'answer_alias': 1, 'other': 2, 'strict_canonical': 2} |
| deepseek7b | phase897_single_candidate | geometry | L27C15791 | 5 | 0 | 0 | 0 | 5 | 0 | 0 | {'answer_alias': 4, 'other': 1} |
| glm4 | phase897_pair_candidate | material | L39C638+L39C1630 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | {'other': 1, 'strict_canonical': 1} |
| glm4 | phase897_pair_candidate | object | L39C11316+L39C5585 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | {'answer_alias': 2} |
| glm4 | phase897_pair_candidate | object | L39C3652+L39C11316 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | {'answer_alias': 2} |
| qwen3 | phase897_pair_candidate | animal | L32C5295+L35C2290 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | {'strict_canonical': 1} |
| qwen3 | phase897_pair_candidate | material | L30C8842+L30C7222 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | {'strict_canonical': 1} |
| glm4 | phase897_pair_candidate | material | L39C638+L39C2682 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | {'answer_alias': 1} |
