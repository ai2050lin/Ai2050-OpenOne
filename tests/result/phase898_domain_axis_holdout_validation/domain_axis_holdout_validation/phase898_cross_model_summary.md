# Phase 898 domain axis holdout validation

## Overall

- models: qwen3, glm4, deepseek7b
- condition_rows: 840
- no_single_pair_conditions: 11
- pair_closure_conditions: 21
- rows: 1832
- single_axis_closure_conditions: 72
- source_candidate_closure_conditions: 78
- sources: 13

## Source summaries

| model | source | domain | subset | conditions | source closure | single closure | pair closure | no-single | single keys | pair keys |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| qwen3 | phase897_pair_candidate | geometry | L31C3531+L35C935 | 40 | 5 | 2 | 5 | 3 | {'L31C3531': 2} | {'L31C3531+L35C935': 3} |
| deepseek7b | phase897_pair_candidate | geometry | L27C15791+L27C15305 | 40 | 7 | 5 | 7 | 2 | {'L27C15305': 3, 'L27C15791': 5} | {'L27C15791+L27C15305': 2} |
| glm4 | phase897_pair_candidate | material | L39C638+L39C1630 | 88 | 2 | 1 | 2 | 1 | {'L39C638': 1} | {'L39C638+L39C1630': 1} |
| glm4 | phase897_pair_candidate | object | L39C11316+L39C5585 | 32 | 2 | 1 | 2 | 1 | {'L39C11316': 1} | {'L39C11316+L39C5585': 1} |
| glm4 | phase897_pair_candidate | object | L39C3652+L39C11316 | 32 | 2 | 1 | 2 | 1 | {'L39C11316': 1} | {'L39C3652+L39C11316': 1} |
| qwen3 | phase897_pair_candidate | animal | L32C5295+L35C2290 | 88 | 1 | 2 | 1 | 1 | {'L32C5295': 1, 'L35C2290': 1} | {'L32C5295+L35C2290': 1} |
| qwen3 | phase897_pair_candidate | material | L30C8842+L30C7222 | 88 | 1 | 2 | 1 | 1 | {'L30C7222': 2} | {'L30C8842+L30C7222': 1} |
| glm4 | phase897_pair_candidate | material | L39C638+L39C2682 | 88 | 1 | 1 | 1 | 1 | {'L39C638': 1} | {'L39C638+L39C2682': 1} |
| deepseek7b | phase897_single_candidate | animal | L27C16651 | 88 | 25 | 25 | 0 | 0 | {'L27C16651': 25} | {} |
| glm4 | phase897_single_candidate | animal | L35C8824 | 88 | 12 | 12 | 0 | 0 | {'L35C8824': 12} | {} |
| qwen3 | phase897_single_candidate | material | L31C2257 | 88 | 9 | 9 | 0 | 0 | {'L31C2257': 9} | {} |
| qwen3 | phase897_single_candidate | geometry | L31C2414 | 40 | 6 | 6 | 0 | 0 | {'L31C2414': 6} | {} |
| deepseek7b | phase897_single_candidate | geometry | L27C15791 | 40 | 5 | 5 | 0 | 0 | {'L27C15791': 5} | {} |
