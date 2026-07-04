# Phase 897 non-color route axis discovery

## Overall

- models: qwen3, glm4, deepseek7b
- activation_candidate_axes: 79
- candidate_axes: 84
- condition_rows: 420
- history_candidate_axes: 5
- known_axis_minimal_pair_conditions: 4
- no_single_pair_conditions: 8
- pair_closure_conditions: 49
- search_rows: 4200
- selected_conditions: 420
- single_axis_closure_conditions: 46

## Domain groups

| model | domain | U size | conditions | single closure | pair closure | no-single pair | known minimal pair | single keys | pair keys |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4 | material | 4 | 24 | 1 | 3 | 2 | 2 | {'L39C638': 1} | {'L39C638+L39C1630': 1, 'L39C638+L39C2682': 1} |
| deepseek7b | geometry | 4 | 20 | 5 | 6 | 1 | 1 | {'L24C3099': 1, 'L27C15207': 2, 'L27C15305': 3, 'L27C15791': 5} | {'L27C15791+L27C15305': 1} |
| qwen3 | animal | 4 | 24 | 4 | 4 | 1 | 1 | {'L31C9629': 3, 'L35C2290': 1} | {'L32C5295+L35C2290': 1} |
| qwen3 | geometry | 4 | 20 | 6 | 6 | 2 | 0 | {'L31C2414': 4, 'L34C6183': 2} | {} |
| qwen3 | material | 4 | 24 | 7 | 7 | 1 | 0 | {'L30C7222': 2, 'L31C2257': 7} | {} |
| glm4 | object | 4 | 16 | 1 | 2 | 1 | 0 | {'L39C11316': 1} | {} |
| deepseek7b | animal | 4 | 24 | 11 | 11 | 0 | 0 | {'L26C8587': 1, 'L27C15369': 2, 'L27C16651': 11} | {} |
| glm4 | animal | 4 | 24 | 4 | 4 | 0 | 0 | {'L35C8824': 4} | {} |
| glm4 | tool | 4 | 20 | 2 | 2 | 0 | 0 | {'L39C13470': 2} | {} |
| deepseek7b | material | 4 | 24 | 2 | 1 | 0 | 0 | {'L27C18590': 2} | {} |
| glm4 | plant | 4 | 16 | 1 | 1 | 0 | 0 | {'L36C4366': 1} | {} |
| deepseek7b | abstract | 4 | 20 | 1 | 1 | 0 | 0 | {'L27C1552': 1} | {} |
| deepseek7b | tool | 4 | 20 | 1 | 1 | 0 | 0 | {'L27C11093': 1} | {} |
| qwen3 | abstract | 4 | 20 | 0 | 0 | 0 | 0 | {} | {} |
| qwen3 | object | 4 | 16 | 0 | 0 | 0 | 0 | {} | {} |
| qwen3 | plant | 4 | 16 | 0 | 0 | 0 | 0 | {} | {} |
| qwen3 | tool | 4 | 20 | 0 | 0 | 0 | 0 | {} | {} |
| glm4 | abstract | 4 | 20 | 0 | 0 | 0 | 0 | {} | {} |
| glm4 | geometry | 4 | 20 | 0 | 0 | 0 | 0 | {} | {} |
| deepseek7b | object | 4 | 16 | 0 | 0 | 0 | 0 | {} | {} |
| deepseek7b | plant | 4 | 16 | 0 | 0 | 0 | 0 | {} | {} |

## Top single axes

| model | domain | subset | rows | closure | mean lift | mean blocker reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| deepseek7b | animal | L27C16651 | 24 | 11 | 2.135 | 0.583 |
| qwen3 | material | L31C2257 | 24 | 7 | 0.565 | 2.042 |
| deepseek7b | geometry | L27C15791 | 20 | 5 | -0.797 | 0.500 |
| qwen3 | geometry | L31C2414 | 20 | 4 | 0.463 | 0.200 |
| glm4 | animal | L35C8824 | 24 | 4 | 0.034 | 0.042 |
| qwen3 | animal | L31C9629 | 24 | 3 | -0.349 | -0.167 |
| deepseek7b | geometry | L27C15305 | 20 | 3 | -0.456 | -0.100 |
| qwen3 | geometry | L34C6183 | 20 | 2 | 1.031 | 0.050 |
| deepseek7b | material | L27C18590 | 24 | 2 | 0.401 | 1.833 |
| glm4 | tool | L39C13470 | 20 | 2 | 0.094 | 0.050 |
| deepseek7b | animal | L27C15369 | 24 | 2 | 0.018 | 0.000 |
| qwen3 | material | L30C7222 | 24 | 2 | -0.021 | -0.083 |
| deepseek7b | geometry | L27C15207 | 20 | 2 | -0.131 | 0.000 |
| deepseek7b | tool | L27C11093 | 20 | 1 | 0.222 | 0.100 |
| glm4 | plant | L36C4366 | 16 | 1 | 0.141 | 0.062 |
| deepseek7b | animal | L26C8587 | 24 | 1 | 0.016 | 0.125 |
| glm4 | object | L39C11316 | 16 | 1 | 0.002 | -0.562 |
| glm4 | material | L39C638 | 24 | 1 | -0.013 | -0.083 |
| deepseek7b | abstract | L27C1552 | 20 | 1 | -0.109 | -1.050 |
| deepseek7b | geometry | L24C3099 | 20 | 1 | -1.206 | -4.850 |
| qwen3 | animal | L35C2290 | 24 | 1 | -2.891 | 0.125 |
| qwen3 | object | L30C5438 | 16 | 0 | 0.387 | 8.812 |
| qwen3 | tool | L33C9689 | 20 | 0 | 0.369 | 0.100 |
| deepseek7b | plant | L27C1709 | 16 | 0 | 0.359 | -0.188 |
| deepseek7b | material | L27C3817 | 24 | 0 | 0.307 | -0.042 |
| qwen3 | object | L33C8825 | 16 | 0 | 0.266 | 7.625 |
| qwen3 | abstract | L35C18 | 20 | 0 | 0.253 | -1.850 |
| glm4 | abstract | L39C12772 | 20 | 0 | 0.177 | -0.950 |
| qwen3 | abstract | L35C219 | 20 | 0 | 0.144 | 2.850 |
| glm4 | abstract | L39C2948 | 20 | 0 | 0.128 | 0.550 |
| glm4 | animal | L39C2948 | 24 | 0 | 0.122 | 0.000 |
| glm4 | object | L39C5585 | 16 | 0 | 0.117 | 0.000 |
| glm4 | material | L39C2682 | 24 | 0 | 0.096 | -0.125 |
| deepseek7b | material | L27C13559 | 24 | 0 | 0.086 | 0.792 |
| glm4 | plant | L39C10302 | 16 | 0 | 0.070 | 0.000 |
| glm4 | tool | L35C13692 | 20 | 0 | 0.062 | 0.000 |
| deepseek7b | object | L25C15220 | 16 | 0 | 0.061 | 0.250 |
| glm4 | animal | L39C12338 | 24 | 0 | 0.060 | 0.000 |
| deepseek7b | object | L27C11125 | 16 | 0 | 0.040 | -17.750 |
| qwen3 | object | L30C8942 | 16 | 0 | 0.039 | 4.562 |

## Top pair axes

| model | domain | subset | rows | closure | no-single | known minimal | mean lift | mean blocker reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | geometry | L27C15791+L27C15305 | 20 | 6 | 1 | 1 | -1.253 | 0.350 |
| glm4 | material | L39C638+L39C1630 | 24 | 2 | 1 | 1 | 0.021 | 0.000 |
| qwen3 | animal | L32C5295+L35C2290 | 24 | 1 | 1 | 1 | -3.526 | 0.000 |
| glm4 | material | L39C638+L39C2682 | 24 | 1 | 1 | 1 | 0.081 | -0.292 |
| qwen3 | geometry | L31C3531+L35C935 | 20 | 2 | 2 | 0 | -1.269 | 0.050 |
| glm4 | object | L39C11316+L39C5585 | 16 | 2 | 1 | 0 | 0.113 | -0.125 |
| glm4 | object | L39C3652+L39C11316 | 16 | 2 | 1 | 0 | -0.205 | 0.125 |
| qwen3 | material | L30C8842+L30C7222 | 24 | 1 | 1 | 0 | -0.453 | -0.583 |
| deepseek7b | animal | L27C16651+L27C1851 | 24 | 11 | 0 | 0 | 2.135 | 0.583 |
| deepseek7b | animal | L27C16651+L26C8587 | 24 | 11 | 0 | 0 | 2.135 | 0.583 |
| deepseek7b | animal | L27C16651+L27C15369 | 24 | 11 | 0 | 0 | 2.133 | 0.583 |
| qwen3 | material | L31C2257+L30C7222 | 24 | 7 | 0 | 0 | 0.602 | 2.000 |
| qwen3 | material | L31C2257+L30C5799 | 24 | 7 | 0 | 0 | 0.542 | 1.958 |
| qwen3 | geometry | L31C2414+L34C6183 | 20 | 6 | 0 | 0 | 1.281 | 0.300 |
| qwen3 | material | L31C2257+L30C8842 | 24 | 5 | 0 | 0 | 0.190 | 1.417 |
| deepseek7b | geometry | L27C15791+L27C15207 | 20 | 5 | 0 | 0 | -0.925 | 0.650 |
| qwen3 | geometry | L31C3531+L31C2414 | 20 | 4 | 0 | 0 | -0.287 | 0.050 |
| qwen3 | geometry | L31C2414+L35C935 | 20 | 4 | 0 | 0 | -0.388 | 0.200 |
| glm4 | animal | L35C8824+L39C2948 | 24 | 4 | 0 | 0 | 0.146 | 0.000 |
| glm4 | animal | L39C12338+L35C8824 | 24 | 4 | 0 | 0 | 0.078 | 0.042 |
| qwen3 | animal | L35C2290+L31C9629 | 24 | 3 | 0 | 0 | -3.422 | -0.083 |
| deepseek7b | geometry | L27C15305+L27C15207 | 20 | 3 | 0 | 0 | -0.606 | 0.000 |
| qwen3 | geometry | L31C3531+L34C6183 | 20 | 2 | 0 | 0 | 0.812 | -0.200 |
| qwen3 | geometry | L34C6183+L35C935 | 20 | 2 | 0 | 0 | 0.287 | 0.100 |
| qwen3 | material | L30C5799+L30C7222 | 24 | 2 | 0 | 0 | -0.008 | 0.042 |
| qwen3 | animal | L32C5295+L31C9629 | 24 | 2 | 0 | 0 | -0.839 | -0.333 |
| glm4 | tool | L39C11316+L39C13470 | 20 | 2 | 0 | 0 | 0.050 | 0.000 |
| glm4 | animal | L35C8824+L39C664 | 24 | 2 | 0 | 0 | -0.286 | -0.250 |
| deepseek7b | animal | L27C1851+L27C15369 | 24 | 2 | 0 | 0 | 0.005 | 0.042 |
| deepseek7b | geometry | L24C3099+L27C15207 | 20 | 2 | 0 | 0 | -1.344 | -4.500 |
| deepseek7b | geometry | L27C15305+L24C3099 | 20 | 2 | 0 | 0 | -1.622 | -5.600 |
| deepseek7b | geometry | L27C15791+L24C3099 | 20 | 2 | 0 | 0 | -1.959 | -4.500 |
| glm4 | plant | L39C10302+L36C4366 | 16 | 1 | 0 | 0 | 0.250 | 0.062 |
| glm4 | plant | L39C3167+L36C4366 | 16 | 1 | 0 | 0 | 0.117 | 0.062 |
| deepseek7b | material | L27C7072+L27C18590 | 24 | 1 | 0 | 0 | 0.349 | 1.625 |
| deepseek7b | tool | L27C11093+L27C11730 | 20 | 1 | 0 | 0 | 0.231 | 0.100 |
| deepseek7b | animal | L27C15369+L26C8587 | 24 | 1 | 0 | 0 | 0.013 | 0.125 |
| deepseek7b | animal | L27C1851+L26C8587 | 24 | 1 | 0 | 0 | 0.010 | 0.125 |
| deepseek7b | abstract | L27C1552+L27C15801 | 20 | 1 | 0 | 0 | -0.241 | 0.050 |
| deepseek7b | abstract | L27C1109+L27C1552 | 20 | 1 | 0 | 0 | -0.411 | -2.850 |
