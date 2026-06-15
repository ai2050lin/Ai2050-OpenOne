# Phase 146 Cross-model Template Router Token Gap Summary

## qwen3

families=['long', 'neutral']; splits=['front_back']; layers=[0, -1]; sites=['input_answer', 'attention_output', 'mlp_input']; scales=[0.25, 0.5, 1.0]

| category | cases | clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax | common_best_paths |
|---|---|---|---|---|---|---|---|
| plant | 2 | 0.00 | +15.47 | +12.52 | 44062.7 | 0.00 | L36 mlp_input s1.0(1); L36 mlp_input s0.5(1) |
| time | 2 | 0.00 | +39.81 | +12.81 | 73654.5 | 0.00 | L36 mlp_input s1.0(1); L36 mlp_input s0.5(1) |

| case | best path | T | R | rec | clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|
| front_back:long:plant | L36 mlp_input s1.0 | +7.87 | +12.29 | +19.34 | False | 74352.3 | 0.00 |
| front_back:long:time | L36 mlp_input s1.0 | +10.27 | +12.45 | +64.78 | False | 58110.7 | 0.00 |
| front_back:neutral:plant | L36 mlp_input s0.5 | +11.96 | +12.76 | +11.59 | False | 13773.1 | 0.00 |
| front_back:neutral:time | L36 mlp_input s0.5 | +10.71 | +13.16 | +14.84 | False | 89198.4 | 0.00 |

## glm4

families=['long', 'neutral']; splits=['front_back']; layers=[0, -1]; sites=['input_answer', 'attention_output', 'mlp_input']; scales=[0.25, 0.5, 1.0]

| category | cases | clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax | common_best_paths |
|---|---|---|---|---|---|---|---|
| plant | 2 | 1.00 | +1.53 | +0.22 | 1210.8 | 0.00 | L40 attention_output s0.25(1); L40 input_answer s0.25(1) |
| time | 2 | 0.00 | +1.11 | +1.40 | 9304.6 | 0.00 | L39 mlp_input s0.25(1); L40 mlp_input s1.0(1) |

| case | best path | T | R | rec | clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|
| front_back:long:plant | L40 attention_output s0.25 | +0.06 | +0.25 | +2.14 | True | 1151.9 | 0.00 |
| front_back:long:time | L39 mlp_input s0.25 | +0.18 | +0.92 | -0.68 | False | 12982.4 | 0.00 |
| front_back:neutral:plant | L40 input_answer s0.25 | -0.02 | +0.19 | +0.92 | True | 1269.8 | 0.00 |
| front_back:neutral:time | L40 mlp_input s1.0 | +0.38 | +1.88 | +2.90 | False | 5626.9 | 0.00 |

## deepseek7b

families=['long', 'short', 'neutral']; splits=['front_back', 'back_front']; layers=[0, -1]; sites=['input_answer', 'attention_output', 'mlp_input']; scales=[0.2, 0.25, 0.3, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]

| category | cases | clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax | common_best_paths |
|---|---|---|---|---|---|---|---|
| container | 6 | 1.00 | +1.07 | +0.16 | 19227.4 | 0.00 | L28 attention_output s0.75(3); L28 mlp_input s0.35(1); L28 input_answer s1.0(1); L28 input_answer s0.35(1) |
| number | 6 | 0.50 | +0.84 | +1.04 | 9589.2 | 0.00 | L28 attention_output s1.5(4); L28 attention_output s0.3(1); L28 input_answer s1.5(1) |
| plant | 6 | 1.00 | +0.66 | +0.03 | 3178.8 | 0.00 | L28 input_answer s1.0(2); L28 attention_output s0.35(1); L28 input_answer s0.5(1); L28 attention_output s0.5(1) |
| time | 6 | 1.00 | +0.63 | +0.10 | 17927.9 | 0.00 | L28 attention_output s0.75(1); L27 attention_output s0.5(1); L28 input_answer s0.75(1); L28 attention_output s0.25(1) |

| case | best path | T | R | rec | clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|
| back_front:long:container | L28 input_answer s0.35 | -0.85 | +0.19 | +0.53 | True | 7062.7 | 0.00 |
| back_front:long:number | L28 attention_output s0.3 | -0.45 | +0.09 | +0.75 | True | 9116.7 | 0.00 |
| back_front:long:plant | L28 attention_output s0.5 | -1.10 | +0.00 | +0.52 | True | 4416.0 | 0.00 |
| back_front:long:time | L28 attention_output s0.25 | -0.63 | +0.08 | +0.66 | True | 18184.0 | 0.00 |
| back_front:neutral:container | L28 attention_output s0.75 | -0.58 | +0.00 | +0.81 | True | 4908.4 | 0.00 |
| back_front:neutral:number | L28 input_answer s1.5 | -1.70 | +0.00 | +0.73 | True | 3382.9 | 0.00 |
| back_front:neutral:plant | L28 input_answer s1.0 | -1.06 | +0.10 | +0.77 | True | 1753.5 | 0.00 |
| back_front:neutral:time | L28 mlp_input s1.5 | -2.59 | +0.05 | +0.65 | True | 34850.0 | 0.00 |
| back_front:short:container | L28 attention_output s0.75 | +0.51 | +0.22 | +1.16 | True | 5844.8 | 0.00 |
| back_front:short:number | L28 attention_output s1.5 | -0.14 | +1.87 | +0.97 | False | 17303.3 | 0.00 |
| back_front:short:plant | L28 attention_output s0.75 | -2.02 | +0.00 | +0.57 | True | 4611.0 | 0.00 |
| back_front:short:time | L28 attention_output s0.5 | -2.46 | +0.00 | +0.50 | True | 8585.0 | 0.00 |
| front_back:long:container | L28 mlp_input s0.35 | +1.14 | +0.17 | +2.14 | True | 85809.7 | 0.00 |
| front_back:long:number | L28 attention_output s1.5 | -0.36 | +1.36 | +0.73 | False | 19943.8 | 0.00 |
| front_back:long:plant | L28 attention_output s0.35 | -0.39 | +0.00 | +0.66 | True | 4337.8 | 0.00 |
| front_back:long:time | L28 attention_output s0.75 | -0.43 | +0.24 | +0.65 | True | 31600.6 | 0.00 |
| front_back:neutral:container | L28 attention_output s0.75 | -1.06 | +0.23 | +0.73 | True | 6278.2 | 0.00 |
| front_back:neutral:number | L28 attention_output s1.5 | +0.37 | +2.90 | +1.08 | False | 3077.0 | 0.00 |
| front_back:neutral:plant | L28 input_answer s0.5 | -1.44 | +0.00 | +0.66 | True | 1434.6 | 0.00 |
| front_back:neutral:time | L28 input_answer s0.75 | -2.89 | +0.00 | +0.55 | True | 9878.4 | 0.00 |
| front_back:short:container | L28 input_answer s1.0 | +0.42 | +0.18 | +1.07 | True | 5460.3 | 0.00 |
| front_back:short:number | L28 attention_output s1.5 | -1.15 | +0.00 | +0.77 | True | 4711.5 | 0.00 |
| front_back:short:plant | L28 input_answer s1.0 | -1.48 | +0.10 | +0.76 | True | 2520.0 | 0.00 |
| front_back:short:time | L27 attention_output s0.5 | -1.04 | +0.24 | +0.75 | True | 4469.3 | 0.00 |

