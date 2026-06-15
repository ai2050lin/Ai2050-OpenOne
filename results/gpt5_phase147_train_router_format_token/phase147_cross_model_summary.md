# Phase 147 Cross-model Train Router Format Token Summary

## qwen3

families=['long', 'neutral']; splits=['front_back']; formats=['plain', 'label_colon']; train/test=8/8

### By category

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| plant | 4 | 0.00 | +15.60 | +12.57 | 48998.9 | 0.00 |
| time | 4 | 0.00 | +85.09 | +13.20 | 85218.6 | 0.00 |

### By format

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| label_colon | 4 | 0.00 | +21.47 | +12.47 | 90267.9 | 0.00 |
| plain | 4 | 0.00 | +79.21 | +13.29 | 43949.7 | 0.00 |

### Cases

| case | train path | train clean | held T | held R | held rec | held clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | L36 mlp_input s1.0 | False | +10.91 | +11.31 | +28.53 | False | 117624.6 | 0.00 |
| front_back:long:label_colon:time | L36 mlp_input s1.0 | False | +9.36 | +12.86 | +36.64 | False | 127883.5 | 0.00 |
| front_back:long:plain:plant | L36 mlp_input s1.0 | False | +8.10 | +12.52 | +13.81 | False | 47785.1 | 0.00 |
| front_back:long:plain:time | L36 mlp_input s1.0 | False | +12.08 | +13.60 | +282.07 | False | 46485.5 | 0.00 |
| front_back:neutral:label_colon:plant | L36 mlp_input s1.0 | False | +11.86 | +13.01 | +9.79 | False | 19858.4 | 0.00 |
| front_back:neutral:label_colon:time | L36 mlp_input s1.0 | False | +10.51 | +12.72 | +10.93 | False | 95705.1 | 0.00 |
| front_back:neutral:plain:plant | L36 mlp_input s0.5 | False | +12.76 | +13.43 | +10.27 | False | 10727.6 | 0.00 |
| front_back:neutral:plain:time | L36 mlp_input s0.5 | False | +12.04 | +13.61 | +10.70 | False | 70800.4 | 0.00 |

## glm4

families=['long', 'neutral']; splits=['front_back']; formats=['plain', 'label_colon']; train/test=8/8

### By category

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| plant | 4 | 0.00 | +6.23 | +1.80 | 1956.5 | 0.00 |
| time | 4 | 0.00 | -0.16 | +1.35 | 4942.8 | 0.00 |

### By format

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| label_colon | 4 | 0.00 | +3.44 | +1.88 | 322.5 | 0.00 |
| plain | 4 | 0.00 | +2.63 | +1.27 | 6576.9 | 0.00 |

### Cases

| case | train path | train clean | held T | held R | held rec | held clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | L39 input_answer s1.0 | False | +0.06 | +2.11 | +1.54 | False | 249.1 | 0.00 |
| front_back:long:label_colon:time | L39 attention_output s1.0 | False | +1.29 | +1.89 | +0.06 | False | 144.1 | 0.00 |
| front_back:long:plain:plant | L40 input_answer s1.0 | False | +0.55 | +0.41 | +1.95 | False | 6868.8 | 0.00 |
| front_back:long:plain:time | L39 mlp_input s0.25 | False | +0.47 | +1.21 | -0.37 | False | 11378.9 | 0.00 |
| front_back:neutral:label_colon:plant | L39 mlp_input s1.0 | False | +1.20 | +2.39 | +10.73 | False | 314.6 | 0.00 |
| front_back:neutral:label_colon:time | L40 mlp_input s1.0 | False | +0.31 | +1.11 | +1.41 | False | 582.0 | 0.00 |
| front_back:neutral:plain:plant | L39 mlp_input s1.0 | False | +1.87 | +2.28 | +10.69 | False | 393.6 | 0.00 |
| front_back:neutral:plain:time | L40 attention_output s1.0 | False | -0.40 | +1.20 | -1.76 | False | 7666.4 | 0.00 |

## deepseek7b

families=['long', 'short', 'neutral']; splits=['front_back', 'back_front']; formats=['plain', 'label_colon', 'answer_one_word', 'multiple_choice']; train/test=8/8

### By category

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| container | 24 | 0.17 | +1.18 | +1.49 | 10498.3 | 0.00 |
| number | 24 | 0.17 | +0.88 | +1.36 | 13163.3 | 0.00 |
| plant | 24 | 0.08 | +1.45 | +2.38 | 2426.9 | 0.00 |
| time | 24 | 0.25 | +4.39 | +1.74 | 17019.6 | 0.00 |

### By format

| group | n | held_clean_rate | mean_rec | mean_release | mean_token_rank | token_argmax |
|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.29 | +1.26 | +1.16 | 16506.2 | 0.00 |
| label_colon | 24 | 0.21 | +1.17 | +2.45 | 6165.5 | 0.00 |
| multiple_choice | 24 | 0.08 | +4.62 | +1.64 | 10654.8 | 0.00 |
| plain | 24 | 0.08 | +0.84 | +1.72 | 9781.6 | 0.00 |

### Cases

| case | train path | train clean | held T | held R | held rec | held clean | token_rank | token_argmax |
|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | L28 mlp_input s1.5 | False | +0.43 | +1.66 | +1.32 | False | 60405.6 | 0.00 |
| back_front:long:answer_one_word:number | L28 input_answer s1.25 | False | +0.38 | +2.67 | +1.24 | False | 17315.0 | 0.00 |
| back_front:long:answer_one_word:plant | L28 input_answer s1.25 | False | +1.23 | +2.19 | +1.62 | False | 386.0 | 0.00 |
| back_front:long:answer_one_word:time | L27 mlp_input s0.35 | True | +0.05 | +0.06 | +1.02 | True | 18269.8 | 0.00 |
| back_front:long:label_colon:container | L28 mlp_input s0.35 | True | +0.57 | +0.22 | +1.19 | True | 22725.5 | 0.00 |
| back_front:long:label_colon:number | L28 attention_output s0.35 | True | -2.69 | +0.00 | +0.30 | False | 20.5 | 0.00 |
| back_front:long:label_colon:plant | L28 attention_output s0.35 | True | -0.90 | +0.26 | +0.67 | False | 62.4 | 0.00 |
| back_front:long:label_colon:time | L28 mlp_input s0.3 | True | -3.61 | +0.00 | +0.07 | False | 8686.0 | 0.00 |
| back_front:long:multiple_choice:container | L28 attention_output s0.2 | True | -0.84 | +0.00 | +0.35 | False | 19.5 | 0.00 |
| back_front:long:multiple_choice:number | L28 attention_output s0.25 | True | -0.31 | +0.40 | +0.79 | False | 164.9 | 0.00 |
| back_front:long:multiple_choice:plant | L28 input_answer s0.2 | True | -0.67 | +0.00 | +0.46 | False | 12.9 | 0.00 |
| back_front:long:multiple_choice:time | L28 attention_output s0.2 | True | -0.50 | +0.18 | +0.65 | True | 71.2 | 0.00 |
| back_front:long:plain:container | L28 input_answer s1.0 | True | -1.09 | +0.00 | +0.30 | False | 3091.1 | 0.00 |
| back_front:long:plain:number | L28 attention_output s0.75 | True | -0.20 | +0.99 | +0.88 | False | 13288.9 | 0.00 |
| back_front:long:plain:plant | L28 mlp_input s1.5 | False | +0.98 | +5.48 | +1.53 | False | 1951.6 | 0.00 |
| back_front:long:plain:time | L28 attention_output s1.0 | True | +0.87 | +0.44 | +1.54 | False | 24778.6 | 0.00 |
| back_front:neutral:answer_one_word:container | L28 mlp_input s0.2 | True | -1.77 | +0.00 | +0.54 | True | 12496.6 | 0.00 |
| back_front:neutral:answer_one_word:number | L28 mlp_input s0.2 | True | -2.73 | +0.00 | +0.48 | False | 20453.6 | 0.00 |
| back_front:neutral:answer_one_word:plant | L28 input_answer s0.3 | True | -3.34 | +0.00 | +0.29 | False | 720.2 | 0.00 |
| back_front:neutral:answer_one_word:time | L28 mlp_input s0.2 | True | -4.20 | +0.00 | +0.31 | False | 16060.4 | 0.00 |
| back_front:neutral:label_colon:container | L28 attention_output s0.75 | True | +0.49 | +1.11 | +1.16 | False | 4427.9 | 0.00 |
| back_front:neutral:label_colon:number | L28 mlp_input s1.5 | False | +0.66 | +1.67 | +1.11 | False | 11668.2 | 0.00 |
| back_front:neutral:label_colon:plant | L28 input_answer s1.0 | True | +1.69 | +3.14 | +1.42 | False | 2601.6 | 0.00 |
| back_front:neutral:label_colon:time | L27 mlp_input s0.75 | True | -1.11 | +0.25 | +0.78 | True | 7842.1 | 0.00 |
| back_front:neutral:multiple_choice:container | L28 input_answer s0.5 | True | +0.54 | +0.29 | +1.49 | False | 456.8 | 0.00 |
| back_front:neutral:multiple_choice:number | L27 mlp_input s1.5 | False | +1.22 | +1.88 | +1.76 | False | 121923.1 | 0.00 |
| back_front:neutral:multiple_choice:plant | L28 attention_output s0.5 | True | -1.28 | +0.93 | +0.40 | False | 9.9 | 0.00 |
| back_front:neutral:multiple_choice:time | L28 input_answer s1.0 | True | -1.35 | +1.17 | +0.44 | False | 590.2 | 0.00 |
| back_front:neutral:plain:container | L28 attention_output s0.75 | True | +1.51 | +1.15 | +1.87 | False | 5306.0 | 0.00 |
| back_front:neutral:plain:number | L28 mlp_input s1.5 | True | -5.21 | +0.00 | +0.28 | False | 25155.0 | 0.00 |
| back_front:neutral:plain:plant | L27 mlp_input s1.5 | False | -0.83 | +2.27 | +0.75 | False | 6623.5 | 0.00 |
| back_front:neutral:plain:time | L27 mlp_input s0.35 | True | -2.70 | +0.00 | +0.64 | True | 995.6 | 0.00 |
| back_front:short:answer_one_word:container | L28 attention_output s1.5 | False | -0.15 | +1.58 | +0.95 | False | 12786.6 | 0.00 |
| back_front:short:answer_one_word:number | L28 attention_output s0.75 | True | -2.24 | +0.00 | +0.57 | True | 5710.4 | 0.00 |
| back_front:short:answer_one_word:plant | L28 input_answer s0.75 | True | -1.15 | +0.00 | +0.78 | True | 1435.2 | 0.00 |
| back_front:short:answer_one_word:time | L28 input_answer s1.25 | True | +0.79 | +2.53 | +1.17 | False | 9726.8 | 0.00 |
| back_front:short:label_colon:container | L28 input_answer s1.5 | False | +5.21 | +11.83 | +2.69 | False | 8724.0 | 0.00 |
| back_front:short:label_colon:number | L27 input_answer s0.5 | True | -0.63 | +0.89 | +0.84 | False | 1527.5 | 0.00 |
| back_front:short:label_colon:plant | L27 attention_output s1.5 | False | +4.23 | +10.30 | +3.49 | False | 2257.1 | 0.00 |
| back_front:short:label_colon:time | L28 attention_output s0.5 | True | +2.15 | +3.92 | +1.93 | False | 939.1 | 0.00 |
| back_front:short:multiple_choice:container | L28 mlp_input s0.75 | True | +0.13 | +0.31 | +1.03 | False | 779.6 | 0.00 |
| back_front:short:multiple_choice:number | L27 attention_output s1.25 | True | -1.29 | +1.39 | +0.70 | False | 760.2 | 0.00 |
| back_front:short:multiple_choice:plant | L27 attention_output s1.0 | True | -1.54 | +0.00 | +0.58 | True | 10.4 | 0.00 |
| back_front:short:multiple_choice:time | L28 attention_output s0.3 | True | +1.33 | +0.89 | +76.51 | False | 10.9 | 0.00 |
| back_front:short:plain:container | L28 input_answer s0.75 | True | +0.68 | +1.27 | +1.13 | False | 9693.6 | 0.00 |
| back_front:short:plain:number | L28 mlp_input s1.5 | True | -0.13 | +0.00 | +0.98 | True | 5012.8 | 0.00 |
| back_front:short:plain:plant | L28 attention_output s0.75 | True | -1.11 | +3.32 | +0.81 | False | 1366.4 | 0.00 |
| back_front:short:plain:time | L28 attention_output s0.75 | True | -1.10 | +2.41 | +0.85 | False | 5292.2 | 0.00 |
| front_back:long:answer_one_word:container | L28 mlp_input s0.2 | True | +1.93 | +0.00 | +3.56 | True | 30634.5 | 0.00 |
| front_back:long:answer_one_word:number | L28 input_answer s0.5 | False | +0.41 | +2.07 | +1.46 | False | 10343.2 | 0.00 |
| front_back:long:answer_one_word:plant | L28 input_answer s0.75 | False | +1.89 | +2.43 | +3.97 | False | 410.4 | 0.00 |
| front_back:long:answer_one_word:time | L27 mlp_input s0.5 | True | +0.46 | +0.64 | +1.31 | False | 47059.6 | 0.00 |
| front_back:long:label_colon:container | L28 attention_output s0.25 | True | +0.08 | +0.31 | +1.07 | False | 2362.8 | 0.00 |
| front_back:long:label_colon:number | L28 attention_output s0.25 | True | -0.53 | +0.10 | +0.72 | True | 14.9 | 0.00 |
| front_back:long:label_colon:plant | L28 attention_output s0.25 | True | -0.06 | +0.31 | +0.95 | False | 42.5 | 0.00 |
| front_back:long:label_colon:time | L27 mlp_input s0.2 | True | +0.31 | +0.00 | +1.08 | True | 550.8 | 0.00 |
| front_back:long:multiple_choice:container | L28 mlp_input s0.75 | True | -1.06 | +0.00 | -0.80 | False | 7139.8 | 0.00 |
| front_back:long:multiple_choice:number | L27 attention_output s1.25 | True | -1.02 | +0.00 | -0.21 | False | 864.2 | 0.00 |
| front_back:long:multiple_choice:plant | L27 attention_output s0.2 | True | -0.19 | +0.00 | -0.56 | False | 14.8 | 0.00 |
| front_back:long:multiple_choice:time | L28 input_answer s1.5 | False | +3.31 | +3.83 | +4.91 | False | 310.9 | 0.00 |
| front_back:long:plain:container | L28 attention_output s0.3 | True | -0.30 | +0.34 | +0.62 | False | 4524.9 | 0.00 |
| front_back:long:plain:number | L28 attention_output s0.35 | True | -0.37 | +0.63 | +0.60 | False | 12185.6 | 0.00 |
| front_back:long:plain:plant | L28 mlp_input s1.5 | False | -0.51 | +2.91 | +0.26 | False | 25654.1 | 0.00 |
| front_back:long:plain:time | L28 attention_output s1.0 | True | -0.17 | +0.66 | +0.81 | False | 32664.5 | 0.00 |
| front_back:neutral:answer_one_word:container | L28 attention_output s1.5 | False | +1.49 | +1.20 | +2.08 | False | 21650.8 | 0.00 |
| front_back:neutral:answer_one_word:number | L27 input_answer s0.5 | True | -0.66 | +0.00 | +0.59 | True | 5646.8 | 0.00 |
| front_back:neutral:answer_one_word:plant | L28 input_answer s1.0 | False | +1.52 | +3.72 | +2.41 | False | 1368.1 | 0.00 |
| front_back:neutral:answer_one_word:time | L27 attention_output s1.25 | False | -2.24 | +0.00 | +0.29 | False | 753.0 | 0.00 |
| front_back:neutral:label_colon:container | L28 attention_output s1.0 | True | -2.36 | +0.91 | +0.52 | False | 22509.4 | 0.00 |
| front_back:neutral:label_colon:number | L28 mlp_input s0.5 | True | -1.47 | +1.03 | +0.70 | False | 10553.5 | 0.00 |
| front_back:neutral:label_colon:plant | L28 input_answer s0.75 | True | +0.91 | +1.91 | +1.29 | False | 1072.5 | 0.00 |
| front_back:neutral:label_colon:time | L28 mlp_input s0.35 | True | -1.97 | +0.24 | +0.63 | True | 36647.8 | 0.00 |
| front_back:neutral:multiple_choice:container | L28 input_answer s1.5 | False | +1.42 | +2.78 | +2.68 | False | 776.2 | 0.00 |
| front_back:neutral:multiple_choice:number | L28 attention_output s1.5 | False | +0.72 | +2.53 | +1.58 | False | 1101.0 | 0.00 |
| front_back:neutral:multiple_choice:plant | L28 attention_output s1.5 | False | +3.35 | +5.93 | +8.28 | False | 22.2 | 0.00 |
| front_back:neutral:multiple_choice:time | L28 mlp_input s1.5 | False | +2.13 | +3.34 | +2.81 | False | 67355.4 | 0.00 |
| front_back:neutral:plain:container | L27 attention_output s1.5 | True | -2.87 | +0.00 | +0.34 | False | 6530.6 | 0.00 |
| front_back:neutral:plain:number | L28 attention_output s1.5 | True | -3.65 | +0.00 | +0.40 | False | 6213.4 | 0.00 |
| front_back:neutral:plain:plant | L28 attention_output s0.75 | True | -0.04 | +1.77 | +0.99 | False | 8201.4 | 0.00 |
| front_back:neutral:plain:time | L28 mlp_input s0.5 | True | -5.31 | +0.00 | +0.28 | False | 21520.1 | 0.00 |
| front_back:short:answer_one_word:container | L28 input_answer s1.25 | True | -1.47 | +0.00 | +0.55 | True | 7572.0 | 0.00 |
| front_back:short:answer_one_word:number | L28 mlp_input s1.5 | False | -0.07 | +2.61 | +0.96 | False | 35561.8 | 0.00 |
| front_back:short:answer_one_word:plant | L28 attention_output s1.5 | False | +1.11 | +2.53 | +1.51 | False | 3905.6 | 0.00 |
| front_back:short:answer_one_word:time | L28 mlp_input s1.5 | False | +0.92 | +1.88 | +1.35 | False | 55476.5 | 0.00 |
| front_back:short:label_colon:container | L27 input_answer s1.0 | True | -2.50 | +1.49 | +0.49 | False | 340.2 | 0.00 |
| front_back:short:label_colon:number | L28 attention_output s1.5 | False | +3.79 | +9.12 | +1.95 | False | 1913.8 | 0.00 |
| front_back:short:label_colon:plant | L27 input_answer s0.75 | True | -0.24 | +1.89 | +0.91 | False | 21.2 | 0.00 |
| front_back:short:label_colon:time | L28 attention_output s0.35 | True | +5.77 | +7.82 | +2.09 | False | 461.6 | 0.00 |
| front_back:short:multiple_choice:container | L28 input_answer s1.5 | False | +1.69 | +4.78 | +1.55 | False | 23.8 | 0.00 |
| front_back:short:multiple_choice:number | L28 attention_output s1.5 | False | +2.23 | +3.83 | +2.08 | False | 4582.0 | 0.00 |
| front_back:short:multiple_choice:plant | L28 input_answer s0.5 | True | +0.13 | +2.67 | +1.05 | False | 13.4 | 0.00 |
| front_back:short:multiple_choice:time | L28 mlp_input s0.75 | True | +2.61 | +2.17 | +2.30 | False | 48702.0 | 0.00 |
| front_back:short:plain:container | L28 input_answer s0.75 | True | +2.13 | +4.57 | +1.57 | False | 6981.8 | 0.00 |
| front_back:short:plain:number | L28 attention_output s1.25 | True | -2.53 | +0.77 | +0.29 | False | 3939.6 | 0.00 |
| front_back:short:plain:plant | L28 input_answer s0.5 | True | -0.38 | +3.18 | +0.91 | False | 82.0 | 0.00 |
| front_back:short:plain:time | L28 attention_output s1.0 | True | +2.77 | +9.18 | +1.59 | False | 3704.4 | 0.00 |

