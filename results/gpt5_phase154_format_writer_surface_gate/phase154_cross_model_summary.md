# Phase 154 Cross-model Format Writer Surface Gate Summary

## qwen3

cases=120, patch_layers=[33, 34, 35, 36], formats=label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice

### By category

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 8.7 | 10.2 | +1.4 | +4.7 | +2.3 | +5.8 | L34:mlp_output | L36:mlp_output | L35:mlp_output |
| number | 30 | 8.3 | 8.2 | +2.5 | +4.2 | +3.9 | +4.8 | L36:mlp_output | L36:mlp_output | L36:mlp_output |
| plant | 30 | 5.1 | 8.6 | +0.7 | +3.9 | +1.1 | +3.8 | L33:attention_output | L36:mlp_output | L36:mlp_output |
| time | 30 | 10.3 | 9.2 | +2.1 | +4.2 | +2.9 | +5.0 | L36:mlp_output | L36:mlp_output | L36:mlp_output |

### By format

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 7.2 | 2.7 | +0.6 | +1.7 | +1.0 | +1.2 | L34:mlp_output | L36:mlp_output | L35:mlp_output |
| label_colon | 24 | 11.4 | 6.1 | +5.2 | +7.9 | +8.0 | +4.7 | L36:mlp_output | L36:mlp_output | L36:mlp_output |
| list_answer | 24 | 7.1 | 6.8 | +0.8 | +2.2 | +1.7 | +1.7 | L36:attention_output | L36:mlp_output | L36:mlp_output |
| multiple_choice | 24 | 1.4 | 10.1 | +0.4 | +2.9 | +0.4 | +4.1 | L33:attention_output | L35:mlp_output | L33:attention_output |
| quoted_answer | 24 | 13.3 | 19.6 | +1.5 | +6.4 | +1.7 | +12.6 | L34:mlp_output | L36:attention_output | L35:attention_output |

### By family

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 10.9 | 12.0 | +2.4 | +5.2 | +3.5 | +6.0 | L36:mlp_output | L36:mlp_output | L36:mlp_output |
| neutral | 40 | 10.2 | 6.8 | +1.5 | +2.4 | +3.0 | +4.3 | L33:attention_output | L36:mlp_output | L36:mlp_output |
| short | 40 | 3.2 | 8.3 | +1.2 | +5.1 | +1.2 | +4.2 | L36:mlp_output | L36:mlp_output | L33:attention_output |

### By split

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 7.3 | 9.3 | +1.3 | +4.3 | +2.3 | +5.0 | L36:mlp_output | L36:mlp_output | L36:mlp_output |
| front_back | 60 | 8.9 | 8.8 | +2.1 | +4.1 | +2.8 | +4.7 | L36:mlp_output | L36:mlp_output | L36:mlp_output |

### By Writer Condition

| mode | component | layer | n | answer_rank_delta | format_rank_delta | answer_argmax_delta | format_argmax_delta |
|---|---|---|---|---|---|---|---|
| format_proj | attention_output | L33 | 120 | +0.0 | +0.1 | +0.001 | +0.000 |
| format_proj | attention_output | L34 | 120 | +0.1 | -0.1 | +0.004 | +0.000 |
| format_proj | attention_output | L35 | 120 | +0.1 | -0.1 | +0.009 | +0.001 |
| format_proj | attention_output | L36 | 120 | -0.2 | +0.9 | +0.004 | +0.004 |
| format_proj | mlp_output | L33 | 120 | -0.1 | +0.0 | +0.007 | -0.005 |
| format_proj | mlp_output | L34 | 120 | +0.2 | +0.2 | +0.004 | -0.005 |
| format_proj | mlp_output | L35 | 120 | +0.2 | +0.7 | +0.036 | -0.017 |
| format_proj | mlp_output | L36 | 120 | +0.2 | +0.6 | +0.001 | -0.084 |
| joint_proj | attention_output | L33 | 120 | +0.1 | +0.2 | +0.000 | +0.000 |
| joint_proj | attention_output | L34 | 120 | +0.1 | +0.2 | +0.003 | +0.001 |
| joint_proj | attention_output | L35 | 120 | +0.2 | +0.3 | +0.003 | +0.000 |
| joint_proj | attention_output | L36 | 120 | -0.2 | +1.4 | -0.008 | +0.014 |
| joint_proj | mlp_output | L33 | 120 | -0.0 | +0.2 | +0.007 | -0.006 |
| joint_proj | mlp_output | L34 | 120 | +0.3 | +0.5 | +0.002 | -0.006 |
| joint_proj | mlp_output | L35 | 120 | +0.1 | +2.2 | +0.016 | -0.011 |
| joint_proj | mlp_output | L36 | 120 | +0.8 | -0.1 | -0.026 | -0.022 |
| semantic_proj | attention_output | L33 | 120 | +0.0 | +0.1 | -0.001 | +0.000 |
| semantic_proj | attention_output | L34 | 120 | +0.1 | +0.3 | +0.000 | +0.001 |
| semantic_proj | attention_output | L35 | 120 | +0.2 | +0.4 | +0.000 | +0.000 |
| semantic_proj | attention_output | L36 | 120 | -0.1 | +1.5 | -0.014 | +0.016 |
| semantic_proj | mlp_output | L33 | 120 | +0.2 | +0.1 | +0.003 | -0.001 |
| semantic_proj | mlp_output | L34 | 120 | +0.3 | +0.2 | +0.003 | +0.000 |
| semantic_proj | mlp_output | L35 | 120 | +0.1 | +3.3 | +0.002 | -0.005 |
| semantic_proj | mlp_output | L36 | 120 | +0.6 | +0.6 | -0.036 | +0.007 |

### Cases

| case | clean_ans | clean_fmt | sem_ans | fmt_fmt | joint_ans | joint_fmt |
|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 10.6 | 4.1 | L34:mlp_output +0.8 | L36:mlp_output +1.5 | L35:mlp_output +2.2 | L36:mlp_output +2.8 |
| back_front:long:answer_one_word:number | 13.4 | 4.0 | L36:mlp_output +1.8 | L36:mlp_output +3.1 | L35:attention_output +1.5 | L36:mlp_output +3.8 |
| back_front:long:answer_one_word:plant | 10.6 | 4.1 | L34:mlp_output +0.6 | L36:mlp_output +2.9 | L34:mlp_output +1.1 | L36:mlp_output +3.0 |
| back_front:long:answer_one_word:time | 12.0 | 6.8 | L34:mlp_output +0.5 | L36:mlp_output +0.8 | L35:mlp_output +1.1 | L36:mlp_output +1.4 |
| back_front:long:label_colon:container | 12.2 | 6.9 | L35:mlp_output +0.5 | L36:mlp_output +6.9 | L36:mlp_output +1.8 | L36:mlp_output +6.4 |
| back_front:long:label_colon:number | 17.4 | 4.4 | L36:mlp_output +11.0 | L36:mlp_output +4.0 | L36:mlp_output +13.6 | L36:mlp_output +3.6 |
| back_front:long:label_colon:plant | 2.4 | 7.1 | L36:attention_output +0.1 | L36:mlp_output +6.4 | L36:mlp_output +0.2 | L36:mlp_output +5.8 |
| back_front:long:label_colon:time | 24.2 | 5.5 | L36:mlp_output +15.0 | L36:mlp_output +5.2 | L36:mlp_output +23.1 | L36:mlp_output +5.8 |
| back_front:long:list_answer:container | 11.1 | 11.5 | L36:attention_output +1.1 | L34:mlp_output +2.8 | L36:mlp_output +4.1 | L34:mlp_output +2.8 |
| back_front:long:list_answer:number | 9.4 | 8.0 | L36:mlp_output +1.0 | L34:mlp_output +0.9 | L36:mlp_output +5.0 | L33:mlp_output +0.6 |
| back_front:long:list_answer:plant | 10.2 | 12.6 | L36:mlp_output +2.2 | L35:mlp_output +2.0 | L36:mlp_output +4.5 | L35:mlp_output +2.6 |
| back_front:long:list_answer:time | 12.9 | 12.0 | L36:mlp_output +1.2 | L35:mlp_output +2.2 | L36:mlp_output +3.5 | L35:mlp_output +2.6 |
| back_front:long:multiple_choice:container | 1.1 | 8.9 | L34:attention_output +1.1 | L36:mlp_output +3.1 | L34:attention_output +1.0 | L36:mlp_output +7.5 |
| back_front:long:multiple_choice:number | 3.2 | 8.2 | L35:mlp_output +0.9 | L34:mlp_output +0.4 | L35:mlp_output +1.4 | L34:attention_output +1.2 |
| back_front:long:multiple_choice:plant | 1.1 | 8.6 | L33:attention_output +0.0 | L36:mlp_output +1.5 | L33:attention_output +0.0 | L34:attention_output +1.5 |
| back_front:long:multiple_choice:time | 1.0 | 9.8 | L36:mlp_output +0.9 | L34:mlp_output +0.8 | L36:mlp_output +0.1 | L35:mlp_output +1.8 |
| back_front:long:quoted_answer:container | 17.1 | 34.8 | L34:attention_output +1.8 | L36:attention_output +12.5 | L34:attention_output +2.0 | L36:attention_output +13.4 |
| back_front:long:quoted_answer:number | 16.1 | 37.5 | L34:mlp_output +0.5 | L36:attention_output +18.5 | L35:attention_output +1.6 | L36:attention_output +19.2 |
| back_front:long:quoted_answer:plant | 15.8 | 26.2 | L35:mlp_output +1.2 | L36:attention_output +14.9 | L35:attention_output +1.4 | L36:attention_output +14.9 |
| back_front:long:quoted_answer:time | 16.8 | 30.6 | L34:mlp_output +0.6 | L36:attention_output +15.8 | L35:attention_output +1.0 | L36:attention_output +15.2 |
| back_front:neutral:answer_one_word:container | 7.4 | 1.0 | L33:attention_output +0.0 | L36:mlp_output +4.1 | L33:attention_output +0.1 | L36:mlp_output +1.0 |
| back_front:neutral:answer_one_word:number | 9.1 | 1.0 | L33:attention_output +0.0 | L36:mlp_output +5.6 | L34:mlp_output +0.0 | L33:attention_output +0.0 |
| back_front:neutral:answer_one_word:plant | 5.5 | 1.1 | L33:attention_output -0.2 | L36:mlp_output +1.4 | L33:attention_output -0.1 | L36:mlp_output +0.6 |
| back_front:neutral:answer_one_word:time | 9.9 | 1.0 | L34:mlp_output +0.2 | L36:mlp_output +3.5 | L36:mlp_output +1.5 | L36:mlp_output +0.2 |
| back_front:neutral:label_colon:container | 9.5 | 2.5 | L36:mlp_output +1.6 | L36:mlp_output +4.2 | L36:mlp_output +1.6 | L36:mlp_output +2.2 |
| back_front:neutral:label_colon:number | 23.2 | 1.0 | L36:attention_output +5.8 | L36:mlp_output +1.0 | L36:mlp_output +26.1 | L33:attention_output +0.0 |
| back_front:neutral:label_colon:plant | 2.0 | 5.2 | L36:mlp_output +0.6 | L36:mlp_output +3.8 | L36:mlp_output +2.2 | L36:mlp_output +2.4 |
| back_front:neutral:label_colon:time | 13.2 | 1.5 | L34:mlp_output +0.5 | L36:mlp_output +1.9 | L36:mlp_output +8.5 | L36:mlp_output +0.4 |
| back_front:neutral:list_answer:container | 7.4 | 4.6 | L36:attention_output +1.0 | L36:mlp_output +2.1 | L36:mlp_output +0.8 | L35:mlp_output +1.6 |
| back_front:neutral:list_answer:number | 6.8 | 4.0 | L36:attention_output +0.2 | L36:mlp_output +1.4 | L36:attention_output +0.2 | L33:attention_output +0.0 |
| back_front:neutral:list_answer:plant | 7.1 | 4.5 | L36:attention_output +1.9 | L36:mlp_output +2.6 | L36:attention_output +1.4 | L36:mlp_output +2.1 |
| back_front:neutral:list_answer:time | 10.6 | 3.4 | L36:attention_output +1.1 | L36:mlp_output +2.6 | L36:mlp_output +1.5 | L36:mlp_output +1.5 |
| back_front:neutral:multiple_choice:container | 1.0 | 15.0 | L33:attention_output +0.0 | L34:mlp_output +0.9 | L33:attention_output +0.0 | L35:mlp_output +2.8 |
| back_front:neutral:multiple_choice:number | 2.0 | 15.5 | L33:attention_output +0.0 | L35:mlp_output +1.5 | L33:attention_output +0.0 | L35:mlp_output +3.5 |
| back_front:neutral:multiple_choice:plant | 1.4 | 9.4 | L33:attention_output +0.2 | L35:mlp_output +2.5 | L36:mlp_output +1.5 | L35:mlp_output +2.5 |
| back_front:neutral:multiple_choice:time | 1.8 | 13.0 | L35:mlp_output +0.2 | L35:mlp_output +2.5 | L36:mlp_output +0.1 | L35:mlp_output +3.2 |
| back_front:neutral:quoted_answer:container | 13.0 | 19.1 | L34:mlp_output +2.4 | L35:mlp_output +2.9 | L33:mlp_output +3.1 | L35:mlp_output +18.9 |
| back_front:neutral:quoted_answer:number | 16.4 | 3.8 | L34:mlp_output +0.1 | L36:mlp_output +3.0 | L33:attention_output +0.5 | L36:mlp_output +16.5 |
| back_front:neutral:quoted_answer:plant | 10.9 | 12.1 | L33:mlp_output +2.2 | L35:mlp_output +0.9 | L36:mlp_output +2.2 | L35:mlp_output +8.9 |
| back_front:neutral:quoted_answer:time | 20.8 | 16.5 | L34:mlp_output +1.1 | L35:mlp_output +1.9 | L33:attention_output +1.4 | L35:mlp_output +18.9 |
| back_front:short:answer_one_word:container | 1.2 | 2.4 | L36:attention_output +0.5 | L35:mlp_output +0.6 | L36:attention_output +0.6 | L35:mlp_output +0.1 |
| back_front:short:answer_one_word:number | 2.0 | 1.0 | L36:mlp_output +0.1 | L35:mlp_output +0.5 | L33:attention_output +0.0 | L35:mlp_output +0.2 |
| back_front:short:answer_one_word:plant | 1.1 | 2.1 | L36:mlp_output +0.6 | L35:mlp_output +0.1 | L36:mlp_output +1.1 | L33:mlp_output +0.0 |
| back_front:short:answer_one_word:time | 2.1 | 1.6 | L36:mlp_output +0.1 | L35:mlp_output +0.5 | L33:attention_output +0.0 | L33:mlp_output +0.2 |
| back_front:short:label_colon:container | 1.8 | 7.4 | L36:mlp_output +1.9 | L36:mlp_output +17.9 | L35:mlp_output +1.5 | L36:mlp_output +11.2 |
| back_front:short:label_colon:number | 2.1 | 4.5 | L36:mlp_output +5.5 | L36:mlp_output +18.4 | L36:mlp_output +5.9 | L36:mlp_output +6.8 |
| back_front:short:label_colon:plant | 2.2 | 16.6 | L36:mlp_output +2.1 | L36:mlp_output +16.2 | L36:mlp_output +1.0 | L36:mlp_output +8.9 |
| back_front:short:label_colon:time | 2.4 | 4.1 | L36:mlp_output +2.4 | L36:mlp_output +15.9 | L36:mlp_output +1.5 | L36:mlp_output +5.8 |
| back_front:short:list_answer:container | 2.6 | 5.5 | L36:attention_output +0.4 | L36:mlp_output +3.8 | L36:attention_output +0.4 | L35:mlp_output +1.0 |
| back_front:short:list_answer:number | 2.6 | 4.6 | L36:mlp_output +1.1 | L36:mlp_output +1.0 | L36:mlp_output +1.8 | L35:mlp_output +0.8 |
| back_front:short:list_answer:plant | 1.8 | 5.9 | L33:attention_output +0.0 | L35:mlp_output +1.4 | L33:attention_output +0.0 | L35:mlp_output +1.1 |
| back_front:short:list_answer:time | 1.9 | 6.2 | L36:attention_output +0.4 | L35:mlp_output +2.6 | L36:attention_output +0.4 | L35:mlp_output +0.8 |
| back_front:short:multiple_choice:container | 1.0 | 13.9 | L33:attention_output +0.0 | L35:mlp_output +10.0 | L33:attention_output +0.0 | L35:mlp_output +8.9 |
| back_front:short:multiple_choice:number | 1.0 | 5.6 | L33:attention_output +0.0 | L35:mlp_output +4.0 | L33:attention_output +0.0 | L35:mlp_output +2.8 |
| back_front:short:multiple_choice:plant | 1.0 | 3.8 | L33:attention_output +0.0 | L35:mlp_output +0.5 | L33:attention_output +0.0 | L35:mlp_output +0.9 |
| back_front:short:multiple_choice:time | 1.6 | 11.2 | L35:mlp_output +0.2 | L35:mlp_output +8.5 | L35:mlp_output +0.5 | L35:mlp_output +6.2 |
| back_front:short:quoted_answer:container | 1.8 | 21.2 | L33:mlp_output +0.1 | L33:attention_output +1.0 | L33:mlp_output +0.5 | L35:mlp_output +15.2 |
| back_front:short:quoted_answer:number | 2.1 | 21.6 | L36:mlp_output +0.1 | L36:attention_output +0.2 | L33:attention_output +0.0 | L35:mlp_output +12.1 |
| back_front:short:quoted_answer:plant | 1.1 | 10.5 | L36:mlp_output +1.5 | L36:mlp_output +1.5 | L36:mlp_output +2.4 | L35:mlp_output +2.4 |
| back_front:short:quoted_answer:time | 4.8 | 19.1 | L35:mlp_output +0.8 | L33:attention_output +0.1 | L33:attention_output +0.1 | L35:mlp_output +10.6 |
| front_back:long:answer_one_word:container | 10.9 | 4.8 | L34:mlp_output +1.8 | L36:mlp_output +1.8 | L35:mlp_output +3.9 | L36:mlp_output +2.9 |
| front_back:long:answer_one_word:number | 12.1 | 5.0 | L34:mlp_output +0.6 | L36:mlp_output +2.5 | L36:mlp_output +1.4 | L36:mlp_output +3.9 |
| front_back:long:answer_one_word:plant | 10.8 | 4.8 | L33:mlp_output +1.0 | L36:mlp_output +3.4 | L34:mlp_output +1.9 | L36:mlp_output +3.4 |
| front_back:long:answer_one_word:time | 11.5 | 6.4 | L34:mlp_output +1.0 | L36:mlp_output +1.4 | L34:mlp_output +1.8 | L36:mlp_output +1.9 |
| front_back:long:label_colon:container | 17.9 | 6.1 | L35:mlp_output +5.9 | L36:mlp_output +7.0 | L35:mlp_output +6.6 | L36:mlp_output +5.4 |
| front_back:long:label_colon:number | 11.9 | 5.4 | L36:mlp_output +22.0 | L36:mlp_output +4.2 | L36:mlp_output +16.4 | L36:mlp_output +3.5 |
| front_back:long:label_colon:plant | 2.9 | 6.8 | L36:attention_output +0.2 | L36:mlp_output +3.5 | L33:attention_output +0.0 | L36:mlp_output +4.4 |
| front_back:long:label_colon:time | 27.1 | 5.9 | L36:mlp_output +12.9 | L36:mlp_output +5.1 | L36:mlp_output +15.8 | L36:mlp_output +5.2 |
| front_back:long:list_answer:container | 13.5 | 11.8 | L34:mlp_output +0.6 | L34:mlp_output +2.1 | L36:mlp_output +3.1 | L35:mlp_output +3.0 |
| front_back:long:list_answer:number | 11.4 | 7.6 | L36:mlp_output +1.4 | L34:mlp_output +1.0 | L36:mlp_output +4.9 | L36:mlp_output +2.2 |
| front_back:long:list_answer:plant | 9.5 | 11.8 | L36:attention_output +0.6 | L35:mlp_output +2.1 | L36:attention_output +1.2 | L35:mlp_output +3.0 |
| front_back:long:list_answer:time | 12.1 | 11.6 | L36:attention_output +1.2 | L35:mlp_output +2.6 | L36:mlp_output +2.9 | L35:mlp_output +3.8 |
| front_back:long:multiple_choice:container | 1.0 | 8.8 | L34:attention_output +0.1 | L36:mlp_output +2.6 | L34:attention_output +0.1 | L36:mlp_output +7.1 |
| front_back:long:multiple_choice:number | 1.5 | 8.9 | L36:mlp_output +2.5 | L34:attention_output +0.4 | L36:mlp_output +1.5 | L35:mlp_output +1.5 |
| front_back:long:multiple_choice:plant | 1.0 | 9.2 | L33:attention_output +0.0 | L36:mlp_output +0.1 | L33:attention_output +0.0 | L36:mlp_output +5.4 |
| front_back:long:multiple_choice:time | 1.0 | 9.2 | L36:mlp_output +0.2 | L36:mlp_output +1.0 | L36:mlp_output +0.1 | L36:mlp_output +3.8 |
| front_back:long:quoted_answer:container | 18.2 | 29.1 | L34:mlp_output +1.1 | L36:attention_output +14.5 | L35:mlp_output +3.5 | L36:attention_output +14.1 |
| front_back:long:quoted_answer:number | 13.6 | 30.1 | L34:mlp_output +0.2 | L36:attention_output +14.6 | L34:mlp_output +0.5 | L36:attention_output +18.9 |
| front_back:long:quoted_answer:plant | 14.6 | 22.0 | L35:attention_output +0.5 | L36:attention_output +14.5 | L35:mlp_output +2.4 | L36:attention_output +14.9 |
| front_back:long:quoted_answer:time | 16.6 | 25.1 | L34:attention_output +0.4 | L36:attention_output +17.6 | L35:attention_output +0.9 | L36:attention_output +17.6 |
| front_back:neutral:answer_one_word:container | 9.1 | 1.5 | L34:mlp_output +0.5 | L36:mlp_output +1.6 | L34:mlp_output +0.5 | L35:mlp_output +0.6 |
| front_back:neutral:answer_one_word:number | 7.1 | 1.4 | L34:mlp_output +0.2 | L36:mlp_output +2.4 | L33:mlp_output +0.4 | L36:mlp_output +0.2 |
| front_back:neutral:answer_one_word:plant | 7.2 | 1.2 | L33:attention_output +0.1 | L36:mlp_output +1.4 | L33:attention_output +0.1 | L36:mlp_output +0.5 |
| front_back:neutral:answer_one_word:time | 9.5 | 1.0 | L33:mlp_output +1.4 | L36:mlp_output +0.8 | L36:attention_output +1.0 | L36:mlp_output +0.5 |
| front_back:neutral:label_colon:container | 35.6 | 2.6 | L36:mlp_output +5.2 | L36:mlp_output +1.8 | L36:mlp_output +17.5 | L36:mlp_output +1.8 |
| front_back:neutral:label_colon:number | 14.4 | 1.0 | L36:mlp_output +6.0 | L36:mlp_output +1.6 | L36:mlp_output +19.8 | L36:mlp_output +1.1 |
| front_back:neutral:label_colon:plant | 4.0 | 4.5 | L36:mlp_output +1.2 | L36:mlp_output +3.0 | L36:mlp_output +4.0 | L35:mlp_output +2.0 |
| front_back:neutral:label_colon:time | 13.4 | 2.6 | L36:mlp_output +9.1 | L36:mlp_output +2.5 | L36:mlp_output +9.0 | L36:mlp_output +1.5 |
| front_back:neutral:list_answer:container | 8.9 | 4.8 | L36:attention_output +1.8 | L35:mlp_output +2.9 | L36:attention_output +1.9 | L35:mlp_output +2.2 |
| front_back:neutral:list_answer:number | 6.5 | 2.4 | L36:attention_output +0.4 | L36:mlp_output +1.9 | L36:attention_output +0.5 | L35:mlp_output +1.1 |
| front_back:neutral:list_answer:plant | 5.6 | 4.8 | L36:attention_output +0.4 | L36:mlp_output +2.8 | L36:mlp_output +1.0 | L33:mlp_output +1.4 |
| front_back:neutral:list_answer:time | 11.2 | 2.2 | L36:attention_output +1.2 | L35:mlp_output +2.0 | L36:mlp_output +1.9 | L35:mlp_output +1.5 |
| front_back:neutral:multiple_choice:container | 1.0 | 14.1 | L33:attention_output +0.0 | L35:mlp_output +3.5 | L33:attention_output +0.0 | L35:mlp_output +4.5 |
| front_back:neutral:multiple_choice:number | 2.8 | 11.8 | L36:mlp_output +1.0 | L35:mlp_output +2.0 | L36:mlp_output +1.5 | L35:mlp_output +3.5 |
| front_back:neutral:multiple_choice:plant | 1.1 | 9.6 | L33:attention_output +0.2 | L35:mlp_output +2.0 | L36:mlp_output +0.2 | L35:mlp_output +3.1 |
| front_back:neutral:multiple_choice:time | 1.0 | 15.8 | L33:attention_output +0.0 | L35:mlp_output +3.2 | L33:attention_output +0.0 | L35:mlp_output +6.2 |
| front_back:neutral:quoted_answer:container | 18.1 | 15.1 | L34:mlp_output +3.1 | L35:mlp_output +1.6 | L33:mlp_output +4.0 | L35:mlp_output +9.4 |
| front_back:neutral:quoted_answer:number | 23.2 | 13.2 | L33:mlp_output +2.2 | L35:mlp_output +3.2 | L33:attention_output +1.2 | L35:mlp_output +19.6 |
| front_back:neutral:quoted_answer:plant | 12.9 | 8.6 | L35:attention_output +1.9 | L35:mlp_output +1.8 | L35:attention_output +1.2 | L35:mlp_output +6.0 |
| front_back:neutral:quoted_answer:time | 34.6 | 19.1 | L36:mlp_output +4.2 | L35:mlp_output +2.0 | L35:attention_output +2.4 | L35:mlp_output +19.4 |
| front_back:short:answer_one_word:container | 3.0 | 2.1 | L35:mlp_output +0.5 | L35:mlp_output +0.8 | L35:mlp_output +0.6 | L35:mlp_output +0.8 |
| front_back:short:answer_one_word:number | 1.9 | 1.2 | L36:attention_output +0.4 | L35:mlp_output +0.4 | L35:mlp_output +0.2 | L33:attention_output +0.0 |
| front_back:short:answer_one_word:plant | 1.0 | 3.1 | L36:mlp_output +0.1 | L33:attention_output +0.2 | L36:mlp_output +0.8 | L35:attention_output +0.4 |
| front_back:short:answer_one_word:time | 3.8 | 1.8 | L36:mlp_output +0.6 | L33:attention_output +0.0 | L35:mlp_output +2.2 | L35:mlp_output +0.1 |
| front_back:short:label_colon:container | 13.0 | 13.4 | L35:attention_output +5.6 | L36:mlp_output +12.4 | L35:attention_output +5.2 | L36:mlp_output +8.8 |
| front_back:short:label_colon:number | 7.6 | 4.5 | L36:mlp_output +8.5 | L36:mlp_output +22.8 | L36:mlp_output +8.5 | L36:mlp_output +9.9 |
| front_back:short:label_colon:plant | 3.4 | 20.1 | L36:mlp_output +0.2 | L36:mlp_output +12.9 | L33:attention_output +0.0 | L36:mlp_output +5.9 |
| front_back:short:label_colon:time | 10.2 | 6.6 | L36:mlp_output +1.8 | L36:mlp_output +11.6 | L35:mlp_output +2.1 | L36:mlp_output +4.0 |
| front_back:short:list_answer:container | 2.0 | 6.6 | L33:attention_output +0.0 | L36:mlp_output +5.1 | L33:attention_output +0.0 | L35:mlp_output +2.4 |
| front_back:short:list_answer:number | 2.0 | 3.6 | L36:attention_output +0.5 | L36:mlp_output +1.8 | L36:attention_output +0.5 | L35:mlp_output +0.6 |
| front_back:short:list_answer:plant | 1.6 | 7.2 | L33:attention_output +0.0 | L36:mlp_output +1.8 | L36:mlp_output +0.1 | L33:mlp_output +0.8 |
| front_back:short:list_answer:time | 1.9 | 6.0 | L36:attention_output +0.2 | L35:mlp_output +1.4 | L36:attention_output +0.2 | L35:mlp_output +1.6 |
| front_back:short:multiple_choice:container | 1.0 | 13.2 | L33:attention_output +0.0 | L35:mlp_output +9.2 | L33:attention_output +0.0 | L35:mlp_output +10.4 |
| front_back:short:multiple_choice:number | 1.5 | 6.0 | L36:mlp_output +1.2 | L35:mlp_output +2.8 | L36:mlp_output +0.5 | L35:mlp_output +1.8 |
| front_back:short:multiple_choice:plant | 1.0 | 3.5 | L33:attention_output +0.0 | L35:mlp_output +0.6 | L33:attention_output +0.0 | L35:mlp_output +1.6 |
| front_back:short:multiple_choice:time | 1.6 | 10.1 | L36:mlp_output +0.1 | L35:mlp_output +7.0 | L35:mlp_output +0.4 | L35:mlp_output +6.0 |
| front_back:short:quoted_answer:container | 8.0 | 13.0 | L35:mlp_output +2.8 | L36:mlp_output +0.4 | L35:mlp_output +2.2 | L35:mlp_output +6.0 |
| front_back:short:quoted_answer:number | 4.1 | 20.0 | L35:mlp_output +1.1 | L33:attention_output +0.2 | L35:attention_output +0.9 | L35:mlp_output +5.5 |
| front_back:short:quoted_answer:plant | 3.5 | 11.4 | L36:mlp_output +1.2 | L36:mlp_output +7.1 | L36:mlp_output +1.1 | L35:attention_output +2.6 |
| front_back:short:quoted_answer:time | 16.1 | 9.8 | L35:mlp_output +4.1 | L36:mlp_output +2.8 | L35:mlp_output +3.0 | L35:mlp_output +3.0 |

## glm4

cases=120, patch_layers=[37, 38, 39, 40], formats=label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice

### By category

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 22.6 | 94.1 | +5.7 | +43.7 | +9.2 | +135.4 | L40:mlp_output | L40:mlp_output | L40:mlp_output |
| number | 30 | 26.9 | 66.6 | +9.7 | +39.4 | +16.7 | +65.5 | L40:mlp_output | L40:mlp_output | L39:attention_output |
| plant | 30 | 15.5 | 147.3 | +6.7 | +51.1 | +7.8 | +190.2 | L39:mlp_output | L39:attention_output | L39:attention_output |
| time | 30 | 34.9 | 86.2 | +10.8 | +44.7 | +21.5 | +107.0 | L40:mlp_output | L40:mlp_output | L39:attention_output |

### By format

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 18.2 | 156.5 | +4.4 | +54.2 | +5.1 | +198.0 | L39:mlp_output | L39:attention_output | L37:mlp_output |
| label_colon | 24 | 30.1 | 62.1 | +10.8 | +68.2 | +22.0 | +66.8 | L40:mlp_output | L40:mlp_output | L39:attention_output |
| list_answer | 24 | 19.7 | 20.8 | +10.3 | +9.0 | +12.2 | +10.3 | L40:mlp_output | L40:mlp_output | L40:mlp_output |
| multiple_choice | 24 | 1.4 | 20.3 | +0.9 | +9.5 | +0.9 | +12.9 | L40:mlp_output | L39:mlp_output | L39:mlp_output |
| quoted_answer | 24 | 55.4 | 232.9 | +14.7 | +82.8 | +28.8 | +334.7 | L40:mlp_output | L39:attention_output | L39:mlp_output |

### By family

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 35.5 | 113.4 | +14.2 | +63.8 | +10.9 | +64.1 | L40:mlp_output | L39:attention_output | L39:attention_output |
| neutral | 40 | 33.9 | 98.6 | +7.6 | +39.7 | +28.3 | +150.0 | L39:attention_output | L40:mlp_output | L39:attention_output |
| short | 40 | 5.5 | 83.6 | +2.8 | +30.6 | +2.2 | +159.5 | L40:mlp_output | L40:mlp_output | L39:attention_output |

### By split

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 24.8 | 98.6 | +8.0 | +43.8 | +14.0 | +102.0 | L40:mlp_output | L40:mlp_output | L39:attention_output |
| front_back | 60 | 25.1 | 98.5 | +8.4 | +45.7 | +13.6 | +147.1 | L40:mlp_output | L39:attention_output | L39:attention_output |

### By Writer Condition

| mode | component | layer | n | answer_rank_delta | format_rank_delta | answer_argmax_delta | format_argmax_delta |
|---|---|---|---|---|---|---|---|
| format_proj | attention_output | L37 | 120 | +2.1 | +3.1 | -0.001 | +0.000 |
| format_proj | attention_output | L38 | 120 | +3.5 | +1.7 | -0.008 | +0.000 |
| format_proj | attention_output | L39 | 120 | +5.2 | +28.5 | -0.005 | +0.000 |
| format_proj | attention_output | L40 | 120 | -0.1 | -1.2 | +0.003 | +0.000 |
| format_proj | mlp_output | L37 | 120 | +0.2 | -8.1 | -0.001 | +0.000 |
| format_proj | mlp_output | L38 | 120 | +0.8 | -1.5 | -0.010 | +0.000 |
| format_proj | mlp_output | L39 | 120 | +3.0 | -54.0 | +0.027 | +0.000 |
| format_proj | mlp_output | L40 | 120 | -12.1 | -26.9 | +0.127 | +0.000 |
| joint_proj | attention_output | L37 | 120 | +1.0 | +24.8 | +0.006 | +0.000 |
| joint_proj | attention_output | L38 | 120 | +2.4 | +27.7 | +0.001 | +0.001 |
| joint_proj | attention_output | L39 | 120 | +5.1 | +112.6 | -0.021 | +0.000 |
| joint_proj | attention_output | L40 | 120 | -0.3 | -1.2 | +0.002 | +0.000 |
| joint_proj | mlp_output | L37 | 120 | +0.2 | -8.8 | +0.000 | +0.000 |
| joint_proj | mlp_output | L38 | 120 | -0.3 | +9.8 | +0.001 | +0.000 |
| joint_proj | mlp_output | L39 | 120 | +3.4 | -48.1 | -0.037 | +0.000 |
| joint_proj | mlp_output | L40 | 120 | -9.3 | -35.6 | +0.035 | +0.000 |
| semantic_proj | attention_output | L37 | 120 | -0.8 | +27.4 | +0.008 | +0.001 |
| semantic_proj | attention_output | L38 | 120 | -0.9 | +36.3 | +0.010 | +0.001 |
| semantic_proj | attention_output | L39 | 120 | -0.5 | +102.6 | +0.003 | +0.000 |
| semantic_proj | attention_output | L40 | 120 | -0.1 | -1.3 | -0.003 | +0.000 |
| semantic_proj | mlp_output | L37 | 120 | +0.1 | -3.6 | +0.000 | +0.000 |
| semantic_proj | mlp_output | L38 | 120 | -0.7 | +14.2 | +0.006 | +0.000 |
| semantic_proj | mlp_output | L39 | 120 | +2.4 | -17.5 | -0.052 | +0.000 |
| semantic_proj | mlp_output | L40 | 120 | +3.0 | +7.0 | -0.066 | +0.001 |

### Cases

| case | clean_ans | clean_fmt | sem_ans | fmt_fmt | joint_ans | joint_fmt |
|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 13.8 | 181.0 | L40:mlp_output +4.2 | L39:attention_output +125.0 | L37:mlp_output +2.4 | L39:attention_output +169.8 |
| back_front:long:answer_one_word:number | 26.2 | 186.8 | L40:mlp_output +11.6 | L39:attention_output +101.2 | L37:mlp_output +0.9 | L37:attention_output +91.1 |
| back_front:long:answer_one_word:plant | 21.1 | 305.1 | L40:mlp_output +8.1 | L39:attention_output +221.8 | L37:mlp_output +1.1 | L37:attention_output +240.5 |
| back_front:long:answer_one_word:time | 24.1 | 171.4 | L39:mlp_output +7.0 | L39:attention_output +123.2 | L37:mlp_output +0.5 | L39:attention_output +111.6 |
| back_front:long:label_colon:container | 48.1 | 59.1 | L38:attention_output +5.9 | L39:attention_output +64.0 | L39:attention_output +31.6 | L39:attention_output +59.1 |
| back_front:long:label_colon:number | 22.6 | 48.1 | L37:attention_output +0.9 | L39:attention_output +66.0 | L39:attention_output +22.1 | L39:attention_output +52.9 |
| back_front:long:label_colon:plant | 12.8 | 81.1 | L39:mlp_output +4.9 | L39:attention_output +77.0 | L39:attention_output +17.1 | L39:attention_output +82.8 |
| back_front:long:label_colon:time | 26.2 | 89.4 | L40:mlp_output +31.9 | L39:attention_output +100.4 | L39:attention_output +31.5 | L39:attention_output +101.9 |
| back_front:long:list_answer:container | 11.9 | 19.4 | L40:mlp_output +8.6 | L39:attention_output +4.2 | L40:mlp_output +9.6 | L38:mlp_output +2.9 |
| back_front:long:list_answer:number | 23.1 | 12.2 | L40:mlp_output +29.0 | L39:attention_output +3.0 | L40:mlp_output +15.6 | L39:attention_output +1.8 |
| back_front:long:list_answer:plant | 23.9 | 23.0 | L40:mlp_output +21.2 | L39:attention_output +5.0 | L40:mlp_output +11.8 | L38:mlp_output +3.4 |
| back_front:long:list_answer:time | 33.9 | 24.6 | L40:mlp_output +23.2 | L40:mlp_output +3.1 | L40:mlp_output +25.1 | L39:attention_output +4.9 |
| back_front:long:multiple_choice:container | 1.0 | 22.8 | L40:mlp_output +1.6 | L39:attention_output +4.4 | L40:mlp_output +1.9 | L39:attention_output +7.5 |
| back_front:long:multiple_choice:number | 1.1 | 21.1 | L39:mlp_output +3.2 | L39:attention_output +3.6 | L39:mlp_output +2.8 | L38:attention_output +5.0 |
| back_front:long:multiple_choice:plant | 1.1 | 15.9 | L39:attention_output +2.5 | L39:attention_output +4.2 | L39:attention_output +3.0 | L39:attention_output +7.8 |
| back_front:long:multiple_choice:time | 1.0 | 17.6 | L37:attention_output +0.0 | L39:attention_output +6.6 | L37:attention_output +0.0 | L38:attention_output +4.9 |
| back_front:long:quoted_answer:container | 21.2 | 244.4 | L39:attention_output +7.2 | L39:attention_output +73.1 | L39:attention_output +6.2 | L39:attention_output +58.2 |
| back_front:long:quoted_answer:number | 86.1 | 179.0 | L40:mlp_output +36.8 | L39:attention_output +72.1 | L39:mlp_output +28.8 | L39:attention_output +60.0 |
| back_front:long:quoted_answer:plant | 97.9 | 383.1 | L39:mlp_output +13.2 | L39:attention_output +146.6 | L37:mlp_output +3.8 | L39:attention_output +268.1 |
| back_front:long:quoted_answer:time | 187.4 | 298.6 | L39:mlp_output +58.2 | L39:attention_output +110.9 | L40:attention_output -8.9 | L39:attention_output +78.1 |
| back_front:neutral:answer_one_word:container | 36.0 | 36.1 | L39:attention_output +9.2 | L40:mlp_output +2.9 | L39:attention_output +14.8 | L40:mlp_output +9.4 |
| back_front:neutral:answer_one_word:number | 35.9 | 14.5 | L37:mlp_output +1.2 | L40:mlp_output +8.6 | L38:attention_output +16.2 | L40:mlp_output +4.9 |
| back_front:neutral:answer_one_word:plant | 22.1 | 190.9 | L39:attention_output +13.4 | L40:attention_output +1.9 | L39:attention_output +12.0 | L39:attention_output +7.4 |
| back_front:neutral:answer_one_word:time | 32.8 | 18.9 | L39:attention_output +16.8 | L40:mlp_output +9.6 | L39:attention_output +26.5 | L40:mlp_output +7.9 |
| back_front:neutral:label_colon:container | 74.0 | 34.1 | L40:mlp_output +7.9 | L40:mlp_output +32.9 | L40:mlp_output +52.2 | L39:attention_output +18.6 |
| back_front:neutral:label_colon:number | 66.5 | 7.1 | L40:mlp_output +38.8 | L40:mlp_output +34.8 | L39:attention_output +44.4 | L40:mlp_output +24.2 |
| back_front:neutral:label_colon:plant | 6.6 | 58.9 | L38:mlp_output +3.0 | L40:mlp_output +33.6 | L39:attention_output +10.4 | L38:mlp_output +15.9 |
| back_front:neutral:label_colon:time | 81.8 | 29.6 | L40:attention_output +2.2 | L40:mlp_output +31.6 | L39:attention_output +73.8 | L40:mlp_output +22.5 |
| back_front:neutral:list_answer:container | 16.5 | 8.2 | L40:mlp_output +3.2 | L40:mlp_output +20.0 | L39:attention_output +7.8 | L40:mlp_output +21.8 |
| back_front:neutral:list_answer:number | 19.2 | 7.4 | L39:attention_output +6.9 | L40:mlp_output +10.2 | L39:attention_output +10.0 | L40:mlp_output +15.4 |
| back_front:neutral:list_answer:plant | 33.1 | 11.2 | L39:attention_output +18.5 | L40:mlp_output +11.5 | L39:attention_output +33.0 | L40:mlp_output +12.2 |
| back_front:neutral:list_answer:time | 28.6 | 9.4 | L40:mlp_output +7.1 | L40:mlp_output +16.2 | L40:mlp_output +22.1 | L40:mlp_output +13.8 |
| back_front:neutral:multiple_choice:container | 1.0 | 28.2 | L40:mlp_output +0.1 | L39:mlp_output +15.6 | L40:mlp_output +0.2 | L39:mlp_output +15.0 |
| back_front:neutral:multiple_choice:number | 3.8 | 42.9 | L40:attention_output +0.0 | L39:mlp_output +25.1 | L40:attention_output +0.0 | L39:mlp_output +25.9 |
| back_front:neutral:multiple_choice:plant | 1.0 | 28.2 | L40:mlp_output +1.2 | L39:mlp_output +16.0 | L40:mlp_output +1.4 | L39:mlp_output +14.8 |
| back_front:neutral:multiple_choice:time | 4.0 | 45.5 | L39:attention_output +0.4 | L39:mlp_output +25.4 | L39:attention_output +0.8 | L39:mlp_output +29.1 |
| back_front:neutral:quoted_answer:container | 5.4 | 392.9 | L38:mlp_output +0.4 | L39:attention_output +89.9 | L39:mlp_output -0.1 | L39:attention_output +458.0 |
| back_front:neutral:quoted_answer:number | 125.1 | 276.1 | L39:attention_output +8.0 | L39:attention_output +57.8 | L39:mlp_output +133.9 | L39:attention_output +391.2 |
| back_front:neutral:quoted_answer:plant | 18.4 | 322.9 | L39:mlp_output +6.8 | L39:attention_output +105.9 | L38:attention_output +8.4 | L39:attention_output +464.1 |
| back_front:neutral:quoted_answer:time | 95.6 | 253.1 | L40:mlp_output +10.0 | L39:attention_output +67.4 | L39:mlp_output +133.5 | L39:attention_output +198.1 |
| back_front:short:answer_one_word:container | 2.4 | 31.0 | L40:mlp_output +0.2 | L40:mlp_output +29.1 | L38:mlp_output +0.1 | L40:mlp_output +47.0 |
| back_front:short:answer_one_word:number | 3.4 | 18.1 | L40:attention_output -0.1 | L40:mlp_output +17.4 | L40:attention_output +0.0 | L40:mlp_output +36.8 |
| back_front:short:answer_one_word:plant | 2.8 | 531.1 | L39:attention_output +0.6 | L39:attention_output +54.1 | L39:attention_output +1.0 | L39:attention_output +1064.2 |
| back_front:short:answer_one_word:time | 4.6 | 26.0 | L40:attention_output +0.1 | L40:mlp_output +19.0 | L37:mlp_output +0.0 | L40:mlp_output +43.8 |
| back_front:short:label_colon:container | 2.2 | 129.9 | L39:attention_output +1.1 | L40:mlp_output +70.9 | L38:attention_output +1.5 | L40:mlp_output +102.5 |
| back_front:short:label_colon:number | 9.2 | 48.8 | L40:mlp_output +5.1 | L40:mlp_output +66.8 | L40:attention_output -0.5 | L40:mlp_output +36.6 |
| back_front:short:label_colon:plant | 1.2 | 71.1 | L39:mlp_output +0.6 | L40:mlp_output +53.8 | L38:attention_output +0.4 | L40:mlp_output +67.1 |
| back_front:short:label_colon:time | 2.9 | 69.8 | L39:attention_output +2.1 | L40:mlp_output +84.0 | L39:attention_output +2.1 | L40:mlp_output +124.5 |
| back_front:short:list_answer:container | 5.1 | 26.8 | L38:mlp_output +0.1 | L40:mlp_output +2.5 | L40:mlp_output +6.6 | L40:mlp_output +7.0 |
| back_front:short:list_answer:number | 5.9 | 25.0 | L40:mlp_output +1.1 | L40:mlp_output +2.4 | L38:mlp_output +0.0 | L39:mlp_output +4.1 |
| back_front:short:list_answer:plant | 3.4 | 52.9 | L40:mlp_output +0.8 | L40:mlp_output +8.1 | L38:attention_output +0.8 | L39:attention_output +18.1 |
| back_front:short:list_answer:time | 5.6 | 35.6 | L39:attention_output +0.6 | L39:attention_output +3.0 | L40:mlp_output +5.4 | L39:mlp_output +10.9 |
| back_front:short:multiple_choice:container | 1.0 | 5.5 | L40:mlp_output +1.0 | L40:mlp_output +1.9 | L40:mlp_output +1.0 | L39:mlp_output +2.6 |
| back_front:short:multiple_choice:number | 1.0 | 12.0 | L40:mlp_output +0.2 | L39:mlp_output +4.9 | L40:mlp_output +0.4 | L39:mlp_output +4.0 |
| back_front:short:multiple_choice:plant | 1.0 | 4.0 | L40:mlp_output +0.8 | L39:mlp_output +0.1 | L40:mlp_output +0.6 | L39:mlp_output +0.2 |
| back_front:short:multiple_choice:time | 1.0 | 8.2 | L37:attention_output +0.0 | L39:mlp_output +2.0 | L39:mlp_output +0.4 | L39:mlp_output +3.9 |
| back_front:short:quoted_answer:container | 10.9 | 240.2 | L40:mlp_output +10.0 | L40:mlp_output +64.9 | L39:mlp_output +0.9 | L39:attention_output +645.0 |
| back_front:short:quoted_answer:number | 10.1 | 91.4 | L39:attention_output +6.6 | L40:mlp_output +60.5 | L39:attention_output +5.4 | L39:attention_output +84.9 |
| back_front:short:quoted_answer:plant | 8.1 | 135.6 | L40:mlp_output +9.1 | L40:mlp_output +76.4 | L39:attention_output +3.4 | L38:attention_output +131.1 |
| back_front:short:quoted_answer:time | 13.8 | 152.4 | L40:mlp_output +4.9 | L40:mlp_output +72.4 | L39:mlp_output +0.5 | L39:attention_output +501.4 |
| front_back:long:answer_one_word:container | 21.9 | 194.2 | L39:mlp_output +4.9 | L39:attention_output +133.2 | L37:mlp_output +0.6 | L39:attention_output +227.0 |
| front_back:long:answer_one_word:number | 26.5 | 161.1 | L39:mlp_output +7.2 | L39:attention_output +129.4 | L37:mlp_output +0.5 | L39:attention_output +183.0 |
| front_back:long:answer_one_word:plant | 18.9 | 328.0 | L39:mlp_output +6.5 | L39:attention_output +106.8 | L37:mlp_output +0.9 | L39:attention_output +105.6 |
| front_back:long:answer_one_word:time | 25.4 | 166.0 | L39:mlp_output +8.8 | L39:attention_output +102.8 | L37:mlp_output -0.1 | L39:attention_output +128.9 |
| front_back:long:label_colon:container | 93.6 | 58.1 | L39:mlp_output +25.0 | L39:attention_output +78.1 | L39:attention_output +37.0 | L39:attention_output +73.6 |
| front_back:long:label_colon:number | 13.2 | 61.0 | L40:mlp_output +7.2 | L39:attention_output +91.4 | L39:attention_output +2.0 | L39:attention_output +73.4 |
| front_back:long:label_colon:plant | 19.8 | 72.9 | L39:attention_output +1.6 | L39:attention_output +70.0 | L39:attention_output +32.5 | L39:attention_output +75.8 |
| front_back:long:label_colon:time | 38.8 | 70.9 | L40:mlp_output +39.6 | L39:attention_output +75.0 | L39:attention_output +71.5 | L39:attention_output +89.6 |
| front_back:long:list_answer:container | 30.2 | 17.6 | L40:mlp_output +23.4 | L39:attention_output +3.1 | L40:mlp_output +19.4 | L39:attention_output +1.4 |
| front_back:long:list_answer:number | 20.4 | 13.1 | L40:mlp_output +13.0 | L39:attention_output +2.6 | L40:mlp_output +3.4 | L39:attention_output +2.2 |
| front_back:long:list_answer:plant | 22.0 | 20.4 | L40:mlp_output +17.4 | L39:attention_output +4.4 | L40:mlp_output +9.4 | L39:attention_output +3.8 |
| front_back:long:list_answer:time | 31.1 | 21.4 | L40:mlp_output +27.6 | L39:attention_output +2.5 | L40:mlp_output +26.6 | L39:attention_output +3.0 |
| front_back:long:multiple_choice:container | 1.0 | 21.5 | L40:mlp_output +3.6 | L39:attention_output +2.4 | L40:mlp_output +4.1 | L39:attention_output +12.6 |
| front_back:long:multiple_choice:number | 3.1 | 17.4 | L39:mlp_output +2.4 | L39:attention_output +3.8 | L39:mlp_output +1.4 | L39:attention_output +15.6 |
| front_back:long:multiple_choice:plant | 1.0 | 14.4 | L39:mlp_output +2.2 | L39:attention_output +5.1 | L39:mlp_output +1.0 | L39:attention_output +13.6 |
| front_back:long:multiple_choice:time | 1.2 | 21.9 | L38:mlp_output +0.1 | L39:attention_output +5.4 | L39:mlp_output +0.5 | L39:attention_output +12.4 |
| front_back:long:quoted_answer:container | 38.1 | 205.9 | L39:attention_output +4.2 | L39:attention_output +88.5 | L39:mlp_output +6.8 | L39:attention_output +21.4 |
| front_back:long:quoted_answer:number | 93.0 | 199.8 | L40:mlp_output +35.6 | L39:attention_output +114.5 | L37:attention_output +5.6 | L39:attention_output +56.8 |
| front_back:long:quoted_answer:plant | 60.9 | 254.0 | L40:mlp_output +14.6 | L39:attention_output +107.8 | L39:mlp_output +11.1 | L39:attention_output +27.6 |
| front_back:long:quoted_answer:time | 175.4 | 232.9 | L39:mlp_output +44.2 | L39:attention_output +111.5 | L40:attention_output -3.9 | L39:attention_output +23.9 |
| front_back:neutral:answer_one_word:container | 36.5 | 69.8 | L38:mlp_output +0.9 | L40:attention_output +0.4 | L38:attention_output +9.0 | L39:attention_output +57.2 |
| front_back:neutral:answer_one_word:number | 21.5 | 42.1 | L40:attention_output +0.2 | L40:attention_output +3.0 | L38:attention_output +11.6 | L39:attention_output +14.2 |
| front_back:neutral:answer_one_word:plant | 16.4 | 252.8 | L39:attention_output +1.0 | L40:attention_output -4.1 | L38:attention_output +4.5 | L39:attention_output +205.6 |
| front_back:neutral:answer_one_word:time | 28.8 | 86.4 | L39:attention_output +0.6 | L40:attention_output +6.6 | L38:attention_output +16.2 | L39:attention_output +117.6 |
| front_back:neutral:label_colon:container | 116.6 | 42.0 | L38:mlp_output +19.6 | L40:mlp_output +96.9 | L39:attention_output +26.2 | L39:attention_output +18.1 |
| front_back:neutral:label_colon:number | 48.8 | 23.8 | L40:mlp_output +46.0 | L40:mlp_output +69.6 | L39:attention_output +42.1 | L40:mlp_output +33.5 |
| front_back:neutral:label_colon:plant | 5.4 | 67.8 | L39:mlp_output +4.5 | L40:mlp_output +133.5 | L39:attention_output +12.4 | L40:mlp_output +66.0 |
| front_back:neutral:label_colon:time | 5.9 | 59.6 | L40:mlp_output +1.1 | L40:mlp_output +64.4 | L39:attention_output +9.1 | L39:attention_output +38.9 |
| front_back:neutral:list_answer:container | 40.0 | 11.0 | L39:mlp_output +4.1 | L40:mlp_output +20.1 | L39:mlp_output +17.0 | L40:mlp_output +16.8 |
| front_back:neutral:list_answer:number | 17.9 | 7.1 | L39:mlp_output +6.4 | L40:mlp_output +16.6 | L39:mlp_output +7.5 | L40:mlp_output +12.6 |
| front_back:neutral:list_answer:plant | 35.5 | 8.9 | L39:attention_output +19.4 | L40:mlp_output +11.5 | L39:attention_output +22.8 | L40:mlp_output +6.1 |
| front_back:neutral:list_answer:time | 41.6 | 10.9 | L40:mlp_output +12.5 | L40:mlp_output +13.9 | L39:attention_output +20.4 | L40:mlp_output +9.1 |
| front_back:neutral:multiple_choice:container | 1.1 | 29.2 | L40:mlp_output +0.2 | L39:mlp_output +24.5 | L40:mlp_output +0.2 | L39:mlp_output +30.8 |
| front_back:neutral:multiple_choice:number | 1.2 | 39.8 | L39:attention_output +0.9 | L39:mlp_output +25.5 | L39:attention_output +1.1 | L39:mlp_output +42.5 |
| front_back:neutral:multiple_choice:plant | 1.0 | 23.5 | L39:mlp_output +0.5 | L39:mlp_output +17.6 | L39:mlp_output +0.2 | L39:mlp_output +20.1 |
| front_back:neutral:multiple_choice:time | 2.4 | 45.4 | L40:attention_output +0.1 | L39:mlp_output +29.2 | L40:attention_output +0.1 | L39:mlp_output +33.4 |
| front_back:neutral:quoted_answer:container | 8.0 | 448.6 | L38:attention_output +0.2 | L39:attention_output +149.5 | L40:attention_output -0.1 | L39:attention_output +1202.6 |
| front_back:neutral:quoted_answer:number | 96.0 | 282.6 | L39:attention_output +15.1 | L39:attention_output +80.6 | L39:mlp_output +141.5 | L39:attention_output +440.6 |
| front_back:neutral:quoted_answer:plant | 15.8 | 236.8 | L39:attention_output +12.5 | L39:attention_output +103.0 | L39:attention_output +17.4 | L39:attention_output +771.1 |
| front_back:neutral:quoted_answer:time | 108.1 | 339.5 | L39:mlp_output +4.6 | L39:attention_output +109.8 | L39:mlp_output +170.4 | L39:attention_output +1093.5 |
| front_back:short:answer_one_word:container | 4.9 | 28.1 | L40:attention_output +0.0 | L40:mlp_output +24.1 | L40:attention_output +0.0 | L40:mlp_output +55.5 |
| front_back:short:answer_one_word:number | 2.6 | 10.1 | L40:mlp_output +0.5 | L40:mlp_output +11.5 | L40:mlp_output +0.8 | L40:mlp_output +25.4 |
| front_back:short:answer_one_word:plant | 2.8 | 649.2 | L39:mlp_output +1.2 | L39:attention_output +45.2 | L39:attention_output +1.5 | L39:attention_output +1621.1 |
| front_back:short:answer_one_word:time | 5.5 | 57.9 | L38:attention_output +0.9 | L40:mlp_output +27.0 | L39:attention_output +1.2 | L39:attention_output +175.5 |
| front_back:short:label_colon:container | 9.0 | 61.0 | L39:attention_output +2.5 | L40:mlp_output +59.5 | L39:attention_output +4.0 | L40:mlp_output +89.4 |
| front_back:short:label_colon:number | 8.8 | 60.5 | L39:mlp_output +3.0 | L40:mlp_output +53.6 | L40:attention_output +0.9 | L40:mlp_output +99.0 |
| front_back:short:label_colon:plant | 2.9 | 84.8 | L39:mlp_output +1.6 | L40:mlp_output +53.1 | L38:attention_output +2.1 | L40:mlp_output +87.5 |
| front_back:short:label_colon:time | 5.5 | 100.1 | L39:attention_output +2.0 | L40:mlp_output +75.4 | L39:attention_output +1.4 | L40:mlp_output +148.9 |
| front_back:short:list_answer:container | 6.2 | 29.6 | L40:mlp_output +0.2 | L40:mlp_output +8.0 | L40:mlp_output +11.8 | L40:mlp_output +12.5 |
| front_back:short:list_answer:number | 4.4 | 17.4 | L40:attention_output +0.0 | L40:mlp_output +10.8 | L38:attention_output +0.4 | L40:mlp_output +15.6 |
| front_back:short:list_answer:plant | 1.5 | 51.2 | L39:mlp_output +2.0 | L40:mlp_output +17.9 | L39:attention_output +3.1 | L39:attention_output +26.9 |
| front_back:short:list_answer:time | 11.4 | 35.5 | L40:attention_output -0.1 | L40:mlp_output +14.5 | L40:mlp_output +2.5 | L40:mlp_output +21.0 |
| front_back:short:multiple_choice:container | 1.0 | 5.6 | L37:attention_output +0.0 | L39:mlp_output +1.4 | L39:mlp_output +0.6 | L40:mlp_output +2.5 |
| front_back:short:multiple_choice:number | 1.2 | 6.4 | L40:mlp_output +0.2 | L39:mlp_output +2.6 | L39:mlp_output +0.2 | L38:attention_output +2.9 |
| front_back:short:multiple_choice:plant | 1.0 | 4.4 | L40:mlp_output +0.5 | L39:mlp_output +0.2 | L39:mlp_output +0.4 | L40:mlp_output +0.2 |
| front_back:short:multiple_choice:time | 1.4 | 6.0 | L37:attention_output +0.0 | L39:mlp_output +1.2 | L37:attention_output +0.0 | L38:attention_output +2.1 |
| front_back:short:quoted_answer:container | 19.4 | 139.9 | L40:mlp_output +21.4 | L40:mlp_output +20.4 | L39:mlp_output +3.4 | L39:attention_output +617.4 |
| front_back:short:quoted_answer:number | 8.9 | 64.2 | L39:attention_output +2.8 | L40:mlp_output +31.8 | L39:attention_output +2.2 | L39:attention_output +108.0 |
| front_back:short:quoted_answer:plant | 6.1 | 134.2 | L40:mlp_output +9.2 | L40:mlp_output +44.9 | L39:mlp_output +5.4 | L39:attention_output +277.6 |
| front_back:short:quoted_answer:time | 20.5 | 81.4 | L40:mlp_output +18.0 | L40:mlp_output +26.8 | L39:attention_output +17.0 | L39:attention_output +54.2 |

## deepseek7b

cases=120, patch_layers=[25, 26, 27, 28], formats=label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice

### By category

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 45.0 | 73.3 | +21.8 | +1477.6 | +97.7 | +524.8 | L28:mlp_output | L28:attention_output | L28:mlp_output |
| number | 30 | 14.0 | 63.4 | +14.2 | +1282.9 | +25.4 | +209.7 | L28:mlp_output | L28:attention_output | L28:mlp_output |
| plant | 30 | 10.7 | 210.4 | +6.5 | +1785.8 | +13.7 | +485.8 | L28:mlp_output | L28:attention_output | L28:mlp_output |
| time | 30 | 28.6 | 295.6 | +13.9 | +3769.2 | +136.0 | +1941.2 | L28:mlp_output | L28:mlp_output | L28:mlp_output |

### By format

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 21.7 | 4.8 | +18.3 | +2070.5 | +23.8 | +16.8 | L28:mlp_output | L28:mlp_output | L28:mlp_output |
| label_colon | 24 | 58.4 | 102.4 | +27.9 | +702.9 | +27.7 | +77.8 | L25:attention_output | L28:attention_output | L25:attention_output |
| list_answer | 24 | 22.9 | 7.6 | +11.9 | +7.3 | +11.9 | +4.9 | L28:attention_output | L28:attention_output | L28:attention_output |
| multiple_choice | 24 | 2.2 | 18.8 | +2.5 | +6.6 | +2.7 | +14.5 | L28:mlp_output | L27:mlp_output | L28:mlp_output |
| quoted_answer | 24 | 17.6 | 669.9 | +9.9 | +7607.0 | +275.0 | +3837.8 | L28:mlp_output | L28:mlp_output | L28:mlp_output |

### By family

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 19.4 | 9.1 | +11.4 | +4.9 | +13.3 | +7.6 | L28:attention_output | L28:attention_output | L28:attention_output |
| neutral | 40 | 32.8 | 167.2 | +22.6 | +3885.0 | +84.4 | +1386.2 | L28:mlp_output | L28:mlp_output | L28:mlp_output |
| short | 40 | 21.6 | 305.8 | +8.3 | +2346.7 | +106.9 | +977.3 | L28:mlp_output | L28:mlp_output | L28:mlp_output |

### By split

| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |
|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 23.0 | 95.3 | +13.2 | +1615.9 | +45.8 | +566.1 | L28:mlp_output | L28:attention_output | L28:mlp_output |
| front_back | 60 | 26.2 | 226.1 | +15.0 | +2541.8 | +90.6 | +1014.7 | L28:mlp_output | L28:mlp_output | L28:mlp_output |

### By Writer Condition

| mode | component | layer | n | answer_rank_delta | format_rank_delta | answer_argmax_delta | format_argmax_delta |
|---|---|---|---|---|---|---|---|
| format_proj | attention_output | L25 | 120 | +0.8 | -14.2 | -0.006 | +0.005 |
| format_proj | attention_output | L26 | 120 | +1.3 | -36.1 | +0.002 | -0.002 |
| format_proj | attention_output | L27 | 120 | -0.3 | -30.2 | +0.006 | +0.000 |
| format_proj | attention_output | L28 | 120 | -2.4 | +74.0 | +0.009 | -0.029 |
| format_proj | mlp_output | L25 | 120 | -0.1 | -5.6 | +0.004 | -0.015 |
| format_proj | mlp_output | L26 | 120 | -0.1 | +83.1 | +0.003 | -0.014 |
| format_proj | mlp_output | L27 | 120 | -0.8 | -4.7 | +0.011 | -0.019 |
| format_proj | mlp_output | L28 | 120 | +73.6 | +2015.9 | +0.024 | +0.042 |
| joint_proj | attention_output | L25 | 120 | +0.6 | -36.0 | -0.004 | +0.001 |
| joint_proj | attention_output | L26 | 120 | -0.0 | -22.4 | +0.002 | -0.003 |
| joint_proj | attention_output | L27 | 120 | -0.0 | -9.0 | +0.000 | +0.001 |
| joint_proj | attention_output | L28 | 120 | -3.5 | -25.3 | +0.006 | -0.021 |
| joint_proj | mlp_output | L25 | 120 | +0.1 | +29.4 | +0.008 | -0.004 |
| joint_proj | mlp_output | L26 | 120 | +3.4 | +90.7 | -0.005 | +0.003 |
| joint_proj | mlp_output | L27 | 120 | +0.1 | -11.2 | -0.005 | -0.003 |
| joint_proj | mlp_output | L28 | 120 | +49.5 | +680.6 | -0.033 | +0.061 |
| semantic_proj | attention_output | L25 | 120 | +1.2 | +6.3 | +0.001 | -0.001 |
| semantic_proj | attention_output | L26 | 120 | -0.5 | -27.1 | +0.001 | -0.005 |
| semantic_proj | attention_output | L27 | 120 | -0.6 | -9.9 | -0.003 | +0.000 |
| semantic_proj | attention_output | L28 | 120 | -2.9 | -28.2 | -0.002 | +0.011 |
| semantic_proj | mlp_output | L25 | 120 | +2.0 | +34.4 | -0.002 | -0.003 |
| semantic_proj | mlp_output | L26 | 120 | +0.7 | +26.4 | -0.006 | +0.009 |
| semantic_proj | mlp_output | L27 | 120 | +0.6 | -41.9 | -0.008 | +0.011 |
| semantic_proj | mlp_output | L28 | 120 | -1.2 | -91.4 | -0.044 | +0.003 |

### Cases

| case | clean_ans | clean_fmt | sem_ans | fmt_fmt | joint_ans | joint_fmt |
|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 17.6 | 6.5 | L28:attention_output +8.0 | L28:attention_output +2.5 | L28:attention_output +9.9 | L28:attention_output +2.6 |
| back_front:long:answer_one_word:number | 30.0 | 5.8 | L28:attention_output +13.8 | L28:attention_output +1.1 | L28:attention_output +23.4 | L28:attention_output +1.4 |
| back_front:long:answer_one_word:plant | 10.6 | 6.6 | L28:mlp_output +35.8 | L28:attention_output +1.5 | L28:mlp_output +42.4 | L28:attention_output +2.0 |
| back_front:long:answer_one_word:time | 31.5 | 5.5 | L28:attention_output +31.2 | L28:attention_output +2.1 | L28:attention_output +42.0 | L28:attention_output +1.1 |
| back_front:long:label_colon:container | 26.0 | 5.9 | L25:attention_output +2.5 | L28:attention_output +3.1 | L25:attention_output +3.8 | L28:mlp_output +7.1 |
| back_front:long:label_colon:number | 7.6 | 7.5 | L27:mlp_output +5.5 | L28:attention_output +3.5 | L27:mlp_output +3.8 | L28:attention_output +5.4 |
| back_front:long:label_colon:plant | 12.2 | 7.8 | L28:mlp_output +2.0 | L28:attention_output +5.0 | L25:attention_output +2.1 | L28:mlp_output +7.1 |
| back_front:long:label_colon:time | 88.1 | 6.5 | L27:mlp_output +29.4 | L28:attention_output +3.4 | L25:mlp_output +19.4 | L28:attention_output +5.0 |
| back_front:long:list_answer:container | 13.4 | 6.4 | L28:attention_output +10.9 | L28:attention_output +2.5 | L28:attention_output +13.6 | L28:attention_output +5.1 |
| back_front:long:list_answer:number | 29.6 | 3.0 | L28:attention_output +29.0 | L28:attention_output +1.1 | L28:attention_output +31.2 | L28:attention_output +2.1 |
| back_front:long:list_answer:plant | 23.8 | 7.4 | L28:attention_output +14.8 | L28:attention_output +2.6 | L28:attention_output +15.8 | L28:attention_output +2.9 |
| back_front:long:list_answer:time | 26.4 | 7.4 | L28:attention_output +24.1 | L28:attention_output +4.6 | L28:attention_output +27.8 | L28:attention_output +7.2 |
| back_front:long:multiple_choice:container | 1.0 | 11.9 | L28:attention_output +0.1 | L28:attention_output +10.0 | L28:attention_output +0.1 | L28:attention_output +14.9 |
| back_front:long:multiple_choice:number | 1.5 | 14.9 | L28:mlp_output +1.6 | L28:attention_output +4.2 | L28:mlp_output +2.4 | L28:attention_output +16.4 |
| back_front:long:multiple_choice:plant | 1.1 | 6.5 | L28:mlp_output +1.1 | L28:attention_output +1.2 | L28:mlp_output +0.2 | L28:attention_output +4.4 |
| back_front:long:multiple_choice:time | 1.2 | 11.5 | L28:mlp_output +1.4 | L28:attention_output +6.8 | L28:mlp_output +3.9 | L28:attention_output +13.2 |
| back_front:long:quoted_answer:container | 5.8 | 8.0 | L28:attention_output +2.4 | L28:attention_output +5.0 | L25:mlp_output +2.9 | L28:attention_output +5.9 |
| back_front:long:quoted_answer:number | 8.9 | 14.9 | L28:mlp_output +2.2 | L28:attention_output +12.2 | L28:attention_output +1.8 | L28:attention_output +19.6 |
| back_front:long:quoted_answer:plant | 6.0 | 13.2 | L25:mlp_output +0.9 | L28:attention_output +7.9 | L26:mlp_output +1.0 | L28:attention_output +7.2 |
| back_front:long:quoted_answer:time | 10.1 | 30.2 | L28:attention_output +1.6 | L28:attention_output +23.6 | L28:attention_output +2.5 | L28:attention_output +29.2 |
| back_front:neutral:answer_one_word:container | 38.8 | 6.5 | L28:mlp_output +45.6 | L28:mlp_output +9515.6 | L28:mlp_output +116.4 | L28:mlp_output +106.9 |
| back_front:neutral:answer_one_word:number | 22.9 | 4.6 | L28:mlp_output +24.0 | L28:mlp_output +17846.4 | L28:mlp_output +38.4 | L28:mlp_output +93.6 |
| back_front:neutral:answer_one_word:plant | 24.8 | 7.2 | L26:attention_output -1.9 | L28:mlp_output +11732.8 | L26:attention_output +2.8 | L28:mlp_output +64.0 |
| back_front:neutral:answer_one_word:time | 32.9 | 4.9 | L28:mlp_output +28.6 | L28:mlp_output +10308.6 | L28:mlp_output +38.4 | L28:mlp_output +65.4 |
| back_front:neutral:label_colon:container | 78.0 | 3.0 | L27:mlp_output +44.6 | L28:attention_output +12.2 | L27:mlp_output +50.5 | L28:mlp_output +7.9 |
| back_front:neutral:label_colon:number | 26.5 | 7.6 | L28:mlp_output +53.9 | L28:attention_output +7.5 | L28:mlp_output +33.6 | L27:attention_output +2.6 |
| back_front:neutral:label_colon:plant | 22.2 | 4.9 | L28:attention_output +1.6 | L28:attention_output +14.2 | L28:attention_output +3.2 | L28:mlp_output +5.9 |
| back_front:neutral:label_colon:time | 129.1 | 6.4 | L26:attention_output -2.4 | L26:mlp_output +3.2 | L26:attention_output -1.4 | L27:attention_output -0.1 |
| back_front:neutral:list_answer:container | 33.8 | 7.0 | L25:attention_output +31.5 | L27:mlp_output +4.2 | L27:attention_output +15.4 | L28:attention_output +6.6 |
| back_front:neutral:list_answer:number | 23.1 | 3.6 | L25:attention_output +5.0 | L28:attention_output +4.5 | L26:attention_output +8.4 | L28:attention_output +6.0 |
| back_front:neutral:list_answer:plant | 13.8 | 10.5 | L25:attention_output +1.6 | L28:attention_output +9.0 | L26:attention_output -1.1 | L28:attention_output +9.6 |
| back_front:neutral:list_answer:time | 62.9 | 10.0 | L25:attention_output +2.5 | L27:mlp_output +8.4 | L25:attention_output +7.5 | L28:attention_output +9.5 |
| back_front:neutral:multiple_choice:container | 2.6 | 35.8 | L26:attention_output +3.6 | L27:mlp_output +11.8 | L26:attention_output +2.4 | L27:mlp_output +16.6 |
| back_front:neutral:multiple_choice:number | 1.5 | 26.0 | L28:mlp_output +4.8 | L27:mlp_output +6.2 | L28:mlp_output +4.4 | L27:mlp_output +5.4 |
| back_front:neutral:multiple_choice:plant | 4.1 | 40.1 | L27:attention_output -1.0 | L27:mlp_output +22.9 | L25:mlp_output -0.8 | L28:mlp_output +16.4 |
| back_front:neutral:multiple_choice:time | 8.0 | 31.1 | L28:mlp_output +17.6 | L27:mlp_output +9.8 | L28:mlp_output +22.1 | L28:mlp_output +17.8 |
| back_front:neutral:quoted_answer:container | 55.0 | 407.9 | L28:mlp_output +39.9 | L27:attention_output +305.9 | L25:mlp_output +18.6 | L25:attention_output +270.8 |
| back_front:neutral:quoted_answer:number | 14.0 | 526.5 | L25:mlp_output +14.9 | L27:mlp_output +607.8 | L25:attention_output +3.4 | L26:mlp_output +574.5 |
| back_front:neutral:quoted_answer:plant | 8.2 | 1231.2 | L27:mlp_output +8.8 | L25:mlp_output +893.1 | L28:mlp_output +5.6 | L25:mlp_output +1276.2 |
| back_front:neutral:quoted_answer:time | 27.4 | 569.4 | L28:mlp_output +13.2 | L27:mlp_output +420.4 | L28:mlp_output +54.6 | L26:attention_output +510.5 |
| back_front:short:answer_one_word:container | 10.9 | 2.4 | L26:mlp_output +3.8 | L28:mlp_output +5.5 | L26:mlp_output +3.6 | L28:mlp_output +4.9 |
| back_front:short:answer_one_word:number | 14.4 | 1.8 | L25:mlp_output +3.2 | L28:mlp_output +4.9 | L26:attention_output +6.2 | L28:mlp_output +3.9 |
| back_front:short:answer_one_word:plant | 6.0 | 1.1 | L25:mlp_output +4.6 | L28:mlp_output +9.0 | L25:mlp_output +3.9 | L28:mlp_output +8.2 |
| back_front:short:answer_one_word:time | 8.0 | 2.1 | L25:mlp_output +3.2 | L28:mlp_output +8.5 | L25:mlp_output +4.8 | L28:mlp_output +7.5 |
| back_front:short:label_colon:container | 220.0 | 312.2 | L25:mlp_output +149.6 | L25:mlp_output +422.5 | L26:mlp_output +98.2 | L26:mlp_output +253.2 |
| back_front:short:label_colon:number | 2.0 | 18.4 | L28:attention_output +12.1 | L27:mlp_output +57.2 | L27:mlp_output +4.9 | L27:mlp_output +34.5 |
| back_front:short:label_colon:plant | 3.6 | 423.6 | L25:attention_output +3.1 | L25:mlp_output +912.4 | L25:attention_output +6.4 | L26:mlp_output +339.9 |
| back_front:short:label_colon:time | 21.1 | 225.0 | L25:mlp_output +21.8 | L27:mlp_output +112.8 | L25:mlp_output +17.0 | L26:mlp_output +116.6 |
| back_front:short:list_answer:container | 14.0 | 12.8 | L25:attention_output +2.8 | L28:attention_output +9.1 | L25:attention_output +2.4 | L27:mlp_output +3.1 |
| back_front:short:list_answer:number | 21.9 | 5.8 | L25:attention_output +1.9 | L28:attention_output +8.6 | L27:attention_output +2.5 | L26:mlp_output +1.5 |
| back_front:short:list_answer:plant | 5.4 | 10.0 | L25:attention_output +0.5 | L28:attention_output +12.4 | L26:attention_output +1.9 | L27:mlp_output +2.6 |
| back_front:short:list_answer:time | 10.8 | 10.0 | L28:mlp_output +7.8 | L28:attention_output +11.0 | L27:mlp_output +6.4 | L27:mlp_output +6.1 |
| back_front:short:multiple_choice:container | 1.0 | 13.4 | L28:mlp_output +0.1 | L28:attention_output +1.1 | L28:mlp_output +0.1 | L28:attention_output +3.0 |
| back_front:short:multiple_choice:number | 1.5 | 14.0 | L28:mlp_output +1.8 | L27:mlp_output +2.5 | L28:mlp_output +1.6 | L28:mlp_output +3.6 |
| back_front:short:multiple_choice:plant | 1.2 | 4.6 | L27:attention_output +0.1 | L26:attention_output +0.2 | L27:attention_output +0.1 | L28:mlp_output +4.0 |
| back_front:short:multiple_choice:time | 1.1 | 10.9 | L28:mlp_output +2.2 | L27:mlp_output +1.2 | L28:mlp_output +1.8 | L28:mlp_output +2.0 |
| back_front:short:quoted_answer:container | 6.2 | 54.0 | L25:attention_output +4.2 | L28:mlp_output +2746.4 | L28:mlp_output +721.4 | L28:mlp_output +1780.5 |
| back_front:short:quoted_answer:number | 5.8 | 86.4 | L27:attention_output +1.2 | L28:mlp_output +3082.6 | L28:mlp_output +251.1 | L28:mlp_output +2341.8 |
| back_front:short:quoted_answer:plant | 21.2 | 208.4 | L25:attention_output +9.4 | L28:mlp_output +8525.2 | L28:mlp_output +136.8 | L28:mlp_output +3176.4 |
| back_front:short:quoted_answer:time | 19.5 | 1171.0 | L28:mlp_output +4.8 | L28:mlp_output +29171.6 | L28:mlp_output +802.8 | L28:mlp_output +22644.4 |
| front_back:long:answer_one_word:container | 43.4 | 5.6 | L28:attention_output +24.9 | L28:mlp_output +2.6 | L28:attention_output +40.4 | L28:attention_output +1.5 |
| front_back:long:answer_one_word:number | 36.2 | 5.9 | L28:attention_output +14.2 | L28:attention_output +1.1 | L28:attention_output +14.9 | L28:attention_output +0.5 |
| front_back:long:answer_one_word:plant | 15.5 | 8.5 | L28:mlp_output +10.9 | L28:attention_output +3.6 | L28:mlp_output +10.1 | L28:mlp_output +3.5 |
| front_back:long:answer_one_word:time | 32.8 | 6.0 | L28:attention_output +28.0 | L28:mlp_output +1.8 | L28:attention_output +46.2 | L28:attention_output +1.4 |
| front_back:long:label_colon:container | 50.2 | 6.6 | L25:mlp_output +18.0 | L28:attention_output +3.5 | L25:attention_output +6.0 | L28:attention_output +4.5 |
| front_back:long:label_colon:number | 2.8 | 9.4 | L27:mlp_output +2.0 | L28:attention_output +5.9 | L27:mlp_output +2.1 | L28:attention_output +7.5 |
| front_back:long:label_colon:plant | 10.8 | 8.0 | L25:attention_output +1.5 | L28:attention_output +5.8 | L25:attention_output +1.5 | L28:attention_output +8.1 |
| front_back:long:label_colon:time | 56.1 | 8.9 | L28:attention_output +33.8 | L28:attention_output +5.6 | L28:attention_output +24.4 | L28:attention_output +7.1 |
| front_back:long:list_answer:container | 31.4 | 6.2 | L28:attention_output +24.1 | L28:attention_output +1.6 | L28:attention_output +31.8 | L28:mlp_output +5.6 |
| front_back:long:list_answer:number | 32.8 | 4.1 | L28:attention_output +31.0 | L28:attention_output +2.4 | L28:attention_output +39.4 | L28:attention_output +4.1 |
| front_back:long:list_answer:plant | 22.8 | 5.9 | L28:attention_output +14.5 | L28:attention_output +2.6 | L28:attention_output +17.9 | L28:mlp_output +6.0 |
| front_back:long:list_answer:time | 26.1 | 6.5 | L28:attention_output +16.5 | L28:attention_output +2.0 | L28:attention_output +19.9 | L28:attention_output +2.2 |
| front_back:long:multiple_choice:container | 1.1 | 11.2 | L28:attention_output +0.2 | L28:attention_output +6.9 | L28:attention_output +0.4 | L28:attention_output +12.4 |
| front_back:long:multiple_choice:number | 3.4 | 11.8 | L26:mlp_output +0.8 | L28:attention_output +9.5 | L28:mlp_output +3.6 | L28:attention_output +20.0 |
| front_back:long:multiple_choice:plant | 1.0 | 5.4 | L25:attention_output +0.0 | L27:mlp_output +1.6 | L25:attention_output +0.0 | L28:attention_output +3.2 |
| front_back:long:multiple_choice:time | 3.1 | 10.6 | L28:mlp_output +5.1 | L28:attention_output +7.2 | L28:mlp_output +3.6 | L28:attention_output +15.0 |
| front_back:long:quoted_answer:container | 15.1 | 6.9 | L27:mlp_output +3.2 | L28:attention_output +2.4 | L26:mlp_output +4.2 | L25:mlp_output +2.9 |
| front_back:long:quoted_answer:number | 13.5 | 16.0 | L26:mlp_output +2.0 | L28:attention_output +8.2 | L27:mlp_output +3.2 | L28:attention_output +15.1 |
| front_back:long:quoted_answer:plant | 7.1 | 19.1 | L27:mlp_output +2.6 | L28:attention_output +10.9 | L25:mlp_output +2.1 | L28:attention_output +13.4 |
| front_back:long:quoted_answer:time | 17.0 | 14.8 | L28:attention_output +5.4 | L28:attention_output +7.5 | L26:mlp_output +12.0 | L28:attention_output +9.8 |
| front_back:neutral:answer_one_word:container | 48.6 | 7.2 | L28:mlp_output +54.6 | L28:mlp_output +31.0 | L26:mlp_output +29.0 | L27:mlp_output +2.6 |
| front_back:neutral:answer_one_word:number | 31.1 | 7.9 | L28:mlp_output +23.8 | L28:mlp_output +41.4 | L27:mlp_output +11.8 | L28:mlp_output +5.0 |
| front_back:neutral:answer_one_word:plant | 13.2 | 4.5 | L28:mlp_output +11.0 | L28:mlp_output +50.4 | L28:mlp_output +4.0 | L28:mlp_output +6.0 |
| front_back:neutral:answer_one_word:time | 21.0 | 5.8 | L28:mlp_output +60.2 | L28:mlp_output +43.9 | L28:mlp_output +69.2 | L28:attention_output +1.8 |
| front_back:neutral:label_colon:container | 206.4 | 4.5 | L25:attention_output +116.5 | L28:mlp_output +19.4 | L26:mlp_output +291.6 | L26:mlp_output +12.4 |
| front_back:neutral:label_colon:number | 18.2 | 13.6 | L28:mlp_output +115.5 | L28:mlp_output +67.9 | L28:mlp_output +39.1 | L27:mlp_output +6.5 |
| front_back:neutral:label_colon:plant | 22.4 | 5.9 | L27:attention_output +25.1 | L28:mlp_output +19.6 | L25:attention_output +19.4 | L26:attention_output +5.2 |
| front_back:neutral:label_colon:time | 77.8 | 3.8 | L26:attention_output +3.1 | L28:mlp_output +12.5 | L25:mlp_output +4.4 | L26:mlp_output -0.4 |
| front_back:neutral:list_answer:container | 69.1 | 6.4 | L27:mlp_output +20.8 | L28:mlp_output +15.2 | L27:mlp_output +13.9 | L28:mlp_output +4.4 |
| front_back:neutral:list_answer:number | 16.4 | 4.2 | L28:mlp_output +7.8 | L28:mlp_output +10.4 | L28:mlp_output +5.4 | L28:mlp_output +5.2 |
| front_back:neutral:list_answer:plant | 14.4 | 8.2 | L27:mlp_output +9.5 | L28:mlp_output +11.9 | L26:attention_output +2.9 | L28:mlp_output +6.8 |
| front_back:neutral:list_answer:time | 27.0 | 6.1 | L28:mlp_output +18.0 | L28:mlp_output +20.6 | L28:mlp_output +9.9 | L28:attention_output +5.0 |
| front_back:neutral:multiple_choice:container | 4.9 | 39.6 | L28:mlp_output +6.6 | L27:mlp_output +10.2 | L28:mlp_output +7.4 | L28:mlp_output +85.5 |
| front_back:neutral:multiple_choice:number | 1.4 | 33.2 | L28:mlp_output +1.9 | L27:mlp_output +10.6 | L28:mlp_output +1.1 | L28:mlp_output +22.1 |
| front_back:neutral:multiple_choice:plant | 1.9 | 32.0 | L27:attention_output +3.4 | L27:mlp_output +11.1 | L26:mlp_output +3.4 | L28:mlp_output +26.0 |
| front_back:neutral:multiple_choice:time | 3.4 | 36.4 | L28:mlp_output +4.9 | L27:mlp_output +9.4 | L28:mlp_output +2.8 | L28:mlp_output +18.0 |
| front_back:neutral:quoted_answer:container | 38.9 | 758.9 | L25:mlp_output +19.2 | L28:mlp_output +19849.5 | L28:mlp_output +891.2 | L28:mlp_output +12737.0 |
| front_back:neutral:quoted_answer:number | 20.8 | 317.4 | L28:mlp_output +26.2 | L28:mlp_output +15271.4 | L28:mlp_output +76.9 | L28:mlp_output +2383.8 |
| front_back:neutral:quoted_answer:plant | 17.8 | 791.1 | L25:attention_output +17.1 | L28:mlp_output +19711.6 | L28:mlp_output +56.8 | L28:mlp_output +6325.4 |
| front_back:neutral:quoted_answer:time | 28.9 | 1656.1 | L28:attention_output +21.2 | L28:mlp_output +48435.8 | L28:mlp_output +1413.2 | L28:mlp_output +30722.2 |
| front_back:short:answer_one_word:container | 11.1 | 3.0 | L28:mlp_output +1.2 | L28:mlp_output +25.6 | L25:attention_output +3.0 | L28:mlp_output +5.2 |
| front_back:short:answer_one_word:number | 6.9 | 1.1 | L25:mlp_output +1.9 | L28:mlp_output +10.0 | L27:attention_output +1.5 | L28:mlp_output +1.4 |
| front_back:short:answer_one_word:plant | 6.8 | 1.5 | L28:mlp_output +6.1 | L28:mlp_output +23.5 | L28:mlp_output +5.2 | L28:mlp_output +5.5 |
| front_back:short:answer_one_word:time | 7.0 | 2.0 | L25:mlp_output +3.1 | L28:mlp_output +18.9 | L25:mlp_output +2.6 | L28:mlp_output +7.4 |
| front_back:short:label_colon:container | 262.1 | 293.8 | L25:mlp_output -13.4 | L28:mlp_output +10297.9 | L26:attention_output -5.0 | L26:mlp_output +223.2 |
| front_back:short:label_colon:number | 6.6 | 222.0 | L28:attention_output +18.1 | L28:mlp_output +809.2 | L28:mlp_output +9.6 | L26:mlp_output +629.9 |
| front_back:short:label_colon:plant | 3.6 | 520.2 | L28:mlp_output +9.1 | L28:mlp_output +1219.8 | L25:attention_output +5.0 | L27:attention_output +193.6 |
| front_back:short:label_colon:time | 47.9 | 333.0 | L26:attention_output +17.6 | L28:mlp_output +2845.4 | L28:attention_output +24.1 | L25:mlp_output -15.5 |
| front_back:short:list_answer:container | 16.5 | 11.5 | L28:mlp_output +3.4 | L27:mlp_output +6.8 | L27:mlp_output +7.2 | L28:attention_output +4.4 |
| front_back:short:list_answer:number | 8.8 | 8.1 | L28:mlp_output +2.9 | L28:attention_output +7.5 | L26:mlp_output +1.6 | L27:mlp_output +1.8 |
| front_back:short:list_answer:plant | 2.4 | 9.1 | L28:mlp_output +1.0 | L28:mlp_output +7.8 | L27:mlp_output +1.0 | L25:mlp_output +3.5 |
| front_back:short:list_answer:time | 4.0 | 12.0 | L28:mlp_output +3.9 | L28:attention_output +8.8 | L27:mlp_output +3.0 | L27:mlp_output +5.0 |
| front_back:short:multiple_choice:container | 1.0 | 15.0 | L28:mlp_output +0.1 | L28:mlp_output +3.9 | L28:mlp_output +0.1 | L28:attention_output +4.0 |
| front_back:short:multiple_choice:number | 3.8 | 13.8 | L28:mlp_output +2.4 | L28:mlp_output +2.6 | L28:mlp_output +2.4 | L28:attention_output +6.2 |
| front_back:short:multiple_choice:plant | 1.1 | 5.2 | L27:mlp_output +0.2 | L28:mlp_output +1.4 | L26:attention_output +0.1 | L28:mlp_output +1.9 |
| front_back:short:multiple_choice:time | 1.1 | 16.4 | L28:mlp_output +1.2 | L28:mlp_output +6.5 | L28:mlp_output +1.1 | L28:attention_output +16.8 |
| front_back:short:quoted_answer:container | 25.6 | 130.1 | L26:mlp_output +24.5 | L28:mlp_output +995.0 | L28:mlp_output +550.9 | L28:mlp_output +148.0 |
| front_back:short:quoted_answer:number | 6.5 | 493.6 | L26:mlp_output +1.8 | L28:mlp_output +577.2 | L28:mlp_output +132.8 | L25:mlp_output +68.4 |
| front_back:short:quoted_answer:plant | 16.0 | 2904.2 | L26:mlp_output +2.6 | L28:mlp_output +10343.4 | L28:mlp_output +60.9 | L26:mlp_output +3039.9 |
| front_back:short:quoted_answer:time | 27.6 | 4648.6 | L28:mlp_output +7.5 | L28:mlp_output +21552.8 | L28:mlp_output +1394.4 | L26:mlp_output +4004.2 |

