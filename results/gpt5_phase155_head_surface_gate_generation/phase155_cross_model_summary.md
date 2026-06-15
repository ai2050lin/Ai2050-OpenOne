# Phase 155 Cross-model Head Surface Gate Generation Summary

## qwen3

cases=120, layer=L36, heads=32

### By category

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.36 | 0.35 | 0.37 | 0.36 | 0.36 | -0.01 | +0.01 | +0.00 | H0 | H0 | H0 |
| number | 30 | 0.25 | 0.26 | 0.26 | 0.26 | 0.26 | +0.01 | +0.01 | +0.00 | H0 | H0 | H0 |
| plant | 30 | 0.52 | 0.50 | 0.51 | 0.51 | 0.52 | -0.03 | -0.01 | -0.01 | H0 | H0 | H0 |
| time | 30 | 0.33 | 0.33 | 0.34 | 0.33 | 0.33 | -0.01 | +0.00 | +0.00 | H0 | H0 | H0 |

### By format

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.14 | 0.13 | 0.15 | 0.15 | 0.14 | -0.01 | +0.02 | +0.01 | H0 | H0 | H24 |
| label_colon | 24 | 0.29 | 0.27 | 0.29 | 0.28 | 0.29 | -0.03 | +0.00 | -0.01 | H0 | H0 | H0 |
| list_answer | 24 | 0.27 | 0.25 | 0.26 | 0.26 | 0.27 | -0.02 | -0.01 | -0.01 | H26 | H0 | H0 |
| multiple_choice | 24 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | +0.01 | +0.01 | +0.01 | H0 | H25 | H25 |
| quoted_answer | 24 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | +0.00 | +0.00 | +0.00 | H8 | H2 | H5 |

### By family

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | -0.00 | +0.00 | +0.00 | H0 | H0 | H0 |
| neutral | 40 | 0.30 | 0.29 | 0.30 | 0.29 | 0.29 | -0.01 | +0.00 | -0.00 | H0 | H0 | H0 |
| short | 40 | 0.56 | 0.55 | 0.57 | 0.57 | 0.57 | -0.01 | +0.01 | +0.00 | H0 | H0 | H0 |

### By split

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.40 | 0.39 | 0.41 | 0.41 | 0.40 | -0.01 | +0.01 | +0.00 | H0 | H0 | H0 |
| front_back | 60 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | -0.00 | +0.00 | -0.00 | H0 | H0 | H0 |

### Cases

| case | clean | ans | fmt | joint | random | top_answer | top_format | top_joint |
|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+0.5 | H8 dF+0.4 | H8 dA+0.5/dF+0.4 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+1.5 | H0 dF+0.9 | H24 dA+1.5/dF+0.9 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H22 dA+0.5 | H8 dF+0.6 | H22 dA+0.5/dF+0.0 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+0.5 | H0 dF+0.5 | H0 dA+0.5/dF+0.5 |
| back_front:long:label_colon:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H3 dA+0.5 | H24 dF+1.6 | H24 dA+0.4/dF+1.6 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+1.9 | H0 dF+1.0 | H24 dA+1.9/dF+0.5 |
| back_front:long:label_colon:plant | 0.38 | 0.25 | 0.38 | 0.38 | 0.38 | H22 dA+0.9 | H0 dF+1.1 | H0 dA+0.0/dF+1.1 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+1.8 | H24 dF+0.6 | H24 dA+1.8/dF+0.6 |
| back_front:long:list_answer:container | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H3 dA+0.4 | H0 dF+4.6 | H0 dA+0.1/dF+4.6 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H6 dA+0.1 | H0 dF+2.4 | H0 dA-0.8/dF+2.4 |
| back_front:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H22 dA+0.9 | H0 dF+4.2 | H0 dA+0.1/dF+4.2 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+0.6 | H0 dF+3.0 | H0 dA+0.2/dF+3.0 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.1 | H25 dF+1.0 | H25 dA+0.0/dF+1.0 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.5 | H0 dF+0.9 | H0 dA+0.5/dF+0.9 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.8 | H0 dA+0.0/dF+0.8 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H11 dF+0.9 | H11 dA+0.0/dF+0.9 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+3.8 | H5 dF+6.9 | H5 dA-0.5/dF+6.9 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+2.9 | H5 dF+8.6 | H5 dA-0.5/dF+8.6 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+4.9 | H5 dF+7.0 | H5 dA-0.4/dF+7.0 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+2.2 | H5 dF+6.9 | H5 dA-1.0/dF+6.9 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+0.8 | H0 dF+0.0 | H0 dA+0.8/dF+0.0 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+0.2 | H0 dF+0.0 | H0 dA+0.2/dF+0.0 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H24 dA+0.9 | H25 dF+0.1 | H24 dA+0.9/dF+0.0 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+0.1 | H0 dF+0.0 | H11 dA+0.1/dF+0.0 |
| back_front:neutral:label_colon:container | 0.25 | 0.25 | 0.38 | 0.25 | 0.25 | H23 dA+0.6 | H0 dF+0.0 | H23 dA+0.6/dF+0.0 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+5.6 | H0 dF+0.0 | H0 dA+5.6/dF+0.0 |
| back_front:neutral:label_colon:plant | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H1 dA+0.1 | H0 dF+0.5 | H0 dA+0.0/dF+0.5 |
| back_front:neutral:label_colon:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H0 dA+1.0 | H0 dF+0.0 | H0 dA+1.0/dF+0.0 |
| back_front:neutral:list_answer:container | 0.50 | 0.50 | 0.62 | 0.50 | 0.50 | H2 dA+0.2 | H0 dF+0.5 | H3 dA+0.0/dF+0.2 |
| back_front:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+0.2 | H0 dF+0.0 | H26 dA+0.2/dF+0.0 |
| back_front:neutral:list_answer:plant | 0.25 | 0.12 | 0.12 | 0.25 | 0.25 | H26 dA+0.4 | H0 dF+0.9 | H3 dA+0.2/dF+0.1 |
| back_front:neutral:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.25 | H26 dA+0.4 | H0 dF+0.6 | H27 dA+0.1/dF+0.1 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+1.9 | H25 dA+0.0/dF+1.9 |
| back_front:neutral:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+5.8 | H25 dA+0.0/dF+5.8 |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H1 dA+0.0 | H25 dF+2.0 | H25 dA+0.0/dF+2.0 |
| back_front:neutral:multiple_choice:time | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H26 dA+0.2 | H25 dF+3.9 | H25 dA-0.2/dF+3.9 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+1.1 | H5 dF+13.1 | H5 dA-1.4/dF+13.1 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+2.2 | H5 dF+2.1 | H24 dA+2.2/dF+0.2 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H27 dA+1.6 | H2 dF+3.8 | H5 dA+0.2/dF+3.1 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H24 dA+4.6 | H2 dF+9.8 | H2 dA-2.5/dF+9.8 |
| back_front:short:answer_one_word:container | 0.50 | 0.38 | 0.62 | 0.62 | 0.50 | H5 dA+0.2 | H4 dF+0.1 | H24 dA+0.1/dF+0.1 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H2 dA+0.2 | H0 dF+0.0 | H2 dA+0.2/dF+0.0 |
| back_front:short:answer_one_word:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H26 dA+0.2 | H1 dF+0.0 | H1 dA+0.0/dF+0.0 |
| back_front:short:answer_one_word:time | 0.25 | 0.25 | 0.38 | 0.38 | 0.38 | H26 dA+0.6 | H24 dF+0.4 | H24 dA+0.0/dF+0.4 |
| back_front:short:label_colon:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | H5 dA+0.5 | H15 dF+0.5 | H24 dA+0.2/dF+0.5 |
| back_front:short:label_colon:number | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H0 dA+0.1 | H0 dF+0.5 | H0 dA+0.1/dF+0.5 |
| back_front:short:label_colon:plant | 0.75 | 0.62 | 0.62 | 0.62 | 0.75 | H27 dA+0.1 | H0 dF+3.1 | H0 dA-0.1/dF+3.1 |
| back_front:short:label_colon:time | 0.62 | 0.62 | 0.75 | 0.62 | 0.62 | H5 dA+0.2 | H0 dF+0.2 | H8 dA+0.1/dF+0.0 |
| back_front:short:list_answer:container | 0.75 | 0.62 | 0.75 | 0.75 | 0.75 | H26 dA+0.2 | H0 dF+1.2 | H0 dA+0.0/dF+1.2 |
| back_front:short:list_answer:number | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H2 dA+0.1 | H27 dF+0.4 | H27 dA+0.1/dF+0.4 |
| back_front:short:list_answer:plant | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H0 dA+0.0 | H0 dF+0.5 | H0 dA+0.0/dF+0.5 |
| back_front:short:list_answer:time | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H26 dA+0.2 | H0 dF+0.8 | H0 dA+0.0/dF+0.8 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+5.0 | H25 dA+0.0/dF+5.0 |
| back_front:short:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+5.1 | H25 dA+0.0/dF+5.1 |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+1.6 | H25 dA+0.0/dF+1.6 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | H25 dA+0.2 | H25 dF+8.1 | H25 dA+0.2/dF+8.1 |
| back_front:short:quoted_answer:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | H25 dA+0.2 | H2 dF+7.1 | H2 dA+0.0/dF+7.1 |
| back_front:short:quoted_answer:number | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H0 dA+0.0 | H2 dF+7.4 | H2 dA+0.0/dF+7.4 |
| back_front:short:quoted_answer:plant | 0.88 | 0.62 | 0.88 | 0.88 | 0.88 | H0 dA+0.5 | H2 dF+1.4 | H2 dA+0.0/dF+1.4 |
| back_front:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H27 dA+0.4 | H2 dF+2.8 | H5 dA+0.1/dF+2.8 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+1.0 | H0 dF+0.2 | H8 dA+1.0/dF+0.1 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H27 dA+0.5 | H0 dF+1.0 | H0 dA+0.4/dF+1.0 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H23 dA+0.1 | H8 dF+0.9 | H8 dA+0.0/dF+0.9 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+0.4 | H3 dF+0.5 | H24 dA+0.4/dF+0.4 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+2.1 | H24 dF+1.1 | H0 dA+2.1/dF+0.8 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H25 dA+2.2 | H24 dF+1.2 | H24 dA+0.6/dF+1.2 |
| front_back:long:label_colon:plant | 0.25 | 0.12 | 0.12 | 0.12 | 0.25 | H22 dA+0.1 | H0 dF+1.5 | H0 dA-0.2/dF+1.5 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+2.5 | H0 dF+0.9 | H24 dA+2.5/dF+0.9 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H3 dA+0.4 | H0 dF+3.4 | H0 dA-0.5/dF+3.4 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H3 dA+0.2 | H0 dF+2.2 | H0 dA+0.0/dF+2.2 |
| front_back:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H23 dA+0.2 | H0 dF+4.2 | H0 dA-0.4/dF+4.2 |
| front_back:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+0.9 | H0 dF+4.4 | H0 dA-0.4/dF+4.4 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.6 | H0 dA+0.0/dF+0.6 |
| front_back:long:multiple_choice:number | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.1 | H0 dF+0.9 | H0 dA+0.1/dF+0.9 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+1.0 | H0 dA+0.0/dF+1.0 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+1.4 | H0 dA+0.0/dF+1.4 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+4.4 | H5 dF+9.1 | H5 dA+0.1/dF+9.1 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+2.8 | H5 dF+7.4 | H5 dA-0.1/dF+7.4 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+4.8 | H5 dF+5.4 | H5 dA+0.0/dF+5.4 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+2.2 | H5 dF+6.1 | H5 dA-0.5/dF+6.1 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+1.2 | H0 dF+0.0 | H24 dA+1.2/dF+0.0 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H5 dA+0.4 | H2 dF+0.0 | H14 dA+0.4/dF+0.0 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+0.5 | H0 dF+0.0 | H26 dA+0.5/dF+0.0 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+1.6 | H0 dF+0.0 | H0 dA+1.6/dF+0.0 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+2.2 | H20 dF+0.2 | H0 dA+2.2/dF+0.1 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+1.9 | H0 dF+0.1 | H0 dA+1.9/dF+0.1 |
| front_back:neutral:label_colon:plant | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | H1 dA+0.4 | H0 dF+1.5 | H0 dA+0.1/dF+1.5 |
| front_back:neutral:label_colon:time | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | H25 dA+1.6 | H0 dF+0.2 | H0 dA+1.4/dF+0.2 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H2 dA+0.6 | H0 dF+1.1 | H20 dA+0.5/dF+0.1 |
| front_back:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H2 dA+0.4 | H0 dF+0.4 | H3 dA+0.0/dF+0.4 |
| front_back:neutral:list_answer:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H2 dA+0.2 | H0 dF+1.1 | H24 dA+0.2/dF+0.5 |
| front_back:neutral:list_answer:time | 0.25 | 0.12 | 0.12 | 0.12 | 0.25 | H15 dA+0.5 | H0 dF+0.9 | H15 dA+0.5/dF+0.4 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+3.5 | H25 dA+0.0/dF+3.5 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.2 | H25 dF+4.4 | H25 dA-0.1/dF+4.4 |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+2.2 | H25 dA+0.0/dF+2.2 |
| front_back:neutral:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+3.6 | H25 dA+0.0/dF+3.6 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H27 dA+2.1 | H5 dF+3.8 | H0 dA+1.5/dF+0.8 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+3.6 | H2 dF+10.1 | H5 dA+0.2/dF+7.4 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.50 | 0.38 | 0.38 | 0.38 | H27 dA+2.2 | H2 dF+4.4 | H5 dA-0.5/dF+4.4 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+4.2 | H2 dF+11.1 | H5 dA+2.6/dF+8.0 |
| front_back:short:answer_one_word:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H2 dA+0.1 | H3 dF+0.4 | H2 dA+0.1/dF+0.1 |
| front_back:short:answer_one_word:number | 0.25 | 0.25 | 0.38 | 0.25 | 0.25 | H1 dA+0.0 | H26 dF+0.1 | H1 dA+0.0/dF+0.0 |
| front_back:short:answer_one_word:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H26 dF+0.4 | H26 dA+0.0/dF+0.4 |
| front_back:short:answer_one_word:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H26 dA+1.2 | H0 dF+0.0 | H26 dA+1.2/dF+0.0 |
| front_back:short:label_colon:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H17 dA+1.5 | H0 dF+1.6 | H0 dA-0.2/dF+1.6 |
| front_back:short:label_colon:number | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H0 dA+1.4 | H0 dF+2.9 | H0 dA+1.4/dF+2.9 |
| front_back:short:label_colon:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H22 dA+0.8 | H0 dF+4.0 | H0 dA+0.1/dF+4.0 |
| front_back:short:label_colon:time | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H5 dA+0.4 | H0 dF+0.9 | H0 dA+0.2/dF+0.9 |
| front_back:short:list_answer:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H0 dA+0.0 | H0 dF+2.6 | H0 dA+0.0/dF+2.6 |
| front_back:short:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | H26 dA+0.1 | H24 dF+0.2 | H24 dA+0.0/dF+0.2 |
| front_back:short:list_answer:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.0 | H0 dF+1.1 | H0 dA+0.0/dF+1.1 |
| front_back:short:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H1 dA+0.1 | H0 dF+0.2 | H27 dA+0.1/dF+0.2 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+5.4 | H25 dA+0.0/dF+5.4 |
| front_back:short:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.0 | H25 dF+4.5 | H25 dA+0.0/dF+4.5 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H25 dF+2.2 | H25 dA+0.0/dF+2.2 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H25 dA+0.9 | H25 dF+7.9 | H25 dA+0.9/dF+7.9 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H18 dA+0.6 | H2 dF+5.1 | H2 dA-0.5/dF+5.1 |
| front_back:short:quoted_answer:number | 0.25 | 0.38 | 0.25 | 0.25 | 0.25 | H26 dA+0.5 | H2 dF+4.9 | H2 dA-0.4/dF+4.9 |
| front_back:short:quoted_answer:plant | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H22 dA+1.1 | H2 dF+3.9 | H2 dA-0.1/dF+3.9 |
| front_back:short:quoted_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H24 dA+1.8 | H2 dF+3.2 | H2 dA-0.9/dF+3.2 |

## glm4

cases=120, layer=L39, heads=32

### By category

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.30 | 0.29 | 0.30 | 0.30 | 0.30 | -0.00 | +0.00 | +0.00 | H0 | H0 | H7 |
| number | 30 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | +0.00 | +0.00 | +0.00 | H28 | H0 | H0 |
| plant | 30 | 0.38 | 0.36 | 0.37 | 0.37 | 0.38 | -0.02 | -0.01 | -0.01 | H0 | H0 | H0 |
| time | 30 | 0.30 | 0.29 | 0.30 | 0.30 | 0.30 | -0.01 | -0.00 | -0.01 | H0 | H0 | H19 |

### By format

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.07 | 0.05 | 0.06 | 0.06 | 0.07 | -0.02 | -0.02 | -0.02 | H0 | H9 | H9 |
| label_colon | 24 | 0.18 | 0.17 | 0.18 | 0.18 | 0.17 | -0.01 | +0.01 | +0.01 | H19 | H7 | H7 |
| list_answer | 24 | 0.16 | 0.14 | 0.15 | 0.15 | 0.16 | -0.02 | -0.01 | -0.01 | H26 | H18 | H18 |
| multiple_choice | 24 | 0.98 | 0.98 | 0.98 | 0.98 | 0.99 | +0.01 | +0.00 | +0.00 | H0 | H0 | H0 |
| quoted_answer | 24 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | +0.01 | +0.01 | +0.01 | H28 | H0 | H0 |

### By family

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | -0.00 | +0.00 | +0.00 | H0 | H0 | H0 |
| neutral | 40 | 0.29 | 0.29 | 0.28 | 0.29 | 0.29 | +0.00 | -0.00 | +0.00 | H19 | H18 | H18 |
| short | 40 | 0.39 | 0.37 | 0.39 | 0.38 | 0.39 | -0.02 | -0.01 | -0.01 | H0 | H0 | H0 |

### By split

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.32 | 0.32 | 0.32 | 0.32 | 0.32 | -0.00 | +0.00 | +0.00 | H0 | H0 | H0 |
| front_back | 60 | 0.28 | 0.26 | 0.27 | 0.27 | 0.28 | -0.01 | -0.01 | -0.01 | H0 | H0 | H0 |

### Cases

| case | clean | ans | fmt | joint | random | top_answer | top_format | top_joint |
|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+2.4 | H9 dF+27.8 | H9 dA+0.6/dF+27.8 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+2.4 | H13 dF+38.4 | H13 dA-0.2/dF+38.4 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+1.5 | H9 dF+69.9 | H9 dA-0.5/dF+69.9 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+0.6 | H9 dF+29.4 | H9 dA-1.4/dF+29.4 |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+2.8 | H12 dF+10.2 | H7 dA+2.1/dF+10.1 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+1.1 | H9 dF+5.2 | H9 dA-0.1/dF+5.2 |
| back_front:long:label_colon:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H13 dA+1.4 | H7 dF+10.6 | H7 dA+0.0/dF+10.6 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H14 dA+0.8 | H12 dF+13.5 | H12 dA-0.6/dF+13.5 |
| back_front:long:list_answer:container | 0.25 | 0.12 | 0.25 | 0.25 | 0.12 | H26 dA+0.5 | H4 dF+2.8 | H7 dA-0.4/dF+2.8 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+0.8 | H4 dF+1.5 | H18 dA+0.5/dF+1.1 |
| back_front:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H24 dA+1.4 | H7 dF+3.1 | H28 dA+0.8/dF+2.2 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+1.5 | H4 dF+3.9 | H7 dA-0.4/dF+3.5 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H19 dF+0.8 | H19 dA+0.0/dF+0.8 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H18 dF+1.1 | H18 dA+0.0/dF+1.1 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H31 dF+0.4 | H31 dA+0.0/dF+0.4 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H19 dF+0.8 | H19 dA+0.0/dF+0.8 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+1.4 | H0 dF+352.1 | H0 dA+0.1/dF+352.1 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+5.8 | H0 dF+265.1 | H0 dA-7.0/dF+265.1 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H14 dA+5.5 | H0 dF+655.0 | H0 dA+2.5/dF+655.0 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H31 dA+13.4 | H0 dF+376.4 | H0 dA+2.1/dF+376.4 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+4.4 | H13 dF+3.6 | H19 dA+4.4/dF+1.8 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+3.6 | H9 dF+1.6 | H19 dA+3.6/dF+0.4 |
| back_front:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+2.4 | H9 dF+32.6 | H9 dA+0.1/dF+32.6 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+3.2 | H13 dF+2.4 | H9 dA+2.8/dF+1.9 |
| back_front:neutral:label_colon:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H19 dA+3.8 | H7 dF+6.2 | H7 dA+2.5/dF+6.2 |
| back_front:neutral:label_colon:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | H7 dA+2.8 | H7 dF+1.0 | H7 dA+2.8/dF+1.0 |
| back_front:neutral:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H28 dA+0.5 | H7 dF+4.5 | H7 dA+0.2/dF+4.5 |
| back_front:neutral:label_colon:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H7 dA+6.8 | H7 dF+4.1 | H7 dA+6.8/dF+4.1 |
| back_front:neutral:list_answer:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H26 dA+1.1 | H18 dF+3.2 | H18 dA+1.0/dF+3.2 |
| back_front:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+1.9 | H18 dF+2.6 | H18 dA+0.6/dF+2.6 |
| back_front:neutral:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H9 dA+2.2 | H18 dF+1.0 | H9 dA+2.2/dF+0.5 |
| back_front:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H26 dA+2.9 | H18 dF+1.6 | H18 dA+1.5/dF+1.6 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+1.9 | H0 dA+0.0/dF+1.9 |
| back_front:neutral:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H19 dA+0.1 | H0 dF+2.6 | H0 dA-0.2/dF+2.6 |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.8 | H0 dA+0.0/dF+0.8 |
| back_front:neutral:multiple_choice:time | 0.88 | 1.00 | 0.88 | 0.88 | 1.00 | H25 dA+0.5 | H19 dF+2.1 | H19 dA+0.1/dF+2.1 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+0.8 | H18 dF+309.0 | H18 dA-0.1/dF+309.0 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+10.2 | H18 dF+245.2 | H18 dA+2.8/dF+245.2 |
| back_front:neutral:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H2 dA+1.0 | H18 dF+311.1 | H18 dA+0.2/dF+311.1 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H28 dA+9.1 | H18 dF+170.0 | H18 dA+7.0/dF+170.0 |
| back_front:short:answer_one_word:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H21 dA+0.2 | H19 dF+6.0 | H19 dA+0.0/dF+6.0 |
| back_front:short:answer_one_word:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H31 dA+0.5 | H19 dF+1.8 | H19 dA+0.4/dF+1.8 |
| back_front:short:answer_one_word:plant | 0.38 | 0.12 | 0.25 | 0.25 | 0.38 | H13 dA+0.5 | H9 dF+120.9 | H9 dA+0.0/dF+120.9 |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+0.4 | H19 dF+9.8 | H19 dA+0.0/dF+9.8 |
| back_front:short:label_colon:container | 0.50 | 0.50 | 0.62 | 0.62 | 0.62 | H11 dA+0.4 | H14 dF+9.4 | H14 dA+0.0/dF+9.4 |
| back_front:short:label_colon:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H11 dA+1.1 | H14 dF+3.6 | H14 dA+0.5/dF+3.6 |
| back_front:short:label_colon:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.25 | H8 dA+0.2 | H24 dF+5.9 | H24 dA+0.0/dF+5.9 |
| back_front:short:label_colon:time | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | H0 dA+0.4 | H14 dF+10.6 | H14 dA-0.6/dF+10.6 |
| back_front:short:list_answer:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H7 dA+0.4 | H7 dF+5.2 | H7 dA+0.4/dF+5.2 |
| back_front:short:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+0.1 | H7 dF+3.9 | H7 dA+0.0/dF+3.9 |
| back_front:short:list_answer:plant | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H19 dA+0.1 | H18 dF+10.6 | H18 dA+0.0/dF+10.6 |
| back_front:short:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H9 dA+0.4 | H19 dF+8.4 | H19 dA+0.2/dF+8.4 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H31 dF+0.4 | H31 dA+0.0/dF+0.4 |
| back_front:short:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H1 dF+0.4 | H1 dA+0.0/dF+0.4 |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.1 | H0 dA+0.0/dF+0.1 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.4 | H0 dA+0.0/dF+0.4 |
| back_front:short:quoted_answer:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H9 dA+0.4 | H0 dF+112.1 | H0 dA+0.0/dF+112.1 |
| back_front:short:quoted_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H24 dA+1.0 | H0 dF+30.2 | H0 dA+0.4/dF+30.2 |
| back_front:short:quoted_answer:plant | 0.25 | 0.38 | 0.38 | 0.38 | 0.25 | H17 dA+0.2 | H0 dF+46.4 | H0 dA+0.0/dF+46.4 |
| back_front:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H5 dA+0.4 | H0 dF+47.6 | H0 dA+0.0/dF+47.6 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+1.6 | H9 dF+30.0 | H9 dA+0.0/dF+30.0 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+2.1 | H9 dF+28.6 | H9 dA-0.6/dF+28.6 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+0.6 | H13 dF+79.8 | H13 dA-0.6/dF+79.8 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H16 dA+1.8 | H9 dF+25.2 | H9 dA-1.2/dF+25.2 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+3.4 | H7 dF+10.6 | H7 dA+0.8/dF+10.6 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+0.9 | H7 dF+11.4 | H7 dA+0.1/dF+11.4 |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+1.2 | H7 dF+8.1 | H7 dA+0.0/dF+8.1 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+2.0 | H12 dF+13.6 | H12 dA-0.1/dF+13.6 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+1.6 | H4 dF+3.2 | H28 dA+1.6/dF+1.0 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+0.8 | H4 dF+2.0 | H24 dA+0.8/dF+0.9 |
| front_back:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H20 dA+0.5 | H4 dF+4.8 | H4 dA-0.8/dF+4.8 |
| front_back:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+1.5 | H4 dF+1.6 | H18 dA+1.1/dF+0.8 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H18 dF+0.4 | H18 dA+0.0/dF+0.4 |
| front_back:long:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 | H11 dA+0.1 | H1 dF+0.8 | H1 dA+0.0/dF+0.8 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H28 dF+0.4 | H28 dA+0.0/dF+0.4 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H1 dF+1.6 | H1 dA+0.0/dF+1.6 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+2.9 | H0 dF+413.1 | H0 dA+0.5/dF+413.1 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+6.4 | H0 dF+375.4 | H0 dA-6.6/dF+375.4 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+2.8 | H0 dF+512.6 | H0 dA+0.1/dF+512.6 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H31 dA+19.0 | H0 dF+343.5 | H0 dA+1.9/dF+343.5 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H31 dA+3.6 | H9 dF+8.6 | H9 dA+1.8/dF+8.6 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H24 dA+3.5 | H13 dF+9.0 | H13 dA+0.9/dF+9.0 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H15 dA+0.6 | H9 dF+40.8 | H9 dA-0.5/dF+40.8 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+2.9 | H9 dF+13.4 | H9 dA+0.1/dF+13.4 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H31 dA+4.9 | H7 dF+9.2 | H7 dA+3.9/dF+9.2 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H7 dA+4.4 | H7 dF+2.9 | H7 dA+4.4/dF+2.9 |
| front_back:neutral:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H5 dA+0.2 | H7 dF+6.6 | H7 dA+0.1/dF+6.6 |
| front_back:neutral:label_colon:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H29 dA+0.5 | H7 dF+8.8 | H7 dA-0.4/dF+8.8 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+4.9 | H18 dF+4.4 | H18 dA+3.5/dF+4.4 |
| front_back:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+1.1 | H28 dF+1.2 | H28 dA+0.9/dF+1.2 |
| front_back:neutral:list_answer:plant | 0.38 | 0.25 | 0.25 | 0.38 | 0.38 | H19 dA+2.4 | H15 dF+3.2 | H28 dA+1.6/dF+2.1 |
| front_back:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H19 dA+2.8 | H0 dF+2.2 | H19 dA+2.8/dF+0.9 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H25 dA+0.1 | H0 dF+1.6 | H0 dA+0.0/dF+1.6 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.1 | H0 dF+1.8 | H0 dA+0.1/dF+1.8 |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+1.4 | H0 dA+0.0/dF+1.4 |
| front_back:neutral:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.1 | H0 dF+2.8 | H0 dA+0.1/dF+2.8 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+0.6 | H18 dF+392.6 | H18 dA-0.1/dF+392.6 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+8.4 | H18 dF+193.6 | H18 dA+3.5/dF+193.6 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.50 | H11 dA+1.0 | H0 dF+212.8 | H0 dA-0.4/dF+212.8 |
| front_back:neutral:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H28 dA+10.2 | H18 dF+205.9 | H18 dA+7.1/dF+205.9 |
| front_back:short:answer_one_word:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H31 dA+0.5 | H19 dF+6.0 | H19 dA+0.2/dF+6.0 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H16 dA+0.2 | H19 dF+0.8 | H19 dA+0.1/dF+0.8 |
| front_back:short:answer_one_word:plant | 0.38 | 0.25 | 0.25 | 0.25 | 0.38 | H11 dA+0.2 | H9 dF+153.0 | H9 dA-0.1/dF+153.0 |
| front_back:short:answer_one_word:time | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | H31 dA+0.9 | H19 dF+13.6 | H19 dA+0.2/dF+13.6 |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H29 dA+0.4 | H14 dF+7.1 | H14 dA-0.1/dF+7.1 |
| front_back:short:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H28 dA+1.2 | H14 dF+4.4 | H26 dA-0.4/dF+4.4 |
| front_back:short:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H13 dA+0.5 | H14 dF+11.0 | H14 dA+0.5/dF+11.0 |
| front_back:short:label_colon:time | 0.38 | 0.12 | 0.38 | 0.38 | 0.38 | H0 dA+0.8 | H14 dF+10.9 | H14 dA-1.1/dF+10.9 |
| front_back:short:list_answer:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H0 dA+0.5 | H7 dF+9.0 | H7 dA+0.4/dF+9.0 |
| front_back:short:list_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H0 dA+0.2 | H28 dF+2.8 | H28 dA+0.1/dF+2.8 |
| front_back:short:list_answer:plant | 0.75 | 0.62 | 0.62 | 0.62 | 0.75 | H2 dA+0.2 | H18 dF+9.8 | H18 dA+0.2/dF+9.8 |
| front_back:short:list_answer:time | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | H19 dA+0.6 | H18 dF+7.8 | H19 dA+0.6/dF+7.5 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.1 | H0 dA+0.0/dF+0.1 |
| front_back:short:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.0 | H1 dF+0.4 | H1 dA+0.0/dF+0.4 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.0 | H0 dA+0.0/dF+0.0 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H0 dF+0.2 | H0 dA+0.0/dF+0.2 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+0.6 | H0 dF+54.2 | H0 dA+0.2/dF+54.2 |
| front_back:short:quoted_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H13 dA+0.6 | H0 dF+15.5 | H0 dA+0.2/dF+15.5 |
| front_back:short:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H24 dA+0.9 | H0 dF+55.0 | H0 dA+0.5/dF+55.0 |
| front_back:short:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H20 dA+1.5 | H15 dF+16.1 | H15 dA+0.4/dF+16.1 |

## deepseek7b

cases=120, layer=L28, heads=28

### By category

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.26 | 0.26 | 0.28 | 0.28 | 0.27 | +0.00 | +0.01 | +0.01 | H11 | H27 | H27 |
| number | 30 | 0.21 | 0.21 | 0.21 | 0.21 | 0.20 | +0.00 | +0.00 | +0.00 | H12 | H27 | H12 |
| plant | 30 | 0.33 | 0.31 | 0.33 | 0.33 | 0.33 | -0.01 | +0.00 | +0.00 | H13 | H12 | H13 |
| time | 30 | 0.22 | 0.22 | 0.23 | 0.21 | 0.22 | +0.00 | +0.01 | -0.01 | H13 | H27 | H27 |

### By format

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.06 | 0.07 | 0.08 | 0.07 | 0.06 | +0.01 | +0.03 | +0.01 | H11 | H12 | H11 |
| label_colon | 24 | 0.06 | 0.05 | 0.06 | 0.06 | 0.05 | -0.01 | +0.00 | +0.00 | H13 | H12 | H12 |
| list_answer | 24 | 0.19 | 0.17 | 0.18 | 0.18 | 0.18 | -0.02 | -0.01 | -0.01 | H13 | H9 | H9 |
| multiple_choice | 24 | 0.88 | 0.88 | 0.90 | 0.89 | 0.88 | +0.00 | +0.02 | +0.01 | H0 | H21 | H21 |
| quoted_answer | 24 | 0.09 | 0.10 | 0.09 | 0.09 | 0.10 | +0.01 | +0.00 | +0.00 | H12 | H27 | H27 |

### By family

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.28 | 0.28 | 0.29 | 0.28 | 0.28 | +0.00 | +0.01 | -0.00 | H13 | H22 | H13 |
| neutral | 40 | 0.20 | 0.20 | 0.21 | 0.20 | 0.20 | -0.00 | +0.01 | +0.00 | H11 | H12 | H11 |
| short | 40 | 0.28 | 0.27 | 0.29 | 0.29 | 0.28 | -0.01 | +0.01 | +0.01 | H13 | H9 | H9 |

### By split

| group | n | clean_hit | top_answer_hit | top_format_hit | top_joint_hit | random_hit | top_answer_delta | top_format_delta | top_joint_delta | top_answer_head | top_format_head | top_joint_head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.27 | 0.27 | 0.29 | 0.28 | 0.27 | +0.00 | +0.02 | +0.01 | H10 | H27 | H12 |
| front_back | 60 | 0.24 | 0.23 | 0.24 | 0.23 | 0.24 | -0.01 | -0.00 | -0.01 | H13 | H27 | H27 |

### Cases

| case | clean | ans | fmt | joint | random | top_answer | top_format | top_joint |
|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H13 dA+3.1 | H11 dF+0.1 | H13 dA+3.1/dF+0.0 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+7.5 | H24 dF+0.1 | H12 dA+7.5/dF+0.0 |
| back_front:long:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H0 dA+0.6 | H10 dF+0.1 | H0 dA+0.6/dF+0.0 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.25 | 0.00 | 0.00 | H12 dA+7.4 | H22 dF+0.1 | H12 dA+7.4/dF-0.1 |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H1 dA+2.1 | H22 dF+0.8 | H1 dA+2.1/dF-0.1 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H7 dA+1.8 | H10 dF+1.0 | H7 dA+1.8/dF+0.8 |
| back_front:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+1.0 | H26 dF+1.2 | H25 dA+0.5/dF+1.1 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H8 dA+5.9 | H7 dF+0.6 | H8 dA+5.9/dF+0.4 |
| back_front:long:list_answer:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H13 dA+3.4 | H26 dF+1.1 | H13 dA+3.4/dF+0.6 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+5.6 | H21 dF+0.2 | H12 dA+5.6/dF+0.0 |
| back_front:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H13 dA+5.2 | H22 dF+1.5 | H13 dA+5.2/dF+1.4 |
| back_front:long:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H13 dA+5.5 | H25 dF+1.6 | H13 dA+5.5/dF+1.5 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H13 dF+1.9 | H13 dA+0.0/dF+1.9 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H8 dF+1.4 | H8 dA+0.0/dF+1.4 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H11 dA+0.1 | H21 dF+0.8 | H21 dA+0.0/dF+0.8 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H13 dF+0.6 | H13 dA+0.0/dF+0.6 |
| back_front:long:quoted_answer:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H10 dA+0.6 | H27 dF+3.8 | H27 dA-0.2/dF+3.8 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H19 dA+0.4 | H27 dF+9.5 | H27 dA-0.8/dF+9.5 |
| back_front:long:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H10 dA+0.1 | H27 dF+6.1 | H27 dA-0.2/dF+6.1 |
| back_front:long:quoted_answer:time | 0.12 | 0.12 | 0.00 | 0.00 | 0.12 | H7 dA+1.1 | H27 dF+14.0 | H27 dA-0.5/dF+14.0 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.25 | 0.25 | 0.25 | 0.00 | H11 dA+16.4 | H11 dF+1.5 | H11 dA+16.4/dF+1.5 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+3.6 | H5 dF+0.2 | H11 dA+3.6/dF-0.1 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H10 dA+3.2 | H12 dF+1.5 | H10 dA+3.2/dF+0.5 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+6.8 | H8 dF+0.5 | H11 dA+6.8/dF+0.1 |
| back_front:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+125.0 | H12 dF+4.2 | H12 dA+125.0/dF+4.2 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H17 dA+1.8 | H10 dF+5.2 | H12 dA-1.2/dF+4.4 |
| back_front:neutral:label_colon:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H5 dA+9.4 | H12 dF+4.0 | H5 dA+9.4/dF-0.1 |
| back_front:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+29.6 | H12 dF+3.4 | H12 dA+29.5/dF+3.4 |
| back_front:neutral:list_answer:container | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H9 dA+6.6 | H13 dF+1.4 | H9 dA+6.6/dF+1.1 |
| back_front:neutral:list_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H10 dA+1.2 | H13 dF+1.0 | H10 dA+1.2/dF+0.1 |
| back_front:neutral:list_answer:plant | 0.38 | 0.25 | 0.38 | 0.38 | 0.38 | H5 dA+3.2 | H9 dF+3.5 | H13 dA+0.6/dF+2.6 |
| back_front:neutral:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+4.9 | H9 dF+3.0 | H12 dA+4.9/dF+0.2 |
| back_front:neutral:multiple_choice:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | H22 dA+0.9 | H13 dF+2.6 | H22 dA+0.9/dF+1.9 |
| back_front:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H1 dA+0.2 | H12 dF+3.0 | H12 dA+0.2/dF+3.0 |
| back_front:neutral:multiple_choice:plant | 0.62 | 0.75 | 0.75 | 0.75 | 0.62 | H10 dA+0.8 | H12 dF+4.6 | H12 dA+0.8/dF+4.6 |
| back_front:neutral:multiple_choice:time | 0.62 | 0.62 | 0.88 | 0.62 | 0.62 | H19 dA+0.9 | H13 dF+2.2 | H5 dA+0.4/dF+1.4 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+5.5 | H27 dF+253.4 | H27 dA-3.8/dF+253.4 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+2.1 | H27 dF+285.8 | H27 dA+1.1/dF+285.8 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H5 dA+1.1 | H27 dF+714.0 | H27 dA+0.5/dF+714.0 |
| back_front:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H22 dA+1.2 | H27 dF+290.8 | H27 dA-0.9/dF+290.8 |
| back_front:short:answer_one_word:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H11 dA+2.1 | H9 dF+1.0 | H11 dA+2.1/dF+0.5 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+3.9 | H10 dF+0.6 | H10 dA+3.9/dF+0.6 |
| back_front:short:answer_one_word:plant | 0.00 | 0.12 | 0.12 | 0.12 | 0.12 | H10 dA+2.1 | H12 dF+0.4 | H10 dA+2.1/dF+0.1 |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+1.6 | H7 dF+0.4 | H11 dA+1.6/dF+0.2 |
| back_front:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+180.2 | H8 dF+94.5 | H12 dA+180.2/dF+80.5 |
| back_front:short:label_colon:number | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | H13 dA+1.4 | H12 dF+16.6 | H12 dA+0.8/dF+16.6 |
| back_front:short:label_colon:plant | 0.25 | 0.25 | 0.38 | 0.38 | 0.25 | H13 dA+1.1 | H8 dF+158.9 | H8 dA-0.4/dF+158.9 |
| back_front:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+3.4 | H8 dF+53.9 | H8 dA+0.8/dF+53.9 |
| back_front:short:list_answer:container | 0.25 | 0.00 | 0.12 | 0.12 | 0.12 | H27 dA+2.4 | H9 dF+3.1 | H9 dA+2.0/dF+3.1 |
| back_front:short:list_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H27 dA+6.0 | H9 dF+3.2 | H9 dA+2.6/dF+3.2 |
| back_front:short:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H7 dA+1.8 | H9 dF+3.2 | H9 dA+0.5/dF+3.2 |
| back_front:short:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+1.6 | H9 dF+2.8 | H9 dA+1.6/dF+2.8 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H12 dF+1.5 | H12 dA+0.0/dF+1.5 |
| back_front:short:multiple_choice:number | 0.88 | 0.88 | 1.00 | 1.00 | 1.00 | H17 dA+0.5 | H23 dF+1.5 | H23 dA+0.1/dF+1.5 |
| back_front:short:multiple_choice:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H0 dA+0.0 | H21 dF+0.5 | H21 dA+0.0/dF+0.5 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H24 dA+0.1 | H21 dF+1.4 | H21 dA+0.0/dF+1.4 |
| back_front:short:quoted_answer:container | 0.25 | 0.25 | 0.38 | 0.38 | 0.38 | H16 dA+1.9 | H27 dF+71.9 | H27 dA+0.0/dF+71.9 |
| back_front:short:quoted_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H9 dA+0.6 | H27 dF+102.4 | H27 dA+0.6/dF+102.4 |
| back_front:short:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+6.8 | H27 dF+306.5 | H27 dA-0.8/dF+306.5 |
| back_front:short:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+3.1 | H27 dF+759.0 | H27 dA+0.6/dF+759.0 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+7.0 | H11 dF+0.2 | H10 dA+7.0/dF+0.0 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+9.2 | H22 dF+0.5 | H12 dA+9.2/dF+0.1 |
| front_back:long:answer_one_word:plant | 0.38 | 0.25 | 0.38 | 0.25 | 0.25 | H22 dA+0.5 | H26 dF+0.5 | H22 dA+0.5/dF+0.2 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+7.8 | H22 dF+0.2 | H13 dA+7.8/dF+0.0 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H1 dA+8.6 | H11 dF+1.1 | H1 dA+8.6/dF-0.4 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H17 dA+0.2 | H10 dF+1.1 | H7 dA+0.0/dF+1.0 |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H23 dA+1.4 | H10 dF+1.4 | H23 dA+1.4/dF+0.2 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H21 dA+10.0 | H10 dF+1.8 | H21 dA+10.0/dF+0.9 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+7.0 | H22 dF+0.9 | H13 dA+7.0/dF+0.8 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+8.5 | H11 dF+0.5 | H12 dA+8.5/dF+0.4 |
| front_back:long:list_answer:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | H13 dA+4.8 | H22 dF+1.4 | H13 dA+4.8/dF+0.6 |
| front_back:long:list_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H13 dA+7.1 | H22 dF+0.9 | H13 dA+7.1/dF+0.5 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H22 dA+0.1 | H26 dF+0.4 | H20 dA+0.0/dF+0.2 |
| front_back:long:multiple_choice:number | 0.50 | 0.50 | 0.62 | 0.62 | 0.50 | H17 dA+0.4 | H12 dF+1.6 | H10 dA+0.0/dF+1.4 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H21 dF+0.9 | H21 dA+0.0/dF+0.9 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H15 dA+0.4 | H13 dF+0.8 | H13 dA+0.2/dF+0.8 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H0 dA+0.8 | H27 dF+2.6 | H27 dA-0.8/dF+2.6 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H2 dA+0.6 | H27 dF+5.6 | H27 dA-0.9/dF+5.6 |
| front_back:long:quoted_answer:plant | 0.25 | 0.38 | 0.25 | 0.25 | 0.25 | H22 dA+0.4 | H13 dF+2.9 | H13 dA-0.1/dF+2.9 |
| front_back:long:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H21 dA+1.0 | H27 dF+4.5 | H27 dA-1.2/dF+4.5 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+20.0 | H12 dF+1.1 | H11 dA+20.0/dF+1.0 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+3.8 | H12 dF+1.2 | H11 dA+3.8/dF+1.0 |
| front_back:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H11 dA+5.2 | H12 dF+2.4 | H11 dA+5.2/dF+1.4 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+7.8 | H9 dF+0.5 | H11 dA+7.8/dF+0.5 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+145.8 | H11 dF+5.1 | H11 dA+145.8/dF+5.1 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+5.8 | H12 dF+8.0 | H12 dA+5.8/dF+8.0 |
| front_back:neutral:label_colon:plant | 0.12 | 0.00 | 0.00 | 0.00 | 0.12 | H13 dA+16.0 | H12 dF+6.1 | H13 dA+16.0/dF+3.6 |
| front_back:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+26.6 | H11 dF+0.9 | H11 dA+26.6/dF+0.9 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H9 dA+20.1 | H9 dF+1.8 | H9 dA+20.1/dF+1.8 |
| front_back:neutral:list_answer:number | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | H12 dA+2.5 | H25 dF+0.9 | H13 dA+2.0/dF+0.4 |
| front_back:neutral:list_answer:plant | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | H12 dA+4.0 | H9 dF+2.5 | H9 dA+3.5/dF+2.5 |
| front_back:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H9 dA+1.8 | H9 dF+1.5 | H9 dA+1.8/dF+1.5 |
| front_back:neutral:multiple_choice:container | 0.62 | 0.62 | 0.75 | 0.75 | 0.88 | H10 dA+1.9 | H23 dF+2.4 | H23 dA+0.9/dF+2.4 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | H22 dA+0.2 | H21 dF+1.4 | H21 dA+0.1/dF+1.4 |
| front_back:neutral:multiple_choice:plant | 0.88 | 0.75 | 0.75 | 0.75 | 0.75 | H10 dA+1.5 | H21 dF+4.9 | H21 dA+0.9/dF+4.9 |
| front_back:neutral:multiple_choice:time | 0.75 | 0.75 | 0.62 | 0.62 | 0.62 | H20 dA+0.2 | H10 dF+3.6 | H10 dA+0.0/dF+3.6 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+6.1 | H27 dF+485.8 | H27 dA+0.4/dF+485.8 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H26 dA+1.6 | H27 dF+243.4 | H27 dA+0.9/dF+243.4 |
| front_back:neutral:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | H8 dA+1.0 | H27 dF+301.8 | H27 dA-0.4/dF+301.8 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H5 dA+1.8 | H27 dF+794.1 | H27 dA+0.0/dF+794.1 |
| front_back:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H11 dA+2.5 | H9 dF+1.1 | H11 dA+2.5/dF+0.6 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H7 dA+1.5 | H7 dF+0.5 | H7 dA+1.5/dF+0.5 |
| front_back:short:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H10 dA+1.5 | H25 dF+0.4 | H10 dA+1.5/dF+0.0 |
| front_back:short:answer_one_word:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H10 dA+0.8 | H9 dF+0.5 | H10 dA+0.8/dF+0.1 |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H1 dA+30.2 | H8 dF+105.2 | H9 dA+5.9/dF+90.5 |
| front_back:short:label_colon:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | H13 dA+3.9 | H12 dF+117.6 | H12 dA+0.8/dF+117.6 |
| front_back:short:label_colon:plant | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | H13 dA+1.0 | H8 dF+182.0 | H8 dA-0.5/dF+182.0 |
| front_back:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H13 dA+11.2 | H8 dF+77.0 | H8 dA+0.6/dF+77.0 |
| front_back:short:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H27 dA+3.0 | H9 dF+2.9 | H9 dA+2.6/dF+2.9 |
| front_back:short:list_answer:number | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | H27 dA+2.8 | H9 dF+1.4 | H9 dA+1.2/dF+1.4 |
| front_back:short:list_answer:plant | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | H27 dA+0.5 | H9 dF+2.0 | H9 dA+0.2/dF+2.0 |
| front_back:short:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H17 dA+1.6 | H12 dF+1.6 | H12 dA+0.1/dF+1.6 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H21 dF+1.0 | H21 dA+0.0/dF+1.0 |
| front_back:short:multiple_choice:number | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | H25 dA+0.1 | H21 dF+1.4 | H21 dA+0.0/dF+1.4 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H21 dF+0.6 | H21 dA+0.0/dF+0.6 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | H0 dA+0.0 | H23 dF+1.1 | H23 dA+0.0/dF+1.1 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H12 dA+12.0 | H27 dF+150.4 | H27 dA+0.6/dF+150.4 |
| front_back:short:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | H27 dA+1.1 | H27 dF+372.9 | H27 dA+1.1/dF+372.9 |
| front_back:short:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H23 dA+1.8 | H27 dF+1757.8 | H27 dA-0.8/dF+1757.8 |
| front_back:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | H12 dA+5.2 | H27 dF+2245.5 | H27 dA-0.8/dF+2245.5 |

