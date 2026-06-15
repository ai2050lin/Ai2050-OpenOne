# Phase 153 Cross-model Format-Syntax Subspace Joint Steering Summary

## qwen3

cases=120, formats=label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer, semantic_scale=0.05, format_scales=[0.05, 0.2]

### By category

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.438 | 0.36 | 0.27 | 0.36 | 0.24 | 0.33 | +0.06 | 5.3 | 2.8 | other | other |
| number | 30 | 0.518 | 0.25 | 0.22 | 0.28 | 0.20 | 0.36 | +0.14 | 4.6 | 2.9 | other | other |
| plant | 30 | 0.494 | 0.52 | 0.47 | 0.52 | 0.39 | 0.53 | +0.07 | 2.9 | 4.0 | other | other |
| time | 30 | 0.529 | 0.33 | 0.35 | 0.34 | 0.26 | 0.43 | +0.08 | 6.0 | 3.2 | other | other |

### By format

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.492 | 0.14 | 0.21 | 0.15 | 0.08 | 0.27 | +0.06 | 4.1 | 2.7 | other | other |
| label_colon | 24 | 0.551 | 0.29 | 0.24 | 0.31 | 0.12 | 0.33 | +0.09 | 5.3 | 3.9 | other | other |
| list_answer | 24 | 0.481 | 0.27 | 0.18 | 0.26 | 0.06 | 0.27 | +0.09 | 4.2 | 1.8 | other | other |
| multiple_choice | 24 | 0.411 | 0.98 | 0.87 | 0.98 | 0.95 | 0.96 | +0.09 | 1.2 | 3.5 | other | canonical |
| quoted_answer | 24 | 0.538 | 0.16 | 0.14 | 0.18 | 0.14 | 0.22 | +0.09 | 8.7 | 4.3 | other | other |

### By family

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.310 | 0.24 | 0.21 | 0.24 | 0.22 | 0.23 | +0.03 | 7.9 | 3.8 | other | other |
| neutral | 40 | 0.487 | 0.30 | 0.28 | 0.30 | 0.21 | 0.33 | +0.05 | 4.5 | 2.2 | other | other |
| short | 40 | 0.688 | 0.56 | 0.50 | 0.58 | 0.38 | 0.68 | +0.18 | 1.7 | 3.7 | other | canonical |

### By split

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.494 | 0.40 | 0.36 | 0.41 | 0.27 | 0.44 | +0.09 | 4.3 | 3.3 | other | other |
| front_back | 60 | 0.496 | 0.33 | 0.30 | 0.34 | 0.27 | 0.38 | +0.09 | 5.1 | 3.2 | other | other |

### Cases

| case | overlap | clean | sem | fmt_int | fmt_lm | best_joint | gain | joint | fmt_group | answer_rank | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.2 |  1.  concrete, abstract  1. |
| back_front:long:answer_one_word:number | 0.239 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.0 |  concrete, abstract  concrete, abstract  concrete, abstract |
| back_front:long:answer_one_word:plant | 0.534 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 4.8 |  concrete or abstract  concrete or abstract  concrete or abstract |
| back_front:long:answer_one_word:time | 0.265 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.8 |  concrete, abstract  concrete, abstract  concrete or abstract |
| back_front:long:label_colon:container | 0.223 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.0 |  [concrete  [concrete  [concrete |
| back_front:long:label_colon:number | 0.523 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.5 |  [category name  [the category  [category name |
| back_front:long:label_colon:plant | 0.568 | 0.38 | 0.12 | 0.38 | 0.12 | 0.25 | +0.12 | joint_lm:0.05 | other | 4.5 |  Abstract, if  Concrete. The  Concrete. The |
| back_front:long:label_colon:time | 0.583 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.8 |  Abstract, because  concrete, because  Abstract, because |
| back_front:long:list_answer:container | 0.143 | 0.50 | 0.38 | 0.50 | 0.38 | 0.50 | +0.12 | joint_lm:0.05 | quote | 9.0 |  "box"  "concrete  "bottle |
| back_front:long:list_answer:number | 0.125 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | quote | 8.8 |  "Concrete"  "Concrete"  "Concrete" |
| back_front:long:list_answer:plant | 0.471 | 0.25 | 0.00 | 0.25 | 0.25 | 0.00 | +0.00 | joint_internal:0.2 | other | 1.5 |  a or an  a or an  concrete object that |
| back_front:long:list_answer:time | 0.122 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | +0.12 | joint_lm:0.05 | quote | 12.1 |  "Morning"  "noon"  "Evening |
| back_front:long:multiple_choice:container | 0.123 | 1.00 | 0.88 | 1.00 | 0.88 | 0.88 | +0.00 | joint_internal:0.05 | other | 1.1 |  container. \n\n  container. \n\n  container. \n\n |
| back_front:long:multiple_choice:number | 0.406 | 1.00 | 0.62 | 1.00 | 0.75 | 1.00 | +0.38 | joint_lm:0.05 | other | 1.6 |  (A)  (A)  A. A |
| back_front:long:multiple_choice:plant | 0.472 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.2 |  container. The  plant. The  plant. The |
| back_front:long:multiple_choice:time | 0.480 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  time. The  time. The  time. The |
| back_front:long:quoted_answer:container | 0.238 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 12.5 | category" ( category" ( X" ( |
| back_front:long:quoted_answer:number | 0.226 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 12.8 | category" ( category" ( category" ( |
| back_front:long:quoted_answer:plant | 0.661 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 7.2 | ..." or " ..." or " ..." or " |
| back_front:long:quoted_answer:time | 0.301 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 14.2 | category" ( category" ( category" ( |
| back_front:neutral:answer_one_word:container | 0.420 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 4.6 |  1.  1.  1. |
| back_front:neutral:answer_one_word:number | 0.510 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 1.0 |  a a a  a a a  a a a |
| back_front:neutral:answer_one_word:plant | 0.516 | 0.12 | 0.50 | 0.12 | 0.00 | 0.38 | -0.12 | joint_lm:0.05 | other | 1.8 |  a. a   plant,  1. |
| back_front:neutral:answer_one_word:time | 0.510 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 4.9 |  1.  1.  1) |
| back_front:neutral:label_colon:container | 0.449 | 0.25 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 2.6 |  a group of  a group of  1. |
| back_front:neutral:label_colon:number | 0.507 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | quote | 3.0 |  the same or  the same or  "  " |
| back_front:neutral:label_colon:plant | 0.493 | 0.62 | 0.12 | 0.62 | 0.00 | 0.75 | +0.62 | joint_lm:0.05 | other | 1.2 |  1.  plant. Flower  plant. What |
| back_front:neutral:label_colon:time | 0.484 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | +0.00 | joint_internal:0.2 | other | 1.5 |  "Morning routine  time of day  "Evening |
| back_front:neutral:list_answer:container | 0.463 | 0.50 | 0.62 | 0.50 | 0.00 | 0.62 | +0.00 | joint_internal:0.2 | option_label | 5.8 |  A term box  A term bag  A term bottle |
| back_front:neutral:list_answer:number | 0.570 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  a. A  a. A  a. A |
| back_front:neutral:list_answer:plant | 0.579 | 0.25 | 0.25 | 0.25 | 0.12 | 0.12 | -0.12 | joint_internal:0.2 | other | 1.0 |  a a a  a flower is  a type of |
| back_front:neutral:list_answer:time | 0.575 | 0.38 | 0.38 | 0.38 | 0.00 | 0.25 | -0.12 | joint_internal:0.05 | other | 6.8 |  Morning is the  noon is the  A time of |
| back_front:neutral:multiple_choice:container | 0.159 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container. Term  container. Why  container. Why |
| back_front:neutral:multiple_choice:number | 0.378 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  number. Term  number. Why  number. Why |
| back_front:neutral:multiple_choice:plant | 0.403 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  plant. Why  plant. Why  plant. Why |
| back_front:neutral:multiple_choice:time | 0.378 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | +0.00 | joint_lm:0.05 | other | 1.2 |  time. Why  time. Why  time. Why |
| back_front:neutral:quoted_answer:container | 0.531 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.8 |  " " " ____" ( ____" " |
| back_front:neutral:quoted_answer:number | 0.637 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 7.5 | ____" for ____" for ____" for |
| back_front:neutral:quoted_answer:plant | 0.644 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 6.9 | ____" and Flower"  ".\n\n" |
| back_front:neutral:quoted_answer:time | 0.639 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 13.6 | Morning" is ____" ( evening" |
| back_front:short:answer_one_word:container | 0.830 | 0.50 | 0.50 | 0.62 | 0.00 | 0.50 | +0.00 | joint_internal:0.05 | other | 1.0 |  a. a  a. bag  bottle, glass |
| back_front:short:answer_one_word:number | 0.815 | 0.00 | 0.62 | 0.00 | 0.00 | 1.00 | +0.38 | joint_lm:0.05 | other | 1.0 |  number, object  number, unit  number, color |
| back_front:short:answer_one_word:plant | 0.246 | 0.88 | 0.88 | 0.88 | 0.75 | 0.88 | +0.00 | joint_internal:0.05 | other | 1.1 |  1.  plant, animal  plant, animal |
| back_front:short:answer_one_word:time | 0.842 | 0.25 | 0.88 | 0.38 | 0.00 | 0.88 | +0.00 | joint_internal:0.05 | other | 1.6 |  morning is a  a. time  the time of |
| back_front:short:label_colon:container | 0.778 | 0.75 | 0.12 | 0.75 | 0.38 | 0.25 | +0.12 | joint_lm:0.05 | other | 2.4 |  a box,  1. Container. A |
| back_front:short:label_colon:number | 0.809 | 0.50 | 0.75 | 0.62 | 0.00 | 1.00 | +0.25 | joint_lm:0.05 | other | 1.0 |  number, and  number, and  number, and |
| back_front:short:label_colon:plant | 0.301 | 0.75 | 0.75 | 0.75 | 0.62 | 0.88 | +0.12 | joint_lm:0.05 | other | 1.4 | Data structure. Plant. The Plant. The |
| back_front:short:label_colon:time | 0.804 | 0.62 | 0.62 | 0.75 | 0.38 | 0.75 | +0.12 | joint_lm:0.05 | other | 1.1 | time, and   (a time, and |
| back_front:short:list_answer:container | 0.798 | 0.75 | 0.25 | 0.75 | 0.00 | 0.25 | +0.00 | joint_internal:0.2 | other | 1.0 |  a box of  a a a  a\n  |
| back_front:short:list_answer:number | 0.791 | 0.25 | 0.00 | 0.25 | 0.00 | 0.25 | +0.25 | joint_lm:0.05 | other | 1.6 |  a. a  a. number  1. |
| back_front:short:list_answer:plant | 0.220 | 0.62 | 0.62 | 0.62 | 0.00 | 1.00 | +0.38 | joint_lm:0.05 | option_label | 1.6 |  A. tree  A. plant  plant\n- |
| back_front:short:list_answer:time | 0.811 | 0.50 | 0.50 | 0.50 | 0.00 | 0.62 | +0.12 | joint_lm:0.05 | other | 1.0 |  a. time  a. a  a. time |
| back_front:short:multiple_choice:container | 0.539 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container. The  container. The  container. The |
| back_front:short:multiple_choice:number | 0.548 | 1.00 | 0.25 | 1.00 | 1.00 | 0.75 | +0.50 | joint_lm:0.05 | other | 1.4 |  number. The  number. The  number. The |
| back_front:short:multiple_choice:plant | 0.543 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. The  plant. The |
| back_front:short:multiple_choice:time | 0.542 | 1.00 | 0.75 | 1.00 | 0.88 | 1.00 | +0.25 | joint_lm:0.05 | other | 1.1 |  time. The  time. The  time. The |
| back_front:short:quoted_answer:container | 0.799 | 0.75 | 0.62 | 0.75 | 0.62 | 0.75 | +0.12 | joint_lm:0.05 | other | 1.2 | box" is bag" is container" or |
| back_front:short:quoted_answer:number | 0.790 | 0.38 | 0.00 | 0.50 | 0.38 | 0.50 | +0.50 | joint_lm:0.05 | other | 1.5 | ______" number" or number" or |
| back_front:short:quoted_answer:plant | 0.267 | 0.88 | 0.75 | 0.88 | 0.50 | 1.00 | +0.25 | joint_lm:0.05 | other | 1.0 | tree" is plant" or plant" or |
| back_front:short:quoted_answer:time | 0.812 | 0.12 | 0.12 | 0.12 | 0.12 | 0.88 | +0.75 | joint_lm:0.05 | other | 1.4 | time of day time" or time of day |
| front_back:long:answer_one_word:container | 0.154 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 9.0 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:number | 0.215 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.5 |  concrete, abstract  concrete, abstract  concrete or abstract |
| front_back:long:answer_one_word:plant | 0.185 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 5.2 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:time | 0.142 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.2 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:label_colon:container | 0.196 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.2 |  [concrete  [category]  [concrete |
| front_back:long:label_colon:number | 0.580 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 6.6 |  [category name  [category name  Concrete. The |
| front_back:long:label_colon:plant | 0.602 | 0.25 | 0.00 | 0.25 | 0.12 | 0.00 | +0.00 | joint_lm:0.05 | other | 4.1 |  Concrete. The  concrete. The  Concrete. The |
| front_back:long:label_colon:time | 0.614 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.1 |  concrete. The  Concrete. The  [concrete |
| front_back:long:list_answer:container | 0.130 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | quote | 9.4 |  "barrel  1.  "Chest |
| front_back:long:list_answer:number | 0.144 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | quote | 10.5 |  "concrete  "concrete  "dozen |
| front_back:long:list_answer:plant | 0.134 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 | +0.12 | joint_lm:0.05 | quote | 9.0 |  "Moss  "Algae  "Bam |
| front_back:long:list_answer:time | 0.143 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | +0.12 | joint_lm:0.05 | quote | 11.0 |  "month"  If the context  "Spring" |
| front_back:long:multiple_choice:container | 0.122 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container. \n\n  container. \n\n  container. The |
| front_back:long:multiple_choice:number | 0.438 | 0.88 | 1.00 | 0.88 | 0.88 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.4 |  number. The  number. The  number. The |
| front_back:long:multiple_choice:plant | 0.484 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. But  plant. The |
| front_back:long:multiple_choice:time | 0.435 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  time. The  time. The  time. The |
| front_back:long:quoted_answer:container | 0.188 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 12.4 | category" ( category" ( category" ( |
| front_back:long:quoted_answer:number | 0.192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.8 | category" ( category" ( category" ( |
| front_back:long:quoted_answer:plant | 0.216 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 11.1 | category" ( ..." (e X" ( |
| front_back:long:quoted_answer:time | 0.174 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 14.2 | category" ( category" ( category" ( |
| front_back:neutral:answer_one_word:container | 0.436 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 1.2 |  a. a  a a a   "chest |
| front_back:neutral:answer_one_word:number | 0.486 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | quote | 3.2 |  1.  "dog,  dozen, semantic |
| front_back:neutral:answer_one_word:plant | 0.519 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_internal:0.2 | other | 1.2 |  a. moss  a type of   bamboo, |
| front_back:neutral:answer_one_word:time | 0.473 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | +0.25 | joint_internal:0.05 | quote | 3.2 |  the following are  "term year  1. |
| front_back:neutral:label_colon:container | 0.215 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 17.4 |  1.  1.  1. |
| front_back:neutral:label_colon:number | 0.481 | 0.00 | 0.12 | 0.00 | 0.00 | 0.25 | +0.12 | joint_lm:0.05 | whitespace | 4.4 |  1.  1.  Number. A |
| front_back:neutral:label_colon:plant | 0.489 | 0.25 | 0.38 | 0.38 | 0.12 | 0.50 | +0.12 | joint_lm:0.05 | other | 1.6 |  plant. Sub  Algae.  1. |
| front_back:neutral:label_colon:time | 0.497 | 0.25 | 0.12 | 0.25 | 0.00 | 0.25 | +0.12 | joint_lm:0.05 | whitespace | 2.8 |  1.  1.  1. |
| front_back:neutral:list_answer:container | 0.493 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_internal:0.2 | other | 1.0 |  a hollow body  a a a  a body cavity |
| front_back:neutral:list_answer:number | 0.549 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | option_label | 2.4 |  A group of  The first and  A dozen is |
| front_back:neutral:list_answer:plant | 0.587 | 0.38 | 0.25 | 0.38 | 0.00 | 0.38 | +0.12 | joint_internal:0.2 | other | 1.5 |  Moss is a  a type of  a type of |
| front_back:neutral:list_answer:time | 0.540 | 0.25 | 0.25 | 0.25 | 0.00 | 0.38 | +0.12 | joint_internal:0.2 | other | 1.2 |  a a a  a a a  a time of |
| front_back:neutral:multiple_choice:container | 0.310 | 1.00 | 0.38 | 1.00 | 1.00 | 1.00 | +0.62 | joint_internal:0.2 | other | 1.2 |  container. I  plant. \n\n  container. I |
| front_back:neutral:multiple_choice:number | 0.382 | 0.88 | 0.75 | 0.88 | 0.88 | 0.88 | +0.12 | joint_lm:0.05 | other | 1.2 |  number. Why  number. Why  number. The |
| front_back:neutral:multiple_choice:plant | 0.363 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  plant. Why  plant. Why  plant. Why |
| front_back:neutral:multiple_choice:time | 0.405 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  time. \n\n  time. Why  time. Why |
| front_back:neutral:quoted_answer:container | 0.560 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.4 | ____"  ____" for ____" " |
| front_back:neutral:quoted_answer:number | 0.596 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 18.0 |  " " "  " " " ____" The |
| front_back:neutral:quoted_answer:plant | 0.648 | 0.38 | 0.38 | 0.38 | 0.25 | 0.38 | +0.00 | joint_lm:0.05 | other | 6.9 | moss" algae" bamboo" |
| front_back:neutral:quoted_answer:time | 0.600 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | quote | 19.0 |  ".\n\n" ____" and  ".\n\n" |
| front_back:short:answer_one_word:container | 0.822 | 0.12 | 0.00 | 0.12 | 0.00 | 0.38 | +0.38 | joint_internal:0.2 | other | 1.4 |  volume, weight  a. a  cabinet, drawer |
| front_back:short:answer_one_word:number | 0.805 | 0.25 | 0.38 | 0.25 | 0.00 | 0.75 | +0.38 | joint_lm:0.05 | other | 1.1 |  a. number  number, object  number, object |
| front_back:short:answer_one_word:plant | 0.821 | 1.00 | 0.75 | 1.00 | 1.00 | 0.88 | +0.12 | joint_lm:0.05 | other | 1.2 |  plant, animal  plant, fungus  plant, animal |
| front_back:short:answer_one_word:time | 0.836 | 0.12 | 0.50 | 0.12 | 0.12 | 0.50 | +0.00 | joint_internal:0.2 | other | 1.5 |  a time period  a time period  a a a |
| front_back:short:label_colon:container | 0.750 | 0.25 | 0.12 | 0.25 | 0.00 | 0.25 | +0.12 | joint_lm:0.05 | other | 7.1 | 1:1  a class of  Furniture. The |
| front_back:short:label_colon:number | 0.776 | 0.38 | 0.62 | 0.50 | 0.00 | 0.88 | +0.25 | joint_lm:0.05 | other | 1.4 |  number, and  number, and  quantity, and |
| front_back:short:label_colon:plant | 0.758 | 0.88 | 0.88 | 0.88 | 0.75 | 0.88 | +0.00 | joint_lm:0.05 | other | 1.1 |  plant. What ____. (  plant. The |
| front_back:short:label_colon:time | 0.752 | 0.50 | 0.62 | 0.50 | 0.50 | 0.75 | +0.12 | joint_lm:0.05 | other | 2.6 | time. A Time. A   (a |
| front_back:short:list_answer:container | 0.786 | 0.25 | 0.12 | 0.25 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 1.0 |  a a a  a a a  a airt |
| front_back:short:list_answer:number | 0.773 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 1.0 |  a. a  a. a  a. number |
| front_back:short:list_answer:plant | 0.783 | 0.88 | 0.38 | 0.75 | 0.00 | 0.62 | +0.25 | joint_lm:0.05 | other | 1.2 |  a. plant  plant\n-  plant\n- |
| front_back:short:list_answer:time | 0.800 | 0.38 | 0.00 | 0.38 | 0.00 | 0.38 | +0.38 | joint_lm:0.05 | other | 1.0 |  a. time  a. a  a. a |
| front_back:short:multiple_choice:container | 0.489 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | +0.12 | joint_internal:0.05 | other | 1.0 |  container. The  container. The  container. The |
| front_back:short:multiple_choice:number | 0.477 | 0.88 | 0.50 | 0.88 | 0.88 | 0.75 | +0.25 | joint_lm:0.05 | other | 1.8 |  number. The  number. The  number. The |
| front_back:short:multiple_choice:plant | 0.511 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. The  plant. The |
| front_back:short:multiple_choice:time | 0.481 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.2 |  time. The  time. The  plant. The |
| front_back:short:quoted_answer:container | 0.800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 5.0 | barrel"  ".\n\nThe furniture" |
| front_back:short:quoted_answer:number | 0.785 | 0.25 | 0.00 | 0.50 | 0.12 | 0.62 | +0.62 | joint_lm:0.05 | other | 2.8 | number" or number" or number" or |
| front_back:short:quoted_answer:plant | 0.800 | 0.50 | 0.62 | 0.50 | 0.50 | 0.50 | -0.12 | joint_internal:0.2 | other | 3.0 | algae", algae"\n\n grass"\n\nWait |
| front_back:short:quoted_answer:time | 0.816 | 0.38 | 0.50 | 0.38 | 0.50 | 0.50 | +0.00 | joint_lm:0.05 | other | 4.4 | time period" time period" ______" |

## glm4

cases=120, formats=label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer, semantic_scale=0.05, format_scales=[0.05, 0.2]

### By category

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.282 | 0.30 | 0.28 | 0.30 | 0.32 | 0.35 | +0.07 | 15.8 | 12.1 | other | other |
| number | 30 | 0.268 | 0.21 | 0.21 | 0.21 | 0.18 | 0.24 | +0.03 | 18.9 | 10.6 | other | other |
| plant | 30 | 0.236 | 0.38 | 0.36 | 0.38 | 0.36 | 0.49 | +0.13 | 10.5 | 17.6 | other | other |
| time | 30 | 0.269 | 0.30 | 0.29 | 0.30 | 0.28 | 0.32 | +0.03 | 27.0 | 14.4 | other | other |

### By format

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.302 | 0.07 | 0.07 | 0.07 | 0.09 | 0.16 | +0.09 | 13.7 | 7.0 | other | other |
| label_colon | 24 | 0.244 | 0.18 | 0.17 | 0.18 | 0.20 | 0.27 | +0.09 | 19.2 | 20.0 | other | other |
| list_answer | 24 | 0.247 | 0.16 | 0.13 | 0.16 | 0.13 | 0.23 | +0.10 | 16.9 | 6.3 | other | other |
| multiple_choice | 24 | 0.240 | 0.98 | 0.97 | 0.98 | 0.94 | 0.98 | +0.01 | 1.2 | 13.6 | other | canonical |
| quoted_answer | 24 | 0.286 | 0.10 | 0.09 | 0.10 | 0.07 | 0.11 | +0.03 | 39.4 | 21.5 | other | other |

### By family

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.290 | 0.21 | 0.21 | 0.21 | 0.21 | 0.24 | +0.03 | 25.0 | 19.7 | other | other |
| neutral | 40 | 0.238 | 0.29 | 0.28 | 0.29 | 0.25 | 0.30 | +0.02 | 24.6 | 13.1 | other | other |
| short | 40 | 0.263 | 0.39 | 0.37 | 0.39 | 0.39 | 0.51 | +0.14 | 4.5 | 8.3 | other | other |

### By split

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.262 | 0.32 | 0.30 | 0.32 | 0.31 | 0.38 | +0.07 | 18.1 | 13.7 | other | other |
| front_back | 60 | 0.265 | 0.28 | 0.27 | 0.28 | 0.26 | 0.32 | +0.05 | 18.1 | 13.6 | other | other |

### Cases

| case | overlap | clean | sem | fmt_int | fmt_lm | best_joint | gain | joint | fmt_group | answer_rank | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.469 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 10.4 | Step 1 Step 1 #Step  |
| back_front:long:answer_one_word:number | 0.315 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 19.5 | Step 1 Step 1 Step 1 |
| back_front:long:answer_one_word:plant | 0.238 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.2 | Solution:\n\nTo Solution:\n\nTo Step 1 |
| back_front:long:answer_one_word:time | 0.364 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 18.5 | Step 1 Step 1 Step 1 |
| back_front:long:label_colon:container | 0.297 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 37.4 |  box.\n Abstract.In Concet |
| back_front:long:label_colon:number | 0.172 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.6 | Abstract.The Abstract.In Abstract.In |
| back_front:long:label_colon:plant | 0.171 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 6.6 | Entity.\n Flower.\n\n  concrete entity.\n\n |
| back_front:long:label_colon:time | 0.270 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.9 | Abstract.In Abstract.In Abstract. Evening |
| back_front:long:list_answer:container | 0.287 | 0.25 | 0.12 | 0.25 | 0.25 | 0.38 | +0.25 | joint_lm:0.05 | other | 9.2 |  Box\n-  Concrete entity\n  concrete entity\n |
| back_front:long:list_answer:number | 0.198 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 18.9 |  Answer\nTo  Answer\n-  Answer\nTo |
| back_front:long:list_answer:plant | 0.223 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 25.1 |  Answer\nTo  Flower# Step  concrete entity\n |
| back_front:long:list_answer:time | 0.332 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 32.5 |  concrete\n-  concrete\n-  concrete\n- |
| back_front:long:multiple_choice:container | 0.184 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container.\n\nThe  container.\n\n#  container.\n\n# |
| back_front:long:multiple_choice:number | 0.248 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  number\n\nThe  number.\n\nThe  number.\n\nThe |
| back_front:long:multiple_choice:plant | 0.252 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.1 |  container.\n\nThe plant\n\nThe plant\n\nIn |
| back_front:long:multiple_choice:time | 0.266 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | time\n\nExplanation time\n\nSolution time\n\nExplanation |
| back_front:long:quoted_answer:container | 0.497 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 15.8 | CONCRETE CONCRETE CONCRETE |
| back_front:long:quoted_answer:number | 0.316 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 67.4 | Answer: " CONCRETE CONCRETE |
| back_front:long:quoted_answer:plant | 0.235 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 43.1 | CONCRETE concrete" CONCRETE |
| back_front:long:quoted_answer:time | 0.335 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 137.6 | CONCRETE CONCRETE CONCRETE |
| back_front:neutral:answer_one_word:container | 0.151 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 23.1 | Step 1 Step 1 Step 1 |
| back_front:neutral:answer_one_word:number | 0.268 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 29.6 | Step 1 Step 1 Step 1 |
| back_front:neutral:answer_one_word:plant | 0.139 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 17.2 | Step 1 Step 1 Step 1 |
| back_front:neutral:answer_one_word:time | 0.148 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 27.9 | Step 1 Step 1 Step 1 |
| back_front:neutral:label_colon:container | 0.440 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | +0.00 | joint_internal:0.05 | other | 70.5 | Term box. Term bag. Term bottle. |
| back_front:neutral:label_colon:number | 0.449 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 32.6 |  Category: Category  Noun.  Noun: |
| back_front:neutral:label_colon:plant | 0.215 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | +0.25 | joint_lm:0.05 | other | 2.6 |  Term tree. Flora.  Plant. Category |
| back_front:neutral:label_colon:time | 0.308 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 60.6 | Time. The Term:noon  Term evening. |
| back_front:neutral:list_answer:container | 0.151 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 14.4 |    -    #1.  Term bottle\n\n |
| back_front:neutral:list_answer:number | 0.161 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 16.9 |  #1.  #1.  #1. |
| back_front:neutral:list_answer:plant | 0.158 | 0.25 | 0.12 | 0.25 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 30.2 |  Semantic group:  flower# Step  semantic group: |
| back_front:neutral:list_answer:time | 0.159 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 21.0 |  morning\n\n-  12:  1. |
| back_front:neutral:multiple_choice:container | 0.315 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  containerThe term  containerThe term  container\n\nExplanation |
| back_front:neutral:multiple_choice:number | 0.204 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.2 |  plant\n\nStep  number\n\nSolution  number\n\nSolution |
| back_front:neutral:multiple_choice:plant | 0.215 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant, container  plant\n\nThe  plant, time |
| back_front:neutral:multiple_choice:time | 0.166 | 0.88 | 0.75 | 0.88 | 0.88 | 0.88 | +0.12 | joint_lm:0.05 | other | 1.9 | time\n\nThe  timeThe term time\n\n# |
| back_front:neutral:quoted_answer:container | 0.154 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 2.9 | Cats" ____"Step Term" " |
| back_front:neutral:quoted_answer:number | 0.238 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 88.9 | ____"Step ________"Step ______" |
| back_front:neutral:quoted_answer:plant | 0.163 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | +0.00 | joint_lm:0.05 | other | 9.8 | Term tree. Flower" Grass" |
| back_front:neutral:quoted_answer:time | 0.145 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 54.5 | morning"\n\n ________"Step Term evening" |
| back_front:short:answer_one_word:container | 0.334 | 0.62 | 0.62 | 0.62 | 0.75 | 0.88 | +0.25 | joint_lm:0.05 | other | 2.0 |  box. The  container. The  container.\n\nThe |
| back_front:short:answer_one_word:number | 0.320 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 3.4 |  one. This  two, three  three, four |
| back_front:short:answer_one_word:plant | 0.331 | 0.38 | 0.38 | 0.38 | 0.25 | 0.88 | +0.50 | joint_lm:0.05 | other | 1.2 |  animal, plant  plant, tree Step 1 |
| back_front:short:answer_one_word:time | 0.316 | 0.00 | 0.00 | 0.00 | 0.12 | 0.25 | +0.25 | joint_lm:0.05 | other | 4.4 | Answer:time Answer\n\n#  "time of |
| back_front:short:label_colon:container | 0.140 | 0.50 | 0.62 | 0.50 | 0.62 | 0.62 | +0.00 | joint_lm:0.05 | other | 1.5 |  Container. A  ContainerA container  ContainerA container |
| back_front:short:label_colon:number | 0.134 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 4.5 |  One. The Category:Two Category:Category |
| back_front:short:label_colon:plant | 0.120 | 0.38 | 0.12 | 0.38 | 0.62 | 0.88 | +0.75 | joint_lm:0.05 | other | 1.0 |  Plant. Trees  plant.Fl Plant.Gr |
| back_front:short:label_colon:time | 0.207 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | +0.00 | joint_internal:0.05 | other | 2.5 | Time of day Time periods. Time period. |
| back_front:short:list_answer:container | 0.350 | 0.12 | 0.12 | 0.12 | 0.50 | 0.50 | +0.38 | joint_lm:0.05 | other | 5.0 |  [ ]   item 1  A container\n |
| back_front:short:list_answer:number | 0.352 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | +0.25 | joint_lm:0.05 | other | 3.6 |  item 1  Answer 1  Answer 1 |
| back_front:short:list_answer:plant | 0.327 | 0.62 | 0.50 | 0.62 | 0.88 | 1.00 | +0.50 | joint_lm:0.05 | other | 2.6 |  type of plant  plant\n-  Plant\n- |
| back_front:short:list_answer:time | 0.342 | 0.38 | 0.38 | 0.38 | 0.38 | 0.50 | +0.12 | joint_lm:0.05 | other | 5.2 |  time of day  type of time  Time of day |
| back_front:short:multiple_choice:container | 0.246 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | container\n\nThe containerThe word containerThe word |
| back_front:short:multiple_choice:number | 0.330 | 1.00 | 1.00 | 1.00 | 0.38 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | number\n\nThe number\n\nThe number\n\nThe |
| back_front:short:multiple_choice:plant | 0.298 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | plant\n\nThe plant\n\nThe plant\n\nThe |
| back_front:short:multiple_choice:time | 0.360 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | time\n\nExplanation time\n\nExplanation time\n\nExplanation |
| back_front:short:quoted_answer:container | 0.320 | 0.25 | 0.12 | 0.25 | 0.12 | 0.25 | +0.12 | joint_internal:0.2 | other | 10.1 | category word". category word." Answer: bottle |
| back_front:short:quoted_answer:number | 0.304 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 8.9 | one" " category word" category word" |
| back_front:short:quoted_answer:plant | 0.307 | 0.25 | 0.12 | 0.25 | 0.38 | 0.50 | +0.38 | joint_lm:0.05 | other | 5.2 | tree" refers flower" is Answer" " |
| back_front:short:quoted_answer:time | 0.291 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 14.1 | type of"\n\n type of" type of day |
| front_back:long:answer_one_word:container | 0.461 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 14.4 | Solution:\n\nTo Step 1  concrete or abstract |
| front_back:long:answer_one_word:number | 0.453 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 17.4 | Step 1 Step 1 #Answer: |
| front_back:long:answer_one_word:plant | 0.253 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 11.4 | Step 1 Step 1 Answer: plant |
| front_back:long:answer_one_word:time | 0.403 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 18.9 | Solution:\n\nTo Step 1 Step 1 |
| front_back:long:label_colon:container | 0.320 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 60.2 | Abstract.Bar Entity.\n Abstract.In |
| front_back:long:label_colon:number | 0.185 | 0.00 | 0.12 | 0.00 | 0.00 | 0.50 | +0.38 | joint_lm:0.05 | other | 2.8 |  Count Noun  Count Noun Abstract.In |
| front_back:long:label_colon:plant | 0.143 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | +0.25 | joint_lm:0.05 | other | 12.2 |  Moss.\n\nIn Algae.\n\n  Plant.In |
| front_back:long:label_colon:time | 0.106 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 20.1 | Month.\n Abstract.In Abstract.In |
| front_back:long:list_answer:container | 0.278 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 22.4 |  Answer\n-  concrete\n-  concrete entity\n |
| front_back:long:list_answer:number | 0.274 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 17.9 |  Answer\n-  Answer\n-  Answer\n- |
| front_back:long:list_answer:plant | 0.241 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 18.4 |  Moss\n-  Algae\n  bamboo\n- |
| front_back:long:list_answer:time | 0.333 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 24.9 |  concrete entity\n  concrete entity\n  Concrete entity\n |
| front_back:long:multiple_choice:container | 0.194 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container.\n\nExplanation  container.\n\nThe  container.\n\nThe |
| front_back:long:multiple_choice:number | 0.246 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | +0.00 | joint_lm:0.05 | other | 2.2 |  number.\n\nThe  number.\n\nThe  number.\n\nThe |
| front_back:long:multiple_choice:plant | 0.239 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant\n\nExplanation  plantThe semantic  plant.\n\nThe |
| front_back:long:multiple_choice:time | 0.234 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 | time\n\n# time\n\n# time\n\nSolution |
| front_back:long:quoted_answer:container | 0.411 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 26.6 | CONCRETE CONCRETE CONCRETE |
| front_back:long:quoted_answer:number | 0.436 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 64.9 | CONCRETE CONCRETE CONCRETE |
| front_back:long:quoted_answer:plant | 0.246 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 38.0 | CONCRETE Entity"Step CONCRETE |
| front_back:long:quoted_answer:time | 0.476 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 128.5 | CONCRETE CONCRETE CONCRETE |
| front_back:neutral:answer_one_word:container | 0.224 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 25.5 | Step 1 Step 1 Step 1 |
| front_back:neutral:answer_one_word:number | 0.274 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 17.2 | Step 1 Step 1 Step 1 |
| front_back:neutral:answer_one_word:plant | 0.230 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 14.1 | Step 1 Step 1 Step 1 |
| front_back:neutral:answer_one_word:time | 0.240 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 21.4 | Step 1 Step 1 Step 1 |
| front_back:neutral:label_colon:container | 0.364 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 57.6 | Barrel. Term case. Term:  |
| front_back:neutral:label_colon:number | 0.449 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 29.9 |  Noun\n\n  Term ten.  English words. |
| front_back:neutral:label_colon:plant | 0.308 | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | +0.25 | joint_lm:0.05 | other | 2.4 | Botany. Algae\n\n Plant. The |
| front_back:neutral:label_colon:time | 0.472 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | +0.00 | joint_lm:0.05 | other | 3.8 | Term month. Term year. Seasons. |
| front_back:neutral:list_answer:container | 0.223 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 25.1 |  1.  1.  1. |
| front_back:neutral:list_answer:number | 0.203 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 14.4 |  #1.  #1.  #1\n |
| front_back:neutral:list_answer:plant | 0.216 | 0.38 | 0.38 | 0.38 | 0.12 | 0.38 | +0.00 | joint_internal:0.05 | other | 35.5 |  mosses\n  Algae\n\n  bamboo# Step |
| front_back:neutral:list_answer:time | 0.195 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 40.9 |  #1.  #1.  spring\n\n- |
| front_back:neutral:multiple_choice:container | 0.246 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.1 | container\n\nThe  plant\n\nThe container\n\nThe |
| front_back:neutral:multiple_choice:number | 0.184 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | +0.00 | joint_internal:0.05 | other | 1.2 |  number\n\nExplanation number\n\nStep number\n\nExplanation |
| front_back:neutral:multiple_choice:plant | 0.258 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plantThe term plant\n\nExplanation plant\n\nExplanation |
| front_back:neutral:multiple_choice:time | 0.254 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.4 |  plant\n\nThe  plant\n\nThe time\n\nThe |
| front_back:neutral:quoted_answer:container | 0.226 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 4.8 | Barrel" ____"Step Chest" |
| front_back:neutral:quoted_answer:number | 0.207 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 62.2 | ______" ____"Step group"\n\nStep |
| front_back:neutral:quoted_answer:plant | 0.168 | 0.38 | 0.38 | 0.38 | 0.25 | 0.38 | +0.00 | joint_lm:0.05 | other | 7.1 | Moss"\n\n Algae" Bamboo |
| front_back:neutral:quoted_answer:time | 0.240 | 0.25 | 0.25 | 0.25 | 0.12 | 0.25 | +0.00 | joint_internal:0.05 | other | 109.6 | Step 1 Term year. Spring"\n\nStep |
| front_back:short:answer_one_word:container | 0.335 | 0.12 | 0.12 | 0.12 | 0.12 | 0.38 | +0.25 | joint_lm:0.05 | other | 4.8 |  container, vessel  "case"  "box" |
| front_back:short:answer_one_word:number | 0.280 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | +0.12 | joint_lm:0.05 | other | 2.9 |  "Answer:  ten, one  dozen. The |
| front_back:short:answer_one_word:plant | 0.421 | 0.38 | 0.38 | 0.38 | 0.38 | 0.88 | +0.50 | joint_lm:0.05 | other | 1.8 | a plant, biological organism  plant\n\n# |
| front_back:short:answer_one_word:time | 0.285 | 0.12 | 0.00 | 0.12 | 0.25 | 0.12 | +0.12 | joint_lm:0.05 | other | 4.9 | Answer\n\n#  answer choices: Answer\n\n# |
| front_back:short:label_colon:container | 0.172 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | +0.12 | joint_lm:0.05 | other | 6.0 |  Container. Bar Case. The Storage containerA |
| front_back:short:label_colon:number | 0.140 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | +0.00 | joint_lm:0.05 | other | 5.4 | Mathematics. Category:Category Counting. |
| front_back:short:label_colon:plant | 0.121 | 0.25 | 0.25 | 0.25 | 0.25 | 0.38 | +0.12 | joint_lm:0.05 | other | 1.6 | Non-vascular  PlantA: Grass. |
| front_back:short:label_colon:time | 0.152 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | +0.00 | joint_internal:0.05 | other | 5.5 | Time period. Time period. Seasons. |
| front_back:short:list_answer:container | 0.210 | 0.25 | 0.00 | 0.25 | 0.38 | 0.38 | +0.38 | joint_lm:0.05 | other | 6.4 |  A container\n  Answer 1  item 1 |
| front_back:short:list_answer:number | 0.210 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 4.0 |  Number\n-  Ten types of  1. |
| front_back:short:list_answer:plant | 0.276 | 0.75 | 0.75 | 0.62 | 0.50 | 1.00 | +0.25 | joint_lm:0.05 | other | 1.0 |  plant\n-  plant\n-  plant\n\n- |
| front_back:short:list_answer:time | 0.216 | 0.12 | 0.00 | 0.12 | 0.00 | 0.25 | +0.25 | joint_lm:0.05 | other | 9.5 |  Answer: month  The word year  Answer: spring |
| front_back:short:multiple_choice:container | 0.161 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | containerThe word  container.\n\nThe containerThe word |
| front_back:short:multiple_choice:number | 0.237 | 0.88 | 0.88 | 0.88 | 0.62 | 0.88 | +0.00 | joint_internal:0.05 | other | 1.2 | number\n\nThe number\n\nThe number\n\nThe |
| front_back:short:multiple_choice:plant | 0.221 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 | plant\n\nM plant\n\nAl plant\n\nB |
| front_back:short:multiple_choice:time | 0.203 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.2 | time\n\nThe time.The plant. The |
| front_back:short:quoted_answer:container | 0.305 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 12.4 | category word" Answer" " category word" |
| front_back:short:quoted_answer:number | 0.254 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 8.8 | category word". category word." category word." |
| front_back:short:quoted_answer:plant | 0.344 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | +0.00 | joint_lm:0.05 | other | 4.1 | moss" _________"\n\n Answer with one |
| front_back:short:quoted_answer:time | 0.245 | 0.25 | 0.25 | 0.25 | 0.12 | 0.25 | +0.00 | joint_internal:0.2 | other | 19.2 | month"# Answer: Year Answer: spring |

## deepseek7b

cases=120, formats=label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer, semantic_scale=0.05, format_scales=[0.05, 0.2]

### By category

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| container | 30 | 0.469 | 0.26 | 0.21 | 0.26 | 0.14 | 0.22 | +0.01 | 20.5 | 10.6 | other | other |
| number | 30 | 0.411 | 0.21 | 0.20 | 0.21 | 0.08 | 0.26 | +0.05 | 42.5 | 56.1 | other | other |
| plant | 30 | 0.456 | 0.33 | 0.31 | 0.30 | 0.18 | 0.37 | +0.06 | 7.6 | 45.3 | other | other |
| time | 30 | 0.408 | 0.22 | 0.18 | 0.22 | 0.15 | 0.21 | +0.03 | 19.2 | 11.7 | other | other |

### By format

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.431 | 0.06 | 0.03 | 0.06 | 0.01 | 0.06 | +0.03 | 21.5 | 31.4 | other | other |
| label_colon | 24 | 0.467 | 0.06 | 0.09 | 0.06 | 0.02 | 0.12 | +0.04 | 31.0 | 63.0 | other | other |
| list_answer | 24 | 0.473 | 0.19 | 0.11 | 0.16 | 0.01 | 0.14 | +0.03 | 26.1 | 16.7 | other | other |
| multiple_choice | 24 | 0.394 | 0.88 | 0.82 | 0.88 | 0.59 | 0.88 | +0.06 | 2.1 | 3.3 | other | canonical |
| quoted_answer | 24 | 0.416 | 0.09 | 0.08 | 0.09 | 0.06 | 0.12 | +0.04 | 31.5 | 40.3 | other | other |

### By family

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 40 | 0.379 | 0.28 | 0.26 | 0.29 | 0.18 | 0.28 | +0.02 | 11.2 | 3.3 | other | other |
| neutral | 40 | 0.426 | 0.20 | 0.18 | 0.19 | 0.05 | 0.24 | +0.06 | 17.9 | 11.4 | other | other |
| short | 40 | 0.504 | 0.28 | 0.24 | 0.27 | 0.17 | 0.28 | +0.04 | 38.3 | 78.0 | other | other |

### By split

| group | n | overlap_max | clean | sem | fmt_int | fmt_lm | best_joint | joint_gain_vs_sem | answer_rank | format_rank | top_fmt_group | best_class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 60 | 0.454 | 0.27 | 0.25 | 0.27 | 0.14 | 0.27 | +0.02 | 12.0 | 17.9 | other | other |
| front_back | 60 | 0.418 | 0.24 | 0.21 | 0.23 | 0.14 | 0.26 | +0.05 | 32.9 | 43.9 | other | other |

### Cases

| case | overlap | clean | sem | fmt_int | fmt_lm | best_joint | gain | joint | fmt_group | answer_rank | examples |
|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.391 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 1.1 |  In  Babab BABBM |
| back_front:long:answer_one_word:number | 0.414 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 19.8 |  concrete, abstract  "Answer:  concrete, abstract |
| back_front:long:answer_one_word:plant | 0.335 | 0.12 | 0.12 | 0.12 | 0.12 | 0.25 | +0.12 | joint_lm:0.05 | other | 8.8 |  concrete, abstract  flower, flowers  grass, concrete |
| back_front:long:answer_one_word:time | 0.227 | 0.00 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 25.5 |  concrete, abstract  concrete, abstract  \boxed{ |
| back_front:long:label_colon:container | 0.522 | 0.00 | 0.25 | 0.00 | 0.00 | 0.25 | +0.00 | joint_lm:0.05 | other | 23.2 |  [box]  [1]  [Bottle |
| back_front:long:label_colon:number | 0.552 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 3.4 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nWait |
| back_front:long:label_colon:plant | 0.419 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.1 |  abstract.\n\nWait  [ ].\n\n  abstract.\n\nGr |
| back_front:long:label_colon:time | 0.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 29.2 |  [1]  [1]  [1] |
| back_front:long:list_answer:container | 0.467 | 0.62 | 0.00 | 0.62 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 4.2 |  1.   (i  1. |
| back_front:long:list_answer:number | 0.496 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 20.4 |  1.  1.  1. |
| back_front:long:list_answer:plant | 0.415 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | +0.00 | joint_lm:0.05 | other | 18.2 |  tree as a  flower\n-  grass is a |
| back_front:long:list_answer:time | 0.228 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 21.9 |  Morning is a  1.  evening\n\n- |
| back_front:long:multiple_choice:container | 0.342 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container.\n\n Why  container.\n\n Why  container.\n\nBut |
| back_front:long:multiple_choice:number | 0.382 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  number.\n\nBut  number.\n\nBut  number.\n\nBut |
| back_front:long:multiple_choice:plant | 0.321 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  container.\n\nBut  plant.\n\nOkay  plant.\n\nOkay |
| back_front:long:multiple_choice:time | 0.353 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  time.\n\nBut  time.\n\nBut  time.\n\nOkay |
| back_front:long:quoted_answer:container | 0.489 | 0.62 | 0.50 | 0.62 | 0.38 | 0.50 | +0.00 | joint_internal:0.05 | other | 3.0 | box" or bag" or bottle", |
| back_front:long:quoted_answer:number | 0.557 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 5.8 |  ".\n\n.\n\n  ".\n\n.\n\n abstract" or |
| back_front:long:quoted_answer:plant | 0.443 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | +0.00 | joint_lm:0.05 | other | 4.2 | tree" or flower" or grass" is |
| back_front:long:quoted_answer:time | 0.270 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 5.9 | morning" concrete" evening" |
| back_front:neutral:answer_one_word:container | 0.492 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 13.6 |  9,  9,  9, |
| back_front:neutral:answer_one_word:number | 0.434 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 9.8 |  1.  1.  21 |
| back_front:neutral:answer_one_word:plant | 0.567 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 13.1 |  **Question:  flower, flower  \n\nQuestion: |
| back_front:neutral:answer_one_word:time | 0.563 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 8.6 |  (e.g  1.  1. |
| back_front:neutral:label_colon:container | 0.413 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 47.4 |  Geometry.   92  Geometry.  |
| back_front:neutral:label_colon:number | 0.520 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | list_marker | 30.6 | 1.1 1.   6. |
| back_front:neutral:label_colon:plant | 0.594 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 13.5 |  62  Geometry, Geometry  62 |
| back_front:neutral:label_colon:time | 0.133 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 75.5 |  1. 1.1  1. |
| back_front:neutral:list_answer:container | 0.563 | 0.38 | 0.00 | 0.25 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 10.8 |  1.  1.  1. |
| back_front:neutral:list_answer:number | 0.482 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | whitespace | 7.5 |  1.  1.  1. |
| back_front:neutral:list_answer:plant | 0.651 | 0.38 | 0.38 | 0.25 | 0.00 | 0.38 | +0.00 | joint_internal:0.2 | other | 14.5 |  The term tree  **Flower  Term\n- |
| back_front:neutral:list_answer:time | 0.635 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | whitespace | 9.8 |  1.  1.  1. |
| back_front:neutral:multiple_choice:container | 0.464 | 0.75 | 0.62 | 0.62 | 0.12 | 0.62 | +0.00 | joint_lm:0.05 | other | 2.0 |  1.  container. Why  container. Explanation |
| back_front:neutral:multiple_choice:number | 0.114 | 0.88 | 1.00 | 1.00 | 0.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  number. Term  number. What  number. \n\n |
| back_front:neutral:multiple_choice:plant | 0.458 | 0.62 | 0.75 | 0.50 | 0.50 | 0.88 | +0.12 | joint_lm:0.05 | other | 2.0 |  plant. Why  plant.\n\nThe  plant. What |
| back_front:neutral:multiple_choice:time | 0.557 | 0.62 | 0.75 | 0.62 | 0.12 | 0.75 | +0.00 | joint_internal:0.2 | other | 6.2 |  container. Explanation  number. Why  container.\n\nExplanation |
| back_front:neutral:quoted_answer:container | 0.460 | 0.00 | 0.12 | 0.00 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 17.0 | word" or word" or bottle" |
| back_front:neutral:quoted_answer:number | 0.480 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 3.9 |  \, \  \, \ .\n\n \ \ |
| back_front:neutral:quoted_answer:plant | 0.503 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | +0.12 | joint_internal:0.05 | other | 9.2 | Word".\n\nOkay flower", " word".\n\nIf |
| back_front:neutral:quoted_answer:time | 0.473 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 6.2 |  \, \ .\n\n \,  \ \ \ |
| back_front:short:answer_one_word:container | 0.616 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | whitespace | 6.5 |   (A  12  'bottle |
| back_front:short:answer_one_word:number | 0.487 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 11.1 |  12   (2  12 |
| back_front:short:answer_one_word:plant | 0.518 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 3.8 |  11  1.   (A |
| back_front:short:answer_one_word:time | 0.546 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 3.9 |   (1   (1   (A |
| back_front:short:label_colon:container | 0.668 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 95.1 | Word, which Word.\n\nBut Geometry.1 |
| back_front:short:label_colon:number | 0.294 | 0.50 | 0.62 | 0.62 | 0.12 | 0.75 | +0.12 | joint_lm:0.05 | other | 1.2 |  Number, and  Number.9  quantity.3 |
| back_front:short:label_colon:plant | 0.272 | 0.25 | 0.38 | 0.25 | 0.12 | 0.62 | +0.25 | joint_lm:0.05 | other | 1.9 | Tree.9 Plant.6 Plant.\n\nBut |
| back_front:short:label_colon:time | 0.669 | 0.00 | 0.12 | 0.00 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 8.8 | Word.\n\nWhat  time.6 Word.\n\n\n |
| back_front:short:list_answer:container | 0.657 | 0.25 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.05 | other | 9.8 |  A.   **A.  A. Question |
| back_front:short:list_answer:number | 0.482 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 16.1 |  The word one  (A)  The word three |
| back_front:short:list_answer:plant | 0.529 | 0.25 | 0.25 | 0.12 | 0.00 | 0.25 | +0.00 | joint_internal:0.2 | other | 4.1 |  **A**  (A)  A\n- |
| back_front:short:list_answer:time | 0.561 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | option_label | 5.5 |  a) a  A.   A. Word |
| back_front:short:multiple_choice:container | 0.633 | 1.00 | 0.75 | 1.00 | 0.88 | 0.75 | +0.00 | joint_lm:0.05 | other | 1.2 |  1.  container\nThe  container.  |
| back_front:short:multiple_choice:number | 0.116 | 0.88 | 0.75 | 1.00 | 0.38 | 0.75 | +0.00 | joint_lm:0.05 | other | 1.1 |  time.\n\nThe  number.\n\n</think>  time.\n\nThe |
| back_front:short:multiple_choice:plant | 0.125 | 0.88 | 0.75 | 0.88 | 0.62 | 1.00 | +0.25 | joint_lm:0.05 | other | 1.0 |  A\n\nThe  plant.\n\n</think>  plant.\n\n</think> |
| back_front:short:multiple_choice:time | 0.628 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  time.\n\nThe  time.\n\nYes  time.\n\nYes |
| back_front:short:quoted_answer:container | 0.498 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | +0.00 | joint_internal:0.2 | other | 5.8 | A" or Answer: " Bottle" |
| back_front:short:quoted_answer:number | 0.462 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | +0.00 | joint_lm:0.05 | other | 5.4 | one" " two" " three" " |
| back_front:short:quoted_answer:plant | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_internal:0.2 | other | 14.9 | The word tree Word" or Answer".\n\nWait |
| back_front:short:quoted_answer:time | 0.380 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | other | 14.2 | Type of Answer Noon" Answer: evening |
| front_back:long:answer_one_word:container | 0.355 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 1.0 | Bam-update BingB Babab |
| front_back:long:answer_one_word:number | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 23.4 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:plant | 0.484 | 0.38 | 0.25 | 0.25 | 0.12 | 0.25 | +0.00 | joint_lm:0.05 | other | 7.6 |  concrete, abstract  " algae "  "bottle |
| front_back:long:answer_one_word:time | 0.167 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 23.9 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:label_colon:container | 0.367 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 37.6 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nC |
| front_back:long:label_colon:number | 0.423 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 1.9 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nWait |
| front_back:long:label_colon:plant | 0.439 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 6.9 |  abstract.\n\nM Algae.\n\n  abstract.\n\nB |
| front_back:long:label_colon:time | 0.173 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 35.1 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nExplanation |
| front_back:long:list_answer:container | 0.391 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 10.9 |  2.  1.   (a |
| front_back:long:list_answer:number | 0.466 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 22.8 |  9 is  1.  1. |
| front_back:long:list_answer:plant | 0.514 | 0.38 | 0.38 | 0.38 | 0.00 | 0.38 | +0.00 | joint_internal:0.2 | other | 19.5 |  moss is a  If the context  If the context |
| front_back:long:list_answer:time | 0.221 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | +0.00 | joint_internal:0.05 | other | 25.0 |  If the context  year is a  If spring is |
| front_back:long:multiple_choice:container | 0.346 | 1.00 | 0.88 | 1.00 | 0.88 | 0.88 | +0.00 | joint_lm:0.05 | other | 1.1 |  container.\n\nBut  container.\n\nBut  container.\n\nI |
| front_back:long:multiple_choice:number | 0.110 | 0.50 | 0.50 | 0.50 | 0.38 | 0.62 | +0.12 | joint_lm:0.05 | other | 1.6 |  number.\n\nBut  number.\n\nBut  container.\n\nOkay |
| front_back:long:multiple_choice:plant | 0.127 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  plant.\n\nOkay  plant.\n\nOkay  plant.\n\nOkay |
| front_back:long:multiple_choice:time | 0.314 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | +0.00 | joint_lm:0.05 | other | 2.2 |  time.\n\nOkay  time.\n\nOkay  container.\n\nWhy |
| front_back:long:quoted_answer:container | 0.416 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 3.8 |  \ \ \  \ \ \  \ \ \ |
| front_back:long:quoted_answer:number | 0.497 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | +0.25 | joint_lm:0.05 | other | 8.5 | number" or number" or dozen" |
| front_back:long:quoted_answer:plant | 0.556 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | +0.00 | joint_lm:0.05 | other | 3.8 | moss" algal"  ".\n\n<think> |
| front_back:long:quoted_answer:time | 0.247 | 0.25 | 0.25 | 0.25 | 0.12 | 0.25 | +0.00 | joint_internal:0.2 | other | 3.4 | month" or year" or spring".\n\nOkay |
| front_back:neutral:answer_one_word:container | 0.361 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 30.6 |  91  9.  chest, heart |
| front_back:neutral:answer_one_word:number | 0.127 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | +0.50 | joint_lm:0.05 | other | 18.1 |  **Question:  **Question:  \n\nThe number |
| front_back:neutral:answer_one_word:plant | 0.512 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 8.4 |  \n\nQuestion:  e.g.,  62 |
| front_back:neutral:answer_one_word:time | 0.115 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 19.6 |  \n\nQuestion:  \n\nQuestion:  'term', |
| front_back:neutral:label_colon:container | 0.572 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 98.9 |  92  91  Geometry. So |
| front_back:neutral:label_colon:number | 0.527 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | whitespace | 42.1 |  1.   (Options  1. |
| front_back:neutral:label_colon:plant | 0.665 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_lm:0.05 | other | 22.1 |  62  Algae.  Geometry, Algebra |
| front_back:neutral:label_colon:time | 0.479 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | whitespace | 23.5 |  9.  9,  1. |
| front_back:neutral:list_answer:container | 0.378 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 20.5 |  1. 1-   61 |
| front_back:neutral:list_answer:number | 0.136 | 0.12 | 0.12 | 0.12 | 0.00 | 0.25 | +0.12 | joint_internal:0.2 | other | 17.0 |  The number of  What is the  Dozen is |
| front_back:neutral:list_answer:plant | 0.588 | 0.25 | 0.12 | 0.12 | 0.00 | 0.25 | +0.12 | joint_internal:0.2 | other | 12.1 |  moss\n-  Algae are  bamboo\n- |
| front_back:neutral:list_answer:time | 0.130 | 0.12 | 0.12 | 0.12 | 0.00 | 0.25 | +0.12 | joint_internal:0.2 | other | 31.0 |  Term\n-  Year:   Spring\n- |
| front_back:neutral:multiple_choice:container | 0.470 | 0.62 | 0.75 | 0.88 | 0.00 | 1.00 | +0.25 | joint_internal:0.2 | other | 2.0 |  time.\n\nWait  container. Explanation  number. Explanation |
| front_back:neutral:multiple_choice:number | 0.408 | 0.88 | 1.00 | 0.88 | 0.25 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  number.\n\nWhat  number. How  number.\n\nA |
| front_back:neutral:multiple_choice:plant | 0.423 | 0.88 | 0.88 | 0.88 | 0.50 | 0.88 | +0.00 | joint_lm:0.05 | other | 1.4 |  plant. I  container. Explanation  plant. Explanation |
| front_back:neutral:multiple_choice:time | 0.515 | 0.75 | 0.25 | 0.62 | 0.50 | 0.88 | +0.62 | joint_internal:0.2 | other | 12.4 | undicularllib ......oneà starttimequal CERT |
| front_back:neutral:quoted_answer:container | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 23.4 | word" or Word".\n\nExample word".\n\nC |
| front_back:neutral:quoted_answer:number | 0.118 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 8.6 | Word" or Word" or word" or |
| front_back:neutral:quoted_answer:plant | 0.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 | joint_lm:0.05 | other | 6.2 | word".\n\nAlright category".\n\nOkay word".\n\nOkay |
| front_back:neutral:quoted_answer:time | 0.126 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 12.8 | month" or word".\n\nExample word".\n\nOkay |
| front_back:short:answer_one_word:container | 0.593 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 5.5 |  12  "case"   (A |
| front_back:short:answer_one_word:number | 0.529 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 239.5 |  <<=否 orders      1. |
| front_back:short:answer_one_word:plant | 0.546 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | whitespace | 4.6 |  11  1.   (Options |
| front_back:short:answer_one_word:time | 0.600 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | whitespace | 7.5 |        \\  1. |
| front_back:short:label_colon:container | 0.264 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 120.2 |  ...? ( Case, which Word.\n\n6 |
| front_back:short:label_colon:number | 0.732 | 0.12 | 0.12 | 0.12 | 0.00 | 0.38 | +0.25 | joint_lm:0.05 | other | 2.5 | Mathematics.  number.9  Mathematics.6 |
| front_back:short:label_colon:plant | 0.282 | 0.25 | 0.38 | 0.12 | 0.12 | 0.50 | +0.12 | joint_lm:0.05 | other | 1.9 | Botanical.\n\n Mathematics. Botany. |
| front_back:short:label_colon:time | 0.696 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 14.1 | Word.6 Word, but Word.\n\nThe |
| front_back:short:list_answer:container | 0.623 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.2 | other | 10.4 |  **A**  Case 1  True\n- |
| front_back:short:list_answer:number | 0.546 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 308.9 | ym<k否   \ sight  1  |
| front_back:short:list_answer:plant | 0.554 | 0.62 | 0.50 | 0.62 | 0.12 | 0.75 | +0.25 | joint_internal:0.2 | other | 1.4 |  a. plant  a. plant  A. plant |
| front_back:short:list_answer:time | 0.629 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_internal:0.05 | whitespace | 3.9 |  (A)  1.  1\n |
| front_back:short:multiple_choice:container | 0.530 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | +0.00 | joint_internal:0.05 | other | 1.0 |  container.\n\n</think>  container.\n\nThe  container.\n\n</think> |
| front_back:short:multiple_choice:number | 0.600 | 0.75 | 0.75 | 0.75 | 0.50 | 0.75 | +0.00 | joint_lm:0.05 | other | 1.8 |  number.\n\n</think>  number.\n\n</think>  number.\n\n</think> |
| front_back:short:multiple_choice:plant | 0.551 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | +0.00 | joint_lm:0.05 | other | 1.0 |  plant.\n\n</think>  plant.\n\n</think>  plant.\n\n</think> |
| front_back:short:multiple_choice:time | 0.569 | 1.00 | 0.25 | 1.00 | 1.00 | 0.25 | +0.00 | joint_internal:0.05 | whitespace | 5.1 |  1.    year  container.  |
| front_back:short:quoted_answer:container | 0.355 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 7.2 |  ".\n\nBut case" or Chest " |
| front_back:short:quoted_answer:number | 0.466 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | joint_lm:0.05 | other | 439.5 |  occupation可以用多少 tt <<= <<= A" or |
| front_back:short:quoted_answer:plant | 0.484 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | +0.00 | joint_internal:0.2 | other | 10.4 | Answer".\n\nThe Answer".\n\nThe Answer".\n\nThe |
| front_back:short:quoted_answer:time | 0.504 | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 | +0.12 | joint_internal:0.05 | other | 132.0 | Month/Area/ Year 1 spring". \ |

