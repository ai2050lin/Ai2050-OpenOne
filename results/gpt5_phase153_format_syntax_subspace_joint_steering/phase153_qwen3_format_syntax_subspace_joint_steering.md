# Phase 153 Format-Syntax Subspace Joint Steering: qwen3

Generated: 2026-06-15 17:24:44

| case | overlap max | clean | sem | fmt_int | fmt_lm | best_joint | joint | fmt_group | answer_rank | examples |
|---|---|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.2 |  1.  concrete, abstract  1. |
| back_front:long:answer_one_word:number | 0.239 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.0 |  concrete, abstract  concrete, abstract  concrete, abstract |
| back_front:long:answer_one_word:plant | 0.534 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 4.8 |  concrete or abstract  concrete or abstract  concrete or abstract |
| back_front:long:answer_one_word:time | 0.265 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.8 |  concrete, abstract  concrete, abstract  concrete or abstract |
| back_front:long:label_colon:container | 0.223 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.0 |  [concrete  [concrete  [concrete |
| back_front:long:label_colon:number | 0.523 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.5 |  [category name  [the category  [category name |
| back_front:long:label_colon:plant | 0.568 | 0.38 | 0.12 | 0.38 | 0.12 | 0.25 | joint_lm:0.05 | other | 4.5 |  Abstract, if  Concrete. The  Concrete. The |
| back_front:long:label_colon:time | 0.583 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 16.8 |  Abstract, because  concrete, because  Abstract, because |
| back_front:long:list_answer:container | 0.143 | 0.50 | 0.38 | 0.50 | 0.38 | 0.50 | joint_lm:0.05 | quote | 9.0 |  "box"  "concrete  "bottle |
| back_front:long:list_answer:number | 0.125 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | quote | 8.8 |  "Concrete"  "Concrete"  "Concrete" |
| back_front:long:list_answer:plant | 0.471 | 0.25 | 0.00 | 0.25 | 0.25 | 0.00 | joint_internal:0.2 | other | 1.5 |  a or an  a or an  concrete object that |
| back_front:long:list_answer:time | 0.122 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | joint_lm:0.05 | quote | 12.1 |  "Morning"  "noon"  "Evening |
| back_front:long:multiple_choice:container | 0.123 | 1.00 | 0.88 | 1.00 | 0.88 | 0.88 | joint_internal:0.05 | other | 1.1 |  container. \n\n  container. \n\n  container. \n\n |
| back_front:long:multiple_choice:number | 0.406 | 1.00 | 0.62 | 1.00 | 0.75 | 1.00 | joint_lm:0.05 | other | 1.6 |  (A)  (A)  A. A |
| back_front:long:multiple_choice:plant | 0.472 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.2 |  container. The  plant. The  plant. The |
| back_front:long:multiple_choice:time | 0.480 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  time. The  time. The  time. The |
| back_front:long:quoted_answer:container | 0.238 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 12.5 | category" ( category" ( X" ( |
| back_front:long:quoted_answer:number | 0.226 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 12.8 | category" ( category" ( category" ( |
| back_front:long:quoted_answer:plant | 0.661 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 7.2 | ..." or " ..." or " ..." or " |
| back_front:long:quoted_answer:time | 0.301 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 14.2 | category" ( category" ( category" ( |
| back_front:neutral:answer_one_word:container | 0.420 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 4.6 |  1.  1.  1. |
| back_front:neutral:answer_one_word:number | 0.510 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 1.0 |  a a a  a a a  a a a |
| back_front:neutral:answer_one_word:plant | 0.516 | 0.12 | 0.50 | 0.12 | 0.00 | 0.38 | joint_lm:0.05 | other | 1.8 |  a. a   plant,  1. |
| back_front:neutral:answer_one_word:time | 0.510 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 4.9 |  1.  1.  1) |
| back_front:neutral:label_colon:container | 0.449 | 0.25 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.05 | other | 2.6 |  a group of  a group of  1. |
| back_front:neutral:label_colon:number | 0.507 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | quote | 3.0 |  the same or  the same or  "  " |
| back_front:neutral:label_colon:plant | 0.493 | 0.62 | 0.12 | 0.62 | 0.00 | 0.75 | joint_lm:0.05 | other | 1.2 |  1.  plant. Flower  plant. What |
| back_front:neutral:label_colon:time | 0.484 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | joint_internal:0.2 | other | 1.5 |  "Morning routine  time of day  "Evening |
| back_front:neutral:list_answer:container | 0.463 | 0.50 | 0.62 | 0.50 | 0.00 | 0.62 | joint_internal:0.2 | option_label | 5.8 |  A term box  A term bag  A term bottle |
| back_front:neutral:list_answer:number | 0.570 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.05 | other | 1.0 |  a. A  a. A  a. A |
| back_front:neutral:list_answer:plant | 0.579 | 0.25 | 0.25 | 0.25 | 0.12 | 0.12 | joint_internal:0.2 | other | 1.0 |  a a a  a flower is  a type of |
| back_front:neutral:list_answer:time | 0.575 | 0.38 | 0.38 | 0.38 | 0.00 | 0.25 | joint_internal:0.05 | other | 6.8 |  Morning is the  noon is the  A time of |
| back_front:neutral:multiple_choice:container | 0.159 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  container. Term  container. Why  container. Why |
| back_front:neutral:multiple_choice:number | 0.378 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  number. Term  number. Why  number. Why |
| back_front:neutral:multiple_choice:plant | 0.403 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  plant. Why  plant. Why  plant. Why |
| back_front:neutral:multiple_choice:time | 0.378 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | joint_lm:0.05 | other | 1.2 |  time. Why  time. Why  time. Why |
| back_front:neutral:quoted_answer:container | 0.531 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.8 |  " " " ____" ( ____" " |
| back_front:neutral:quoted_answer:number | 0.637 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 7.5 | ____" for ____" for ____" for |
| back_front:neutral:quoted_answer:plant | 0.644 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | joint_lm:0.05 | other | 6.9 | ____" and Flower"  ".\n\n" |
| back_front:neutral:quoted_answer:time | 0.639 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | joint_lm:0.05 | other | 13.6 | Morning" is ____" ( evening" |
| back_front:short:answer_one_word:container | 0.830 | 0.50 | 0.50 | 0.62 | 0.00 | 0.50 | joint_internal:0.05 | other | 1.0 |  a. a  a. bag  bottle, glass |
| back_front:short:answer_one_word:number | 0.815 | 0.00 | 0.62 | 0.00 | 0.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  number, object  number, unit  number, color |
| back_front:short:answer_one_word:plant | 0.246 | 0.88 | 0.88 | 0.88 | 0.75 | 0.88 | joint_internal:0.05 | other | 1.1 |  1.  plant, animal  plant, animal |
| back_front:short:answer_one_word:time | 0.842 | 0.25 | 0.88 | 0.38 | 0.00 | 0.88 | joint_internal:0.05 | other | 1.6 |  morning is a  a. time  the time of |
| back_front:short:label_colon:container | 0.778 | 0.75 | 0.12 | 0.75 | 0.38 | 0.25 | joint_lm:0.05 | other | 2.4 |  a box,  1. Container. A |
| back_front:short:label_colon:number | 0.809 | 0.50 | 0.75 | 0.62 | 0.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  number, and  number, and  number, and |
| back_front:short:label_colon:plant | 0.301 | 0.75 | 0.75 | 0.75 | 0.62 | 0.88 | joint_lm:0.05 | other | 1.4 | Data structure. Plant. The Plant. The |
| back_front:short:label_colon:time | 0.804 | 0.62 | 0.62 | 0.75 | 0.38 | 0.75 | joint_lm:0.05 | other | 1.1 | time, and   (a time, and |
| back_front:short:list_answer:container | 0.798 | 0.75 | 0.25 | 0.75 | 0.00 | 0.25 | joint_internal:0.2 | other | 1.0 |  a box of  a a a  a\n  |
| back_front:short:list_answer:number | 0.791 | 0.25 | 0.00 | 0.25 | 0.00 | 0.25 | joint_lm:0.05 | other | 1.6 |  a. a  a. number  1. |
| back_front:short:list_answer:plant | 0.220 | 0.62 | 0.62 | 0.62 | 0.00 | 1.00 | joint_lm:0.05 | option_label | 1.6 |  A. tree  A. plant  plant\n- |
| back_front:short:list_answer:time | 0.811 | 0.50 | 0.50 | 0.50 | 0.00 | 0.62 | joint_lm:0.05 | other | 1.0 |  a. time  a. a  a. time |
| back_front:short:multiple_choice:container | 0.539 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  container. The  container. The  container. The |
| back_front:short:multiple_choice:number | 0.548 | 1.00 | 0.25 | 1.00 | 1.00 | 0.75 | joint_lm:0.05 | other | 1.4 |  number. The  number. The  number. The |
| back_front:short:multiple_choice:plant | 0.543 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. The  plant. The |
| back_front:short:multiple_choice:time | 0.542 | 1.00 | 0.75 | 1.00 | 0.88 | 1.00 | joint_lm:0.05 | other | 1.1 |  time. The  time. The  time. The |
| back_front:short:quoted_answer:container | 0.799 | 0.75 | 0.62 | 0.75 | 0.62 | 0.75 | joint_lm:0.05 | other | 1.2 | box" is bag" is container" or |
| back_front:short:quoted_answer:number | 0.790 | 0.38 | 0.00 | 0.50 | 0.38 | 0.50 | joint_lm:0.05 | other | 1.5 | ______" number" or number" or |
| back_front:short:quoted_answer:plant | 0.267 | 0.88 | 0.75 | 0.88 | 0.50 | 1.00 | joint_lm:0.05 | other | 1.0 | tree" is plant" or plant" or |
| back_front:short:quoted_answer:time | 0.812 | 0.12 | 0.12 | 0.12 | 0.12 | 0.88 | joint_lm:0.05 | other | 1.4 | time of day time" or time of day |
| front_back:long:answer_one_word:container | 0.154 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 9.0 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:number | 0.215 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.5 |  concrete, abstract  concrete, abstract  concrete or abstract |
| front_back:long:answer_one_word:plant | 0.185 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 5.2 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:time | 0.142 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.2 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:label_colon:container | 0.196 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.2 |  [concrete  [category]  [concrete |
| front_back:long:label_colon:number | 0.580 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 6.6 |  [category name  [category name  Concrete. The |
| front_back:long:label_colon:plant | 0.602 | 0.25 | 0.00 | 0.25 | 0.12 | 0.00 | joint_lm:0.05 | other | 4.1 |  Concrete. The  concrete. The  Concrete. The |
| front_back:long:label_colon:time | 0.614 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 16.1 |  concrete. The  Concrete. The  [concrete |
| front_back:long:list_answer:container | 0.130 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | quote | 9.4 |  "barrel  1.  "Chest |
| front_back:long:list_answer:number | 0.144 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | quote | 10.5 |  "concrete  "concrete  "dozen |
| front_back:long:list_answer:plant | 0.134 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 | joint_lm:0.05 | quote | 9.0 |  "Moss  "Algae  "Bam |
| front_back:long:list_answer:time | 0.143 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | joint_lm:0.05 | quote | 11.0 |  "month"  If the context  "Spring" |
| front_back:long:multiple_choice:container | 0.122 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | joint_internal:0.05 | other | 1.0 |  container. \n\n  container. \n\n  container. The |
| front_back:long:multiple_choice:number | 0.438 | 0.88 | 1.00 | 0.88 | 0.88 | 1.00 | joint_lm:0.05 | other | 1.4 |  number. The  number. The  number. The |
| front_back:long:multiple_choice:plant | 0.484 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. But  plant. The |
| front_back:long:multiple_choice:time | 0.435 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  time. The  time. The  time. The |
| front_back:long:quoted_answer:container | 0.188 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 12.4 | category" ( category" ( category" ( |
| front_back:long:quoted_answer:number | 0.192 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.8 | category" ( category" ( category" ( |
| front_back:long:quoted_answer:plant | 0.216 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 11.1 | category" ( ..." (e X" ( |
| front_back:long:quoted_answer:time | 0.174 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 14.2 | category" ( category" ( category" ( |
| front_back:neutral:answer_one_word:container | 0.436 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 1.2 |  a. a  a a a   "chest |
| front_back:neutral:answer_one_word:number | 0.486 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | quote | 3.2 |  1.  "dog,  dozen, semantic |
| front_back:neutral:answer_one_word:plant | 0.519 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_internal:0.2 | other | 1.2 |  a. moss  a type of   bamboo, |
| front_back:neutral:answer_one_word:time | 0.473 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | joint_internal:0.05 | quote | 3.2 |  the following are  "term year  1. |
| front_back:neutral:label_colon:container | 0.215 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 17.4 |  1.  1.  1. |
| front_back:neutral:label_colon:number | 0.481 | 0.00 | 0.12 | 0.00 | 0.00 | 0.25 | joint_lm:0.05 | whitespace | 4.4 |  1.  1.  Number. A |
| front_back:neutral:label_colon:plant | 0.489 | 0.25 | 0.38 | 0.38 | 0.12 | 0.50 | joint_lm:0.05 | other | 1.6 |  plant. Sub  Algae.  1. |
| front_back:neutral:label_colon:time | 0.497 | 0.25 | 0.12 | 0.25 | 0.00 | 0.25 | joint_lm:0.05 | whitespace | 2.8 |  1.  1.  1. |
| front_back:neutral:list_answer:container | 0.493 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_internal:0.2 | other | 1.0 |  a hollow body  a a a  a body cavity |
| front_back:neutral:list_answer:number | 0.549 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | option_label | 2.4 |  A group of  The first and  A dozen is |
| front_back:neutral:list_answer:plant | 0.587 | 0.38 | 0.25 | 0.38 | 0.00 | 0.38 | joint_internal:0.2 | other | 1.5 |  Moss is a  a type of  a type of |
| front_back:neutral:list_answer:time | 0.540 | 0.25 | 0.25 | 0.25 | 0.00 | 0.38 | joint_internal:0.2 | other | 1.2 |  a a a  a a a  a time of |
| front_back:neutral:multiple_choice:container | 0.310 | 1.00 | 0.38 | 1.00 | 1.00 | 1.00 | joint_internal:0.2 | other | 1.2 |  container. I  plant. \n\n  container. I |
| front_back:neutral:multiple_choice:number | 0.382 | 0.88 | 0.75 | 0.88 | 0.88 | 0.88 | joint_lm:0.05 | other | 1.2 |  number. Why  number. Why  number. The |
| front_back:neutral:multiple_choice:plant | 0.363 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  plant. Why  plant. Why  plant. Why |
| front_back:neutral:multiple_choice:time | 0.405 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  time. \n\n  time. Why  time. Why |
| front_back:neutral:quoted_answer:container | 0.560 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 10.4 | ____"  ____" for ____" " |
| front_back:neutral:quoted_answer:number | 0.596 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 18.0 |  " " "  " " " ____" The |
| front_back:neutral:quoted_answer:plant | 0.648 | 0.38 | 0.38 | 0.38 | 0.25 | 0.38 | joint_lm:0.05 | other | 6.9 | moss" algae" bamboo" |
| front_back:neutral:quoted_answer:time | 0.600 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | quote | 19.0 |  ".\n\n" ____" and  ".\n\n" |
| front_back:short:answer_one_word:container | 0.822 | 0.12 | 0.00 | 0.12 | 0.00 | 0.38 | joint_internal:0.2 | other | 1.4 |  volume, weight  a. a  cabinet, drawer |
| front_back:short:answer_one_word:number | 0.805 | 0.25 | 0.38 | 0.25 | 0.00 | 0.75 | joint_lm:0.05 | other | 1.1 |  a. number  number, object  number, object |
| front_back:short:answer_one_word:plant | 0.821 | 1.00 | 0.75 | 1.00 | 1.00 | 0.88 | joint_lm:0.05 | other | 1.2 |  plant, animal  plant, fungus  plant, animal |
| front_back:short:answer_one_word:time | 0.836 | 0.12 | 0.50 | 0.12 | 0.12 | 0.50 | joint_internal:0.2 | other | 1.5 |  a time period  a time period  a a a |
| front_back:short:label_colon:container | 0.750 | 0.25 | 0.12 | 0.25 | 0.00 | 0.25 | joint_lm:0.05 | other | 7.1 | 1:1  a class of  Furniture. The |
| front_back:short:label_colon:number | 0.776 | 0.38 | 0.62 | 0.50 | 0.00 | 0.88 | joint_lm:0.05 | other | 1.4 |  number, and  number, and  quantity, and |
| front_back:short:label_colon:plant | 0.758 | 0.88 | 0.88 | 0.88 | 0.75 | 0.88 | joint_lm:0.05 | other | 1.1 |  plant. What ____. (  plant. The |
| front_back:short:label_colon:time | 0.752 | 0.50 | 0.62 | 0.50 | 0.50 | 0.75 | joint_lm:0.05 | other | 2.6 | time. A Time. A   (a |
| front_back:short:list_answer:container | 0.786 | 0.25 | 0.12 | 0.25 | 0.00 | 0.12 | joint_internal:0.2 | other | 1.0 |  a a a  a a a  a airt |
| front_back:short:list_answer:number | 0.773 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 1.0 |  a. a  a. a  a. number |
| front_back:short:list_answer:plant | 0.783 | 0.88 | 0.38 | 0.75 | 0.00 | 0.62 | joint_lm:0.05 | other | 1.2 |  a. plant  plant\n-  plant\n- |
| front_back:short:list_answer:time | 0.800 | 0.38 | 0.00 | 0.38 | 0.00 | 0.38 | joint_lm:0.05 | other | 1.0 |  a. time  a. a  a. a |
| front_back:short:multiple_choice:container | 0.489 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  container. The  container. The  container. The |
| front_back:short:multiple_choice:number | 0.477 | 0.88 | 0.50 | 0.88 | 0.88 | 0.75 | joint_lm:0.05 | other | 1.8 |  number. The  number. The  number. The |
| front_back:short:multiple_choice:plant | 0.511 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  plant. The  plant. The  plant. The |
| front_back:short:multiple_choice:time | 0.481 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.2 |  time. The  time. The  plant. The |
| front_back:short:quoted_answer:container | 0.800 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 5.0 | barrel"  ".\n\nThe furniture" |
| front_back:short:quoted_answer:number | 0.785 | 0.25 | 0.00 | 0.50 | 0.12 | 0.62 | joint_lm:0.05 | other | 2.8 | number" or number" or number" or |
| front_back:short:quoted_answer:plant | 0.800 | 0.50 | 0.62 | 0.50 | 0.50 | 0.50 | joint_internal:0.2 | other | 3.0 | algae", algae"\n\n grass"\n\nWait |
| front_back:short:quoted_answer:time | 0.816 | 0.38 | 0.50 | 0.38 | 0.50 | 0.50 | joint_lm:0.05 | other | 4.4 | time period" time period" ______" |
