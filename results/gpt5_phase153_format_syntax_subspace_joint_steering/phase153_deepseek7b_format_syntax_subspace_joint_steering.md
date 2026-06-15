# Phase 153 Format-Syntax Subspace Joint Steering: deepseek7b

Generated: 2026-06-15 17:11:19

| case | overlap max | clean | sem | fmt_int | fmt_lm | best_joint | joint | fmt_group | answer_rank | examples |
|---|---|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.391 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | joint_internal:0.2 | other | 1.1 |  In  Babab BABBM |
| back_front:long:answer_one_word:number | 0.414 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 19.8 |  concrete, abstract  "Answer:  concrete, abstract |
| back_front:long:answer_one_word:plant | 0.335 | 0.12 | 0.12 | 0.12 | 0.12 | 0.25 | joint_lm:0.05 | other | 8.8 |  concrete, abstract  flower, flowers  grass, concrete |
| back_front:long:answer_one_word:time | 0.227 | 0.00 | 0.12 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 25.5 |  concrete, abstract  concrete, abstract  \boxed{ |
| back_front:long:label_colon:container | 0.522 | 0.00 | 0.25 | 0.00 | 0.00 | 0.25 | joint_lm:0.05 | other | 23.2 |  [box]  [1]  [Bottle |
| back_front:long:label_colon:number | 0.552 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 3.4 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nWait |
| back_front:long:label_colon:plant | 0.419 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.1 |  abstract.\n\nWait  [ ].\n\n  abstract.\n\nGr |
| back_front:long:label_colon:time | 0.540 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 29.2 |  [1]  [1]  [1] |
| back_front:long:list_answer:container | 0.467 | 0.62 | 0.00 | 0.62 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 4.2 |  1.   (i  1. |
| back_front:long:list_answer:number | 0.496 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 20.4 |  1.  1.  1. |
| back_front:long:list_answer:plant | 0.415 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | joint_lm:0.05 | other | 18.2 |  tree as a  flower\n-  grass is a |
| back_front:long:list_answer:time | 0.228 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 21.9 |  Morning is a  1.  evening\n\n- |
| back_front:long:multiple_choice:container | 0.342 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | joint_internal:0.05 | other | 1.0 |  container.\n\n Why  container.\n\n Why  container.\n\nBut |
| back_front:long:multiple_choice:number | 0.382 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | joint_lm:0.05 | other | 1.0 |  number.\n\nBut  number.\n\nBut  number.\n\nBut |
| back_front:long:multiple_choice:plant | 0.321 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | joint_lm:0.05 | other | 1.0 |  container.\n\nBut  plant.\n\nOkay  plant.\n\nOkay |
| back_front:long:multiple_choice:time | 0.353 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | joint_lm:0.05 | other | 1.0 |  time.\n\nBut  time.\n\nBut  time.\n\nOkay |
| back_front:long:quoted_answer:container | 0.489 | 0.62 | 0.50 | 0.62 | 0.38 | 0.50 | joint_internal:0.05 | other | 3.0 | box" or bag" or bottle", |
| back_front:long:quoted_answer:number | 0.557 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 5.8 |  ".\n\n.\n\n  ".\n\n.\n\n abstract" or |
| back_front:long:quoted_answer:plant | 0.443 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | joint_lm:0.05 | other | 4.2 | tree" or flower" or grass" is |
| back_front:long:quoted_answer:time | 0.270 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.05 | other | 5.9 | morning" concrete" evening" |
| back_front:neutral:answer_one_word:container | 0.492 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 13.6 |  9,  9,  9, |
| back_front:neutral:answer_one_word:number | 0.434 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 9.8 |  1.  1.  21 |
| back_front:neutral:answer_one_word:plant | 0.567 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 13.1 |  **Question:  flower, flower  \n\nQuestion: |
| back_front:neutral:answer_one_word:time | 0.563 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 8.6 |  (e.g  1.  1. |
| back_front:neutral:label_colon:container | 0.413 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 47.4 |  Geometry.   92  Geometry.  |
| back_front:neutral:label_colon:number | 0.520 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | list_marker | 30.6 | 1.1 1.   6. |
| back_front:neutral:label_colon:plant | 0.594 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 13.5 |  62  Geometry, Geometry  62 |
| back_front:neutral:label_colon:time | 0.133 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 75.5 |  1. 1.1  1. |
| back_front:neutral:list_answer:container | 0.563 | 0.38 | 0.00 | 0.25 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 10.8 |  1.  1.  1. |
| back_front:neutral:list_answer:number | 0.482 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | joint_internal:0.05 | whitespace | 7.5 |  1.  1.  1. |
| back_front:neutral:list_answer:plant | 0.651 | 0.38 | 0.38 | 0.25 | 0.00 | 0.38 | joint_internal:0.2 | other | 14.5 |  The term tree  **Flower  Term\n- |
| back_front:neutral:list_answer:time | 0.635 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | joint_internal:0.05 | whitespace | 9.8 |  1.  1.  1. |
| back_front:neutral:multiple_choice:container | 0.464 | 0.75 | 0.62 | 0.62 | 0.12 | 0.62 | joint_lm:0.05 | other | 2.0 |  1.  container. Why  container. Explanation |
| back_front:neutral:multiple_choice:number | 0.114 | 0.88 | 1.00 | 1.00 | 0.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  number. Term  number. What  number. \n\n |
| back_front:neutral:multiple_choice:plant | 0.458 | 0.62 | 0.75 | 0.50 | 0.50 | 0.88 | joint_lm:0.05 | other | 2.0 |  plant. Why  plant.\n\nThe  plant. What |
| back_front:neutral:multiple_choice:time | 0.557 | 0.62 | 0.75 | 0.62 | 0.12 | 0.75 | joint_internal:0.2 | other | 6.2 |  container. Explanation  number. Why  container.\n\nExplanation |
| back_front:neutral:quoted_answer:container | 0.460 | 0.00 | 0.12 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 17.0 | word" or word" or bottle" |
| back_front:neutral:quoted_answer:number | 0.480 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 3.9 |  \, \  \, \ .\n\n \ \ |
| back_front:neutral:quoted_answer:plant | 0.503 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | joint_internal:0.05 | other | 9.2 | Word".\n\nOkay flower", " word".\n\nIf |
| back_front:neutral:quoted_answer:time | 0.473 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 6.2 |  \, \ .\n\n \,  \ \ \ |
| back_front:short:answer_one_word:container | 0.616 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.2 | whitespace | 6.5 |   (A  12  'bottle |
| back_front:short:answer_one_word:number | 0.487 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 11.1 |  12   (2  12 |
| back_front:short:answer_one_word:plant | 0.518 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 3.8 |  11  1.   (A |
| back_front:short:answer_one_word:time | 0.546 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 3.9 |   (1   (1   (A |
| back_front:short:label_colon:container | 0.668 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 95.1 | Word, which Word.\n\nBut Geometry.1 |
| back_front:short:label_colon:number | 0.294 | 0.50 | 0.62 | 0.62 | 0.12 | 0.75 | joint_lm:0.05 | other | 1.2 |  Number, and  Number.9  quantity.3 |
| back_front:short:label_colon:plant | 0.272 | 0.25 | 0.38 | 0.25 | 0.12 | 0.62 | joint_lm:0.05 | other | 1.9 | Tree.9 Plant.6 Plant.\n\nBut |
| back_front:short:label_colon:time | 0.669 | 0.00 | 0.12 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 8.8 | Word.\n\nWhat  time.6 Word.\n\n\n |
| back_front:short:list_answer:container | 0.657 | 0.25 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.05 | other | 9.8 |  A.   **A.  A. Question |
| back_front:short:list_answer:number | 0.482 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.2 | other | 16.1 |  The word one  (A)  The word three |
| back_front:short:list_answer:plant | 0.529 | 0.25 | 0.25 | 0.12 | 0.00 | 0.25 | joint_internal:0.2 | other | 4.1 |  **A**  (A)  A\n- |
| back_front:short:list_answer:time | 0.561 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | option_label | 5.5 |  a) a  A.   A. Word |
| back_front:short:multiple_choice:container | 0.633 | 1.00 | 0.75 | 1.00 | 0.88 | 0.75 | joint_lm:0.05 | other | 1.2 |  1.  container\nThe  container.  |
| back_front:short:multiple_choice:number | 0.116 | 0.88 | 0.75 | 1.00 | 0.38 | 0.75 | joint_lm:0.05 | other | 1.1 |  time.\n\nThe  number.\n\n</think>  time.\n\nThe |
| back_front:short:multiple_choice:plant | 0.125 | 0.88 | 0.75 | 0.88 | 0.62 | 1.00 | joint_lm:0.05 | other | 1.0 |  A\n\nThe  plant.\n\n</think>  plant.\n\n</think> |
| back_front:short:multiple_choice:time | 0.628 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_lm:0.05 | other | 1.0 |  time.\n\nThe  time.\n\nYes  time.\n\nYes |
| back_front:short:quoted_answer:container | 0.498 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | joint_internal:0.2 | other | 5.8 | A" or Answer: " Bottle" |
| back_front:short:quoted_answer:number | 0.462 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | joint_lm:0.05 | other | 5.4 | one" " two" " three" " |
| back_front:short:quoted_answer:plant | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_internal:0.2 | other | 14.9 | The word tree Word" or Answer".\n\nWait |
| back_front:short:quoted_answer:time | 0.380 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.05 | other | 14.2 | Type of Answer Noon" Answer: evening |
| front_back:long:answer_one_word:container | 0.355 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | joint_internal:0.2 | other | 1.0 | Bam-update BingB Babab |
| front_back:long:answer_one_word:number | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 23.4 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:answer_one_word:plant | 0.484 | 0.38 | 0.25 | 0.25 | 0.12 | 0.25 | joint_lm:0.05 | other | 7.6 |  concrete, abstract  " algae "  "bottle |
| front_back:long:answer_one_word:time | 0.167 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 23.9 |  concrete, abstract  concrete, abstract  concrete, abstract |
| front_back:long:label_colon:container | 0.367 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 37.6 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nC |
| front_back:long:label_colon:number | 0.423 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 1.9 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nWait |
| front_back:long:label_colon:plant | 0.439 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 6.9 |  abstract.\n\nM Algae.\n\n  abstract.\n\nB |
| front_back:long:label_colon:time | 0.173 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 35.1 |  abstract.\n\nWait  abstract.\n\nWait  abstract.\n\nExplanation |
| front_back:long:list_answer:container | 0.391 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 10.9 |  2.  1.   (a |
| front_back:long:list_answer:number | 0.466 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 22.8 |  9 is  1.  1. |
| front_back:long:list_answer:plant | 0.514 | 0.38 | 0.38 | 0.38 | 0.00 | 0.38 | joint_internal:0.2 | other | 19.5 |  moss is a  If the context  If the context |
| front_back:long:list_answer:time | 0.221 | 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | joint_internal:0.05 | other | 25.0 |  If the context  year is a  If spring is |
| front_back:long:multiple_choice:container | 0.346 | 1.00 | 0.88 | 1.00 | 0.88 | 0.88 | joint_lm:0.05 | other | 1.1 |  container.\n\nBut  container.\n\nBut  container.\n\nI |
| front_back:long:multiple_choice:number | 0.110 | 0.50 | 0.50 | 0.50 | 0.38 | 0.62 | joint_lm:0.05 | other | 1.6 |  number.\n\nBut  number.\n\nBut  container.\n\nOkay |
| front_back:long:multiple_choice:plant | 0.127 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | joint_internal:0.05 | other | 1.0 |  plant.\n\nOkay  plant.\n\nOkay  plant.\n\nOkay |
| front_back:long:multiple_choice:time | 0.314 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | joint_lm:0.05 | other | 2.2 |  time.\n\nOkay  time.\n\nOkay  container.\n\nWhy |
| front_back:long:quoted_answer:container | 0.416 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 3.8 |  \ \ \  \ \ \  \ \ \ |
| front_back:long:quoted_answer:number | 0.497 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | joint_lm:0.05 | other | 8.5 | number" or number" or dozen" |
| front_back:long:quoted_answer:plant | 0.556 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | joint_lm:0.05 | other | 3.8 | moss" algal"  ".\n\n<think> |
| front_back:long:quoted_answer:time | 0.247 | 0.25 | 0.25 | 0.25 | 0.12 | 0.25 | joint_internal:0.2 | other | 3.4 | month" or year" or spring".\n\nOkay |
| front_back:neutral:answer_one_word:container | 0.361 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 30.6 |  91  9.  chest, heart |
| front_back:neutral:answer_one_word:number | 0.127 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | joint_lm:0.05 | other | 18.1 |  **Question:  **Question:  \n\nThe number |
| front_back:neutral:answer_one_word:plant | 0.512 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 8.4 |  \n\nQuestion:  e.g.,  62 |
| front_back:neutral:answer_one_word:time | 0.115 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 19.6 |  \n\nQuestion:  \n\nQuestion:  'term', |
| front_back:neutral:label_colon:container | 0.572 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 98.9 |  92  91  Geometry. So |
| front_back:neutral:label_colon:number | 0.527 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.05 | whitespace | 42.1 |  1.   (Options  1. |
| front_back:neutral:label_colon:plant | 0.665 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_lm:0.05 | other | 22.1 |  62  Algae.  Geometry, Algebra |
| front_back:neutral:label_colon:time | 0.479 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.05 | whitespace | 23.5 |  9.  9,  1. |
| front_back:neutral:list_answer:container | 0.378 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 20.5 |  1. 1-   61 |
| front_back:neutral:list_answer:number | 0.136 | 0.12 | 0.12 | 0.12 | 0.00 | 0.25 | joint_internal:0.2 | other | 17.0 |  The number of  What is the  Dozen is |
| front_back:neutral:list_answer:plant | 0.588 | 0.25 | 0.12 | 0.12 | 0.00 | 0.25 | joint_internal:0.2 | other | 12.1 |  moss\n-  Algae are  bamboo\n- |
| front_back:neutral:list_answer:time | 0.130 | 0.12 | 0.12 | 0.12 | 0.00 | 0.25 | joint_internal:0.2 | other | 31.0 |  Term\n-  Year:   Spring\n- |
| front_back:neutral:multiple_choice:container | 0.470 | 0.62 | 0.75 | 0.88 | 0.00 | 1.00 | joint_internal:0.2 | other | 2.0 |  time.\n\nWait  container. Explanation  number. Explanation |
| front_back:neutral:multiple_choice:number | 0.408 | 0.88 | 1.00 | 0.88 | 0.25 | 1.00 | joint_lm:0.05 | other | 1.0 |  number.\n\nWhat  number. How  number.\n\nA |
| front_back:neutral:multiple_choice:plant | 0.423 | 0.88 | 0.88 | 0.88 | 0.50 | 0.88 | joint_lm:0.05 | other | 1.4 |  plant. I  container. Explanation  plant. Explanation |
| front_back:neutral:multiple_choice:time | 0.515 | 0.75 | 0.25 | 0.62 | 0.50 | 0.88 | joint_internal:0.2 | other | 12.4 | undicularllib ......oneà starttimequal CERT |
| front_back:neutral:quoted_answer:container | 0.372 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 23.4 | word" or Word".\n\nExample word".\n\nC |
| front_back:neutral:quoted_answer:number | 0.118 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 8.6 | Word" or Word" or word" or |
| front_back:neutral:quoted_answer:plant | 0.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | joint_lm:0.05 | other | 6.2 | word".\n\nAlright category".\n\nOkay word".\n\nOkay |
| front_back:neutral:quoted_answer:time | 0.126 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 12.8 | month" or word".\n\nExample word".\n\nOkay |
| front_back:short:answer_one_word:container | 0.593 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 5.5 |  12  "case"   (A |
| front_back:short:answer_one_word:number | 0.529 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 239.5 |  <<=否 orders      1. |
| front_back:short:answer_one_word:plant | 0.546 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | whitespace | 4.6 |  11  1.   (Options |
| front_back:short:answer_one_word:time | 0.600 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | joint_lm:0.05 | whitespace | 7.5 |        \\  1. |
| front_back:short:label_colon:container | 0.264 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 120.2 |  ...? ( Case, which Word.\n\n6 |
| front_back:short:label_colon:number | 0.732 | 0.12 | 0.12 | 0.12 | 0.00 | 0.38 | joint_lm:0.05 | other | 2.5 | Mathematics.  number.9  Mathematics.6 |
| front_back:short:label_colon:plant | 0.282 | 0.25 | 0.38 | 0.12 | 0.12 | 0.50 | joint_lm:0.05 | other | 1.9 | Botanical.\n\n Mathematics. Botany. |
| front_back:short:label_colon:time | 0.696 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 14.1 | Word.6 Word, but Word.\n\nThe |
| front_back:short:list_answer:container | 0.623 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.2 | other | 10.4 |  **A**  Case 1  True\n- |
| front_back:short:list_answer:number | 0.546 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 308.9 | ym<k否   \ sight  1  |
| front_back:short:list_answer:plant | 0.554 | 0.62 | 0.50 | 0.62 | 0.12 | 0.75 | joint_internal:0.2 | other | 1.4 |  a. plant  a. plant  A. plant |
| front_back:short:list_answer:time | 0.629 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_internal:0.05 | whitespace | 3.9 |  (A)  1.  1\n |
| front_back:short:multiple_choice:container | 0.530 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | joint_internal:0.05 | other | 1.0 |  container.\n\n</think>  container.\n\nThe  container.\n\n</think> |
| front_back:short:multiple_choice:number | 0.600 | 0.75 | 0.75 | 0.75 | 0.50 | 0.75 | joint_lm:0.05 | other | 1.8 |  number.\n\n</think>  number.\n\n</think>  number.\n\n</think> |
| front_back:short:multiple_choice:plant | 0.551 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | joint_lm:0.05 | other | 1.0 |  plant.\n\n</think>  plant.\n\n</think>  plant.\n\n</think> |
| front_back:short:multiple_choice:time | 0.569 | 1.00 | 0.25 | 1.00 | 1.00 | 0.25 | joint_internal:0.05 | whitespace | 5.1 |  1.    year  container.  |
| front_back:short:quoted_answer:container | 0.355 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 7.2 |  ".\n\nBut case" or Chest " |
| front_back:short:quoted_answer:number | 0.466 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | joint_lm:0.05 | other | 439.5 |  occupation可以用多少 tt <<= <<= A" or |
| front_back:short:quoted_answer:plant | 0.484 | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | joint_internal:0.2 | other | 10.4 | Answer".\n\nThe Answer".\n\nThe Answer".\n\nThe |
| front_back:short:quoted_answer:time | 0.504 | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 | joint_internal:0.05 | other | 132.0 | Month/Area/ Year 1 spring". \ |
