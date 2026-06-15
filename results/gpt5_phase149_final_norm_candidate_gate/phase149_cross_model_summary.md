# Phase 149 Cross-model Final-Norm Candidate Gate Summary

## qwen3

### By category

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 0.00 | 1.00 | 1.00 | +3.88 | 32677.6 | 0.00 | 1.00 | 1.00 | 32677.6 |
| time | 2 | 0.00 | 0.56 | 1.44 | +0.05 | 43163.2 | 0.00 | 0.00 | 0.56 | 43163.2 |

### By format

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.00 | 0.78 | 1.22 | +1.96 | 37920.4 | 0.00 | 0.50 | 0.78 | 37920.4 |

### By family

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 0.00 | 1.00 | 1.00 | +1.95 | 65370.7 | 0.00 | 0.50 | 1.00 | 65370.7 |
| neutral | 2 | 0.00 | 0.56 | 1.44 | +1.97 | 10470.1 | 0.00 | 0.50 | 0.56 | 10470.1 |

### Cases

| case | variant | cand_arg | cand_rank | full_rank | full_arg | lens_in/out | top |
|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 64209.4 | 0.00 | 1.00/1.00 | .fhir إنش 改革委 |
| front_back:long:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 66532.0 | 0.00 | 0.00/1.00 | 改革委 إنش .fhir |
| front_back:neutral:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 1145.8 | 0.00 | 1.00/1.00 | 改革委 إنش 有网友 |
| front_back:neutral:label_colon:time | final_norm_output_lm:4.0:0.0 | 0.12 | 1.88 | 19794.4 | 0.00 | 0.00/0.12 | 改革委 PRECATED  overposting |

## glm4

### By category

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 0.00 | 0.94 | 1.06 | +3.45 | 84.2 | 0.00 | 1.00 | 0.94 | 84.2 |
| time | 2 | 0.00 | 1.00 | 1.00 | +5.36 | 97.3 | 0.00 | 0.94 | 1.00 | 97.3 |

### By format

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.00 | 0.97 | 1.03 | +4.40 | 90.8 | 0.00 | 0.97 | 0.97 | 90.8 |

### By family

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 0.00 | 0.94 | 1.06 | +5.66 | 56.6 | 0.00 | 1.00 | 0.94 | 56.6 |
| neutral | 2 | 0.00 | 1.00 | 1.00 | +3.14 | 124.9 | 0.00 | 0.94 | 1.00 | 124.9 |

### Cases

| case | variant | cand_arg | cand_rank | full_rank | full_arg | lens_in/out | top |
|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 72.1 | 0.00 | 1.00/0.88 |  natural Location Objects |
| front_back:long:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 41.0 | 0.00 | 1.00/1.00 | Abstract ## Location |
| front_back:neutral:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 96.2 | 0.00 | 1.00/1.00 | Common Bi Bot |
| front_back:neutral:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 153.6 | 0.00 | 0.88/1.00 |    N  Articles |

## deepseek7b

### By category

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| container | 18 | 0.22 | 0.61 | 1.51 | +0.99 | 3736.7 | 0.00 | 0.13 | 0.61 | 3717.4 |
| number | 18 | 0.17 | 0.46 | 1.94 | +0.05 | 6494.7 | 0.00 | 0.18 | 0.46 | 6494.7 |
| plant | 18 | 0.11 | 0.99 | 1.01 | +4.99 | 101.8 | 0.00 | 0.85 | 0.99 | 101.8 |
| time | 18 | 0.28 | 0.66 | 1.46 | +1.01 | 7025.0 | 0.00 | 0.12 | 0.66 | 7025.0 |

### By format

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.29 | 0.61 | 1.51 | +1.10 | 5550.9 | 0.00 | 0.21 | 0.61 | 5550.9 |
| label_colon | 24 | 0.21 | 0.83 | 1.23 | +2.01 | 1742.2 | 0.00 | 0.38 | 0.83 | 1742.2 |
| multiple_choice | 24 | 0.08 | 0.59 | 1.70 | +2.17 | 5725.5 | 0.00 | 0.36 | 0.59 | 5711.0 |

### By family

| group | n | prev_clean | cand_arg | cand_rank | cand_margin | full_rank | full_arg | lens_in_arg | lens_out_arg | best_full_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| long | 24 | 0.25 | 0.63 | 1.56 | +1.93 | 3510.2 | 0.00 | 0.34 | 0.63 | 3510.2 |
| neutral | 24 | 0.17 | 0.67 | 1.53 | +0.94 | 6592.1 | 0.00 | 0.26 | 0.67 | 6592.1 |
| short | 24 | 0.17 | 0.74 | 1.35 | +2.41 | 2916.4 | 0.00 | 0.36 | 0.74 | 2901.9 |

### Cases

| case | variant | cand_arg | cand_rank | full_rank | full_arg | lens_in/out | top |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 0.12 | 2.00 | 26407.6 | 0.00 | 0.00/0.12 | !=- -disabled .${ |
| back_front:long:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 0.00 | 3.00 | 3266.9 | 0.00 | 0.00/0.00 |  either  computer  cat |
| back_front:long:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 33.5 | 0.00 | 1.00/1.00 |  either  tree  concrete |
| back_front:long:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 0.62 | 1.50 | 6416.9 | 0.00 | 0.00/0.62 |  C    (%) |
| back_front:long:label_colon:container | final_norm_output_lm:4.0:0.0 | 0.12 | 2.00 | 6960.2 | 0.00 | 0.00/0.12 |   -sw -schema |
| back_front:long:label_colon:number | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 5.1 | 0.00 | 1.00/1.00 |  abstract {{ Abstract |
| back_front:long:label_colon:plant | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 16.1 | 0.00 | 0.00/0.88 |  abstract  concrete  semantic |
| back_front:long:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 2319.4 | 0.00 | 0.62/1.00 |   <<<< ;break |
| back_front:long:multiple_choice:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 4.4 | 0.00 | 1.00/1.00 |  container  plant  box |
| back_front:long:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.50 | 1.88 | 47.2 | 0.00 | 0.00/0.50 |  container  plant  number |
| back_front:long:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 2.9 | 0.00 | 1.00/1.00 |  plant  container  tree |
| back_front:long:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 19.4 | 0.00 | 0.00/0.88 |  time  number  container |
| back_front:neutral:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 1579.8 | 0.00 | 0.00/1.00 |  <<=  polys 一分钟 |
| back_front:neutral:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 6991.6 | 0.00 | 0.00/1.00 |  <<= 这个地图  soaked |
| back_front:neutral:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 60.2 | 0.00 | 1.00/1.00 |  tree **\n  Which |
| back_front:neutral:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 3583.2 | 0.00 | 0.00/1.00 | ym ACHE 一分钟 |
| back_front:neutral:label_colon:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 156.4 | 0.00 | 0.00/1.00 |  implode  unset  Yii |
| back_front:neutral:label_colon:number | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 3241.1 | 0.00 | 0.12/0.88 | ﻿using  fkk  asn |
| back_front:neutral:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 153.6 | 0.00 | 0.88/1.00 | Tree  tree  Tree |
| back_front:neutral:label_colon:time | final_norm_output_lm:4.0:0.0 | 0.75 | 1.25 | 1719.8 | 0.00 | 0.00/0.75 | ky ÷ ze |
| back_front:neutral:multiple_choice:container | final_norm_output_lm:4.0:0.0 | 0.25 | 1.88 | 65.0 | 0.00 | 0.00/0.25 |  plant  ?\n\n  container |
| back_front:neutral:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.00 | 4.00 | 78759.9 | 0.00 | 0.00/0.00 |  genetically 懿 档 |
| back_front:neutral:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 2.4 | 0.00 | 1.00/1.00 |  container  plant plant |
| back_front:neutral:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.00 | 2.00 | 80.0 | 0.00 | 0.00/0.00 |  time  ?\n\n  ?\n |
| back_front:short:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 0.75 | 1.38 | 3906.1 | 0.00 | 0.00/0.75 |    either  \n |
| back_front:short:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 0.00 | 2.00 | 1656.4 | 0.00 | 0.00/0.00 |    either  e |
| back_front:short:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 215.9 | 0.00 | 0.88/1.00 |  \n    \n\n |
| back_front:short:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 1227.1 | 0.00 | 0.00/1.00 |  \n\n  \n  either |
| back_front:short:label_colon:container | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 1134.8 | 0.00 | 0.00/0.88 |  **  \n   |
| back_front:short:label_colon:number | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 258.2 | 0.00 | 0.50/1.00 |  quantity  \  Number |
| back_front:short:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 263.6 | 0.00 | 1.00/1.00 |  \n  Category  \ |
| back_front:short:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 26.0 | 0.00 | 0.00/1.00 |  time time  word |
| back_front:short:multiple_choice:container | final_norm_output_suppress:0.0:1.0 | 1.00 | 1.00 | 662.1 | 0.00 | 0.00/1.00 | 在这个  cords  <<= |
| back_front:short:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.00 | 3.00 | 250.1 | 0.00 | 0.00/0.00 |  number  time  ** |
| back_front:short:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 3.9 | 0.00 | 1.00/1.00 |  \  }\  number |
| back_front:short:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 4.9 | 0.00 | 0.25/0.88 |  time time  Time |
| front_back:long:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 0.00 | 2.50 | 9803.2 | 0.00 | 0.00/0.00 | {}", eworld >-- |
| front_back:long:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 0.62 | 1.38 | 1825.2 | 0.00 | 0.00/0.62 |  concrete  either  abstract |
| front_back:long:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 50.4 | 0.00 | 1.00/1.00 |  either  concrete  plant |
| front_back:long:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 0.12 | 2.38 | 23010.6 | 0.00 | 0.00/0.12 | born  centr  deep |
| front_back:long:label_colon:container | final_norm_output_lm:4.0:0.0 | 0.12 | 2.00 | 401.0 | 0.00 | 0.00/0.12 |  abstract Abstract  concrete |
| front_back:long:label_colon:number | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 3.6 | 0.00 | 1.00/1.00 |  abstract  quantity Abstract |
| front_back:long:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 9.1 | 0.00 | 0.25/1.00 | Abstract  abstract plant |
| front_back:long:label_colon:time | final_norm_output_lm:4.0:0.0 | 0.62 | 1.38 | 121.4 | 0.00 | 0.25/0.62 |  [ 1  {\ |
| front_back:long:multiple_choice:container | final_norm_output_lm:4.0:0.0 | 0.75 | 1.25 | 3205.6 | 0.00 | 0.00/0.75 |  re （  Acad |
| front_back:long:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.00 | 3.75 | 239.2 | 0.00 | 0.00/0.00 |  container  number  Options |
| front_back:long:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 2.4 | 0.00 | 1.00/1.00 |  plant plant  moss |
| front_back:long:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.75 | 1.25 | 73.5 | 0.00 | 0.00/0.75 |  time  Time  A |
| front_back:neutral:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 0.38 | 2.12 | 5377.0 | 0.00 | 0.00/0.38 |  \n\n  \n  How |
| front_back:neutral:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 0.12 | 1.88 | 978.0 | 0.00 | 0.00/0.12 |  "  What  number |
| front_back:neutral:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 91.5 | 0.00 | 1.00/1.00 |  \n\n  \n  Caught |
| front_back:neutral:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 64.5 | 0.00 | 0.00/0.88 |  What  month  length |
| front_back:neutral:label_colon:container | final_norm_output_lm:4.0:0.0 | 0.62 | 1.50 | 6346.9 | 0.00 | 0.00/0.62 |  Algebra  Mathematics  Geometry |
| front_back:neutral:label_colon:number | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 3298.0 | 0.00 | 0.12/1.00 | 用  <<= !=- |
| front_back:neutral:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 48.5 | 0.00 | 1.00/1.00 | 1  bot  Ge |
| front_back:neutral:label_colon:time | final_norm_output_lm:4.0:0.0 | 0.25 | 2.75 | 14812.2 | 0.00 | 0.00/0.25 |  wildcard plier 特点是 |
| front_back:neutral:multiple_choice:container | final_norm_output_lm:4.0:0.0 | 0.00 | 2.50 | 117.6 | 0.00 | 0.00/0.00 |  time  Time  ?\n\n |
| front_back:neutral:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.00 | 2.38 | 310.5 | 0.00 | 0.00/0.00 |  number  container  time |
| front_back:neutral:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 3.9 | 0.00 | 1.00/1.00 |  ?\n  plant plant |
| front_back:neutral:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 30368.4 | 0.00 | 0.12/0.88 |  soaked "—  affairs |
| front_back:short:answer_one_word:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 1090.6 | 0.00 | 0.00/1.00 |  \n  "  either |
| front_back:short:answer_one_word:number | final_norm_output_lm:4.0:0.0 | 0.25 | 1.75 | 14163.0 | 0.00 | 0.00/0.25 | .${  !$  asn |
| front_back:short:answer_one_word:plant | final_norm_output_lm:4.0:0.0 | 0.88 | 1.12 | 867.8 | 0.00 | 0.25/0.88 |  the  each   |
| front_back:short:answer_one_word:time | final_norm_output_lm:4.0:0.0 | 0.00 | 2.12 | 20554.2 | 0.00 | 0.00/0.00 |  fkk  asn  krb |
| front_back:short:label_colon:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 37.4 | 0.00 | 0.38/1.00 | math sign  physical |
| front_back:short:label_colon:number | final_norm_output_lm:4.0:0.0 | 0.88 | 1.25 | 371.9 | 0.00 | 0.50/0.88 |  \n    number |
| front_back:short:label_colon:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 4.8 | 0.00 | 1.00/1.00 | Plant Bot Math |
| front_back:short:label_colon:time | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 104.5 | 0.00 | 0.50/1.00 |  {  \n   |
| front_back:short:multiple_choice:container | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 4.8 | 0.00 | 1.00/1.00 |  container  \ container |
| front_back:short:multiple_choice:number | final_norm_output_lm:4.0:0.0 | 0.00 | 2.50 | 1238.8 | 0.00 | 0.00/0.00 |  number  \  Number |
| front_back:short:multiple_choice:plant | final_norm_output_lm:4.0:0.0 | 1.00 | 1.00 | 2.2 | 0.00 | 1.00/1.00 |  plant plant  A |
| front_back:short:multiple_choice:time | final_norm_output_lm:4.0:0.0 | 0.25 | 2.12 | 21943.9 | 0.00 | 0.38/0.25 | - 主意 在这个 |

