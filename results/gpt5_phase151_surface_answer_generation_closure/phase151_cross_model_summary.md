# Phase 151 Cross-model Surface Answer Generation Closure Summary

## qwen3

### By category

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 0.25 | 0.00 | 0.00 | 1421.4 | 98068.7 | 32677.6 | 0.00 | other |
| time | 2 | 0.06 | 0.00 | 0.00 | 381.4 | 105437.1 | 43163.2 | 0.00 | other |

### By format

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.16 | 0.00 | 0.00 | 901.4 | 101752.9 | 37920.4 | 0.00 | other |

### By family

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 0.12 | 0.00 | 0.00 | 1708.1 | 141236.2 | 65370.7 | 0.00 | other |
| neutral | 2 | 0.19 | 0.00 | 0.00 | 94.6 | 62269.6 | 10470.1 | 0.00 | other |

### Cases

| case | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | greedy_class | examples |
|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | 0.00 | 2721.6 | 145027.5 | 64209.4 | other | .fhir .fhir .fhir |
| front_back:long:label_colon:time | 0.00 | 694.6 | 137444.9 | 66532.0 | other | 改革委 改革委 改革委 |
| front_back:neutral:label_colon:plant | 0.00 | 121.1 | 51109.9 | 1145.8 | other | 改革委 改革委 إنش |
| front_back:neutral:label_colon:time | 0.00 | 68.1 | 73429.4 | 19794.4 | other | 改革委 改革委 改革委 |

## glm4

### By category

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 0.06 | 0.00 | 0.12 | 15.7 | 91.4 | 84.2 | 0.00 | other |
| time | 2 | 0.06 | 0.06 | 0.06 | 9.3 | 97.3 | 97.3 | 0.06 | other |

### By format

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.06 | 0.03 | 0.09 | 12.5 | 94.4 | 90.8 | 0.03 | other |

### By family

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 0.00 | 0.00 | 0.06 | 15.5 | 56.6 | 56.6 | 0.00 | other |
| neutral | 2 | 0.12 | 0.06 | 0.12 | 9.5 | 132.2 | 124.9 | 0.06 | other |

### Cases

| case | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | greedy_class | examples |
|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | 0.12 | 25.4 | 72.1 | 72.1 | other |  natural Living  Objects |
| front_back:long:label_colon:time | 0.00 | 5.6 | 41.0 | 41.0 | other | Abstract Abstract Abstract |
| front_back:neutral:label_colon:plant | 0.12 | 6.0 | 110.8 | 96.2 | other | Common Al Common |
| front_back:neutral:label_colon:time | 0.12 | 13.0 | 153.6 | 153.6 | other |     Term |

## deepseek7b

### By category

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| container | 18 | 0.28 | 0.12 | 0.15 | 950.6 | 29464.9 | 3717.4 | 0.19 | format_only |
| number | 18 | 0.19 | 0.10 | 0.19 | 735.6 | 12310.4 | 6494.7 | 0.32 | other |
| plant | 18 | 0.33 | 0.28 | 0.38 | 6.7 | 178.2 | 101.8 | 0.40 | canonical |
| time | 18 | 0.20 | 0.15 | 0.19 | 2670.8 | 24552.4 | 7025.0 | 0.29 | format_only |

### By format

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.04 | 0.02 | 0.03 | 1043.1 | 14596.1 | 5550.9 | 0.03 | format_only |
| label_colon | 24 | 0.05 | 0.03 | 0.10 | 605.3 | 5674.8 | 1742.2 | 0.10 | other |
| multiple_choice | 24 | 0.66 | 0.44 | 0.56 | 1624.4 | 29608.6 | 5711.0 | 0.77 | canonical |

### By family

| group | n | clean_exp_arg | support_exp_arg | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | good_greedy | top_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 24 | 0.27 | 0.23 | 0.24 | 736.4 | 13544.5 | 3510.2 | 0.31 | other |
| neutral | 24 | 0.17 | 0.08 | 0.16 | 1470.0 | 22226.8 | 6592.1 | 0.27 | other |
| short | 24 | 0.31 | 0.18 | 0.29 | 1066.3 | 14108.1 | 2901.9 | 0.32 | format_only |

### Cases

| case | final_exp_arg | final_exp_rank | final_canon_rank | final_syn_rank | greedy_class | examples |
|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.00 | 6030.8 | 33890.9 | 26407.6 | format_only | !=- !=- jsx |
| back_front:long:answer_one_word:number | 0.00 | 51.9 | 4842.9 | 3266.9 | other |  either  either  either |
| back_front:long:answer_one_word:plant | 0.12 | 2.4 | 49.0 | 33.5 | other |  either  flower  either |
| back_front:long:answer_one_word:time | 0.00 | 88.6 | 6416.9 | 6416.9 | format_only |  C  C   |
| back_front:long:label_colon:container | 0.00 | 2315.9 | 16587.0 | 6960.2 | format_only |       |
| back_front:long:label_colon:number | 0.00 | 3.2 | 3451.8 | 5.1 | other |  abstract  abstract  abstract |
| back_front:long:label_colon:plant | 0.00 | 5.4 | 16.1 | 16.1 | other |  abstract  abstract  abstract |
| back_front:long:label_colon:time | 0.00 | 646.6 | 20833.1 | 2319.4 | format_only |   <<<<   |
| back_front:long:multiple_choice:container | 1.00 | 1.0 | 4.4 | 4.4 | canonical |  container  container  container |
| back_front:long:multiple_choice:number | 0.75 | 1.2 | 47.2 | 47.2 | canonical |  container  container  number |
| back_front:long:multiple_choice:plant | 1.00 | 1.0 | 2.9 | 2.9 | canonical |  plant  plant  plant |
| back_front:long:multiple_choice:time | 0.88 | 1.2 | 19.4 | 19.4 | canonical |  time  time  time |
| back_front:neutral:answer_one_word:container | 0.00 | 971.6 | 96208.0 | 1579.8 | format_only |  <<= 一刀  polys |
| back_front:neutral:answer_one_word:number | 0.00 | 2301.2 | 6991.6 | 6991.6 | format_only |  <<=  <<=  <<= |
| back_front:neutral:answer_one_word:plant | 0.25 | 5.8 | 164.6 | 60.2 | format_only |  tree  flower  grass |
| back_front:neutral:answer_one_word:time | 0.00 | 1830.8 | 48164.1 | 3583.2 | other | ym 一分钟 ighb |
| back_front:neutral:label_colon:container | 0.00 | 13.6 | 156.4 | 156.4 | other |  implode  unset  implode |
| back_front:neutral:label_colon:number | 0.00 | 1207.5 | 15620.5 | 3241.1 | other | ﻿using ﻿using ﻿using |
| back_front:neutral:label_colon:plant | 0.12 | 12.8 | 382.9 | 153.6 | other | Tree  bot  DNA |
| back_front:neutral:label_colon:time | 0.00 | 423.1 | 19924.5 | 1719.8 | format_only | ky ÷ ÷ |
| back_front:neutral:multiple_choice:container | 0.12 | 5.9 | 65.0 | 65.0 | option_like |  plant  container  ?\n\n |
| back_front:neutral:multiple_choice:number | 0.00 | 5548.8 | 150567.4 | 78759.9 | option_like |  genetically 懿  genetically |
| back_front:neutral:multiple_choice:plant | 0.88 | 1.1 | 2.4 | 2.4 | canonical |  container  plant  plant |
| back_front:neutral:multiple_choice:time | 0.38 | 2.5 | 80.0 | 80.0 | format_only |  time  ?\n\n  ?\n\n |
| back_front:short:answer_one_word:container | 0.00 | 4.0 | 7139.4 | 3906.1 | format_only |       |
| back_front:short:answer_one_word:number | 0.00 | 8.8 | 3713.1 | 1656.4 | format_only |       |
| back_front:short:answer_one_word:plant | 0.00 | 15.2 | 447.8 | 215.9 | format_only |  \n  \n   |
| back_front:short:answer_one_word:time | 0.00 | 13.5 | 2952.1 | 1227.1 | format_only |  \n\n  \n\n  \n\n |
| back_front:short:label_colon:container | 0.00 | 90.9 | 2316.8 | 1134.8 | format_only |  **  mathematics  ** |
| back_front:short:label_colon:number | 0.38 | 2.2 | 1378.6 | 258.2 | other |  quantity  \  Category |
| back_front:short:label_colon:plant | 0.12 | 24.4 | 786.4 | 263.6 | format_only |  \n  \n  Category |
| back_front:short:label_colon:time | 0.62 | 6.1 | 43.9 | 26.0 | canonical |  time  time  time |
| back_front:short:multiple_choice:container | 0.00 | 314.2 | 151940.0 | 314.2 | format_only |  cords -  outcry |
| back_front:short:multiple_choice:number | 1.00 | 1.0 | 250.1 | 250.1 | canonical |  number  number  number |
| back_front:short:multiple_choice:plant | 0.75 | 1.8 | 3.9 | 3.9 | canonical |  \  \  plant |
| back_front:short:multiple_choice:time | 0.88 | 1.5 | 4.9 | 4.9 | canonical |  time  time  time |
| front_back:long:answer_one_word:container | 0.00 | 5073.9 | 55988.9 | 9803.2 | format_only | {}", {}", {}", |
| front_back:long:answer_one_word:number | 0.00 | 66.9 | 8840.5 | 1825.2 | other |  concrete  concrete  concrete |
| front_back:long:answer_one_word:plant | 0.25 | 2.2 | 70.2 | 50.4 | other |  either  either  either |
| front_back:long:answer_one_word:time | 0.00 | 1135.0 | 23010.6 | 23010.6 | other | born  deep 为核心 |
| front_back:long:label_colon:container | 0.00 | 61.0 | 444.2 | 401.0 | other |  abstract  abstract  abstract |
| front_back:long:label_colon:number | 0.12 | 2.0 | 3306.4 | 3.6 | other |  abstract  abstract  abstract |
| front_back:long:label_colon:plant | 0.00 | 3.4 | 9.1 | 9.1 | other | Abstract Al  abstract |
| front_back:long:label_colon:time | 0.00 | 81.2 | 152.0 | 121.4 | format_only |  [ 1  {\ |
| front_back:long:multiple_choice:container | 0.00 | 2094.6 | 146770.5 | 3205.6 | other |  re  re  re |
| front_back:long:multiple_choice:number | 0.00 | 2.4 | 239.2 | 239.2 | option_like |  container  container  container |
| front_back:long:multiple_choice:plant | 1.00 | 1.0 | 2.4 | 2.4 | canonical |  plant  plant  plant |
| front_back:long:multiple_choice:time | 0.75 | 1.2 | 73.5 | 73.5 | canonical |  time  time  plant |
| front_back:neutral:answer_one_word:container | 0.00 | 15.1 | 5593.6 | 5377.0 | format_only |  \n\n  \n\n  \n |
| front_back:neutral:answer_one_word:number | 0.00 | 14.1 | 1997.4 | 978.0 | other |  "  " _\n\n |
| front_back:neutral:answer_one_word:plant | 0.00 | 24.4 | 105.1 | 91.5 | format_only |  \n\n  al  \n\n |
| front_back:neutral:answer_one_word:time | 0.00 | 11.4 | 99.6 | 64.5 | other |  What  What  What |
| front_back:neutral:label_colon:container | 0.00 | 57.5 | 11870.6 | 6346.9 | other |  Algebra 1  chest |
| front_back:neutral:label_colon:number | 0.00 | 132.1 | 3298.0 | 3298.0 | format_only | 用  <<=  <<= |
| front_back:neutral:label_colon:plant | 0.00 | 10.6 | 75.1 | 48.5 | other | 1  Algebra  Algebra |
| front_back:neutral:label_colon:time | 0.00 | 9358.6 | 34146.9 | 14812.2 | other |  wildcard [root  wildcard |
| front_back:neutral:multiple_choice:container | 0.62 | 1.6 | 117.6 | 117.6 | canonical |  time  time  container |
| front_back:neutral:multiple_choice:number | 0.88 | 1.5 | 310.5 | 310.5 | canonical |  number  number  number |
| front_back:neutral:multiple_choice:plant | 0.50 | 2.1 | 3.9 | 3.9 | canonical |  ?\n  ?\n  plant |
| front_back:neutral:multiple_choice:time | 0.00 | 13327.2 | 137496.5 | 30368.4 | option_like |  soaked  soaked  soaked |
| front_back:short:answer_one_word:container | 0.00 | 20.1 | 1228.2 | 1090.6 | format_only |  \n  \  " |
| front_back:short:answer_one_word:number | 0.00 | 3867.6 | 14163.0 | 14163.0 | format_only | .${ .${  <<= |
| front_back:short:answer_one_word:plant | 0.00 | 4.2 | 1078.8 | 867.8 | format_only |  the     |
| front_back:short:answer_one_word:time | 0.00 | 3473.8 | 27149.5 | 20554.2 | other |  fkk  fkk  fkk |
| front_back:short:label_colon:container | 0.00 | 37.4 | 42.0 | 37.4 | other | math case Vehicle |
| front_back:short:label_colon:number | 0.12 | 7.9 | 1242.9 | 371.9 | format_only |  \n    Number |
| front_back:short:label_colon:plant | 1.00 | 1.0 | 4.8 | 4.8 | canonical | Plant Plant Plant |
| front_back:short:label_colon:time | 0.00 | 23.8 | 104.5 | 104.5 | format_only |  {  \ Word |
| front_back:short:multiple_choice:container | 1.00 | 1.0 | 4.8 | 4.8 | canonical |  container  container  container |
| front_back:short:multiple_choice:number | 0.12 | 20.6 | 1326.2 | 1238.8 | format_only |  number  \  container |
| front_back:short:multiple_choice:plant | 0.88 | 1.1 | 2.2 | 2.2 | canonical |  plant  plant  plant |
| front_back:short:multiple_choice:time | 0.00 | 17648.8 | 121271.5 | 21943.9 | format_only | -  soaked  soaked |

