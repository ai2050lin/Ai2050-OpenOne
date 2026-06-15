# Phase 151 Surface Answer Generation Closure: deepseek7b

Generated: 2026-06-15 14:34:09

| case | clean expanded arg | support expanded arg | final expanded arg | final expanded rank | final canonical rank | greedy class | examples |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.12 | 0.00 | 0.00 | 6030.8 | 33890.9 | format_only | !=- | !=- | jsx |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 51.9 | 4842.9 | other |  either |  either |  either |
| back_front:long:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 2.4 | 49.0 | other |  either |  flower |  either |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 88.6 | 6416.9 | format_only |  C |  C |   |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 2315.9 | 16587.0 | format_only |   |   |   |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 3.2 | 3451.8 | other |  abstract |  abstract |  abstract |
| back_front:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 5.4 | 16.1 | other |  abstract |  abstract |  abstract |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 646.6 | 20833.1 | format_only |   | <<<< |   |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.0 | 4.4 | canonical |  container |  container |  container |
| back_front:long:multiple_choice:number | 0.75 | 0.62 | 0.75 | 1.2 | 47.2 | canonical |  container |  container |  number |
| back_front:long:multiple_choice:plant | 0.88 | 0.88 | 1.00 | 1.0 | 2.9 | canonical |  plant |  plant |  plant |
| back_front:long:multiple_choice:time | 0.88 | 0.88 | 0.88 | 1.2 | 19.4 | canonical |  time |  time |  time |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 971.6 | 96208.0 | format_only |  <<= | 一刀 |  polys |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 2301.2 | 6991.6 | format_only |  <<= |  <<= |  <<= |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.25 | 5.8 | 164.6 | format_only |  tree |  flower |  grass |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 1830.8 | 48164.1 | other | ym | 一分钟 | ighb |
| back_front:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 13.6 | 156.4 | other |  implode |  unset |  implode |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 1207.5 | 15620.5 | other | ﻿using | ﻿using | ﻿using |
| back_front:neutral:label_colon:plant | 0.12 | 0.00 | 0.12 | 12.8 | 382.9 | other | Tree |  bot |  DNA |
| back_front:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 423.1 | 19924.5 | format_only | ky | ÷ | ÷ |
| back_front:neutral:multiple_choice:container | 0.62 | 0.00 | 0.12 | 5.9 | 65.0 | option_like |  plant |  container |  ?\n\n |
| back_front:neutral:multiple_choice:number | 0.62 | 0.00 | 0.00 | 5548.8 | 150567.4 | option_like |  genetically | 懿 |  genetically |
| back_front:neutral:multiple_choice:plant | 0.50 | 0.75 | 0.88 | 1.1 | 2.4 | canonical |  container |  plant |  plant |
| back_front:neutral:multiple_choice:time | 0.00 | 0.12 | 0.38 | 2.5 | 80.0 | format_only |  time |  ?\n\n |  ?\n\n |
| back_front:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | 4.0 | 7139.4 | format_only |   |   |   |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 8.8 | 3713.1 | format_only |   |   |   |
| back_front:short:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 15.2 | 447.8 | format_only |  \n |  \n |   |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | 13.5 | 2952.1 | format_only |  \n\n |  \n\n |  \n\n |
| back_front:short:label_colon:container | 0.00 | 0.00 | 0.00 | 90.9 | 2316.8 | format_only |  ** |  mathematics |  ** |
| back_front:short:label_colon:number | 0.50 | 0.00 | 0.38 | 2.2 | 1378.6 | other |  quantity |  \ |  Category |
| back_front:short:label_colon:plant | 0.12 | 0.00 | 0.12 | 24.4 | 786.4 | format_only |  \n |  \n |  Category |
| back_front:short:label_colon:time | 0.00 | 0.00 | 0.62 | 6.1 | 43.9 | canonical |  time |  time |  time |
| back_front:short:multiple_choice:container | 1.00 | 0.00 | 0.00 | 314.2 | 151940.0 | format_only |  cords | - |  outcry |
| back_front:short:multiple_choice:number | 0.50 | 0.62 | 1.00 | 1.0 | 250.1 | canonical |  number |  number |  number |
| back_front:short:multiple_choice:plant | 0.88 | 0.75 | 0.75 | 1.8 | 3.9 | canonical |  \ |  \ |  plant |
| back_front:short:multiple_choice:time | 0.88 | 0.88 | 0.88 | 1.5 | 4.9 | canonical |  time |  time |  time |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 5073.9 | 55988.9 | format_only | {}", | {}", | {}", |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 66.9 | 8840.5 | other |  concrete |  concrete |  concrete |
| front_back:long:answer_one_word:plant | 0.25 | 0.25 | 0.25 | 2.2 | 70.2 | other |  either |  either |  either |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 1135.0 | 23010.6 | other | born |  deep | 为核心 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 61.0 | 444.2 | other |  abstract |  abstract |  abstract |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.12 | 2.0 | 3306.4 | other |  abstract |  abstract |  abstract |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 3.4 | 9.1 | other | Abstract | Al |  abstract |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 81.2 | 152.0 | format_only |  [ | 1 |  {\ |
| front_back:long:multiple_choice:container | 0.88 | 0.00 | 0.00 | 2094.6 | 146770.5 | other |  re |  re |  re |
| front_back:long:multiple_choice:number | 0.12 | 0.00 | 0.00 | 2.4 | 239.2 | option_like |  container |  container |  container |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.0 | 2.4 | canonical |  plant |  plant |  plant |
| front_back:long:multiple_choice:time | 0.50 | 0.75 | 0.75 | 1.2 | 73.5 | canonical |  time |  time |  plant |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 15.1 | 5593.6 | format_only |  \n\n |  \n\n |  \n |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 14.1 | 1997.4 | other |  " |  " | _\n\n |
| front_back:neutral:answer_one_word:plant | 0.12 | 0.00 | 0.00 | 24.4 | 105.1 | format_only |  \n\n |  al |  \n\n |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 11.4 | 99.6 | other |  What |  What |  What |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 57.5 | 11870.6 | other |  Algebra | 1 |  chest |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 132.1 | 3298.0 | format_only | 用 |  <<= |  <<= |
| front_back:neutral:label_colon:plant | 0.12 | 0.00 | 0.00 | 10.6 | 75.1 | other | 1 |  Algebra |  Algebra |
| front_back:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 9358.6 | 34146.9 | other |  wildcard | [root |  wildcard |
| front_back:neutral:multiple_choice:container | 0.38 | 0.25 | 0.62 | 1.6 | 117.6 | canonical |  time |  time |  container |
| front_back:neutral:multiple_choice:number | 0.62 | 0.62 | 0.88 | 1.5 | 310.5 | canonical |  number |  number |  number |
| front_back:neutral:multiple_choice:plant | 0.50 | 0.12 | 0.50 | 2.1 | 3.9 | canonical |  ?\n |  ?\n |  plant |
| front_back:neutral:multiple_choice:time | 0.38 | 0.00 | 0.00 | 13327.2 | 137496.5 | option_like |  soaked |  soaked |  soaked |
| front_back:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | 20.1 | 1228.2 | format_only |  \n |  \ |  " |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 3867.6 | 14163.0 | format_only | .${ | .${ |  <<= |
| front_back:short:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 4.2 | 1078.8 | format_only |  the |   |   |
| front_back:short:answer_one_word:time | 0.12 | 0.00 | 0.00 | 3473.8 | 27149.5 | other |  fkk |  fkk |  fkk |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 37.4 | 42.0 | other | math | case | Vehicle |
| front_back:short:label_colon:number | 0.12 | 0.00 | 0.12 | 7.9 | 1242.9 | format_only |  \n |   |  Number |
| front_back:short:label_colon:plant | 0.25 | 0.62 | 1.00 | 1.0 | 4.8 | canonical | Plant | Plant | Plant |
| front_back:short:label_colon:time | 0.00 | 0.00 | 0.00 | 23.8 | 104.5 | format_only |  { |  \ | Word |
| front_back:short:multiple_choice:container | 1.00 | 0.88 | 1.00 | 1.0 | 4.8 | canonical |  container |  container |  container |
| front_back:short:multiple_choice:number | 0.25 | 0.00 | 0.12 | 20.6 | 1326.2 | format_only |  number |  \ |  container |
| front_back:short:multiple_choice:plant | 0.88 | 0.50 | 0.88 | 1.1 | 2.2 | canonical |  plant |  plant |  plant |
| front_back:short:multiple_choice:time | 0.88 | 0.00 | 0.00 | 17648.8 | 121271.5 | format_only | - |  soaked |  soaked |
