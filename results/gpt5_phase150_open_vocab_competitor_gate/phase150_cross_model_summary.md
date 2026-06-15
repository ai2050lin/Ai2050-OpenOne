# Phase 150 Cross-model Open-Vocab Competitor Gate Summary

## qwen3

### By category

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 1.00 | 0.50 | 14547.4 | 14548.7 | 32677.6 | 0.00 | 68741.5 | format_or_fragment |
| time | 2 | 0.56 | 0.00 | 20758.4 | 20761.2 | 43163.2 | 0.00 | 111794.3 | non_ascii_or_fragment |

### By format

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.78 | 0.25 | 17652.9 | 17654.9 | 37920.4 | 0.00 | 90267.9 | non_ascii_or_fragment |

### By family

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 1.00 | 0.00 | 29594.2 | 29597.6 | 65370.7 | 0.00 | 122754.1 | format_or_fragment |
| neutral | 2 | 0.56 | 0.50 | 5711.6 | 5712.3 | 10470.1 | 0.00 | 57781.8 | non_ascii_or_fragment |

### Cases

| case | cand4 | semantic | alpha_rank | nonfmt_rank | full_rank | arg_class | top_tokens |
|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | 1.00 | 0.00 | 28369.8 | 28372.4 | 64209.4 | format_or_fragment | .fhir إنش 改革委 نموذ |
| front_back:long:label_colon:time | 1.00 | 0.00 | 30818.8 | 30822.8 | 66532.0 | non_ascii_or_fragment | 改革委 إنش .fhir .FixedSingle |
| front_back:neutral:label_colon:plant | 1.00 | 1.00 | 725.0 | 725.0 | 1145.8 | non_ascii_or_fragment | 改革委 إنش 有网友  overposting |
| front_back:neutral:label_colon:time | 0.12 | 0.00 | 10698.1 | 10699.6 | 19794.4 | non_ascii_or_fragment | 改革委 PRECATED  overposting .FixedSingle |

## glm4

### By category

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| plant | 2 | 0.94 | 0.88 | 78.4 | 78.8 | 84.2 | 0.00 | 281.9 | alphabetic_other |
| time | 2 | 1.00 | 0.94 | 84.6 | 85.1 | 97.3 | 0.00 | 363.1 | alphabetic_other |

### By format

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| label_colon | 4 | 0.97 | 0.91 | 81.5 | 81.9 | 90.8 | 0.00 | 322.5 | alphabetic_other |

### By family

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 2 | 0.94 | 0.88 | 50.9 | 51.0 | 56.6 | 0.00 | 196.6 | alphabetic_other |
| neutral | 2 | 1.00 | 0.94 | 112.1 | 112.9 | 124.9 | 0.00 | 448.3 | alphabetic_other |

### Cases

| case | cand4 | semantic | alpha_rank | nonfmt_rank | full_rank | arg_class | top_tokens |
|---|---|---|---|---|---|---|---|
| front_back:long:label_colon:plant | 0.88 | 0.88 | 63.8 | 64.0 | 72.1 | alphabetic_other |  natural Objects  Natural Abstract |
| front_back:long:label_colon:time | 1.00 | 0.88 | 38.0 | 38.0 | 41.0 | alphabetic_other | Abstract ## Location Time |
| front_back:neutral:label_colon:plant | 1.00 | 0.88 | 93.1 | 93.6 | 96.2 | alphabetic_other | Common Bi Bot Pl |
| front_back:neutral:label_colon:time | 1.00 | 1.00 | 131.1 | 132.1 | 153.6 | whitespace | Term  Terms Time  Term |

## deepseek7b

### By category

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| container | 18 | 0.61 | 0.41 | 2286.3 | 2725.4 | 3717.4 | 0.00 | 11976.8 | punctuation |
| number | 18 | 0.46 | 0.27 | 3395.2 | 4276.5 | 6494.7 | 0.00 | 13873.6 | alphabetic_other |
| plant | 18 | 0.99 | 0.97 | 85.4 | 86.0 | 101.8 | 0.00 | 798.1 | target_synonym |
| time | 18 | 0.66 | 0.37 | 3690.6 | 4627.6 | 7022.2 | 0.00 | 17728.5 | alphabetic_other |

### By format

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 24 | 0.61 | 0.39 | 3270.3 | 3962.5 | 5550.9 | 0.00 | 16489.6 | alphabetic_other |
| label_colon | 24 | 0.83 | 0.61 | 1073.6 | 1261.2 | 1742.2 | 0.00 | 6165.5 | alphabetic_other |
| multiple_choice | 24 | 0.59 | 0.51 | 2749.3 | 3562.8 | 5709.0 | 0.00 | 10627.7 | target_synonym |

### By family

| group | n | cand4_arg | semantic_arg | alphabetic_rank | nonfmt_rank | full_rank | full_arg | support_full_rank | top_arg_class |
|---|---|---|---|---|---|---|---|---|---|
| long | 24 | 0.63 | 0.48 | 1920.1 | 2404.5 | 3510.2 | 0.00 | 9484.9 | alphabetic_other |
| neutral | 24 | 0.67 | 0.39 | 3473.8 | 4333.8 | 6592.1 | 0.00 | 15341.9 | alphabetic_other |
| short | 24 | 0.74 | 0.64 | 1699.2 | 2048.2 | 2899.8 | 0.00 | 8455.9 | target_synonym |

### Cases

| case | cand4 | semantic | alpha_rank | nonfmt_rank | full_rank | arg_class | top_tokens |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.12 | 0.00 | 12991.8 | 17235.1 | 26407.6 | punctuation | !=- -disabled .${ $max |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 2822.6 | 2868.9 | 3266.9 | alphabetic_other |  either  computer  cat  \ |
| back_front:long:answer_one_word:plant | 1.00 | 1.00 | 21.9 | 21.9 | 33.5 | alphabetic_other |  tree  either  concrete  \ |
| back_front:long:answer_one_word:time | 0.62 | 0.00 | 3554.6 | 4223.8 | 6416.9 | whitespace |  C    (%)  hum |
| back_front:long:label_colon:container | 0.12 | 0.00 | 3814.9 | 4872.1 | 6960.2 | whitespace |   medi -schema <<< |
| back_front:long:label_colon:number | 1.00 | 1.00 | 4.8 | 4.8 | 5.1 | alphabetic_other |  abstract {{ Abstract  {{ |
| back_front:long:label_colon:plant | 0.88 | 0.88 | 12.0 | 12.0 | 16.1 | alphabetic_other |  abstract  concrete  semantic Abstract |
| back_front:long:label_colon:time | 1.00 | 0.25 | 1356.4 | 1729.1 | 2319.4 | whitespace |   <<<< ;break ='') |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 3.8 | 3.8 | 4.4 | target_synonym |  container  plant container  Container |
| back_front:long:multiple_choice:number | 0.50 | 0.50 | 34.1 | 34.1 | 47.2 | target_synonym |  container  plant  number  Container |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 2.9 | 2.9 | 2.9 | target_synonym |  plant  tree plant  ?\n\n |
| back_front:long:multiple_choice:time | 0.88 | 0.88 | 12.8 | 12.8 | 19.4 | target_synonym |  time  number  container  plant |
| back_front:neutral:answer_one_word:container | 1.00 | 0.50 | 891.0 | 1121.9 | 1579.8 | punctuation |  <<= 这个地图  polys 一分钟 |
| back_front:neutral:answer_one_word:number | 1.00 | 0.00 | 3947.4 | 4835.1 | 6991.6 | punctuation |  <<= 这个地图  soaked .§ |
| back_front:neutral:answer_one_word:plant | 1.00 | 0.88 | 34.6 | 35.0 | 60.2 | punctuation |  tree **\n  "  \n |
| back_front:neutral:answer_one_word:time | 1.00 | 0.12 | 1944.5 | 2446.0 | 3583.2 | alphabetic_other | ym 一分钟 ighb  polys |
| back_front:neutral:label_colon:container | 1.00 | 1.00 | 136.8 | 139.9 | 156.4 | alphabetic_other |  unset  Container  implode  Yii |
| back_front:neutral:label_colon:number | 0.88 | 0.38 | 2124.4 | 2522.5 | 3241.1 | non_ascii_or_fragment | ﻿using  fkk  asn صند |
| back_front:neutral:label_colon:plant | 1.00 | 1.00 | 136.1 | 138.6 | 153.6 | alphabetic_other | Tree  Data  Ge  General |
| back_front:neutral:label_colon:time | 0.75 | 0.00 | 1096.2 | 1239.8 | 1719.8 | non_ascii_or_fragment | ky ÷ ze  Capital |
| back_front:neutral:multiple_choice:container | 0.25 | 0.25 | 41.5 | 41.9 | 65.0 | other_category |  plant  ?\n\n  container  ?\n |
| back_front:neutral:multiple_choice:number | 0.00 | 0.00 | 36781.8 | 48737.5 | 78759.9 | alphabetic_other | 懿  genetically 档 imized |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 2.4 | 2.4 | 2.4 | target_synonym |  plant plant  ?\n\n  number |
| back_front:neutral:multiple_choice:time | 0.00 | 0.00 | 40.9 | 40.9 | 80.0 | punctuation |  time  ?\n\n  ?\n  ....\n\n |
| back_front:short:answer_one_word:container | 0.75 | 0.12 | 3409.8 | 3460.9 | 3906.1 | whitespace |    either  \n  a |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 1322.6 | 1327.2 | 1656.4 | whitespace |    either  e   \n |
| back_front:short:answer_one_word:plant | 1.00 | 1.00 | 167.1 | 168.0 | 215.9 | whitespace |  \n    \n\n   \n |
| back_front:short:answer_one_word:time | 1.00 | 1.00 | 1055.8 | 1065.4 | 1227.1 | whitespace |  \n\n  \n  either  ' |
| back_front:short:label_colon:container | 0.88 | 0.75 | 948.1 | 957.9 | 1134.8 | punctuation |  **  \n    " |
| back_front:short:label_colon:number | 1.00 | 0.88 | 212.2 | 214.1 | 258.2 | generic_continuation |  \  Number  Count  Category |
| back_front:short:label_colon:plant | 1.00 | 1.00 | 213.0 | 214.8 | 263.6 | whitespace |  \n  Category  \  each |
| back_front:short:label_colon:time | 1.00 | 1.00 | 16.5 | 16.5 | 26.0 | target_synonym |  word  Word  { math |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 178.4 | 204.6 | 314.2 | punctuation | � anos  outcry  GPI |
| back_front:short:multiple_choice:number | 0.00 | 0.00 | 87.6 | 96.0 | 250.1 | target_synonym |  number  **  \  " |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 2.5 | 2.5 | 3.9 | target_synonym |  \  }\  number  "\n\n |
| back_front:short:multiple_choice:time | 0.88 | 0.88 | 4.5 | 4.5 | 4.9 | target_synonym |  time time  Time  options |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 5266.5 | 6718.1 | 9803.2 | punctuation | {}", eworld >--  >( |
| front_back:long:answer_one_word:number | 0.62 | 0.12 | 1522.5 | 1536.6 | 1825.2 | alphabetic_other |  concrete  either  abstract  \ |
| front_back:long:answer_one_word:plant | 1.00 | 1.00 | 36.2 | 36.2 | 50.4 | alphabetic_other |  either  concrete  plant  animal |
| front_back:long:answer_one_word:time | 0.12 | 0.00 | 12040.5 | 15309.2 | 23010.6 | alphabetic_other | born  deep  centr (proc |
| front_back:long:label_colon:container | 0.12 | 0.00 | 308.4 | 325.8 | 401.0 | alphabetic_other |  abstract Abstract  concrete  semantic |
| front_back:long:label_colon:number | 1.00 | 1.00 | 3.6 | 3.6 | 3.6 | alphabetic_other |  abstract  quantity Abstract quantity |
| front_back:long:label_colon:plant | 1.00 | 1.00 | 7.0 | 7.0 | 9.1 | alphabetic_other | Abstract  abstract plant {{ |
| front_back:long:label_colon:time | 0.62 | 0.12 | 71.9 | 76.1 | 121.4 | punctuation |  [ 1  {\ 2 |
| front_back:long:multiple_choice:container | 0.75 | 0.00 | 1999.2 | 2472.6 | 3205.6 | alphabetic_other |  re （  Acad 提出的 |
| front_back:long:multiple_choice:number | 0.00 | 0.00 | 137.8 | 144.9 | 239.2 | other_category |  container  number  Options  time |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 2.4 | 2.4 | 2.4 | target_synonym |  plant plant  Options  ?\n\n |
| front_back:long:multiple_choice:time | 0.75 | 0.75 | 54.5 | 54.5 | 73.5 | target_synonym |  time  Time  A  D |
| front_back:neutral:answer_one_word:container | 0.38 | 0.00 | 4980.9 | 5048.2 | 5377.0 | whitespace |  \n\n  \n  How  For |
| front_back:neutral:answer_one_word:number | 0.12 | 0.12 | 820.4 | 823.9 | 978.0 | object_token |  "  What  Answer  How |
| front_back:neutral:answer_one_word:plant | 1.00 | 1.00 | 77.4 | 77.5 | 91.5 | whitespace |  \n\n  \n  '  ( |
| front_back:neutral:answer_one_word:time | 0.88 | 0.88 | 50.8 | 50.8 | 64.5 | alphabetic_other |  What  length  "  How |
| front_back:neutral:label_colon:container | 0.62 | 0.25 | 5145.4 | 5413.0 | 6346.9 | whitespace |  Algebra  Geometry  Mathematics   |
| front_back:neutral:label_colon:number | 1.00 | 0.00 | 1618.6 | 2033.5 | 3298.0 | punctuation | 和发展 ∏ 用  <<= |
| front_back:neutral:label_colon:plant | 1.00 | 1.00 | 40.0 | 40.0 | 48.5 | alphabetic_other |  Data  Algebra  Mathematics 1 |
| front_back:neutral:label_colon:time | 0.25 | 0.00 | 8126.1 | 9931.8 | 14812.2 | alphabetic_other |  wildcard plier [root =\' |
| front_back:neutral:multiple_choice:container | 0.00 | 0.00 | 85.4 | 85.9 | 117.6 | target_synonym |  time  Time  ?\n\n  container |
| front_back:neutral:multiple_choice:number | 0.00 | 0.00 | 240.4 | 245.6 | 310.5 | target_synonym |  number  container  time  plant |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 2.8 | 2.8 | 3.9 | target_synonym |  ?\n  plant plant  ?\n\n |
| front_back:neutral:multiple_choice:time | 0.88 | 0.00 | 15006.9 | 18957.5 | 30368.4 | alphabetic_other |  soaked "—  affairs ?): |
| front_back:short:answer_one_word:container | 1.00 | 0.88 | 915.1 | 918.0 | 1090.6 | punctuation |  \n  "  either  \n\n |
| front_back:short:answer_one_word:number | 0.25 | 0.00 | 8401.9 | 10445.5 | 14163.0 | punctuation | .${  !$  asn  <<= |
| front_back:short:answer_one_word:plant | 0.88 | 0.62 | 773.1 | 777.8 | 867.8 | whitespace |  the  each    a |
| front_back:short:answer_one_word:time | 0.00 | 0.00 | 11437.9 | 14550.0 | 20554.2 | alphabetic_other |  fkk  asn  krb  mex |
| front_back:short:label_colon:container | 1.00 | 0.62 | 33.6 | 34.4 | 37.4 | alphabetic_other | balance Contents ...  physical |
| front_back:short:label_colon:number | 0.88 | 0.88 | 268.6 | 270.8 | 371.9 | whitespace |  \n    (  A |
| front_back:short:label_colon:plant | 1.00 | 1.00 | 4.4 | 4.4 | 4.8 | target_synonym | Plant Bot Math {{{ |
| front_back:short:label_colon:time | 1.00 | 0.75 | 66.6 | 67.5 | 104.5 | punctuation |  {  \n    Math |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 2.5 | 2.5 | 4.8 | target_synonym |  container  \ container  { |
| front_back:short:multiple_choice:number | 0.00 | 0.00 | 762.6 | 831.6 | 1238.8 | punctuation |  \  **  It  { |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 2.1 | 2.1 | 2.2 | target_synonym |  plant plant  A  __ |
| front_back:short:multiple_choice:time | 0.25 | 0.00 | 10493.5 | 13520.1 | 21894.0 | punctuation |  (--  soaked 在这个  thighs |

