# Phase 152 Natural Surface Iterative Generation: deepseek7b

Generated: 2026-06-15 16:41:43

| case | clean hit | remove_restore hit | best_add hit | best_add | fmt_first_later | clean class | best class | examples |
|---|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:container | 0.25 | 0.00 | 0.12 | additive_support_lm:0.05 | 0.12 | other | other |  concrete, abstract |  concrete, abstract |  "bottle |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  concrete, abstract |  concrete, abstract |  concrete, abstract |
| back_front:long:answer_one_word:plant | 0.12 | 0.12 | 0.50 | additive_support_lm:0.5 | 0.00 | other | other |  concrete, abstract |  flower, flowers |  grass, grass |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.25 | additive_support_lm:0.2 | 0.25 | other | other |  'time' |  'time', |  'nighttime |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.25 | additive_support:0.05 | 0.25 | other | other |  [1] |  [bag] |  [Bottle |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  abstract.\n\nWait |  abstract.\n\nWait |  abstract.\n\nWait |
| back_front:long:label_colon:plant | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  abstract.\n\nWait |  abstract.\n\nWait |  abstract.\n\nGr |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  [ ].\n\n |  [ ].\n\n |  [1] |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  container.\n\n Why |  container.\n\nBut |  container.\n\nBut |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  container.\n\n Why |  container.\n\nBut |  number.\n\nBut |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  container.\n\nBut |  plant.\n\nOkay |  plant.\n\nOkay |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  time.\n\nBut |  time.\n\nBut |  time.\n\nOkay |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.38 | additive_support_lm:0.05 | 0.12 | other | other |  term, concept |  **Term bag |  e.g. |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  1. |  1. |  21 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.25 | 0.38 | additive_support_lm:0.05 | 0.00 | other | other |  \n\nQuestion: |  flower, flower |  \n\nQuestion: |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  (Include in |  9, |  1. |
| back_front:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  Geometry. Term |  92 |  Geometry. So |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other | 1.1 | 1.1 |  6. |
| back_front:neutral:label_colon:plant | 0.12 | 0.00 | 0.12 | additive_support:0.05 | 0.00 | other | other |  Geometry. Sub |  Geometry.\n\nOkay |  Geometry. Sub |
| back_front:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  1. | 12. |   I'm |
| back_front:neutral:multiple_choice:container | 0.75 | 0.75 | 1.00 | additive_support_lm:0.5 | 0.25 | canonical | option_like |  plant.\n\nWhat |  ?\n\n plant time |  ?\n\n plant time |
| back_front:neutral:multiple_choice:number | 0.88 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  plant.\n\nTerm |  number. What |  number.\n\nBut |
| back_front:neutral:multiple_choice:plant | 0.62 | 1.00 | 1.00 | additive_support_lm:0.05 | 0.12 | canonical | canonical |  plant.\n\nWait |  plant.\n\nExplanation |  plant.\n\nWait |
| back_front:neutral:multiple_choice:time | 0.62 | 0.12 | 1.00 | additive_support_lm:0.05 | 0.00 | option_like | option_like |  container.\n\nContext |  number.\n\nExplanation |  container.\n\nExplanation |
| back_front:short:answer_one_word:container | 0.12 | 0.00 | 0.12 | additive_support:0.05 | 0.12 | other | other |   (A |  12 |  "bottle |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.25 | additive_support_lm:0.5 | 0.25 | other | other |   the number |   (2 |   (1 |
| back_front:short:answer_one_word:plant | 0.00 | 0.00 | 0.38 | additive_support_lm:0.2 | 0.00 | other | other |  the word tree |  flower, and |  1. |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |   (A |  "Noon |  12 |
| back_front:short:label_colon:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other | Word, where | Word.\n\nBut | Geometry.\n\n9 |
| back_front:short:label_colon:number | 0.50 | 0.00 | 0.88 | additive_support_lm:0.05 | 0.00 | other | canonical |  Number; Quantity |  Number.6 |  quantity.3 |
| back_front:short:label_colon:plant | 0.25 | 0.00 | 0.62 | additive_support_lm:0.1 | 0.00 | other | canonical | Tree.<br | Plant.\n\nFl | Botany.\n\n |
| back_front:short:label_colon:time | 0.00 | 0.00 | 0.12 | additive_support:0.05 | 0.00 | other | other | Word.\n\nWhat |  time.\n\nWhich | Word.\n\n9 |
| back_front:short:multiple_choice:container | 1.00 | 0.12 | 1.00 | additive_support:0.2 | 0.00 | canonical | option_like | quia全是 ideas | zza-- | zza-zza |
| back_front:short:multiple_choice:number | 0.88 | 0.75 | 1.00 | additive_support_lm:0.05 | 0.00 | canonical | option_like |  time.\n\nOkay |  time.\n\nWait |  time.\n\nWait |
| back_front:short:multiple_choice:plant | 0.88 | 0.75 | 1.00 | additive_support_lm:0.1 | 0.00 | canonical | canonical |  tree refers to |  plant.\n\n</think> |  plant.\n\n</think> |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  time.\n\nThe |  time.\n\nYes |  time.\n\nYes |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  1. |  1. |   (1 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  concrete, abstract |  concrete, abstract |  concrete, abstract |
| front_back:long:answer_one_word:plant | 0.38 | 0.25 | 0.38 | additive_support_lm:0.5 | 0.00 | other | other |  either " concrete |  either ' concrete |  either "abstract |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.25 | additive_support:0.2 | 0.25 | other | other |  'abstract' |  'year' |  '','',' |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  abstract.\n\nWait |  abstract.\n\nWait |  abstract.\n\nWait |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.25 | additive_support_lm:0.2 | 0.00 | other | other |  abstract.\n\nWait |  abstract.\n\nWait |  abstract quantity.\n\n |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.12 | additive_support:0.5 | 0.00 | other | other |  abstract.\n\nM |  abstract.\n\nAl |  abstract.\n\nB |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  abstract.\n\nWait |  abstract.\n\nWait |  abstract.\n\nExplanation |
| front_back:long:multiple_choice:container | 1.00 | 0.00 | 1.00 | additive_support_lm:0.05 | 0.00 | canonical | canonical |  container.\n\nBut |  container.\n\nWait |  container.\n\nBut |
| front_back:long:multiple_choice:number | 0.50 | 1.00 | 1.00 | additive_support_lm:0.5 | 0.00 | other | option_like |  number.\n\nBut |  number.\n\nBut |  container.\n\nOkay |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  plant.\n\nOkay |  plant.\n\nOkay |  plant.\n\nOkay |
| front_back:long:multiple_choice:time | 1.00 | 0.88 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  time.\n\nOkay |  time.\n\nOkay |  container.\n\nWhy |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  \n\nQuestion: |  \n\nQuestion: |  chest.\n\nThe |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  **Question**: |  **Question: |  \n\nThe term |
| front_back:neutral:answer_one_word:plant | 0.12 | 0.00 | 0.38 | additive_support_lm:0.05 | 0.00 | other | other |  \n\nQuestion: |  algae.\n\nQuestion |  tree, animal |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  \n\nThe term |  \n\nQuestion: |  'term', |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  Geometry. The |  91 |  Geometry.\n\n<think> |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  1. |   (Options | 1,  |
| front_back:neutral:label_colon:plant | 0.12 | 0.00 | 0.12 | additive_support:0.05 | 0.00 | other | other |  62 |  Algae. |  Geometry, Algebra |
| front_back:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  9. |  9, |  1. |
| front_back:neutral:multiple_choice:container | 0.62 | 0.50 | 1.00 | additive_support_lm:0.05 | 0.00 | canonical | canonical |  container.\n\nWait |  container.\n\nWait |  container.\n\nWait |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  number.\n\nWhat |  number.\n\nWait |  number.\n\nWait |
| front_back:neutral:multiple_choice:plant | 0.88 | 0.25 | 1.00 | additive_support_lm:0.1 | 0.00 | canonical | canonical |  plant.\n\nExplanation |  container.\n\nWait |  plant.\n\nExplanation |
| front_back:neutral:multiple_choice:time | 0.75 | 1.00 | 1.00 | additive_support_lm:0.05 | 0.00 | canonical | option_like |  period. I |  year. I |  container. I |
| front_back:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  \textbf |  case, and |  12 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | other |  <<= !$ tape |     |  1. |
| front_back:short:answer_one_word:plant | 0.00 | 0.00 | 0.38 | additive_support_lm:0.1 | 0.00 | other | other |  moss refers to |  plant, animal |  A) plant |
| front_back:short:answer_one_word:time | 0.25 | 0.00 | 0.00 | additive_support:0.05 | 0.00 | other | format_only |     |    \\ |  1. |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.12 | additive_support_lm:0.1 | 0.00 | other | other | container. and | Case.\n\nCase | Word.\n\n Category |
| front_back:short:label_colon:number | 0.12 | 0.00 | 0.62 | additive_support_lm:0.2 | 0.00 | other | synonym |  quantity.9 |  number.\n\nBut |  Math. What |
| front_back:short:label_colon:plant | 0.25 | 0.62 | 0.62 | additive_support_lm:0.2 | 0.00 | other | canonical | Plant.\n\nM | Mathematics..."\n\n | Botany<think> |
| front_back:short:label_colon:time | 0.00 | 0.00 | 0.38 | additive_support_lm:0.5 | 0.00 | other | canonical |  time category  |   (  |  word origin.\n\n |
| front_back:short:multiple_choice:container | 1.00 | 0.88 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  container.\n\n</think> |  container.\n\nCase |  container.\n\n</think> |
| front_back:short:multiple_choice:number | 0.75 | 0.25 | 0.75 | additive_support:0.05 | 0.00 | option_like | option_like |  number.\n\n</think> |  number.\n\n</think> |  container. Wait |
| front_back:short:multiple_choice:plant | 1.00 | 0.62 | 1.00 | additive_support:0.05 | 0.00 | canonical | canonical |  plant.\n\n</think> |  plant.\n\n</think> |  plant.\n\n</think> |
| front_back:short:multiple_choice:time | 1.00 | 0.38 | 0.75 | additive_support_lm:0.05 | 0.00 | canonical | option_like |  1. |  year refers to |  container. Is |
