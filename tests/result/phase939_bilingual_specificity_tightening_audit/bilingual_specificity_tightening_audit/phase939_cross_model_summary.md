# Phase 939 bilingual specificity tightening audit

## Evidence

- bilingual_specific_semantic_transfer_retained: 2
- partial_specific_semantic_transfer_retained: 1

## Condition Rows

| model | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | target_direction | 1.0 | 912 | 0.31023848684210525 | 0.4624794407894737 | 495 | 64 |
| qwen3 | template_subtracted | 1.0 | 912 | 0.3115919682017544 | 0.46023506030701755 | 493 | 66 |
| qwen3 | wrong_mean_subtracted | 1.0 | 912 | 0.2719298245614035 | 0.31078673245614036 | 499 | 42 |
| qwen3 | specific_direction | 1.0 | 912 | 0.27309484649122806 | 0.30838815789473684 | 496 | 45 |
| qwen3 | template_shift_same_norm | 1.0 | 912 | 0.5189144736842105 | 0.2861670778508772 | 531 | 32 |
| qwen3 | baseline | None | 912 | 0.0 | 0.0 | 0 | 0 |
| qwen3 | random_same_norm | 1.0 | 912 | -0.041015625 | -0.03769188596491228 | 288 | 6 |
| qwen3 | wrong_mean_direction | 1.0 | 912 | -0.24753289473684212 | -0.4036458333333333 | 269 | 4 |
| qwen3 | negative_target_direction | 1.0 | 912 | -0.5816543311403509 | -0.7162828947368421 | 168 | 2 |
| qwen3 | wrong_label_direction | 1.0 | 912 | -0.07146038925438597 | -0.7343578673245614 | 355 | 10 |
| glm4 | template_subtracted | 1.0 | 765 | 4.899981809129902 | 2.4554001353145423 | 574 | 262 |
| glm4 | wrong_mean_subtracted | 1.0 | 765 | 4.964400467218137 | 2.4009647543913397 | 573 | 254 |
| glm4 | specific_direction | 1.0 | 765 | 4.965112783394608 | 2.399420126123366 | 566 | 254 |
| glm4 | target_direction | 1.0 | 765 | 4.9952756395526965 | 2.2863359438827615 | 572 | 250 |
| glm4 | template_shift_same_norm | 1.0 | 765 | 4.756602966707517 | 2.170129314746732 | 569 | 242 |
| glm4 | random_same_norm | 1.0 | 765 | 4.375563597834967 | 1.9768090740527982 | 546 | 229 |
| glm4 | wrong_label_direction | 1.0 | 765 | 5.004369638480392 | 1.9000338286356209 | 549 | 215 |
| glm4 | wrong_mean_direction | 1.0 | 765 | 4.176855787888072 | 1.8808852251838235 | 544 | 230 |
| glm4 | negative_target_direction | 1.0 | 765 | 3.921994437423407 | 1.6942457012101715 | 536 | 214 |
| glm4 | baseline | None | 765 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | target_direction | 1.0 | 912 | 0.23152562191611842 | 0.2305023293746145 | 470 | 31 |
| deepseek7b | template_subtracted | 1.0 | 912 | 0.2323925620631168 | 0.22361037605687192 | 457 | 32 |
| deepseek7b | template_shift_same_norm | 1.0 | 912 | 0.483207434938665 | 0.2125598673234906 | 538 | 27 |
| deepseek7b | specific_direction | 1.0 | 912 | 0.1919708251953125 | 0.07002746849729304 | 364 | 18 |
| deepseek7b | wrong_mean_subtracted | 1.0 | 912 | 0.1886425352933114 | 0.06821187337239583 | 371 | 18 |
| deepseek7b | baseline | None | 912 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | random_same_norm | 1.0 | 912 | 0.1403142025596217 | -0.007800871865791187 | 392 | 9 |
| deepseek7b | wrong_label_direction | 1.0 | 912 | -0.03445407800507127 | -0.15236857899448328 | 322 | 10 |
| deepseek7b | wrong_mean_direction | 1.0 | 912 | 0.02968262789542215 | -0.1961963134899474 | 349 | 5 |
| deepseek7b | negative_target_direction | 1.0 | 912 | -0.030526412160773026 | -0.27575996465850294 | 294 | 4 |

## Language Pair Rows

| model | pair | condition | alpha | rows | mean margin delta | rank improved | new winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | en->zh | template_shift_same_norm | 1.0 | 304 | 0.7119140625 | 278 | 14 |
| qwen3 | en->en | target_direction | 1.0 | 152 | 0.6492598684210527 | 70 | 16 |
| qwen3 | en->en | template_subtracted | 1.0 | 152 | 0.6373355263157895 | 71 | 16 |
| qwen3 | zh->en | template_subtracted | 1.0 | 304 | 0.4569284539473684 | 114 | 23 |
| qwen3 | zh->en | target_direction | 1.0 | 304 | 0.45682565789473684 | 123 | 23 |
| qwen3 | en->zh | template_subtracted | 1.0 | 304 | 0.4350328947368421 | 204 | 20 |
| qwen3 | en->zh | target_direction | 1.0 | 304 | 0.43277138157894735 | 199 | 18 |
| qwen3 | en->en | wrong_mean_subtracted | 1.0 | 152 | 0.38260690789473684 | 55 | 9 |
| qwen3 | en->en | specific_direction | 1.0 | 152 | 0.3784950657894737 | 56 | 9 |
| qwen3 | zh->zh | target_direction | 1.0 | 152 | 0.3464226973684211 | 103 | 7 |
| qwen3 | zh->zh | template_subtracted | 1.0 | 152 | 0.34015213815789475 | 104 | 7 |
| qwen3 | zh->en | wrong_mean_subtracted | 1.0 | 304 | 0.32113486842105265 | 109 | 17 |
| qwen3 | zh->en | specific_direction | 1.0 | 304 | 0.32010690789473684 | 107 | 18 |
| qwen3 | en->zh | wrong_mean_subtracted | 1.0 | 304 | 0.30535567434210525 | 224 | 11 |
| qwen3 | en->zh | specific_direction | 1.0 | 304 | 0.3012438322368421 | 223 | 13 |
| qwen3 | zh->zh | specific_direction | 1.0 | 152 | 0.22913240131578946 | 110 | 5 |
| qwen3 | zh->zh | wrong_mean_subtracted | 1.0 | 152 | 0.22913240131578946 | 111 | 5 |
| qwen3 | en->en | template_shift_same_norm | 1.0 | 152 | 0.11060855263157894 | 60 | 8 |
| qwen3 | zh->en | template_shift_same_norm | 1.0 | 304 | 0.07966694078947369 | 91 | 9 |
| qwen3 | zh->zh | template_shift_same_norm | 1.0 | 152 | 0.023231907894736843 | 102 | 1 |
| qwen3 | en->en | baseline | None | 152 | 0.0 | 0 | 0 |
| qwen3 | en->zh | baseline | None | 304 | 0.0 | 0 | 0 |
| qwen3 | zh->en | baseline | None | 304 | 0.0 | 0 | 0 |
| qwen3 | zh->zh | baseline | None | 152 | 0.0 | 0 | 0 |
| qwen3 | zh->en | random_same_norm | 1.0 | 304 | -0.0078125 | 59 | 2 |
| qwen3 | zh->zh | random_same_norm | 1.0 | 152 | -0.01665296052631579 | 58 | 0 |
| qwen3 | en->en | random_same_norm | 1.0 | 152 | -0.026110197368421052 | 40 | 2 |
| qwen3 | en->zh | random_same_norm | 1.0 | 304 | -0.08388157894736842 | 131 | 2 |
| qwen3 | en->zh | wrong_mean_direction | 1.0 | 304 | -0.265625 | 136 | 1 |
| qwen3 | zh->zh | wrong_mean_direction | 1.0 | 152 | -0.32195723684210525 | 55 | 0 |
| qwen3 | zh->en | wrong_mean_direction | 1.0 | 304 | -0.44233141447368424 | 53 | 2 |
| qwen3 | zh->zh | negative_target_direction | 1.0 | 152 | -0.606188322368421 | 37 | 0 |
| qwen3 | en->zh | negative_target_direction | 1.0 | 304 | -0.6138466282894737 | 75 | 0 |
| qwen3 | zh->zh | wrong_label_direction | 1.0 | 152 | -0.6297286184210527 | 93 | 2 |
| qwen3 | en->zh | wrong_label_direction | 1.0 | 304 | -0.6608244243421053 | 176 | 5 |
| qwen3 | en->en | wrong_mean_direction | 1.0 | 152 | -0.6840049342105263 | 25 | 1 |
| qwen3 | zh->en | negative_target_direction | 1.0 | 304 | -0.7456825657894737 | 40 | 2 |
| qwen3 | zh->en | wrong_label_direction | 1.0 | 304 | -0.7515419407894737 | 48 | 2 |
| qwen3 | en->en | wrong_label_direction | 1.0 | 152 | -0.9516858552631579 | 38 | 1 |
| qwen3 | en->en | negative_target_direction | 1.0 | 152 | -0.9724506578947368 | 16 | 0 |
| glm4 | zh->en | target_direction | 1.0 | 304 | 4.323055869654605 | 297 | 151 |
| glm4 | zh->en | template_subtracted | 1.0 | 304 | 4.3111620451274675 | 297 | 148 |
| glm4 | zh->en | specific_direction | 1.0 | 304 | 4.154793187191612 | 298 | 146 |
| glm4 | zh->en | wrong_mean_subtracted | 1.0 | 304 | 4.153148450349507 | 298 | 145 |
| glm4 | en->en | template_subtracted | 1.0 | 103 | 4.116315230582524 | 103 | 59 |
| glm4 | en->en | specific_direction | 1.0 | 103 | 4.071746529884709 | 103 | 53 |
| glm4 | en->en | wrong_mean_subtracted | 1.0 | 103 | 4.066285364836165 | 103 | 52 |
| glm4 | zh->en | template_shift_same_norm | 1.0 | 304 | 4.000629625822368 | 298 | 145 |
| glm4 | zh->en | random_same_norm | 1.0 | 304 | 3.929311651932566 | 297 | 136 |
| glm4 | en->en | wrong_label_direction | 1.0 | 103 | 3.9083548240291264 | 103 | 45 |
| glm4 | zh->en | wrong_label_direction | 1.0 | 304 | 3.6566354851973686 | 301 | 128 |
| glm4 | zh->en | wrong_mean_direction | 1.0 | 304 | 3.5324377762643913 | 298 | 127 |
| glm4 | en->en | wrong_mean_direction | 1.0 | 103 | 3.5200242718446604 | 102 | 49 |
| glm4 | en->en | target_direction | 1.0 | 103 | 3.491922026699029 | 98 | 48 |
| glm4 | en->en | negative_target_direction | 1.0 | 103 | 3.4641800667475726 | 102 | 48 |
| glm4 | en->en | random_same_norm | 1.0 | 103 | 3.4254778519417477 | 103 | 49 |
| glm4 | zh->en | negative_target_direction | 1.0 | 304 | 3.2793725666246916 | 298 | 119 |
| glm4 | en->en | template_shift_same_norm | 1.0 | 103 | 3.0318188713592233 | 94 | 47 |
| glm4 | zh->zh | template_subtracted | 1.0 | 152 | 0.8086965460526315 | 97 | 30 |
| glm4 | zh->zh | target_direction | 1.0 | 152 | 0.8086515727796053 | 92 | 31 |
| glm4 | zh->zh | wrong_mean_subtracted | 1.0 | 152 | 0.6713224712171053 | 84 | 29 |
| glm4 | zh->zh | specific_direction | 1.0 | 152 | 0.6623021175986842 | 82 | 28 |
| glm4 | zh->zh | template_shift_same_norm | 1.0 | 152 | 0.6109040912828947 | 84 | 25 |
| glm4 | zh->zh | random_same_norm | 1.0 | 152 | 0.3091826187936883 | 73 | 20 |
| glm4 | en->zh | wrong_mean_subtracted | 1.0 | 206 | 0.25879854368932037 | 88 | 28 |
| glm4 | en->zh | specific_direction | 1.0 | 206 | 0.2545604520631068 | 83 | 27 |
| glm4 | en->zh | template_shift_same_norm | 1.0 | 206 | 0.1884599704186893 | 93 | 25 |
| glm4 | en->zh | template_subtracted | 1.0 | 206 | 0.10138591284890777 | 77 | 25 |
| glm4 | zh->zh | wrong_mean_direction | 1.0 | 152 | 0.03554494757401316 | 68 | 21 |
| glm4 | en->en | baseline | None | 103 | 0.0 | 0 | 0 |
| glm4 | en->zh | baseline | None | 206 | 0.0 | 0 | 0 |
| glm4 | zh->en | baseline | None | 304 | 0.0 | 0 | 0 |
| glm4 | zh->zh | baseline | None | 152 | 0.0 | 0 | 0 |
| glm4 | en->zh | negative_target_direction | 1.0 | 206 | -0.008738101107402913 | 77 | 33 |
| glm4 | en->zh | wrong_mean_direction | 1.0 | 206 | -0.014316595873786407 | 76 | 33 |
| glm4 | en->zh | wrong_label_direction | 1.0 | 206 | -0.1681299302184466 | 82 | 23 |
| glm4 | zh->zh | wrong_label_direction | 1.0 | 152 | -0.17116506476151316 | 63 | 19 |
| glm4 | en->zh | target_direction | 1.0 | 206 | -0.23177182095722088 | 85 | 20 |
| glm4 | zh->zh | negative_target_direction | 1.0 | 152 | -0.36738024259868424 | 59 | 14 |
| glm4 | en->zh | random_same_norm | 1.0 | 206 | -0.39840668613470875 | 73 | 24 |
| deepseek7b | en->zh | template_shift_same_norm | 1.0 | 304 | 0.38604425129137543 | 245 | 12 |
| deepseek7b | en->en | target_direction | 1.0 | 152 | 0.33223684210526316 | 77 | 6 |
| deepseek7b | en->en | template_subtracted | 1.0 | 152 | 0.31863563939144735 | 76 | 5 |
| deepseek7b | zh->en | target_direction | 1.0 | 304 | 0.24861546566611842 | 150 | 8 |
| deepseek7b | zh->en | template_subtracted | 1.0 | 304 | 0.23495322779605263 | 145 | 9 |
| deepseek7b | en->zh | template_subtracted | 1.0 | 304 | 0.21000199568899056 | 155 | 11 |
| deepseek7b | en->zh | target_direction | 1.0 | 304 | 0.2087009831478721 | 160 | 10 |
| deepseek7b | zh->en | template_shift_same_norm | 1.0 | 304 | 0.2021580746299342 | 134 | 8 |
| deepseek7b | zh->zh | target_direction | 1.0 | 152 | 0.13614423651444285 | 83 | 7 |
| deepseek7b | zh->zh | template_subtracted | 1.0 | 152 | 0.13311616997969777 | 81 | 7 |
| deepseek7b | zh->en | wrong_mean_subtracted | 1.0 | 304 | 0.09228515625 | 126 | 6 |
| deepseek7b | zh->zh | template_shift_same_norm | 1.0 | 152 | 0.09117738824141652 | 89 | 4 |
| deepseek7b | zh->en | specific_direction | 1.0 | 304 | 0.08539782072368421 | 120 | 5 |
| deepseek7b | en->en | specific_direction | 1.0 | 152 | 0.07953844572368421 | 66 | 1 |
| deepseek7b | zh->zh | wrong_mean_subtracted | 1.0 | 152 | 0.06765024285567434 | 55 | 6 |
| deepseek7b | zh->zh | specific_direction | 1.0 | 152 | 0.06594125848067434 | 58 | 4 |
| deepseek7b | en->zh | wrong_mean_subtracted | 1.0 | 304 | 0.052941974840666116 | 123 | 5 |
| deepseek7b | en->zh | specific_direction | 1.0 | 304 | 0.051944732666015625 | 120 | 8 |
| deepseek7b | en->en | wrong_mean_subtracted | 1.0 | 152 | 0.05116673519736842 | 67 | 1 |
| deepseek7b | zh->zh | wrong_label_direction | 1.0 | 152 | 0.027567311337119656 | 69 | 1 |
| deepseek7b | en->zh | random_same_norm | 1.0 | 304 | 0.017779802021227385 | 140 | 4 |
| deepseek7b | en->en | template_shift_same_norm | 1.0 | 152 | 0.007777163856907895 | 70 | 3 |
| deepseek7b | en->en | baseline | None | 152 | 0.0 | 0 | 0 |
| deepseek7b | en->zh | baseline | None | 304 | 0.0 | 0 | 0 |
| deepseek7b | zh->en | baseline | None | 304 | 0.0 | 0 | 0 |
| deepseek7b | zh->zh | baseline | None | 152 | 0.0 | 0 | 0 |
| deepseek7b | zh->zh | random_same_norm | 1.0 | 152 | -0.0026914696944387337 | 78 | 0 |
| deepseek7b | en->en | random_same_norm | 1.0 | 152 | -0.019820363898026317 | 57 | 2 |
| deepseek7b | zh->en | random_same_norm | 1.0 | 304 | -0.02992650082236842 | 117 | 3 |
| deepseek7b | zh->zh | wrong_mean_direction | 1.0 | 152 | -0.06784649899131373 | 71 | 0 |
| deepseek7b | zh->zh | negative_target_direction | 1.0 | 152 | -0.10834332516318873 | 58 | 0 |
| deepseek7b | zh->en | wrong_label_direction | 1.0 | 304 | -0.11445055509868421 | 114 | 3 |
| deepseek7b | zh->en | wrong_mean_direction | 1.0 | 304 | -0.14860454358552633 | 93 | 1 |
| deepseek7b | zh->en | negative_target_direction | 1.0 | 304 | -0.18094675164473684 | 87 | 1 |
| deepseek7b | en->zh | wrong_label_direction | 1.0 | 304 | -0.1897036401849044 | 93 | 2 |
| deepseek7b | en->zh | wrong_mean_direction | 1.0 | 304 | -0.2039191346419485 | 145 | 3 |
| deepseek7b | en->en | wrong_label_direction | 1.0 | 152 | -0.3334703947368421 | 46 | 4 |
| deepseek7b | en->zh | negative_target_direction | 1.0 | 304 | -0.34299669767680924 | 109 | 2 |
| deepseek7b | en->en | wrong_mean_direction | 1.0 | 152 | -0.4042840254934211 | 40 | 1 |
| deepseek7b | en->en | negative_target_direction | 1.0 | 152 | -0.49832956414473684 | 40 | 1 |

## Top Specificity Rows

| model | relation | pair | condition | alpha | rows | margin | control best | specificity gain |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| glm4 | color | zh->zh | target_direction | 1.0 | 54 | 0.97265625 | -0.01837384259259259 | 0.9910300925925926 |
| glm4 | color | zh->zh | template_subtracted | 1.0 | 54 | 0.9321469907407407 | -0.01837384259259259 | 0.9505208333333333 |
| qwen3 | function | en->en | target_direction | 1.0 | 38 | 1.0921052631578947 | 0.18914473684210525 | 0.9029605263157894 |
| qwen3 | function | en->en | template_subtracted | 1.0 | 38 | 1.0625 | 0.18914473684210525 | 0.8733552631578947 |
| qwen3 | color | en->zh | template_subtracted | 1.0 | 108 | 0.9163773148148148 | 0.07609953703703703 | 0.8402777777777777 |
| qwen3 | color | en->zh | target_direction | 1.0 | 108 | 0.9123263888888888 | 0.07609953703703703 | 0.8362268518518519 |
| glm4 | color | zh->en | target_direction | 1.0 | 108 | 4.720811631944445 | 3.9258174189814814 | 0.7949942129629632 |
| glm4 | color | zh->en | template_subtracted | 1.0 | 108 | 4.708948206018518 | 3.9258174189814814 | 0.7831307870370368 |
| qwen3 | function | en->en | wrong_mean_subtracted | 1.0 | 38 | 0.9194078947368421 | 0.18914473684210525 | 0.730263157894737 |
| qwen3 | function | zh->en | template_subtracted | 1.0 | 76 | 0.9547697368421053 | 0.23684210526315788 | 0.7179276315789475 |
| qwen3 | function | en->en | specific_direction | 1.0 | 38 | 0.90625 | 0.18914473684210525 | 0.7171052631578947 |
| qwen3 | function | zh->en | target_direction | 1.0 | 76 | 0.9490131578947368 | 0.23684210526315788 | 0.712171052631579 |
| qwen3 | color | en->en | target_direction | 1.0 | 54 | 0.7210648148148148 | 0.013888888888888888 | 0.7071759259259259 |
| qwen3 | color | en->en | template_subtracted | 1.0 | 54 | 0.7210648148148148 | 0.013888888888888888 | 0.7071759259259259 |
| glm4 | color | zh->zh | wrong_mean_subtracted | 1.0 | 54 | 0.6112557870370371 | -0.01837384259259259 | 0.6296296296296297 |
| glm4 | color | zh->zh | specific_direction | 1.0 | 54 | 0.5956307870370371 | -0.01837384259259259 | 0.6140046296296297 |
| deepseek7b | category | en->en | target_direction | 1.0 | 60 | 0.6364583333333333 | 0.0290771484375 | 0.6073811848958334 |
| qwen3 | function | zh->en | specific_direction | 1.0 | 76 | 0.834703947368421 | 0.23684210526315788 | 0.5978618421052632 |
| qwen3 | function | zh->en | wrong_mean_subtracted | 1.0 | 76 | 0.8322368421052632 | 0.23684210526315788 | 0.5953947368421053 |
| deepseek7b | category | en->en | template_subtracted | 1.0 | 60 | 0.6025227864583333 | 0.0290771484375 | 0.5734456380208334 |
| qwen3 | color | zh->zh | template_subtracted | 1.0 | 54 | 0.6197916666666666 | 0.052083333333333336 | 0.5677083333333333 |
| qwen3 | color | zh->zh | target_direction | 1.0 | 54 | 0.6186342592592593 | 0.052083333333333336 | 0.5665509259259259 |
| qwen3 | color | en->zh | wrong_mean_subtracted | 1.0 | 108 | 0.5954861111111112 | 0.07609953703703703 | 0.5193865740740742 |
| qwen3 | color | en->zh | specific_direction | 1.0 | 108 | 0.5911458333333334 | 0.07609953703703703 | 0.5150462962962963 |
| glm4 | color | zh->en | specific_direction | 1.0 | 108 | 4.361870659722222 | 3.9258174189814814 | 0.4360532407407409 |
| glm4 | color | en->zh | wrong_mean_subtracted | 1.0 | 108 | -0.08409288194444445 | -0.5185366030092593 | 0.4344437210648149 |
| qwen3 | color | zh->en | template_subtracted | 1.0 | 108 | 0.5179398148148148 | 0.08391203703703703 | 0.43402777777777773 |
| glm4 | color | zh->en | wrong_mean_subtracted | 1.0 | 108 | 4.35767505787037 | 3.9258174189814814 | 0.43185763888888884 |
| qwen3 | color | zh->en | target_direction | 1.0 | 108 | 0.515625 | 0.08391203703703703 | 0.43171296296296297 |
| glm4 | color | en->zh | specific_direction | 1.0 | 108 | -0.10033275462962964 | -0.5185366030092593 | 0.41820384837962965 |
| qwen3 | function | zh->zh | target_direction | 1.0 | 38 | 0.43667763157894735 | 0.029605263157894735 | 0.4070723684210526 |
| qwen3 | function | zh->zh | template_subtracted | 1.0 | 38 | 0.43050986842105265 | 0.029605263157894735 | 0.4009046052631579 |
| qwen3 | color | en->en | wrong_mean_subtracted | 1.0 | 54 | 0.41203703703703703 | 0.013888888888888888 | 0.39814814814814814 |
| qwen3 | color | en->en | specific_direction | 1.0 | 54 | 0.4097222222222222 | 0.013888888888888888 | 0.3958333333333333 |
| qwen3 | function | zh->zh | wrong_mean_subtracted | 1.0 | 38 | 0.37870065789473684 | 0.029605263157894735 | 0.3490953947368421 |
| qwen3 | color | zh->zh | specific_direction | 1.0 | 54 | 0.3975694444444444 | 0.052083333333333336 | 0.3454861111111111 |
| qwen3 | function | zh->zh | specific_direction | 1.0 | 38 | 0.36800986842105265 | 0.029605263157894735 | 0.3384046052631579 |
| qwen3 | color | zh->zh | wrong_mean_subtracted | 1.0 | 54 | 0.3900462962962963 | 0.052083333333333336 | 0.33796296296296297 |
| glm4 | function | en->en | target_direction | 1.0 | 19 | 3.823190789473684 | 3.5945723684210527 | 0.22861842105263142 |
| qwen3 | color | zh->en | wrong_mean_subtracted | 1.0 | 108 | 0.3003472222222222 | 0.08391203703703703 | 0.21643518518518517 |
| qwen3 | color | zh->en | specific_direction | 1.0 | 108 | 0.2957175925925926 | 0.08391203703703703 | 0.21180555555555558 |
| deepseek7b | function | en->en | specific_direction | 1.0 | 38 | 0.18092105263157895 | -0.029605263157894735 | 0.2105263157894737 |
| glm4 | function | en->en | wrong_mean_subtracted | 1.0 | 19 | 3.801809210526316 | 3.5945723684210527 | 0.20723684210526327 |
| glm4 | function | en->en | template_subtracted | 1.0 | 19 | 3.7985197368421053 | 3.5945723684210527 | 0.20394736842105265 |
| glm4 | function | en->en | specific_direction | 1.0 | 19 | 3.7820723684210527 | 3.5945723684210527 | 0.1875 |
| glm4 | function | zh->en | specific_direction | 1.0 | 76 | 4.0327919407894735 | 3.8708881578947367 | 0.16190378289473673 |
| glm4 | function | zh->en | wrong_mean_subtracted | 1.0 | 76 | 4.032175164473684 | 3.8708881578947367 | 0.16128700657894735 |
| qwen3 | category | en->en | target_direction | 1.0 | 60 | 0.30416666666666664 | 0.14791666666666667 | 0.15624999999999997 |
| qwen3 | category | en->en | template_subtracted | 1.0 | 60 | 0.29270833333333335 | 0.14791666666666667 | 0.14479166666666668 |
| deepseek7b | function | zh->zh | target_direction | 1.0 | 38 | 0.38464034231085525 | 0.24318012438322367 | 0.14146021792763158 |
| deepseek7b | function | zh->zh | template_subtracted | 1.0 | 38 | 0.37676198858963816 | 0.24318012438322367 | 0.1335818642064145 |
| deepseek7b | function | en->en | target_direction | 1.0 | 38 | 0.10361842105263158 | -0.029605263157894735 | 0.13322368421052633 |
| deepseek7b | function | en->en | wrong_mean_subtracted | 1.0 | 38 | 0.10361842105263158 | -0.029605263157894735 | 0.13322368421052633 |
| glm4 | color | en->zh | template_subtracted | 1.0 | 108 | -0.39345748336226855 | -0.5185366030092593 | 0.12507911964699076 |
| deepseek7b | function | en->en | template_subtracted | 1.0 | 38 | 0.09539473684210527 | -0.029605263157894735 | 0.125 |
| glm4 | function | zh->en | target_direction | 1.0 | 76 | 3.993112664473684 | 3.8708881578947367 | 0.12222450657894735 |
| glm4 | function | zh->en | template_subtracted | 1.0 | 76 | 3.989206414473684 | 3.8708881578947367 | 0.11831825657894735 |
| glm4 | color | en->en | specific_direction | 1.0 | 54 | 4.008716724537037 | 3.892107928240741 | 0.11660879629629628 |
| deepseek7b | function | zh->en | target_direction | 1.0 | 76 | 0.4917763157894737 | 0.3873355263157895 | 0.10444078947368418 |
| glm4 | color | en->en | wrong_mean_subtracted | 1.0 | 54 | 3.991355613425926 | 3.892107928240741 | 0.09924768518518512 |
| qwen3 | category | zh->en | target_direction | 1.0 | 120 | 0.0921875 | 0.009895833333333333 | 0.08229166666666668 |
| deepseek7b | function | zh->en | template_subtracted | 1.0 | 76 | 0.46463815789473684 | 0.3873355263157895 | 0.07730263157894735 |
| qwen3 | category | zh->en | template_subtracted | 1.0 | 120 | 0.08671875 | 0.009895833333333333 | 0.07682291666666666 |
| deepseek7b | category | zh->en | target_direction | 1.0 | 120 | 0.2670654296875 | 0.2090087890625 | 0.05805664062500002 |
| deepseek7b | category | zh->en | template_subtracted | 1.0 | 120 | 0.25823567708333334 | 0.2090087890625 | 0.04922688802083333 |
| qwen3 | category | zh->zh | target_direction | 1.0 | 60 | 0.044270833333333336 | -0.0046875 | 0.04895833333333333 |
| qwen3 | category | zh->zh | template_subtracted | 1.0 | 60 | 0.03125 | -0.0046875 | 0.0359375 |
| deepseek7b | function | zh->zh | wrong_mean_subtracted | 1.0 | 38 | 0.2747385125411184 | 0.24318012438322367 | 0.031558388157894746 |
| deepseek7b | color | en->en | template_subtracted | 1.0 | 54 | 0.16030092592592593 | 0.13194444444444445 | 0.028356481481481483 |
| glm4 | color | en->en | template_subtracted | 1.0 | 54 | 3.9195963541666665 | 3.892107928240741 | 0.027488425925925597 |
| deepseek7b | color | en->en | target_direction | 1.0 | 54 | 0.1550925925925926 | 0.13194444444444445 | 0.02314814814814814 |
| deepseek7b | function | zh->zh | specific_direction | 1.0 | 38 | 0.2646131013569079 | 0.24318012438322367 | 0.021432976973684237 |
| glm4 | category | zh->en | target_direction | 1.0 | 120 | 4.174039713541666 | 4.160286458333333 | 0.013753255208333037 |
| deepseek7b | color | zh->zh | target_direction | 1.0 | 54 | 0.004050925925925926 | -0.008391203703703705 | 0.01244212962962963 |
| deepseek7b | color | zh->zh | specific_direction | 1.0 | 54 | 0.001591435185185185 | -0.008391203703703705 | 0.00998263888888889 |
| deepseek7b | color | zh->zh | template_subtracted | 1.0 | 54 | 0.00043402777777777775 | -0.008391203703703705 | 0.008825231481481483 |
| deepseek7b | color | zh->zh | wrong_mean_subtracted | 1.0 | 54 | -0.0007233796296296296 | -0.008391203703703705 | 0.007667824074074075 |
| qwen3 | category | zh->en | specific_direction | 1.0 | 120 | 0.016145833333333335 | 0.009895833333333333 | 0.006250000000000002 |
| qwen3 | category | zh->en | wrong_mean_subtracted | 1.0 | 120 | 0.016145833333333335 | 0.009895833333333333 | 0.006250000000000002 |
| deepseek7b | color | en->zh | specific_direction | 1.0 | 108 | 0.01837384259259259 | 0.018012152777777776 | 0.0003616898148148147 |
| deepseek7b | color | en->zh | wrong_mean_subtracted | 1.0 | 108 | 0.01750578703703704 | 0.018012152777777776 | -0.0005063657407407378 |
| glm4 | category | zh->en | template_subtracted | 1.0 | 120 | 4.157059733072916 | 4.160286458333333 | -0.003226725260416785 |
| qwen3 | category | zh->zh | specific_direction | 1.0 | 60 | -0.010416666666666666 | -0.0046875 | -0.005729166666666666 |
| qwen3 | category | zh->zh | wrong_mean_subtracted | 1.0 | 60 | -0.010416666666666666 | -0.0046875 | -0.005729166666666666 |
| glm4 | category | en->en | template_subtracted | 1.0 | 30 | 4.6716796875 | 4.678580729166667 | -0.0069010416666666075 |
| deepseek7b | color | zh->en | target_direction | 1.0 | 108 | 0.05700231481481482 | 0.0642361111111111 | -0.007233796296296287 |
| deepseek7b | color | en->zh | template_subtracted | 1.0 | 108 | 0.01048900462962963 | 0.018012152777777776 | -0.007523148148148147 |
| glm4 | category | en->en | target_direction | 1.0 | 30 | 4.6705078125 | 4.678580729166667 | -0.00807291666666643 |
| deepseek7b | color | en->zh | target_direction | 1.0 | 108 | 0.00853587962962963 | 0.018012152777777776 | -0.009476273148148147 |
| deepseek7b | color | en->en | specific_direction | 1.0 | 54 | 0.11921296296296297 | 0.13194444444444445 | -0.012731481481481483 |
| deepseek7b | color | zh->en | wrong_mean_subtracted | 1.0 | 108 | 0.04861111111111111 | 0.0642361111111111 | -0.015624999999999993 |
| deepseek7b | color | zh->en | template_subtracted | 1.0 | 108 | 0.047453703703703706 | 0.0642361111111111 | -0.0167824074074074 |
| deepseek7b | category | zh->zh | template_subtracted | 1.0 | 60 | 0.09822107950846354 | 0.11616134643554688 | -0.017940266927083337 |
| deepseek7b | category | zh->zh | target_direction | 1.0 | 60 | 0.09764734903971355 | 0.11616134643554688 | -0.01851399739583333 |
| deepseek7b | color | zh->en | specific_direction | 1.0 | 108 | 0.03211805555555555 | 0.0642361111111111 | -0.03211805555555555 |
| deepseek7b | color | en->en | wrong_mean_subtracted | 1.0 | 54 | 0.09375 | 0.13194444444444445 | -0.03819444444444445 |
| deepseek7b | category | en->en | specific_direction | 1.0 | 60 | -0.020377604166666667 | 0.0290771484375 | -0.049454752604166666 |
| deepseek7b | category | en->en | wrong_mean_subtracted | 1.0 | 60 | -0.020377604166666667 | 0.0290771484375 | -0.049454752604166666 |
| deepseek7b | function | zh->en | wrong_mean_subtracted | 1.0 | 76 | 0.33223684210526316 | 0.3873355263157895 | -0.05509868421052633 |
| deepseek7b | function | zh->en | specific_direction | 1.0 | 76 | 0.328125 | 0.3873355263157895 | -0.05921052631578949 |
| glm4 | category | zh->en | specific_direction | 1.0 | 120 | 4.04569091796875 | 4.160286458333333 | -0.11459554036458286 |
| glm4 | category | zh->en | wrong_mean_subtracted | 1.0 | 120 | 4.04569091796875 | 4.160286458333333 | -0.11459554036458286 |
| glm4 | category | zh->zh | template_subtracted | 1.0 | 60 | -0.01328125 | 0.10260416666666666 | -0.11588541666666666 |
| deepseek7b | category | zh->zh | specific_direction | 1.0 | 60 | -0.0019694010416666666 | 0.11616134643554688 | -0.11813074747721354 |
| deepseek7b | category | zh->zh | wrong_mean_subtracted | 1.0 | 60 | -0.0019694010416666666 | 0.11616134643554688 | -0.11813074747721354 |
| qwen3 | category | en->en | specific_direction | 1.0 | 60 | 0.016145833333333335 | 0.14791666666666667 | -0.13177083333333334 |
| qwen3 | category | en->en | wrong_mean_subtracted | 1.0 | 60 | 0.016145833333333335 | 0.14791666666666667 | -0.13177083333333334 |
| glm4 | category | zh->zh | target_direction | 1.0 | 60 | -0.05974934895833333 | 0.10260416666666666 | -0.162353515625 |
| deepseek7b | category | zh->en | specific_direction | 1.0 | 120 | -0.020377604166666667 | 0.2090087890625 | -0.22938639322916668 |
| deepseek7b | category | zh->en | wrong_mean_subtracted | 1.0 | 120 | -0.020377604166666667 | 0.2090087890625 | -0.22938639322916668 |
| glm4 | function | zh->zh | wrong_mean_subtracted | 1.0 | 38 | 2.0759662828947367 | 2.3077199835526314 | -0.2317537006578947 |
| deepseek7b | function | en->zh | target_direction | 1.0 | 76 | 0.21041548879523025 | 0.45585070158305924 | -0.24543521278782898 |
| glm4 | function | zh->zh | specific_direction | 1.0 | 38 | 2.062088815789474 | 2.3077199835526314 | -0.24563116776315752 |
| deepseek7b | function | en->zh | wrong_mean_subtracted | 1.0 | 76 | 0.19000083521792763 | 0.45585070158305924 | -0.26584986636513164 |
| glm4 | category | zh->zh | specific_direction | 1.0 | 60 | -0.16422526041666666 | 0.10260416666666666 | -0.26682942708333335 |
| glm4 | category | zh->zh | wrong_mean_subtracted | 1.0 | 60 | -0.16422526041666666 | 0.10260416666666666 | -0.26682942708333335 |
| deepseek7b | category | en->zh | template_subtracted | 1.0 | 120 | 0.4053571065266927 | 0.6730623881022135 | -0.2677052815755208 |
| deepseek7b | function | en->zh | template_subtracted | 1.0 | 76 | 0.18506501850328946 | 0.45585070158305924 | -0.2707856830797698 |
| deepseek7b | function | en->zh | specific_direction | 1.0 | 76 | 0.1847783138877467 | 0.45585070158305924 | -0.2710723876953125 |
| glm4 | function | en->zh | specific_direction | 1.0 | 38 | 1.9244449013157894 | 2.1991159539473686 | -0.2746710526315792 |
| deepseek7b | category | en->zh | target_direction | 1.0 | 120 | 0.3877637227376302 | 0.6730623881022135 | -0.2852986653645833 |
| glm4 | function | en->zh | template_subtracted | 1.0 | 38 | 1.907483552631579 | 2.1991159539473686 | -0.2916324013157896 |
| glm4 | function | en->zh | target_direction | 1.0 | 38 | 1.9019325657894737 | 2.1991159539473686 | -0.2971833881578949 |
| glm4 | function | en->zh | wrong_mean_subtracted | 1.0 | 38 | 1.9012643914473684 | 2.1991159539473686 | -0.2978515625000002 |
| glm4 | category | en->en | specific_direction | 1.0 | 30 | 4.368660481770833 | 4.678580729166667 | -0.30992024739583357 |
| glm4 | category | en->en | wrong_mean_subtracted | 1.0 | 30 | 4.368660481770833 | 4.678580729166667 | -0.30992024739583357 |
| glm4 | category | en->zh | target_direction | 1.0 | 60 | -0.14235026041666668 | 0.19925130208333333 | -0.3416015625 |
| glm4 | category | en->zh | template_subtracted | 1.0 | 60 | -0.1517578125 | 0.19925130208333333 | -0.35100911458333334 |
| glm4 | function | zh->zh | target_direction | 1.0 | 38 | 1.946751644736842 | 2.3077199835526314 | -0.3609683388157894 |
| glm4 | category | en->zh | specific_direction | 1.0 | 60 | -0.16422526041666666 | 0.19925130208333333 | -0.3634765625 |
| glm4 | category | en->zh | wrong_mean_subtracted | 1.0 | 60 | -0.16422526041666666 | 0.19925130208333333 | -0.3634765625 |
| glm4 | function | zh->zh | template_subtracted | 1.0 | 38 | 1.931126644736842 | 2.3077199835526314 | -0.3765933388157894 |
| glm4 | color | en->zh | target_direction | 1.0 | 108 | -1.0321983054832176 | -0.5185366030092593 | -0.5136617024739583 |
| qwen3 | function | en->zh | target_direction | 1.0 | 76 | 0.4276315789473684 | 0.9901315789473685 | -0.5625 |
| qwen3 | function | en->zh | template_subtracted | 1.0 | 76 | 0.421875 | 0.9901315789473685 | -0.5682565789473685 |
| qwen3 | function | en->zh | wrong_mean_subtracted | 1.0 | 76 | 0.39165296052631576 | 0.9901315789473685 | -0.5984786184210527 |
| qwen3 | function | en->zh | specific_direction | 1.0 | 76 | 0.3813733552631579 | 0.9901315789473685 | -0.6087582236842106 |
| deepseek7b | category | en->zh | specific_direction | 1.0 | 120 | -0.0019694010416666666 | 0.6730623881022135 | -0.6750317891438802 |
| deepseek7b | category | en->zh | wrong_mean_subtracted | 1.0 | 120 | -0.0019694010416666666 | 0.6730623881022135 | -0.6750317891438802 |
| glm4 | color | en->en | target_direction | 1.0 | 54 | 2.7205946180555554 | 3.892107928240741 | -1.1715133101851856 |
| qwen3 | category | en->zh | template_subtracted | 1.0 | 120 | 0.01015625 | 1.217578125 | -1.2074218749999999 |
| qwen3 | category | en->zh | target_direction | 1.0 | 120 | 0.004427083333333333 | 1.217578125 | -1.2131510416666667 |
| qwen3 | category | en->zh | specific_direction | 1.0 | 120 | -0.010416666666666666 | 1.217578125 | -1.2279947916666667 |
| qwen3 | category | en->zh | wrong_mean_subtracted | 1.0 | 120 | -0.010416666666666666 | 1.217578125 | -1.2279947916666667 |
