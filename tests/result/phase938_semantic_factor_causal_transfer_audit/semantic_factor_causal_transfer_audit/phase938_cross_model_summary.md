# Phase 938 semantic factor causal transfer audit

## Evidence

- semantic_factor_causal_transfer_positive: 3

## Condition Rows

| model | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | target_direction | 1.0 | 152 | 0.4453125 | 0.6525493421052632 | 70 | 18 |
| qwen3 | target_direction | 0.5 | 152 | 0.25370065789473684 | 0.37397203947368424 | 61 | 11 |
| qwen3 | random_same_norm | 0.5 | 152 | 0.02549342105263158 | 0.012335526315789474 | 41 | 3 |
| qwen3 | baseline | None | 152 | 0.0 | 0.0 | 0 | 0 |
| qwen3 | random_same_norm | 1.0 | 152 | 0.00390625 | -0.013980263157894737 | 42 | 5 |
| qwen3 | wrong_label_direction | 0.5 | 152 | 0.019942434210526317 | -0.3178453947368421 | 37 | 0 |
| qwen3 | negative_target_direction | 0.5 | 152 | -0.3145559210526316 | -0.4428453947368421 | 18 | 1 |
| qwen3 | wrong_label_direction | 1.0 | 152 | -0.06620065789473684 | -0.9551809210526315 | 37 | 1 |
| qwen3 | negative_target_direction | 1.0 | 152 | -0.7436266447368421 | -0.9740953947368421 | 19 | 0 |
| glm4 | target_direction | 1.0 | 152 | 4.464483963815789 | 2.217702765213816 | 104 | 63 |
| glm4 | target_direction | 0.5 | 152 | 4.39569091796875 | 2.13885498046875 | 103 | 62 |
| glm4 | random_same_norm | 0.5 | 152 | 4.2470863743832235 | 1.9550684878700657 | 103 | 56 |
| glm4 | random_same_norm | 1.0 | 152 | 4.225521689967105 | 1.9296232524671053 | 102 | 56 |
| glm4 | wrong_label_direction | 0.5 | 152 | 4.1834459806743425 | 1.709337736430921 | 103 | 51 |
| glm4 | negative_target_direction | 0.5 | 152 | 4.00799560546875 | 1.6627486379523027 | 100 | 51 |
| glm4 | wrong_label_direction | 1.0 | 152 | 4.0520276521381575 | 1.3991185238486843 | 102 | 46 |
| glm4 | negative_target_direction | 1.0 | 152 | 3.6735614977384867 | 1.2803665964226973 | 101 | 47 |
| glm4 | baseline | None | 152 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | target_direction | 1.0 | 152 | 0.3828125 | 0.3458059210526316 | 77 | 5 |
| deepseek7b | target_direction | 0.5 | 152 | 0.31365645559210525 | 0.21024362664473684 | 69 | 1 |
| deepseek7b | random_same_norm | 0.5 | 152 | 0.24180201480263158 | 0.1122789884868421 | 66 | 1 |
| deepseek7b | random_same_norm | 1.0 | 152 | 0.17957024825246712 | 0.06464426141036184 | 73 | 2 |
| deepseek7b | baseline | None | 152 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | wrong_label_direction | 0.5 | 152 | 0.02470960115131579 | -0.126708984375 | 43 | 3 |
| deepseek7b | negative_target_direction | 0.5 | 152 | 0.18397923519736842 | -0.17971319901315788 | 36 | 0 |
| deepseek7b | wrong_label_direction | 1.0 | 152 | -0.15810032894736842 | -0.31805098684210525 | 46 | 4 |
| deepseek7b | negative_target_direction | 1.0 | 152 | 0.005473889802631579 | -0.48188219572368424 | 38 | 1 |

## Top Relation Conditions

| model | relation | condition | alpha | rows | mean logit delta | mean margin delta | rank improved | new winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | function | target_direction | 1.0 | 38 | 6.3969983552631575 | 2.6422697368421053 | 21 | 16 |
| glm4 | function | target_direction | 0.5 | 38 | 6.201274671052632 | 2.4161184210526314 | 22 | 15 |
| glm4 | category | target_direction | 1.0 | 60 | 1.8919270833333333 | 2.2707845052083333 | 30 | 21 |
| glm4 | category | target_direction | 0.5 | 60 | 1.8447347005208334 | 2.1965576171875 | 28 | 21 |
| glm4 | category | random_same_norm | 1.0 | 60 | 1.8581705729166667 | 2.187858072916667 | 28 | 21 |
| glm4 | category | random_same_norm | 0.5 | 60 | 1.8445719401041667 | 2.1791422526041666 | 29 | 20 |
| glm4 | category | wrong_label_direction | 0.5 | 60 | 1.831005859375 | 2.1574055989583334 | 28 | 20 |
| glm4 | category | wrong_label_direction | 1.0 | 60 | 1.8215169270833333 | 2.1293294270833334 | 28 | 20 |
| glm4 | category | negative_target_direction | 0.5 | 60 | 1.8388753255208334 | 2.1156819661458335 | 26 | 20 |
| glm4 | category | negative_target_direction | 1.0 | 60 | 1.8521891276041667 | 2.055183919270833 | 27 | 20 |
| glm4 | color | negative_target_direction | 1.0 | 54 | 5.9311342592592595 | 1.8967013888888888 | 53 | 26 |
| glm4 | color | wrong_label_direction | 1.0 | 54 | 5.949652777777778 | 1.8952546296296295 | 53 | 25 |
| glm4 | color | negative_target_direction | 0.5 | 54 | 5.933449074074074 | 1.8888888888888888 | 53 | 26 |
| glm4 | color | wrong_label_direction | 0.5 | 54 | 5.949074074074074 | 1.8877314814814814 | 53 | 26 |
| glm4 | color | target_direction | 0.5 | 54 | 5.9594907407407405 | 1.8796296296296295 | 53 | 26 |
| glm4 | color | random_same_norm | 0.5 | 54 | 5.951967592592593 | 1.8790509259259258 | 53 | 26 |
| glm4 | color | random_same_norm | 1.0 | 54 | 5.953125 | 1.8764467592592593 | 53 | 26 |
| glm4 | color | target_direction | 1.0 | 54 | 5.962962962962963 | 1.8599537037037037 | 53 | 26 |
| glm4 | function | random_same_norm | 0.5 | 38 | 5.617804276315789 | 1.7092927631578947 | 21 | 10 |
| glm4 | function | random_same_norm | 1.0 | 38 | 5.508429276315789 | 1.597450657894737 | 21 | 9 |
| qwen3 | function | target_direction | 1.0 | 38 | 0.8108552631578947 | 1.0904605263157894 | 20 | 9 |
| glm4 | function | wrong_label_direction | 0.5 | 38 | 5.388774671052632 | 0.7483552631578947 | 22 | 5 |
| qwen3 | color | target_direction | 1.0 | 54 | 0.7442129629629629 | 0.7314814814814815 | 26 | 6 |
| deepseek7b | category | target_direction | 1.0 | 60 | 0.5625 | 0.6364583333333333 | 34 | 4 |
| glm4 | function | negative_target_direction | 0.5 | 38 | 4.6967516447368425 | 0.626233552631579 | 21 | 5 |
| qwen3 | function | target_direction | 0.5 | 38 | 0.4605263157894737 | 0.5592105263157895 | 17 | 2 |
| deepseek7b | category | target_direction | 0.5 | 60 | 0.3857421875 | 0.39251302083333334 | 33 | 1 |
| qwen3 | color | target_direction | 0.5 | 54 | 0.4027777777777778 | 0.3854166666666667 | 23 | 5 |
| qwen3 | category | target_direction | 1.0 | 60 | -0.05520833333333333 | 0.30416666666666664 | 24 | 3 |
| qwen3 | category | target_direction | 0.5 | 60 | -0.011458333333333333 | 0.24635416666666668 | 21 | 4 |
| deepseek7b | function | target_direction | 1.0 | 38 | 0.02631578947368421 | 0.16776315789473684 | 20 | 1 |
| deepseek7b | color | target_direction | 1.0 | 54 | 0.4340277777777778 | 0.14814814814814814 | 23 | 0 |
| deepseek7b | category | random_same_norm | 0.5 | 60 | 0.27350260416666666 | 0.1443359375 | 32 | 1 |
| deepseek7b | category | random_same_norm | 1.0 | 60 | 0.26168212890625 | 0.12782796223958334 | 35 | 1 |
| deepseek7b | color | target_direction | 0.5 | 54 | 0.4774305555555556 | 0.10590277777777778 | 17 | 0 |
| deepseek7b | color | random_same_norm | 0.5 | 54 | 0.4195601851851852 | 0.09664351851851852 | 18 | 0 |
| deepseek7b | function | random_same_norm | 0.5 | 38 | -0.06085526315789474 | 0.08388157894736842 | 16 | 0 |
| deepseek7b | function | target_direction | 0.5 | 38 | -0.03289473684210526 | 0.07072368421052631 | 19 | 0 |
| deepseek7b | color | wrong_label_direction | 0.5 | 54 | 0.2534722222222222 | 0.052083333333333336 | 12 | 0 |
| qwen3 | category | random_same_norm | 0.5 | 60 | 0.04583333333333333 | 0.03854166666666667 | 20 | 0 |
| deepseek7b | color | negative_target_direction | 0.5 | 54 | 0.6597222222222222 | 0.032407407407407406 | 17 | 0 |
| deepseek7b | color | random_same_norm | 1.0 | 54 | 0.4519675925925926 | 0.031828703703703706 | 21 | 0 |
| qwen3 | category | random_same_norm | 1.0 | 60 | 0.043229166666666666 | 0.030208333333333334 | 22 | 0 |
| qwen3 | color | random_same_norm | 0.5 | 54 | 0.03587962962962963 | 0.02546296296296296 | 10 | 1 |
| deepseek7b | color | wrong_label_direction | 1.0 | 54 | -0.0005787037037037037 | 0.023726851851851853 | 11 | 0 |
| qwen3 | color | random_same_norm | 1.0 | 54 | 0.010416666666666666 | 0.023148148148148147 | 7 | 3 |
| deepseek7b | function | random_same_norm | 1.0 | 38 | -0.3371710526315789 | 0.011513157894736841 | 17 | 1 |
| deepseek7b | color | negative_target_direction | 1.0 | 54 | 0.6927083333333334 | 0.009837962962962963 | 18 | 0 |
| qwen3 | category | baseline | None | 60 | 0.0 | 0.0 | 0 | 0 |
| qwen3 | color | baseline | None | 54 | 0.0 | 0.0 | 0 | 0 |
| qwen3 | function | baseline | None | 38 | 0.0 | 0.0 | 0 | 0 |
| glm4 | category | baseline | None | 60 | 0.0 | 0.0 | 0 | 0 |
| glm4 | color | baseline | None | 54 | 0.0 | 0.0 | 0 | 0 |
| glm4 | function | baseline | None | 38 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | category | baseline | None | 60 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | color | baseline | None | 54 | 0.0 | 0.0 | 0 | 0 |
| deepseek7b | function | baseline | None | 38 | 0.0 | 0.0 | 0 | 0 |
| qwen3 | color | wrong_label_direction | 0.5 | 54 | 0.19212962962962962 | -0.006944444444444444 | 10 | 0 |
| qwen3 | color | wrong_label_direction | 1.0 | 54 | 0.3541666666666667 | -0.011574074074074073 | 16 | 0 |
| qwen3 | category | wrong_label_direction | 0.5 | 60 | 0.10572916666666667 | -0.025 | 21 | 0 |
| qwen3 | function | random_same_norm | 0.5 | 38 | -0.02138157894736842 | -0.047697368421052634 | 11 | 2 |
| deepseek7b | function | wrong_label_direction | 1.0 | 38 | -0.8914473684210527 | -0.08388157894736842 | 9 | 2 |
| deepseek7b | function | wrong_label_direction | 0.5 | 38 | -0.47368421052631576 | -0.10361842105263158 | 6 | 2 |
| deepseek7b | function | negative_target_direction | 0.5 | 38 | -0.19407894736842105 | -0.10526315789473684 | 7 | 0 |
| qwen3 | function | random_same_norm | 1.0 | 38 | -0.06743421052631579 | -0.13651315789473684 | 13 | 2 |
| qwen3 | category | wrong_label_direction | 1.0 | 60 | 0.159375 | -0.22708333333333333 | 19 | 1 |
| qwen3 | category | negative_target_direction | 0.5 | 60 | -0.019791666666666666 | -0.29270833333333335 | 17 | 0 |
| deepseek7b | category | wrong_label_direction | 0.5 | 60 | 0.13447265625 | -0.30224609375 | 25 | 1 |
| qwen3 | color | negative_target_direction | 0.5 | 54 | -0.33101851851851855 | -0.38078703703703703 | 0 | 1 |
| deepseek7b | function | negative_target_direction | 1.0 | 38 | -0.5657894736842105 | -0.3815789473684211 | 7 | 1 |
| deepseek7b | category | negative_target_direction | 0.5 | 60 | -0.004752604166666666 | -0.4177734375 | 12 | 0 |
| glm4 | function | wrong_label_direction | 1.0 | 38 | 4.877261513157895 | -0.4588815789473684 | 21 | 1 |
| qwen3 | category | negative_target_direction | 1.0 | 60 | -0.09427083333333333 | -0.6322916666666667 | 18 | 0 |
| qwen3 | function | negative_target_direction | 0.5 | 38 | -0.756578947368421 | -0.7680921052631579 | 1 | 0 |
| deepseek7b | category | wrong_label_direction | 1.0 | 60 | 0.16458333333333333 | -0.7739583333333333 | 26 | 2 |
| qwen3 | color | negative_target_direction | 1.0 | 54 | -0.7233796296296297 | -0.7939814814814815 | 0 | 0 |
| glm4 | function | negative_target_direction | 1.0 | 38 | 3.341282894736842 | -0.8188733552631579 | 21 | 1 |
| deepseek7b | category | negative_target_direction | 1.0 | 60 | -0.25123697916666665 | -0.9879557291666666 | 13 | 0 |
| qwen3 | function | wrong_label_direction | 0.5 | 38 | -0.36019736842105265 | -1.2220394736842106 | 6 | 0 |
| qwen3 | function | negative_target_direction | 1.0 | 38 | -1.7976973684210527 | -1.769736842105263 | 1 | 0 |
| qwen3 | function | wrong_label_direction | 1.0 | 38 | -1.019736842105263 | -3.445723684210526 | 2 | 0 |
