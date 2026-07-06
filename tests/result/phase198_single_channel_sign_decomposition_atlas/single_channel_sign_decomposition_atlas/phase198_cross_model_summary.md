# Phase 198 single-channel sign decomposition atlas

## Evidence

- single_channel_mixed_sign_decomposition_positive: 1
- single_channel_support_positive: 1
- single_channel_suppressor_only: 1

## Top Channel Eval Rows

| model | relation | pair | hidden | channel | source | sign | boundary slope | relation slope | target slope | rows |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | color | en->en | 36 | 249 | candidate | support_channel | 0.5432692307692308 | 0.03245192307692307 | 0.5865384615384616 | 26 |
| qwen3 | color | en->en | 36 | 16 | candidate | mixed_side_effect_channel | 0.12980769230769232 | -0.03365384615384616 | 0.15625 | 26 |
| qwen3 | function | zh->zh | 27 | 2 | candidate | support_channel | 0.05598958333333333 | -0.009114583333333332 | 0.014322916666666668 | 12 |
| glm4 | function | zh->en | 30 | 1165 | candidate | mixed_side_effect_channel | 0.04817708333333348 | -0.02473958333333326 | -0.05078125 | 12 |
| glm4 | color | en->en | 30 | 1165 | candidate | support_channel | 0.03125 | 0.004807692307692307 | 0.021634615384615384 | 26 |
| qwen3 | function | zh->zh | 27 | 58 | candidate | support_channel | 0.02734375 | 0.013020833333333332 | 0.011718749999999998 | 12 |
| qwen3 | function | en->en | 27 | 3 | candidate | support_channel | 0.020833333333333336 | 0.04166666666666667 | 0.03125 | 12 |
| glm4 | color | zh->en | 30 | 260 | candidate | near_zero_or_correlational | 0.017857142857142856 | 0.008928571428571428 | 0.017857142857142856 | 14 |
| deepseek7b | function | en->en | 14 | 16221 | candidate | near_zero_or_correlational | 0.015625 | 0.0 | 0.026041666666666664 | 12 |
| qwen3 | function | zh->en | 27 | 106 | candidate | near_zero_or_correlational | 0.015625 | -0.010416666666666666 | 0.03645833333333333 | 12 |
| qwen3 | function | zh->en | 27 | 3 | candidate | mixed_side_effect_channel | 0.015625 | 0.020833333333333332 | 0.015625 | 12 |
| qwen3 | function | zh->zh | 27 | 3 | candidate | near_zero_or_correlational | 0.01171875 | -0.00390625 | 0.01171875 | 12 |
| qwen3 | function | en->en | 27 | 376 | candidate | near_zero_or_correlational | 0.010416666666666668 | 0.005208333333333333 | 0.005208333333333332 | 12 |
| qwen3 | color | zh->en | 36 | 310 | candidate | near_zero_or_correlational | 0.008928571428571428 | 0.00892857142857143 | 0.00892857142857143 | 14 |
| glm4 | function | en->en | 30 | 8633 | candidate | near_zero_or_correlational | 0.006944444444444642 | 0.00694444444444442 | 0.010416666666666075 | 9 |
| glm4 | color | zh->zh | 30 | 5532 | candidate | near_zero_or_correlational | 0.0068359375 | 0.008103590745192308 | 0.016451322115384616 | 26 |
| glm4 | color | zh->zh | 30 | 1165 | candidate | near_zero_or_correlational | 0.006047175480769231 | 0.0027231069711538465 | 0.003643329326923076 | 26 |
| qwen3 | color | zh->zh | 36 | 2509 | candidate | near_zero_or_correlational | 0.004807692307692308 | 0.0 | 0.0 | 26 |
| glm4 | function | zh->en | 30 | 8633 | candidate | near_zero_or_correlational | 0.0026041666666669627 | 0.015625 | 0.01302083333333326 | 12 |
| qwen3 | color | zh->zh | 36 | 1579 | candidate | near_zero_or_correlational | 0.001201923076923077 | 0.001201923076923077 | 0.001201923076923077 | 26 |
| qwen3 | color | en->en | 36 | 2509 | candidate | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 26 |
| qwen3 | color | en->zh | 36 | 1579 | candidate | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 14 |
| qwen3 | color | en->zh | 36 | 310 | candidate | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 14 |
| qwen3 | color | zh->en | 36 | 134 | candidate | near_zero_or_correlational | 0.0 | -0.00892857142857143 | 0.0 | 14 |
| qwen3 | color | zh->en | 36 | 2509 | candidate | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 14 |
| qwen3 | function | zh->en | 27 | 2 | candidate | near_zero_or_correlational | 0.0 | -0.005208333333333333 | -0.005208333333333333 | 12 |
| glm4 | color | zh->zh | 30 | 8633 | candidate | near_zero_or_correlational | -0.004056490384615385 | 4.695012019230796e-05 | 0.007962740384615386 | 26 |
| deepseek7b | function | en->en | 14 | 3033 | candidate | mixed_side_effect_channel | -0.005208333333333336 | 0.03125 | 0.03645833333333333 | 12 |
| glm4 | color | zh->en | 30 | 5775 | candidate | near_zero_or_correlational | -0.006696428571428571 | -0.006696428571428572 | -0.011160714285714284 | 14 |
| glm4 | color | en->en | 30 | 8633 | candidate | near_zero_or_correlational | -0.007211538461538462 | 0.003605769230769231 | -0.002403846153846154 | 26 |
| qwen3 | color | en->zh | 36 | 2509 | candidate | near_zero_or_correlational | -0.008928571428571428 | 0.0 | 0.0 | 14 |
| qwen3 | color | zh->zh | 36 | 134 | candidate | near_zero_or_correlational | -0.009615384615384616 | 0.0 | -0.004807692307692308 | 26 |
| glm4 | color | zh->en | 30 | 4906 | candidate | near_zero_or_correlational | -0.013392857142857144 | -0.004464285714285715 | 0.0 | 14 |
| glm4 | function | en->en | 30 | 1165 | candidate | mixed_side_effect_channel | -0.013888888888889284 | -0.02083333333333326 | -0.013888888888889284 | 9 |
| qwen3 | function | en->en | 27 | 2 | candidate | near_zero_or_correlational | -0.015625 | -0.010416666666666668 | -0.005208333333333336 | 12 |
| glm4 | function | zh->en | 30 | 5532 | candidate | suppressor_or_blocker_channel | -0.02994791666666652 | 0.02473958333333337 | 0.01171875 | 12 |
| glm4 | color | en->en | 30 | 5532 | candidate | suppressor_or_blocker_channel | -0.040865384615384616 | 0.007211538461538461 | -0.019230769230769232 | 26 |
| deepseek7b | function | en->en | 14 | 6030 | candidate | suppressor_or_blocker_channel | -0.04166666666666667 | -0.010416666666666668 | 0.0 | 12 |
| glm4 | function | en->en | 30 | 5532 | candidate | suppressor_or_blocker_channel | -0.08333333333333393 | -0.03125 | -0.027777777777778567 | 9 |
| glm4 | function | zh->en | 30 | 4932 | same_layer_random | support_channel | 0.04166666666666652 | -0.01041666666666663 | 0.015625 | 12 |
| glm4 | color | en->en | 30 | 5571 | same_layer_random | support_channel | 0.02403846153846154 | 0.021634615384615384 | 0.01201923076923077 | 26 |
| deepseek7b | function | en->en | 14 | 3380 | same_layer_random | support_channel | 0.020833333333333336 | 0.03125 | 0.005208333333333333 | 12 |
| glm4 | function | en->en | 30 | 5990 | same_layer_random | near_zero_or_correlational | 0.017361111111110716 | 0.01736111111111094 | 0.013888888888889284 | 9 |
| qwen3 | color | zh->zh | 36 | 3505 | same_layer_random | near_zero_or_correlational | 0.014423076923076924 | 0.002403846153846154 | 0.0 | 26 |
| glm4 | function | en->en | 30 | 2785 | same_layer_random | near_zero_or_correlational | 0.013888888888889284 | 0.0 | 0.013888888888889284 | 9 |
| glm4 | color | zh->en | 30 | 2418 | same_layer_random | near_zero_or_correlational | 0.013392857142857142 | 0.004464285714285714 | 0.013392857142857142 | 14 |
| qwen3 | color | zh->zh | 36 | 3710 | same_layer_random | near_zero_or_correlational | 0.010817307692307692 | 0.003605769230769231 | 0.006009615384615385 | 26 |
| qwen3 | color | en->en | 36 | 1332 | same_layer_random | near_zero_or_correlational | 0.007211538461538462 | 0.009615384615384614 | 0.009615384615384616 | 26 |
| qwen3 | function | zh->zh | 27 | 4163 | same_layer_random | near_zero_or_correlational | 0.005208333333333333 | -0.01171875 | 0.005208333333333333 | 12 |
| qwen3 | color | zh->en | 36 | 1820 | same_layer_random | near_zero_or_correlational | 0.004464285714285719 | 0.0 | 0.0 | 14 |
| qwen3 | color | en->zh | 36 | 7140 | same_layer_random | near_zero_or_correlational | 0.004464285714285714 | 0.004464285714285714 | 0.004464285714285714 | 14 |
| glm4 | color | en->en | 30 | 6841 | same_layer_random | near_zero_or_correlational | 0.002403846153846154 | 0.0012019230769230779 | 0.004807692307692308 | 26 |
| qwen3 | color | zh->zh | 36 | 3496 | same_layer_random | near_zero_or_correlational | 0.002403846153846154 | 0.004807692307692308 | 0.002403846153846154 | 26 |
| glm4 | color | zh->en | 30 | 6462 | same_layer_random | near_zero_or_correlational | 0.002232142857142858 | -0.006696428571428571 | -0.011160714285714284 | 14 |
| qwen3 | color | en->en | 36 | 1458 | same_layer_random | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 26 |
| qwen3 | color | en->zh | 36 | 4517 | same_layer_random | near_zero_or_correlational | 0.0 | 0.0 | 0.0 | 14 |
| qwen3 | color | zh->en | 36 | 1448 | same_layer_random | near_zero_or_correlational | 0.0 | -0.008928571428571428 | 0.0 | 14 |
| qwen3 | color | zh->en | 36 | 7506 | same_layer_random | near_zero_or_correlational | 0.0 | 0.008928571428571428 | 0.0 | 14 |
| qwen3 | function | en->en | 27 | 566 | same_layer_random | near_zero_or_correlational | 0.0 | 0.005208333333333333 | 0.0 | 12 |
| qwen3 | color | en->zh | 36 | 2069 | same_layer_random | near_zero_or_correlational | -0.004464285714285714 | -0.004464285714285714 | -0.004464285714285714 | 14 |
| qwen3 | function | zh->en | 27 | 444 | same_layer_random | near_zero_or_correlational | -0.005208333333333333 | -0.015625 | 0.0 | 12 |
| deepseek7b | function | en->en | 14 | 18897 | same_layer_random | mixed_side_effect_channel | -0.005208333333333336 | -0.020833333333333332 | -0.005208333333333336 | 12 |
| glm4 | color | zh->zh | 30 | 12939 | same_layer_random | near_zero_or_correlational | -0.005671574519230769 | -0.00279822716346154 | -0.0008638822115384619 | 26 |
| glm4 | color | en->en | 30 | 12455 | same_layer_random | near_zero_or_correlational | -0.007211538461538462 | -0.007211538461538461 | -0.01201923076923077 | 26 |
| glm4 | color | zh->zh | 30 | 10223 | same_layer_random | near_zero_or_correlational | -0.008263221153846154 | -0.006742037259615385 | -0.005859375 | 26 |
| qwen3 | function | en->en | 27 | 1790 | same_layer_random | near_zero_or_correlational | -0.010416666666666666 | 0.015625 | -0.010416666666666668 | 12 |
| qwen3 | function | zh->en | 27 | 5305 | same_layer_random | mixed_side_effect_channel | -0.010416666666666666 | -0.026041666666666664 | -0.020833333333333332 | 12 |
| qwen3 | function | zh->zh | 27 | 3262 | same_layer_random | near_zero_or_correlational | -0.011718749999999998 | 0.0 | 0.014322916666666666 | 12 |
| glm4 | color | zh->zh | 30 | 3276 | same_layer_random | near_zero_or_correlational | -0.01378455528846154 | 0.001201923076923077 | 0.005446213942307692 | 26 |
| glm4 | color | zh->en | 30 | 12739 | same_layer_random | near_zero_or_correlational | -0.015625 | 0.0 | 0.006696428571428571 | 14 |
| qwen3 | function | en->en | 27 | 3458 | same_layer_random | near_zero_or_correlational | -0.015625 | -0.015625 | 0.0 | 12 |
| qwen3 | color | en->en | 36 | 474 | same_layer_random | near_zero_or_correlational | -0.01682692307692308 | -0.01682692307692308 | -0.01682692307692308 | 26 |
| qwen3 | function | zh->zh | 27 | 2563 | same_layer_random | near_zero_or_correlational | -0.016927083333333332 | -0.00390625 | -0.022135416666666668 | 12 |
| glm4 | function | zh->en | 30 | 11252 | same_layer_random | mixed_side_effect_channel | -0.01953125 | 0.03255208333333337 | 0.0013020833333332593 | 12 |
| glm4 | function | zh->en | 30 | 7902 | same_layer_random | suppressor_or_blocker_channel | -0.0234375 | 0.01302083333333326 | 0.01302083333333326 | 12 |
| deepseek7b | function | en->en | 14 | 5261 | same_layer_random | suppressor_or_blocker_channel | -0.026041666666666664 | -0.03125 | -0.07291666666666666 | 12 |
| glm4 | function | en->en | 30 | 844 | same_layer_random | suppressor_or_blocker_channel | -0.03472222222222143 | -0.02083333333333326 | -0.020833333333333925 | 9 |
| qwen3 | function | zh->en | 27 | 7194 | same_layer_random | suppressor_or_blocker_channel | -0.046875 | -0.026041666666666664 | -0.036458333333333336 | 12 |
