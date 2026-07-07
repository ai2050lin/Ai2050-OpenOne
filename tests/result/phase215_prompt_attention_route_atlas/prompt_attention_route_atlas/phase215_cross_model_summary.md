# Phase 215 prompt attention route atlas

Selected trajectory rows: 111
Attention route rows: 113408
Route delta rows: 8160

| model | pattern | anchor | layer | head | success | drift | max delta | trigger:any delta | answer_slot delta | object delta | target delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | answer_list | gen_after_step_6 | 5 | 19 | 6 | 2 | 0.932231 | 0.004183 | 0.001488 | -0.003373 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 5 | 19 | 6 | 2 | 0.931478 | 0.004461 | 0.000854 | -0.002753 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 24 | 20 | 6 | 2 | 0.874512 | -0.871338 | -0.874512 | 0.003064 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 5 | 14 | 6 | 2 | 0.861654 | -0.004527 | -0.002383 | -0.000331 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 24 | 16 | 6 | 2 | 0.833008 | -0.833008 | -0.828257 | 0.066925 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 25 | 3 | 6 | 2 | 0.743001 | -0.023031 | -0.011688 | 0.004247 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 5 | 14 | 6 | 2 | 0.683594 | 0.009094 | 0.002701 | 0.004171 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 24 | 20 | 6 | 2 | 0.673177 | -0.657878 | -0.673177 | 0.008077 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 26 | 27 | 6 | 2 | 0.667480 | -0.648438 | -0.667480 | 0.013404 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 28 | 8 | 8 | 0.664310 | 0.616241 | -0.011033 | -0.000101 | -0.020390 |
| deepseek7b | answer_explain | gen_after_step_1 | 25 | 1 | 6 | 2 | 0.664144 | -0.620117 | -0.664144 | 0.012049 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 2 | 13 | 6 | 2 | 0.654785 | 0.074544 | 0.026530 | 0.033241 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 10 | 8 | 8 | 0.653900 | 0.553711 | -0.020562 | -0.012211 | -0.039357 |
| deepseek7b | answer_explain | gen_after_step_1 | 25 | 23 | 6 | 2 | 0.640259 | -0.024821 | -0.010913 | -0.640259 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 14 | 15 | 6 | 2 | 0.622721 | 0.117134 | 0.024083 | -0.038013 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 25 | 25 | 6 | 2 | 0.620117 | 0.286540 | 0.024963 | -0.009140 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 24 | 5 | 6 | 2 | 0.618896 | -0.546875 | -0.618896 | 0.012414 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 2 | 1 | 6 | 2 | 0.615885 | 0.062500 | 0.027018 | 0.010385 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 24 | 15 | 6 | 2 | 0.612061 | -0.526042 | -0.612061 | -0.004842 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 30 | 13 | 8 | 8 | 0.611042 | 0.113714 | 0.015761 | 0.004252 | 0.000218 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 11 | 8 | 8 | 0.609737 | 0.538269 | -0.018767 | -0.001423 | -0.004142 |
| deepseek7b | answer_explain | gen_after_step_6 | 23 | 12 | 6 | 2 | 0.609456 | 0.557007 | 0.017111 | -0.000335 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 25 | 1 | 6 | 2 | 0.606852 | -0.534017 | -0.591878 | 0.029260 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 2 | 13 | 6 | 2 | 0.602214 | 0.101644 | 0.040446 | 0.053523 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 2 | 1 | 6 | 2 | 0.583333 | 0.026611 | 0.023112 | 0.013555 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 27 | 8 | 6 | 2 | 0.571615 | 0.419596 | 0.371908 | 0.016083 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 25 | 2 | 6 | 2 | 0.568685 | -0.568685 | -0.518799 | -0.001829 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 24 | 16 | 6 | 2 | 0.563883 | -0.563883 | -0.561361 | 0.088338 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 30 | 4 | 8 | 8 | 0.559937 | -0.013748 | -0.025063 | 0.000101 | 0.000149 |
| deepseek7b | answer_list | gen_after_step_1 | 25 | 2 | 6 | 2 | 0.559570 | -0.559570 | -0.509399 | 0.000130 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_3 | 24 | 20 | 6 | 2 | 0.555990 | 0.047852 | 0.022786 | 0.012077 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 18 | 8 | 8 | 0.550369 | 0.550369 | 0.011822 | -0.059233 | -0.271638 |
| deepseek7b | answer_explain | gen_after_step_1 | 26 | 8 | 6 | 2 | 0.548828 | -0.000509 | -0.001119 | 0.000038 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 26 | 0 | 6 | 2 | 0.546143 | 0.052409 | 0.010854 | -0.546143 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 26 | 27 | 6 | 2 | 0.543783 | -0.189128 | -0.198242 | 0.014760 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 23 | 11 | 6 | 2 | 0.543254 | 0.028442 | 0.009115 | -0.543254 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 25 | 15 | 6 | 2 | 0.542969 | -0.474894 | -0.440659 | -0.001794 | 0.000000 |
| glm4 | answer_repeat | gen_after_step_3 | 12 | 21 | 8 | 8 | 0.538727 | 0.521877 | -0.001264 | -0.000075 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_3 | 23 | 23 | 6 | 2 | 0.538086 | 0.041667 | 0.028564 | 0.000819 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 28 | 18 | 8 | 8 | 0.537994 | 0.034607 | -0.005886 | -0.000854 | -0.005206 |
| deepseek7b | answer_list | gen_after_step_6 | 25 | 0 | 6 | 2 | 0.537760 | 0.060425 | 0.027313 | 0.016235 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 25 | 8 | 8 | 0.535488 | 0.513245 | -0.003075 | -0.005574 | -0.247326 |
| glm4 | answer_target_seeded | gen_after_step_6 | 12 | 25 | 8 | 8 | 0.535400 | -0.021294 | -0.011190 | -0.005033 | -0.000570 |
| deepseek7b | answer_list | gen_after_step_1 | 23 | 6 | 6 | 2 | 0.527990 | 0.019918 | 0.002380 | -0.527990 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 27 | 10 | 6 | 2 | 0.523437 | 0.211588 | 0.178060 | 0.033366 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 23 | 27 | 6 | 2 | 0.520182 | 0.011780 | 0.008779 | 0.000119 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 22 | 10 | 6 | 2 | 0.517578 | 0.306722 | 0.110596 | -0.039135 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 22 | 7 | 6 | 2 | 0.517171 | 0.025024 | 0.018921 | -0.517171 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 24 | 14 | 6 | 2 | 0.516276 | 0.041382 | 0.000758 | 0.014923 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 25 | 3 | 6 | 2 | 0.514974 | 0.015299 | 0.014608 | -0.006251 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 25 | 14 | 6 | 2 | 0.514974 | 0.192952 | 0.174723 | 0.002183 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 27 | 7 | 6 | 2 | 0.514079 | -0.262655 | -0.064397 | -0.003693 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 23 | 0 | 6 | 2 | 0.511963 | 0.035619 | 0.013526 | 0.035838 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_6 | 23 | 10 | 6 | 2 | 0.511393 | 0.020391 | 0.014070 | -0.000407 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_1 | 24 | 18 | 6 | 2 | 0.510173 | 0.068654 | 0.029489 | 0.016479 | 0.000000 |
| glm4 | answer_target_seeded | gen_after_step_6 | 29 | 23 | 8 | 8 | 0.509644 | -0.037704 | -0.038107 | 0.001260 | -0.024881 |
| glm4 | answer_explain | gen_after_step_3 | 27 | 21 | 7 | 8 | 0.509147 | -0.001167 | -0.001129 | -0.000002 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 24 | 19 | 6 | 2 | 0.507487 | 0.012263 | 0.002307 | -0.045013 | 0.000000 |
| deepseek7b | answer_list | gen_after_step_1 | 14 | 17 | 6 | 2 | 0.505859 | -0.505859 | 0.022217 | 0.001013 | 0.000000 |
| deepseek7b | answer_explain | gen_after_step_6 | 22 | 15 | 6 | 2 | 0.504476 | -0.008499 | -0.001061 | 0.017136 | 0.000000 |
