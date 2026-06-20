# Phase558 Fast Prototype/Object-Binding Summary

## qwen3

pair=vehicle_tool, window=[10, 12, 14], combos=['all'], routes=['forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12

| model | combo | layers | route | condition | donor category | donor variant | base margin | margin | margin delta | remove delta | restore gain | target rank | top1 | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.448 | +1.464 | +1.016 | +0.276 | +0.740 | 150.2 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.448 | +1.406 | +0.958 | +0.276 | +0.682 | 155.4 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.448 | +1.401 | +0.953 | +0.276 | +0.677 | 154.7 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +2.253 | +3.440 | +1.188 | +0.516 | +0.672 | 202.2 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.448 | +1.391 | +0.943 | +0.276 | +0.667 | 146.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.448 | +1.391 | +0.943 | +0.276 | +0.667 | 148.7 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.448 | +1.385 | +0.937 | +0.276 | +0.661 | 146.4 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.448 | +1.375 | +0.927 | +0.276 | +0.651 | 156.4 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.448 | +1.370 | +0.922 | +0.276 | +0.646 | 152.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.448 | +1.359 | +0.911 | +0.276 | +0.635 | 150.3 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.448 | +1.354 | +0.906 | +0.276 | +0.630 | 147.0 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.448 | +1.339 | +0.891 | +0.276 | +0.615 | 144.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +2.253 | +3.352 | +1.099 | +0.516 | +0.583 | 206.4 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.448 | +1.249 | +0.801 | +0.276 | +0.525 | 513.0 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +2.253 | +3.263 | +1.010 | +0.516 | +0.495 | 210.1 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +2.253 | +3.260 | +1.008 | +0.516 | +0.492 | 198.6 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +2.253 | +3.255 | +1.003 | +0.516 | +0.487 | 210.5 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | +2.253 | +3.242 | +0.990 | +0.516 | +0.474 | 213.7 | 0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +2.253 | +3.242 | +0.990 | +0.516 | +0.474 | 213.7 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +2.253 | +3.219 | +0.966 | +0.516 | +0.451 | 212.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +2.253 | +3.216 | +0.964 | +0.516 | +0.448 | 210.0 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | +0.448 | +1.167 | +0.719 | +0.276 | +0.443 | 283.2 | 0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.448 | +1.161 | +0.714 | +0.276 | +0.438 | 176.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +2.253 | +3.104 | +0.852 | +0.516 | +0.336 | 221.9 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +2.253 | +3.083 | +0.831 | +0.516 | +0.315 | 215.5 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | +2.253 | +3.068 | +0.815 | +0.516 | +0.299 | 204.8 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +2.253 | +2.979 | +0.727 | +0.516 | +0.211 | 212.5 | 0.00 | restore_without_drop |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | +0.448 | +0.724 | +0.276 | +0.276 | +0.000 | 330.2 | 0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | +2.253 | +2.768 | +0.516 | +0.516 | +0.000 | 255.2 | 0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | +0.448 | +0.448 | +0.000 | +0.276 | -0.276 | 400.1 | 0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | +2.253 | +2.276 | +0.023 | +0.516 | -0.492 | 295.5 | 0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +2.253 | +2.016 | -0.237 | +0.516 | -0.753 | 656.0 | 0.00 | restore_fail |

## glm4

pair=vehicle_tool, window=[24, 26, 28], combos=['L24', 'L24+L28', 'all'], routes=['forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12

| model | combo | layers | route | condition | donor category | donor variant | base margin | margin | margin delta | remove delta | restore gain | target rank | top1 | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | -1.276 | +4.058 | +5.334 | -0.311 | +5.646 | 618.4 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | -1.276 | +3.645 | +4.921 | -0.311 | +5.232 | 1227.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | -1.276 | +3.605 | +4.881 | -0.311 | +5.192 | 2078.0 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | -1.276 | +3.190 | +4.466 | -0.342 | +4.809 | 1516.0 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.288 | +4.378 | +4.089 | -0.673 | +4.762 | 865.7 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.288 | +4.373 | +4.084 | -0.673 | +4.757 | 630.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | -1.276 | +3.153 | +4.429 | -0.311 | +4.740 | 1348.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | -1.276 | +3.147 | +4.423 | -0.311 | +4.734 | 950.6 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | -1.276 | +3.044 | +4.320 | -0.311 | +4.632 | 1120.2 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | -1.276 | +2.991 | +4.267 | -0.311 | +4.579 | 1304.8 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | -1.276 | +2.936 | +4.212 | -0.342 | +4.554 | 3110.3 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | -1.276 | +2.934 | +4.210 | -0.342 | +4.552 | 1438.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | -1.276 | +2.953 | +4.229 | -0.311 | +4.541 | 1283.7 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | -1.276 | +2.722 | +3.998 | -0.342 | +4.340 | 1118.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | -1.276 | +2.740 | +4.016 | -0.311 | +4.327 | 482.5 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.288 | +3.924 | +3.635 | -0.673 | +4.308 | 1584.8 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | -1.276 | +2.616 | +3.892 | -0.211 | +4.103 | 1209.3 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.288 | +3.648 | +3.360 | -0.673 | +4.033 | 1581.8 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.288 | +3.697 | +3.408 | -0.596 | +4.005 | 1323.2 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | -1.276 | +2.319 | +3.595 | -0.311 | +3.906 | 4319.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | -1.276 | +2.255 | +3.531 | -0.311 | +3.842 | 5739.9 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.288 | +3.452 | +3.163 | -0.673 | +3.836 | 1081.3 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | -1.276 | +2.238 | +3.515 | -0.211 | +3.725 | 2326.5 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.288 | +3.339 | +3.050 | -0.673 | +3.723 | 1235.8 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | -1.276 | +2.088 | +3.364 | -0.342 | +3.706 | 1668.4 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | -1.276 | +2.084 | +3.360 | -0.342 | +3.703 | 1495.5 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | -1.276 | +2.185 | +3.461 | -0.211 | +3.672 | 1468.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.288 | +3.284 | +2.996 | -0.673 | +3.669 | 1142.3 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | -1.276 | +2.049 | +3.325 | -0.342 | +3.667 | 2037.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.288 | +3.347 | +3.059 | -0.596 | +3.655 | 1642.0 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | -1.276 | +1.978 | +3.254 | -0.342 | +3.597 | 1995.3 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | -1.276 | +1.960 | +3.236 | -0.342 | +3.578 | 5867.6 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | -1.276 | +1.943 | +3.219 | -0.342 | +3.562 | 6324.2 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | +0.288 | +3.164 | +2.875 | -0.673 | +3.548 | 1344.6 | 0.00 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.288 | +3.164 | +2.875 | -0.673 | +3.548 | 1344.6 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | -1.276 | +2.009 | +3.285 | -0.211 | +3.496 | 1200.5 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.288 | +3.093 | +2.804 | -0.673 | +3.477 | 465.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.288 | +3.161 | +2.873 | -0.596 | +3.469 | 2592.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.288 | +3.068 | +2.780 | -0.673 | +3.453 | 3200.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.288 | +3.125 | +2.836 | -0.596 | +3.432 | 1022.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | -1.276 | +1.826 | +3.103 | -0.311 | +3.414 | 1611.8 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.288 | +2.973 | +2.684 | -0.673 | +3.357 | 3920.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | -1.276 | +1.510 | +2.786 | -0.342 | +3.129 | 997.2 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | -1.276 | +1.544 | +2.820 | -0.211 | +3.031 | 5465.5 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.288 | +2.794 | +2.505 | -0.462 | +2.968 | 2023.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.288 | +2.490 | +2.202 | -0.596 | +2.798 | 4849.4 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.288 | +2.604 | +2.316 | -0.462 | +2.778 | 2537.4 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.288 | +2.427 | +2.139 | -0.596 | +2.735 | 5457.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.288 | +2.417 | +2.128 | -0.596 | +2.725 | 1815.6 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | -1.276 | +1.223 | +2.500 | -0.211 | +2.710 | 3048.9 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.288 | +2.523 | +2.235 | -0.462 | +2.698 | 1621.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.288 | +2.377 | +2.088 | -0.596 | +2.685 | 1909.0 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | -1.276 | +1.190 | +2.466 | -0.211 | +2.677 | 3594.7 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.288 | +2.339 | +2.051 | -0.596 | +2.647 | 2044.2 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | -1.276 | +1.157 | +2.433 | -0.211 | +2.644 | 3444.3 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | -1.276 | +1.133 | +2.409 | -0.211 | +2.620 | 3357.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | -1.276 | +0.970 | +2.246 | -0.342 | +2.589 | 3460.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | +0.288 | +2.222 | +1.934 | -0.596 | +2.530 | 2254.7 | 0.00 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.288 | +2.222 | +1.934 | -0.596 | +2.530 | 2254.7 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | -1.276 | +0.922 | +2.198 | -0.311 | +2.509 | 535.9 | 0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | -1.276 | +0.999 | +2.275 | -0.211 | +2.486 | 11702.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.288 | +1.950 | +1.661 | -0.673 | +2.334 | 1431.2 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | -1.276 | +0.825 | +2.101 | -0.211 | +2.312 | 3283.4 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.288 | +1.938 | +1.649 | -0.462 | +2.112 | 5513.8 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | -1.276 | +0.591 | +1.867 | -0.211 | +2.078 | 1479.1 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.288 | +1.889 | +1.601 | -0.462 | +2.064 | 2049.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.288 | +1.679 | +1.391 | -0.596 | +1.987 | 1044.6 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | -1.276 | +0.471 | +1.747 | -0.211 | +1.958 | 10107.5 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.288 | +1.692 | +1.404 | -0.462 | +1.866 | 7355.2 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | -1.276 | +0.133 | +1.409 | -0.342 | +1.751 | 627.5 | 0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.288 | +1.505 | +1.216 | -0.462 | +1.679 | 3010.2 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.288 | +1.403 | +1.115 | -0.462 | +1.577 | 2844.8 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.288 | +1.324 | +1.036 | -0.462 | +1.498 | 3095.6 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | +0.288 | +1.237 | +0.949 | -0.462 | +1.411 | 3616.3 | 0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.288 | +1.237 | +0.949 | -0.462 | +1.411 | 3616.3 | 0.00 | rank_restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.288 | +0.886 | +0.598 | -0.596 | +1.194 | 3327.3 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.288 | +0.733 | +0.445 | -0.462 | +0.907 | 1687.8 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | -1.276 | -0.747 | +0.529 | -0.211 | +0.740 | 735.2 | 0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.288 | +0.492 | +0.204 | -0.462 | +0.666 | 19780.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | +0.288 | +0.270 | -0.019 | -0.673 | +0.654 | 6559.1 | 0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | +0.288 | +0.270 | -0.019 | -0.596 | +0.578 | 6583.8 | 0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.288 | +0.306 | +0.018 | -0.462 | +0.480 | 4968.3 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | +0.288 | +0.276 | -0.013 | -0.462 | +0.450 | 6485.0 | 0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | -1.276 | -1.290 | -0.014 | -0.342 | +0.328 | 587.1 | 0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | -1.276 | -1.392 | -0.116 | -0.342 | +0.226 | 23237.1 | 0.00 | rank_restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | -1.276 | -1.376 | -0.100 | -0.311 | +0.211 | 585.2 | 0.00 | rank_necessity_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | -1.276 | -1.292 | -0.016 | -0.211 | +0.195 | 577.2 | 0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | -1.276 | -1.437 | -0.161 | -0.311 | +0.150 | 23747.2 | 0.00 | rank_restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | -1.276 | -1.487 | -0.211 | -0.211 | +0.000 | 526.7 | 0.00 | rank_necessity_drop |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | +0.288 | -0.174 | -0.462 | -0.462 | +0.000 | 8139.3 | 0.00 | rank_necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | -1.276 | -1.618 | -0.342 | -0.342 | +0.000 | 527.7 | 0.00 | rank_necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | +0.288 | -0.308 | -0.596 | -0.596 | +0.000 | 8318.0 | 0.00 | rank_necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | -1.276 | -1.587 | -0.311 | -0.311 | +0.000 | 535.2 | 0.00 | rank_necessity_drop |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | +0.288 | -0.384 | -0.673 | -0.673 | +0.000 | 8709.6 | 0.00 | rank_necessity_drop |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.288 | -1.441 | -1.730 | -0.673 | -1.057 | 31670.4 | 0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.288 | -1.371 | -1.659 | -0.596 | -1.063 | 30830.7 | 0.00 | restore_fail |

## deepseek7b

pair=vehicle_tool, window=[16, 18, 20], combos=['all'], routes=['forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12

| model | combo | layers | route | condition | donor category | donor variant | base margin | margin | margin delta | remove delta | restore gain | target rank | top1 | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | +0.807 | +1.757 | +0.949 | +0.949 | +0.000 | 210.1 | 0.00 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | +0.559 | +2.706 | +2.148 | +2.148 | +0.000 | 277.5 | 0.00 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.559 | +2.625 | +2.066 | +2.148 | -0.081 | 107.1 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.559 | +2.552 | +1.993 | +2.148 | -0.154 | 119.7 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | +0.807 | +1.473 | +0.665 | +0.949 | -0.284 | 212.8 | 0.00 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat0_add | vehicle | repeat0 | +0.807 | +0.875 | +0.068 | +0.949 | -0.882 | 103.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat1_add | vehicle | repeat1 | +0.807 | +0.779 | -0.029 | +0.949 | -0.978 | 97.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | +0.807 | +0.750 | -0.057 | +0.949 | -1.007 | 238.8 | 0.00 | rank_necessity_drop |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.559 | +1.457 | +0.898 | +2.148 | -1.249 | 241.8 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.807 | +0.457 | -0.350 | +0.949 | -1.299 | 11195.2 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.559 | +1.383 | +0.824 | +2.148 | -1.324 | 126.6 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.559 | +1.311 | +0.753 | +2.148 | -1.395 | 202.2 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | +0.559 | +1.163 | +0.604 | +2.148 | -1.544 | 291.5 | 0.00 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.559 | +1.163 | +0.604 | +2.148 | -1.544 | 291.5 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.559 | +1.128 | +0.569 | +2.148 | -1.579 | 151.6 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.559 | +1.086 | +0.527 | +2.148 | -1.620 | 210.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.559 | +1.068 | +0.509 | +2.148 | -1.639 | 175.7 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat5_add | vehicle | repeat5 | +0.807 | +0.115 | -0.693 | +0.949 | -1.642 | 187.4 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | +0.807 | -0.333 | -1.141 | +0.949 | -2.090 | 153.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat3_add | vehicle | repeat3 | +0.807 | -0.375 | -1.182 | +0.949 | -2.132 | 92.2 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | +0.807 | -0.385 | -1.193 | +0.949 | -2.142 | 201.4 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | +0.559 | +0.510 | -0.048 | +2.148 | -2.196 | 299.3 | 0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca1_cache_add | vehicle | pca1_cache | +0.807 | -0.482 | -1.289 | +0.949 | -2.238 | 128.3 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_pca3_cache_add | vehicle | pca3_cache | +0.807 | -0.547 | -1.354 | +0.949 | -2.303 | 152.6 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_mean_cache_add | vehicle | mean_cache | +0.807 | -0.596 | -1.404 | +0.949 | -2.353 | 123.2 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.559 | +0.161 | -0.397 | +2.148 | -2.545 | 253.3 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.559 | +0.044 | -0.514 | +2.148 | -2.662 | 458.6 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | +0.559 | +0.035 | -0.523 | +2.148 | -2.671 | 20891.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.559 | -0.068 | -0.626 | +2.148 | -2.774 | 204.9 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | +0.807 | -1.186 | -1.993 | +0.949 | -2.943 | 203.1 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat4_add | vehicle | repeat4 | +0.807 | -1.294 | -2.102 | +0.949 | -3.051 | 248.2 | 0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat2_add | vehicle | repeat2 | +0.807 | -1.755 | -2.563 | +0.949 | -3.512 | 151.6 | 0.00 | restore_fail |

## Prototype Split

| model | combo | route | same | shuffle | best repeat | best repeat id | mean | pca1 | pca3 | random | same-shuffle | repeat-mean | pca3-random |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | all | forbidden_definition:top_p<-forbidden_definition | +0.474 | +0.448 | +0.672 | repeat0 | +0.487 | +0.495 | +0.451 | -0.753 | +0.026 | +0.185 | +1.203 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +0.677 | +0.651 | +0.740 | repeat0 | +0.635 | +0.667 | +0.646 | +0.525 | +0.026 | +0.104 | +0.121 |
| glm4 | L24 | forbidden_definition:top_p<-forbidden_definition | +1.411 | +1.679 | +2.968 | repeat2 | +2.064 | +1.577 | +1.498 | +0.666 | -0.268 | +0.904 | +0.832 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_definition | +2.620 | +2.677 | +4.103 | repeat2 | +3.496 | +2.710 | +2.644 | +1.958 | -0.057 | +0.606 | +0.686 |
| glm4 | L24+L28 | forbidden_definition:top_p<-forbidden_definition | +2.530 | +2.685 | +4.005 | repeat2 | +3.432 | +2.725 | +2.647 | -1.063 | -0.155 | +0.572 | +3.710 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_definition | +3.597 | +3.667 | +4.809 | repeat2 | +4.340 | +3.703 | +3.706 | +0.226 | -0.070 | +0.469 | +3.480 |
| glm4 | all | forbidden_definition:top_p<-forbidden_definition | +3.548 | +3.669 | +4.762 | repeat2 | +4.757 | +3.836 | +3.723 | -1.057 | -0.121 | +0.005 | +4.780 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +4.541 | +4.579 | +5.232 | repeat2 | +5.646 | +4.734 | +4.632 | +0.150 | -0.038 | -0.414 | +4.482 |
| deepseek7b | all | forbidden_definition:top_p<-forbidden_definition | -1.544 | -1.395 | -0.081 | repeat0 | -1.579 | -1.639 | -1.620 | -2.671 | -0.148 | +1.497 | +1.051 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_definition | -2.142 | -2.090 | -0.882 | repeat0 | -2.353 | -2.238 | -2.303 | -1.299 | -0.052 | +1.471 | -1.004 |

## Repeat Matrix

| model | combo | route | repeat0 | repeat1 | repeat2 | repeat3 | repeat4 | repeat5 | spread |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | all | forbidden_definition:top_p<-forbidden_definition | +0.672 | +0.336 | +0.492 | +0.583 | +0.211 | +0.315 | +0.461 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +0.740 | +0.667 | +0.661 | +0.682 | +0.630 | +0.615 | +0.125 |
| glm4 | L24 | forbidden_definition:top_p<-forbidden_definition | +2.112 | +1.866 | +2.968 | +0.480 | +2.778 | +2.698 | +2.487 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_definition | +3.031 | +2.486 | +4.103 | +2.312 | +3.725 | +3.672 | +1.791 |
| glm4 | L24+L28 | forbidden_definition:top_p<-forbidden_definition | +2.798 | +2.735 | +4.005 | +1.194 | +3.469 | +3.655 | +2.811 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_definition | +3.578 | +3.562 | +4.809 | +2.589 | +4.554 | +4.552 | +2.220 |
| glm4 | all | forbidden_definition:top_p<-forbidden_definition | +3.453 | +3.357 | +4.762 | +2.334 | +4.308 | +4.033 | +2.428 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +3.906 | +3.842 | +5.232 | +3.414 | +5.192 | +4.740 | +1.818 |
| deepseek7b | all | forbidden_definition:top_p<-forbidden_definition | -0.081 | -0.154 | -2.774 | -1.324 | -2.662 | -1.249 | +2.693 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_definition | -0.882 | -0.978 | -3.512 | -2.132 | -3.051 | -1.642 | +2.630 |
