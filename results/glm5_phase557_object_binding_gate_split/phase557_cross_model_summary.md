# Phase557 Object-Binding and Generic-Gate Split Summary

## qwen3

pair=vehicle_tool, window=[10, 12, 14], combos=['L14', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_same_add', 'resid_donor_vehicle_near_add', 'resid_donor_vehicle_far_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_vehicle_repeat_add', 'resid_donor_vehicle_random_cache_add', 'resid_donor_tool_same_add', 'resid_donor_furniture_same_add', 'resid_donor_animal_same_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.22 | 0.42 | +0.20 | +0.02 | +0.18 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.74 | +0.20 | +0.04 | +0.16 | +0.02 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.22 | 0.36 | +0.15 | +0.02 | +0.12 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.22 | 0.35 | +0.14 | +0.02 | +0.11 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.71 | +0.17 | +0.06 | +0.10 | +0.04 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.22 | 0.28 | +0.06 | +0.02 | +0.04 | -0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.62 | +0.08 | +0.04 | +0.04 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.62 | +0.08 | +0.06 | +0.02 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.22 | 0.20 | -0.02 | +0.02 | -0.04 | +0.00 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.54 | +0.00 | +0.04 | -0.04 | +0.05 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.22 | 0.17 | -0.05 | +0.02 | -0.07 | -0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.54 | 0.50 | -0.04 | +0.04 | -0.08 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.22 | 0.15 | -0.07 | +0.02 | -0.09 | -0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.51 | -0.03 | +0.06 | -0.09 | +0.08 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.54 | 0.49 | -0.05 | +0.04 | -0.09 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.48 | -0.06 | +0.04 | -0.10 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.22 | 0.12 | -0.09 | +0.02 | -0.11 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.22 | 0.11 | -0.10 | +0.02 | -0.12 | -0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.47 | -0.07 | +0.06 | -0.14 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.47 | -0.07 | +0.06 | -0.14 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.54 | 0.44 | -0.10 | +0.06 | -0.17 | +0.06 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.54 | 0.40 | -0.15 | +0.04 | -0.19 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.41 | -0.14 | +0.06 | -0.20 | +0.07 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.54 | 0.40 | -0.15 | +0.06 | -0.21 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.54 | 0.38 | -0.17 | +0.04 | -0.21 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.54 | 0.38 | -0.17 | +0.06 | -0.23 | +0.04 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.22 | 0.01 | -0.21 | +0.02 | -0.23 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.22 | 0.00 | -0.22 | +0.02 | -0.24 | -0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.54 | 0.33 | -0.21 | +0.06 | -0.27 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.54 | 0.31 | -0.23 | +0.04 | -0.27 | +0.04 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.54 | 0.32 | -0.22 | +0.06 | -0.28 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.54 | 0.30 | -0.24 | +0.04 | -0.28 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.54 | 0.31 | -0.23 | +0.06 | -0.29 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.54 | 0.28 | -0.26 | +0.06 | -0.32 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.54 | 0.25 | -0.29 | +0.04 | -0.33 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.54 | 0.22 | -0.32 | +0.04 | -0.36 | +0.00 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.54 | 0.08 | -0.46 | +0.04 | -0.50 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.54 | 0.07 | -0.47 | +0.04 | -0.51 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.54 | 0.06 | -0.48 | +0.06 | -0.54 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.54 | 0.05 | -0.49 | +0.06 | -0.55 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.22 | 0.30 | +0.08 | +0.02 | +0.06 | +0.03 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.54 | 0.64 | +0.09 | +0.04 | +0.05 | +0.00 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.54 | 0.64 | +0.09 | +0.04 | +0.05 | +0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.54 | 0.64 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.54 | 0.64 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | -0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.22 | 0.24 | +0.02 | +0.02 | +0.00 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.22 | 0.24 | +0.02 | +0.02 | +0.00 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.54 | 0.58 | +0.04 | +0.04 | +0.00 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.54 | 0.58 | +0.04 | +0.04 | +0.00 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.54 | 0.60 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.54 | 0.60 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.54 | 0.56 | +0.02 | +0.06 | -0.04 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.54 | 0.56 | +0.02 | +0.06 | -0.04 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.22 | 0.20 | -0.02 | +0.02 | -0.04 | -0.02 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.00 | weak_drop |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.00 | weak_drop |

## glm4

pair=vehicle_tool, window=[24, 26, 28], combos=['L24', 'L24+L28', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_same_add', 'resid_donor_vehicle_near_add', 'resid_donor_vehicle_far_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_vehicle_repeat_add', 'resid_donor_vehicle_random_cache_add', 'resid_donor_tool_same_add', 'resid_donor_furniture_same_add', 'resid_donor_animal_same_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.72 | +0.43 | -0.20 | +0.62 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.75 | +0.46 | -0.11 | +0.57 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.74 | +0.45 | -0.10 | +0.55 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.81 | +0.43 | -0.09 | +0.52 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.70 | +0.41 | -0.10 | +0.51 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.68 | +0.39 | -0.11 | +0.50 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.56 | +0.27 | -0.20 | +0.47 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.73 | +0.34 | -0.09 | +0.44 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.59 | +0.30 | -0.10 | +0.41 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.57 | +0.28 | -0.10 | +0.39 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.67 | +0.28 | -0.09 | +0.37 | -0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.43 | +0.14 | -0.20 | +0.33 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.51 | +0.22 | -0.11 | +0.33 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.49 | +0.20 | -0.10 | +0.30 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.45 | +0.16 | -0.10 | +0.26 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.42 | +0.12 | -0.11 | +0.24 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.50 | +0.11 | -0.09 | +0.21 | -0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.49 | +0.10 | -0.09 | +0.20 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.27 | -0.02 | -0.20 | +0.18 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.32 | +0.03 | -0.10 | +0.14 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.31 | +0.02 | -0.11 | +0.14 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.29 | +0.00 | -0.11 | +0.11 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.41 | +0.02 | -0.09 | +0.11 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.40 | +0.01 | -0.09 | +0.10 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.16 | -0.14 | -0.20 | +0.06 | +0.00 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.03 | weak_restore |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.29 | 0.11 | -0.18 | -0.20 | +0.02 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.29 | 0.07 | -0.22 | -0.20 | -0.02 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.06 | -0.23 | -0.20 | -0.03 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.15 | -0.15 | -0.10 | -0.04 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.29 | 0.10 | -0.19 | -0.11 | -0.07 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.29 | 0.11 | -0.18 | -0.10 | -0.07 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.01 | -0.28 | -0.20 | -0.08 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.01 | -0.28 | -0.20 | -0.08 | +0.04 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.09 | -0.20 | -0.11 | -0.08 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.05 | -0.24 | -0.11 | -0.12 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.06 | -0.23 | -0.10 | -0.12 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.39 | 0.17 | -0.22 | -0.09 | -0.13 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.03 | -0.26 | -0.11 | -0.15 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.29 | 0.04 | -0.25 | -0.10 | -0.15 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.03 | -0.26 | -0.11 | -0.15 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.39 | 0.14 | -0.25 | -0.09 | -0.16 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.39 | 0.12 | -0.26 | -0.09 | -0.17 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | -0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.39 | 0.06 | -0.32 | -0.09 | -0.23 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.39 | 0.05 | -0.33 | -0.09 | -0.24 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.39 | 0.04 | -0.34 | -0.09 | -0.25 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.39 | 0.03 | -0.35 | -0.09 | -0.26 | -0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.39 | 0.02 | -0.36 | -0.09 | -0.27 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.39 | 0.02 | -0.36 | -0.09 | -0.27 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.29 | 0.56 | +0.27 | -0.10 | +0.38 | +0.00 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.29 | 0.56 | +0.27 | -0.10 | +0.38 | +0.00 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.29 | 0.49 | +0.20 | -0.11 | +0.31 | +0.00 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.29 | 0.49 | +0.20 | -0.11 | +0.31 | +0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.29 | 0.36 | +0.07 | -0.20 | +0.27 | +0.01 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.29 | 0.36 | +0.07 | -0.20 | +0.27 | +0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.39 | 0.53 | +0.15 | -0.09 | +0.24 | -0.01 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.29 | 0.29 | +0.00 | -0.20 | +0.20 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.29 | 0.29 | +0.00 | -0.20 | +0.20 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.39 | 0.42 | +0.03 | -0.09 | +0.12 | +0.01 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.29 | 0.30 | +0.01 | -0.10 | +0.11 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.29 | 0.30 | +0.01 | -0.10 | +0.11 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.39 | 0.40 | +0.01 | -0.09 | +0.10 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.29 | 0.27 | -0.02 | -0.11 | +0.09 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.29 | 0.27 | -0.02 | -0.11 | +0.09 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.39 | 0.38 | -0.01 | -0.09 | +0.08 | +0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.39 | 0.35 | -0.03 | -0.09 | +0.06 | +0.02 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.29 | 0.09 | -0.20 | -0.20 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.29 | 0.09 | -0.20 | -0.20 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.29 | 0.18 | -0.11 | -0.11 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.29 | 0.18 | -0.11 | -0.11 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | +0.00 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.29 | 0.19 | -0.10 | -0.10 | +0.00 | +0.00 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.29 | 0.19 | -0.10 | -0.10 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | -0.01 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | +0.01 | weak_drop |

## deepseek7b

pair=vehicle_tool, window=[16, 18, 20], combos=['L16+L20', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_same_add', 'resid_donor_vehicle_near_add', 'resid_donor_vehicle_far_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_vehicle_repeat_add', 'resid_donor_vehicle_random_cache_add', 'resid_donor_tool_same_add', 'resid_donor_furniture_same_add', 'resid_donor_animal_same_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.16 | 0.49 | +0.33 | +0.05 | +0.28 | +0.05 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.16 | 0.45 | +0.29 | +0.03 | +0.26 | +0.11 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.40 | +0.22 | +0.04 | +0.18 | +0.24 | restore_without_drop_or_leaky |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.41 | +0.23 | +0.06 | +0.17 | +0.18 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.31 | +0.14 | +0.04 | +0.09 | +0.01 | restore_without_drop_or_leaky |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.28 | +0.10 | +0.06 | +0.04 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.16 | 0.23 | +0.07 | +0.03 | +0.04 | +0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.23 | +0.05 | +0.04 | +0.01 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.16 | 0.20 | +0.04 | +0.05 | -0.01 | +0.08 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.20 | +0.02 | +0.06 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.16 | 0.16 | +0.00 | +0.05 | -0.05 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.09 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.17 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.18 | 0.16 | -0.02 | +0.04 | -0.06 | +0.14 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.16 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.18 | 0.16 | -0.02 | +0.06 | -0.08 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.18 | 0.16 | -0.02 | +0.06 | -0.08 | +0.16 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.16 | 0.11 | -0.04 | +0.05 | -0.09 | -0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.18 | 0.11 | -0.06 | +0.04 | -0.10 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.16 | 0.08 | -0.07 | +0.03 | -0.10 | -0.04 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.18 | 0.12 | -0.05 | +0.06 | -0.11 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.16 | 0.09 | -0.06 | +0.05 | -0.11 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.16 | 0.09 | -0.06 | +0.05 | -0.11 | +0.04 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.18 | 0.11 | -0.06 | +0.06 | -0.12 | -0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.18 | 0.09 | -0.08 | +0.04 | -0.12 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.16 | 0.06 | -0.09 | +0.03 | -0.12 | +0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.18 | 0.06 | -0.11 | +0.04 | -0.16 | +0.09 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.18 | 0.05 | -0.12 | +0.04 | -0.17 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.18 | 0.05 | -0.12 | +0.04 | -0.17 | +0.10 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.18 | 0.06 | -0.11 | +0.06 | -0.18 | +0.05 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.18 | 0.06 | -0.11 | +0.06 | -0.18 | +0.06 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.16 | 0.00 | -0.16 | +0.03 | -0.19 | -0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.18 | 0.05 | -0.12 | +0.06 | -0.19 | +0.15 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.16 | 0.00 | -0.16 | +0.05 | -0.21 | -0.10 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.18 | 0.00 | -0.18 | +0.04 | -0.22 | -0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.18 | 0.00 | -0.18 | +0.04 | -0.22 | -0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.18 | 0.00 | -0.18 | +0.06 | -0.24 | -0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_random_cache_add | vehicle | random_cache | 0.18 | 0.00 | -0.18 | +0.06 | -0.24 | -0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.18 | 0.27 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.18 | 0.27 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.16 | 0.19 | +0.03 | +0.03 | +0.00 | +0.11 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  | 0.16 | 0.21 | +0.05 | +0.05 | +0.00 | +0.11 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.16 | 0.18 | +0.02 | +0.03 | -0.01 | +0.09 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.16 | 0.18 | +0.02 | +0.05 | -0.03 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  | 0.16 | 0.18 | +0.02 | +0.05 | -0.03 | +0.09 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.18 | 0.19 | +0.01 | +0.06 | -0.05 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.18 | 0.19 | +0.01 | +0.06 | -0.05 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  | 0.16 | 0.11 | -0.04 | +0.03 | -0.07 | +0.05 | flat |

## Object Binding Split

| model | combo | route | same | near | far | shuffle | repeat | random cache | same-near | same-far | same-shuffle | same-random |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | L14 | forbidden_definition:top_p<-forbidden_definition | -0.04 | +0.11 | +0.05 | +0.05 | +0.12 | -0.24 | -0.16 | -0.09 | -0.09 | +0.20 |
| qwen3 | L14 | forbidden_sentence_completion:temperature<-forbidden_definition | -0.21 | -0.10 | -0.11 | -0.11 | -0.04 | -0.50 | -0.10 | -0.09 | -0.09 | +0.29 |
| qwen3 | L14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | -0.09 | +0.04 | -0.01 | -0.01 | +0.16 | -0.51 | -0.14 | -0.08 | -0.08 | +0.42 |
| qwen3 | all | forbidden_definition:top_p<-forbidden_definition | +0.04 | +0.02 | +0.14 | +0.14 | +0.18 | -0.23 | +0.02 | -0.09 | -0.09 | +0.27 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_definition | -0.21 | -0.20 | -0.14 | -0.14 | -0.09 | -0.55 | -0.01 | -0.07 | -0.07 | +0.34 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.01 | +0.02 | +0.01 | +0.01 | +0.10 | -0.54 | -0.01 | +0.00 | +0.00 | +0.55 |
| glm4 | L24 | forbidden_definition:top_p<-forbidden_definition | +0.10 | +0.11 | +0.26 | +0.26 | +0.37 | -0.27 | -0.01 | -0.16 | -0.16 | +0.38 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_definition | +0.34 | +0.34 | +0.38 | +0.38 | +0.62 | -0.08 | +0.00 | -0.03 | -0.03 | +0.43 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.33 | +0.31 | +0.31 | +0.31 | +0.47 | -0.08 | +0.02 | +0.02 | +0.02 | +0.42 |
| glm4 | L24+L28 | forbidden_definition:top_p<-forbidden_definition | +0.17 | +0.20 | +0.17 | +0.17 | +0.44 | -0.27 | -0.03 | +0.00 | +0.00 | +0.44 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_definition | +0.35 | +0.33 | +0.40 | +0.40 | +0.57 | -0.12 | +0.02 | -0.04 | -0.04 | +0.48 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.24 | +0.35 | +0.29 | +0.29 | +0.50 | -0.15 | -0.11 | -0.05 | -0.05 | +0.39 |
| glm4 | all | forbidden_definition:top_p<-forbidden_definition | +0.21 | +0.26 | +0.19 | +0.19 | +0.52 | -0.24 | -0.05 | +0.02 | +0.02 | +0.45 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +0.39 | +0.29 | +0.29 | +0.29 | +0.51 | -0.12 | +0.09 | +0.09 | +0.09 | +0.51 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.26 | +0.41 | +0.23 | +0.23 | +0.55 | -0.15 | -0.15 | +0.03 | +0.03 | +0.41 |
| deepseek7b | L16+L20 | forbidden_definition:top_p<-forbidden_definition | +0.04 | -0.04 | +0.03 | +0.03 | +0.26 | -0.19 | +0.08 | +0.01 | +0.01 | +0.23 |
| deepseek7b | L16+L20 | forbidden_sentence_completion:temperature<-forbidden_definition | -0.06 | -0.05 | +0.02 | +0.02 | +0.18 | -0.22 | -0.01 | -0.08 | -0.08 | +0.16 |
| deepseek7b | L16+L20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | -0.05 | +0.09 | -0.04 | -0.04 | +0.01 | -0.22 | -0.15 | -0.01 | -0.01 | +0.17 |
| deepseek7b | all | forbidden_definition:top_p<-forbidden_definition | -0.01 | -0.05 | +0.05 | +0.05 | +0.28 | -0.21 | +0.04 | -0.06 | -0.06 | +0.20 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_definition | -0.08 | -0.07 | +0.00 | +0.00 | +0.17 | -0.24 | -0.01 | -0.08 | -0.08 | +0.16 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | -0.08 | +0.04 | +0.00 | +0.00 | -0.04 | -0.24 | -0.12 | -0.08 | -0.08 | +0.16 |

## Strongest Restores

| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.72 | +0.43 | -0.20 | +0.62 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.75 | +0.46 | -0.11 | +0.57 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.74 | +0.45 | -0.10 | +0.55 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.81 | +0.43 | -0.09 | +0.52 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.70 | +0.41 | -0.10 | +0.51 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.68 | +0.39 | -0.11 | +0.50 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.29 | 0.56 | +0.27 | -0.20 | +0.47 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.73 | +0.34 | -0.09 | +0.44 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.59 | +0.30 | -0.10 | +0.41 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.57 | +0.28 | -0.10 | +0.39 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.39 | 0.67 | +0.28 | -0.09 | +0.37 | -0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.43 | +0.14 | -0.20 | +0.33 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.51 | +0.22 | -0.11 | +0.33 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.49 | +0.20 | -0.10 | +0.30 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.45 | +0.16 | -0.10 | +0.26 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.29 | 0.42 | +0.12 | -0.11 | +0.24 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.50 | +0.11 | -0.09 | +0.21 | -0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.49 | +0.10 | -0.09 | +0.20 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.27 | -0.02 | -0.20 | +0.18 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.32 | +0.03 | -0.10 | +0.14 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_same_add | animal | same | 0.29 | 0.31 | +0.02 | -0.11 | +0.14 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.29 | +0.00 | -0.11 | +0.11 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.39 | 0.41 | +0.02 | -0.09 | +0.11 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.39 | 0.40 | +0.01 | -0.09 | +0.10 | +0.00 | restore_success |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.16 | 0.49 | +0.33 | +0.05 | +0.28 | +0.05 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.16 | 0.45 | +0.29 | +0.03 | +0.26 | +0.11 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.22 | 0.42 | +0.20 | +0.02 | +0.18 | -0.01 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.40 | +0.22 | +0.04 | +0.18 | +0.24 | restore_without_drop_or_leaky |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.41 | +0.23 | +0.06 | +0.17 | +0.18 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.74 | +0.20 | +0.04 | +0.16 | +0.02 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.22 | 0.36 | +0.15 | +0.02 | +0.12 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.22 | 0.35 | +0.14 | +0.02 | +0.11 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.54 | 0.71 | +0.17 | +0.06 | +0.10 | +0.04 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.31 | +0.14 | +0.04 | +0.09 | +0.01 | restore_without_drop_or_leaky |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.16 | -0.14 | -0.20 | +0.06 | +0.00 | weak_restore |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.22 | 0.28 | +0.06 | +0.02 | +0.04 | -0.02 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_same_add | tool | same | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.03 | weak_restore |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.16 | 0.23 | +0.07 | +0.03 | +0.04 | +0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.18 | 0.28 | +0.10 | +0.06 | +0.04 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.62 | +0.08 | +0.04 | +0.04 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_near_add | vehicle | near | 0.54 | 0.62 | +0.08 | +0.06 | +0.02 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | +0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_same_add | animal | same | 0.29 | 0.11 | -0.18 | -0.20 | +0.02 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_same_add | vehicle | same | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_repeat_add | vehicle | repeat | 0.18 | 0.23 | +0.05 | +0.04 | +0.01 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_same_add | vehicle | same | 0.16 | 0.20 | +0.04 | +0.05 | -0.01 | +0.08 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_same_add | tool | same | 0.29 | 0.07 | -0.22 | -0.20 | -0.02 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.06 | -0.23 | -0.20 | -0.03 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_same_add | furniture | same | 0.29 | 0.15 | -0.15 | -0.10 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_far_add | vehicle | far | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_near_add | vehicle | near | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |

