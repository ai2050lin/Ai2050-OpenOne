# Phase556 Donor Specificity Decomposition Summary

## qwen3

pair=vehicle_tool, window=[10, 12, 14], combos=['L14', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_add', 'resid_donor_tool_add', 'resid_donor_furniture_add', 'resid_donor_animal_add', 'resid_donor_fruit_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_tool_shuffle_add', 'resid_donor_furniture_shuffle_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.22 | 0.35 | +0.14 | +0.02 | +0.11 | -0.01 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.62 | +0.08 | +0.04 | +0.04 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.62 | +0.08 | +0.06 | +0.02 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.22 | 0.22 | +0.00 | +0.02 | -0.02 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.22 | 0.21 | -0.01 | +0.02 | -0.03 | -0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.22 | 0.20 | -0.02 | +0.02 | -0.04 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.22 | 0.19 | -0.03 | +0.02 | -0.05 | -0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.22 | 0.17 | -0.05 | +0.02 | -0.07 | -0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.54 | 0.50 | -0.04 | +0.04 | -0.08 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.22 | 0.15 | -0.07 | +0.02 | -0.09 | -0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.48 | -0.06 | +0.04 | -0.10 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.22 | 0.12 | -0.09 | +0.02 | -0.11 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.22 | 0.11 | -0.10 | +0.02 | -0.12 | -0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.47 | -0.07 | +0.06 | -0.14 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.54 | 0.43 | -0.11 | +0.04 | -0.16 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.54 | 0.44 | -0.10 | +0.06 | -0.17 | +0.06 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.54 | 0.40 | -0.15 | +0.04 | -0.19 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.54 | 0.41 | -0.14 | +0.06 | -0.20 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.41 | -0.14 | +0.06 | -0.20 | +0.07 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.54 | 0.39 | -0.16 | +0.06 | -0.22 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.54 | 0.38 | -0.17 | +0.06 | -0.23 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.54 | 0.35 | -0.19 | +0.04 | -0.23 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.54 | 0.36 | -0.18 | +0.06 | -0.24 | +0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.54 | 0.34 | -0.20 | +0.04 | -0.24 | +0.05 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.54 | 0.34 | -0.20 | +0.06 | -0.26 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.54 | 0.34 | -0.20 | +0.06 | -0.26 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.54 | 0.33 | -0.21 | +0.06 | -0.27 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.54 | 0.33 | -0.21 | +0.06 | -0.27 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.54 | 0.31 | -0.23 | +0.04 | -0.27 | +0.04 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.54 | 0.32 | -0.22 | +0.06 | -0.28 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.54 | 0.30 | -0.24 | +0.04 | -0.28 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.54 | 0.30 | -0.24 | +0.04 | -0.28 | +0.04 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.54 | 0.31 | -0.23 | +0.06 | -0.29 | +0.00 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.54 | 0.29 | -0.25 | +0.04 | -0.29 | +0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.54 | 0.28 | -0.26 | +0.06 | -0.32 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.54 | 0.25 | -0.29 | +0.04 | -0.33 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.54 | 0.25 | -0.29 | +0.04 | -0.33 | +0.02 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.54 | 0.22 | -0.32 | +0.04 | -0.36 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.22 | 0.30 | +0.08 | +0.02 | +0.06 | +0.03 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.54 | 0.64 | +0.09 | +0.04 | +0.05 | +0.00 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.54 | 0.64 | +0.09 | +0.04 | +0.05 | +0.00 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.54 | 0.64 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.54 | 0.64 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | -0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.22 | 0.24 | +0.02 | +0.02 | +0.00 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.22 | 0.24 | +0.02 | +0.02 | +0.00 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.54 | 0.58 | +0.04 | +0.04 | +0.00 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.54 | 0.58 | +0.04 | +0.04 | +0.00 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.54 | 0.60 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.54 | 0.60 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.54 | 0.56 | +0.02 | +0.06 | -0.04 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.54 | 0.56 | +0.02 | +0.06 | -0.04 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.22 | 0.20 | -0.02 | +0.02 | -0.04 | -0.02 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.00 | weak_drop |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.54 | 0.47 | -0.07 | +0.04 | -0.11 | +0.00 | weak_drop |

## glm4

pair=vehicle_tool, window=[24, 26, 28], combos=['L24', 'L24+L28', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_add', 'resid_donor_tool_add', 'resid_donor_furniture_add', 'resid_donor_animal_add', 'resid_donor_fruit_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_tool_shuffle_add', 'resid_donor_furniture_shuffle_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.59 | +0.30 | -0.10 | +0.41 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.51 | +0.22 | -0.11 | +0.33 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.49 | +0.20 | -0.10 | +0.30 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.39 | +0.09 | -0.10 | +0.20 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.49 | +0.10 | -0.09 | +0.20 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.27 | -0.02 | -0.20 | +0.18 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.32 | +0.03 | -0.10 | +0.14 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.31 | +0.02 | -0.11 | +0.14 | +0.02 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.29 | +0.00 | -0.11 | +0.11 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.21 | -0.08 | -0.20 | +0.11 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.41 | +0.02 | -0.09 | +0.11 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.28 | -0.01 | -0.10 | +0.09 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.26 | -0.03 | -0.10 | +0.07 | +0.00 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.16 | -0.14 | -0.20 | +0.06 | +0.00 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.00 | weak_restore |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.22 | -0.07 | -0.11 | +0.04 | +0.00 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.02 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.03 | weak_restore |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.12 | -0.17 | -0.20 | +0.03 | +0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.11 | -0.18 | -0.20 | +0.02 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.19 | -0.10 | -0.11 | +0.01 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.07 | -0.22 | -0.20 | -0.02 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.16 | -0.14 | -0.11 | -0.02 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.06 | -0.23 | -0.20 | -0.03 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.15 | -0.15 | -0.10 | -0.04 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.05 | -0.24 | -0.20 | -0.04 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.05 | -0.24 | -0.20 | -0.04 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.10 | -0.19 | -0.11 | -0.07 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.11 | -0.18 | -0.10 | -0.07 | +0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.39 | 0.22 | -0.17 | -0.09 | -0.07 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.09 | -0.20 | -0.11 | -0.08 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.39 | 0.21 | -0.18 | -0.09 | -0.08 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.09 | -0.20 | -0.10 | -0.09 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.39 | 0.19 | -0.20 | -0.09 | -0.10 | +0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.05 | -0.24 | -0.11 | -0.12 | +0.02 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.39 | 0.17 | -0.22 | -0.09 | -0.13 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.39 | 0.17 | -0.22 | -0.09 | -0.13 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.03 | -0.26 | -0.11 | -0.15 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.39 | 0.14 | -0.25 | -0.09 | -0.16 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.39 | 0.12 | -0.26 | -0.09 | -0.17 | -0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.39 | 0.10 | -0.28 | -0.09 | -0.19 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.39 | 0.08 | -0.30 | -0.09 | -0.21 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.39 | 0.08 | -0.30 | -0.09 | -0.21 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.39 | 0.07 | -0.31 | -0.09 | -0.22 | -0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.39 | 0.06 | -0.32 | -0.09 | -0.23 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.39 | 0.05 | -0.33 | -0.09 | -0.24 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.39 | 0.04 | -0.34 | -0.09 | -0.25 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.39 | 0.03 | -0.35 | -0.09 | -0.26 | -0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.39 | 0.02 | -0.36 | -0.09 | -0.27 | -0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.29 | 0.56 | +0.27 | -0.10 | +0.38 | +0.00 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.29 | 0.56 | +0.27 | -0.10 | +0.38 | +0.00 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.29 | 0.49 | +0.20 | -0.11 | +0.31 | +0.00 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.29 | 0.49 | +0.20 | -0.11 | +0.31 | +0.00 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.29 | 0.36 | +0.07 | -0.20 | +0.27 | +0.01 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.29 | 0.36 | +0.07 | -0.20 | +0.27 | +0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.39 | 0.53 | +0.15 | -0.09 | +0.24 | -0.01 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.29 | 0.29 | +0.00 | -0.20 | +0.20 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.29 | 0.29 | +0.00 | -0.20 | +0.20 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.39 | 0.42 | +0.03 | -0.09 | +0.12 | +0.01 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.29 | 0.30 | +0.01 | -0.10 | +0.11 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.29 | 0.30 | +0.01 | -0.10 | +0.11 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.39 | 0.40 | +0.01 | -0.09 | +0.10 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.29 | 0.27 | -0.02 | -0.11 | +0.09 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.29 | 0.27 | -0.02 | -0.11 | +0.09 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.39 | 0.38 | -0.01 | -0.09 | +0.08 | +0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.39 | 0.35 | -0.03 | -0.09 | +0.06 | +0.02 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.29 | 0.09 | -0.20 | -0.20 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.29 | 0.09 | -0.20 | -0.20 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.29 | 0.18 | -0.11 | -0.11 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.29 | 0.18 | -0.11 | -0.11 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | +0.00 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.29 | 0.19 | -0.10 | -0.10 | +0.00 | +0.00 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.29 | 0.19 | -0.10 | -0.10 | +0.00 | +0.00 | necessity_drop |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | -0.01 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.39 | 0.29 | -0.09 | -0.09 | +0.00 | +0.01 | weak_drop |

## deepseek7b

pair=vehicle_tool, window=[16, 18, 20], combos=['L16+L20', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_random_perp', 'resid_donor_vehicle_add', 'resid_donor_tool_add', 'resid_donor_furniture_add', 'resid_donor_animal_add', 'resid_donor_fruit_add', 'resid_donor_vehicle_shuffle_add', 'resid_donor_tool_shuffle_add', 'resid_donor_furniture_shuffle_add'], routes=['forbidden_sentence_completion:temperature<-forbidden_sentence_completion', 'forbidden_sentence_completion:temperature<-forbidden_definition', 'forbidden_definition:top_p<-forbidden_definition'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137]

| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.31 | +0.14 | +0.04 | +0.09 | +0.01 | restore_without_drop_or_leaky |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.28 | +0.10 | +0.06 | +0.04 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.16 | 0.17 | +0.01 | +0.03 | -0.02 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.16 | 0.19 | +0.03 | +0.05 | -0.02 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.16 | 0.16 | +0.00 | +0.05 | -0.05 | +0.05 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.09 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.17 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.16 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.16 | 0.12 | -0.03 | +0.05 | -0.08 | +0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.16 | -0.02 | +0.06 | -0.08 | +0.11 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.16 | 0.11 | -0.04 | +0.05 | -0.09 | -0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.11 | -0.06 | +0.04 | -0.10 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.16 | 0.08 | -0.07 | +0.03 | -0.10 | -0.04 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.12 | -0.05 | +0.06 | -0.11 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.16 | 0.09 | -0.06 | +0.05 | -0.11 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.16 | 0.09 | -0.06 | +0.05 | -0.11 | +0.04 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.16 | 0.06 | -0.09 | +0.03 | -0.12 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.18 | 0.11 | -0.06 | +0.06 | -0.12 | -0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.18 | 0.09 | -0.08 | +0.04 | -0.12 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.18 | 0.09 | -0.08 | +0.04 | -0.12 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.16 | 0.06 | -0.09 | +0.03 | -0.12 | +0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.18 | 0.10 | -0.07 | +0.06 | -0.14 | +0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.18 | 0.10 | -0.07 | +0.06 | -0.14 | +0.04 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.18 | 0.07 | -0.10 | +0.04 | -0.15 | +0.06 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.18 | 0.09 | -0.08 | +0.06 | -0.15 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.16 | 0.04 | -0.11 | +0.03 | -0.15 | +0.04 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.18 | 0.06 | -0.11 | +0.04 | -0.16 | +0.09 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.05 | -0.12 | +0.04 | -0.17 | +0.01 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.18 | 0.05 | -0.12 | +0.04 | -0.17 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.05 | -0.12 | +0.04 | -0.17 | +0.10 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.16 | 0.03 | -0.12 | +0.05 | -0.18 | +0.04 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.06 | -0.11 | +0.06 | -0.18 | +0.05 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.18 | 0.06 | -0.11 | +0.06 | -0.18 | +0.06 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.03 | -0.15 | +0.04 | -0.19 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.18 | 0.03 | -0.15 | +0.04 | -0.19 | +0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.05 | -0.12 | +0.06 | -0.19 | +0.15 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.18 | 0.04 | -0.14 | +0.06 | -0.20 | +0.17 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.02 | -0.16 | +0.06 | -0.22 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.18 | 0.27 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.18 | 0.27 | +0.09 | +0.06 | +0.03 | +0.02 | positive_add_or_release |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | add_perp |  |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | add_perp |  |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_perp |  |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_perp |  |  |  | 0.18 | 0.22 | +0.04 | +0.04 | +0.00 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.16 | 0.19 | +0.03 | +0.03 | +0.00 | +0.11 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_perp |  |  |  | 0.16 | 0.21 | +0.05 | +0.05 | +0.00 | +0.11 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.16 | 0.18 | +0.02 | +0.03 | -0.01 | +0.09 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.16 | 0.18 | +0.02 | +0.05 | -0.03 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | add_perp |  |  |  | 0.16 | 0.18 | +0.02 | +0.05 | -0.03 | +0.09 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.18 | 0.19 | +0.01 | +0.06 | -0.05 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.18 | 0.19 | +0.01 | +0.06 | -0.05 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_remove_random_perp |  |  |  | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_remove_random_perp |  |  |  | 0.16 | 0.11 | -0.04 | +0.03 | -0.07 | +0.05 | flat |

## Decomposition By Route

| model | combo | route | vehicle | tool | unrelated best | vehicle shuffled | tool shuffled | category gap | task/shared gap | shuffle loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | L14 | forbidden_definition:top_p<-forbidden_definition | +0.11 | -0.12 | -0.01 | +0.05 | -0.05 | +0.13 | -0.11 | +0.06 |
| qwen3 | L14 | forbidden_sentence_completion:temperature<-forbidden_definition | -0.10 | -0.28 | -0.08 | -0.11 | -0.29 | -0.02 | -0.20 | +0.01 |
| qwen3 | L14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.04 | -0.33 | -0.16 | -0.01 | -0.28 | +0.20 | -0.18 | +0.05 |
| qwen3 | all | forbidden_definition:top_p<-forbidden_definition | +0.02 | -0.11 | +0.03 | +0.14 | -0.04 | -0.01 | -0.15 | -0.11 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_definition | -0.20 | -0.32 | -0.22 | -0.14 | -0.24 | +0.02 | -0.10 | -0.06 |
| qwen3 | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.02 | -0.27 | -0.17 | +0.01 | -0.26 | +0.19 | -0.10 | +0.01 |
| glm4 | L24 | forbidden_definition:top_p<-forbidden_definition | +0.11 | -0.23 | -0.10 | +0.26 | -0.19 | +0.22 | -0.12 | -0.15 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_definition | +0.34 | -0.02 | +0.04 | +0.38 | -0.04 | +0.30 | -0.06 | -0.03 |
| glm4 | L24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.31 | +0.04 | +0.18 | +0.31 | +0.04 | +0.14 | -0.14 | +0.00 |
| glm4 | L24+L28 | forbidden_definition:top_p<-forbidden_definition | +0.20 | -0.22 | -0.13 | +0.17 | -0.21 | +0.32 | -0.09 | +0.03 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_definition | +0.33 | -0.06 | -0.06 | +0.40 | -0.12 | +0.40 | +0.00 | -0.06 |
| glm4 | L24+L28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.35 | +0.11 | +0.14 | +0.29 | +0.04 | +0.22 | -0.02 | +0.06 |
| glm4 | all | forbidden_definition:top_p<-forbidden_definition | +0.26 | -0.13 | -0.07 | +0.19 | -0.08 | +0.33 | -0.05 | +0.07 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_definition | +0.29 | +0.03 | +0.03 | +0.29 | +0.09 | +0.26 | +0.00 | +0.00 |
| glm4 | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.41 | +0.30 | +0.14 | +0.23 | +0.20 | +0.10 | +0.17 | +0.18 |
| deepseek7b | L16+L20 | forbidden_definition:top_p<-forbidden_definition | -0.04 | -0.10 | -0.04 | +0.03 | -0.12 | +0.00 | -0.06 | -0.07 |
| deepseek7b | L16+L20 | forbidden_sentence_completion:temperature<-forbidden_definition | -0.05 | -0.17 | -0.05 | +0.02 | -0.19 | +0.00 | -0.11 | -0.07 |
| deepseek7b | L16+L20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.09 | -0.10 | -0.12 | -0.04 | -0.12 | +0.20 | +0.02 | +0.14 |
| deepseek7b | all | forbidden_definition:top_p<-forbidden_definition | -0.05 | -0.11 | -0.09 | +0.05 | -0.08 | +0.04 | -0.02 | -0.10 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_definition | -0.07 | -0.19 | -0.07 | +0.00 | -0.14 | +0.00 | -0.11 | -0.07 |
| deepseek7b | all | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | +0.04 | -0.11 | -0.12 | +0.00 | -0.14 | +0.16 | +0.01 | +0.04 |

## Strongest Restores

| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.59 | +0.30 | -0.10 | +0.41 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.57 | +0.28 | -0.11 | +0.40 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.47 | +0.18 | -0.20 | +0.38 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.53 | +0.24 | -0.11 | +0.35 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.44 | +0.15 | -0.20 | +0.34 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.51 | +0.22 | -0.11 | +0.33 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.01 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.41 | +0.11 | -0.20 | +0.31 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.49 | +0.20 | -0.10 | +0.30 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.02 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.48 | +0.19 | -0.10 | +0.29 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.47 | +0.18 | -0.11 | +0.29 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.55 | +0.17 | -0.09 | +0.26 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.29 | 0.42 | +0.12 | -0.10 | +0.23 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.39 | +0.09 | -0.10 | +0.20 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.49 | +0.10 | -0.09 | +0.20 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.48 | +0.09 | -0.09 | +0.19 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.27 | -0.02 | -0.20 | +0.18 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.39 | 0.46 | +0.07 | -0.09 | +0.17 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.32 | +0.03 | -0.10 | +0.14 | +0.00 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.31 | +0.02 | -0.11 | +0.14 | +0.02 | restore_success |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.21 | -0.08 | -0.20 | +0.11 | +0.01 | restore_success |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.29 | +0.00 | -0.11 | +0.11 | +0.00 | restore_success |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.39 | 0.41 | +0.02 | -0.09 | +0.11 | -0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.00 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.29 | +0.00 | -0.10 | +0.10 | +0.01 | restore_success |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.28 | -0.01 | -0.10 | +0.09 | +0.00 | restore_success |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.22 | 0.38 | +0.16 | +0.02 | +0.14 | -0.02 | restore_without_drop_or_leaky |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.22 | 0.35 | +0.14 | +0.02 | +0.11 | -0.01 | restore_without_drop_or_leaky |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.31 | +0.14 | +0.04 | +0.09 | +0.01 | restore_without_drop_or_leaky |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.26 | -0.03 | -0.10 | +0.07 | +0.00 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.16 | -0.14 | -0.20 | +0.06 | +0.00 | weak_restore |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.22 | 0.29 | +0.07 | +0.02 | +0.05 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.16 | 0.26 | +0.10 | +0.05 | +0.05 | +0.07 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.03 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.02 | weak_restore |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.14 | -0.16 | -0.20 | +0.04 | +0.00 | weak_restore |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.22 | -0.07 | -0.11 | +0.04 | +0.00 | weak_restore |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.28 | +0.10 | +0.06 | +0.04 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.62 | +0.08 | +0.04 | +0.04 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.12 | -0.17 | -0.20 | +0.03 | +0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.01 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.22 | -0.07 | -0.10 | +0.03 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.16 | 0.22 | +0.06 | +0.03 | +0.03 | +0.05 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.22 | 0.27 | +0.05 | +0.02 | +0.03 | -0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.62 | +0.08 | +0.06 | +0.02 | +0.02 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.22 | 0.26 | +0.04 | +0.02 | +0.02 | +0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.04 | +0.02 | +0.12 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.11 | -0.18 | -0.20 | +0.02 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.61 | +0.07 | +0.06 | +0.01 | +0.03 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.19 | -0.10 | -0.11 | +0.01 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.24 | +0.06 | +0.06 | +0.00 | +0.14 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.03 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.22 | 0.23 | +0.01 | +0.02 | -0.01 | -0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.54 | 0.57 | +0.03 | +0.04 | -0.01 | +0.02 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.07 | -0.22 | -0.20 | -0.02 | +0.00 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.22 | 0.22 | +0.00 | +0.02 | -0.02 | -0.03 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.16 | -0.14 | -0.11 | -0.02 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.16 | 0.17 | +0.01 | +0.03 | -0.02 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.16 | 0.19 | +0.03 | +0.05 | -0.02 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.22 | 0.21 | -0.01 | +0.02 | -0.03 | -0.03 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.06 | -0.23 | -0.20 | -0.03 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.15 | -0.15 | -0.10 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_vehicle_shuffle_add | vehicle | shuffle | add | 0.18 | 0.18 | +0.00 | +0.04 | -0.04 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.16 | 0.15 | -0.01 | +0.03 | -0.04 | +0.02 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.29 | 0.05 | -0.24 | -0.20 | -0.04 | +0.00 | restore_fail |
| glm4 | L24 | 24 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.05 | -0.24 | -0.20 | -0.04 | +0.00 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.22 | 0.20 | -0.02 | +0.02 | -0.04 | -0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.22 | 0.19 | -0.03 | +0.02 | -0.05 | -0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.17 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.17 | -0.01 | +0.04 | -0.05 | +0.09 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.16 | 0.16 | +0.00 | +0.05 | -0.05 | +0.05 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.01 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.00 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.11 | -0.18 | -0.11 | -0.06 | +0.00 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.15 | -0.03 | +0.04 | -0.07 | +0.09 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.10 | -0.19 | -0.11 | -0.07 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.29 | 0.11 | -0.18 | -0.10 | -0.07 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.22 | 0.17 | -0.05 | +0.02 | -0.07 | -0.02 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.39 | 0.22 | -0.17 | -0.09 | -0.07 | -0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.16 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.18 | 0.17 | -0.01 | +0.06 | -0.07 | +0.09 | restore_fail |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_furniture_add | furniture | aligned | add | 0.29 | 0.09 | -0.20 | -0.11 | -0.08 | +0.00 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.39 | 0.21 | -0.18 | -0.09 | -0.08 | +0.00 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.18 | 0.16 | -0.02 | +0.06 | -0.08 | +0.11 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_shuffle_add | tool | shuffle | add | 0.16 | 0.12 | -0.03 | +0.05 | -0.08 | +0.03 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.54 | 0.50 | -0.04 | +0.04 | -0.08 | +0.01 | restore_fail |
| qwen3 | L14 | 14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_animal_add | animal | aligned | add | 0.22 | 0.15 | -0.07 | +0.02 | -0.09 | -0.02 | restore_fail |
| glm4 | all | 24,26,28 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_furniture_shuffle_add | furniture | shuffle | add | 0.29 | 0.09 | -0.20 | -0.10 | -0.09 | +0.01 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_furniture_add | furniture | aligned | add | 0.16 | 0.11 | -0.04 | +0.05 | -0.09 | -0.03 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.11 | -0.06 | +0.04 | -0.10 | +0.02 | restore_fail |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.16 | 0.08 | -0.07 | +0.03 | -0.10 | -0.04 | restore_fail |
| qwen3 | L14 | 14 | forbidden_sentence_completion:temperature<-forbidden_definition | resid_donor_vehicle_add | vehicle | aligned | add | 0.54 | 0.48 | -0.06 | +0.04 | -0.10 | +0.01 | restore_fail |
| glm4 | L24 | 24 | forbidden_definition:top_p<-forbidden_definition | resid_donor_fruit_add | fruit | aligned | add | 0.39 | 0.19 | -0.20 | -0.09 | -0.10 | +0.01 | restore_fail |
| qwen3 | all | 10,12,14 | forbidden_definition:top_p<-forbidden_definition | resid_donor_tool_add | tool | aligned | add | 0.22 | 0.12 | -0.09 | +0.02 | -0.11 | -0.03 | restore_fail |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion:temperature<-forbidden_sentence_completion | resid_donor_tool_add | tool | aligned | add | 0.18 | 0.12 | -0.05 | +0.06 | -0.11 | +0.00 | restore_fail |

