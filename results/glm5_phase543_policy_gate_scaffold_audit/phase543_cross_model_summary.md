# Phase543 Policy Gate and Scaffold Sensitivity Summary

## qwen3

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'choose_pair', 'label_only'], windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=12, max_new_tokens=10, checkpoints=[1, 3, 5, 10], alpha=6.0, attn=sdpa

| source | scaffold | condition | best win | target hit curve | first target | rankT | gate class |
|---|---|---|---|---|---:|---:|---|
| vehicle_furniture | direct | baseline | center | k1=0.00, k3=0.00, k5=0.25, k10=0.50 | 0.00 | 416.1 | baseline |
| vehicle_furniture | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.33, k10=0.42 | 0.00 | 353.2 | rank_only |
| vehicle_furniture | direct | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.50, k10=0.50 | 0.00 | 190.9 | rank_only |
| vehicle_furniture | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.42, k10=0.50 | 0.00 | 337.2 | rank_only |
| vehicle_furniture | one_word | baseline | center | k1=0.17, k3=0.17, k5=0.17, k10=0.33 | 0.17 | 14.8 | baseline |
| vehicle_furniture | one_word | residual_perp | center | k1=0.17, k3=0.17, k5=0.17, k10=0.42 | 0.17 | 12.3 | no_gate |
| vehicle_furniture | one_word | residual_parallel | extended | k1=0.25, k3=0.25, k5=0.25, k10=0.42 | 0.25 | 8.3 | first_step_only |
| vehicle_furniture | one_word | residual_full | center | k1=0.17, k3=0.17, k5=0.17, k10=0.42 | 0.17 | 11.8 | no_gate |
| vehicle_furniture | choose_pair | baseline | center | k1=0.00, k3=0.75, k5=1.00, k10=1.00 | 0.00 | 13.4 | baseline |
| vehicle_furniture | choose_pair | residual_perp | center | k1=0.00, k3=0.83, k5=1.00, k10=1.00 | 0.00 | 9.6 | no_gate |
| vehicle_furniture | choose_pair | residual_parallel | center | k1=0.00, k3=0.83, k5=1.00, k10=1.00 | 0.00 | 12.2 | no_gate |
| vehicle_furniture | choose_pair | residual_full | center | k1=0.00, k3=0.83, k5=1.00, k10=1.00 | 0.00 | 9.3 | no_gate |
| vehicle_furniture | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 718.5 | baseline |
| vehicle_furniture | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 468.5 | rank_only |
| vehicle_furniture | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 488.9 | rank_only |
| vehicle_furniture | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.08 | 0.00 | 457.8 | no_gate |
| vehicle_tool | direct | baseline | center | k1=0.00, k3=0.00, k5=0.25, k10=0.50 | 0.00 | 416.1 | baseline |
| vehicle_tool | direct | residual_perp | extended | k1=0.00, k3=0.00, k5=0.08, k10=0.42 | 0.00 | 271.5 | rank_only |
| vehicle_tool | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.33, k10=0.50 | 0.00 | 210.3 | rank_only |
| vehicle_tool | direct | residual_full | extended | k1=0.00, k3=0.00, k5=0.25, k10=0.50 | 0.00 | 245.8 | rank_only |
| vehicle_tool | one_word | baseline | center | k1=0.17, k3=0.17, k5=0.17, k10=0.33 | 0.17 | 14.8 | baseline |
| vehicle_tool | one_word | residual_perp | center | k1=0.17, k3=0.17, k5=0.17, k10=0.42 | 0.17 | 10.8 | no_gate |
| vehicle_tool | one_word | residual_parallel | extended | k1=0.25, k3=0.25, k5=0.25, k10=0.25 | 0.25 | 9.8 | first_step_only |
| vehicle_tool | one_word | residual_full | center | k1=0.17, k3=0.25, k5=0.25, k10=0.42 | 0.17 | 10.8 | no_gate |
| vehicle_tool | choose_pair | baseline | center | k1=0.00, k3=0.92, k5=0.92, k10=1.00 | 0.00 | 17.9 | baseline |
| vehicle_tool | choose_pair | residual_perp | center | k1=0.00, k3=0.92, k5=1.00, k10=1.00 | 0.00 | 9.0 | no_gate |
| vehicle_tool | choose_pair | residual_parallel | center | k1=0.00, k3=0.92, k5=0.92, k10=1.00 | 0.00 | 16.4 | no_gate |
| vehicle_tool | choose_pair | residual_full | center | k1=0.00, k3=0.92, k5=1.00, k10=1.00 | 0.00 | 9.2 | no_gate |
| vehicle_tool | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 718.5 | baseline |
| vehicle_tool | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 564.2 | rank_only |
| vehicle_tool | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 440.4 | rank_only |
| vehicle_tool | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 523.0 | rank_only |
| vehicle_clothing | direct | baseline | center | k1=0.00, k3=0.00, k5=0.25, k10=0.50 | 0.00 | 416.1 | baseline |
| vehicle_clothing | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.42, k10=0.42 | 0.00 | 356.3 | rank_only |
| vehicle_clothing | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.42, k10=0.42 | 0.00 | 230.8 | rank_only |
| vehicle_clothing | direct | residual_full | extended | k1=0.00, k3=0.00, k5=0.42, k10=0.42 | 0.00 | 351.2 | rank_only |
| vehicle_clothing | one_word | baseline | center | k1=0.17, k3=0.17, k5=0.17, k10=0.33 | 0.17 | 14.8 | baseline |
| vehicle_clothing | one_word | residual_perp | center | k1=0.17, k3=0.17, k5=0.17, k10=0.42 | 0.17 | 10.1 | no_gate |
| vehicle_clothing | one_word | residual_parallel | extended | k1=0.25, k3=0.25, k5=0.25, k10=0.42 | 0.25 | 9.0 | first_step_only |
| vehicle_clothing | one_word | residual_full | center | k1=0.17, k3=0.17, k5=0.17, k10=0.33 | 0.17 | 9.9 | no_gate |
| vehicle_clothing | choose_pair | baseline | center | k1=0.00, k3=0.67, k5=0.92, k10=0.92 | 0.00 | 27.4 | baseline |
| vehicle_clothing | choose_pair | residual_perp | center | k1=0.00, k3=0.75, k5=0.92, k10=0.92 | 0.00 | 16.2 | no_gate |
| vehicle_clothing | choose_pair | residual_parallel | center | k1=0.00, k3=0.67, k5=0.92, k10=0.92 | 0.00 | 24.6 | no_gate |
| vehicle_clothing | choose_pair | residual_full | extended | k1=0.00, k3=0.83, k5=1.00, k10=1.00 | 0.00 | 12.2 | no_gate |
| vehicle_clothing | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 718.5 | baseline |
| vehicle_clothing | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 426.8 | rank_only |
| vehicle_clothing | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 447.2 | rank_only |
| vehicle_clothing | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 403.7 | rank_only |

## glm4

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'choose_pair', 'label_only'], windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=12, max_new_tokens=10, checkpoints=[1, 3, 5, 10], alpha=6.0, attn=sdpa

| source | scaffold | condition | best win | target hit curve | first target | rankT | gate class |
|---|---|---|---|---|---:|---:|---|
| vehicle_furniture | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 2244.1 | baseline |
| vehicle_furniture | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.17 | 0.00 | 380.8 | weak_gate_open |
| vehicle_furniture | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.33, k10=0.58 | 0.00 | 88.2 | gate_open |
| vehicle_furniture | direct | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.33 | 0.00 | 198.9 | gate_open |
| vehicle_furniture | one_word | baseline | center | k1=0.17, k3=0.25, k5=0.33, k10=0.33 | 0.17 | 71.9 | baseline |
| vehicle_furniture | one_word | residual_perp | extended | k1=0.08, k3=0.25, k5=0.50, k10=0.50 | 0.08 | 38.7 | weak_gate_open |
| vehicle_furniture | one_word | residual_parallel | center | k1=0.58, k3=1.00, k5=1.00, k10=1.00 | 0.58 | 4.9 | gate_open |
| vehicle_furniture | one_word | residual_full | extended | k1=0.33, k3=0.67, k5=0.83, k10=0.83 | 0.33 | 18.4 | gate_open |
| vehicle_furniture | choose_pair | baseline | center | k1=0.00, k3=0.33, k5=0.83, k10=0.92 | 0.00 | 23.5 | baseline |
| vehicle_furniture | choose_pair | residual_perp | center | k1=0.00, k3=0.50, k5=0.92, k10=0.92 | 0.00 | 15.6 | no_gate |
| vehicle_furniture | choose_pair | residual_parallel | center | k1=0.33, k3=0.92, k5=1.00, k10=1.00 | 0.33 | 4.4 | first_step_only |
| vehicle_furniture | choose_pair | residual_full | center | k1=0.00, k3=0.50, k5=0.92, k10=0.92 | 0.00 | 8.8 | no_gate |
| vehicle_furniture | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 571.8 | baseline |
| vehicle_furniture | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 200.5 | rank_only |
| vehicle_furniture | label_only | residual_parallel | center | k1=0.00, k3=0.33, k5=0.33, k10=0.42 | 0.00 | 16.5 | gate_open |
| vehicle_furniture | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 108.7 | rank_only |
| vehicle_tool | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 2244.1 | baseline |
| vehicle_tool | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 407.9 | rank_only |
| vehicle_tool | direct | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.58, k10=0.92 | 0.00 | 101.2 | gate_open |
| vehicle_tool | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 213.6 | rank_only |
| vehicle_tool | one_word | baseline | center | k1=0.17, k3=0.25, k5=0.33, k10=0.33 | 0.17 | 71.9 | baseline |
| vehicle_tool | one_word | residual_perp | extended | k1=0.08, k3=0.17, k5=0.42, k10=0.50 | 0.08 | 35.0 | weak_gate_open |
| vehicle_tool | one_word | residual_parallel | center | k1=0.58, k3=1.00, k5=1.00, k10=1.00 | 0.58 | 5.4 | gate_open |
| vehicle_tool | one_word | residual_full | extended | k1=0.33, k3=0.58, k5=0.67, k10=0.67 | 0.33 | 16.7 | gate_open |
| vehicle_tool | choose_pair | baseline | center | k1=0.00, k3=0.42, k5=0.83, k10=0.92 | 0.00 | 8.4 | baseline |
| vehicle_tool | choose_pair | residual_perp | center | k1=0.00, k3=0.58, k5=1.00, k10=1.00 | 0.00 | 10.8 | no_gate |
| vehicle_tool | choose_pair | residual_parallel | center | k1=0.17, k3=0.75, k5=0.92, k10=0.92 | 0.17 | 5.3 | first_step_only |
| vehicle_tool | choose_pair | residual_full | center | k1=0.00, k3=0.58, k5=1.00, k10=1.00 | 0.00 | 8.8 | no_gate |
| vehicle_tool | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 571.8 | baseline |
| vehicle_tool | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 180.5 | rank_only |
| vehicle_tool | label_only | residual_parallel | center | k1=0.08, k3=0.33, k5=0.33, k10=0.50 | 0.08 | 9.2 | gate_open |
| vehicle_tool | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 73.8 | rank_only |
| vehicle_clothing | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 2244.1 | baseline |
| vehicle_clothing | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.08, k10=0.08 | 0.00 | 650.6 | no_gate |
| vehicle_clothing | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.25, k10=0.58 | 0.00 | 82.4 | gate_open |
| vehicle_clothing | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.08, k10=0.08 | 0.00 | 377.2 | no_gate |
| vehicle_clothing | one_word | baseline | center | k1=0.17, k3=0.25, k5=0.33, k10=0.33 | 0.17 | 71.9 | baseline |
| vehicle_clothing | one_word | residual_perp | center | k1=0.08, k3=0.25, k5=0.50, k10=0.58 | 0.08 | 57.5 | gate_open |
| vehicle_clothing | one_word | residual_parallel | center | k1=0.58, k3=1.00, k5=1.00, k10=1.00 | 0.58 | 5.0 | gate_open |
| vehicle_clothing | one_word | residual_full | center | k1=0.17, k3=0.33, k5=0.50, k10=0.58 | 0.17 | 34.0 | gate_open |
| vehicle_clothing | choose_pair | baseline | center | k1=0.00, k3=0.42, k5=0.83, k10=0.92 | 0.00 | 20.5 | baseline |
| vehicle_clothing | choose_pair | residual_perp | extended | k1=0.00, k3=0.50, k5=0.83, k10=0.83 | 0.00 | 40.2 | no_gate |
| vehicle_clothing | choose_pair | residual_parallel | center | k1=0.17, k3=0.75, k5=1.00, k10=1.00 | 0.17 | 7.0 | first_step_only |
| vehicle_clothing | choose_pair | residual_full | center | k1=0.00, k3=0.58, k5=0.83, k10=0.83 | 0.00 | 18.0 | no_gate |
| vehicle_clothing | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 571.8 | baseline |
| vehicle_clothing | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 202.8 | rank_only |
| vehicle_clothing | label_only | residual_parallel | center | k1=0.00, k3=0.25, k5=0.25, k10=0.33 | 0.00 | 14.3 | gate_open |
| vehicle_clothing | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 101.5 | rank_only |

## deepseek7b

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'choose_pair', 'label_only'], windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=12, max_new_tokens=10, checkpoints=[1, 3, 5, 10], alpha=6.0, attn=sdpa

| source | scaffold | condition | best win | target hit curve | first target | rankT | gate class |
|---|---|---|---|---|---:|---:|---|
| vehicle_furniture | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 50752.3 | baseline |
| vehicle_furniture | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 48826.6 | rank_only |
| vehicle_furniture | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 44206.5 | rank_only |
| vehicle_furniture | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 48774.0 | rank_only |
| vehicle_furniture | one_word | baseline | center | k1=0.08, k3=0.25, k5=0.25, k10=0.25 | 0.08 | 64.6 | baseline |
| vehicle_furniture | one_word | residual_perp | center | k1=0.08, k3=0.17, k5=0.17, k10=0.17 | 0.08 | 50.6 | no_gate |
| vehicle_furniture | one_word | residual_parallel | center | k1=0.08, k3=0.25, k5=0.33, k10=0.33 | 0.08 | 55.9 | no_gate |
| vehicle_furniture | one_word | residual_full | center | k1=0.08, k3=0.17, k5=0.17, k10=0.17 | 0.08 | 50.8 | no_gate |
| vehicle_furniture | choose_pair | baseline | center | k1=0.00, k3=0.58, k5=0.75, k10=0.75 | 0.00 | 203.7 | baseline |
| vehicle_furniture | choose_pair | residual_perp | center | k1=0.00, k3=0.58, k5=0.75, k10=0.75 | 0.00 | 260.0 | no_gate |
| vehicle_furniture | choose_pair | residual_parallel | extended | k1=0.00, k3=0.58, k5=0.83, k10=0.83 | 0.00 | 133.8 | no_gate |
| vehicle_furniture | choose_pair | residual_full | center | k1=0.00, k3=0.58, k5=0.75, k10=0.75 | 0.00 | 258.8 | no_gate |
| vehicle_furniture | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 18062.2 | baseline |
| vehicle_furniture | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 20162.5 | no_gate |
| vehicle_furniture | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 16551.1 | rank_only |
| vehicle_furniture | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 19757.6 | no_gate |
| vehicle_tool | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 50752.3 | baseline |
| vehicle_tool | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 47051.5 | rank_only |
| vehicle_tool | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 34756.8 | rank_only |
| vehicle_tool | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 47325.8 | rank_only |
| vehicle_tool | one_word | baseline | center | k1=0.08, k3=0.25, k5=0.25, k10=0.25 | 0.08 | 64.6 | baseline |
| vehicle_tool | one_word | residual_perp | center | k1=0.08, k3=0.17, k5=0.17, k10=0.25 | 0.08 | 52.0 | no_gate |
| vehicle_tool | one_word | residual_parallel | extended | k1=0.17, k3=0.33, k5=0.33, k10=0.33 | 0.17 | 32.2 | first_step_only |
| vehicle_tool | one_word | residual_full | center | k1=0.08, k3=0.17, k5=0.17, k10=0.25 | 0.08 | 51.2 | no_gate |
| vehicle_tool | choose_pair | baseline | center | k1=0.00, k3=0.50, k5=0.67, k10=0.83 | 0.00 | 60.2 | baseline |
| vehicle_tool | choose_pair | residual_perp | center | k1=0.00, k3=0.58, k5=0.75, k10=0.83 | 0.00 | 53.4 | no_gate |
| vehicle_tool | choose_pair | residual_parallel | extended | k1=0.00, k3=0.83, k5=1.00, k10=1.00 | 0.00 | 39.5 | weak_gate_open |
| vehicle_tool | choose_pair | residual_full | center | k1=0.00, k3=0.58, k5=0.75, k10=0.83 | 0.00 | 53.2 | no_gate |
| vehicle_tool | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 18062.2 | baseline |
| vehicle_tool | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 18158.8 | no_gate |
| vehicle_tool | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 8354.8 | rank_only |
| vehicle_tool | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 18318.1 | no_gate |
| vehicle_clothing | direct | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 50752.3 | baseline |
| vehicle_clothing | direct | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 46805.8 | rank_only |
| vehicle_clothing | direct | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 43083.8 | rank_only |
| vehicle_clothing | direct | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 46918.7 | rank_only |
| vehicle_clothing | one_word | baseline | center | k1=0.08, k3=0.25, k5=0.25, k10=0.25 | 0.08 | 64.6 | baseline |
| vehicle_clothing | one_word | residual_perp | extended | k1=0.08, k3=0.17, k5=0.17, k10=0.25 | 0.08 | 50.0 | no_gate |
| vehicle_clothing | one_word | residual_parallel | center | k1=0.08, k3=0.25, k5=0.25, k10=0.25 | 0.08 | 48.2 | no_gate |
| vehicle_clothing | one_word | residual_full | center | k1=0.08, k3=0.17, k5=0.17, k10=0.17 | 0.08 | 53.5 | no_gate |
| vehicle_clothing | choose_pair | baseline | center | k1=0.00, k3=0.50, k5=0.83, k10=0.83 | 0.00 | 437.2 | baseline |
| vehicle_clothing | choose_pair | residual_perp | center | k1=0.00, k3=0.50, k5=0.83, k10=0.83 | 0.00 | 533.4 | no_gate |
| vehicle_clothing | choose_pair | residual_parallel | center | k1=0.00, k3=0.50, k5=0.92, k10=0.92 | 0.00 | 375.1 | no_gate |
| vehicle_clothing | choose_pair | residual_full | center | k1=0.00, k3=0.50, k5=0.83, k10=0.83 | 0.00 | 529.8 | no_gate |
| vehicle_clothing | label_only | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 18062.2 | baseline |
| vehicle_clothing | label_only | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 15260.2 | rank_only |
| vehicle_clothing | label_only | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 15410.5 | rank_only |
| vehicle_clothing | label_only | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00 | 0.00 | 15633.7 | rank_only |

## Best Positive Rows

| model | source | scaffold | condition | win | base hit | hit | gain | rank improve | gate ratio | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| glm4 | vehicle_tool | direct | residual_parallel | extended | 0.00 | 0.92 | +0.92 | 2142.8 | 0.00043 | gate_open |
| glm4 | vehicle_furniture | one_word | residual_parallel | center | 0.33 | 1.00 | +0.67 | 67.0 | 0.00995 | gate_open |
| glm4 | vehicle_clothing | one_word | residual_parallel | center | 0.33 | 1.00 | +0.67 | 66.9 | 0.00996 | gate_open |
| glm4 | vehicle_tool | one_word | residual_parallel | center | 0.33 | 1.00 | +0.67 | 66.5 | 0.01003 | gate_open |
| glm4 | vehicle_clothing | direct | residual_parallel | center | 0.00 | 0.58 | +0.58 | 2161.7 | 0.00027 | gate_open |
| glm4 | vehicle_furniture | direct | residual_parallel | center | 0.00 | 0.58 | +0.58 | 2155.8 | 0.00027 | gate_open |
| glm4 | vehicle_tool | label_only | residual_parallel | center | 0.00 | 0.50 | +0.50 | 562.5 | 0.00089 | gate_open |
| glm4 | vehicle_furniture | one_word | residual_full | extended | 0.33 | 0.83 | +0.50 | 53.5 | 0.00935 | gate_open |
| glm4 | vehicle_furniture | label_only | residual_parallel | center | 0.00 | 0.42 | +0.42 | 555.2 | 0.00075 | gate_open |
| glm4 | vehicle_furniture | direct | residual_full | extended | 0.00 | 0.33 | +0.33 | 2045.2 | 0.00016 | gate_open |
| glm4 | vehicle_clothing | label_only | residual_parallel | center | 0.00 | 0.33 | +0.33 | 557.4 | 0.00060 | gate_open |
| glm4 | vehicle_tool | one_word | residual_full | extended | 0.33 | 0.67 | +0.33 | 55.2 | 0.00603 | gate_open |
| glm4 | vehicle_clothing | one_word | residual_full | center | 0.33 | 0.58 | +0.25 | 37.9 | 0.00659 | gate_open |
| glm4 | vehicle_clothing | one_word | residual_perp | center | 0.33 | 0.58 | +0.25 | 14.4 | 0.01734 | gate_open |
| glm4 | vehicle_tool | one_word | residual_perp | extended | 0.33 | 0.50 | +0.17 | 36.9 | 0.00451 | weak_gate_open |
| glm4 | vehicle_furniture | one_word | residual_perp | extended | 0.33 | 0.50 | +0.17 | 33.3 | 0.00501 | weak_gate_open |
| glm4 | vehicle_furniture | direct | residual_perp | center | 0.00 | 0.17 | +0.17 | 1863.3 | 0.00009 | weak_gate_open |
| deepseek7b | vehicle_tool | choose_pair | residual_parallel | extended | 0.83 | 1.00 | +0.17 | 20.8 | 0.00803 | weak_gate_open |
| deepseek7b | vehicle_furniture | choose_pair | residual_parallel | extended | 0.75 | 0.83 | +0.08 | 69.8 | 0.00119 | no_gate |
| glm4 | vehicle_furniture | choose_pair | residual_parallel | center | 0.92 | 1.00 | +0.08 | 19.1 | 0.00437 | first_step_only |
| qwen3 | vehicle_clothing | choose_pair | residual_full | extended | 0.92 | 1.00 | +0.08 | 15.2 | 0.00549 | no_gate |
| glm4 | vehicle_clothing | choose_pair | residual_parallel | center | 0.92 | 1.00 | +0.08 | 13.5 | 0.00617 | first_step_only |
| qwen3 | vehicle_furniture | one_word | residual_parallel | extended | 0.33 | 0.42 | +0.08 | 6.5 | 0.01282 | first_step_only |
| qwen3 | vehicle_clothing | one_word | residual_parallel | extended | 0.33 | 0.42 | +0.08 | 5.8 | 0.01429 | first_step_only |
| qwen3 | vehicle_clothing | one_word | residual_perp | center | 0.33 | 0.42 | +0.08 | 4.8 | 0.01754 | no_gate |
| qwen3 | vehicle_tool | one_word | residual_perp | center | 0.33 | 0.42 | +0.08 | 4.1 | 0.02041 | no_gate |
| qwen3 | vehicle_tool | one_word | residual_full | center | 0.33 | 0.42 | +0.08 | 4.1 | 0.02041 | no_gate |
| qwen3 | vehicle_furniture | one_word | residual_full | center | 0.33 | 0.42 | +0.08 | 3.1 | 0.02703 | no_gate |
| qwen3 | vehicle_furniture | one_word | residual_perp | center | 0.33 | 0.42 | +0.08 | 2.5 | 0.03333 | no_gate |
| glm4 | vehicle_tool | choose_pair | residual_full | center | 0.92 | 1.00 | +0.08 | -0.4 | 0.00000 | no_gate |
| glm4 | vehicle_tool | choose_pair | residual_perp | center | 0.92 | 1.00 | +0.08 | -2.4 | 0.00000 | no_gate |
| glm4 | vehicle_clothing | direct | residual_full | center | 0.00 | 0.08 | +0.08 | 1866.9 | 0.00004 | no_gate |
| glm4 | vehicle_clothing | direct | residual_perp | center | 0.00 | 0.08 | +0.08 | 1593.5 | 0.00005 | no_gate |
| qwen3 | vehicle_furniture | label_only | residual_full | center | 0.00 | 0.08 | +0.08 | 260.7 | 0.00032 | no_gate |
| deepseek7b | vehicle_tool | one_word | residual_parallel | extended | 0.25 | 0.33 | +0.08 | 32.4 | 0.00257 | first_step_only |
| deepseek7b | vehicle_furniture | one_word | residual_parallel | center | 0.25 | 0.33 | +0.08 | 8.7 | 0.00962 | no_gate |

