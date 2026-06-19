# Phase544 Natural Answer and Decode-Mode Policy Gate Summary

## qwen3

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'natural_qa', 'definition', 'sentence_completion'], modes=['greedy', 'temperature', 'top_p', 'beam'], windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=10, max_new_tokens=12, alpha=6.0, attn=sdpa

| source | scaffold | mode | condition | win | family target curve | exact target curve | rankT | class |
|---|---|---|---|---|---|---|---:|---|
| vehicle_furniture | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.90 | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | 454.5 | baseline |
| vehicle_furniture | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.50, k10=0.70, k12=0.90 | k1=0.00, k3=0.00, k5=0.40, k10=0.50, k12=0.60 | 372.0 | rank_only |
| vehicle_furniture | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.60, k10=0.80, k12=0.90 | k1=0.00, k3=0.00, k5=0.50, k10=0.50, k12=0.70 | 219.2 | rank_only |
| vehicle_furniture | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.60, k10=0.80, k12=1.00 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.70 | 353.1 | no_gate |
| vehicle_furniture | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.20 | 454.5 | baseline |
| vehicle_furniture | direct | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.30 | 372.0 | no_gate |
| vehicle_furniture | direct | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.40, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.30, k10=0.30, k12=0.30 | 193.5 | weak_natural_gate |
| vehicle_furniture | direct | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | 353.1 | rank_only |
| vehicle_furniture | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 454.5 | baseline |
| vehicle_furniture | direct | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 372.0 | natural_gate_open |
| vehicle_furniture | direct | top_p | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.50 | k1=0.00, k3=0.10, k5=0.10, k10=0.20, k12=0.20 | 193.5 | natural_gate_open |
| vehicle_furniture | direct | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 353.1 | weak_natural_gate |
| vehicle_furniture | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 451.4 | baseline |
| vehicle_furniture | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | 373.8 | rank_only |
| vehicle_furniture | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=0.80 | k1=0.00, k3=0.00, k5=0.30, k10=0.50, k12=0.60 | 217.0 | rank_only |
| vehicle_furniture | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.20, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 356.2 | rank_only |
| vehicle_furniture | one_word | greedy | baseline | center | k1=0.20, k3=0.20, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 4.4 | baseline |
| vehicle_furniture | one_word | greedy | residual_perp | center | k1=0.20, k3=0.20, k5=0.40, k10=0.80, k12=0.80 | k1=0.20, k3=0.20, k5=0.20, k10=0.50, k12=0.50 | 3.9 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | residual_parallel | center | k1=0.30, k3=0.30, k5=0.50, k10=0.70, k12=0.80 | k1=0.30, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 3.5 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | residual_full | center | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.9 | label_only_gain |
| vehicle_furniture | one_word | temperature | baseline | center | k1=0.10, k3=0.10, k5=0.30, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.20, k10=0.30, k12=0.40 | 4.4 | baseline |
| vehicle_furniture | one_word | temperature | residual_perp | extended | k1=0.30, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.20, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | 3.7 | no_gate |
| vehicle_furniture | one_word | temperature | residual_parallel | extended | k1=0.60, k3=0.70, k5=0.70, k10=0.80, k12=0.80 | k1=0.40, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | 3.4 | weak_natural_gate |
| vehicle_furniture | one_word | temperature | residual_full | extended | k1=0.10, k3=0.20, k5=0.20, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.40, k12=0.40 | 3.6 | no_gate |
| vehicle_furniture | one_word | top_p | baseline | center | k1=0.30, k3=0.30, k5=0.30, k10=0.80, k12=0.80 | k1=0.30, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | 4.4 | baseline |
| vehicle_furniture | one_word | top_p | residual_perp | extended | k1=0.20, k3=0.30, k5=0.30, k10=0.80, k12=0.80 | k1=0.20, k3=0.20, k5=0.20, k10=0.50, k12=0.50 | 3.7 | no_gate |
| vehicle_furniture | one_word | top_p | residual_parallel | extended | k1=0.50, k3=0.50, k5=0.60, k10=0.90, k12=0.90 | k1=0.30, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | 3.4 | no_gate |
| vehicle_furniture | one_word | top_p | residual_full | extended | k1=0.30, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.30, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 3.6 | no_gate |
| vehicle_furniture | one_word | beam | baseline | center | k1=0.10, k3=0.10, k5=0.20, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.30, k12=0.40 | 4.5 | baseline |
| vehicle_furniture | one_word | beam | residual_perp | center | k1=0.20, k3=0.20, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 3.9 | no_gate |
| vehicle_furniture | one_word | beam | residual_parallel | center | k1=0.10, k3=0.10, k5=0.30, k10=0.60, k12=0.70 | k1=0.10, k3=0.10, k5=0.10, k10=0.20, k12=0.30 | 3.5 | no_gate |
| vehicle_furniture | one_word | beam | residual_full | center | k1=0.20, k3=0.20, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.9 | no_gate |
| vehicle_furniture | natural_qa | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_furniture | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 81.2 | no_gate |
| vehicle_furniture | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 66.5 | no_gate |
| vehicle_furniture | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 78.8 | no_gate |
| vehicle_furniture | natural_qa | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | 84.5 | baseline |
| vehicle_furniture | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 81.2 | label_only_gain |
| vehicle_furniture | natural_qa | temperature | residual_parallel | center | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 66.5 | label_only_gain |
| vehicle_furniture | natural_qa | temperature | residual_full | center | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.70, k10=0.70, k12=0.80 | 78.8 | label_only_gain |
| vehicle_furniture | natural_qa | top_p | baseline | center | k1=0.00, k3=0.40, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_furniture | natural_qa | top_p | residual_perp | extended | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.80 | 84.9 | no_gate |
| vehicle_furniture | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 66.5 | no_gate |
| vehicle_furniture | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 78.8 | no_gate |
| vehicle_furniture | natural_qa | beam | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 86.3 | baseline |
| vehicle_furniture | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 79.7 | no_gate |
| vehicle_furniture | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 68.1 | no_gate |
| vehicle_furniture | natural_qa | beam | residual_full | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 77.4 | no_gate |
| vehicle_furniture | definition | greedy | baseline | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | 137.1 | baseline |
| vehicle_furniture | definition | greedy | residual_perp | extended | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | 119.4 | no_gate |
| vehicle_furniture | definition | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 91.7 | no_gate |
| vehicle_furniture | definition | greedy | residual_full | extended | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 114.8 | no_gate |
| vehicle_furniture | definition | temperature | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.50, k12=0.50 | 137.1 | baseline |
| vehicle_furniture | definition | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.10, k5=0.40, k10=0.60, k12=0.60 | 122.6 | weak_natural_gate |
| vehicle_furniture | definition | temperature | residual_parallel | center | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.80 | k1=0.00, k3=0.20, k5=0.20, k10=0.40, k12=0.50 | 91.7 | weak_natural_gate |
| vehicle_furniture | definition | temperature | residual_full | center | k1=0.00, k3=0.50, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 118.6 | weak_natural_gate |
| vehicle_furniture | definition | top_p | baseline | center | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 137.1 | baseline |
| vehicle_furniture | definition | top_p | residual_perp | extended | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | 119.4 | no_gate |
| vehicle_furniture | definition | top_p | residual_parallel | center | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.30, k10=0.50, k12=0.50 | 91.7 | no_gate |
| vehicle_furniture | definition | top_p | residual_full | extended | k1=0.00, k3=0.20, k5=0.40, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.30, k10=0.50, k12=0.60 | 114.8 | no_gate |
| vehicle_furniture | definition | beam | baseline | center | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 138.8 | baseline |
| vehicle_furniture | definition | beam | residual_perp | center | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 122.3 | label_only_gain |
| vehicle_furniture | definition | beam | residual_parallel | center | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 92.3 | no_gate |
| vehicle_furniture | definition | beam | residual_full | center | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 118.7 | label_only_gain |
| vehicle_furniture | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 68.9 | baseline |
| vehicle_furniture | sentence_completion | greedy | residual_perp | center | k1=0.00, k3=0.20, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 69.5 | no_gate |
| vehicle_furniture | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 51.2 | label_only_gain |
| vehicle_furniture | sentence_completion | greedy | residual_full | extended | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 64.0 | no_gate |
| vehicle_furniture | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.80 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.20 | 68.9 | baseline |
| vehicle_furniture | sentence_completion | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.50, k10=0.50, k12=0.60 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 69.5 | no_gate |
| vehicle_furniture | sentence_completion | temperature | residual_parallel | center | k1=0.00, k3=0.40, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 51.2 | no_gate |
| vehicle_furniture | sentence_completion | temperature | residual_full | center | k1=0.00, k3=0.20, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 67.9 | no_gate |
| vehicle_furniture | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.40, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 68.9 | baseline |
| vehicle_furniture | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.30, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.30 | 69.5 | label_only_gain |
| vehicle_furniture | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.40, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.20 | 51.2 | weak_natural_gate |
| vehicle_furniture | sentence_completion | top_p | residual_full | extended | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 64.0 | weak_natural_gate |
| vehicle_furniture | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 68.7 | baseline |
| vehicle_furniture | sentence_completion | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 70.0 | label_only_gain |
| vehicle_furniture | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.10, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.20 | 50.5 | label_only_gain |
| vehicle_furniture | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.10, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 67.9 | no_gate |
| vehicle_tool | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.90 | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | 454.5 | baseline |
| vehicle_tool | direct | greedy | residual_perp | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | 288.5 | rank_only |
| vehicle_tool | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=1.00 | k1=0.00, k3=0.00, k5=0.30, k10=0.50, k12=0.60 | 216.3 | no_gate |
| vehicle_tool | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.20, k10=0.70, k12=0.90 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 313.4 | rank_only |
| vehicle_tool | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 454.5 | baseline |
| vehicle_tool | direct | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.40, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.30 | 288.5 | natural_gate_open |
| vehicle_tool | direct | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.60 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 208.5 | weak_natural_gate |
| vehicle_tool | direct | temperature | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.60 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 261.7 | weak_natural_gate |
| vehicle_tool | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.30 | 454.5 | baseline |
| vehicle_tool | direct | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.30, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 288.5 | rank_only |
| vehicle_tool | direct | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | 216.3 | weak_natural_gate |
| vehicle_tool | direct | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.30, k10=0.40, k12=0.40 | 261.7 | weak_natural_gate |
| vehicle_tool | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 451.4 | baseline |
| vehicle_tool | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.20, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.50 | 332.2 | rank_only |
| vehicle_tool | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | 216.2 | rank_only |
| vehicle_tool | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.20, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.50 | 311.0 | rank_only |
| vehicle_tool | one_word | greedy | baseline | center | k1=0.20, k3=0.20, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 4.4 | baseline |
| vehicle_tool | one_word | greedy | residual_perp | center | k1=0.20, k3=0.20, k5=0.50, k10=0.80, k12=0.80 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 4.1 | weak_natural_gate |
| vehicle_tool | one_word | greedy | residual_parallel | center | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 3.7 | no_gate |
| vehicle_tool | one_word | greedy | residual_full | center | k1=0.20, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 4.1 | weak_natural_gate |
| vehicle_tool | one_word | temperature | baseline | center | k1=0.20, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 4.4 | baseline |
| vehicle_tool | one_word | temperature | residual_perp | center | k1=0.20, k3=0.30, k5=0.30, k10=0.80, k12=0.80 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 4.1 | weak_natural_gate |
| vehicle_tool | one_word | temperature | residual_parallel | extended | k1=0.20, k3=0.20, k5=0.40, k10=0.90, k12=0.90 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.4 | weak_natural_gate |
| vehicle_tool | one_word | temperature | residual_full | center | k1=0.30, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.30, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | 4.1 | label_only_gain |
| vehicle_tool | one_word | top_p | baseline | center | k1=0.20, k3=0.20, k5=0.30, k10=0.40, k12=0.50 | k1=0.20, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 4.4 | baseline |
| vehicle_tool | one_word | top_p | residual_perp | extended | k1=0.10, k3=0.10, k5=0.40, k10=0.70, k12=0.70 | k1=0.10, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 3.9 | weak_natural_gate |
| vehicle_tool | one_word | top_p | residual_parallel | center | k1=0.30, k3=0.30, k5=0.30, k10=0.70, k12=0.90 | k1=0.30, k3=0.30, k5=0.30, k10=0.30, k12=0.50 | 3.7 | natural_gate_open |
| vehicle_tool | one_word | top_p | residual_full | extended | k1=0.20, k3=0.60, k5=0.60, k10=0.90, k12=0.90 | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | 3.8 | natural_gate_open |
| vehicle_tool | one_word | beam | baseline | center | k1=0.10, k3=0.10, k5=0.20, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.30, k12=0.40 | 4.5 | baseline |
| vehicle_tool | one_word | beam | residual_perp | extended | k1=0.20, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 3.8 | no_gate |
| vehicle_tool | one_word | beam | residual_parallel | center | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 3.6 | no_gate |
| vehicle_tool | one_word | beam | residual_full | center | k1=0.20, k3=0.30, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 4.1 | no_gate |
| vehicle_tool | natural_qa | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_tool | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 73.7 | no_gate |
| vehicle_tool | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 67.5 | no_gate |
| vehicle_tool | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 71.0 | no_gate |
| vehicle_tool | natural_qa | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_tool | natural_qa | temperature | residual_perp | extended | k1=0.00, k3=0.50, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | 69.4 | no_gate |
| vehicle_tool | natural_qa | temperature | residual_parallel | center | k1=0.00, k3=0.40, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 67.5 | no_gate |
| vehicle_tool | natural_qa | temperature | residual_full | center | k1=0.00, k3=0.40, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 71.0 | no_gate |
| vehicle_tool | natural_qa | top_p | baseline | center | k1=0.00, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_tool | natural_qa | top_p | residual_perp | center | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.80, k12=0.80 | 73.7 | no_gate |
| vehicle_tool | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 67.5 | no_gate |
| vehicle_tool | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 71.0 | no_gate |
| vehicle_tool | natural_qa | beam | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 86.3 | baseline |
| vehicle_tool | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 72.7 | no_gate |
| vehicle_tool | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 68.2 | no_gate |
| vehicle_tool | natural_qa | beam | residual_full | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 72.4 | no_gate |
| vehicle_tool | definition | greedy | baseline | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | 137.1 | baseline |
| vehicle_tool | definition | greedy | residual_perp | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 111.1 | no_gate |
| vehicle_tool | definition | greedy | residual_parallel | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 93.1 | no_gate |
| vehicle_tool | definition | greedy | residual_full | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 108.0 | no_gate |
| vehicle_tool | definition | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.60, k12=0.70 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | 137.1 | baseline |
| vehicle_tool | definition | temperature | residual_perp | extended | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.50, k10=0.60, k12=0.60 | 105.3 | no_gate |
| vehicle_tool | definition | temperature | residual_parallel | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 93.1 | weak_natural_gate |
| vehicle_tool | definition | temperature | residual_full | center | k1=0.00, k3=0.30, k5=0.50, k10=0.80, k12=0.90 | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | 108.0 | weak_natural_gate |
| vehicle_tool | definition | top_p | baseline | center | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 137.1 | baseline |
| vehicle_tool | definition | top_p | residual_perp | center | k1=0.00, k3=0.60, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.70, k10=0.70, k12=0.70 | 111.1 | no_gate |
| vehicle_tool | definition | top_p | residual_parallel | center | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.70, k12=0.70 | 93.1 | no_gate |
| vehicle_tool | definition | top_p | residual_full | center | k1=0.00, k3=0.40, k5=0.50, k10=0.70, k12=0.80 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 108.0 | no_gate |
| vehicle_tool | definition | beam | baseline | center | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 138.8 | baseline |
| vehicle_tool | definition | beam | residual_perp | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 110.3 | no_gate |
| vehicle_tool | definition | beam | residual_parallel | center | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 90.9 | no_gate |
| vehicle_tool | definition | beam | residual_full | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 108.1 | no_gate |
| vehicle_tool | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 68.9 | baseline |
| vehicle_tool | sentence_completion | greedy | residual_perp | extended | k1=0.00, k3=0.30, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 49.4 | label_only_gain |
| vehicle_tool | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 53.4 | no_gate |
| vehicle_tool | sentence_completion | greedy | residual_full | center | k1=0.00, k3=0.30, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 54.7 | label_only_gain |
| vehicle_tool | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | 68.9 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_perp | center | k1=0.00, k3=0.50, k5=0.80, k10=0.80, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 56.5 | weak_natural_gate |
| vehicle_tool | sentence_completion | temperature | residual_parallel | center | k1=0.10, k3=0.40, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 53.4 | no_gate |
| vehicle_tool | sentence_completion | temperature | residual_full | extended | k1=0.00, k3=0.20, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 47.1 | no_gate |
| vehicle_tool | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | 68.9 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.20, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.20 | 56.5 | weak_natural_gate |
| vehicle_tool | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.30, k5=0.70, k10=0.70, k12=0.90 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.40 | 53.4 | weak_natural_gate |
| vehicle_tool | sentence_completion | top_p | residual_full | center | k1=0.00, k3=0.10, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 54.7 | no_gate |
| vehicle_tool | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 68.7 | baseline |
| vehicle_tool | sentence_completion | beam | residual_perp | center | k1=0.00, k3=0.10, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.20 | 56.3 | label_only_gain |
| vehicle_tool | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.20, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 53.5 | label_only_gain |
| vehicle_tool | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 54.3 | label_only_gain |
| vehicle_clothing | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.90 | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | 454.5 | baseline |
| vehicle_clothing | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.50, k10=0.50, k12=0.60 | 369.4 | rank_only |
| vehicle_clothing | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.60, k10=0.80, k12=0.90 | k1=0.00, k3=0.00, k5=0.50, k10=0.50, k12=0.60 | 235.7 | rank_only |
| vehicle_clothing | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.40, k10=0.40, k12=0.50 | 355.7 | rank_only |
| vehicle_clothing | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 454.5 | baseline |
| vehicle_clothing | direct | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.20, k10=0.20, k12=0.20 | 380.8 | natural_gate_open |
| vehicle_clothing | direct | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 207.5 | natural_gate_open |
| vehicle_clothing | direct | temperature | residual_full | extended | k1=0.00, k3=0.00, k5=0.20, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 361.4 | natural_gate_open |
| vehicle_clothing | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 454.5 | baseline |
| vehicle_clothing | direct | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.60 | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | 380.8 | natural_gate_open |
| vehicle_clothing | direct | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | 235.7 | natural_gate_open |
| vehicle_clothing | direct | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.20, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 355.7 | natural_gate_open |
| vehicle_clothing | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 451.4 | baseline |
| vehicle_clothing | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.40 | 368.0 | rank_only |
| vehicle_clothing | direct | beam | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=0.90 | k1=0.00, k3=0.00, k5=0.30, k10=0.50, k12=0.60 | 210.0 | no_gate |
| vehicle_clothing | direct | beam | residual_full | extended | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.40 | 362.8 | rank_only |
| vehicle_clothing | one_word | greedy | baseline | center | k1=0.20, k3=0.20, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 4.4 | baseline |
| vehicle_clothing | one_word | greedy | residual_perp | center | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.9 | label_only_gain |
| vehicle_clothing | one_word | greedy | residual_parallel | extended | k1=0.30, k3=0.30, k5=0.40, k10=0.60, k12=0.80 | k1=0.30, k3=0.30, k5=0.30, k10=0.40, k12=0.50 | 3.4 | weak_natural_gate |
| vehicle_clothing | one_word | greedy | residual_full | center | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 3.8 | no_gate |
| vehicle_clothing | one_word | temperature | baseline | center | k1=0.20, k3=0.20, k5=0.30, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.20, k12=0.30 | 4.4 | baseline |
| vehicle_clothing | one_word | temperature | residual_perp | center | k1=0.10, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.10, k3=0.20, k5=0.30, k10=0.40, k12=0.40 | 3.9 | weak_natural_gate |
| vehicle_clothing | one_word | temperature | residual_parallel | center | k1=0.20, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 3.6 | label_only_gain |
| vehicle_clothing | one_word | temperature | residual_full | extended | k1=0.20, k3=0.20, k5=0.20, k10=0.50, k12=0.60 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.50 | 3.7 | label_only_gain |
| vehicle_clothing | one_word | top_p | baseline | center | k1=0.20, k3=0.40, k5=0.40, k10=0.80, k12=0.80 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 4.4 | baseline |
| vehicle_clothing | one_word | top_p | residual_perp | center | k1=0.10, k3=0.30, k5=0.30, k10=0.50, k12=0.60 | k1=0.10, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 3.9 | no_gate |
| vehicle_clothing | one_word | top_p | residual_parallel | extended | k1=0.20, k3=0.40, k5=0.50, k10=0.90, k12=0.90 | k1=0.20, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | 3.4 | no_gate |
| vehicle_clothing | one_word | top_p | residual_full | center | k1=0.20, k3=0.30, k5=0.40, k10=0.70, k12=0.80 | k1=0.20, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 3.8 | no_gate |
| vehicle_clothing | one_word | beam | baseline | center | k1=0.10, k3=0.10, k5=0.20, k10=0.50, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.30, k12=0.40 | 4.5 | baseline |
| vehicle_clothing | one_word | beam | residual_perp | center | k1=0.20, k3=0.20, k5=0.30, k10=0.80, k12=0.80 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.9 | weak_natural_gate |
| vehicle_clothing | one_word | beam | residual_parallel | center | k1=0.10, k3=0.10, k5=0.20, k10=0.50, k12=0.70 | k1=0.10, k3=0.10, k5=0.10, k10=0.20, k12=0.40 | 3.6 | no_gate |
| vehicle_clothing | one_word | beam | residual_full | center | k1=0.20, k3=0.20, k5=0.30, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 3.8 | no_gate |
| vehicle_clothing | natural_qa | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_clothing | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 74.5 | no_gate |
| vehicle_clothing | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 72.3 | no_gate |
| vehicle_clothing | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 73.4 | no_gate |
| vehicle_clothing | natural_qa | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 84.5 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | 74.5 | weak_natural_gate |
| vehicle_clothing | natural_qa | temperature | residual_parallel | extended | k1=0.00, k3=0.50, k5=0.60, k10=0.80, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.80, k12=0.80 | 72.3 | natural_gate_open |
| vehicle_clothing | natural_qa | temperature | residual_full | center | k1=0.00, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.80 | 73.4 | natural_gate_open |
| vehicle_clothing | natural_qa | top_p | baseline | center | k1=0.00, k3=0.40, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 84.5 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_perp | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 74.5 | no_gate |
| vehicle_clothing | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.80 | 72.3 | no_gate |
| vehicle_clothing | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | 73.4 | no_gate |
| vehicle_clothing | natural_qa | beam | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 86.3 | baseline |
| vehicle_clothing | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 74.5 | no_gate |
| vehicle_clothing | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.30, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 72.7 | no_gate |
| vehicle_clothing | natural_qa | beam | residual_full | center | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | 72.4 | no_gate |
| vehicle_clothing | definition | greedy | baseline | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | 137.1 | baseline |
| vehicle_clothing | definition | greedy | residual_perp | center | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 129.5 | no_gate |
| vehicle_clothing | definition | greedy | residual_parallel | center | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 98.5 | no_gate |
| vehicle_clothing | definition | greedy | residual_full | center | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | 126.9 | no_gate |
| vehicle_clothing | definition | temperature | baseline | center | k1=0.00, k3=0.10, k5=0.40, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.40, k10=0.60, k12=0.60 | 137.1 | baseline |
| vehicle_clothing | definition | temperature | residual_perp | center | k1=0.00, k3=0.40, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.30, k5=0.40, k10=0.50, k12=0.60 | 129.5 | no_gate |
| vehicle_clothing | definition | temperature | residual_parallel | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 88.2 | weak_natural_gate |
| vehicle_clothing | definition | temperature | residual_full | center | k1=0.00, k3=0.40, k5=0.50, k10=0.70, k12=0.80 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.70 | 126.9 | weak_natural_gate |
| vehicle_clothing | definition | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | 137.1 | baseline |
| vehicle_clothing | definition | top_p | residual_perp | center | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | 129.5 | weak_natural_gate |
| vehicle_clothing | definition | top_p | residual_parallel | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 88.2 | weak_natural_gate |
| vehicle_clothing | definition | top_p | residual_full | center | k1=0.00, k3=0.40, k5=0.50, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.40, k10=0.80, k12=0.80 | 126.9 | weak_natural_gate |
| vehicle_clothing | definition | beam | baseline | center | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.60, k12=0.60 | 138.8 | baseline |
| vehicle_clothing | definition | beam | residual_perp | center | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 129.1 | label_only_gain |
| vehicle_clothing | definition | beam | residual_parallel | center | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 97.8 | no_gate |
| vehicle_clothing | definition | beam | residual_full | extended | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.80, k12=0.80 | 123.5 | label_only_gain |
| vehicle_clothing | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 68.9 | baseline |
| vehicle_clothing | sentence_completion | greedy | residual_perp | extended | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 82.6 | no_gate |
| vehicle_clothing | sentence_completion | greedy | residual_parallel | extended | k1=0.00, k3=0.20, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 49.2 | no_gate |
| vehicle_clothing | sentence_completion | greedy | residual_full | extended | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 78.7 | no_gate |
| vehicle_clothing | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.30, k10=0.30, k12=0.30 | 68.9 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_perp | extended | k1=0.10, k3=0.40, k5=0.70, k10=0.80, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 82.6 | weak_natural_gate |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.90 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 55.9 | weak_natural_gate |
| vehicle_clothing | sentence_completion | temperature | residual_full | extended | k1=0.00, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 78.7 | weak_natural_gate |
| vehicle_clothing | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 68.9 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.40, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 83.0 | weak_natural_gate |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 55.9 | weak_natural_gate |
| vehicle_clothing | sentence_completion | top_p | residual_full | extended | k1=0.00, k3=0.40, k5=0.90, k10=0.90, k12=1.00 | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.60 | 78.7 | natural_gate_open |
| vehicle_clothing | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 68.7 | baseline |
| vehicle_clothing | sentence_completion | beam | residual_perp | center | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 82.5 | label_only_gain |
| vehicle_clothing | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.10, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.20 | 55.1 | label_only_gain |
| vehicle_clothing | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.20, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 79.5 | label_only_gain |

## glm4

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'natural_qa', 'definition', 'sentence_completion'], modes=['greedy', 'temperature', 'top_p', 'beam'], windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=10, max_new_tokens=12, alpha=6.0, attn=sdpa

| source | scaffold | mode | condition | win | family target curve | exact target curve | rankT | class |
|---|---|---|---|---|---|---|---:|---|
| vehicle_furniture | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_furniture | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | 355.6 | label_only_gain |
| vehicle_furniture | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.30, k10=0.60, k12=0.60 | 76.9 | natural_gate_open |
| vehicle_furniture | direct | greedy | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.00, k10=0.50, k12=0.50 | 181.9 | natural_gate_open |
| vehicle_furniture | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_furniture | direct | temperature | residual_perp | center | k1=0.00, k3=0.10, k5=0.20, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.30 | 355.6 | natural_gate_open |
| vehicle_furniture | direct | temperature | residual_parallel | center | k1=0.10, k3=0.10, k5=0.30, k10=0.90, k12=0.90 | k1=0.10, k3=0.10, k5=0.20, k10=0.80, k12=0.80 | 76.9 | natural_gate_open |
| vehicle_furniture | direct | temperature | residual_full | extended | k1=0.00, k3=0.00, k5=0.30, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 181.9 | weak_natural_gate |
| vehicle_furniture | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 2437.8 | baseline |
| vehicle_furniture | direct | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 355.6 | rank_only |
| vehicle_furniture | direct | top_p | residual_parallel | center | k1=0.00, k3=0.10, k5=0.20, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.20, k10=0.60, k12=0.70 | 76.9 | natural_gate_open |
| vehicle_furniture | direct | top_p | residual_full | extended | k1=0.00, k3=0.10, k5=0.30, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.20 | 181.9 | natural_gate_open |
| vehicle_furniture | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2410.6 | baseline |
| vehicle_furniture | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 358.6 | label_only_gain |
| vehicle_furniture | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | 78.1 | natural_gate_open |
| vehicle_furniture | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 205.8 | label_only_gain |
| vehicle_furniture | one_word | greedy | baseline | center | k1=0.20, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.20, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 44.5 | baseline |
| vehicle_furniture | one_word | greedy | residual_perp | extended | k1=0.20, k3=0.50, k5=0.70, k10=0.80, k12=0.90 | k1=0.20, k3=0.40, k5=0.60, k10=0.60, k12=0.60 | 32.6 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | residual_parallel | center | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 4.1 | natural_gate_open |
| vehicle_furniture | one_word | greedy | residual_full | center | k1=0.40, k3=0.70, k5=0.80, k10=0.80, k12=0.90 | k1=0.40, k3=0.70, k5=0.80, k10=0.80, k12=0.90 | 16.0 | weak_natural_gate |
| vehicle_furniture | one_word | temperature | baseline | center | k1=0.20, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | k1=0.20, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 44.5 | baseline |
| vehicle_furniture | one_word | temperature | residual_perp | center | k1=0.30, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.20, k3=0.30, k5=0.50, k10=0.60, k12=0.60 | 30.2 | natural_gate_open |
| vehicle_furniture | one_word | temperature | residual_parallel | center | k1=0.30, k3=0.80, k5=0.80, k10=1.00, k12=1.00 | k1=0.30, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | 4.1 | natural_gate_open |
| vehicle_furniture | one_word | temperature | residual_full | extended | k1=0.20, k3=0.70, k5=0.80, k10=0.90, k12=0.90 | k1=0.20, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | 15.4 | natural_gate_open |
| vehicle_furniture | one_word | top_p | baseline | center | k1=0.10, k3=0.20, k5=0.40, k10=0.60, k12=0.70 | k1=0.10, k3=0.20, k5=0.30, k10=0.50, k12=0.50 | 44.5 | baseline |
| vehicle_furniture | one_word | top_p | residual_perp | center | k1=0.30, k3=0.30, k5=0.50, k10=0.80, k12=0.90 | k1=0.20, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | 30.2 | weak_natural_gate |
| vehicle_furniture | one_word | top_p | residual_parallel | center | k1=0.40, k3=0.70, k5=0.80, k10=0.90, k12=1.00 | k1=0.40, k3=0.60, k5=0.70, k10=0.80, k12=1.00 | 4.1 | natural_gate_open |
| vehicle_furniture | one_word | top_p | residual_full | extended | k1=0.40, k3=0.80, k5=0.90, k10=1.00, k12=1.00 | k1=0.10, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 15.4 | natural_gate_open |
| vehicle_furniture | one_word | beam | baseline | center | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 5.3 | baseline |
| vehicle_furniture | one_word | beam | residual_perp | center | k1=0.20, k3=0.40, k5=0.60, k10=0.90, k12=0.90 | k1=0.10, k3=0.20, k5=0.40, k10=0.70, k12=0.70 | 2.4 | label_only_gain |
| vehicle_furniture | one_word | beam | residual_parallel | center | k1=0.90, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.90, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 1.0 | weak_natural_gate |
| vehicle_furniture | one_word | beam | residual_full | center | k1=0.40, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.40, k3=0.40, k5=0.50, k10=0.60, k12=0.70 | 1.3 | label_only_gain |
| vehicle_furniture | natural_qa | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.60, k10=0.60, k12=0.60 | 141.1 | baseline |
| vehicle_furniture | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | 107.9 | weak_natural_gate |
| vehicle_furniture | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | 44.5 | weak_natural_gate |
| vehicle_furniture | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.80 | 84.3 | weak_natural_gate |
| vehicle_furniture | natural_qa | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.50, k12=0.50 | 141.1 | baseline |
| vehicle_furniture | natural_qa | temperature | residual_perp | extended | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.60, k12=0.70 | 119.7 | natural_gate_open |
| vehicle_furniture | natural_qa | temperature | residual_parallel | center | k1=0.00, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | 44.5 | natural_gate_open |
| vehicle_furniture | natural_qa | temperature | residual_full | extended | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=0.90 | 84.1 | natural_gate_open |
| vehicle_furniture | natural_qa | top_p | baseline | center | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.40, k10=0.50, k12=0.50 | 141.1 | baseline |
| vehicle_furniture | natural_qa | top_p | residual_perp | center | k1=0.00, k3=0.20, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | 107.9 | no_gate |
| vehicle_furniture | natural_qa | top_p | residual_parallel | extended | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | 50.1 | natural_gate_open |
| vehicle_furniture | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.60, k10=0.80, k12=0.80 | 84.3 | weak_natural_gate |
| vehicle_furniture | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 141.9 | baseline |
| vehicle_furniture | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.20, k5=0.70, k10=0.70, k12=0.80 | k1=0.00, k3=0.20, k5=0.70, k10=0.70, k12=0.80 | 110.4 | weak_natural_gate |
| vehicle_furniture | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | 44.1 | weak_natural_gate |
| vehicle_furniture | natural_qa | beam | residual_full | center | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | 84.2 | weak_natural_gate |
| vehicle_furniture | definition | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.40, k10=0.70, k12=0.70 | 92.1 | baseline |
| vehicle_furniture | definition | greedy | residual_perp | center | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | 25.6 | weak_natural_gate |
| vehicle_furniture | definition | greedy | residual_parallel | center | k1=0.00, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.90, k5=0.90, k10=1.00, k12=1.00 | 13.4 | natural_gate_open |
| vehicle_furniture | definition | greedy | residual_full | extended | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=0.90 | 15.6 | weak_natural_gate |
| vehicle_furniture | definition | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | 92.1 | baseline |
| vehicle_furniture | definition | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.20, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.10, k10=0.50, k12=0.60 | 25.6 | rank_only |
| vehicle_furniture | definition | temperature | residual_parallel | center | k1=0.00, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | 13.4 | label_only_gain |
| vehicle_furniture | definition | temperature | residual_full | center | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 17.1 | rank_only |
| vehicle_furniture | definition | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.30, k10=0.50, k12=0.50 | 92.1 | baseline |
| vehicle_furniture | definition | top_p | residual_perp | center | k1=0.00, k3=0.20, k5=0.30, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.30, k10=0.60, k12=0.60 | 25.6 | rank_only |
| vehicle_furniture | definition | top_p | residual_parallel | center | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | 13.4 | natural_gate_open |
| vehicle_furniture | definition | top_p | residual_full | extended | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.80 | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.80 | 15.6 | weak_natural_gate |
| vehicle_furniture | definition | beam | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 91.1 | baseline |
| vehicle_furniture | definition | beam | residual_perp | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 25.6 | rank_only |
| vehicle_furniture | definition | beam | residual_parallel | center | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.90, k12=0.90 | 13.6 | natural_gate_open |
| vehicle_furniture | definition | beam | residual_full | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 17.4 | rank_only |
| vehicle_furniture | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.5 | baseline |
| vehicle_furniture | sentence_completion | greedy | residual_perp | center | k1=0.00, k3=0.40, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | 38.0 | label_only_gain |
| vehicle_furniture | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | 10.9 | natural_gate_open |
| vehicle_furniture | sentence_completion | greedy | residual_full | center | k1=0.00, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | 26.1 | label_only_gain |
| vehicle_furniture | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 36.5 | baseline |
| vehicle_furniture | sentence_completion | temperature | residual_perp | extended | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 40.3 | no_gate |
| vehicle_furniture | sentence_completion | temperature | residual_parallel | extended | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 13.8 | natural_gate_open |
| vehicle_furniture | sentence_completion | temperature | residual_full | extended | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.40, k10=0.50, k12=0.50 | 23.2 | natural_gate_open |
| vehicle_furniture | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 36.5 | baseline |
| vehicle_furniture | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 38.0 | no_gate |
| vehicle_furniture | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.70, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.70, k5=0.70, k10=0.80, k12=0.80 | 10.9 | weak_natural_gate |
| vehicle_furniture | sentence_completion | top_p | residual_full | center | k1=0.00, k3=0.30, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | 26.1 | label_only_gain |
| vehicle_furniture | sentence_completion | beam | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.8 | baseline |
| vehicle_furniture | sentence_completion | beam | residual_perp | extended | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 40.0 | natural_gate_open |
| vehicle_furniture | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.50, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.70, k12=0.70 | 10.8 | natural_gate_open |
| vehicle_furniture | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | 25.7 | weak_natural_gate |
| vehicle_tool | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_tool | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.50, k12=0.60 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 402.5 | weak_natural_gate |
| vehicle_tool | direct | greedy | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.50, k10=1.00, k12=1.00 | k1=0.00, k3=0.00, k5=0.50, k10=1.00, k12=1.00 | 86.0 | natural_gate_open |
| vehicle_tool | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.50, k12=0.60 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 207.0 | weak_natural_gate |
| vehicle_tool | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_tool | direct | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 416.6 | natural_gate_open |
| vehicle_tool | direct | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.30, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.50, k12=0.50 | 51.6 | natural_gate_open |
| vehicle_tool | direct | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 207.0 | natural_gate_open |
| vehicle_tool | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_tool | direct | top_p | residual_perp | center | k1=0.00, k3=0.10, k5=0.20, k10=0.30, k12=0.30 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.20 | 402.5 | label_only_gain |
| vehicle_tool | direct | top_p | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.30, k10=0.80, k12=0.80 | 86.0 | natural_gate_open |
| vehicle_tool | direct | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | 195.2 | label_only_gain |
| vehicle_tool | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2410.6 | baseline |
| vehicle_tool | direct | beam | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 415.6 | natural_gate_open |
| vehicle_tool | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.40, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.40, k10=0.80, k12=0.80 | 51.8 | natural_gate_open |
| vehicle_tool | direct | beam | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 196.2 | weak_natural_gate |
| vehicle_tool | one_word | greedy | baseline | center | k1=0.20, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.20, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 44.5 | baseline |
| vehicle_tool | one_word | greedy | residual_perp | center | k1=0.30, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.20, k3=0.20, k5=0.40, k10=0.50, k12=0.50 | 21.9 | weak_natural_gate |
| vehicle_tool | one_word | greedy | residual_parallel | center | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 4.5 | natural_gate_open |
| vehicle_tool | one_word | greedy | residual_full | extended | k1=0.40, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | k1=0.40, k3=0.60, k5=0.70, k10=0.70, k12=0.80 | 11.5 | natural_gate_open |
| vehicle_tool | one_word | temperature | baseline | center | k1=0.10, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | k1=0.10, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 44.5 | baseline |
| vehicle_tool | one_word | temperature | residual_perp | extended | k1=0.20, k3=0.50, k5=0.90, k10=1.00, k12=1.00 | k1=0.20, k3=0.30, k5=0.60, k10=0.60, k12=0.70 | 24.8 | natural_gate_open |
| vehicle_tool | one_word | temperature | residual_parallel | center | k1=0.30, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | k1=0.30, k3=0.60, k5=0.90, k10=0.90, k12=0.90 | 4.5 | natural_gate_open |
| vehicle_tool | one_word | temperature | residual_full | extended | k1=0.10, k3=0.30, k5=0.80, k10=0.90, k12=0.90 | k1=0.10, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 11.5 | natural_gate_open |
| vehicle_tool | one_word | top_p | baseline | center | k1=0.10, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 44.5 | baseline |
| vehicle_tool | one_word | top_p | residual_perp | extended | k1=0.20, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | 24.8 | label_only_gain |
| vehicle_tool | one_word | top_p | residual_parallel | center | k1=0.60, k3=0.90, k5=1.00, k10=1.00, k12=1.00 | k1=0.60, k3=0.90, k5=1.00, k10=1.00, k12=1.00 | 4.5 | weak_natural_gate |
| vehicle_tool | one_word | top_p | residual_full | center | k1=0.40, k3=0.70, k5=0.80, k10=1.00, k12=1.00 | k1=0.30, k3=0.40, k5=0.50, k10=0.50, k12=0.50 | 10.5 | weak_natural_gate |
| vehicle_tool | one_word | beam | baseline | center | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 5.3 | baseline |
| vehicle_tool | one_word | beam | residual_perp | center | k1=0.20, k3=0.50, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 3.0 | weak_natural_gate |
| vehicle_tool | one_word | beam | residual_parallel | center | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 1.0 | weak_natural_gate |
| vehicle_tool | one_word | beam | residual_full | center | k1=0.30, k3=0.70, k5=0.80, k10=0.90, k12=0.90 | k1=0.20, k3=0.50, k5=0.60, k10=0.70, k12=0.70 | 1.7 | label_only_gain |
| vehicle_tool | natural_qa | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.60, k10=0.60, k12=0.60 | 141.1 | baseline |
| vehicle_tool | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 83.9 | weak_natural_gate |
| vehicle_tool | natural_qa | greedy | residual_parallel | extended | k1=0.00, k3=0.50, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.50, k5=1.00, k10=1.00, k12=1.00 | 43.8 | natural_gate_open |
| vehicle_tool | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.50, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.50, k5=0.70, k10=0.80, k12=0.80 | 68.9 | weak_natural_gate |
| vehicle_tool | natural_qa | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 141.1 | baseline |
| vehicle_tool | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 83.9 | weak_natural_gate |
| vehicle_tool | natural_qa | temperature | residual_parallel | extended | k1=0.00, k3=0.60, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.60, k5=0.90, k10=1.00, k12=1.00 | 43.8 | natural_gate_open |
| vehicle_tool | natural_qa | temperature | residual_full | extended | k1=0.00, k3=0.50, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.50, k5=0.60, k10=0.60, k12=0.60 | 69.9 | weak_natural_gate |
| vehicle_tool | natural_qa | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.60, k12=0.70 | k1=0.00, k3=0.20, k5=0.30, k10=0.40, k12=0.40 | 141.1 | baseline |
| vehicle_tool | natural_qa | top_p | residual_perp | extended | k1=0.00, k3=0.50, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.50, k5=0.60, k10=0.70, k12=0.70 | 89.7 | natural_gate_open |
| vehicle_tool | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.50, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.50, k5=0.80, k10=0.80, k12=0.80 | 41.0 | weak_natural_gate |
| vehicle_tool | natural_qa | top_p | residual_full | extended | k1=0.00, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.80, k10=0.80, k12=0.80 | 69.9 | weak_natural_gate |
| vehicle_tool | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 141.9 | baseline |
| vehicle_tool | natural_qa | beam | residual_perp | extended | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.70, k10=0.80, k12=0.80 | 89.6 | weak_natural_gate |
| vehicle_tool | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.30, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.30, k5=1.00, k10=1.00, k12=1.00 | 41.0 | natural_gate_open |
| vehicle_tool | natural_qa | beam | residual_full | extended | k1=0.00, k3=0.30, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.70, k10=0.80, k12=0.80 | 69.5 | weak_natural_gate |
| vehicle_tool | definition | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.40, k10=0.70, k12=0.70 | 92.1 | baseline |
| vehicle_tool | definition | greedy | residual_perp | extended | k1=0.00, k3=0.60, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.70, k10=0.70, k12=0.70 | 30.0 | weak_natural_gate |
| vehicle_tool | definition | greedy | residual_parallel | center | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=0.90 | 15.1 | weak_natural_gate |
| vehicle_tool | definition | greedy | residual_full | extended | k1=0.00, k3=0.80, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.80, k5=0.80, k10=0.90, k12=0.90 | 21.5 | weak_natural_gate |
| vehicle_tool | definition | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | 92.1 | baseline |
| vehicle_tool | definition | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.70, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.50, k10=0.70, k12=0.70 | 29.2 | natural_gate_open |
| vehicle_tool | definition | temperature | residual_parallel | center | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=1.00 | k1=0.00, k3=0.90, k5=0.90, k10=0.90, k12=1.00 | 15.1 | natural_gate_open |
| vehicle_tool | definition | temperature | residual_full | extended | k1=0.00, k3=0.30, k5=0.40, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 21.5 | weak_natural_gate |
| vehicle_tool | definition | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 92.1 | baseline |
| vehicle_tool | definition | top_p | residual_perp | extended | k1=0.00, k3=0.40, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.40, k10=0.50, k12=0.50 | 30.0 | natural_gate_open |
| vehicle_tool | definition | top_p | residual_parallel | extended | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.90, k10=1.00, k12=1.00 | 16.7 | natural_gate_open |
| vehicle_tool | definition | top_p | residual_full | center | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | 21.8 | natural_gate_open |
| vehicle_tool | definition | beam | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 91.1 | baseline |
| vehicle_tool | definition | beam | residual_perp | center | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 29.7 | no_gate |
| vehicle_tool | definition | beam | residual_parallel | center | k1=0.00, k3=0.70, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.60, k5=0.70, k10=0.90, k12=0.90 | 15.3 | natural_gate_open |
| vehicle_tool | definition | beam | residual_full | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.60, k5=0.60, k10=0.70, k12=0.70 | 21.5 | weak_natural_gate |
| vehicle_tool | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.5 | baseline |
| vehicle_tool | sentence_completion | greedy | residual_perp | center | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.60 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 27.2 | no_gate |
| vehicle_tool | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | 11.0 | natural_gate_open |
| vehicle_tool | sentence_completion | greedy | residual_full | extended | k1=0.00, k3=0.50, k5=0.70, k10=0.70, k12=0.80 | k1=0.00, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | 19.3 | natural_gate_open |
| vehicle_tool | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.20, k5=0.30, k10=0.50, k12=0.50 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 36.5 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_perp | extended | k1=0.00, k3=0.30, k5=0.40, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 29.4 | natural_gate_open |
| vehicle_tool | sentence_completion | temperature | residual_parallel | extended | k1=0.00, k3=0.40, k5=0.70, k10=0.90, k12=1.00 | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.90 | 11.8 | natural_gate_open |
| vehicle_tool | sentence_completion | temperature | residual_full | center | k1=0.10, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 20.0 | natural_gate_open |
| vehicle_tool | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.50 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.5 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_perp | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 29.4 | natural_gate_open |
| vehicle_tool | sentence_completion | top_p | residual_parallel | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | 11.8 | natural_gate_open |
| vehicle_tool | sentence_completion | top_p | residual_full | extended | k1=0.00, k3=0.30, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.20, k5=0.40, k10=0.60, k12=0.70 | 19.3 | natural_gate_open |
| vehicle_tool | sentence_completion | beam | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.8 | baseline |
| vehicle_tool | sentence_completion | beam | residual_perp | extended | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.70 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 29.2 | natural_gate_open |
| vehicle_tool | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | 11.2 | natural_gate_open |
| vehicle_tool | sentence_completion | beam | residual_full | extended | k1=0.00, k3=0.30, k5=0.80, k10=0.80, k12=0.90 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 19.2 | natural_gate_open |
| vehicle_clothing | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_clothing | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 629.4 | label_only_gain |
| vehicle_clothing | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.20, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.20, k10=0.60, k12=0.60 | 70.9 | natural_gate_open |
| vehicle_clothing | direct | greedy | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 355.4 | weak_natural_gate |
| vehicle_clothing | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2437.8 | baseline |
| vehicle_clothing | direct | temperature | residual_perp | center | k1=0.00, k3=0.10, k5=0.10, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.20 | 629.4 | weak_natural_gate |
| vehicle_clothing | direct | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.30 | 70.9 | natural_gate_open |
| vehicle_clothing | direct | temperature | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 355.4 | natural_gate_open |
| vehicle_clothing | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 2437.8 | baseline |
| vehicle_clothing | direct | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.20, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 629.4 | weak_natural_gate |
| vehicle_clothing | direct | top_p | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.20, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.20, k10=0.70, k12=0.70 | 182.8 | natural_gate_open |
| vehicle_clothing | direct | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | 355.4 | weak_natural_gate |
| vehicle_clothing | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 2410.6 | baseline |
| vehicle_clothing | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.50, k12=0.50 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 633.9 | label_only_gain |
| vehicle_clothing | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.40, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.40, k10=0.80, k12=0.80 | 72.4 | natural_gate_open |
| vehicle_clothing | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.40, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 369.4 | label_only_gain |
| vehicle_clothing | one_word | greedy | baseline | center | k1=0.20, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.20, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 44.5 | baseline |
| vehicle_clothing | one_word | greedy | residual_perp | center | k1=0.10, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.10, k3=0.20, k5=0.40, k10=0.50, k12=0.50 | 51.2 | no_gate |
| vehicle_clothing | one_word | greedy | residual_parallel | center | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.60, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 4.3 | natural_gate_open |
| vehicle_clothing | one_word | greedy | residual_full | center | k1=0.20, k3=0.50, k5=0.60, k10=0.70, k12=0.70 | k1=0.20, k3=0.40, k5=0.50, k10=0.60, k12=0.60 | 30.3 | label_only_gain |
| vehicle_clothing | one_word | temperature | baseline | center | k1=0.20, k3=0.50, k5=0.60, k10=0.80, k12=0.90 | k1=0.20, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | 44.5 | baseline |
| vehicle_clothing | one_word | temperature | residual_perp | center | k1=0.10, k3=0.50, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.40, k10=0.50, k12=0.60 | 51.2 | no_gate |
| vehicle_clothing | one_word | temperature | residual_parallel | center | k1=0.40, k3=0.90, k5=1.00, k10=1.00, k12=1.00 | k1=0.40, k3=0.80, k5=0.90, k10=1.00, k12=1.00 | 4.3 | label_only_gain |
| vehicle_clothing | one_word | temperature | residual_full | extended | k1=0.10, k3=0.40, k5=0.60, k10=0.80, k12=1.00 | k1=0.10, k3=0.30, k5=0.40, k10=0.60, k12=0.70 | 32.3 | label_only_gain |
| vehicle_clothing | one_word | top_p | baseline | center | k1=0.20, k3=0.50, k5=0.50, k10=0.80, k12=0.80 | k1=0.20, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 44.5 | baseline |
| vehicle_clothing | one_word | top_p | residual_perp | extended | k1=0.10, k3=0.50, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | 62.1 | label_only_gain |
| vehicle_clothing | one_word | top_p | residual_parallel | center | k1=0.40, k3=0.80, k5=0.90, k10=1.00, k12=1.00 | k1=0.40, k3=0.80, k5=0.90, k10=0.90, k12=1.00 | 4.3 | weak_natural_gate |
| vehicle_clothing | one_word | top_p | residual_full | extended | k1=0.10, k3=0.60, k5=0.80, k10=0.90, k12=0.90 | k1=0.10, k3=0.30, k5=0.50, k10=0.60, k12=0.70 | 32.3 | label_only_gain |
| vehicle_clothing | one_word | beam | baseline | center | k1=0.00, k3=0.40, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 5.3 | baseline |
| vehicle_clothing | one_word | beam | residual_perp | center | k1=0.10, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.00, k5=0.60, k10=0.70, k12=0.70 | 3.1 | label_only_gain |
| vehicle_clothing | one_word | beam | residual_parallel | center | k1=0.80, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | k1=0.80, k3=1.00, k5=1.00, k10=1.00, k12=1.00 | 1.0 | weak_natural_gate |
| vehicle_clothing | one_word | beam | residual_full | center | k1=0.20, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | k1=0.10, k3=0.10, k5=0.70, k10=0.80, k12=0.80 | 1.7 | label_only_gain |
| vehicle_clothing | natural_qa | greedy | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.60, k10=0.60, k12=0.60 | 141.1 | baseline |
| vehicle_clothing | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.30, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.70, k10=0.80, k12=0.80 | 135.9 | weak_natural_gate |
| vehicle_clothing | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=0.90 | 43.1 | weak_natural_gate |
| vehicle_clothing | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.80 | 105.0 | weak_natural_gate |
| vehicle_clothing | natural_qa | temperature | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.50 | 141.1 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.60, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.50, k10=0.80, k12=0.90 | 135.9 | natural_gate_open |
| vehicle_clothing | natural_qa | temperature | residual_parallel | extended | k1=0.00, k3=0.60, k5=0.90, k10=1.00, k12=1.00 | k1=0.00, k3=0.60, k5=0.90, k10=0.90, k12=0.90 | 52.0 | natural_gate_open |
| vehicle_clothing | natural_qa | temperature | residual_full | center | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.70 | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | 105.0 | no_gate |
| vehicle_clothing | natural_qa | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.50 | 141.1 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_perp | center | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.70, k10=0.70, k12=0.70 | 135.9 | weak_natural_gate |
| vehicle_clothing | natural_qa | top_p | residual_parallel | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.30, k5=0.60, k10=0.90, k12=0.90 | 52.0 | weak_natural_gate |
| vehicle_clothing | natural_qa | top_p | residual_full | extended | k1=0.00, k3=0.70, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.70, k5=0.70, k10=0.70, k12=0.70 | 112.1 | label_only_gain |
| vehicle_clothing | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 141.9 | baseline |
| vehicle_clothing | natural_qa | beam | residual_perp | extended | k1=0.00, k3=0.20, k5=0.70, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.70, k10=0.80, k12=0.80 | 160.1 | weak_natural_gate |
| vehicle_clothing | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.50, k5=0.90, k10=0.90, k12=0.90 | 43.8 | weak_natural_gate |
| vehicle_clothing | natural_qa | beam | residual_full | center | k1=0.00, k3=0.20, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | 104.6 | weak_natural_gate |
| vehicle_clothing | definition | greedy | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.40, k5=0.40, k10=0.70, k12=0.70 | 92.1 | baseline |
| vehicle_clothing | definition | greedy | residual_perp | center | k1=0.00, k3=0.50, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.50, k5=0.60, k10=0.80, k12=0.80 | 64.6 | weak_natural_gate |
| vehicle_clothing | definition | greedy | residual_parallel | center | k1=0.00, k3=0.80, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.70, k5=0.70, k10=0.90, k12=0.90 | 17.7 | weak_natural_gate |
| vehicle_clothing | definition | greedy | residual_full | center | k1=0.00, k3=0.80, k5=0.90, k10=0.90, k12=0.90 | k1=0.00, k3=0.80, k5=0.90, k10=0.90, k12=0.90 | 42.2 | weak_natural_gate |
| vehicle_clothing | definition | temperature | baseline | center | k1=0.00, k3=0.10, k5=0.30, k10=0.40, k12=0.50 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.30 | 92.1 | baseline |
| vehicle_clothing | definition | temperature | residual_perp | center | k1=0.00, k3=0.20, k5=0.40, k10=0.60, k12=0.70 | k1=0.00, k3=0.20, k5=0.30, k10=0.40, k12=0.50 | 64.6 | weak_natural_gate |
| vehicle_clothing | definition | temperature | residual_parallel | center | k1=0.00, k3=0.60, k5=0.70, k10=0.80, k12=0.90 | k1=0.00, k3=0.50, k5=0.60, k10=0.70, k12=0.80 | 17.7 | natural_gate_open |
| vehicle_clothing | definition | temperature | residual_full | center | k1=0.00, k3=0.30, k5=0.50, k10=0.80, k12=0.90 | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.80 | 42.2 | natural_gate_open |
| vehicle_clothing | definition | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 92.1 | baseline |
| vehicle_clothing | definition | top_p | residual_perp | extended | k1=0.00, k3=0.30, k5=0.50, k10=0.80, k12=0.80 | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | 77.2 | natural_gate_open |
| vehicle_clothing | definition | top_p | residual_parallel | center | k1=0.00, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.70, k5=1.00, k10=1.00, k12=1.00 | 17.7 | natural_gate_open |
| vehicle_clothing | definition | top_p | residual_full | extended | k1=0.00, k3=0.30, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.40, k10=0.60, k12=0.60 | 43.2 | weak_natural_gate |
| vehicle_clothing | definition | beam | baseline | center | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.40, k5=0.40, k10=0.60, k12=0.60 | 91.1 | baseline |
| vehicle_clothing | definition | beam | residual_perp | center | k1=0.00, k3=0.40, k5=0.40, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.40, k10=0.80, k12=0.80 | 65.2 | weak_natural_gate |
| vehicle_clothing | definition | beam | residual_parallel | center | k1=0.00, k3=0.70, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.70, k5=0.70, k10=0.80, k12=0.80 | 17.7 | natural_gate_open |
| vehicle_clothing | definition | beam | residual_full | center | k1=0.00, k3=0.50, k5=0.50, k10=0.80, k12=0.80 | k1=0.00, k3=0.50, k5=0.50, k10=0.80, k12=0.80 | 41.7 | weak_natural_gate |
| vehicle_clothing | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.5 | baseline |
| vehicle_clothing | sentence_completion | greedy | residual_perp | center | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.50 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 49.6 | no_gate |
| vehicle_clothing | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.70, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.70, k5=0.70, k10=0.80, k12=0.80 | 13.3 | natural_gate_open |
| vehicle_clothing | sentence_completion | greedy | residual_full | extended | k1=0.00, k3=0.30, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 32.1 | no_gate |
| vehicle_clothing | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.10, k5=0.30, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 36.5 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_perp | center | k1=0.10, k3=0.20, k5=0.30, k10=0.50, k12=0.50 | k1=0.00, k3=0.10, k5=0.20, k10=0.30, k12=0.30 | 49.6 | weak_natural_gate |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | center | k1=0.00, k3=0.80, k5=1.00, k10=1.00, k12=1.00 | k1=0.00, k3=0.80, k5=0.80, k10=0.90, k12=0.90 | 13.3 | natural_gate_open |
| vehicle_clothing | sentence_completion | temperature | residual_full | extended | k1=0.00, k3=0.40, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.50, k10=0.50, k12=0.50 | 32.1 | natural_gate_open |
| vehicle_clothing | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.20, k10=0.40, k12=0.40 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 36.5 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.30, k5=0.40, k10=0.50, k12=0.60 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 49.6 | weak_natural_gate |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | extended | k1=0.00, k3=0.40, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | 16.6 | natural_gate_open |
| vehicle_clothing | sentence_completion | top_p | residual_full | center | k1=0.00, k3=0.40, k5=0.80, k10=0.80, k12=0.90 | k1=0.00, k3=0.40, k5=0.50, k10=0.50, k12=0.50 | 31.2 | natural_gate_open |
| vehicle_clothing | sentence_completion | beam | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 36.8 | baseline |
| vehicle_clothing | sentence_completion | beam | residual_perp | extended | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 62.1 | weak_natural_gate |
| vehicle_clothing | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.70, k5=0.90, k10=0.90, k12=1.00 | k1=0.00, k3=0.70, k5=0.80, k10=0.80, k12=0.80 | 13.2 | natural_gate_open |
| vehicle_clothing | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.20, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 31.3 | weak_natural_gate |

## deepseek7b

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], scaffolds=['direct', 'one_word', 'natural_qa', 'definition', 'sentence_completion'], modes=['greedy', 'temperature', 'top_p', 'beam'], windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=10, max_new_tokens=12, alpha=6.0, attn=sdpa

| source | scaffold | mode | condition | win | family target curve | exact target curve | rankT | class |
|---|---|---|---|---|---|---|---:|---|
| vehicle_furniture | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_furniture | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46633.9 | rank_only |
| vehicle_furniture | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 41953.4 | weak_natural_gate |
| vehicle_furniture | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46846.7 | rank_only |
| vehicle_furniture | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_furniture | direct | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46633.9 | rank_only |
| vehicle_furniture | direct | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 35464.9 | weak_natural_gate |
| vehicle_furniture | direct | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46846.7 | rank_only |
| vehicle_furniture | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_furniture | direct | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45520.1 | weak_natural_gate |
| vehicle_furniture | direct | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 41953.4 | weak_natural_gate |
| vehicle_furniture | direct | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45712.2 | weak_natural_gate |
| vehicle_furniture | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49374.7 | baseline |
| vehicle_furniture | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 47689.3 | weak_natural_gate |
| vehicle_furniture | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 43505.9 | rank_only |
| vehicle_furniture | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 47795.6 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | baseline | center | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 35.8 | baseline |
| vehicle_furniture | one_word | greedy | residual_perp | center | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 26.2 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | residual_parallel | center | k1=0.10, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | k1=0.10, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 30.6 | weak_natural_gate |
| vehicle_furniture | one_word | greedy | residual_full | center | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 26.2 | weak_natural_gate |
| vehicle_furniture | one_word | temperature | baseline | center | k1=0.30, k3=0.30, k5=0.30, k10=0.40, k12=0.50 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.40 | 35.8 | baseline |
| vehicle_furniture | one_word | temperature | residual_perp | center | k1=0.10, k3=0.50, k5=0.50, k10=0.50, k12=0.60 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 26.2 | no_gate |
| vehicle_furniture | one_word | temperature | residual_parallel | center | k1=0.20, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 30.6 | no_gate |
| vehicle_furniture | one_word | temperature | residual_full | extended | k1=0.10, k3=0.30, k5=0.30, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 23.9 | no_gate |
| vehicle_furniture | one_word | top_p | baseline | center | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.00, k3=0.20, k5=0.20, k10=0.20, k12=0.20 | 35.8 | baseline |
| vehicle_furniture | one_word | top_p | residual_perp | center | k1=0.00, k3=0.10, k5=0.20, k10=0.60, k12=0.60 | k1=0.00, k3=0.10, k5=0.20, k10=0.30, k12=0.30 | 26.2 | natural_gate_open |
| vehicle_furniture | one_word | top_p | residual_parallel | extended | k1=0.00, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 24.3 | natural_gate_open |
| vehicle_furniture | one_word | top_p | residual_full | center | k1=0.20, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 26.2 | no_gate |
| vehicle_furniture | one_word | beam | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 35.6 | baseline |
| vehicle_furniture | one_word | beam | residual_perp | center | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | k1=0.00, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 27.1 | no_gate |
| vehicle_furniture | one_word | beam | residual_parallel | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 30.0 | no_gate |
| vehicle_furniture | one_word | beam | residual_full | center | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 26.3 | no_gate |
| vehicle_furniture | natural_qa | greedy | baseline | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 384.8 | baseline |
| vehicle_furniture | natural_qa | greedy | residual_perp | extended | k1=0.00, k3=0.10, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.80 | 304.5 | weak_natural_gate |
| vehicle_furniture | natural_qa | greedy | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.80 | 266.0 | weak_natural_gate |
| vehicle_furniture | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 330.9 | rank_only |
| vehicle_furniture | natural_qa | temperature | baseline | center | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.10, k5=0.40, k10=0.50, k12=0.50 | 384.8 | baseline |
| vehicle_furniture | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.40, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.40, k10=0.50, k12=0.50 | 334.3 | rank_only |
| vehicle_furniture | natural_qa | temperature | residual_parallel | center | k1=0.00, k3=0.20, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.50, k10=0.50, k12=0.50 | 323.6 | rank_only |
| vehicle_furniture | natural_qa | temperature | residual_full | center | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.50 | k1=0.00, k3=0.10, k5=0.40, k10=0.40, k12=0.40 | 330.9 | rank_only |
| vehicle_furniture | natural_qa | top_p | baseline | center | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.30, k10=0.40, k12=0.40 | 384.8 | baseline |
| vehicle_furniture | natural_qa | top_p | residual_perp | extended | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.50 | 304.5 | weak_natural_gate |
| vehicle_furniture | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.60 | k1=0.00, k3=0.10, k5=0.40, k10=0.40, k12=0.40 | 323.6 | weak_natural_gate |
| vehicle_furniture | natural_qa | top_p | residual_full | extended | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.90 | k1=0.00, k3=0.10, k5=0.40, k10=0.50, k12=0.60 | 313.1 | natural_gate_open |
| vehicle_furniture | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 528.3 | baseline |
| vehicle_furniture | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 456.0 | rank_only |
| vehicle_furniture | natural_qa | beam | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.80 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 398.0 | weak_natural_gate |
| vehicle_furniture | natural_qa | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 472.8 | rank_only |
| vehicle_furniture | definition | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1261.7 | baseline |
| vehicle_furniture | definition | greedy | residual_perp | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 740.4 | no_gate |
| vehicle_furniture | definition | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 1025.5 | no_gate |
| vehicle_furniture | definition | greedy | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 765.3 | no_gate |
| vehicle_furniture | definition | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 1261.7 | baseline |
| vehicle_furniture | definition | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 888.3 | rank_only |
| vehicle_furniture | definition | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1025.5 | label_only_gain |
| vehicle_furniture | definition | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | 902.5 | label_only_gain |
| vehicle_furniture | definition | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.20 | 1261.7 | baseline |
| vehicle_furniture | definition | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 740.4 | rank_only |
| vehicle_furniture | definition | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 1025.5 | rank_only |
| vehicle_furniture | definition | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.30 | 902.5 | rank_only |
| vehicle_furniture | definition | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1281.1 | baseline |
| vehicle_furniture | definition | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 872.6 | rank_only |
| vehicle_furniture | definition | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1017.2 | rank_only |
| vehicle_furniture | definition | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 886.9 | rank_only |
| vehicle_furniture | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1257.6 | baseline |
| vehicle_furniture | sentence_completion | greedy | residual_perp | extended | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.20 | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.20 | 962.9 | rank_only |
| vehicle_furniture | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 1019.1 | rank_only |
| vehicle_furniture | sentence_completion | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1043.2 | rank_only |
| vehicle_furniture | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 1257.6 | baseline |
| vehicle_furniture | sentence_completion | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 962.9 | rank_only |
| vehicle_furniture | sentence_completion | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 1019.1 | rank_only |
| vehicle_furniture | sentence_completion | temperature | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 982.4 | rank_only |
| vehicle_furniture | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 1257.6 | baseline |
| vehicle_furniture | sentence_completion | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.20, k10=0.20, k12=0.20 | 962.9 | weak_natural_gate |
| vehicle_furniture | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1019.1 | rank_only |
| vehicle_furniture | sentence_completion | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 982.4 | weak_natural_gate |
| vehicle_furniture | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1354.0 | baseline |
| vehicle_furniture | sentence_completion | beam | residual_perp | extended | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 1062.6 | weak_natural_gate |
| vehicle_furniture | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1112.0 | rank_only |
| vehicle_furniture | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1145.7 | rank_only |
| vehicle_tool | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_tool | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44877.8 | rank_only |
| vehicle_tool | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 32730.0 | weak_natural_gate |
| vehicle_tool | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45221.5 | rank_only |
| vehicle_tool | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_tool | direct | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44877.8 | rank_only |
| vehicle_tool | direct | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 32730.0 | rank_only |
| vehicle_tool | direct | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45221.5 | rank_only |
| vehicle_tool | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_tool | direct | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44289.4 | weak_natural_gate |
| vehicle_tool | direct | top_p | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 27383.0 | weak_natural_gate |
| vehicle_tool | direct | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45221.5 | rank_only |
| vehicle_tool | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49374.7 | baseline |
| vehicle_tool | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46228.5 | weak_natural_gate |
| vehicle_tool | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 33631.6 | rank_only |
| vehicle_tool | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 46084.9 | weak_natural_gate |
| vehicle_tool | one_word | greedy | baseline | center | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 35.8 | baseline |
| vehicle_tool | one_word | greedy | residual_perp | center | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 27.8 | weak_natural_gate |
| vehicle_tool | one_word | greedy | residual_parallel | extended | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.20, k3=0.40, k5=0.40, k10=0.40, k12=0.50 | 17.1 | weak_natural_gate |
| vehicle_tool | one_word | greedy | residual_full | center | k1=0.20, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.10, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 27.0 | natural_gate_open |
| vehicle_tool | one_word | temperature | baseline | center | k1=0.00, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 35.8 | baseline |
| vehicle_tool | one_word | temperature | residual_perp | extended | k1=0.20, k3=0.20, k5=0.40, k10=0.50, k12=0.50 | k1=0.10, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | 26.8 | no_gate |
| vehicle_tool | one_word | temperature | residual_parallel | center | k1=0.20, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.10, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 21.5 | no_gate |
| vehicle_tool | one_word | temperature | residual_full | center | k1=0.00, k3=0.30, k5=0.40, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.30 | 27.0 | natural_gate_open |
| vehicle_tool | one_word | top_p | baseline | center | k1=0.00, k3=0.30, k5=0.30, k10=0.50, k12=0.60 | k1=0.00, k3=0.20, k5=0.20, k10=0.40, k12=0.40 | 35.8 | baseline |
| vehicle_tool | one_word | top_p | residual_perp | extended | k1=0.10, k3=0.40, k5=0.50, k10=0.70, k12=0.70 | k1=0.00, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 26.8 | no_gate |
| vehicle_tool | one_word | top_p | residual_parallel | extended | k1=0.30, k3=0.60, k5=0.70, k10=0.80, k12=0.80 | k1=0.20, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | 17.1 | weak_natural_gate |
| vehicle_tool | one_word | top_p | residual_full | center | k1=0.30, k3=0.50, k5=0.50, k10=0.70, k12=0.70 | k1=0.20, k3=0.20, k5=0.20, k10=0.30, k12=0.30 | 27.0 | no_gate |
| vehicle_tool | one_word | beam | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 35.6 | baseline |
| vehicle_tool | one_word | beam | residual_perp | center | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.00, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 27.4 | no_gate |
| vehicle_tool | one_word | beam | residual_parallel | center | k1=0.10, k3=0.40, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.40, k5=0.50, k10=0.50, k12=0.50 | 21.3 | no_gate |
| vehicle_tool | one_word | beam | residual_full | extended | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 26.0 | no_gate |
| vehicle_tool | natural_qa | greedy | baseline | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 384.8 | baseline |
| vehicle_tool | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 321.5 | rank_only |
| vehicle_tool | natural_qa | greedy | residual_parallel | center | k1=0.00, k3=0.10, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.80 | 242.4 | weak_natural_gate |
| vehicle_tool | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 322.5 | rank_only |
| vehicle_tool | natural_qa | temperature | baseline | center | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | k1=0.00, k3=0.10, k5=0.30, k10=0.40, k12=0.40 | 384.8 | baseline |
| vehicle_tool | natural_qa | temperature | residual_perp | extended | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.40, k10=0.50, k12=0.50 | 300.2 | no_gate |
| vehicle_tool | natural_qa | temperature | residual_parallel | center | k1=0.00, k3=0.40, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.40, k5=0.60, k10=0.70, k12=0.70 | 242.4 | weak_natural_gate |
| vehicle_tool | natural_qa | temperature | residual_full | extended | k1=0.00, k3=0.30, k5=0.70, k10=0.70, k12=0.80 | k1=0.00, k3=0.30, k5=0.60, k10=0.60, k12=0.70 | 292.1 | weak_natural_gate |
| vehicle_tool | natural_qa | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.10, k5=0.40, k10=0.40, k12=0.40 | 384.8 | baseline |
| vehicle_tool | natural_qa | top_p | residual_perp | center | k1=0.00, k3=0.20, k5=0.70, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.50, k10=0.60, k12=0.60 | 321.5 | weak_natural_gate |
| vehicle_tool | natural_qa | top_p | residual_parallel | extended | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.20, k5=0.80, k10=0.80, k12=0.80 | 199.9 | weak_natural_gate |
| vehicle_tool | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.20, k5=0.60, k10=0.70, k12=0.70 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 322.5 | no_gate |
| vehicle_tool | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 528.3 | baseline |
| vehicle_tool | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 459.1 | rank_only |
| vehicle_tool | natural_qa | beam | residual_parallel | center | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.80 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 351.7 | weak_natural_gate |
| vehicle_tool | natural_qa | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 451.2 | rank_only |
| vehicle_tool | definition | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1261.7 | baseline |
| vehicle_tool | definition | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 873.8 | rank_only |
| vehicle_tool | definition | greedy | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 543.9 | no_gate |
| vehicle_tool | definition | greedy | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 759.8 | no_gate |
| vehicle_tool | definition | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1261.7 | baseline |
| vehicle_tool | definition | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 873.8 | rank_only |
| vehicle_tool | definition | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.20, k10=0.30, k12=0.40 | 727.4 | natural_gate_open |
| vehicle_tool | definition | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 857.7 | weak_natural_gate |
| vehicle_tool | definition | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 1261.7 | baseline |
| vehicle_tool | definition | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 873.8 | rank_only |
| vehicle_tool | definition | top_p | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | 543.9 | label_only_gain |
| vehicle_tool | definition | top_p | residual_full | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 759.8 | rank_only |
| vehicle_tool | definition | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1281.1 | baseline |
| vehicle_tool | definition | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 904.3 | rank_only |
| vehicle_tool | definition | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 744.9 | rank_only |
| vehicle_tool | definition | beam | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 721.3 | weak_natural_gate |
| vehicle_tool | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1257.6 | baseline |
| vehicle_tool | sentence_completion | greedy | residual_perp | extended | k1=0.00, k3=0.10, k5=0.20, k10=0.30, k12=0.30 | k1=0.00, k3=0.10, k5=0.20, k10=0.30, k12=0.30 | 981.3 | no_gate |
| vehicle_tool | sentence_completion | greedy | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | 480.3 | no_gate |
| vehicle_tool | sentence_completion | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1009.0 | rank_only |
| vehicle_tool | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1257.6 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 981.3 | weak_natural_gate |
| vehicle_tool | sentence_completion | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.40 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 480.3 | natural_gate_open |
| vehicle_tool | sentence_completion | temperature | residual_full | center | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.10, k5=0.10, k10=0.10, k12=0.10 | 1009.0 | natural_gate_open |
| vehicle_tool | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1257.6 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_perp | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 981.3 | natural_gate_open |
| vehicle_tool | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 620.5 | weak_natural_gate |
| vehicle_tool | sentence_completion | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1009.0 | weak_natural_gate |
| vehicle_tool | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1354.0 | baseline |
| vehicle_tool | sentence_completion | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1129.4 | rank_only |
| vehicle_tool | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 669.9 | rank_only |
| vehicle_tool | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1114.2 | rank_only |
| vehicle_clothing | direct | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_clothing | direct | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44957.3 | rank_only |
| vehicle_clothing | direct | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 41572.6 | rank_only |
| vehicle_clothing | direct | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44911.2 | rank_only |
| vehicle_clothing | direct | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_clothing | direct | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 43301.0 | rank_only |
| vehicle_clothing | direct | temperature | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 41572.6 | rank_only |
| vehicle_clothing | direct | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44911.2 | rank_only |
| vehicle_clothing | direct | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49019.2 | baseline |
| vehicle_clothing | direct | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 44957.3 | weak_natural_gate |
| vehicle_clothing | direct | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 41572.6 | weak_natural_gate |
| vehicle_clothing | direct | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 44911.2 | rank_only |
| vehicle_clothing | direct | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 49374.7 | baseline |
| vehicle_clothing | direct | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45635.5 | rank_only |
| vehicle_clothing | direct | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 42160.4 | rank_only |
| vehicle_clothing | direct | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 45670.0 | rank_only |
| vehicle_clothing | one_word | greedy | baseline | center | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.30 | 35.8 | baseline |
| vehicle_clothing | one_word | greedy | residual_perp | extended | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.30, k12=0.40 | 26.6 | weak_natural_gate |
| vehicle_clothing | one_word | greedy | residual_parallel | extended | k1=0.10, k3=0.40, k5=0.40, k10=0.40, k12=0.40 | k1=0.10, k3=0.30, k5=0.30, k10=0.30, k12=0.40 | 20.6 | weak_natural_gate |
| vehicle_clothing | one_word | greedy | residual_full | center | k1=0.20, k3=0.50, k5=0.50, k10=0.50, k12=0.50 | k1=0.10, k3=0.20, k5=0.20, k10=0.20, k12=0.30 | 29.5 | weak_natural_gate |
| vehicle_clothing | one_word | temperature | baseline | center | k1=0.10, k3=0.30, k5=0.40, k10=0.60, k12=0.60 | k1=0.10, k3=0.30, k5=0.40, k10=0.50, k12=0.50 | 35.8 | baseline |
| vehicle_clothing | one_word | temperature | residual_perp | center | k1=0.20, k3=0.40, k5=0.40, k10=0.60, k12=0.70 | k1=0.10, k3=0.10, k5=0.10, k10=0.30, k12=0.40 | 28.9 | no_gate |
| vehicle_clothing | one_word | temperature | residual_parallel | center | k1=0.10, k3=0.40, k5=0.50, k10=0.60, k12=0.70 | k1=0.10, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 25.2 | no_gate |
| vehicle_clothing | one_word | temperature | residual_full | extended | k1=0.20, k3=0.50, k5=0.50, k10=0.60, k12=0.60 | k1=0.10, k3=0.10, k5=0.10, k10=0.20, k12=0.20 | 28.2 | no_gate |
| vehicle_clothing | one_word | top_p | baseline | center | k1=0.20, k3=0.40, k5=0.40, k10=0.50, k12=0.60 | k1=0.10, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 35.8 | baseline |
| vehicle_clothing | one_word | top_p | residual_perp | center | k1=0.00, k3=0.10, k5=0.20, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.30 | 28.9 | no_gate |
| vehicle_clothing | one_word | top_p | residual_parallel | center | k1=0.20, k3=0.50, k5=0.50, k10=0.60, k12=0.70 | k1=0.20, k3=0.30, k5=0.30, k10=0.40, k12=0.40 | 25.2 | no_gate |
| vehicle_clothing | one_word | top_p | residual_full | extended | k1=0.10, k3=0.30, k5=0.40, k10=0.40, k12=0.60 | k1=0.00, k3=0.20, k5=0.30, k10=0.30, k12=0.40 | 28.2 | no_gate |
| vehicle_clothing | one_word | beam | baseline | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 35.6 | baseline |
| vehicle_clothing | one_word | beam | residual_perp | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.40 | 29.0 | no_gate |
| vehicle_clothing | one_word | beam | residual_parallel | center | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | k1=0.00, k3=0.30, k5=0.40, k10=0.40, k12=0.50 | 25.0 | no_gate |
| vehicle_clothing | one_word | beam | residual_full | center | k1=0.00, k3=0.20, k5=0.30, k10=0.40, k12=0.40 | k1=0.00, k3=0.20, k5=0.30, k10=0.40, k12=0.40 | 28.2 | no_gate |
| vehicle_clothing | natural_qa | greedy | baseline | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 384.8 | baseline |
| vehicle_clothing | natural_qa | greedy | residual_perp | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 333.9 | rank_only |
| vehicle_clothing | natural_qa | greedy | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.80, k10=0.90, k12=0.90 | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.80 | 251.1 | weak_natural_gate |
| vehicle_clothing | natural_qa | greedy | residual_full | center | k1=0.00, k3=0.10, k5=0.70, k10=0.70, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 321.0 | rank_only |
| vehicle_clothing | natural_qa | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.00, k5=0.50, k10=0.50, k12=0.50 | 384.8 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_perp | center | k1=0.00, k3=0.30, k5=0.60, k10=0.70, k12=0.80 | k1=0.00, k3=0.20, k5=0.40, k10=0.50, k12=0.60 | 333.9 | weak_natural_gate |
| vehicle_clothing | natural_qa | temperature | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.70 | 251.1 | weak_natural_gate |
| vehicle_clothing | natural_qa | temperature | residual_full | extended | k1=0.00, k3=0.10, k5=0.80, k10=0.80, k12=0.80 | k1=0.00, k3=0.00, k5=0.50, k10=0.50, k12=0.60 | 299.0 | weak_natural_gate |
| vehicle_clothing | natural_qa | top_p | baseline | center | k1=0.00, k3=0.10, k5=0.40, k10=0.60, k12=0.60 | k1=0.00, k3=0.10, k5=0.40, k10=0.40, k12=0.40 | 384.8 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_perp | extended | k1=0.00, k3=0.20, k5=0.60, k10=0.60, k12=0.60 | k1=0.00, k3=0.20, k5=0.40, k10=0.40, k12=0.40 | 301.7 | rank_only |
| vehicle_clothing | natural_qa | top_p | residual_parallel | center | k1=0.00, k3=0.10, k5=0.60, k10=0.80, k12=0.80 | k1=0.00, k3=0.10, k5=0.40, k10=0.50, k12=0.60 | 298.8 | weak_natural_gate |
| vehicle_clothing | natural_qa | top_p | residual_full | center | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.60 | k1=0.00, k3=0.10, k5=0.50, k10=0.50, k12=0.50 | 321.0 | rank_only |
| vehicle_clothing | natural_qa | beam | baseline | center | k1=0.00, k3=0.10, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.10, k5=0.50, k10=0.60, k12=0.60 | 528.3 | baseline |
| vehicle_clothing | natural_qa | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 459.7 | rank_only |
| vehicle_clothing | natural_qa | beam | residual_parallel | extended | k1=0.00, k3=0.10, k5=0.60, k10=0.70, k12=0.80 | k1=0.00, k3=0.10, k5=0.50, k10=0.70, k12=0.70 | 353.6 | weak_natural_gate |
| vehicle_clothing | natural_qa | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.60, k10=0.60, k12=0.70 | k1=0.00, k3=0.00, k5=0.50, k10=0.60, k12=0.60 | 470.9 | rank_only |
| vehicle_clothing | definition | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1261.7 | baseline |
| vehicle_clothing | definition | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 856.1 | rank_only |
| vehicle_clothing | definition | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | 986.2 | no_gate |
| vehicle_clothing | definition | greedy | residual_full | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 853.2 | rank_only |
| vehicle_clothing | definition | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | 1261.7 | baseline |
| vehicle_clothing | definition | temperature | residual_perp | extended | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 738.3 | no_gate |
| vehicle_clothing | definition | temperature | residual_parallel | center | k1=0.00, k3=0.10, k5=0.10, k10=0.40, k12=0.50 | k1=0.00, k3=0.10, k5=0.10, k10=0.30, k12=0.30 | 986.2 | natural_gate_open |
| vehicle_clothing | definition | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.10, k12=0.10 | 853.2 | rank_only |
| vehicle_clothing | definition | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 1261.7 | baseline |
| vehicle_clothing | definition | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 856.1 | weak_natural_gate |
| vehicle_clothing | definition | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 986.2 | weak_natural_gate |
| vehicle_clothing | definition | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.30 | 853.2 | weak_natural_gate |
| vehicle_clothing | definition | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1281.1 | baseline |
| vehicle_clothing | definition | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 876.7 | rank_only |
| vehicle_clothing | definition | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1009.0 | rank_only |
| vehicle_clothing | definition | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 874.0 | rank_only |
| vehicle_clothing | sentence_completion | greedy | baseline | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 1257.6 | baseline |
| vehicle_clothing | sentence_completion | greedy | residual_perp | center | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 978.9 | rank_only |
| vehicle_clothing | sentence_completion | greedy | residual_parallel | center | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | 983.1 | rank_only |
| vehicle_clothing | sentence_completion | greedy | residual_full | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.30, k12=0.30 | k1=0.00, k3=0.00, k5=0.10, k10=0.20, k12=0.20 | 966.2 | no_gate |
| vehicle_clothing | sentence_completion | temperature | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1257.6 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 978.9 | rank_only |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | extended | k1=0.00, k3=0.00, k5=0.10, k10=0.10, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 746.2 | rank_only |
| vehicle_clothing | sentence_completion | temperature | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1017.1 | rank_only |
| vehicle_clothing | sentence_completion | top_p | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1257.6 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.10 | 978.9 | label_only_gain |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 983.1 | rank_only |
| vehicle_clothing | sentence_completion | top_p | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.20, k12=0.20 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1017.1 | rank_only |
| vehicle_clothing | sentence_completion | beam | baseline | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1354.0 | baseline |
| vehicle_clothing | sentence_completion | beam | residual_perp | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1136.1 | rank_only |
| vehicle_clothing | sentence_completion | beam | residual_parallel | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1033.9 | rank_only |
| vehicle_clothing | sentence_completion | beam | residual_full | center | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | k1=0.00, k3=0.00, k5=0.00, k10=0.00, k12=0.00 | 1125.4 | rank_only |

## Best Family-Hit Positive Rows

| model | source | scaffold | mode | condition | win | base family | family | gain | exact gain | rank improve | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| glm4 | vehicle_clothing | sentence_completion | temperature | residual_parallel | center | 0.30 | 1.00 | +0.70 | +0.90 | 23.2 | natural_gate_open |
| glm4 | vehicle_furniture | direct | temperature | residual_parallel | center | 0.20 | 0.90 | +0.70 | +0.80 | 2360.9 | natural_gate_open |
| glm4 | vehicle_tool | direct | temperature | residual_parallel | center | 0.00 | 0.70 | +0.70 | +0.50 | 2386.2 | natural_gate_open |
| glm4 | vehicle_tool | direct | top_p | residual_parallel | extended | 0.20 | 0.80 | +0.60 | +0.80 | 2351.8 | natural_gate_open |
| qwen3 | vehicle_clothing | direct | top_p | residual_parallel | center | 0.20 | 0.80 | +0.60 | +0.30 | 218.8 | natural_gate_open |
| glm4 | vehicle_tool | direct | greedy | residual_parallel | extended | 0.40 | 1.00 | +0.60 | +1.00 | 2351.8 | natural_gate_open |
| glm4 | vehicle_tool | definition | top_p | residual_parallel | extended | 0.40 | 1.00 | +0.60 | +0.70 | 75.4 | natural_gate_open |
| glm4 | vehicle_clothing | sentence_completion | beam | residual_parallel | center | 0.40 | 1.00 | +0.60 | +0.60 | 23.6 | natural_gate_open |
| glm4 | vehicle_tool | definition | temperature | residual_parallel | center | 0.40 | 1.00 | +0.60 | +0.60 | 77.0 | natural_gate_open |
| glm4 | vehicle_tool | definition | temperature | residual_perp | center | 0.40 | 1.00 | +0.60 | +0.30 | 62.9 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_parallel | extended | 0.50 | 1.00 | +0.50 | +0.80 | 24.7 | natural_gate_open |
| glm4 | vehicle_clothing | sentence_completion | top_p | residual_parallel | extended | 0.40 | 0.90 | +0.50 | +0.70 | 19.9 | natural_gate_open |
| glm4 | vehicle_furniture | sentence_completion | greedy | residual_parallel | center | 0.50 | 1.00 | +0.50 | +0.70 | 25.6 | natural_gate_open |
| glm4 | vehicle_clothing | definition | top_p | residual_parallel | center | 0.50 | 1.00 | +0.50 | +0.70 | 74.4 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | greedy | residual_parallel | center | 0.50 | 1.00 | +0.50 | +0.60 | 25.5 | natural_gate_open |
| glm4 | vehicle_clothing | direct | top_p | residual_parallel | extended | 0.30 | 0.80 | +0.50 | +0.60 | 2255.0 | natural_gate_open |
| glm4 | vehicle_furniture | one_word | temperature | residual_parallel | center | 0.50 | 1.00 | +0.50 | +0.50 | 40.4 | natural_gate_open |
| glm4 | vehicle_furniture | sentence_completion | beam | residual_parallel | center | 0.40 | 0.90 | +0.50 | +0.50 | 26.0 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_full | extended | 0.50 | 1.00 | +0.50 | +0.50 | 17.2 | natural_gate_open |
| glm4 | vehicle_clothing | sentence_completion | top_p | residual_full | center | 0.40 | 0.90 | +0.50 | +0.40 | 5.3 | natural_gate_open |
| glm4 | vehicle_furniture | sentence_completion | temperature | residual_parallel | extended | 0.40 | 0.90 | +0.50 | +0.40 | 22.7 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | beam | residual_parallel | center | 0.40 | 0.90 | +0.50 | +0.40 | 25.6 | natural_gate_open |
| glm4 | vehicle_tool | definition | top_p | residual_perp | extended | 0.40 | 0.90 | +0.50 | +0.20 | 62.1 | natural_gate_open |
| qwen3 | vehicle_clothing | direct | temperature | residual_full | extended | 0.10 | 0.60 | +0.50 | +0.20 | 93.1 | natural_gate_open |
| deepseek7b | vehicle_furniture | natural_qa | top_p | residual_full | extended | 0.40 | 0.90 | +0.50 | +0.20 | 71.7 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | beam | residual_full | extended | 0.40 | 0.90 | +0.50 | +0.10 | 17.6 | natural_gate_open |
| glm4 | vehicle_tool | direct | temperature | residual_perp | extended | 0.00 | 0.50 | +0.50 | +0.00 | 2021.2 | natural_gate_open |
| qwen3 | vehicle_clothing | direct | top_p | residual_full | center | 0.20 | 0.70 | +0.50 | +0.10 | 98.8 | natural_gate_open |
| glm4 | vehicle_furniture | direct | beam | residual_parallel | center | 0.40 | 0.80 | +0.40 | +0.80 | 2332.5 | natural_gate_open |
| glm4 | vehicle_tool | direct | beam | residual_parallel | center | 0.40 | 0.80 | +0.40 | +0.80 | 2358.8 | natural_gate_open |
| glm4 | vehicle_clothing | direct | beam | residual_parallel | center | 0.40 | 0.80 | +0.40 | +0.80 | 2338.2 | natural_gate_open |
| glm4 | vehicle_tool | natural_qa | temperature | residual_parallel | extended | 0.60 | 1.00 | +0.40 | +0.70 | 97.3 | natural_gate_open |
| glm4 | vehicle_tool | one_word | temperature | residual_parallel | center | 0.60 | 1.00 | +0.40 | +0.60 | 40.0 | natural_gate_open |
| glm4 | vehicle_clothing | sentence_completion | greedy | residual_parallel | center | 0.50 | 0.90 | +0.40 | +0.60 | 23.2 | natural_gate_open |
| glm4 | vehicle_furniture | natural_qa | top_p | residual_parallel | extended | 0.60 | 1.00 | +0.40 | +0.50 | 91.0 | natural_gate_open |
| glm4 | vehicle_clothing | definition | temperature | residual_parallel | center | 0.50 | 0.90 | +0.40 | +0.50 | 74.4 | natural_gate_open |
| glm4 | vehicle_clothing | definition | temperature | residual_full | center | 0.50 | 0.90 | +0.40 | +0.50 | 49.9 | natural_gate_open |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_parallel | extended | 0.50 | 0.90 | +0.40 | +0.50 | 24.7 | natural_gate_open |
| glm4 | vehicle_furniture | natural_qa | temperature | residual_parallel | center | 0.60 | 1.00 | +0.40 | +0.40 | 96.6 | natural_gate_open |
| glm4 | vehicle_clothing | natural_qa | temperature | residual_parallel | extended | 0.60 | 1.00 | +0.40 | +0.40 | 89.1 | natural_gate_open |
| glm4 | vehicle_tool | one_word | temperature | residual_perp | extended | 0.60 | 1.00 | +0.40 | +0.40 | 19.7 | natural_gate_open |
| glm4 | vehicle_clothing | direct | temperature | residual_parallel | center | 0.10 | 0.50 | +0.40 | +0.30 | 2366.9 | natural_gate_open |
| qwen3 | vehicle_tool | one_word | top_p | residual_parallel | center | 0.50 | 0.90 | +0.40 | +0.20 | 0.7 | natural_gate_open |
| qwen3 | vehicle_tool | one_word | top_p | residual_full | extended | 0.50 | 0.90 | +0.40 | +0.20 | 0.6 | natural_gate_open |
| glm4 | vehicle_clothing | direct | temperature | residual_full | extended | 0.10 | 0.50 | +0.40 | +0.20 | 2082.4 | natural_gate_open |
| glm4 | vehicle_furniture | one_word | temperature | residual_perp | center | 0.50 | 0.90 | +0.40 | +0.20 | 14.3 | natural_gate_open |
| glm4 | vehicle_furniture | one_word | temperature | residual_full | extended | 0.50 | 0.90 | +0.40 | +0.20 | 29.1 | natural_gate_open |
| deepseek7b | vehicle_tool | sentence_completion | temperature | residual_parallel | extended | 0.00 | 0.40 | +0.40 | +0.10 | 777.3 | natural_gate_open |

## Decode-Mode Max Gain

| model | mode | max family gain | row |
|---|---|---:|---|
| deepseek7b | beam | +0.10 | vehicle_furniture natural_qa residual_parallel extended |
| deepseek7b | greedy | +0.30 | vehicle_tool one_word residual_full center |
| deepseek7b | temperature | +0.40 | vehicle_tool sentence_completion residual_parallel extended |
| deepseek7b | top_p | +0.50 | vehicle_furniture natural_qa residual_full extended |
| glm4 | beam | +0.60 | vehicle_clothing sentence_completion residual_parallel center |
| glm4 | greedy | +0.60 | vehicle_tool direct residual_parallel extended |
| glm4 | temperature | +0.70 | vehicle_furniture direct residual_parallel center |
| glm4 | top_p | +0.60 | vehicle_tool direct residual_parallel extended |
| qwen3 | beam | +0.20 | vehicle_clothing one_word residual_perp center |
| qwen3 | greedy | +0.20 | vehicle_furniture one_word residual_perp center |
| qwen3 | temperature | +0.50 | vehicle_clothing direct residual_full extended |
| qwen3 | top_p | +0.60 | vehicle_clothing direct residual_parallel center |

