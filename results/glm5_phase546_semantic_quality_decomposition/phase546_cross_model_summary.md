# Phase546 Semantic Quality Decomposition Summary

## qwen3

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'10-12-14': [10, 12, 14]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | vehicle_tool | natural_qa | top_p | baseline | 10-12-14 | 0.98 | 0.98 | +0.00 | 0.85 | +0.00 | 0.12 | +0.00 | 0.87 | 0.02 | 0.00 | flat |
| qwen3 | vehicle_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.98 | 1.00 | +0.02 | 0.88 | +0.02 | 0.12 | +0.00 | 0.88 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.98 | 0.98 | +0.00 | 0.88 | +0.02 | 0.10 | -0.02 | 0.89 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | natural_qa | temperature | baseline | 10-12-14 | 0.98 | 0.98 | +0.00 | 0.88 | +0.00 | 0.10 | +0.00 | 0.89 | 0.02 | 0.00 | flat |
| qwen3 | vehicle_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.98 | 0.96 | -0.02 | 0.85 | -0.02 | 0.10 | +0.00 | 0.89 | 0.02 | 0.00 | flat |
| qwen3 | vehicle_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.98 | 0.98 | +0.00 | 0.88 | +0.00 | 0.10 | +0.00 | 0.89 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | definition | top_p | baseline | 10-12-14 | 0.77 | 0.77 | +0.00 | 0.65 | +0.00 | 0.12 | +0.00 | 0.84 | 0.00 | 0.10 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_parallel | 10-12-14 | 0.77 | 0.85 | +0.08 | 0.73 | +0.08 | 0.12 | +0.00 | 0.85 | 0.00 | 0.04 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_full | 10-12-14 | 0.77 | 0.85 | +0.08 | 0.73 | +0.08 | 0.12 | +0.00 | 0.85 | 0.00 | 0.04 | flat |
| qwen3 | vehicle_tool | definition | temperature | baseline | 10-12-14 | 0.79 | 0.79 | +0.00 | 0.62 | +0.00 | 0.17 | +0.00 | 0.79 | 0.04 | 0.02 | flat |
| qwen3 | vehicle_tool | definition | temperature | residual_parallel | 10-12-14 | 0.79 | 0.71 | -0.08 | 0.62 | +0.00 | 0.08 | -0.08 | 0.88 | 0.00 | 0.02 | flat |
| qwen3 | vehicle_tool | definition | temperature | residual_full | 10-12-14 | 0.79 | 0.83 | +0.04 | 0.69 | +0.06 | 0.15 | -0.02 | 0.82 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.75 | 0.75 | +0.00 | 0.27 | +0.00 | 0.48 | +0.00 | 0.36 | 0.04 | 0.02 | flat |
| qwen3 | vehicle_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.75 | 0.75 | +0.00 | 0.31 | +0.04 | 0.44 | -0.04 | 0.42 | 0.04 | 0.02 | flat |
| qwen3 | vehicle_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.75 | 0.79 | +0.04 | 0.38 | +0.10 | 0.42 | -0.06 | 0.47 | 0.04 | 0.00 | flat |
| qwen3 | vehicle_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.83 | 0.83 | +0.00 | 0.33 | +0.00 | 0.50 | +0.00 | 0.40 | 0.04 | 0.02 | flat |
| qwen3 | vehicle_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.83 | 0.75 | -0.08 | 0.31 | -0.02 | 0.44 | -0.06 | 0.42 | 0.08 | 0.02 | flat |
| qwen3 | vehicle_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.83 | 0.71 | -0.12 | 0.29 | -0.04 | 0.42 | -0.08 | 0.41 | 0.04 | 0.00 | negative |
| qwen3 | fruit_vegetable | natural_qa | top_p | baseline | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.88 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_vegetable | natural_qa | top_p | residual_parallel | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.88 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_vegetable | natural_qa | top_p | residual_full | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.88 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_vegetable | natural_qa | temperature | baseline | 10-12-14 | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| qwen3 | fruit_vegetable | natural_qa | temperature | residual_parallel | 10-12-14 | 0.85 | 0.88 | +0.02 | 0.88 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_vegetable | natural_qa | temperature | residual_full | 10-12-14 | 0.85 | 0.88 | +0.02 | 0.88 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| qwen3 | fruit_vegetable | definition | top_p | baseline | 10-12-14 | 0.69 | 0.69 | +0.00 | 0.62 | +0.00 | 0.06 | +0.00 | 0.91 | 0.00 | 0.02 | flat |
| qwen3 | fruit_vegetable | definition | top_p | residual_parallel | 10-12-14 | 0.69 | 0.67 | -0.02 | 0.62 | +0.00 | 0.04 | -0.02 | 0.94 | 0.00 | 0.06 | flat |
| qwen3 | fruit_vegetable | definition | top_p | residual_full | 10-12-14 | 0.69 | 0.67 | -0.02 | 0.58 | -0.04 | 0.08 | +0.02 | 0.88 | 0.00 | 0.04 | flat |
| qwen3 | fruit_vegetable | definition | temperature | baseline | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.54 | +0.00 | 0.12 | +0.00 | 0.81 | 0.00 | 0.06 | flat |
| qwen3 | fruit_vegetable | definition | temperature | residual_parallel | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.58 | +0.04 | 0.08 | -0.04 | 0.88 | 0.00 | 0.02 | flat |
| qwen3 | fruit_vegetable | definition | temperature | residual_full | 10-12-14 | 0.67 | 0.50 | -0.17 | 0.46 | -0.08 | 0.04 | -0.08 | 0.92 | 0.00 | 0.08 | negative |
| qwen3 | fruit_vegetable | sentence_completion | top_p | baseline | 10-12-14 | 0.17 | 0.17 | +0.00 | 0.17 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.10 | flat |
| qwen3 | fruit_vegetable | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.17 | 0.12 | -0.04 | 0.12 | -0.04 | 0.00 | +0.00 | 1.00 | 0.00 | 0.06 | flat |
| qwen3 | fruit_vegetable | sentence_completion | top_p | residual_full | 10-12-14 | 0.17 | 0.10 | -0.06 | 0.10 | -0.06 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| qwen3 | fruit_vegetable | sentence_completion | temperature | baseline | 10-12-14 | 0.17 | 0.17 | +0.00 | 0.15 | +0.00 | 0.02 | +0.00 | 0.88 | 0.00 | 0.06 | flat |
| qwen3 | fruit_vegetable | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.17 | 0.15 | -0.02 | 0.12 | -0.02 | 0.02 | +0.00 | 0.86 | 0.00 | 0.04 | flat |
| qwen3 | fruit_vegetable | sentence_completion | temperature | residual_full | 10-12-14 | 0.17 | 0.10 | -0.06 | 0.10 | -0.04 | 0.00 | -0.02 | 1.00 | 0.02 | 0.02 | flat |
| qwen3 | animal_tool | natural_qa | top_p | baseline | 10-12-14 | 0.58 | 0.58 | +0.00 | 0.21 | +0.00 | 0.38 | +0.00 | 0.36 | 0.00 | 0.08 | flat |
| qwen3 | animal_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.58 | 0.75 | +0.17 | 0.23 | +0.02 | 0.52 | +0.15 | 0.31 | 0.00 | 0.06 | weak_positive |
| qwen3 | animal_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.58 | 0.71 | +0.12 | 0.23 | +0.02 | 0.48 | +0.10 | 0.32 | 0.00 | 0.10 | weak_positive |
| qwen3 | animal_tool | natural_qa | temperature | baseline | 10-12-14 | 0.58 | 0.58 | +0.00 | 0.19 | +0.00 | 0.40 | +0.00 | 0.32 | 0.00 | 0.08 | flat |
| qwen3 | animal_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.58 | 0.65 | +0.06 | 0.21 | +0.02 | 0.44 | +0.04 | 0.32 | 0.00 | 0.08 | flat |
| qwen3 | animal_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.58 | 0.67 | +0.08 | 0.23 | +0.04 | 0.44 | +0.04 | 0.34 | 0.00 | 0.08 | flat |
| qwen3 | animal_tool | definition | top_p | baseline | 10-12-14 | 0.46 | 0.46 | +0.00 | 0.17 | +0.00 | 0.29 | +0.00 | 0.36 | 0.00 | 0.02 | flat |
| qwen3 | animal_tool | definition | top_p | residual_parallel | 10-12-14 | 0.46 | 0.54 | +0.08 | 0.21 | +0.04 | 0.33 | +0.04 | 0.38 | 0.00 | 0.06 | flat |
| qwen3 | animal_tool | definition | top_p | residual_full | 10-12-14 | 0.46 | 0.35 | -0.10 | 0.15 | -0.02 | 0.21 | -0.08 | 0.41 | 0.00 | 0.06 | negative |
| qwen3 | animal_tool | definition | temperature | baseline | 10-12-14 | 0.46 | 0.46 | +0.00 | 0.17 | +0.00 | 0.29 | +0.00 | 0.36 | 0.00 | 0.06 | flat |
| qwen3 | animal_tool | definition | temperature | residual_parallel | 10-12-14 | 0.46 | 0.50 | +0.04 | 0.17 | +0.00 | 0.33 | +0.04 | 0.33 | 0.00 | 0.02 | flat |
| qwen3 | animal_tool | definition | temperature | residual_full | 10-12-14 | 0.46 | 0.40 | -0.06 | 0.15 | -0.02 | 0.25 | -0.04 | 0.37 | 0.00 | 0.04 | flat |
| qwen3 | animal_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.19 | 0.19 | +0.00 | 0.17 | +0.00 | 0.02 | +0.00 | 0.89 | 0.00 | 0.00 | flat |
| qwen3 | animal_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.19 | 0.25 | +0.06 | 0.19 | +0.02 | 0.06 | +0.04 | 0.75 | 0.00 | 0.00 | flat |
| qwen3 | animal_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.19 | 0.12 | -0.06 | 0.10 | -0.06 | 0.02 | +0.00 | 0.83 | 0.00 | 0.04 | flat |
| qwen3 | animal_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.17 | 0.17 | +0.00 | 0.15 | +0.00 | 0.02 | +0.00 | 0.88 | 0.02 | 0.02 | flat |
| qwen3 | animal_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.17 | 0.25 | +0.08 | 0.21 | +0.06 | 0.04 | +0.02 | 0.83 | 0.00 | 0.00 | flat |
| qwen3 | animal_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.17 | 0.12 | -0.04 | 0.10 | -0.04 | 0.02 | +0.00 | 0.83 | 0.00 | 0.04 | flat |
| qwen3 | fruit_tool | natural_qa | top_p | baseline | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.88 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.90 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.88 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| qwen3 | fruit_tool | natural_qa | temperature | baseline | 10-12-14 | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| qwen3 | fruit_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.85 | 0.85 | +0.00 | 0.85 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.85 | 0.90 | +0.04 | 0.90 | +0.04 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | fruit_tool | definition | top_p | baseline | 10-12-14 | 0.69 | 0.69 | +0.00 | 0.62 | +0.00 | 0.06 | +0.00 | 0.91 | 0.00 | 0.02 | flat |
| qwen3 | fruit_tool | definition | top_p | residual_parallel | 10-12-14 | 0.69 | 0.77 | +0.08 | 0.69 | +0.06 | 0.08 | +0.02 | 0.89 | 0.00 | 0.02 | flat |
| qwen3 | fruit_tool | definition | top_p | residual_full | 10-12-14 | 0.69 | 0.69 | +0.00 | 0.67 | +0.04 | 0.02 | -0.04 | 0.97 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | definition | temperature | baseline | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.54 | +0.00 | 0.12 | +0.00 | 0.81 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | definition | temperature | residual_parallel | 10-12-14 | 0.67 | 0.62 | -0.04 | 0.54 | +0.00 | 0.08 | -0.04 | 0.87 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | definition | temperature | residual_full | 10-12-14 | 0.67 | 0.65 | -0.02 | 0.56 | +0.02 | 0.08 | -0.04 | 0.87 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.17 | 0.17 | +0.00 | 0.17 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.10 | flat |
| qwen3 | fruit_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.17 | 0.08 | -0.08 | 0.06 | -0.10 | 0.02 | +0.02 | 0.75 | 0.00 | 0.08 | flat |
| qwen3 | fruit_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.17 | 0.08 | -0.08 | 0.08 | -0.08 | 0.00 | +0.00 | 1.00 | 0.02 | 0.02 | flat |
| qwen3 | fruit_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.17 | 0.17 | +0.00 | 0.15 | +0.00 | 0.02 | +0.00 | 0.88 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.17 | 0.17 | -0.00 | 0.17 | +0.02 | 0.00 | -0.02 | 1.00 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.17 | 0.19 | +0.02 | 0.19 | +0.04 | 0.00 | -0.02 | 1.00 | 0.00 | 0.02 | flat |

## glm4

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'24-26-28': [24, 26, 28]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | vehicle_tool | natural_qa | top_p | baseline | 24-26-28 | 0.71 | 0.71 | +0.00 | 0.58 | +0.00 | 0.12 | +0.00 | 0.82 | 0.04 | 0.12 | flat |
| glm4 | vehicle_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.71 | 0.96 | +0.25 | 0.92 | +0.33 | 0.04 | -0.08 | 0.96 | 0.00 | 0.04 | label_driven_positive |
| glm4 | vehicle_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.71 | 0.75 | +0.04 | 0.71 | +0.12 | 0.04 | -0.08 | 0.94 | 0.00 | 0.12 | flat |
| glm4 | vehicle_tool | natural_qa | temperature | baseline | 24-26-28 | 0.67 | 0.67 | +0.00 | 0.60 | +0.00 | 0.06 | +0.00 | 0.91 | 0.06 | 0.12 | flat |
| glm4 | vehicle_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.67 | 0.96 | +0.29 | 0.94 | +0.33 | 0.02 | -0.04 | 0.98 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.67 | 0.77 | +0.10 | 0.69 | +0.08 | 0.08 | +0.02 | 0.89 | 0.00 | 0.08 | weak_positive |
| glm4 | vehicle_tool | definition | top_p | baseline | 24-26-28 | 0.54 | 0.54 | +0.00 | 0.46 | +0.00 | 0.08 | +0.00 | 0.85 | 0.04 | 0.06 | flat |
| glm4 | vehicle_tool | definition | top_p | residual_parallel | 24-26-28 | 0.54 | 0.83 | +0.29 | 0.79 | +0.33 | 0.04 | -0.04 | 0.95 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | definition | top_p | residual_full | 24-26-28 | 0.54 | 0.75 | +0.21 | 0.71 | +0.25 | 0.04 | -0.04 | 0.94 | 0.00 | 0.06 | label_driven_positive |
| glm4 | vehicle_tool | definition | temperature | baseline | 24-26-28 | 0.48 | 0.48 | +0.00 | 0.44 | +0.00 | 0.04 | +0.00 | 0.91 | 0.04 | 0.06 | flat |
| glm4 | vehicle_tool | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.81 | +0.33 | 0.79 | +0.35 | 0.02 | -0.02 | 0.97 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | definition | temperature | residual_full | 24-26-28 | 0.48 | 0.67 | +0.19 | 0.60 | +0.17 | 0.06 | +0.02 | 0.91 | 0.00 | 0.04 | weak_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.35 | 0.35 | +0.00 | 0.15 | +0.00 | 0.21 | +0.00 | 0.41 | 0.15 | 0.04 | flat |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.35 | 0.85 | +0.50 | 0.79 | +0.65 | 0.06 | -0.15 | 0.93 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.35 | 0.69 | +0.33 | 0.27 | +0.12 | 0.42 | +0.21 | 0.39 | 0.00 | 0.02 | semantic_paraphrase_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.12 | +0.00 | 0.19 | +0.00 | 0.40 | 0.23 | 0.02 | flat |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.31 | 0.83 | +0.52 | 0.75 | +0.62 | 0.08 | -0.10 | 0.90 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.31 | 0.75 | +0.44 | 0.31 | +0.19 | 0.44 | +0.25 | 0.42 | 0.00 | 0.00 | semantic_paraphrase_positive |
| glm4 | fruit_vegetable | natural_qa | top_p | baseline | 24-26-28 | 0.50 | 0.50 | +0.00 | 0.38 | +0.00 | 0.12 | +0.00 | 0.75 | 0.00 | 0.12 | flat |
| glm4 | fruit_vegetable | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.90 | +0.40 | 0.90 | +0.52 | 0.00 | -0.12 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | natural_qa | top_p | residual_full | 24-26-28 | 0.50 | 0.56 | +0.06 | 0.54 | +0.17 | 0.02 | -0.10 | 0.96 | 0.00 | 0.06 | flat |
| glm4 | fruit_vegetable | natural_qa | temperature | baseline | 24-26-28 | 0.46 | 0.46 | +0.00 | 0.31 | +0.00 | 0.15 | +0.00 | 0.68 | 0.00 | 0.08 | flat |
| glm4 | fruit_vegetable | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.88 | +0.42 | 0.88 | +0.56 | 0.00 | -0.15 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | natural_qa | temperature | residual_full | 24-26-28 | 0.46 | 0.50 | +0.04 | 0.46 | +0.15 | 0.04 | -0.10 | 0.92 | 0.00 | 0.04 | flat |
| glm4 | fruit_vegetable | definition | top_p | baseline | 24-26-28 | 0.48 | 0.48 | +0.00 | 0.44 | +0.00 | 0.04 | +0.00 | 0.91 | 0.00 | 0.02 | flat |
| glm4 | fruit_vegetable | definition | top_p | residual_parallel | 24-26-28 | 0.48 | 0.85 | +0.37 | 0.85 | +0.42 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | definition | top_p | residual_full | 24-26-28 | 0.48 | 0.50 | +0.02 | 0.44 | +0.00 | 0.06 | +0.02 | 0.88 | 0.00 | 0.02 | flat |
| glm4 | fruit_vegetable | definition | temperature | baseline | 24-26-28 | 0.48 | 0.48 | +0.00 | 0.40 | +0.00 | 0.08 | +0.00 | 0.83 | 0.00 | 0.04 | flat |
| glm4 | fruit_vegetable | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.83 | +0.35 | 0.83 | +0.44 | 0.00 | -0.08 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | definition | temperature | residual_full | 24-26-28 | 0.48 | 0.44 | -0.04 | 0.38 | -0.02 | 0.06 | -0.02 | 0.86 | 0.00 | 0.00 | flat |
| glm4 | fruit_vegetable | sentence_completion | top_p | baseline | 24-26-28 | 0.08 | 0.08 | +0.00 | 0.08 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.08 | 0.90 | +0.81 | 0.90 | +0.81 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_full | 24-26-28 | 0.08 | 0.46 | +0.38 | 0.46 | +0.38 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | baseline | 24-26-28 | 0.21 | 0.21 | +0.00 | 0.17 | +0.00 | 0.04 | +0.00 | 0.80 | 0.00 | 0.06 | flat |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.21 | 0.88 | +0.67 | 0.88 | +0.71 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_full | 24-26-28 | 0.21 | 0.44 | +0.23 | 0.44 | +0.27 | 0.00 | -0.04 | 1.00 | 0.00 | 0.02 | label_driven_positive |
| glm4 | animal_tool | natural_qa | top_p | baseline | 24-26-28 | 0.50 | 0.50 | +0.00 | 0.40 | +0.00 | 0.10 | +0.00 | 0.79 | 0.00 | 0.10 | flat |
| glm4 | animal_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.98 | +0.48 | 0.98 | +0.58 | 0.00 | -0.10 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.50 | 0.35 | -0.15 | 0.27 | -0.12 | 0.08 | -0.02 | 0.76 | 0.00 | 0.12 | negative |
| glm4 | animal_tool | natural_qa | temperature | baseline | 24-26-28 | 0.46 | 0.46 | +0.00 | 0.42 | +0.00 | 0.04 | +0.00 | 0.91 | 0.00 | 0.27 | flat |
| glm4 | animal_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.98 | +0.52 | 0.98 | +0.56 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.46 | 0.29 | -0.17 | 0.21 | -0.21 | 0.08 | +0.04 | 0.71 | 0.00 | 0.08 | negative |
| glm4 | animal_tool | definition | top_p | baseline | 24-26-28 | 0.58 | 0.58 | +0.00 | 0.46 | +0.00 | 0.12 | +0.00 | 0.79 | 0.00 | 0.00 | flat |
| glm4 | animal_tool | definition | top_p | residual_parallel | 24-26-28 | 0.58 | 0.88 | +0.29 | 0.88 | +0.42 | 0.00 | -0.12 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | definition | top_p | residual_full | 24-26-28 | 0.58 | 0.35 | -0.23 | 0.27 | -0.19 | 0.08 | -0.04 | 0.76 | 0.00 | 0.00 | negative |
| glm4 | animal_tool | definition | temperature | baseline | 24-26-28 | 0.54 | 0.54 | +0.00 | 0.46 | +0.00 | 0.08 | +0.00 | 0.85 | 0.00 | 0.00 | flat |
| glm4 | animal_tool | definition | temperature | residual_parallel | 24-26-28 | 0.54 | 0.96 | +0.42 | 0.94 | +0.48 | 0.02 | -0.06 | 0.98 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | definition | temperature | residual_full | 24-26-28 | 0.54 | 0.38 | -0.17 | 0.29 | -0.17 | 0.08 | +0.00 | 0.78 | 0.00 | 0.00 | negative |
| glm4 | animal_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.04 | 0.04 | +0.00 | 0.04 | +0.00 | 0.00 | +0.00 | 1.00 | 0.02 | 0.04 | flat |
| glm4 | animal_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.04 | 0.98 | +0.94 | 0.98 | +0.94 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.04 | 0.25 | +0.21 | 0.25 | +0.21 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.04 | 0.04 | +0.00 | 0.04 | +0.00 | 0.00 | +0.00 | 1.00 | 0.02 | 0.02 | flat |
| glm4 | animal_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.04 | 0.90 | +0.85 | 0.90 | +0.85 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.04 | 0.19 | +0.15 | 0.19 | +0.15 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | weak_positive |
| glm4 | fruit_tool | natural_qa | top_p | baseline | 24-26-28 | 0.50 | 0.50 | +0.00 | 0.38 | +0.00 | 0.12 | +0.00 | 0.75 | 0.00 | 0.12 | flat |
| glm4 | fruit_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.85 | +0.35 | 0.83 | +0.46 | 0.02 | -0.10 | 0.98 | 0.00 | 0.02 | label_driven_positive |
| glm4 | fruit_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.50 | 0.48 | -0.02 | 0.33 | -0.04 | 0.15 | +0.02 | 0.70 | 0.00 | 0.04 | flat |
| glm4 | fruit_tool | natural_qa | temperature | baseline | 24-26-28 | 0.46 | 0.46 | +0.00 | 0.31 | +0.00 | 0.15 | +0.00 | 0.68 | 0.00 | 0.08 | flat |
| glm4 | fruit_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.83 | +0.38 | 0.83 | +0.52 | 0.00 | -0.15 | 1.00 | 0.00 | 0.04 | label_driven_positive |
| glm4 | fruit_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.46 | 0.50 | +0.04 | 0.38 | +0.06 | 0.12 | -0.02 | 0.75 | 0.00 | 0.02 | flat |
| glm4 | fruit_tool | definition | top_p | baseline | 24-26-28 | 0.48 | 0.48 | +0.00 | 0.44 | +0.00 | 0.04 | +0.00 | 0.91 | 0.00 | 0.02 | flat |
| glm4 | fruit_tool | definition | top_p | residual_parallel | 24-26-28 | 0.48 | 0.90 | +0.42 | 0.90 | +0.46 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | definition | top_p | residual_full | 24-26-28 | 0.48 | 0.58 | +0.10 | 0.54 | +0.10 | 0.04 | +0.00 | 0.93 | 0.00 | 0.00 | weak_positive |
| glm4 | fruit_tool | definition | temperature | baseline | 24-26-28 | 0.48 | 0.48 | +0.00 | 0.40 | +0.00 | 0.08 | +0.00 | 0.83 | 0.00 | 0.04 | flat |
| glm4 | fruit_tool | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.81 | +0.33 | 0.81 | +0.42 | 0.00 | -0.08 | 1.00 | 0.00 | 0.02 | label_driven_positive |
| glm4 | fruit_tool | definition | temperature | residual_full | 24-26-28 | 0.48 | 0.54 | +0.06 | 0.44 | +0.04 | 0.10 | +0.02 | 0.81 | 0.00 | 0.00 | flat |
| glm4 | fruit_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.08 | 0.08 | +0.00 | 0.08 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| glm4 | fruit_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.08 | 0.92 | +0.83 | 0.92 | +0.83 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.08 | 0.52 | +0.44 | 0.50 | +0.42 | 0.02 | +0.02 | 0.96 | 0.00 | 0.04 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.21 | 0.21 | +0.00 | 0.17 | +0.00 | 0.04 | +0.00 | 0.80 | 0.02 | 0.06 | flat |
| glm4 | fruit_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.21 | 0.88 | +0.67 | 0.88 | +0.71 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.21 | 0.56 | +0.35 | 0.54 | +0.38 | 0.02 | -0.02 | 0.96 | 0.00 | 0.02 | label_driven_positive |

## deepseek7b

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'16-18-20': [16, 18, 20]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | vehicle_tool | natural_qa | top_p | baseline | 16-18-20 | 0.77 | 0.77 | +0.00 | 0.58 | +0.00 | 0.19 | +0.00 | 0.76 | 0.02 | 0.12 | flat |
| deepseek7b | vehicle_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.77 | 0.81 | +0.04 | 0.65 | +0.06 | 0.17 | -0.02 | 0.79 | 0.04 | 0.10 | flat |
| deepseek7b | vehicle_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.77 | 0.71 | -0.06 | 0.54 | -0.04 | 0.17 | -0.02 | 0.76 | 0.02 | 0.23 | flat |
| deepseek7b | vehicle_tool | natural_qa | temperature | baseline | 16-18-20 | 0.71 | 0.71 | +0.00 | 0.50 | +0.00 | 0.21 | +0.00 | 0.71 | 0.00 | 0.21 | flat |
| deepseek7b | vehicle_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.71 | 0.71 | +0.00 | 0.58 | +0.08 | 0.12 | -0.08 | 0.82 | 0.00 | 0.17 | flat |
| deepseek7b | vehicle_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.71 | 0.65 | -0.06 | 0.50 | +0.00 | 0.15 | -0.06 | 0.77 | 0.02 | 0.21 | flat |
| deepseek7b | vehicle_tool | definition | top_p | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.12 | +0.00 | 0.02 | +0.00 | 0.86 | 0.00 | 0.12 | flat |
| deepseek7b | vehicle_tool | definition | top_p | residual_parallel | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.12 | +0.00 | 0.02 | +0.00 | 0.86 | 0.00 | 0.04 | flat |
| deepseek7b | vehicle_tool | definition | top_p | residual_full | 16-18-20 | 0.15 | 0.12 | -0.02 | 0.10 | -0.02 | 0.02 | +0.00 | 0.83 | 0.00 | 0.08 | flat |
| deepseek7b | vehicle_tool | definition | temperature | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.15 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| deepseek7b | vehicle_tool | definition | temperature | residual_parallel | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.12 | -0.02 | 0.02 | +0.02 | 0.86 | 0.02 | 0.02 | flat |
| deepseek7b | vehicle_tool | definition | temperature | residual_full | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.12 | -0.02 | 0.02 | +0.02 | 0.86 | 0.00 | 0.08 | flat |
| deepseek7b | vehicle_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.19 | 0.19 | +0.00 | 0.08 | +0.00 | 0.10 | +0.00 | 0.44 | 0.00 | 0.10 | flat |
| deepseek7b | vehicle_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.19 | 0.19 | +0.00 | 0.08 | +0.00 | 0.10 | +0.00 | 0.44 | 0.00 | 0.10 | flat |
| deepseek7b | vehicle_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.19 | 0.10 | -0.08 | 0.04 | -0.04 | 0.06 | -0.04 | 0.40 | 0.00 | 0.10 | flat |
| deepseek7b | vehicle_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.04 | +0.00 | 0.06 | +0.00 | 0.40 | 0.00 | 0.02 | flat |
| deepseek7b | vehicle_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.10 | 0.06 | -0.04 | 0.06 | +0.02 | 0.00 | -0.06 | 1.00 | 0.00 | 0.08 | flat |
| deepseek7b | vehicle_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.08 | +0.04 | 0.06 | +0.00 | 0.57 | 0.00 | 0.06 | flat |
| deepseek7b | fruit_vegetable | natural_qa | top_p | baseline | 16-18-20 | 0.71 | 0.71 | +0.00 | 0.71 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.06 | flat |
| deepseek7b | fruit_vegetable | natural_qa | top_p | residual_parallel | 16-18-20 | 0.71 | 0.67 | -0.04 | 0.67 | -0.04 | 0.00 | +0.00 | 1.00 | 0.00 | 0.10 | flat |
| deepseek7b | fruit_vegetable | natural_qa | top_p | residual_full | 16-18-20 | 0.71 | 0.73 | +0.02 | 0.73 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | natural_qa | temperature | baseline | 16-18-20 | 0.67 | 0.67 | +0.00 | 0.67 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.10 | flat |
| deepseek7b | fruit_vegetable | natural_qa | temperature | residual_parallel | 16-18-20 | 0.67 | 0.67 | +0.00 | 0.65 | -0.02 | 0.02 | +0.02 | 0.97 | 0.00 | 0.06 | flat |
| deepseek7b | fruit_vegetable | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.77 | +0.10 | 0.75 | +0.08 | 0.02 | +0.02 | 0.97 | 0.00 | 0.02 | weak_positive |
| deepseek7b | fruit_vegetable | definition | top_p | baseline | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.33 | +0.00 | 0.02 | +0.00 | 0.94 | 0.00 | 0.04 | flat |
| deepseek7b | fruit_vegetable | definition | top_p | residual_parallel | 16-18-20 | 0.35 | 0.33 | -0.02 | 0.31 | -0.02 | 0.02 | +0.00 | 0.94 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | definition | top_p | residual_full | 16-18-20 | 0.35 | 0.38 | +0.02 | 0.35 | +0.02 | 0.02 | +0.00 | 0.94 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | definition | temperature | baseline | 16-18-20 | 0.33 | 0.33 | +0.00 | 0.33 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | definition | temperature | residual_parallel | 16-18-20 | 0.33 | 0.29 | -0.04 | 0.29 | -0.04 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| deepseek7b | fruit_vegetable | definition | temperature | residual_full | 16-18-20 | 0.33 | 0.35 | +0.02 | 0.33 | +0.00 | 0.02 | +0.02 | 0.94 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | top_p | baseline | 16-18-20 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.00 | 0.02 | +0.02 | 0.00 | +0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.04 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | temperature | baseline | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.00 | +0.00 | 0.02 | +0.00 | 0.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.02 | +0.02 | 0.00 | -0.02 | 1.00 | 0.02 | 0.04 | flat |
| deepseek7b | fruit_vegetable | sentence_completion | temperature | residual_full | 16-18-20 | 0.02 | 0.00 | -0.02 | 0.00 | +0.00 | 0.00 | -0.02 | 0.00 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | natural_qa | top_p | baseline | 16-18-20 | 0.83 | 0.83 | +0.00 | 0.60 | +0.00 | 0.23 | +0.00 | 0.72 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.83 | 0.81 | -0.02 | 0.60 | +0.00 | 0.21 | -0.02 | 0.74 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.83 | 0.79 | -0.04 | 0.54 | -0.06 | 0.25 | +0.02 | 0.68 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | natural_qa | temperature | baseline | 16-18-20 | 0.77 | 0.77 | +0.00 | 0.56 | +0.00 | 0.21 | +0.00 | 0.73 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.77 | 0.81 | +0.04 | 0.65 | +0.08 | 0.17 | -0.04 | 0.79 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.77 | 0.77 | +0.00 | 0.52 | -0.04 | 0.25 | +0.04 | 0.68 | 0.00 | 0.08 | flat |
| deepseek7b | animal_tool | definition | top_p | baseline | 16-18-20 | 0.27 | 0.27 | +0.00 | 0.21 | +0.00 | 0.06 | +0.00 | 0.77 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | top_p | residual_parallel | 16-18-20 | 0.27 | 0.27 | -0.00 | 0.25 | +0.04 | 0.02 | -0.04 | 0.92 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | top_p | residual_full | 16-18-20 | 0.27 | 0.31 | +0.04 | 0.25 | +0.04 | 0.06 | +0.00 | 0.80 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | temperature | baseline | 16-18-20 | 0.29 | 0.29 | +0.00 | 0.19 | +0.00 | 0.10 | +0.00 | 0.64 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | temperature | residual_parallel | 16-18-20 | 0.29 | 0.23 | -0.06 | 0.21 | +0.02 | 0.02 | -0.08 | 0.91 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | temperature | residual_full | 16-18-20 | 0.29 | 0.29 | +0.00 | 0.19 | +0.00 | 0.10 | +0.00 | 0.64 | 0.00 | 0.02 | flat |
| deepseek7b | animal_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.02 | flat |
| deepseek7b | animal_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.00 | 0.02 | +0.02 | 0.02 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| deepseek7b | animal_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.02 | 0.04 | +0.02 | 0.04 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | natural_qa | top_p | baseline | 16-18-20 | 0.71 | 0.71 | +0.00 | 0.71 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.06 | flat |
| deepseek7b | fruit_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.71 | 0.71 | +0.00 | 0.71 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.06 | flat |
| deepseek7b | fruit_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.71 | 0.75 | +0.04 | 0.73 | +0.02 | 0.02 | +0.02 | 0.97 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | natural_qa | temperature | baseline | 16-18-20 | 0.67 | 0.67 | +0.00 | 0.67 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.10 | flat |
| deepseek7b | fruit_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.67 | 0.75 | +0.08 | 0.75 | +0.08 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| deepseek7b | fruit_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.69 | +0.02 | 0.69 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.08 | flat |
| deepseek7b | fruit_tool | definition | top_p | baseline | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.33 | +0.00 | 0.02 | +0.00 | 0.94 | 0.00 | 0.04 | flat |
| deepseek7b | fruit_tool | definition | top_p | residual_parallel | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.35 | +0.02 | 0.00 | -0.02 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | definition | top_p | residual_full | 16-18-20 | 0.35 | 0.42 | +0.06 | 0.42 | +0.08 | 0.00 | -0.02 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | definition | temperature | baseline | 16-18-20 | 0.33 | 0.33 | +0.00 | 0.33 | +0.00 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | definition | temperature | residual_parallel | 16-18-20 | 0.33 | 0.35 | +0.02 | 0.35 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_tool | definition | temperature | residual_full | 16-18-20 | 0.33 | 0.23 | -0.10 | 0.23 | -0.10 | 0.00 | +0.00 | 1.00 | 0.00 | 0.02 | negative |
| deepseek7b | fruit_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.00 | 0.02 | +0.02 | 0.02 | +0.02 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.04 | +0.04 | 0.02 | +0.02 | 0.02 | +0.02 | 0.50 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.00 | +0.00 | 0.02 | +0.00 | 0.00 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.02 | 0.04 | +0.02 | 0.02 | +0.02 | 0.02 | +0.00 | 0.50 | 0.00 | 0.00 | flat |
| deepseek7b | fruit_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.02 | 0.02 | +0.00 | 0.00 | +0.00 | 0.02 | +0.00 | 0.00 | 0.00 | 0.04 | flat |

## Best Semantic Gains

| model | pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | animal_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.04 | 0.98 | +0.94 | 0.98 | +0.94 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | animal_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.04 | 0.90 | +0.85 | 0.90 | +0.85 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.08 | 0.92 | +0.83 | 0.92 | +0.83 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.08 | 0.90 | +0.81 | 0.90 | +0.81 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.21 | 0.88 | +0.67 | 0.88 | +0.71 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.21 | 0.88 | +0.67 | 0.88 | +0.71 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.31 | 0.83 | +0.52 | 0.75 | +0.62 | 0.08 | -0.10 | 0.90 | 0.00 | 0.02 | label_driven_positive |
| glm4 | animal_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.98 | +0.52 | 0.98 | +0.56 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.35 | 0.85 | +0.50 | 0.79 | +0.65 | 0.06 | -0.15 | 0.93 | 0.00 | 0.02 | label_driven_positive |
| glm4 | animal_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.98 | +0.48 | 0.98 | +0.58 | 0.00 | -0.10 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.08 | 0.52 | +0.44 | 0.50 | +0.42 | 0.02 | +0.02 | 0.96 | 0.00 | 0.04 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.31 | 0.75 | +0.44 | 0.31 | +0.19 | 0.44 | +0.25 | 0.42 | 0.00 | 0.00 | semantic_paraphrase_positive |
| glm4 | animal_tool | definition | temperature | residual_parallel | 24-26-28 | 0.54 | 0.96 | +0.42 | 0.94 | +0.48 | 0.02 | -0.06 | 0.98 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | definition | top_p | residual_parallel | 24-26-28 | 0.48 | 0.90 | +0.42 | 0.90 | +0.46 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.88 | +0.42 | 0.88 | +0.56 | 0.00 | -0.15 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.90 | +0.40 | 0.90 | +0.52 | 0.00 | -0.12 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_full | 24-26-28 | 0.08 | 0.46 | +0.38 | 0.46 | +0.38 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.46 | 0.83 | +0.38 | 0.83 | +0.52 | 0.00 | -0.15 | 1.00 | 0.00 | 0.04 | label_driven_positive |
| glm4 | fruit_vegetable | definition | top_p | residual_parallel | 24-26-28 | 0.48 | 0.85 | +0.37 | 0.85 | +0.42 | 0.00 | -0.04 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_vegetable | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.83 | +0.35 | 0.83 | +0.44 | 0.00 | -0.08 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | fruit_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.50 | 0.85 | +0.35 | 0.83 | +0.46 | 0.02 | -0.10 | 0.98 | 0.00 | 0.02 | label_driven_positive |
| glm4 | fruit_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.21 | 0.56 | +0.35 | 0.54 | +0.38 | 0.02 | -0.02 | 0.96 | 0.00 | 0.02 | label_driven_positive |
| glm4 | fruit_tool | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.81 | +0.33 | 0.81 | +0.42 | 0.00 | -0.08 | 1.00 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.35 | 0.69 | +0.33 | 0.27 | +0.12 | 0.42 | +0.21 | 0.39 | 0.00 | 0.02 | semantic_paraphrase_positive |
| glm4 | vehicle_tool | definition | temperature | residual_parallel | 24-26-28 | 0.48 | 0.81 | +0.33 | 0.79 | +0.35 | 0.02 | -0.02 | 0.97 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.67 | 0.96 | +0.29 | 0.94 | +0.33 | 0.02 | -0.04 | 0.98 | 0.00 | 0.02 | label_driven_positive |
| glm4 | animal_tool | definition | top_p | residual_parallel | 24-26-28 | 0.58 | 0.88 | +0.29 | 0.88 | +0.42 | 0.00 | -0.12 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | definition | top_p | residual_parallel | 24-26-28 | 0.54 | 0.83 | +0.29 | 0.79 | +0.33 | 0.04 | -0.04 | 0.95 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.71 | 0.96 | +0.25 | 0.92 | +0.33 | 0.04 | -0.08 | 0.96 | 0.00 | 0.04 | label_driven_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_full | 24-26-28 | 0.21 | 0.44 | +0.23 | 0.44 | +0.27 | 0.00 | -0.04 | 1.00 | 0.00 | 0.02 | label_driven_positive |
| glm4 | vehicle_tool | definition | top_p | residual_full | 24-26-28 | 0.54 | 0.75 | +0.21 | 0.71 | +0.25 | 0.04 | -0.04 | 0.94 | 0.00 | 0.06 | label_driven_positive |
| glm4 | animal_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.04 | 0.25 | +0.21 | 0.25 | +0.21 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | label_driven_positive |
| glm4 | vehicle_tool | definition | temperature | residual_full | 24-26-28 | 0.48 | 0.67 | +0.19 | 0.60 | +0.17 | 0.06 | +0.02 | 0.91 | 0.00 | 0.04 | weak_positive |
| qwen3 | animal_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.58 | 0.75 | +0.17 | 0.23 | +0.02 | 0.52 | +0.15 | 0.31 | 0.00 | 0.06 | weak_positive |
| glm4 | animal_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.04 | 0.19 | +0.15 | 0.19 | +0.15 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | weak_positive |
| qwen3 | animal_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.58 | 0.71 | +0.12 | 0.23 | +0.02 | 0.48 | +0.10 | 0.32 | 0.00 | 0.10 | weak_positive |
| deepseek7b | fruit_vegetable | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.77 | +0.10 | 0.75 | +0.08 | 0.02 | +0.02 | 0.97 | 0.00 | 0.02 | weak_positive |
| glm4 | vehicle_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.67 | 0.77 | +0.10 | 0.69 | +0.08 | 0.08 | +0.02 | 0.89 | 0.00 | 0.08 | weak_positive |
| glm4 | fruit_tool | definition | top_p | residual_full | 24-26-28 | 0.48 | 0.58 | +0.10 | 0.54 | +0.10 | 0.04 | +0.00 | 0.93 | 0.00 | 0.00 | weak_positive |
| qwen3 | animal_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.58 | 0.67 | +0.08 | 0.23 | +0.04 | 0.44 | +0.04 | 0.34 | 0.00 | 0.08 | flat |
| qwen3 | fruit_tool | definition | top_p | residual_parallel | 10-12-14 | 0.69 | 0.77 | +0.08 | 0.69 | +0.06 | 0.08 | +0.02 | 0.89 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.67 | 0.75 | +0.08 | 0.75 | +0.08 | 0.00 | +0.00 | 1.00 | 0.00 | 0.04 | flat |
| qwen3 | animal_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.17 | 0.25 | +0.08 | 0.21 | +0.06 | 0.04 | +0.02 | 0.83 | 0.00 | 0.00 | flat |
| qwen3 | animal_tool | definition | top_p | residual_parallel | 10-12-14 | 0.46 | 0.54 | +0.08 | 0.21 | +0.04 | 0.33 | +0.04 | 0.38 | 0.00 | 0.06 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_parallel | 10-12-14 | 0.77 | 0.85 | +0.08 | 0.73 | +0.08 | 0.12 | +0.00 | 0.85 | 0.00 | 0.04 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_full | 10-12-14 | 0.77 | 0.85 | +0.08 | 0.73 | +0.08 | 0.12 | +0.00 | 0.85 | 0.00 | 0.04 | flat |
| qwen3 | animal_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.58 | 0.65 | +0.06 | 0.21 | +0.02 | 0.44 | +0.04 | 0.32 | 0.00 | 0.08 | flat |
| deepseek7b | fruit_tool | definition | top_p | residual_full | 16-18-20 | 0.35 | 0.42 | +0.06 | 0.42 | +0.08 | 0.00 | -0.02 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | animal_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.19 | 0.25 | +0.06 | 0.19 | +0.02 | 0.06 | +0.04 | 0.75 | 0.00 | 0.00 | flat |
| glm4 | fruit_tool | definition | temperature | residual_full | 24-26-28 | 0.48 | 0.54 | +0.06 | 0.44 | +0.04 | 0.10 | +0.02 | 0.81 | 0.00 | 0.00 | flat |
| glm4 | fruit_vegetable | natural_qa | top_p | residual_full | 24-26-28 | 0.50 | 0.56 | +0.06 | 0.54 | +0.17 | 0.02 | -0.10 | 0.96 | 0.00 | 0.06 | flat |
| qwen3 | fruit_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.85 | 0.90 | +0.04 | 0.90 | +0.04 | 0.00 | +0.00 | 1.00 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | definition | temperature | residual_full | 10-12-14 | 0.79 | 0.83 | +0.04 | 0.69 | +0.06 | 0.15 | -0.02 | 0.82 | 0.00 | 0.00 | flat |
| qwen3 | vehicle_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.75 | 0.79 | +0.04 | 0.38 | +0.10 | 0.42 | -0.06 | 0.47 | 0.04 | 0.00 | flat |
| deepseek7b | fruit_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.04 | +0.04 | 0.02 | +0.02 | 0.02 | +0.02 | 0.50 | 0.00 | 0.02 | flat |
| deepseek7b | vehicle_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.08 | +0.04 | 0.06 | +0.00 | 0.57 | 0.00 | 0.06 | flat |
| qwen3 | animal_tool | definition | temperature | residual_parallel | 10-12-14 | 0.46 | 0.50 | +0.04 | 0.17 | +0.00 | 0.33 | +0.04 | 0.33 | 0.00 | 0.02 | flat |
| deepseek7b | fruit_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.71 | 0.75 | +0.04 | 0.73 | +0.02 | 0.02 | +0.02 | 0.97 | 0.00 | 0.00 | flat |
| deepseek7b | animal_tool | definition | top_p | residual_full | 16-18-20 | 0.27 | 0.31 | +0.04 | 0.25 | +0.04 | 0.06 | +0.00 | 0.80 | 0.00 | 0.00 | flat |
| glm4 | fruit_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.46 | 0.50 | +0.04 | 0.38 | +0.06 | 0.12 | -0.02 | 0.75 | 0.00 | 0.02 | flat |

## Pair Max Semantic Gain

| model | pair | max semantic gain | exact gain | non-exact gain | row | class |
|---|---|---:|---:|---:|---|---|
| deepseek7b | animal_tool | +0.04 | +0.08 | -0.04 | natural_qa temperature residual_parallel 16-18-20 | flat |
| deepseek7b | fruit_tool | +0.08 | +0.08 | +0.00 | natural_qa temperature residual_parallel 16-18-20 | flat |
| deepseek7b | fruit_vegetable | +0.10 | +0.08 | +0.02 | natural_qa temperature residual_full 16-18-20 | weak_positive |
| deepseek7b | vehicle_tool | +0.04 | +0.04 | +0.00 | sentence_completion temperature residual_full 16-18-20 | flat |
| glm4 | animal_tool | +0.94 | +0.94 | +0.00 | sentence_completion top_p residual_parallel 24-26-28 | label_driven_positive |
| glm4 | fruit_tool | +0.83 | +0.83 | +0.00 | sentence_completion top_p residual_parallel 24-26-28 | label_driven_positive |
| glm4 | fruit_vegetable | +0.81 | +0.81 | +0.00 | sentence_completion top_p residual_parallel 24-26-28 | label_driven_positive |
| glm4 | vehicle_tool | +0.52 | +0.62 | -0.10 | sentence_completion temperature residual_parallel 24-26-28 | label_driven_positive |
| qwen3 | animal_tool | +0.17 | +0.02 | +0.15 | natural_qa top_p residual_parallel 10-12-14 | weak_positive |
| qwen3 | fruit_tool | +0.08 | +0.06 | +0.02 | definition top_p residual_parallel 10-12-14 | flat |
| qwen3 | fruit_vegetable | +0.02 | +0.02 | +0.00 | natural_qa temperature residual_parallel 10-12-14 | flat |
| qwen3 | vehicle_tool | +0.08 | +0.08 | +0.00 | definition top_p residual_parallel 10-12-14 | flat |

## Representative Samples

| model | window | pair | scaffold | mode | condition | seed | quality | prompt | suffix |
|---|---|---|---|---|---|---:|---|---|---|
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | baseline | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a vehicle that uses jet propulsion to move through the air or |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | baseline | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically made of wood, that is used |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a vehicle that propels itself by expelling mass, typically |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically used for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | residual_full | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a vehicle that uses the propulsion of exhaust gases to move through |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | top_p | residual_full | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically made of wood or other materials, |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | baseline | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a type of vehicle or device that uses a propulsion system to |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | baseline | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically made of wood, that is used |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a type of vehicle designed to travel through the air or space |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically used for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | residual_full | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a type of vehicle or machine that can be used to travel |
| qwen3 | 10-12-14 | vehicle_tool | natural_qa | temperature | residual_full | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a small boat, typically made of wood or other materials, |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | baseline | 101 | generic_only | A rocket is best defined as | : (a) An object that has no engine. ( |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | baseline | 101 | family_non_exact | A canoe is best defined as |  a small boat with oars. Canoes are used for |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | residual_parallel | 101 | exact_label | A rocket is best defined as | :  A.  A vehicle that propels itself by |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | residual_parallel | 101 | family_non_exact | A canoe is best defined as |  a watercraft used for transportation, recreation, or fishing. |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | residual_full | 101 | exact_label | A rocket is best defined as | :  A.  A vehicle that moves by expelling |
| qwen3 | 10-12-14 | vehicle_tool | definition | top_p | residual_full | 101 | family_non_exact | A canoe is best defined as |  a watercraft used for transportation or recreation. It is typically |

| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | baseline | 101 | other | What kind of thing is a rocket? A rocket is |  an artificial satellite, also known as an artificial satellite. The |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | baseline | 101 | generic_only | What kind of thing is a canoe? A canoe is |  a type of# Introduction  A canoe is a type of |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | generic_only | What kind of thing is a rocket? A rocket is |  an **inanimate object**. Refined to its essence |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a type of boat, typically made of wood, but it |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | residual_full | 101 | exact_label | What kind of thing is a rocket? A rocket is |  an extremely powerful motorized vehicle that operates through the propulsion of |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | top_p | residual_full | 101 | generic_only | What kind of thing is a canoe? A canoe is |  a type of a _______.  A canoe is a type |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | baseline | 101 | wrong_family | What kind of thing is a rocket? A rocket is |  typically identified as a technological device designed to project itself through space |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | baseline | 101 | generic_only | What kind of thing is a canoe? A canoe is |  a type of# Introduction  A canoe is a type of |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | exact_label | What kind of thing is a rocket? A rocket is |  an (a) "vehicle" if you consider it as |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | exact_label | What kind of thing is a canoe? A canoe is |  a type of# Vehicle # Watercraft A canoe |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | residual_full | 101 | other | What kind of thing is a rocket? A rocket is |  not merely a complex arrangement of parts to which a particular function |
| glm4 | 24-26-28 | vehicle_tool | natural_qa | temperature | residual_full | 101 | exact_label | What kind of thing is a canoe? A canoe is |  a type of vessel, which is a vehicle, which is |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | baseline | 101 | other | A rocket is best defined as | : a. An artificial satellite launched into orbit around the Earth |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | baseline | 101 | other | A canoe is best defined as |  a light, narrow watercraft, typically with an open deck |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | residual_parallel | 101 | exact_label | A rocket is best defined as |  an_____ vehicle. 首先，我们需要了解火箭的定义。火箭 |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | residual_parallel | 101 | exact_label | A canoe is best defined as |  a small, narrow boat typically used in a vehicle class called |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | residual_full | 101 | exact_label | A rocket is best defined as |  an air-breathing vehicle that carriesA rocket is a complex |
| glm4 | 24-26-28 | vehicle_tool | definition | top_p | residual_full | 101 | other | A canoe is best defined as |  a small, lightweight, open water vessel. In a typical |

| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | baseline | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a self-propelled flying vehicle that is usually propelled by an |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | baseline | 101 | generic_only | What kind of thing is a canoe? A canoe is |  a type of watercraft, specifically a single-seater or |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a self-propelled flying vehicle that uses jet propulsion to move |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | residual_parallel | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a type of vessel used for transportation or recreation. It is |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | residual_full | 101 | exact_label | What kind of thing is a rocket? A rocket is |  a self-propelled flying vehicle designed to carry and expel |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | top_p | residual_full | 101 | generic_only | What kind of thing is a canoe? A canoe is |  a type of watercraft, specifically a narrow, lightweight vessel |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | baseline | 101 | other | What kind of thing is a rocket? A rocket is |  an extremely fast flying machine that is generally used for space travel |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | baseline | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a type of vessel used for transportation or recreation. It is |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | exact_label | What kind of thing is a rocket? A rocket is |  an vehicle, an aerospace vehicle, yes? Or is it |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | residual_parallel | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a type of vessel used for transportation or recreation on water. |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | residual_full | 101 | exact_label | What kind of thing is a rocket? A rocket is |  an ornithopter. Or it is a vehicle. |
| deepseek7b | 16-18-20 | vehicle_tool | natural_qa | temperature | residual_full | 101 | family_non_exact | What kind of thing is a canoe? A canoe is |  a type of vessel used for transportation or recreation. It is |
| deepseek7b | 16-18-20 | vehicle_tool | definition | top_p | baseline | 101 | other | A rocket is best defined as |  something that:  a) Is propelled in a straight line by |
| deepseek7b | 16-18-20 | vehicle_tool | definition | top_p | residual_parallel | 101 | other | A rocket is best defined as |  something that...?   A) fuels its engine   B |
| deepseek7b | 16-18-20 | vehicle_tool | definition | top_p | residual_parallel | 101 | other | A canoe is best defined as |  a9. a canoe is best defined as9 Question |
| deepseek7b | 16-18-20 | vehicle_tool | definition | top_p | residual_full | 101 | other | A rocket is best defined as |  something that...?   A) is fast   B) |
| deepseek7b | 16-18-20 | vehicle_tool | definition | temperature | baseline | 101 | other | A rocket is best defined as |  what? A rocket is best defined as something that uses a |
| deepseek7b | 16-18-20 | vehicle_tool | definition | temperature | baseline | 101 | other | A canoe is best defined as |  a9.4 inches wide, 58 inches long |

