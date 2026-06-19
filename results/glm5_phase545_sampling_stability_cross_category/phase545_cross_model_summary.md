# Phase545 Sampling Stability and Cross-Category Summary

## qwen3

pairs=['vehicle_clothing', 'vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'10-12-14': [10, 12, 14]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| pair | scaffold | mode | condition | win | base family | family mean | std | gain | exact | comp family | stability | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| vehicle_clothing | natural_qa | top_p | baseline | 10-12-14 | 0.97 | 0.97 | 0.05 | +0.00 | 0.86 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_parallel | 10-12-14 | 0.97 | 0.97 | 0.05 | +0.00 | 0.86 | 0.00 | 0.00 | flat |
| vehicle_clothing | natural_qa | top_p | residual_full | 10-12-14 | 0.97 | 0.98 | 0.04 | +0.02 | 0.88 | 0.00 | 0.38 | flat |
| vehicle_clothing | natural_qa | temperature | baseline | 10-12-14 | 0.98 | 0.98 | 0.04 | +0.00 | 0.86 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_parallel | 10-12-14 | 0.98 | 0.97 | 0.05 | -0.02 | 0.84 | 0.00 | -0.29 | flat |
| vehicle_clothing | natural_qa | temperature | residual_full | 10-12-14 | 0.98 | 0.97 | 0.05 | -0.02 | 0.84 | 0.00 | -0.29 | flat |
| vehicle_clothing | definition | top_p | baseline | 10-12-14 | 0.83 | 0.83 | 0.09 | +0.00 | 0.62 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | top_p | residual_parallel | 10-12-14 | 0.83 | 0.92 | 0.06 | +0.09 | 0.72 | 0.00 | 1.55 | flat |
| vehicle_clothing | definition | top_p | residual_full | 10-12-14 | 0.83 | 0.91 | 0.08 | +0.08 | 0.75 | 0.00 | 0.94 | flat |
| vehicle_clothing | definition | temperature | baseline | 10-12-14 | 0.80 | 0.80 | 0.09 | +0.00 | 0.58 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | temperature | residual_parallel | 10-12-14 | 0.80 | 0.80 | 0.14 | +0.00 | 0.66 | 0.00 | 0.00 | flat |
| vehicle_clothing | definition | temperature | residual_full | 10-12-14 | 0.80 | 0.92 | 0.11 | +0.12 | 0.70 | 0.00 | 1.17 | weak_positive |
| vehicle_clothing | sentence_completion | top_p | baseline | 10-12-14 | 0.86 | 0.86 | 0.10 | +0.00 | 0.25 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.86 | 0.88 | 0.06 | +0.02 | 0.36 | 0.00 | 0.25 | flat |
| vehicle_clothing | sentence_completion | top_p | residual_full | 10-12-14 | 0.86 | 0.80 | 0.09 | -0.06 | 0.28 | 0.00 | -0.72 | flat |
| vehicle_clothing | sentence_completion | temperature | baseline | 10-12-14 | 0.83 | 0.83 | 0.15 | +0.00 | 0.33 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.83 | 0.84 | 0.12 | +0.02 | 0.33 | 0.00 | 0.13 | flat |
| vehicle_clothing | sentence_completion | temperature | residual_full | 10-12-14 | 0.83 | 0.83 | 0.12 | +0.00 | 0.30 | 0.00 | 0.00 | flat |
| vehicle_tool | natural_qa | top_p | baseline | 10-12-14 | 0.97 | 0.97 | 0.05 | +0.00 | 0.86 | 0.05 | 0.00 | baseline |
| vehicle_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.97 | 0.98 | 0.04 | +0.02 | 0.88 | 0.03 | 0.38 | flat |
| vehicle_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.97 | 0.97 | 0.05 | +0.00 | 0.88 | 0.06 | 0.00 | flat |
| vehicle_tool | natural_qa | temperature | baseline | 10-12-14 | 0.98 | 0.98 | 0.04 | +0.00 | 0.86 | 0.08 | 0.00 | baseline |
| vehicle_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.98 | 0.97 | 0.05 | -0.02 | 0.86 | 0.05 | -0.29 | flat |
| vehicle_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.98 | 0.97 | 0.05 | -0.02 | 0.86 | 0.05 | -0.29 | flat |
| vehicle_tool | definition | top_p | baseline | 10-12-14 | 0.83 | 0.83 | 0.09 | +0.00 | 0.62 | 0.05 | 0.00 | baseline |
| vehicle_tool | definition | top_p | residual_parallel | 10-12-14 | 0.83 | 0.92 | 0.09 | +0.09 | 0.72 | 0.02 | 1.08 | flat |
| vehicle_tool | definition | top_p | residual_full | 10-12-14 | 0.83 | 0.91 | 0.10 | +0.08 | 0.70 | 0.05 | 0.75 | flat |
| vehicle_tool | definition | temperature | baseline | 10-12-14 | 0.80 | 0.80 | 0.09 | +0.00 | 0.58 | 0.08 | 0.00 | baseline |
| vehicle_tool | definition | temperature | residual_parallel | 10-12-14 | 0.80 | 0.83 | 0.12 | +0.03 | 0.64 | 0.06 | 0.25 | flat |
| vehicle_tool | definition | temperature | residual_full | 10-12-14 | 0.80 | 0.91 | 0.10 | +0.11 | 0.72 | 0.03 | 1.06 | weak_positive |
| vehicle_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.86 | 0.86 | 0.10 | +0.00 | 0.25 | 0.11 | 0.00 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.86 | 0.89 | 0.10 | +0.03 | 0.33 | 0.11 | 0.32 | flat |
| vehicle_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.86 | 0.88 | 0.12 | +0.02 | 0.36 | 0.06 | 0.12 | flat |
| vehicle_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.83 | 0.83 | 0.15 | +0.00 | 0.33 | 0.09 | 0.00 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.83 | 0.80 | 0.12 | -0.03 | 0.30 | 0.08 | -0.25 | flat |
| vehicle_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.83 | 0.83 | 0.09 | +0.00 | 0.34 | 0.09 | 0.00 | flat |
| fruit_vegetable | natural_qa | top_p | baseline | 10-12-14 | 0.88 | 0.88 | 0.00 | +0.00 | 0.88 | 0.00 | 0.00 | baseline |
| fruit_vegetable | natural_qa | top_p | residual_parallel | 10-12-14 | 0.88 | 0.88 | 0.00 | +0.00 | 0.88 | 0.00 | 0.00 | flat |
| fruit_vegetable | natural_qa | top_p | residual_full | 10-12-14 | 0.88 | 0.88 | 0.00 | +0.00 | 0.88 | 0.00 | 0.00 | flat |
| fruit_vegetable | natural_qa | temperature | baseline | 10-12-14 | 0.86 | 0.86 | 0.04 | +0.00 | 0.86 | 0.00 | 0.00 | baseline |
| fruit_vegetable | natural_qa | temperature | residual_parallel | 10-12-14 | 0.86 | 0.88 | 0.00 | +0.02 | 0.88 | 0.00 | 15625.00 | flat |
| fruit_vegetable | natural_qa | temperature | residual_full | 10-12-14 | 0.86 | 0.88 | 0.00 | +0.02 | 0.88 | 0.02 | 15625.00 | flat |
| fruit_vegetable | definition | top_p | baseline | 10-12-14 | 0.73 | 0.73 | 0.12 | +0.00 | 0.59 | 0.05 | 0.00 | baseline |
| fruit_vegetable | definition | top_p | residual_parallel | 10-12-14 | 0.73 | 0.69 | 0.06 | -0.05 | 0.61 | 0.05 | -0.75 | flat |
| fruit_vegetable | definition | top_p | residual_full | 10-12-14 | 0.73 | 0.70 | 0.09 | -0.03 | 0.53 | 0.02 | -0.36 | flat |
| fruit_vegetable | definition | temperature | baseline | 10-12-14 | 0.70 | 0.70 | 0.12 | +0.00 | 0.55 | 0.03 | 0.00 | baseline |
| fruit_vegetable | definition | temperature | residual_parallel | 10-12-14 | 0.70 | 0.67 | 0.12 | -0.03 | 0.53 | 0.03 | -0.25 | flat |
| fruit_vegetable | definition | temperature | residual_full | 10-12-14 | 0.70 | 0.55 | 0.12 | -0.16 | 0.45 | 0.00 | -1.26 | negative |
| fruit_vegetable | sentence_completion | top_p | baseline | 10-12-14 | 0.20 | 0.20 | 0.09 | +0.00 | 0.20 | 0.02 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.20 | 0.16 | 0.14 | -0.05 | 0.16 | 0.00 | -0.34 | flat |
| fruit_vegetable | sentence_completion | top_p | residual_full | 10-12-14 | 0.20 | 0.14 | 0.10 | -0.06 | 0.12 | 0.00 | -0.64 | flat |
| fruit_vegetable | sentence_completion | temperature | baseline | 10-12-14 | 0.20 | 0.20 | 0.09 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.20 | 0.16 | 0.05 | -0.05 | 0.14 | 0.00 | -0.87 | flat |
| fruit_vegetable | sentence_completion | temperature | residual_full | 10-12-14 | 0.20 | 0.12 | 0.11 | -0.08 | 0.11 | 0.03 | -0.72 | flat |
| animal_tool | natural_qa | top_p | baseline | 10-12-14 | 0.66 | 0.66 | 0.08 | +0.00 | 0.22 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.66 | 0.73 | 0.07 | +0.08 | 0.22 | 0.00 | 1.04 | flat |
| animal_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.66 | 0.72 | 0.05 | +0.06 | 0.22 | 0.00 | 1.15 | flat |
| animal_tool | natural_qa | temperature | baseline | 10-12-14 | 0.64 | 0.64 | 0.04 | +0.00 | 0.17 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.64 | 0.66 | 0.08 | +0.02 | 0.17 | 0.00 | 0.19 | flat |
| animal_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.64 | 0.72 | 0.05 | +0.08 | 0.20 | 0.00 | 1.44 | flat |
| animal_tool | definition | top_p | baseline | 10-12-14 | 0.52 | 0.52 | 0.15 | +0.00 | 0.17 | 0.00 | 0.00 | baseline |
| animal_tool | definition | top_p | residual_parallel | 10-12-14 | 0.52 | 0.56 | 0.12 | +0.05 | 0.20 | 0.00 | 0.37 | flat |
| animal_tool | definition | top_p | residual_full | 10-12-14 | 0.52 | 0.41 | 0.14 | -0.11 | 0.16 | 0.00 | -0.80 | negative |
| animal_tool | definition | temperature | baseline | 10-12-14 | 0.45 | 0.45 | 0.15 | +0.00 | 0.14 | 0.00 | 0.00 | baseline |
| animal_tool | definition | temperature | residual_parallel | 10-12-14 | 0.45 | 0.52 | 0.13 | +0.06 | 0.20 | 0.00 | 0.47 | flat |
| animal_tool | definition | temperature | residual_full | 10-12-14 | 0.45 | 0.53 | 0.18 | +0.08 | 0.22 | 0.00 | 0.42 | flat |
| animal_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.23 | 0.23 | 0.18 | +0.00 | 0.17 | 0.00 | 0.00 | baseline |
| animal_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.23 | 0.28 | 0.12 | +0.05 | 0.19 | 0.00 | 0.39 | flat |
| animal_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.23 | 0.12 | 0.06 | -0.11 | 0.11 | 0.00 | -1.75 | negative |
| animal_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.20 | 0.20 | 0.11 | +0.00 | 0.14 | 0.03 | 0.00 | baseline |
| animal_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.20 | 0.27 | 0.10 | +0.06 | 0.22 | 0.00 | 0.64 | flat |
| animal_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.20 | 0.12 | 0.09 | -0.08 | 0.11 | 0.02 | -0.88 | flat |
| fruit_tool | natural_qa | top_p | baseline | 10-12-14 | 0.88 | 0.88 | 0.00 | +0.00 | 0.88 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.88 | 0.89 | 0.04 | +0.02 | 0.89 | 0.00 | 0.38 | flat |
| fruit_tool | natural_qa | top_p | residual_full | 10-12-14 | 0.88 | 0.88 | 0.00 | +0.00 | 0.88 | 0.00 | 0.00 | flat |
| fruit_tool | natural_qa | temperature | baseline | 10-12-14 | 0.86 | 0.86 | 0.04 | +0.00 | 0.86 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | temperature | residual_parallel | 10-12-14 | 0.86 | 0.86 | 0.04 | +0.00 | 0.86 | 0.00 | 0.00 | flat |
| fruit_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.86 | 0.89 | 0.04 | +0.03 | 0.89 | 0.00 | 0.76 | flat |
| fruit_tool | definition | top_p | baseline | 10-12-14 | 0.73 | 0.73 | 0.12 | +0.00 | 0.59 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | top_p | residual_parallel | 10-12-14 | 0.73 | 0.77 | 0.07 | +0.03 | 0.64 | 0.00 | 0.42 | flat |
| fruit_tool | definition | top_p | residual_full | 10-12-14 | 0.73 | 0.67 | 0.06 | -0.06 | 0.66 | 0.00 | -1.03 | flat |
| fruit_tool | definition | temperature | baseline | 10-12-14 | 0.70 | 0.70 | 0.12 | +0.00 | 0.55 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | temperature | residual_parallel | 10-12-14 | 0.70 | 0.67 | 0.15 | -0.03 | 0.50 | 0.00 | -0.21 | flat |
| fruit_tool | definition | temperature | residual_full | 10-12-14 | 0.70 | 0.66 | 0.15 | -0.05 | 0.52 | 0.00 | -0.31 | flat |
| fruit_tool | sentence_completion | top_p | baseline | 10-12-14 | 0.20 | 0.20 | 0.09 | +0.00 | 0.20 | 0.00 | 0.00 | baseline |
| fruit_tool | sentence_completion | top_p | residual_parallel | 10-12-14 | 0.20 | 0.12 | 0.11 | -0.08 | 0.11 | 0.00 | -0.72 | flat |
| fruit_tool | sentence_completion | top_p | residual_full | 10-12-14 | 0.20 | 0.11 | 0.07 | -0.09 | 0.11 | 0.02 | -1.25 | flat |
| fruit_tool | sentence_completion | temperature | baseline | 10-12-14 | 0.20 | 0.20 | 0.09 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| fruit_tool | sentence_completion | temperature | residual_parallel | 10-12-14 | 0.20 | 0.19 | 0.12 | -0.02 | 0.19 | 0.00 | -0.12 | flat |
| fruit_tool | sentence_completion | temperature | residual_full | 10-12-14 | 0.20 | 0.19 | 0.06 | -0.02 | 0.19 | 0.02 | -0.25 | flat |

## glm4

pairs=['vehicle_clothing', 'vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'24-26-28': [24, 26, 28]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| pair | scaffold | mode | condition | win | base family | family mean | std | gain | exact | comp family | stability | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| vehicle_clothing | natural_qa | top_p | baseline | 24-26-28 | 0.73 | 0.73 | 0.10 | +0.00 | 0.56 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_parallel | 24-26-28 | 0.73 | 0.94 | 0.06 | +0.20 | 0.92 | 0.00 | 3.25 | stable_positive |
| vehicle_clothing | natural_qa | top_p | residual_full | 24-26-28 | 0.73 | 0.88 | 0.09 | +0.14 | 0.78 | 0.00 | 1.59 | weak_positive |
| vehicle_clothing | natural_qa | temperature | baseline | 24-26-28 | 0.73 | 0.73 | 0.16 | +0.00 | 0.58 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_parallel | 24-26-28 | 0.73 | 0.89 | 0.07 | +0.16 | 0.88 | 0.00 | 2.09 | weak_positive |
| vehicle_clothing | natural_qa | temperature | residual_full | 24-26-28 | 0.73 | 0.86 | 0.07 | +0.12 | 0.80 | 0.00 | 1.67 | weak_positive |
| vehicle_clothing | definition | top_p | baseline | 24-26-28 | 0.62 | 0.62 | 0.12 | +0.00 | 0.44 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | top_p | residual_parallel | 24-26-28 | 0.62 | 0.84 | 0.10 | +0.22 | 0.80 | 0.00 | 2.11 | stable_positive |
| vehicle_clothing | definition | top_p | residual_full | 24-26-28 | 0.62 | 0.81 | 0.09 | +0.19 | 0.69 | 0.00 | 2.12 | weak_positive |
| vehicle_clothing | definition | temperature | baseline | 24-26-28 | 0.64 | 0.64 | 0.12 | +0.00 | 0.41 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | temperature | residual_parallel | 24-26-28 | 0.64 | 0.84 | 0.08 | +0.20 | 0.83 | 0.02 | 2.46 | stable_positive |
| vehicle_clothing | definition | temperature | residual_full | 24-26-28 | 0.64 | 0.67 | 0.22 | +0.03 | 0.59 | 0.00 | 0.14 | flat |
| vehicle_clothing | sentence_completion | top_p | baseline | 24-26-28 | 0.52 | 0.52 | 0.21 | +0.00 | 0.14 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.52 | 0.89 | 0.12 | +0.38 | 0.75 | 0.00 | 3.24 | stable_positive |
| vehicle_clothing | sentence_completion | top_p | residual_full | 24-26-28 | 0.52 | 0.61 | 0.13 | +0.09 | 0.28 | 0.00 | 0.71 | flat |
| vehicle_clothing | sentence_completion | temperature | baseline | 24-26-28 | 0.36 | 0.36 | 0.07 | +0.00 | 0.11 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.36 | 0.84 | 0.15 | +0.48 | 0.69 | 0.00 | 3.23 | stable_positive |
| vehicle_clothing | sentence_completion | temperature | residual_full | 24-26-28 | 0.36 | 0.61 | 0.07 | +0.25 | 0.25 | 0.00 | 3.34 | stable_positive |
| vehicle_tool | natural_qa | top_p | baseline | 24-26-28 | 0.73 | 0.73 | 0.10 | +0.00 | 0.56 | 0.06 | 0.00 | baseline |
| vehicle_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.73 | 0.97 | 0.05 | +0.23 | 0.92 | 0.00 | 4.33 | stable_positive |
| vehicle_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.73 | 0.81 | 0.14 | +0.08 | 0.67 | 0.00 | 0.56 | flat |
| vehicle_tool | natural_qa | temperature | baseline | 24-26-28 | 0.73 | 0.73 | 0.16 | +0.00 | 0.58 | 0.09 | 0.00 | baseline |
| vehicle_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.73 | 0.98 | 0.04 | +0.25 | 0.94 | 0.00 | 6.05 | stable_positive |
| vehicle_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.73 | 0.86 | 0.07 | +0.12 | 0.69 | 0.00 | 1.67 | weak_positive |
| vehicle_tool | definition | top_p | baseline | 24-26-28 | 0.62 | 0.62 | 0.12 | +0.00 | 0.44 | 0.05 | 0.00 | baseline |
| vehicle_tool | definition | top_p | residual_parallel | 24-26-28 | 0.62 | 0.91 | 0.08 | +0.28 | 0.84 | 0.00 | 3.40 | stable_positive |
| vehicle_tool | definition | top_p | residual_full | 24-26-28 | 0.62 | 0.80 | 0.09 | +0.17 | 0.72 | 0.00 | 1.98 | weak_positive |
| vehicle_tool | definition | temperature | baseline | 24-26-28 | 0.64 | 0.64 | 0.12 | +0.00 | 0.41 | 0.06 | 0.00 | baseline |
| vehicle_tool | definition | temperature | residual_parallel | 24-26-28 | 0.64 | 0.91 | 0.08 | +0.27 | 0.88 | 0.00 | 3.21 | stable_positive |
| vehicle_tool | definition | temperature | residual_full | 24-26-28 | 0.64 | 0.78 | 0.14 | +0.14 | 0.59 | 0.02 | 1.03 | weak_positive |
| vehicle_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.52 | 0.52 | 0.21 | +0.00 | 0.14 | 0.19 | 0.00 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.52 | 0.94 | 0.09 | +0.42 | 0.77 | 0.00 | 4.77 | stable_positive |
| vehicle_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.52 | 0.73 | 0.12 | +0.22 | 0.30 | 0.00 | 1.89 | stable_positive |
| vehicle_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.36 | 0.36 | 0.07 | +0.00 | 0.11 | 0.25 | 0.00 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.36 | 0.84 | 0.14 | +0.48 | 0.75 | 0.00 | 3.56 | stable_positive |
| vehicle_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.36 | 0.84 | 0.08 | +0.48 | 0.38 | 0.02 | 5.86 | stable_positive |
| fruit_vegetable | natural_qa | top_p | baseline | 24-26-28 | 0.62 | 0.62 | 0.12 | +0.00 | 0.52 | 0.00 | 0.00 | baseline |
| fruit_vegetable | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 0.98 | 0.04 | +0.36 | 1.00 | 0.00 | 8.69 | stable_positive |
| fruit_vegetable | natural_qa | top_p | residual_full | 24-26-28 | 0.62 | 0.67 | 0.12 | +0.05 | 0.64 | 0.00 | 0.38 | flat |
| fruit_vegetable | natural_qa | temperature | baseline | 24-26-28 | 0.55 | 0.55 | 0.14 | +0.00 | 0.44 | 0.02 | 0.00 | baseline |
| fruit_vegetable | natural_qa | temperature | residual_parallel | 24-26-28 | 0.55 | 1.00 | 0.00 | +0.45 | 1.00 | 0.00 | 453125.00 | stable_positive |
| fruit_vegetable | natural_qa | temperature | residual_full | 24-26-28 | 0.55 | 0.69 | 0.14 | +0.14 | 0.59 | 0.00 | 1.01 | weak_positive |
| fruit_vegetable | definition | top_p | baseline | 24-26-28 | 0.56 | 0.56 | 0.17 | +0.00 | 0.52 | 0.02 | 0.00 | baseline |
| fruit_vegetable | definition | top_p | residual_parallel | 24-26-28 | 0.56 | 0.95 | 0.06 | +0.39 | 0.98 | 0.00 | 6.45 | stable_positive |
| fruit_vegetable | definition | top_p | residual_full | 24-26-28 | 0.56 | 0.59 | 0.15 | +0.03 | 0.56 | 0.00 | 0.21 | flat |
| fruit_vegetable | definition | temperature | baseline | 24-26-28 | 0.58 | 0.58 | 0.15 | +0.00 | 0.47 | 0.00 | 0.00 | baseline |
| fruit_vegetable | definition | temperature | residual_parallel | 24-26-28 | 0.58 | 0.94 | 0.06 | +0.36 | 0.95 | 0.00 | 5.75 | stable_positive |
| fruit_vegetable | definition | temperature | residual_full | 24-26-28 | 0.58 | 0.52 | 0.15 | -0.06 | 0.47 | 0.00 | -0.43 | flat |
| fruit_vegetable | sentence_completion | top_p | baseline | 24-26-28 | 0.19 | 0.19 | 0.09 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.19 | 0.98 | 0.04 | +0.80 | 0.98 | 0.00 | 19.28 | stable_positive |
| fruit_vegetable | sentence_completion | top_p | residual_full | 24-26-28 | 0.19 | 0.58 | 0.19 | +0.39 | 0.56 | 0.00 | 2.09 | stable_positive |
| fruit_vegetable | sentence_completion | temperature | baseline | 24-26-28 | 0.25 | 0.25 | 0.12 | +0.00 | 0.20 | 0.00 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.25 | 0.98 | 0.04 | +0.73 | 0.98 | 0.00 | 17.76 | stable_positive |
| fruit_vegetable | sentence_completion | temperature | residual_full | 24-26-28 | 0.25 | 0.55 | 0.09 | +0.30 | 0.52 | 0.00 | 3.41 | stable_positive |
| animal_tool | natural_qa | top_p | baseline | 24-26-28 | 0.62 | 0.62 | 0.11 | +0.00 | 0.52 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 1.00 | 0.00 | +0.38 | 1.00 | 0.00 | 375000.00 | stable_positive |
| animal_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.62 | 0.52 | 0.13 | -0.11 | 0.33 | 0.00 | -0.83 | negative |
| animal_tool | natural_qa | temperature | baseline | 24-26-28 | 0.56 | 0.56 | 0.11 | +0.00 | 0.50 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.56 | 1.00 | 0.00 | +0.44 | 0.97 | 0.00 | 437500.00 | stable_positive |
| animal_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.56 | 0.47 | 0.14 | -0.09 | 0.31 | 0.00 | -0.69 | flat |
| animal_tool | definition | top_p | baseline | 24-26-28 | 0.73 | 0.73 | 0.12 | +0.00 | 0.53 | 0.00 | 0.00 | baseline |
| animal_tool | definition | top_p | residual_parallel | 24-26-28 | 0.73 | 0.97 | 0.08 | +0.23 | 0.97 | 0.00 | 2.83 | stable_positive |
| animal_tool | definition | top_p | residual_full | 24-26-28 | 0.73 | 0.56 | 0.20 | -0.17 | 0.42 | 0.00 | -0.87 | negative |
| animal_tool | definition | temperature | baseline | 24-26-28 | 0.67 | 0.67 | 0.12 | +0.00 | 0.55 | 0.00 | 0.00 | baseline |
| animal_tool | definition | temperature | residual_parallel | 24-26-28 | 0.67 | 1.00 | 0.00 | +0.33 | 0.97 | 0.00 | 328125.00 | stable_positive |
| animal_tool | definition | temperature | residual_full | 24-26-28 | 0.67 | 0.48 | 0.17 | -0.19 | 0.38 | 0.00 | -1.10 | negative |
| animal_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.11 | 0.11 | 0.12 | +0.00 | 0.05 | 0.05 | 0.00 | baseline |
| animal_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.11 | 1.00 | 0.00 | +0.89 | 1.00 | 0.00 | 890625.00 | stable_positive |
| animal_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.11 | 0.28 | 0.08 | +0.17 | 0.23 | 0.02 | 2.08 | weak_positive |
| animal_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.09 | 0.09 | 0.08 | +0.00 | 0.05 | 0.03 | 0.00 | baseline |
| animal_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.09 | 0.98 | 0.04 | +0.89 | 0.95 | 0.00 | 21.54 | stable_positive |
| animal_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.09 | 0.30 | 0.11 | +0.20 | 0.22 | 0.00 | 1.90 | stable_positive |
| fruit_tool | natural_qa | top_p | baseline | 24-26-28 | 0.62 | 0.62 | 0.12 | +0.00 | 0.52 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 0.98 | 0.04 | +0.36 | 0.98 | 0.00 | 8.69 | stable_positive |
| fruit_tool | natural_qa | top_p | residual_full | 24-26-28 | 0.62 | 0.61 | 0.12 | -0.02 | 0.47 | 0.00 | -0.13 | flat |
| fruit_tool | natural_qa | temperature | baseline | 24-26-28 | 0.55 | 0.55 | 0.14 | +0.00 | 0.44 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.55 | 1.00 | 0.00 | +0.45 | 1.00 | 0.00 | 453125.00 | stable_positive |
| fruit_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.55 | 0.69 | 0.14 | +0.14 | 0.56 | 0.00 | 1.01 | weak_positive |
| fruit_tool | definition | top_p | baseline | 24-26-28 | 0.56 | 0.56 | 0.17 | +0.00 | 0.52 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | top_p | residual_parallel | 24-26-28 | 0.56 | 0.97 | 0.05 | +0.41 | 0.97 | 0.00 | 7.51 | stable_positive |
| fruit_tool | definition | top_p | residual_full | 24-26-28 | 0.56 | 0.64 | 0.13 | +0.08 | 0.59 | 0.00 | 0.59 | flat |
| fruit_tool | definition | temperature | baseline | 24-26-28 | 0.58 | 0.58 | 0.15 | +0.00 | 0.47 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | temperature | residual_parallel | 24-26-28 | 0.58 | 0.94 | 0.06 | +0.36 | 0.94 | 0.00 | 5.75 | stable_positive |
| fruit_tool | definition | temperature | residual_full | 24-26-28 | 0.58 | 0.59 | 0.14 | +0.02 | 0.52 | 0.00 | 0.11 | flat |
| fruit_tool | sentence_completion | top_p | baseline | 24-26-28 | 0.19 | 0.19 | 0.09 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| fruit_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.19 | 0.97 | 0.05 | +0.78 | 0.97 | 0.00 | 14.43 | stable_positive |
| fruit_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.19 | 0.58 | 0.16 | +0.39 | 0.53 | 0.00 | 2.37 | stable_positive |
| fruit_tool | sentence_completion | temperature | baseline | 24-26-28 | 0.25 | 0.25 | 0.12 | +0.00 | 0.20 | 0.02 | 0.00 | baseline |
| fruit_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.25 | 0.97 | 0.08 | +0.72 | 0.98 | 0.00 | 8.69 | stable_positive |
| fruit_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.25 | 0.59 | 0.16 | +0.34 | 0.53 | 0.00 | 2.12 | stable_positive |

## deepseek7b

pairs=['vehicle_clothing', 'vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['natural_qa', 'definition', 'sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full'], windows={'16-18-20': [16, 18, 20]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| pair | scaffold | mode | condition | win | base family | family mean | std | gain | exact | comp family | stability | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| vehicle_clothing | natural_qa | top_p | baseline | 16-18-20 | 0.78 | 0.78 | 0.15 | +0.00 | 0.59 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | top_p | residual_parallel | 16-18-20 | 0.78 | 0.77 | 0.10 | -0.02 | 0.62 | 0.00 | -0.16 | flat |
| vehicle_clothing | natural_qa | top_p | residual_full | 16-18-20 | 0.78 | 0.77 | 0.07 | -0.02 | 0.56 | 0.00 | -0.21 | flat |
| vehicle_clothing | natural_qa | temperature | baseline | 16-18-20 | 0.78 | 0.78 | 0.10 | +0.00 | 0.50 | 0.00 | 0.00 | baseline |
| vehicle_clothing | natural_qa | temperature | residual_parallel | 16-18-20 | 0.78 | 0.86 | 0.10 | +0.08 | 0.59 | 0.00 | 0.80 | flat |
| vehicle_clothing | natural_qa | temperature | residual_full | 16-18-20 | 0.78 | 0.83 | 0.09 | +0.05 | 0.58 | 0.00 | 0.54 | flat |
| vehicle_clothing | definition | top_p | baseline | 16-18-20 | 0.20 | 0.20 | 0.16 | +0.00 | 0.12 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | top_p | residual_parallel | 16-18-20 | 0.20 | 0.16 | 0.12 | -0.05 | 0.12 | 0.00 | -0.39 | flat |
| vehicle_clothing | definition | top_p | residual_full | 16-18-20 | 0.20 | 0.16 | 0.08 | -0.05 | 0.08 | 0.00 | -0.57 | flat |
| vehicle_clothing | definition | temperature | baseline | 16-18-20 | 0.19 | 0.19 | 0.14 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| vehicle_clothing | definition | temperature | residual_parallel | 16-18-20 | 0.19 | 0.19 | 0.18 | +0.00 | 0.14 | 0.00 | 0.00 | flat |
| vehicle_clothing | definition | temperature | residual_full | 16-18-20 | 0.19 | 0.17 | 0.06 | -0.02 | 0.09 | 0.00 | -0.26 | flat |
| vehicle_clothing | sentence_completion | top_p | baseline | 16-18-20 | 0.22 | 0.22 | 0.14 | +0.00 | 0.09 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.22 | 0.17 | 0.16 | -0.05 | 0.03 | 0.00 | -0.28 | flat |
| vehicle_clothing | sentence_completion | top_p | residual_full | 16-18-20 | 0.22 | 0.17 | 0.18 | -0.05 | 0.08 | 0.00 | -0.27 | flat |
| vehicle_clothing | sentence_completion | temperature | baseline | 16-18-20 | 0.12 | 0.12 | 0.12 | +0.00 | 0.05 | 0.00 | 0.00 | baseline |
| vehicle_clothing | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.12 | 0.14 | 0.12 | +0.02 | 0.06 | 0.00 | 0.13 | flat |
| vehicle_clothing | sentence_completion | temperature | residual_full | 16-18-20 | 0.12 | 0.19 | 0.09 | +0.06 | 0.09 | 0.00 | 0.71 | flat |
| vehicle_tool | natural_qa | top_p | baseline | 16-18-20 | 0.78 | 0.78 | 0.15 | +0.00 | 0.59 | 0.06 | 0.00 | baseline |
| vehicle_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.78 | 0.88 | 0.06 | +0.09 | 0.70 | 0.05 | 1.50 | flat |
| vehicle_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.78 | 0.75 | 0.11 | -0.03 | 0.56 | 0.08 | -0.29 | flat |
| vehicle_tool | natural_qa | temperature | baseline | 16-18-20 | 0.78 | 0.78 | 0.10 | +0.00 | 0.50 | 0.06 | 0.00 | baseline |
| vehicle_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.78 | 0.84 | 0.14 | +0.06 | 0.59 | 0.02 | 0.46 | flat |
| vehicle_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.78 | 0.77 | 0.12 | -0.02 | 0.50 | 0.06 | -0.13 | flat |
| vehicle_tool | definition | top_p | baseline | 16-18-20 | 0.20 | 0.20 | 0.16 | +0.00 | 0.12 | 0.00 | 0.00 | baseline |
| vehicle_tool | definition | top_p | residual_parallel | 16-18-20 | 0.20 | 0.20 | 0.09 | +0.00 | 0.17 | 0.00 | 0.00 | flat |
| vehicle_tool | definition | top_p | residual_full | 16-18-20 | 0.20 | 0.19 | 0.11 | -0.02 | 0.12 | 0.02 | -0.14 | flat |
| vehicle_tool | definition | temperature | baseline | 16-18-20 | 0.19 | 0.19 | 0.14 | +0.00 | 0.16 | 0.02 | 0.00 | baseline |
| vehicle_tool | definition | temperature | residual_parallel | 16-18-20 | 0.19 | 0.16 | 0.12 | -0.03 | 0.14 | 0.02 | -0.26 | flat |
| vehicle_tool | definition | temperature | residual_full | 16-18-20 | 0.19 | 0.19 | 0.09 | +0.00 | 0.12 | 0.00 | 0.00 | flat |
| vehicle_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.22 | 0.22 | 0.14 | +0.00 | 0.09 | 0.02 | 0.00 | baseline |
| vehicle_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.22 | 0.19 | 0.14 | -0.03 | 0.08 | 0.02 | -0.22 | flat |
| vehicle_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.22 | 0.09 | 0.08 | -0.12 | 0.03 | 0.00 | -1.51 | negative |
| vehicle_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.12 | 0.12 | 0.12 | +0.00 | 0.05 | 0.02 | 0.00 | baseline |
| vehicle_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.12 | 0.09 | 0.10 | -0.03 | 0.06 | 0.00 | -0.30 | flat |
| vehicle_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.12 | 0.17 | 0.14 | +0.05 | 0.08 | 0.00 | 0.34 | flat |
| fruit_vegetable | natural_qa | top_p | baseline | 16-18-20 | 0.73 | 0.73 | 0.10 | +0.00 | 0.73 | 0.00 | 0.00 | baseline |
| fruit_vegetable | natural_qa | top_p | residual_parallel | 16-18-20 | 0.73 | 0.69 | 0.11 | -0.05 | 0.69 | 0.00 | -0.43 | flat |
| fruit_vegetable | natural_qa | top_p | residual_full | 16-18-20 | 0.73 | 0.77 | 0.10 | +0.03 | 0.75 | 0.00 | 0.32 | flat |
| fruit_vegetable | natural_qa | temperature | baseline | 16-18-20 | 0.67 | 0.67 | 0.09 | +0.00 | 0.67 | 0.00 | 0.00 | baseline |
| fruit_vegetable | natural_qa | temperature | residual_parallel | 16-18-20 | 0.67 | 0.70 | 0.09 | +0.03 | 0.69 | 0.02 | 0.36 | flat |
| fruit_vegetable | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.75 | 0.09 | +0.08 | 0.73 | 0.02 | 0.88 | flat |
| fruit_vegetable | definition | top_p | baseline | 16-18-20 | 0.31 | 0.31 | 0.20 | +0.00 | 0.30 | 0.00 | 0.00 | baseline |
| fruit_vegetable | definition | top_p | residual_parallel | 16-18-20 | 0.31 | 0.30 | 0.16 | -0.02 | 0.28 | 0.03 | -0.09 | flat |
| fruit_vegetable | definition | top_p | residual_full | 16-18-20 | 0.31 | 0.34 | 0.14 | +0.03 | 0.31 | 0.03 | 0.23 | flat |
| fruit_vegetable | definition | temperature | baseline | 16-18-20 | 0.36 | 0.36 | 0.12 | +0.00 | 0.34 | 0.02 | 0.00 | baseline |
| fruit_vegetable | definition | temperature | residual_parallel | 16-18-20 | 0.36 | 0.27 | 0.17 | -0.09 | 0.27 | 0.00 | -0.55 | flat |
| fruit_vegetable | definition | temperature | residual_full | 16-18-20 | 0.36 | 0.34 | 0.12 | -0.02 | 0.33 | 0.02 | -0.13 | flat |
| fruit_vegetable | sentence_completion | top_p | baseline | 16-18-20 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.00 | 0.03 | 0.05 | +0.03 | 0.00 | 0.00 | 0.58 | flat |
| fruit_vegetable | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.00 | flat |
| fruit_vegetable | sentence_completion | temperature | baseline | 16-18-20 | 0.02 | 0.02 | 0.04 | +0.00 | 0.00 | 0.00 | 0.00 | baseline |
| fruit_vegetable | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.02 | 0.03 | 0.05 | +0.02 | 0.03 | 0.02 | 0.29 | flat |
| fruit_vegetable | sentence_completion | temperature | residual_full | 16-18-20 | 0.02 | 0.00 | 0.00 | -0.02 | 0.00 | 0.00 | -15625.00 | flat |
| animal_tool | natural_qa | top_p | baseline | 16-18-20 | 0.83 | 0.83 | 0.11 | +0.00 | 0.59 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.83 | 0.83 | 0.12 | +0.00 | 0.61 | 0.00 | 0.00 | flat |
| animal_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.83 | 0.78 | 0.08 | -0.05 | 0.53 | 0.00 | -0.57 | flat |
| animal_tool | natural_qa | temperature | baseline | 16-18-20 | 0.78 | 0.78 | 0.08 | +0.00 | 0.56 | 0.00 | 0.00 | baseline |
| animal_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.78 | 0.80 | 0.12 | +0.02 | 0.62 | 0.00 | 0.13 | flat |
| animal_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.78 | 0.75 | 0.09 | -0.03 | 0.53 | 0.00 | -0.35 | flat |
| animal_tool | definition | top_p | baseline | 16-18-20 | 0.27 | 0.27 | 0.13 | +0.00 | 0.20 | 0.00 | 0.00 | baseline |
| animal_tool | definition | top_p | residual_parallel | 16-18-20 | 0.27 | 0.27 | 0.19 | +0.00 | 0.23 | 0.00 | 0.00 | flat |
| animal_tool | definition | top_p | residual_full | 16-18-20 | 0.27 | 0.27 | 0.16 | +0.00 | 0.22 | 0.00 | 0.00 | flat |
| animal_tool | definition | temperature | baseline | 16-18-20 | 0.25 | 0.25 | 0.19 | +0.00 | 0.16 | 0.00 | 0.00 | baseline |
| animal_tool | definition | temperature | residual_parallel | 16-18-20 | 0.25 | 0.22 | 0.12 | -0.03 | 0.19 | 0.00 | -0.26 | flat |
| animal_tool | definition | temperature | residual_full | 16-18-20 | 0.25 | 0.23 | 0.13 | -0.02 | 0.16 | 0.00 | -0.12 | flat |
| animal_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.03 | 0.03 | 0.05 | +0.00 | 0.00 | 0.00 | 0.00 | baseline |
| animal_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.03 | 0.03 | 0.05 | +0.00 | 0.02 | 0.00 | 0.00 | flat |
| animal_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.03 | 0.02 | 0.04 | -0.02 | 0.02 | 0.00 | -0.38 | flat |
| animal_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.05 | 0.05 | 0.06 | +0.00 | 0.02 | 0.00 | 0.00 | baseline |
| animal_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.05 | 0.08 | 0.09 | +0.03 | 0.03 | 0.00 | 0.36 | flat |
| animal_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.05 | 0.06 | 0.09 | +0.02 | 0.06 | 0.00 | 0.18 | flat |
| fruit_tool | natural_qa | top_p | baseline | 16-18-20 | 0.73 | 0.73 | 0.10 | +0.00 | 0.73 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.73 | 0.73 | 0.12 | +0.00 | 0.73 | 0.00 | 0.00 | flat |
| fruit_tool | natural_qa | top_p | residual_full | 16-18-20 | 0.73 | 0.77 | 0.10 | +0.03 | 0.75 | 0.00 | 0.32 | flat |
| fruit_tool | natural_qa | temperature | baseline | 16-18-20 | 0.67 | 0.67 | 0.09 | +0.00 | 0.67 | 0.00 | 0.00 | baseline |
| fruit_tool | natural_qa | temperature | residual_parallel | 16-18-20 | 0.67 | 0.72 | 0.08 | +0.05 | 0.72 | 0.00 | 0.57 | flat |
| fruit_tool | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.70 | 0.06 | +0.03 | 0.70 | 0.00 | 0.52 | flat |
| fruit_tool | definition | top_p | baseline | 16-18-20 | 0.31 | 0.31 | 0.20 | +0.00 | 0.30 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | top_p | residual_parallel | 16-18-20 | 0.31 | 0.34 | 0.22 | +0.03 | 0.34 | 0.00 | 0.14 | flat |
| fruit_tool | definition | top_p | residual_full | 16-18-20 | 0.31 | 0.33 | 0.19 | +0.02 | 0.33 | 0.00 | 0.08 | flat |
| fruit_tool | definition | temperature | baseline | 16-18-20 | 0.36 | 0.36 | 0.12 | +0.00 | 0.34 | 0.00 | 0.00 | baseline |
| fruit_tool | definition | temperature | residual_parallel | 16-18-20 | 0.36 | 0.34 | 0.17 | -0.02 | 0.31 | 0.00 | -0.09 | flat |
| fruit_tool | definition | temperature | residual_full | 16-18-20 | 0.36 | 0.22 | 0.08 | -0.14 | 0.22 | 0.00 | -1.70 | negative |
| fruit_tool | sentence_completion | top_p | baseline | 16-18-20 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.00 | baseline |
| fruit_tool | sentence_completion | top_p | residual_parallel | 16-18-20 | 0.00 | 0.02 | 0.04 | +0.02 | 0.02 | 0.00 | 0.38 | flat |
| fruit_tool | sentence_completion | top_p | residual_full | 16-18-20 | 0.00 | 0.03 | 0.05 | +0.03 | 0.02 | 0.00 | 0.58 | flat |
| fruit_tool | sentence_completion | temperature | baseline | 16-18-20 | 0.02 | 0.02 | 0.04 | +0.00 | 0.00 | 0.00 | 0.00 | baseline |
| fruit_tool | sentence_completion | temperature | residual_parallel | 16-18-20 | 0.02 | 0.03 | 0.05 | +0.02 | 0.02 | 0.00 | 0.29 | flat |
| fruit_tool | sentence_completion | temperature | residual_full | 16-18-20 | 0.02 | 0.03 | 0.05 | +0.02 | 0.00 | 0.00 | 0.29 | flat |

## Best Stable Positive Rows

| model | pair | scaffold | mode | condition | win | base | family | std | gain | stability | exact | comp | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | animal_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.11 | 1.00 | 0.00 | +0.89 | 890625.00 | 1.00 | 0.00 | stable_positive |
| glm4 | animal_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.09 | 0.98 | 0.04 | +0.89 | 21.54 | 0.95 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.19 | 0.98 | 0.04 | +0.80 | 19.28 | 0.98 | 0.00 | stable_positive |
| glm4 | fruit_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.19 | 0.97 | 0.05 | +0.78 | 14.43 | 0.97 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.25 | 0.98 | 0.04 | +0.73 | 17.76 | 0.98 | 0.00 | stable_positive |
| glm4 | fruit_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.25 | 0.97 | 0.08 | +0.72 | 8.69 | 0.98 | 0.00 | stable_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.36 | 0.84 | 0.08 | +0.48 | 5.86 | 0.38 | 0.02 | stable_positive |
| glm4 | vehicle_tool | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.36 | 0.84 | 0.14 | +0.48 | 3.56 | 0.75 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | sentence_completion | temperature | residual_parallel | 24-26-28 | 0.36 | 0.84 | 0.15 | +0.48 | 3.23 | 0.69 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | natural_qa | temperature | residual_parallel | 24-26-28 | 0.55 | 1.00 | 0.00 | +0.45 | 453125.00 | 1.00 | 0.00 | stable_positive |
| glm4 | fruit_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.55 | 1.00 | 0.00 | +0.45 | 453125.00 | 1.00 | 0.00 | stable_positive |
| glm4 | animal_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.56 | 1.00 | 0.00 | +0.44 | 437500.00 | 0.97 | 0.00 | stable_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.52 | 0.94 | 0.09 | +0.42 | 4.77 | 0.77 | 0.00 | stable_positive |
| glm4 | fruit_tool | definition | top_p | residual_parallel | 24-26-28 | 0.56 | 0.97 | 0.05 | +0.41 | 7.51 | 0.97 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | definition | top_p | residual_parallel | 24-26-28 | 0.56 | 0.95 | 0.06 | +0.39 | 6.45 | 0.98 | 0.00 | stable_positive |
| glm4 | fruit_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.19 | 0.58 | 0.16 | +0.39 | 2.37 | 0.53 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | sentence_completion | top_p | residual_full | 24-26-28 | 0.19 | 0.58 | 0.19 | +0.39 | 2.09 | 0.56 | 0.00 | stable_positive |
| glm4 | animal_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 1.00 | 0.00 | +0.38 | 375000.00 | 1.00 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | sentence_completion | top_p | residual_parallel | 24-26-28 | 0.52 | 0.89 | 0.12 | +0.38 | 3.24 | 0.75 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 0.98 | 0.04 | +0.36 | 8.69 | 1.00 | 0.00 | stable_positive |
| glm4 | fruit_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.62 | 0.98 | 0.04 | +0.36 | 8.69 | 0.98 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | definition | temperature | residual_parallel | 24-26-28 | 0.58 | 0.94 | 0.06 | +0.36 | 5.75 | 0.95 | 0.00 | stable_positive |
| glm4 | fruit_tool | definition | temperature | residual_parallel | 24-26-28 | 0.58 | 0.94 | 0.06 | +0.36 | 5.75 | 0.94 | 0.00 | stable_positive |
| glm4 | fruit_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.25 | 0.59 | 0.16 | +0.34 | 2.12 | 0.53 | 0.00 | stable_positive |
| glm4 | animal_tool | definition | temperature | residual_parallel | 24-26-28 | 0.67 | 1.00 | 0.00 | +0.33 | 328125.00 | 0.97 | 0.00 | stable_positive |
| glm4 | fruit_vegetable | sentence_completion | temperature | residual_full | 24-26-28 | 0.25 | 0.55 | 0.09 | +0.30 | 3.41 | 0.52 | 0.00 | stable_positive |
| glm4 | vehicle_tool | definition | top_p | residual_parallel | 24-26-28 | 0.62 | 0.91 | 0.08 | +0.28 | 3.40 | 0.84 | 0.00 | stable_positive |
| glm4 | vehicle_tool | definition | temperature | residual_parallel | 24-26-28 | 0.64 | 0.91 | 0.08 | +0.27 | 3.21 | 0.88 | 0.00 | stable_positive |
| glm4 | vehicle_tool | natural_qa | temperature | residual_parallel | 24-26-28 | 0.73 | 0.98 | 0.04 | +0.25 | 6.05 | 0.94 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | sentence_completion | temperature | residual_full | 24-26-28 | 0.36 | 0.61 | 0.07 | +0.25 | 3.34 | 0.25 | 0.00 | stable_positive |
| glm4 | vehicle_tool | natural_qa | top_p | residual_parallel | 24-26-28 | 0.73 | 0.97 | 0.05 | +0.23 | 4.33 | 0.92 | 0.00 | stable_positive |
| glm4 | animal_tool | definition | top_p | residual_parallel | 24-26-28 | 0.73 | 0.97 | 0.08 | +0.23 | 2.83 | 0.97 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | definition | top_p | residual_parallel | 24-26-28 | 0.62 | 0.84 | 0.10 | +0.22 | 2.11 | 0.80 | 0.00 | stable_positive |
| glm4 | vehicle_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.52 | 0.73 | 0.12 | +0.22 | 1.89 | 0.30 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | natural_qa | top_p | residual_parallel | 24-26-28 | 0.73 | 0.94 | 0.06 | +0.20 | 3.25 | 0.92 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | definition | temperature | residual_parallel | 24-26-28 | 0.64 | 0.84 | 0.08 | +0.20 | 2.46 | 0.83 | 0.02 | stable_positive |
| glm4 | animal_tool | sentence_completion | temperature | residual_full | 24-26-28 | 0.09 | 0.30 | 0.11 | +0.20 | 1.90 | 0.22 | 0.00 | stable_positive |
| glm4 | vehicle_clothing | definition | top_p | residual_full | 24-26-28 | 0.62 | 0.81 | 0.09 | +0.19 | 2.12 | 0.69 | 0.00 | weak_positive |
| glm4 | animal_tool | sentence_completion | top_p | residual_full | 24-26-28 | 0.11 | 0.28 | 0.08 | +0.17 | 2.08 | 0.23 | 0.02 | weak_positive |
| glm4 | vehicle_tool | definition | top_p | residual_full | 24-26-28 | 0.62 | 0.80 | 0.09 | +0.17 | 1.98 | 0.72 | 0.00 | weak_positive |
| glm4 | vehicle_clothing | natural_qa | temperature | residual_parallel | 24-26-28 | 0.73 | 0.89 | 0.07 | +0.16 | 2.09 | 0.88 | 0.00 | weak_positive |
| glm4 | vehicle_clothing | natural_qa | top_p | residual_full | 24-26-28 | 0.73 | 0.88 | 0.09 | +0.14 | 1.59 | 0.78 | 0.00 | weak_positive |
| glm4 | vehicle_tool | definition | temperature | residual_full | 24-26-28 | 0.64 | 0.78 | 0.14 | +0.14 | 1.03 | 0.59 | 0.02 | weak_positive |
| glm4 | fruit_vegetable | natural_qa | temperature | residual_full | 24-26-28 | 0.55 | 0.69 | 0.14 | +0.14 | 1.01 | 0.59 | 0.00 | weak_positive |
| glm4 | fruit_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.55 | 0.69 | 0.14 | +0.14 | 1.01 | 0.56 | 0.00 | weak_positive |
| glm4 | vehicle_clothing | natural_qa | temperature | residual_full | 24-26-28 | 0.73 | 0.86 | 0.07 | +0.12 | 1.67 | 0.80 | 0.00 | weak_positive |
| glm4 | vehicle_tool | natural_qa | temperature | residual_full | 24-26-28 | 0.73 | 0.86 | 0.07 | +0.12 | 1.67 | 0.69 | 0.00 | weak_positive |
| qwen3 | vehicle_clothing | definition | temperature | residual_full | 10-12-14 | 0.80 | 0.92 | 0.11 | +0.12 | 1.17 | 0.70 | 0.00 | weak_positive |
| qwen3 | vehicle_tool | definition | temperature | residual_full | 10-12-14 | 0.80 | 0.91 | 0.10 | +0.11 | 1.06 | 0.72 | 0.03 | weak_positive |
| qwen3 | vehicle_clothing | definition | top_p | residual_parallel | 10-12-14 | 0.83 | 0.92 | 0.06 | +0.09 | 1.55 | 0.72 | 0.00 | flat |
| deepseek7b | vehicle_tool | natural_qa | top_p | residual_parallel | 16-18-20 | 0.78 | 0.88 | 0.06 | +0.09 | 1.50 | 0.70 | 0.05 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_parallel | 10-12-14 | 0.83 | 0.92 | 0.09 | +0.09 | 1.08 | 0.72 | 0.02 | flat |
| glm4 | vehicle_clothing | sentence_completion | top_p | residual_full | 24-26-28 | 0.52 | 0.61 | 0.13 | +0.09 | 0.71 | 0.28 | 0.00 | flat |
| qwen3 | animal_tool | natural_qa | temperature | residual_full | 10-12-14 | 0.64 | 0.72 | 0.05 | +0.08 | 1.44 | 0.20 | 0.00 | flat |
| qwen3 | animal_tool | natural_qa | top_p | residual_parallel | 10-12-14 | 0.66 | 0.73 | 0.07 | +0.08 | 1.04 | 0.22 | 0.00 | flat |
| qwen3 | vehicle_clothing | definition | top_p | residual_full | 10-12-14 | 0.83 | 0.91 | 0.08 | +0.08 | 0.94 | 0.75 | 0.00 | flat |
| deepseek7b | fruit_vegetable | natural_qa | temperature | residual_full | 16-18-20 | 0.67 | 0.75 | 0.09 | +0.08 | 0.88 | 0.73 | 0.02 | flat |
| deepseek7b | vehicle_clothing | natural_qa | temperature | residual_parallel | 16-18-20 | 0.78 | 0.86 | 0.10 | +0.08 | 0.80 | 0.59 | 0.00 | flat |
| qwen3 | vehicle_tool | definition | top_p | residual_full | 10-12-14 | 0.83 | 0.91 | 0.10 | +0.08 | 0.75 | 0.70 | 0.05 | flat |
| glm4 | fruit_tool | definition | top_p | residual_full | 24-26-28 | 0.56 | 0.64 | 0.13 | +0.08 | 0.59 | 0.59 | 0.00 | flat |

## Pair Max Gain

| model | pair | max gain | row |
|---|---|---:|---|
| deepseek7b | animal_tool | +0.03 | sentence_completion temperature residual_parallel 16-18-20 std=0.09 cls=flat |
| deepseek7b | fruit_tool | +0.05 | natural_qa temperature residual_parallel 16-18-20 std=0.08 cls=flat |
| deepseek7b | fruit_vegetable | +0.08 | natural_qa temperature residual_full 16-18-20 std=0.09 cls=flat |
| deepseek7b | vehicle_clothing | +0.08 | natural_qa temperature residual_parallel 16-18-20 std=0.10 cls=flat |
| deepseek7b | vehicle_tool | +0.09 | natural_qa top_p residual_parallel 16-18-20 std=0.06 cls=flat |
| glm4 | animal_tool | +0.89 | sentence_completion top_p residual_parallel 24-26-28 std=0.00 cls=stable_positive |
| glm4 | fruit_tool | +0.78 | sentence_completion top_p residual_parallel 24-26-28 std=0.05 cls=stable_positive |
| glm4 | fruit_vegetable | +0.80 | sentence_completion top_p residual_parallel 24-26-28 std=0.04 cls=stable_positive |
| glm4 | vehicle_clothing | +0.48 | sentence_completion temperature residual_parallel 24-26-28 std=0.15 cls=stable_positive |
| glm4 | vehicle_tool | +0.48 | sentence_completion temperature residual_parallel 24-26-28 std=0.14 cls=stable_positive |
| qwen3 | animal_tool | +0.08 | natural_qa top_p residual_parallel 10-12-14 std=0.07 cls=flat |
| qwen3 | fruit_tool | +0.03 | natural_qa temperature residual_full 10-12-14 std=0.04 cls=flat |
| qwen3 | fruit_vegetable | +0.02 | natural_qa temperature residual_parallel 10-12-14 std=0.00 cls=flat |
| qwen3 | vehicle_clothing | +0.12 | definition temperature residual_full 10-12-14 std=0.11 cls=weak_positive |
| qwen3 | vehicle_tool | +0.11 | definition temperature residual_full 10-12-14 std=0.10 cls=weak_positive |

