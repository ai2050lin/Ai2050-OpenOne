# Phase548 Paraphrase Candidate Robustness Summary

## qwen3

pair=vehicle_tool, scaffolds=['forbidden_definition', 'forbidden_sentence_completion', 'forbidden_natural_qa'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout', 'random_full', 'random_perp'], windows={'10-12-14': [10, 12, 14]}, train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | forbidden_definition | top_p | baseline | 10-12-14 | 0.22 | 0.22 | +0.00 | 0.03 | 0.03 | +0.00 | 0.35 | 0.03 | 0.12 | +0.00 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.22 | 0.25 | +0.03 | 0.03 | 0.05 | +0.02 | 0.26 | 0.00 | 0.12 | -0.00 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | residual_full | 10-12-14 | 0.22 | 0.29 | +0.07 | 0.03 | 0.06 | +0.03 | 0.30 | 0.01 | 0.19 | +0.06 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.22 | 0.30 | +0.08 | 0.03 | 0.06 | +0.03 | 0.29 | 0.00 | 0.23 | +0.10 | +0.07 | weak_clean |
| qwen3 | forbidden_definition | top_p | readout | 10-12-14 | 0.22 | 0.25 | +0.03 | 0.03 | 0.05 | +0.02 | 0.26 | 0.00 | 0.12 | -0.00 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | random_full | 10-12-14 | 0.22 | 0.24 | +0.02 | 0.03 | 0.03 | +0.00 | 0.31 | 0.01 | 0.16 | +0.03 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | random_perp | 10-12-14 | 0.22 | 0.29 | +0.07 | 0.03 | 0.04 | +0.01 | 0.28 | 0.01 | 0.18 | +0.05 | +0.07 | flat |
| qwen3 | forbidden_definition | temperature | baseline | 10-12-14 | 0.21 | 0.21 | +0.00 | 0.04 | 0.04 | +0.00 | 0.30 | 0.00 | 0.10 | +0.00 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.21 | 0.23 | +0.02 | 0.04 | 0.04 | +0.00 | 0.31 | 0.03 | 0.10 | -0.00 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | residual_full | 10-12-14 | 0.21 | 0.29 | +0.08 | 0.04 | 0.03 | -0.01 | 0.28 | 0.01 | 0.22 | +0.11 | +0.04 | weak_clean |
| qwen3 | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.21 | 0.28 | +0.07 | 0.04 | 0.05 | +0.01 | 0.32 | 0.01 | 0.18 | +0.07 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | readout | 10-12-14 | 0.21 | 0.23 | +0.02 | 0.04 | 0.04 | +0.00 | 0.31 | 0.03 | 0.10 | -0.00 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | random_full | 10-12-14 | 0.21 | 0.18 | -0.03 | 0.04 | 0.06 | +0.02 | 0.32 | 0.02 | 0.04 | -0.06 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | random_perp | 10-12-14 | 0.21 | 0.25 | +0.04 | 0.04 | 0.05 | +0.01 | 0.30 | 0.02 | 0.14 | +0.03 | +0.04 | flat |
| qwen3 | forbidden_sentence_completion | top_p | baseline | 10-12-14 | 0.66 | 0.66 | +0.00 | 0.04 | 0.04 | +0.00 | 0.62 | 0.00 | 0.61 | +0.00 | -0.10 | flat |
| qwen3 | forbidden_sentence_completion | top_p | residual_parallel | 10-12-14 | 0.66 | 0.64 | -0.02 | 0.04 | 0.04 | +0.00 | 0.52 | 0.01 | 0.58 | -0.03 | -0.10 | flat |
| qwen3 | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.66 | 0.71 | +0.05 | 0.04 | 0.05 | +0.01 | 0.59 | 0.00 | 0.66 | +0.04 | -0.10 | flat |
| qwen3 | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.66 | 0.68 | +0.02 | 0.04 | 0.05 | +0.01 | 0.62 | 0.00 | 0.61 | +0.00 | -0.10 | flat |
| qwen3 | forbidden_sentence_completion | top_p | readout | 10-12-14 | 0.66 | 0.64 | -0.02 | 0.04 | 0.04 | +0.00 | 0.52 | 0.01 | 0.58 | -0.03 | -0.10 | flat |
| qwen3 | forbidden_sentence_completion | top_p | random_full | 10-12-14 | 0.66 | 0.55 | -0.10 | 0.04 | 0.05 | +0.01 | 0.53 | 0.00 | 0.50 | -0.11 | -0.10 | negative |
| qwen3 | forbidden_sentence_completion | top_p | random_perp | 10-12-14 | 0.66 | 0.52 | -0.14 | 0.04 | 0.06 | +0.02 | 0.62 | 0.01 | 0.43 | -0.19 | -0.10 | negative |
| qwen3 | forbidden_sentence_completion | temperature | baseline | 10-12-14 | 0.54 | 0.54 | +0.00 | 0.01 | 0.01 | +0.00 | 0.57 | 0.01 | 0.52 | +0.00 | -0.05 | flat |
| qwen3 | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.54 | 0.53 | -0.01 | 0.01 | 0.03 | +0.02 | 0.53 | 0.01 | 0.49 | -0.03 | -0.05 | flat |
| qwen3 | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.54 | 0.66 | +0.11 | 0.01 | 0.01 | +0.00 | 0.64 | 0.00 | 0.65 | +0.12 | -0.05 | weak_clean |
| qwen3 | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.54 | 0.64 | +0.09 | 0.01 | 0.03 | +0.02 | 0.58 | 0.00 | 0.60 | +0.08 | -0.05 | weak_clean |
| qwen3 | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.54 | 0.53 | -0.01 | 0.01 | 0.03 | +0.02 | 0.53 | 0.01 | 0.49 | -0.03 | -0.05 | flat |
| qwen3 | forbidden_sentence_completion | temperature | random_full | 10-12-14 | 0.54 | 0.47 | -0.07 | 0.01 | 0.03 | +0.02 | 0.53 | 0.03 | 0.41 | -0.11 | -0.05 | negative |
| qwen3 | forbidden_sentence_completion | temperature | random_perp | 10-12-14 | 0.54 | 0.49 | -0.05 | 0.01 | 0.01 | +0.00 | 0.59 | 0.01 | 0.47 | -0.05 | -0.05 | flat |
| qwen3 | forbidden_natural_qa | top_p | baseline | 10-12-14 | 0.48 | 0.48 | +0.00 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | 0.28 | +0.00 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.48 | 0.46 | -0.02 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.27 | -0.01 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.48 | 0.53 | +0.05 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.39 | +0.10 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.48 | 0.53 | +0.05 | 0.07 | 0.10 | +0.03 | 0.00 | 0.00 | 0.38 | +0.09 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.48 | 0.46 | -0.02 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.27 | -0.01 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | random_full | 10-12-14 | 0.48 | 0.48 | +0.00 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.26 | -0.02 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | random_perp | 10-12-14 | 0.48 | 0.47 | -0.01 | 0.07 | 0.10 | +0.03 | 0.00 | 0.00 | 0.30 | +0.02 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | temperature | baseline | 10-12-14 | 0.44 | 0.44 | +0.00 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | 0.27 | +0.00 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.44 | 0.40 | -0.04 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | 0.21 | -0.06 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.44 | 0.49 | +0.05 | 0.07 | 0.06 | -0.01 | 0.01 | 0.00 | 0.36 | +0.09 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.44 | 0.48 | +0.04 | 0.07 | 0.07 | +0.00 | 0.01 | 0.00 | 0.35 | +0.08 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.44 | 0.40 | -0.04 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | 0.21 | -0.06 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | random_full | 10-12-14 | 0.44 | 0.40 | -0.04 | 0.07 | 0.06 | -0.01 | 0.00 | 0.00 | 0.22 | -0.05 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | temperature | random_perp | 10-12-14 | 0.44 | 0.46 | +0.02 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | 0.29 | +0.02 | +0.02 | flat |

## glm4

pair=vehicle_tool, scaffolds=['forbidden_definition', 'forbidden_sentence_completion', 'forbidden_natural_qa'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout', 'random_full', 'random_perp'], windows={'24-26-28': [24, 26, 28]}, train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | forbidden_definition | top_p | baseline | 24-26-28 | 0.39 | 0.39 | +0.00 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | 0.21 | +0.00 | -0.04 | flat |
| glm4 | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.39 | 0.46 | +0.07 | 0.01 | 0.19 | +0.18 | 0.02 | 0.00 | 0.20 | -0.01 | -0.04 | label_leak |
| glm4 | forbidden_definition | top_p | residual_full | 24-26-28 | 0.39 | 0.54 | +0.16 | 0.01 | 0.01 | +0.00 | 0.03 | 0.00 | 0.51 | +0.30 | -0.04 | robust_clean_positive |
| glm4 | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.39 | 0.53 | +0.15 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | 0.49 | +0.28 | -0.04 | weak_clean |
| glm4 | forbidden_definition | top_p | readout | 24-26-28 | 0.39 | 0.46 | +0.07 | 0.01 | 0.19 | +0.18 | 0.02 | 0.00 | 0.20 | -0.01 | -0.04 | label_leak |
| glm4 | forbidden_definition | top_p | random_full | 24-26-28 | 0.39 | 0.32 | -0.06 | 0.01 | 0.00 | -0.01 | 0.05 | 0.00 | 0.22 | +0.01 | -0.04 | flat |
| glm4 | forbidden_definition | top_p | random_perp | 24-26-28 | 0.39 | 0.34 | -0.04 | 0.01 | 0.02 | +0.01 | 0.07 | 0.00 | 0.13 | -0.08 | -0.04 | flat |
| glm4 | forbidden_definition | temperature | baseline | 24-26-28 | 0.36 | 0.36 | +0.00 | 0.00 | 0.00 | +0.00 | 0.06 | 0.00 | 0.23 | +0.00 | +0.02 | flat |
| glm4 | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.20 | +0.20 | 0.02 | 0.01 | 0.10 | -0.12 | +0.02 | label_leak |
| glm4 | forbidden_definition | temperature | residual_full | 24-26-28 | 0.36 | 0.49 | +0.12 | 0.00 | 0.02 | +0.02 | 0.04 | 0.00 | 0.46 | +0.23 | +0.02 | weak_clean |
| glm4 | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.36 | 0.45 | +0.08 | 0.00 | 0.02 | +0.02 | 0.05 | 0.00 | 0.41 | +0.18 | +0.02 | weak_clean |
| glm4 | forbidden_definition | temperature | readout | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.20 | +0.20 | 0.02 | 0.01 | 0.10 | -0.12 | +0.02 | label_leak |
| glm4 | forbidden_definition | temperature | random_full | 24-26-28 | 0.36 | 0.39 | +0.02 | 0.00 | 0.01 | +0.01 | 0.04 | 0.00 | 0.25 | +0.02 | +0.02 | flat |
| glm4 | forbidden_definition | temperature | random_perp | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.00 | +0.00 | 0.10 | 0.00 | 0.20 | -0.03 | +0.02 | flat |
| glm4 | forbidden_sentence_completion | top_p | baseline | 24-26-28 | 0.24 | 0.24 | +0.00 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | 0.24 | +0.00 | +0.00 | flat |
| glm4 | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.24 | 0.32 | +0.08 | 0.00 | 0.10 | +0.10 | 0.70 | 0.00 | 0.22 | -0.02 | +0.00 | weak_clean |
| glm4 | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.24 | 0.45 | +0.21 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | 0.45 | +0.21 | +0.00 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.24 | 0.41 | +0.17 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | 0.41 | +0.17 | +0.00 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.24 | 0.32 | +0.08 | 0.00 | 0.10 | +0.10 | 0.70 | 0.00 | 0.22 | -0.02 | +0.00 | weak_clean |
| glm4 | forbidden_sentence_completion | top_p | random_full | 24-26-28 | 0.24 | 0.24 | +0.00 | 0.00 | 0.00 | +0.00 | 0.71 | 0.01 | 0.22 | -0.02 | +0.00 | flat |
| glm4 | forbidden_sentence_completion | top_p | random_perp | 24-26-28 | 0.24 | 0.21 | -0.03 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | 0.17 | -0.07 | +0.00 | flat |
| glm4 | forbidden_sentence_completion | temperature | baseline | 24-26-28 | 0.29 | 0.29 | +0.00 | 0.00 | 0.00 | +0.00 | 0.60 | 0.00 | 0.26 | +0.00 | -0.01 | flat |
| glm4 | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.29 | 0.34 | +0.05 | 0.00 | 0.18 | +0.18 | 0.70 | 0.01 | 0.16 | -0.10 | -0.01 | label_leak |
| glm4 | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.29 | 0.50 | +0.21 | 0.00 | 0.00 | +0.00 | 0.60 | 0.00 | 0.50 | +0.24 | -0.01 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.29 | 0.56 | +0.27 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | 0.55 | +0.29 | -0.01 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.29 | 0.34 | +0.05 | 0.00 | 0.18 | +0.18 | 0.70 | 0.01 | 0.16 | -0.10 | -0.01 | label_leak |
| glm4 | forbidden_sentence_completion | temperature | random_full | 24-26-28 | 0.29 | 0.28 | -0.01 | 0.00 | 0.01 | +0.01 | 0.67 | 0.00 | 0.25 | -0.01 | -0.01 | flat |
| glm4 | forbidden_sentence_completion | temperature | random_perp | 24-26-28 | 0.29 | 0.27 | -0.02 | 0.00 | 0.01 | +0.01 | 0.53 | 0.01 | 0.20 | -0.06 | -0.01 | flat |
| glm4 | forbidden_natural_qa | top_p | baseline | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | 0.12 | +0.00 | +0.08 | flat |
| glm4 | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.31 | 0.43 | +0.11 | 0.01 | 0.25 | +0.24 | 0.02 | 0.00 | 0.15 | +0.02 | +0.08 | label_leak |
| glm4 | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.31 | 0.50 | +0.19 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | 0.43 | +0.30 | +0.08 | robust_clean_positive |
| glm4 | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.31 | 0.54 | +0.23 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | 0.51 | +0.39 | +0.08 | robust_clean_positive |
| glm4 | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.31 | 0.43 | +0.11 | 0.01 | 0.25 | +0.24 | 0.02 | 0.00 | 0.15 | +0.02 | +0.08 | label_leak |
| glm4 | forbidden_natural_qa | top_p | random_full | 24-26-28 | 0.31 | 0.34 | +0.03 | 0.01 | 0.04 | +0.03 | 0.02 | 0.00 | 0.11 | -0.01 | +0.08 | flat |
| glm4 | forbidden_natural_qa | top_p | random_perp | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | 0.23 | +0.10 | +0.08 | weak_clean |
| glm4 | forbidden_natural_qa | temperature | baseline | 24-26-28 | 0.30 | 0.30 | +0.00 | 0.01 | 0.01 | +0.00 | 0.02 | 0.00 | 0.12 | +0.00 | +0.04 | flat |
| glm4 | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.30 | 0.42 | +0.11 | 0.01 | 0.27 | +0.26 | 0.04 | 0.00 | 0.11 | -0.01 | +0.04 | label_leak |
| glm4 | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.30 | 0.52 | +0.22 | 0.01 | 0.03 | +0.02 | 0.07 | 0.00 | 0.47 | +0.34 | +0.04 | robust_clean_positive |
| glm4 | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.30 | 0.52 | +0.22 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | 0.47 | +0.34 | +0.04 | robust_clean_positive |
| glm4 | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.30 | 0.42 | +0.11 | 0.01 | 0.27 | +0.26 | 0.04 | 0.00 | 0.11 | -0.01 | +0.04 | label_leak |
| glm4 | forbidden_natural_qa | temperature | random_full | 24-26-28 | 0.30 | 0.34 | +0.04 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | 0.15 | +0.02 | +0.04 | flat |
| glm4 | forbidden_natural_qa | temperature | random_perp | 24-26-28 | 0.30 | 0.33 | +0.03 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | 0.17 | +0.04 | +0.04 | flat |

## deepseek7b

pair=vehicle_tool, scaffolds=['forbidden_definition', 'forbidden_sentence_completion', 'forbidden_natural_qa'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout', 'random_full', 'random_perp'], windows={'16-18-20': [16, 18, 20]}, train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], alpha=6.0

| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | forbidden_definition | top_p | baseline | 16-18-20 | 0.16 | 0.16 | +0.00 | 0.10 | 0.10 | +0.00 | 0.21 | 0.06 | -0.03 | +0.00 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.15 | +0.04 | 0.18 | 0.09 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | residual_full | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.18 | +0.07 | 0.18 | 0.09 | -0.08 | -0.05 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.25 | 0.10 | -0.15 | -0.11 | +0.01 | negative |
| deepseek7b | forbidden_definition | top_p | readout | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.15 | +0.04 | 0.18 | 0.09 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | random_full | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.10 | 0.14 | +0.03 | 0.18 | 0.05 | -0.04 | -0.01 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | random_perp | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.10 | 0.14 | +0.03 | 0.18 | 0.07 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | temperature | baseline | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.15 | +0.00 | 0.22 | 0.05 | -0.03 | +0.00 | +0.02 | flat |
| deepseek7b | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.19 | +0.04 | 0.30 | 0.10 | -0.14 | -0.10 | +0.02 | negative |
| deepseek7b | forbidden_definition | temperature | residual_full | 16-18-20 | 0.17 | 0.20 | +0.03 | 0.15 | 0.12 | -0.02 | 0.23 | 0.09 | -0.05 | -0.02 | +0.02 | flat |
| deepseek7b | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.17 | 0.16 | -0.01 | 0.15 | 0.10 | -0.04 | 0.26 | 0.10 | -0.06 | -0.03 | +0.02 | flat |
| deepseek7b | forbidden_definition | temperature | readout | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.19 | +0.04 | 0.30 | 0.10 | -0.14 | -0.10 | +0.02 | negative |
| deepseek7b | forbidden_definition | temperature | random_full | 16-18-20 | 0.17 | 0.19 | +0.02 | 0.15 | 0.15 | +0.00 | 0.27 | 0.09 | -0.07 | -0.04 | +0.02 | flat |
| deepseek7b | forbidden_definition | temperature | random_perp | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.11 | -0.03 | 0.22 | 0.05 | -0.00 | +0.03 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | baseline | 16-18-20 | 0.21 | 0.21 | +0.00 | 0.00 | 0.00 | +0.00 | 0.55 | 0.00 | 0.21 | +0.00 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.21 | 0.29 | +0.08 | 0.00 | 0.04 | +0.04 | 0.62 | 0.00 | 0.25 | +0.04 | +0.01 | weak_clean |
| deepseek7b | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.21 | 0.22 | +0.01 | 0.00 | 0.03 | +0.03 | 0.59 | 0.00 | 0.19 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.21 | 0.19 | -0.02 | 0.00 | 0.01 | +0.01 | 0.56 | 0.01 | 0.17 | -0.04 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.21 | 0.29 | +0.08 | 0.00 | 0.04 | +0.04 | 0.62 | 0.00 | 0.25 | +0.04 | +0.01 | weak_clean |
| deepseek7b | forbidden_sentence_completion | top_p | random_full | 16-18-20 | 0.21 | 0.16 | -0.05 | 0.00 | 0.00 | +0.00 | 0.64 | 0.01 | 0.15 | -0.06 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | random_perp | 16-18-20 | 0.21 | 0.22 | +0.01 | 0.00 | 0.01 | +0.01 | 0.54 | 0.00 | 0.21 | +0.00 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | baseline | 16-18-20 | 0.18 | 0.18 | +0.00 | 0.02 | 0.02 | +0.00 | 0.48 | 0.00 | 0.16 | +0.00 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.18 | 0.24 | +0.06 | 0.02 | 0.02 | +0.00 | 0.51 | 0.00 | 0.21 | +0.05 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | residual_full | 16-18-20 | 0.18 | 0.17 | -0.01 | 0.02 | 0.03 | +0.01 | 0.56 | 0.00 | 0.14 | -0.02 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.18 | 0.27 | +0.09 | 0.02 | 0.04 | +0.02 | 0.49 | 0.01 | 0.22 | +0.06 | +0.02 | weak_clean |
| deepseek7b | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.18 | 0.24 | +0.06 | 0.02 | 0.02 | +0.00 | 0.51 | 0.00 | 0.21 | +0.05 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | random_full | 16-18-20 | 0.18 | 0.15 | -0.03 | 0.02 | 0.03 | +0.01 | 0.64 | 0.00 | 0.11 | -0.04 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | random_perp | 16-18-20 | 0.18 | 0.20 | +0.02 | 0.02 | 0.02 | +0.00 | 0.49 | 0.00 | 0.18 | +0.02 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | top_p | baseline | 16-18-20 | 0.19 | 0.19 | +0.00 | 0.19 | 0.19 | +0.00 | 0.21 | 0.00 | -0.05 | +0.00 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.19 | 0.15 | -0.04 | 0.19 | 0.21 | +0.02 | 0.22 | 0.00 | -0.15 | -0.09 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.19 | 0.16 | -0.03 | 0.19 | 0.23 | +0.04 | 0.23 | 0.00 | -0.17 | -0.11 | -0.01 | negative |
| deepseek7b | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.09 | -0.04 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.19 | 0.15 | -0.04 | 0.19 | 0.21 | +0.02 | 0.22 | 0.00 | -0.15 | -0.09 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | top_p | random_full | 16-18-20 | 0.19 | 0.18 | -0.01 | 0.19 | 0.17 | -0.02 | 0.26 | 0.00 | -0.02 | +0.03 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | top_p | random_perp | 16-18-20 | 0.19 | 0.12 | -0.06 | 0.19 | 0.18 | -0.01 | 0.24 | 0.00 | -0.11 | -0.06 | -0.01 | flat |
| deepseek7b | forbidden_natural_qa | temperature | baseline | 16-18-20 | 0.16 | 0.16 | +0.00 | 0.15 | 0.15 | +0.00 | 0.27 | 0.00 | -0.09 | +0.00 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.15 | 0.20 | +0.05 | 0.30 | 0.00 | -0.05 | +0.04 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.16 | 0.21 | +0.05 | 0.15 | 0.20 | +0.05 | 0.31 | 0.00 | -0.07 | +0.02 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.15 | 0.19 | +0.04 | 0.26 | 0.00 | -0.09 | +0.00 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.15 | 0.20 | +0.05 | 0.30 | 0.00 | -0.05 | +0.04 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | random_full | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.15 | 0.19 | +0.04 | 0.29 | 0.00 | -0.07 | +0.02 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | random_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.15 | 0.17 | +0.02 | 0.28 | 0.01 | -0.05 | +0.04 | +0.02 | flat |

## Best Strict Clean Non-Object Gains

| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.29 | 0.56 | +0.27 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | 0.55 | +0.29 | -0.01 | robust_clean_positive |
| glm4 | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.31 | 0.54 | +0.23 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | 0.51 | +0.39 | +0.08 | robust_clean_positive |
| glm4 | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.30 | 0.52 | +0.22 | 0.01 | 0.03 | +0.02 | 0.07 | 0.00 | 0.47 | +0.34 | +0.04 | robust_clean_positive |
| glm4 | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.30 | 0.52 | +0.22 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | 0.47 | +0.34 | +0.04 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.24 | 0.45 | +0.21 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | 0.45 | +0.21 | +0.00 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.29 | 0.50 | +0.21 | 0.00 | 0.00 | +0.00 | 0.60 | 0.00 | 0.50 | +0.24 | -0.01 | robust_clean_positive |
| glm4 | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.31 | 0.50 | +0.19 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | 0.43 | +0.30 | +0.08 | robust_clean_positive |
| glm4 | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.24 | 0.41 | +0.17 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | 0.41 | +0.17 | +0.00 | robust_clean_positive |
| glm4 | forbidden_definition | top_p | residual_full | 24-26-28 | 0.39 | 0.54 | +0.16 | 0.01 | 0.01 | +0.00 | 0.03 | 0.00 | 0.51 | +0.30 | -0.04 | robust_clean_positive |
| glm4 | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.39 | 0.53 | +0.15 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | 0.49 | +0.28 | -0.04 | weak_clean |
| glm4 | forbidden_definition | temperature | residual_full | 24-26-28 | 0.36 | 0.49 | +0.12 | 0.00 | 0.02 | +0.02 | 0.04 | 0.00 | 0.46 | +0.23 | +0.02 | weak_clean |
| qwen3 | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.54 | 0.66 | +0.11 | 0.01 | 0.01 | +0.00 | 0.64 | 0.00 | 0.65 | +0.12 | -0.05 | weak_clean |
| glm4 | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.30 | 0.42 | +0.11 | 0.01 | 0.27 | +0.26 | 0.04 | 0.00 | 0.11 | -0.01 | +0.04 | label_leak |
| glm4 | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.30 | 0.42 | +0.11 | 0.01 | 0.27 | +0.26 | 0.04 | 0.00 | 0.11 | -0.01 | +0.04 | label_leak |
| glm4 | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.31 | 0.43 | +0.11 | 0.01 | 0.25 | +0.24 | 0.02 | 0.00 | 0.15 | +0.02 | +0.08 | label_leak |
| glm4 | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.31 | 0.43 | +0.11 | 0.01 | 0.25 | +0.24 | 0.02 | 0.00 | 0.15 | +0.02 | +0.08 | label_leak |
| qwen3 | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.54 | 0.64 | +0.09 | 0.01 | 0.03 | +0.02 | 0.58 | 0.00 | 0.60 | +0.08 | -0.05 | weak_clean |
| deepseek7b | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.18 | 0.27 | +0.09 | 0.02 | 0.04 | +0.02 | 0.49 | 0.01 | 0.22 | +0.06 | +0.02 | weak_clean |
| glm4 | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.36 | 0.45 | +0.08 | 0.00 | 0.02 | +0.02 | 0.05 | 0.00 | 0.41 | +0.18 | +0.02 | weak_clean |
| qwen3 | forbidden_definition | temperature | residual_full | 10-12-14 | 0.21 | 0.29 | +0.08 | 0.04 | 0.03 | -0.01 | 0.28 | 0.01 | 0.22 | +0.11 | +0.04 | weak_clean |
| deepseek7b | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.21 | 0.29 | +0.08 | 0.00 | 0.04 | +0.04 | 0.62 | 0.00 | 0.25 | +0.04 | +0.01 | weak_clean |
| deepseek7b | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.21 | 0.29 | +0.08 | 0.00 | 0.04 | +0.04 | 0.62 | 0.00 | 0.25 | +0.04 | +0.01 | weak_clean |
| glm4 | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.24 | 0.32 | +0.08 | 0.00 | 0.10 | +0.10 | 0.70 | 0.00 | 0.22 | -0.02 | +0.00 | weak_clean |
| glm4 | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.24 | 0.32 | +0.08 | 0.00 | 0.10 | +0.10 | 0.70 | 0.00 | 0.22 | -0.02 | +0.00 | weak_clean |
| glm4 | forbidden_natural_qa | top_p | random_perp | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | 0.23 | +0.10 | +0.08 | weak_clean |
| qwen3 | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.22 | 0.30 | +0.08 | 0.03 | 0.06 | +0.03 | 0.29 | 0.00 | 0.23 | +0.10 | +0.07 | weak_clean |
| qwen3 | forbidden_definition | top_p | residual_full | 10-12-14 | 0.22 | 0.29 | +0.07 | 0.03 | 0.06 | +0.03 | 0.30 | 0.01 | 0.19 | +0.06 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | random_perp | 10-12-14 | 0.22 | 0.29 | +0.07 | 0.03 | 0.04 | +0.01 | 0.28 | 0.01 | 0.18 | +0.05 | +0.07 | flat |
| qwen3 | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.21 | 0.28 | +0.07 | 0.04 | 0.05 | +0.01 | 0.32 | 0.01 | 0.18 | +0.07 | +0.04 | flat |
| glm4 | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.39 | 0.46 | +0.07 | 0.01 | 0.19 | +0.18 | 0.02 | 0.00 | 0.20 | -0.01 | -0.04 | label_leak |
| glm4 | forbidden_definition | top_p | readout | 24-26-28 | 0.39 | 0.46 | +0.07 | 0.01 | 0.19 | +0.18 | 0.02 | 0.00 | 0.20 | -0.01 | -0.04 | label_leak |
| deepseek7b | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.18 | 0.24 | +0.06 | 0.02 | 0.02 | +0.00 | 0.51 | 0.00 | 0.21 | +0.05 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.18 | 0.24 | +0.06 | 0.02 | 0.02 | +0.00 | 0.51 | 0.00 | 0.21 | +0.05 | +0.02 | flat |
| qwen3 | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.66 | 0.71 | +0.05 | 0.04 | 0.05 | +0.01 | 0.59 | 0.00 | 0.66 | +0.04 | -0.10 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.16 | 0.21 | +0.05 | 0.15 | 0.20 | +0.05 | 0.31 | 0.00 | -0.07 | +0.02 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.48 | 0.53 | +0.05 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.39 | +0.10 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.48 | 0.53 | +0.05 | 0.07 | 0.10 | +0.03 | 0.00 | 0.00 | 0.38 | +0.09 | +0.00 | flat |
| qwen3 | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.44 | 0.49 | +0.05 | 0.07 | 0.06 | -0.01 | 0.01 | 0.00 | 0.36 | +0.09 | +0.02 | flat |
| glm4 | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.29 | 0.34 | +0.05 | 0.00 | 0.18 | +0.18 | 0.70 | 0.01 | 0.16 | -0.10 | -0.01 | label_leak |
| glm4 | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.29 | 0.34 | +0.05 | 0.00 | 0.18 | +0.18 | 0.70 | 0.01 | 0.16 | -0.10 | -0.01 | label_leak |
| qwen3 | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.44 | 0.48 | +0.04 | 0.07 | 0.07 | +0.00 | 0.01 | 0.00 | 0.35 | +0.08 | +0.02 | flat |
| glm4 | forbidden_natural_qa | temperature | random_full | 24-26-28 | 0.30 | 0.34 | +0.04 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | 0.15 | +0.02 | +0.04 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.15 | 0.20 | +0.05 | 0.30 | 0.00 | -0.05 | +0.04 | +0.02 | flat |
| deepseek7b | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.15 | 0.20 | +0.05 | 0.30 | 0.00 | -0.05 | +0.04 | +0.02 | flat |
| qwen3 | forbidden_definition | temperature | random_perp | 10-12-14 | 0.21 | 0.25 | +0.04 | 0.04 | 0.05 | +0.01 | 0.30 | 0.02 | 0.14 | +0.03 | +0.04 | flat |
| deepseek7b | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.15 | +0.04 | 0.18 | 0.09 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | readout | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.15 | +0.04 | 0.18 | 0.09 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | residual_full | 16-18-20 | 0.16 | 0.20 | +0.04 | 0.10 | 0.18 | +0.07 | 0.18 | 0.09 | -0.08 | -0.05 | +0.01 | flat |
| glm4 | forbidden_natural_qa | temperature | random_perp | 24-26-28 | 0.30 | 0.33 | +0.03 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | 0.17 | +0.04 | +0.04 | flat |
| qwen3 | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.22 | 0.25 | +0.03 | 0.03 | 0.05 | +0.02 | 0.26 | 0.00 | 0.12 | -0.00 | +0.07 | flat |
| qwen3 | forbidden_definition | top_p | readout | 10-12-14 | 0.22 | 0.25 | +0.03 | 0.03 | 0.05 | +0.02 | 0.26 | 0.00 | 0.12 | -0.00 | +0.07 | flat |
| glm4 | forbidden_natural_qa | top_p | random_full | 24-26-28 | 0.31 | 0.34 | +0.03 | 0.01 | 0.04 | +0.03 | 0.02 | 0.00 | 0.11 | -0.01 | +0.08 | flat |
| deepseek7b | forbidden_definition | temperature | residual_full | 16-18-20 | 0.17 | 0.20 | +0.03 | 0.15 | 0.12 | -0.02 | 0.23 | 0.09 | -0.05 | -0.02 | +0.02 | flat |
| glm4 | forbidden_definition | temperature | random_full | 24-26-28 | 0.36 | 0.39 | +0.02 | 0.00 | 0.01 | +0.01 | 0.04 | 0.00 | 0.25 | +0.02 | +0.02 | flat |
| qwen3 | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.66 | 0.68 | +0.02 | 0.04 | 0.05 | +0.01 | 0.62 | 0.00 | 0.61 | +0.00 | -0.10 | flat |
| deepseek7b | forbidden_natural_qa | temperature | random_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.15 | 0.17 | +0.02 | 0.28 | 0.01 | -0.05 | +0.04 | +0.02 | flat |
| qwen3 | forbidden_definition | top_p | random_full | 10-12-14 | 0.22 | 0.24 | +0.02 | 0.03 | 0.03 | +0.00 | 0.31 | 0.01 | 0.16 | +0.03 | +0.07 | flat |
| deepseek7b | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.15 | 0.19 | +0.04 | 0.26 | 0.00 | -0.09 | +0.00 | +0.02 | flat |
| deepseek7b | forbidden_definition | temperature | random_full | 16-18-20 | 0.17 | 0.19 | +0.02 | 0.15 | 0.15 | +0.00 | 0.27 | 0.09 | -0.07 | -0.04 | +0.02 | flat |
| deepseek7b | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.25 | 0.10 | -0.15 | -0.11 | +0.01 | negative |
| qwen3 | forbidden_natural_qa | temperature | random_perp | 10-12-14 | 0.44 | 0.46 | +0.02 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | 0.29 | +0.02 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | temperature | random_perp | 16-18-20 | 0.18 | 0.20 | +0.02 | 0.02 | 0.02 | +0.00 | 0.49 | 0.00 | 0.18 | +0.02 | +0.02 | flat |
| qwen3 | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.21 | 0.23 | +0.02 | 0.04 | 0.04 | +0.00 | 0.31 | 0.03 | 0.10 | -0.00 | +0.04 | flat |
| qwen3 | forbidden_definition | temperature | readout | 10-12-14 | 0.21 | 0.23 | +0.02 | 0.04 | 0.04 | +0.00 | 0.31 | 0.03 | 0.10 | -0.00 | +0.04 | flat |
| deepseek7b | forbidden_natural_qa | temperature | random_full | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.15 | 0.19 | +0.04 | 0.29 | 0.00 | -0.07 | +0.02 | +0.02 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | random_perp | 16-18-20 | 0.21 | 0.22 | +0.01 | 0.00 | 0.01 | +0.01 | 0.54 | 0.00 | 0.21 | +0.00 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | random_full | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.10 | 0.14 | +0.03 | 0.18 | 0.05 | -0.04 | -0.01 | +0.01 | flat |
| deepseek7b | forbidden_definition | top_p | random_perp | 16-18-20 | 0.16 | 0.17 | +0.01 | 0.10 | 0.14 | +0.03 | 0.18 | 0.07 | -0.05 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.21 | 0.22 | +0.01 | 0.00 | 0.03 | +0.03 | 0.59 | 0.00 | 0.19 | -0.02 | +0.01 | flat |
| deepseek7b | forbidden_definition | temperature | random_perp | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.11 | -0.03 | 0.22 | 0.05 | -0.00 | +0.03 | +0.02 | flat |
| qwen3 | forbidden_natural_qa | top_p | random_full | 10-12-14 | 0.48 | 0.48 | +0.00 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | 0.26 | -0.02 | +0.00 | flat |
| glm4 | forbidden_sentence_completion | top_p | random_full | 24-26-28 | 0.24 | 0.24 | +0.00 | 0.00 | 0.00 | +0.00 | 0.71 | 0.01 | 0.22 | -0.02 | +0.00 | flat |
| deepseek7b | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.09 | -0.04 | -0.01 | flat |
| deepseek7b | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.19 | +0.04 | 0.30 | 0.10 | -0.14 | -0.10 | +0.02 | negative |
| deepseek7b | forbidden_definition | temperature | readout | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.15 | 0.19 | +0.04 | 0.30 | 0.10 | -0.14 | -0.10 | +0.02 | negative |
| glm4 | forbidden_definition | temperature | random_perp | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.00 | +0.00 | 0.10 | 0.00 | 0.20 | -0.03 | +0.02 | flat |
| qwen3 | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.54 | 0.53 | -0.01 | 0.01 | 0.03 | +0.02 | 0.53 | 0.01 | 0.49 | -0.03 | -0.05 | flat |
| qwen3 | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.54 | 0.53 | -0.01 | 0.01 | 0.03 | +0.02 | 0.53 | 0.01 | 0.49 | -0.03 | -0.05 | flat |
| glm4 | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.20 | +0.20 | 0.02 | 0.01 | 0.10 | -0.12 | +0.02 | label_leak |
| glm4 | forbidden_definition | temperature | readout | 24-26-28 | 0.36 | 0.35 | -0.01 | 0.00 | 0.20 | +0.20 | 0.02 | 0.01 | 0.10 | -0.12 | +0.02 | label_leak |

## Model Max Strict Clean Gain

| model | max clean gain | random gain | score gain | row | class |
|---|---:|---:|---:|---|---|
| deepseek7b | +0.09 | +0.02 | +0.06 | forbidden_sentence_completion temperature residual_perp 16-18-20 | weak_clean |
| glm4 | +0.27 | -0.01 | +0.29 | forbidden_sentence_completion temperature residual_perp 24-26-28 | robust_clean_positive |
| qwen3 | +0.11 | -0.05 | +0.12 | forbidden_sentence_completion temperature residual_full 10-12-14 | weak_clean |

## Readable Samples

| model | window | scaffold | mode | condition | seed | object | quality | clean-no | labels | target terms | suffix |
|---|---|---|---|---|---:|---|---|---|---|---|---|
| qwen3 | 10-12-14 | forbidden_definition | top_p | baseline | 101 | tram | clean_synonym | True |  | transport | : a method of transport that uses rails for movement and is |
| qwen3 | 10-12-14 | forbidden_definition | top_p | baseline | 101 | subway | other | False |  |  |  a system of elevated or underground rail lines that connect different parts |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_parallel | 101 | tram | clean_synonym | True |  | transport,propelled | : A mobile, self-propelled public transport system designed for |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_parallel | 101 | subway | clean_synonym | True |  | travel |  a system of ___________ that allows people to travel from |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_parallel | 101 | helicopter | generic_only | False |  |  |  a...?  Helicopter is best described as a type of |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_full | 101 | tram | clean_synonym | True |  | transportation | :  Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_full | 101 | subway | clean_synonym | True |  | travel |  a system of ________ that allows people to travel from one |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_perp | 101 | tram | clean_synonym | True |  | propelled | : a mobile, self-propelled structure that operates on rails |
| qwen3 | 10-12-14 | forbidden_definition | top_p | residual_perp | 101 | subway | clean_synonym | True |  | travel |  a system of ________ that allows people to travel from one |
| qwen3 | 10-12-14 | forbidden_definition | top_p | readout | 101 | tram | clean_synonym | True |  | transport,propelled | : A mobile, self-propelled public transport system designed for |
| qwen3 | 10-12-14 | forbidden_definition | top_p | readout | 101 | subway | clean_synonym | True |  | travel |  a system of ___________ that allows people to travel from |
| qwen3 | 10-12-14 | forbidden_definition | top_p | readout | 101 | helicopter | generic_only | False |  |  |  a...?  Helicopter is best described as a type of |
| qwen3 | 10-12-14 | forbidden_definition | top_p | random_full | 101 | tram | clean_synonym | True |  | transport | : a method of transport that uses rails for movement and is |
| qwen3 | 10-12-14 | forbidden_definition | top_p | random_full | 101 | subway | other | False |  |  |  a system of ______ that connects different parts of a city. |
| qwen3 | 10-12-14 | forbidden_definition | top_p | random_perp | 101 | tram | clean_synonym | True |  | transport | : A type of public transport that operates on rails. Yes |
| qwen3 | 10-12-14 | forbidden_definition | top_p | random_perp | 101 | subway | clean_synonym | True |  | travel |  a system of __________ that allows people to travel from one |
| qwen3 | 10-12-14 | forbidden_definition | temperature | baseline | 101 | tram | clean_synonym | True |  | transportation | ...  Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | temperature | baseline | 101 | subway | clean_synonym | True |  | carry |  a system of tracks and tunnels, with trains that carry people |
| qwen3 | 10-12-14 | forbidden_definition | temperature | baseline | 101 | helicopter | other | False |  |  | ... \| by Steve B. \| Medium Define helicopter without |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_parallel | 101 | tram | other | False |  |  | ...  Tram is best described as a specialized form of public |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_parallel | 101 | subway | other | False |  |  |  a system of below-ground tunnels connected to above-ground stations. |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_parallel | 101 | helicopter | other | False |  |  | ... - Brainly.com Define helicopter without using these words |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_full | 101 | tram | clean_synonym | True |  | transportation |   Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_full | 101 | subway | clean_synonym | True |  | travel |  a system of tracks that connect stations, allowing people to travel |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_full | 101 | helicopter | clean_synonym | True |  | aircraft | ... ?  Helicopter is best described as a type of aircraft |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_perp | 101 | tram | clean_synonym | True |  | transportation |   Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_perp | 101 | subway | other | False |  |  |  a system of... Okay, so I need to define " |
| qwen3 | 10-12-14 | forbidden_definition | temperature | residual_perp | 101 | helicopter | other | False |  |  | ... - Brainly.com Define helicopter without using these words |
| qwen3 | 10-12-14 | forbidden_definition | temperature | readout | 101 | tram | other | False |  |  | ...  Tram is best described as a specialized form of public |
| qwen3 | 10-12-14 | forbidden_definition | temperature | readout | 101 | subway | other | False |  |  |  a system of below-ground tunnels connected to above-ground stations. |
| qwen3 | 10-12-14 | forbidden_definition | temperature | readout | 101 | helicopter | other | False |  |  | ... - Brainly.com Define helicopter without using these words |
| qwen3 | 10-12-14 | forbidden_definition | temperature | random_full | 101 | tram | clean_synonym | True |  | transportation | ...  Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | temperature | random_full | 101 | subway | other | False |  |  |  a system of... Okay, so I need to define " |
| qwen3 | 10-12-14 | forbidden_definition | temperature | random_full | 101 | helicopter | other | False |  |  | ... (please use 3-5 concise and precise words |
| qwen3 | 10-12-14 | forbidden_definition | temperature | random_perp | 101 | tram | clean_synonym | True |  | transportation | ...  Tram is best described as a type of public transportation |
| qwen3 | 10-12-14 | forbidden_definition | temperature | random_perp | 101 | subway | clean_synonym | True |  | transportation |  a system of transportation that connects different parts of a city. |

| glm4 | 24-26-28 | forbidden_definition | top_p | baseline | 101 | tram | clean_synonym | True |  | transportation | :Tram is a form of transportation that operates along a |
| glm4 | 24-26-28 | forbidden_definition | top_p | baseline | 101 | subway | other | False |  |  |  as a an system underground of for for transporting transporting people people |
| glm4 | 24-26-28 | forbidden_definition | top_p | baseline | 101 | helicopter | clean_synonym | True |  | aircraft |  an aircraft that employs:  rotary wings.  An aircraft that |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_parallel | 101 | tram | other | False |  |  | :A:   vehicleA with: a wheels) arrangement that |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_parallel | 101 | subway | clean_synonym | True |  | transportation |  as a an transportation underground system system consisting that of includes trains |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_parallel | 101 | helicopter | clean_synonym | True |  | craft |  a craft that is capable of vertical take-off and landing, |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_parallel | 101 | tractor | synonym_with_label_violation | False | vehicles | machine,carry | :  A machine used to pull vehicles or carry heavy loads. |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_full | 101 | tram | clean_synonym | True |  | transportation | :A:   publicA transportation:  AA systemApublic |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_full | 101 | subway | clean_synonym | True |  | transportation |  as a a form large of network public public transportation transportation transportation |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_full | 101 | helicopter | wrong_synonym | False |  |  | :  A flying device that uses a rotating system of to blades |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_full | 101 | tractor | other | False |  |  |  a:  Agricultural power unit  An agricultural agricultural engine |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_perp | 101 | tram | clean_synonym | True |  | transportation | :A:   tramA is: a a publicB transportation |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_perp | 101 | subway | other | False |  |  |  as a a transit large system underground urban. designed    for |
| glm4 | 24-26-28 | forbidden_definition | top_p | residual_perp | 101 | helicopter | wrong_synonym | False |  |  | :  A flying device that uses a rotating blade to create lift |
| glm4 | 24-26-28 | forbidden_definition | top_p | readout | 101 | tram | other | False |  |  | :A:   vehicleA with: a wheels) arrangement that |
| glm4 | 24-26-28 | forbidden_definition | top_p | readout | 101 | subway | clean_synonym | True |  | transportation |  as a an transportation underground system system consisting that of includes trains |
| glm4 | 24-26-28 | forbidden_definition | top_p | readout | 101 | helicopter | clean_synonym | True |  | craft |  a craft that is capable of vertical take-off and landing, |
| glm4 | 24-26-28 | forbidden_definition | top_p | readout | 101 | tractor | synonym_with_label_violation | False | vehicles | machine,carry | :  A machine used to pull vehicles or carry heavy loads. |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_full | 101 | tram | clean_synonym | True |  | transportation | :A:   meansA of: public A transportation transportation system |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_full | 101 | subway | other | False |  |  |  as a an:   extensivea underground network network of of tunnels |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_full | 101 | helicopter | clean_synonym | True |  | aircraft |  an aircraft that employs vertical lift capabilities, using fans or rot |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_perp | 101 | tram | clean_synonym | True |  | transportation | :Tram is a means of transportation that runs along tracks |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_perp | 101 | subway | other | False |  |  |  as anystem extensive of network tracks tunnels that that carries carries |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_perp | 101 | helicopter | clean_synonym | True |  | aircraft | :A:   rotatingAn aircraft:   that consisting has of a |
| glm4 | 24-26-28 | forbidden_definition | top_p | random_perp | 101 | tractor | wrong_synonym | False |  |  |  a large agricultural implement designed to harness mechanical power for plowing |
| glm4 | 24-26-28 | forbidden_definition | temperature | baseline | 101 | tram | clean_synonym | True |  | transportation | :Tram is a form of transportation that operates along tracks |
| glm4 | 24-26-28 | forbidden_definition | temperature | baseline | 101 | subway | clean_synonym | True |  | transport |  as a an mode underground of transport public that. carriesSub |
| glm4 | 24-26-28 | forbidden_definition | temperature | baseline | 101 | helicopter | clean_synonym | True |  | craft |  an aerial craft that can take off and land vertically, using |
| glm4 | 24-26-28 | forbidden_definition | temperature | baseline | 101 | tractor | wrong_synonym | False |  |  | :  A device used in agriculture that performs various tasks, such |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_parallel | 101 | tram | other | False |  |  | :A:   vehicleA or: transit AA (A |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_parallel | 101 | subway | synonym_with_label_violation | False | vehicles | transport |  as a an transport underground system network with. vehicles vehicles. |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_parallel | 101 | helicopter | other | False |  |  |  a form of aviation that utilizes a rotating... wing system to |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_parallel | 101 | tractor | synonym_with_label_violation | False | vehicles | machine,carry | :  A machine used to pull vehicles or carry heavy loads. |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_full | 101 | tram | clean_synonym | True |  | transport | :A:   publicA transport:  AA system tram system |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_full | 101 | subway | clean_synonym | True |  | transportation |  as a a shared large transportation a system that that that that |
| glm4 | 24-26-28 | forbidden_definition | temperature | residual_full | 101 | helicopter | other | False |  |  | :  A flying platform with a rotating wing(s), system) |

| deepseek7b | 16-18-20 | forbidden_definition | top_p | baseline | 101 | tram | other | False |  |  | ...   So, I need to figure out a word that |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | baseline | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | baseline | 101 | helicopter | clean_synonym | True |  | airplane | ... The options are: A) a kind of airplane that |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | baseline | 101 | tractor | other | False |  |  |  what?  Okay, so I have to figure out what a |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_parallel | 101 | tram | generic_only | False |  |  |  which of these: object, concept, thing.  So, |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_parallel | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_parallel | 101 | helicopter | other | False |  |  | ... Hmm, okay, so I need to think about what |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_parallel | 101 | tractor | other | False |  |  |  what?  Okay, so I have to figure out what a |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_full | 101 | tram | other | False |  |  | ... (needs to fit into the sentence: Tram is |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_full | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_full | 101 | helicopter | other | False |  |  | ... Hmm, okay, so I need to think about what |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_full | 101 | tractor | other | False |  |  |  what?  Okay, so I have to figure out what a |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_perp | 101 | tram | label_violation | False | vehicle,vehicles |  | ... (must not use the words: vehicle, vehicles, |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_perp | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels, powered |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_perp | 101 | helicopter | other | False |  |  | ... I need to find a synonym or rephrase the sentence |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | residual_perp | 101 | tractor | other | False |  |  |  what?  Okay, so I have to figure out what a |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | readout | 101 | tram | generic_only | False |  |  |  which of these: object, concept, thing.  So, |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | readout | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | readout | 101 | helicopter | other | False |  |  | ... Hmm, okay, so I need to think about what |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | readout | 101 | tractor | other | False |  |  |  what?  Okay, so I have to figure out what a |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_full | 101 | tram | other | False |  |  | ...   So, I need to figure out a word that |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_full | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels, powered |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_full | 101 | helicopter | other | False |  |  | ... Let me think about the characteristics of a helicopter. It |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_full | 101 | tractor | other | False |  |  |  what?  Okay, so I have this question where I need |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_perp | 101 | tram | other | False |  |  | ...   So, I need to figure out a word that |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_perp | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels, powered |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_perp | 101 | helicopter | other | False |  |  | ... I need to find a synonym or rephrase the sentence |
| deepseek7b | 16-18-20 | forbidden_definition | top_p | random_perp | 101 | tractor | other | False |  |  |  what?  Okay, so I have this question where I need |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | baseline | 101 | tram | synonym_with_label_violation | False | vehicle | transportation | ?  Tram is best described as a public transportation vehicle designed |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | baseline | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | baseline | 101 | helicopter | other | False |  |  | : Please reason step by step, and keep your explanation natural |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | baseline | 101 | tractor | other | False |  |  |  what?  Okay, I need to figure out what tractor is |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | residual_parallel | 101 | tram | other | False |  |  | :  A. a place where people go for work. B |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | residual_parallel | 101 | subway | other | False |  |  |  a system that transports people through a network of tunnels. The |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | residual_parallel | 101 | helicopter | other | False |  |  | : Please reason step by step, and do not guess.  |
| deepseek7b | 16-18-20 | forbidden_definition | temperature | residual_parallel | 101 | tractor | other | False |  |  |  what?  Okay, I need to figure out what tractor is |

