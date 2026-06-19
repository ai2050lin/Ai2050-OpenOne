# Phase547 Label-Forbidden Paraphrase Gate Summary

## qwen3

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['forbidden_natural_qa', 'forbidden_definition', 'forbidden_sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout'], windows={'10-12-14': [10, 12, 14]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 10-12-14 | 0.31 | 0.31 | +0.00 | 0.12 | 0.12 | +0.00 | 0.23 | 0.08 | -0.04 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.31 | 0.29 | -0.02 | 0.12 | 0.15 | +0.02 | 0.23 | 0.02 | -0.08 | -0.04 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.31 | 0.35 | +0.04 | 0.12 | 0.23 | +0.10 | 0.08 | 0.10 | 0.04 | +0.08 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.31 | 0.33 | +0.02 | 0.12 | 0.23 | +0.10 | 0.06 | 0.08 | 0.04 | +0.08 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.31 | 0.29 | -0.02 | 0.12 | 0.15 | +0.02 | 0.23 | 0.02 | -0.08 | -0.04 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 10-12-14 | 0.40 | 0.40 | +0.00 | 0.04 | 0.04 | +0.00 | 0.19 | 0.10 | 0.17 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.40 | 0.27 | -0.12 | 0.04 | 0.12 | +0.08 | 0.23 | 0.04 | -0.08 | -0.25 | negative |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.40 | 0.42 | +0.02 | 0.04 | 0.12 | +0.08 | 0.06 | 0.08 | 0.23 | +0.06 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.40 | 0.38 | -0.02 | 0.04 | 0.12 | +0.08 | 0.08 | 0.10 | 0.17 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.40 | 0.27 | -0.12 | 0.04 | 0.12 | +0.08 | 0.23 | 0.04 | -0.08 | -0.25 | negative |
| qwen3 | vehicle_tool | forbidden_definition | top_p | baseline | 10-12-14 | 0.04 | 0.04 | +0.00 | 0.04 | 0.04 | +0.00 | 0.10 | 0.27 | -0.10 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.04 | +0.00 | 0.08 | 0.23 | 0.08 | +0.19 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_full | 10-12-14 | 0.04 | 0.23 | +0.19 | 0.04 | 0.04 | +0.00 | 0.04 | 0.25 | 0.15 | +0.25 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.00 | -0.04 | 0.08 | 0.29 | 0.12 | +0.23 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | readout | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.04 | +0.00 | 0.08 | 0.23 | 0.08 | +0.19 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | temperature | baseline | 10-12-14 | 0.19 | 0.19 | +0.00 | 0.02 | 0.02 | +0.00 | 0.08 | 0.19 | 0.08 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.19 | 0.12 | -0.06 | 0.02 | 0.04 | +0.02 | 0.08 | 0.19 | 0.00 | -0.08 | flat |
| qwen3 | vehicle_tool | forbidden_definition | temperature | residual_full | 10-12-14 | 0.19 | 0.27 | +0.08 | 0.02 | 0.04 | +0.02 | 0.04 | 0.21 | 0.19 | +0.10 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.19 | 0.23 | +0.04 | 0.02 | 0.00 | -0.02 | 0.04 | 0.25 | 0.19 | +0.10 | flat |
| qwen3 | vehicle_tool | forbidden_definition | temperature | readout | 10-12-14 | 0.19 | 0.12 | -0.06 | 0.02 | 0.04 | +0.02 | 0.08 | 0.19 | 0.00 | -0.08 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | baseline | 10-12-14 | 0.52 | 0.52 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.52 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | residual_parallel | 10-12-14 | 0.52 | 0.50 | -0.02 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | 0.46 | -0.06 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.52 | 0.67 | +0.15 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | 0.62 | +0.10 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.52 | 0.67 | +0.15 | 0.00 | 0.02 | +0.02 | 0.00 | 0.02 | 0.65 | +0.12 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | readout | 10-12-14 | 0.52 | 0.50 | -0.02 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | 0.46 | -0.06 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | baseline | 10-12-14 | 0.54 | 0.54 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.06 | 0.52 | +0.00 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.54 | 0.50 | -0.04 | 0.02 | 0.04 | +0.02 | 0.00 | 0.08 | 0.46 | -0.06 | flat |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.54 | 0.62 | +0.08 | 0.02 | 0.08 | +0.06 | 0.02 | 0.04 | 0.52 | +0.00 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.54 | 0.69 | +0.15 | 0.02 | 0.06 | +0.04 | 0.02 | 0.00 | 0.60 | +0.08 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.54 | 0.50 | -0.04 | 0.02 | 0.04 | +0.02 | 0.00 | 0.08 | 0.46 | -0.06 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | baseline | 10-12-14 | 0.73 | 0.73 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.10 | 0.73 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.73 | 0.75 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.75 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.73 | 0.62 | -0.10 | 0.00 | 0.02 | +0.02 | 0.02 | 0.12 | 0.58 | -0.15 | negative |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.73 | 0.62 | -0.10 | 0.00 | 0.02 | +0.02 | 0.00 | 0.12 | 0.60 | -0.12 | negative |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.73 | 0.75 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.75 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | baseline | 10-12-14 | 0.60 | 0.60 | +0.00 | 0.08 | 0.08 | +0.00 | 0.00 | 0.12 | 0.52 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.60 | 0.67 | +0.06 | 0.08 | 0.06 | -0.02 | 0.00 | 0.10 | 0.60 | +0.08 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.60 | 0.54 | -0.06 | 0.08 | 0.06 | -0.02 | 0.02 | 0.23 | 0.46 | -0.06 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.60 | 0.60 | +0.00 | 0.08 | 0.04 | -0.04 | 0.00 | 0.19 | 0.56 | +0.04 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.60 | 0.67 | +0.06 | 0.08 | 0.06 | -0.02 | 0.00 | 0.10 | 0.60 | +0.08 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | top_p | baseline | 10-12-14 | 0.40 | 0.40 | +0.00 | 0.04 | 0.04 | +0.00 | 0.00 | 0.17 | 0.35 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.40 | 0.38 | -0.02 | 0.04 | 0.04 | +0.00 | 0.00 | 0.15 | 0.33 | -0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | top_p | residual_full | 10-12-14 | 0.40 | 0.29 | -0.10 | 0.04 | 0.02 | -0.02 | 0.00 | 0.25 | 0.27 | -0.08 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.40 | 0.31 | -0.08 | 0.04 | 0.00 | -0.04 | 0.00 | 0.23 | 0.31 | -0.04 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | top_p | readout | 10-12-14 | 0.40 | 0.38 | -0.02 | 0.04 | 0.04 | +0.00 | 0.00 | 0.15 | 0.33 | -0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | temperature | baseline | 10-12-14 | 0.33 | 0.33 | +0.00 | 0.04 | 0.04 | +0.00 | 0.00 | 0.23 | 0.29 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.33 | 0.25 | -0.08 | 0.04 | 0.08 | +0.04 | 0.00 | 0.25 | 0.17 | -0.12 | negative |
| qwen3 | fruit_vegetable | forbidden_definition | temperature | residual_full | 10-12-14 | 0.33 | 0.31 | -0.02 | 0.04 | 0.08 | +0.04 | 0.00 | 0.23 | 0.23 | -0.06 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.33 | 0.31 | -0.02 | 0.04 | 0.04 | +0.00 | 0.00 | 0.25 | 0.27 | -0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_definition | temperature | readout | 10-12-14 | 0.33 | 0.25 | -0.08 | 0.04 | 0.08 | +0.04 | 0.00 | 0.25 | 0.17 | -0.12 | negative |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | top_p | baseline | 10-12-14 | 0.15 | 0.15 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.15 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_parallel | 10-12-14 | 0.15 | 0.17 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.17 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.15 | 0.12 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.12 | -0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.15 | 0.17 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.17 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | top_p | readout | 10-12-14 | 0.15 | 0.17 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.17 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | temperature | baseline | 10-12-14 | 0.15 | 0.15 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.15 | +0.00 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.15 | 0.08 | -0.06 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.08 | -0.06 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.15 | 0.10 | -0.04 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.10 | -0.04 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.15 | 0.15 | +0.00 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.12 | -0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.15 | 0.08 | -0.06 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.08 | -0.06 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | top_p | baseline | 10-12-14 | 0.90 | 0.90 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.90 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.90 | 0.92 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.92 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.90 | 0.88 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.10 | 0.88 | -0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.90 | 0.88 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.10 | 0.88 | -0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.90 | 0.92 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.92 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | baseline | 10-12-14 | 0.88 | 0.88 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.06 | 0.88 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.90 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.90 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.88 | 0.92 | +0.04 | 0.00 | 0.00 | +0.00 | 0.00 | 0.06 | 0.92 | +0.04 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.90 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_definition | top_p | baseline | 10-12-14 | 0.44 | 0.44 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.10 | 0.44 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.44 | 0.42 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.12 | 0.42 | -0.02 | flat |
| qwen3 | animal_tool | forbidden_definition | top_p | residual_full | 10-12-14 | 0.44 | 0.42 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.42 | -0.02 | flat |
| qwen3 | animal_tool | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.44 | 0.46 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.46 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_definition | top_p | readout | 10-12-14 | 0.44 | 0.42 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.12 | 0.42 | -0.02 | flat |
| qwen3 | animal_tool | forbidden_definition | temperature | baseline | 10-12-14 | 0.56 | 0.56 | +0.00 | 0.00 | 0.00 | +0.00 | 0.02 | 0.08 | 0.54 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.56 | 0.46 | -0.10 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.46 | -0.08 | flat |
| qwen3 | animal_tool | forbidden_definition | temperature | residual_full | 10-12-14 | 0.56 | 0.35 | -0.21 | 0.00 | 0.02 | +0.02 | 0.00 | 0.08 | 0.33 | -0.21 | negative |
| qwen3 | animal_tool | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.56 | 0.42 | -0.15 | 0.00 | 0.02 | +0.02 | 0.00 | 0.12 | 0.40 | -0.15 | negative |
| qwen3 | animal_tool | forbidden_definition | temperature | readout | 10-12-14 | 0.56 | 0.46 | -0.10 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.46 | -0.08 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | top_p | baseline | 10-12-14 | 0.02 | 0.02 | +0.00 | 0.00 | 0.00 | +0.00 | 0.04 | 0.00 | -0.02 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | top_p | residual_parallel | 10-12-14 | 0.02 | 0.00 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.00 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.02 | 0.04 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.06 | 0.04 | +0.06 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.02 | 0.00 | -0.02 | 0.00 | 0.06 | +0.06 | 0.00 | 0.04 | -0.06 | -0.04 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | top_p | readout | 10-12-14 | 0.02 | 0.00 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.00 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | temperature | baseline | 10-12-14 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.00 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.02 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.02 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.00 | +0.00 | flat |
| qwen3 | animal_tool | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.02 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | baseline | 10-12-14 | 0.75 | 0.75 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.75 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.75 | 0.77 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.77 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.75 | 0.75 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.12 | 0.75 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.75 | 0.77 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.77 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.75 | 0.77 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.77 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | temperature | baseline | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.19 | 0.65 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.19 | 0.65 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.67 | 0.69 | +0.02 | 0.02 | 0.02 | +0.00 | 0.00 | 0.19 | 0.67 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.17 | 0.65 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.67 | 0.67 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.19 | 0.65 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | baseline | 10-12-14 | 0.38 | 0.38 | +0.00 | 0.04 | 0.04 | +0.00 | 0.02 | 0.23 | 0.31 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.02 | 0.23 | 0.40 | +0.08 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_full | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.00 | 0.17 | 0.42 | +0.10 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.38 | 0.50 | +0.12 | 0.04 | 0.00 | -0.04 | 0.00 | 0.10 | 0.50 | +0.19 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | top_p | readout | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.02 | 0.23 | 0.40 | +0.08 | flat |
| qwen3 | fruit_tool | forbidden_definition | temperature | baseline | 10-12-14 | 0.35 | 0.35 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.27 | 0.35 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_definition | temperature | residual_parallel | 10-12-14 | 0.35 | 0.35 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.21 | 0.35 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_definition | temperature | residual_full | 10-12-14 | 0.35 | 0.44 | +0.08 | 0.00 | 0.00 | +0.00 | 0.00 | 0.12 | 0.44 | +0.08 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.35 | 0.44 | +0.08 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.44 | +0.08 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | temperature | readout | 10-12-14 | 0.35 | 0.35 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.21 | 0.35 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | top_p | baseline | 10-12-14 | 0.10 | 0.10 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.10 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | top_p | residual_parallel | 10-12-14 | 0.10 | 0.12 | +0.02 | 0.00 | 0.02 | +0.02 | 0.00 | 0.02 | 0.10 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.10 | 0.12 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.12 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.10 | 0.12 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.12 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | top_p | readout | 10-12-14 | 0.10 | 0.12 | +0.02 | 0.00 | 0.02 | +0.02 | 0.00 | 0.02 | 0.10 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | temperature | baseline | 10-12-14 | 0.15 | 0.15 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.15 | +0.00 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | temperature | residual_parallel | 10-12-14 | 0.15 | 0.10 | -0.04 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.08 | -0.06 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.15 | 0.10 | -0.04 | 0.00 | 0.02 | +0.02 | 0.00 | 0.06 | 0.08 | -0.06 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.15 | 0.12 | -0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.06 | 0.12 | -0.02 | flat |
| qwen3 | fruit_tool | forbidden_sentence_completion | temperature | readout | 10-12-14 | 0.15 | 0.10 | -0.04 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.08 | -0.06 | flat |

## glm4

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['forbidden_natural_qa', 'forbidden_definition', 'forbidden_sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout'], windows={'24-26-28': [24, 26, 28]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 24-26-28 | 0.38 | 0.38 | +0.00 | 0.00 | 0.00 | +0.00 | 0.19 | 0.33 | 0.19 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.38 | 0.27 | -0.10 | 0.00 | 0.46 | +0.46 | 0.06 | 0.02 | -0.25 | -0.44 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.38 | 0.54 | +0.17 | 0.00 | 0.04 | +0.04 | 0.06 | 0.12 | 0.44 | +0.25 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.38 | 0.50 | +0.12 | 0.00 | 0.02 | +0.02 | 0.10 | 0.15 | 0.38 | +0.19 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.38 | 0.27 | -0.10 | 0.00 | 0.46 | +0.46 | 0.06 | 0.02 | -0.25 | -0.44 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.02 | 0.02 | +0.00 | 0.23 | 0.25 | 0.06 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.02 | 0.44 | +0.42 | 0.04 | 0.00 | -0.17 | -0.23 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.31 | 0.50 | +0.19 | 0.02 | 0.08 | +0.06 | 0.04 | 0.10 | 0.38 | +0.31 | synonym_with_label_leak |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.31 | 0.52 | +0.21 | 0.02 | 0.06 | +0.04 | 0.08 | 0.12 | 0.38 | +0.31 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.02 | 0.44 | +0.42 | 0.04 | 0.00 | -0.17 | -0.23 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_definition | top_p | baseline | 24-26-28 | 0.29 | 0.29 | +0.00 | 0.02 | 0.02 | +0.00 | 0.06 | 0.08 | 0.21 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.29 | 0.40 | +0.10 | 0.02 | 0.23 | +0.21 | 0.02 | 0.00 | 0.15 | -0.06 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_full | 24-26-28 | 0.29 | 0.54 | +0.25 | 0.02 | 0.02 | +0.00 | 0.02 | 0.02 | 0.50 | +0.29 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.29 | 0.56 | +0.27 | 0.02 | 0.02 | +0.00 | 0.00 | 0.04 | 0.54 | +0.33 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | readout | 24-26-28 | 0.29 | 0.40 | +0.10 | 0.02 | 0.23 | +0.21 | 0.02 | 0.00 | 0.15 | -0.06 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_definition | temperature | baseline | 24-26-28 | 0.25 | 0.25 | +0.00 | 0.00 | 0.00 | +0.00 | 0.15 | 0.04 | 0.10 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.25 | 0.31 | +0.06 | 0.00 | 0.17 | +0.17 | 0.02 | 0.00 | 0.13 | +0.02 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_full | 24-26-28 | 0.25 | 0.44 | +0.19 | 0.00 | 0.02 | +0.02 | 0.02 | 0.00 | 0.40 | +0.29 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.25 | 0.44 | +0.19 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.44 | +0.33 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | temperature | readout | 24-26-28 | 0.25 | 0.31 | +0.06 | 0.00 | 0.17 | +0.17 | 0.02 | 0.00 | 0.13 | +0.02 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | baseline | 24-26-28 | 0.33 | 0.33 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.33 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.33 | 0.25 | -0.08 | 0.00 | 0.06 | +0.06 | 0.00 | 0.02 | 0.19 | -0.15 | negative |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.33 | 0.52 | +0.19 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.52 | +0.19 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.33 | 0.54 | +0.21 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.54 | +0.21 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.33 | 0.25 | -0.08 | 0.00 | 0.06 | +0.06 | 0.00 | 0.02 | 0.19 | -0.15 | negative |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | baseline | 24-26-28 | 0.31 | 0.31 | +0.00 | 0.00 | 0.00 | +0.00 | 0.02 | 0.04 | 0.29 | +0.00 | flat |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.00 | 0.17 | +0.17 | 0.00 | 0.00 | 0.23 | -0.06 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.31 | 0.56 | +0.25 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.56 | +0.27 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.31 | 0.65 | +0.33 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.65 | +0.35 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.00 | 0.17 | +0.17 | 0.00 | 0.00 | 0.23 | -0.06 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | baseline | 24-26-28 | 0.08 | 0.08 | +0.00 | 0.48 | 0.48 | +0.00 | 0.02 | 0.19 | -0.42 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.08 | 0.00 | -0.08 | 0.48 | 0.88 | +0.40 | 0.00 | 0.08 | -0.88 | -0.46 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.08 | 0.10 | +0.02 | 0.48 | 0.62 | +0.15 | 0.00 | 0.08 | -0.52 | -0.10 | negative |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.08 | 0.27 | +0.19 | 0.48 | 0.48 | +0.00 | 0.00 | 0.08 | -0.21 | +0.21 | clean_paraphrase_positive |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.08 | 0.00 | -0.08 | 0.48 | 0.88 | +0.40 | 0.00 | 0.08 | -0.88 | -0.46 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | baseline | 24-26-28 | 0.19 | 0.19 | +0.00 | 0.38 | 0.38 | +0.00 | 0.02 | 0.17 | -0.21 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.19 | 0.00 | -0.19 | 0.38 | 0.81 | +0.44 | 0.00 | 0.04 | -0.81 | -0.60 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.19 | 0.19 | +0.00 | 0.38 | 0.56 | +0.19 | 0.00 | 0.04 | -0.38 | -0.17 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.19 | 0.25 | +0.06 | 0.38 | 0.46 | +0.08 | 0.00 | 0.06 | -0.21 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.19 | 0.00 | -0.19 | 0.38 | 0.81 | +0.44 | 0.00 | 0.04 | -0.81 | -0.60 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_definition | top_p | baseline | 24-26-28 | 0.29 | 0.29 | +0.00 | 0.12 | 0.12 | +0.00 | 0.00 | 0.15 | 0.17 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.29 | 0.12 | -0.17 | 0.12 | 0.40 | +0.27 | 0.00 | 0.02 | -0.27 | -0.44 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_definition | top_p | residual_full | 24-26-28 | 0.29 | 0.35 | +0.06 | 0.12 | 0.10 | -0.02 | 0.00 | 0.04 | 0.25 | +0.08 | flat |
| glm4 | fruit_vegetable | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.29 | 0.42 | +0.12 | 0.12 | 0.10 | -0.02 | 0.00 | 0.02 | 0.31 | +0.15 | weak_clean_positive |
| glm4 | fruit_vegetable | forbidden_definition | top_p | readout | 24-26-28 | 0.29 | 0.12 | -0.17 | 0.12 | 0.40 | +0.27 | 0.00 | 0.02 | -0.27 | -0.44 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_definition | temperature | baseline | 24-26-28 | 0.27 | 0.27 | +0.00 | 0.19 | 0.19 | +0.00 | 0.00 | 0.08 | 0.08 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.27 | 0.17 | -0.10 | 0.19 | 0.38 | +0.19 | 0.00 | 0.10 | -0.21 | -0.29 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_definition | temperature | residual_full | 24-26-28 | 0.27 | 0.35 | +0.08 | 0.19 | 0.08 | -0.10 | 0.00 | 0.02 | 0.27 | +0.19 | weak_clean_positive |
| glm4 | fruit_vegetable | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.27 | 0.44 | +0.17 | 0.19 | 0.10 | -0.08 | 0.00 | 0.02 | 0.33 | +0.25 | clean_paraphrase_positive |
| glm4 | fruit_vegetable | forbidden_definition | temperature | readout | 24-26-28 | 0.27 | 0.17 | -0.10 | 0.19 | 0.38 | +0.19 | 0.00 | 0.10 | -0.21 | -0.29 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_sentence_completion | top_p | baseline | 24-26-28 | 0.15 | 0.15 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.15 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.15 | 0.02 | -0.12 | 0.00 | 0.35 | +0.35 | 0.00 | 0.00 | -0.33 | -0.48 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.15 | 0.04 | -0.10 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.04 | -0.10 | negative |
| glm4 | fruit_vegetable | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.15 | 0.06 | -0.08 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.04 | -0.10 | negative |
| glm4 | fruit_vegetable | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.15 | 0.02 | -0.12 | 0.00 | 0.35 | +0.35 | 0.00 | 0.00 | -0.33 | -0.48 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_sentence_completion | temperature | baseline | 24-26-28 | 0.25 | 0.25 | +0.00 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.25 | +0.00 | flat |
| glm4 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.25 | 0.06 | -0.19 | 0.00 | 0.38 | +0.38 | 0.00 | 0.00 | -0.31 | -0.56 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.25 | 0.12 | -0.12 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.10 | -0.15 | negative |
| glm4 | fruit_vegetable | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.25 | 0.10 | -0.15 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.10 | -0.15 | negative |
| glm4 | fruit_vegetable | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.25 | 0.06 | -0.19 | 0.00 | 0.38 | +0.38 | 0.00 | 0.00 | -0.31 | -0.56 | label_gate_leak |
| glm4 | animal_tool | forbidden_natural_qa | top_p | baseline | 24-26-28 | 0.35 | 0.35 | +0.00 | 0.12 | 0.12 | +0.00 | 0.02 | 0.04 | 0.21 | +0.00 | flat |
| glm4 | animal_tool | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.35 | 0.19 | -0.17 | 0.12 | 0.60 | +0.48 | 0.00 | 0.00 | -0.42 | -0.62 | label_gate_leak |
| glm4 | animal_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.35 | 0.44 | +0.08 | 0.12 | 0.02 | -0.10 | 0.00 | 0.02 | 0.42 | +0.21 | weak_clean_positive |
| glm4 | animal_tool | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.35 | 0.38 | +0.02 | 0.12 | 0.02 | -0.10 | 0.00 | 0.02 | 0.35 | +0.15 | flat |
| glm4 | animal_tool | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.35 | 0.19 | -0.17 | 0.12 | 0.60 | +0.48 | 0.00 | 0.00 | -0.42 | -0.62 | label_gate_leak |
| glm4 | animal_tool | forbidden_natural_qa | temperature | baseline | 24-26-28 | 0.40 | 0.40 | +0.00 | 0.04 | 0.04 | +0.00 | 0.02 | 0.02 | 0.33 | +0.00 | flat |
| glm4 | animal_tool | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.40 | 0.23 | -0.17 | 0.04 | 0.48 | +0.44 | 0.00 | 0.02 | -0.25 | -0.58 | label_gate_leak |
| glm4 | animal_tool | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.40 | 0.38 | -0.02 | 0.04 | 0.04 | +0.00 | 0.00 | 0.02 | 0.33 | +0.00 | flat |
| glm4 | animal_tool | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.40 | 0.38 | -0.02 | 0.04 | 0.02 | -0.02 | 0.00 | 0.00 | 0.35 | +0.02 | flat |
| glm4 | animal_tool | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.40 | 0.23 | -0.17 | 0.04 | 0.48 | +0.44 | 0.00 | 0.02 | -0.25 | -0.58 | label_gate_leak |
| glm4 | animal_tool | forbidden_definition | top_p | baseline | 24-26-28 | 0.54 | 0.54 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.02 | 0.52 | +0.00 | flat |
| glm4 | animal_tool | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.54 | 0.38 | -0.17 | 0.02 | 0.38 | +0.35 | 0.00 | 0.00 | 0.00 | -0.52 | label_gate_leak |
| glm4 | animal_tool | forbidden_definition | top_p | residual_full | 24-26-28 | 0.54 | 0.27 | -0.27 | 0.02 | 0.00 | -0.02 | 0.00 | 0.00 | 0.27 | -0.25 | negative |
| glm4 | animal_tool | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.54 | 0.25 | -0.29 | 0.02 | 0.00 | -0.02 | 0.00 | 0.00 | 0.25 | -0.27 | negative |
| glm4 | animal_tool | forbidden_definition | top_p | readout | 24-26-28 | 0.54 | 0.38 | -0.17 | 0.02 | 0.38 | +0.35 | 0.00 | 0.00 | 0.00 | -0.52 | label_gate_leak |
| glm4 | animal_tool | forbidden_definition | temperature | baseline | 24-26-28 | 0.38 | 0.38 | +0.00 | 0.04 | 0.04 | +0.00 | 0.00 | 0.02 | 0.33 | +0.00 | flat |
| glm4 | animal_tool | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.38 | 0.33 | -0.04 | 0.04 | 0.42 | +0.38 | 0.00 | 0.00 | -0.08 | -0.42 | label_gate_leak |
| glm4 | animal_tool | forbidden_definition | temperature | residual_full | 24-26-28 | 0.38 | 0.27 | -0.10 | 0.04 | 0.02 | -0.02 | 0.00 | 0.00 | 0.25 | -0.08 | flat |
| glm4 | animal_tool | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.38 | 0.23 | -0.15 | 0.04 | 0.00 | -0.04 | 0.00 | 0.00 | 0.23 | -0.10 | negative |
| glm4 | animal_tool | forbidden_definition | temperature | readout | 24-26-28 | 0.38 | 0.33 | -0.04 | 0.04 | 0.42 | +0.38 | 0.00 | 0.00 | -0.08 | -0.42 | label_gate_leak |
| glm4 | animal_tool | forbidden_sentence_completion | top_p | baseline | 24-26-28 | 0.06 | 0.06 | +0.00 | 0.00 | 0.00 | +0.00 | 0.02 | 0.00 | 0.04 | +0.00 | flat |
| glm4 | animal_tool | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.06 | 0.02 | -0.04 | 0.00 | 0.52 | +0.52 | 0.00 | 0.02 | -0.50 | -0.54 | label_gate_leak |
| glm4 | animal_tool | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.06 | 0.02 | -0.04 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | -0.02 | -0.06 | flat |
| glm4 | animal_tool | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.06 | 0.02 | -0.04 | 0.00 | 0.02 | +0.02 | 0.00 | 0.00 | 0.00 | -0.04 | flat |
| glm4 | animal_tool | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.06 | 0.02 | -0.04 | 0.00 | 0.52 | +0.52 | 0.00 | 0.02 | -0.50 | -0.54 | label_gate_leak |
| glm4 | animal_tool | forbidden_sentence_completion | temperature | baseline | 24-26-28 | 0.06 | 0.06 | +0.00 | 0.00 | 0.00 | +0.00 | 0.02 | 0.02 | 0.04 | +0.00 | flat |
| glm4 | animal_tool | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.06 | 0.00 | -0.06 | 0.00 | 0.60 | +0.60 | 0.00 | 0.00 | -0.60 | -0.65 | label_gate_leak |
| glm4 | animal_tool | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.06 | 0.02 | -0.04 | 0.00 | 0.12 | +0.12 | 0.00 | 0.00 | -0.10 | -0.15 | negative |
| glm4 | animal_tool | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.06 | 0.08 | +0.02 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | 0.04 | -0.00 | flat |
| glm4 | animal_tool | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.06 | 0.00 | -0.06 | 0.00 | 0.60 | +0.60 | 0.00 | 0.00 | -0.60 | -0.65 | label_gate_leak |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | baseline | 24-26-28 | 0.08 | 0.08 | +0.00 | 0.50 | 0.50 | +0.00 | 0.00 | 0.23 | -0.42 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | residual_parallel | 24-26-28 | 0.08 | 0.02 | -0.06 | 0.50 | 0.75 | +0.25 | 0.02 | 0.15 | -0.75 | -0.33 | label_gate_leak |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.08 | 0.19 | +0.10 | 0.50 | 0.56 | +0.06 | 0.00 | 0.10 | -0.38 | +0.04 | weak_clean_positive |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.08 | 0.27 | +0.19 | 0.50 | 0.54 | +0.04 | 0.00 | 0.06 | -0.27 | +0.15 | clean_paraphrase_positive |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | readout | 24-26-28 | 0.08 | 0.02 | -0.06 | 0.50 | 0.75 | +0.25 | 0.02 | 0.15 | -0.75 | -0.33 | label_gate_leak |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | baseline | 24-26-28 | 0.15 | 0.15 | +0.00 | 0.46 | 0.46 | +0.00 | 0.00 | 0.15 | -0.31 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | residual_parallel | 24-26-28 | 0.15 | 0.08 | -0.06 | 0.46 | 0.67 | +0.21 | 0.02 | 0.19 | -0.60 | -0.29 | label_gate_leak |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.15 | 0.19 | +0.04 | 0.46 | 0.46 | +0.00 | 0.00 | 0.04 | -0.27 | +0.04 | flat |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.15 | 0.12 | -0.02 | 0.46 | 0.50 | +0.04 | 0.00 | 0.04 | -0.38 | -0.06 | flat |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | readout | 24-26-28 | 0.15 | 0.08 | -0.06 | 0.46 | 0.67 | +0.21 | 0.02 | 0.19 | -0.60 | -0.29 | label_gate_leak |
| glm4 | fruit_tool | forbidden_definition | top_p | baseline | 24-26-28 | 0.38 | 0.38 | +0.00 | 0.04 | 0.04 | +0.00 | 0.00 | 0.08 | 0.33 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.38 | 0.08 | -0.29 | 0.04 | 0.21 | +0.17 | 0.00 | 0.12 | -0.12 | -0.46 | label_gate_leak |
| glm4 | fruit_tool | forbidden_definition | top_p | residual_full | 24-26-28 | 0.38 | 0.33 | -0.04 | 0.04 | 0.10 | +0.06 | 0.00 | 0.00 | 0.23 | -0.10 | negative |
| glm4 | fruit_tool | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.38 | 0.33 | -0.04 | 0.04 | 0.08 | +0.04 | 0.00 | 0.02 | 0.25 | -0.08 | flat |
| glm4 | fruit_tool | forbidden_definition | top_p | readout | 24-26-28 | 0.38 | 0.08 | -0.29 | 0.04 | 0.21 | +0.17 | 0.00 | 0.12 | -0.12 | -0.46 | label_gate_leak |
| glm4 | fruit_tool | forbidden_definition | temperature | baseline | 24-26-28 | 0.29 | 0.29 | +0.00 | 0.06 | 0.06 | +0.00 | 0.00 | 0.12 | 0.23 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.29 | 0.10 | -0.19 | 0.06 | 0.27 | +0.21 | 0.00 | 0.08 | -0.17 | -0.40 | label_gate_leak |
| glm4 | fruit_tool | forbidden_definition | temperature | residual_full | 24-26-28 | 0.29 | 0.38 | +0.08 | 0.06 | 0.02 | -0.04 | 0.00 | 0.04 | 0.35 | +0.12 | weak_clean_positive |
| glm4 | fruit_tool | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.29 | 0.31 | +0.02 | 0.06 | 0.02 | -0.04 | 0.00 | 0.02 | 0.29 | +0.06 | flat |
| glm4 | fruit_tool | forbidden_definition | temperature | readout | 24-26-28 | 0.29 | 0.10 | -0.19 | 0.06 | 0.27 | +0.21 | 0.00 | 0.08 | -0.17 | -0.40 | label_gate_leak |
| glm4 | fruit_tool | forbidden_sentence_completion | top_p | baseline | 24-26-28 | 0.10 | 0.10 | +0.00 | 0.02 | 0.02 | +0.00 | 0.04 | 0.00 | 0.04 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | top_p | residual_parallel | 24-26-28 | 0.10 | 0.02 | -0.08 | 0.02 | 0.46 | +0.44 | 0.08 | 0.00 | -0.52 | -0.56 | label_gate_leak |
| glm4 | fruit_tool | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.10 | 0.12 | +0.02 | 0.02 | 0.02 | +0.00 | 0.02 | 0.00 | 0.08 | +0.04 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.10 | 0.08 | -0.02 | 0.02 | 0.04 | +0.02 | 0.02 | 0.00 | 0.02 | -0.02 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | top_p | readout | 24-26-28 | 0.10 | 0.02 | -0.08 | 0.02 | 0.46 | +0.44 | 0.08 | 0.00 | -0.52 | -0.56 | label_gate_leak |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | baseline | 24-26-28 | 0.12 | 0.12 | +0.00 | 0.02 | 0.02 | +0.00 | 0.04 | 0.00 | 0.06 | +0.00 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.12 | 0.04 | -0.08 | 0.02 | 0.35 | +0.33 | 0.04 | 0.00 | -0.35 | -0.42 | label_gate_leak |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.12 | 0.12 | +0.00 | 0.02 | 0.04 | +0.02 | 0.02 | 0.00 | 0.06 | -0.00 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.12 | 0.17 | +0.04 | 0.02 | 0.00 | -0.02 | 0.04 | 0.00 | 0.12 | +0.06 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.12 | 0.04 | -0.08 | 0.02 | 0.35 | +0.33 | 0.04 | 0.00 | -0.35 | -0.42 | label_gate_leak |

## deepseek7b

pairs=['vehicle_tool', 'fruit_vegetable', 'animal_tool', 'fruit_tool'], scaffolds=['forbidden_natural_qa', 'forbidden_definition', 'forbidden_sentence_completion'], modes=['top_p', 'temperature'], conditions=['baseline', 'residual_parallel', 'residual_full', 'residual_perp', 'readout'], windows={'16-18-20': [16, 18, 20]}, train_n=12, test_n=8, sample_seeds=[101, 103, 107, 109, 113, 127], alpha=6.0

| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | baseline | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.27 | 0.27 | +0.00 | 0.10 | 0.10 | -0.27 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.27 | 0.31 | +0.04 | 0.08 | 0.08 | -0.25 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.27 | 0.29 | +0.02 | 0.06 | 0.12 | -0.25 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.10 | 0.12 | +0.02 | 0.27 | 0.23 | -0.04 | 0.06 | 0.12 | -0.17 | +0.10 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.27 | 0.31 | +0.04 | 0.08 | 0.08 | -0.25 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | temperature | baseline | 16-18-20 | 0.17 | 0.17 | +0.00 | 0.29 | 0.29 | +0.00 | 0.02 | 0.21 | -0.15 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.17 | 0.12 | -0.04 | 0.29 | 0.27 | -0.02 | 0.10 | 0.19 | -0.25 | -0.10 | negative |
| deepseek7b | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.17 | 0.19 | +0.02 | 0.29 | 0.27 | -0.02 | 0.08 | 0.12 | -0.17 | -0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.17 | 0.19 | +0.02 | 0.29 | 0.25 | -0.04 | 0.08 | 0.19 | -0.15 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.17 | 0.12 | -0.04 | 0.29 | 0.27 | -0.02 | 0.10 | 0.19 | -0.25 | -0.10 | negative |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | baseline | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.17 | 0.17 | +0.00 | 0.00 | 0.06 | -0.04 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.19 | +0.02 | 0.00 | 0.06 | -0.02 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_full | 16-18-20 | 0.12 | 0.19 | +0.06 | 0.17 | 0.15 | -0.02 | 0.00 | 0.04 | 0.04 | +0.08 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.15 | -0.02 | 0.00 | 0.04 | 0.02 | +0.06 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | readout | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.19 | +0.02 | 0.00 | 0.06 | -0.02 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.19 | 0.19 | +0.00 | 0.00 | 0.10 | -0.04 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.19 | 0.06 | -0.12 | 0.00 | 0.10 | 0.08 | +0.12 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | residual_full | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.19 | 0.27 | +0.08 | 0.00 | 0.06 | -0.12 | -0.08 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.15 | 0.21 | +0.06 | 0.19 | 0.25 | +0.06 | 0.00 | 0.06 | -0.04 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | readout | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.19 | 0.06 | -0.12 | 0.00 | 0.10 | 0.08 | +0.12 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | top_p | baseline | 16-18-20 | 0.23 | 0.23 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.02 | 0.21 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.23 | 0.21 | -0.02 | 0.02 | 0.04 | +0.02 | 0.00 | 0.04 | 0.17 | -0.04 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.23 | 0.21 | -0.02 | 0.02 | 0.02 | +0.00 | 0.00 | 0.00 | 0.19 | -0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.23 | 0.15 | -0.08 | 0.02 | 0.00 | -0.02 | 0.00 | 0.02 | 0.15 | -0.06 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.23 | 0.21 | -0.02 | 0.02 | 0.04 | +0.02 | 0.00 | 0.04 | 0.17 | -0.04 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | temperature | baseline | 16-18-20 | 0.21 | 0.21 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.02 | 0.19 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.21 | 0.21 | +0.00 | 0.02 | 0.06 | +0.04 | 0.00 | 0.04 | 0.15 | -0.04 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | temperature | residual_full | 16-18-20 | 0.21 | 0.15 | -0.06 | 0.02 | 0.02 | +0.00 | 0.00 | 0.06 | 0.12 | -0.06 | flat |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.21 | 0.10 | -0.10 | 0.02 | 0.02 | +0.00 | 0.00 | 0.06 | 0.08 | -0.10 | negative |
| deepseek7b | vehicle_tool | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.21 | 0.21 | +0.00 | 0.02 | 0.06 | +0.04 | 0.00 | 0.04 | 0.15 | -0.04 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | baseline | 16-18-20 | 0.08 | 0.08 | +0.00 | 0.15 | 0.15 | +0.00 | 0.00 | 0.40 | -0.06 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.08 | 0.15 | +0.06 | 0.15 | 0.17 | +0.02 | 0.00 | 0.29 | -0.02 | +0.04 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.08 | 0.06 | -0.02 | 0.15 | 0.19 | +0.04 | 0.00 | 0.35 | -0.12 | -0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.08 | 0.08 | +0.00 | 0.15 | 0.19 | +0.04 | 0.02 | 0.27 | -0.12 | -0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.08 | 0.04 | -0.04 | 0.15 | 0.29 | +0.15 | 0.00 | 0.17 | -0.25 | -0.19 | negative |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | baseline | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.17 | 0.17 | +0.00 | 0.00 | 0.29 | -0.10 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.17 | 0.15 | -0.02 | 0.00 | 0.31 | -0.04 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.17 | 0.25 | +0.08 | 0.00 | 0.29 | -0.21 | -0.10 | negative |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.17 | 0.23 | +0.06 | 0.00 | 0.33 | -0.19 | -0.08 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.17 | 0.38 | +0.21 | 0.00 | 0.23 | -0.31 | -0.21 | label_gate_leak |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | baseline | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.40 | 0.40 | +0.00 | 0.00 | 0.02 | -0.33 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.40 | 0.40 | +0.00 | 0.00 | 0.08 | -0.35 | -0.02 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | residual_full | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.40 | 0.40 | +0.00 | 0.00 | 0.02 | -0.33 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.40 | 0.38 | -0.02 | 0.00 | 0.04 | -0.27 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | readout | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.40 | 0.46 | +0.06 | 0.00 | 0.06 | -0.42 | -0.08 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | baseline | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.23 | 0.23 | +0.00 | 0.00 | 0.04 | -0.10 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.12 | 0.10 | -0.02 | 0.23 | 0.15 | -0.08 | 0.00 | 0.04 | -0.04 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | residual_full | 16-18-20 | 0.12 | 0.08 | -0.04 | 0.23 | 0.21 | -0.02 | 0.00 | 0.04 | -0.12 | -0.02 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.12 | 0.19 | +0.06 | 0.23 | 0.23 | +0.00 | 0.00 | 0.02 | -0.04 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | readout | 16-18-20 | 0.12 | 0.15 | +0.02 | 0.23 | 0.21 | -0.02 | 0.00 | 0.04 | -0.06 | +0.04 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | baseline | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.06 | 0.06 | +0.00 | 0.00 | 0.00 | 0.06 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.06 | 0.06 | +0.00 | 0.00 | 0.02 | 0.06 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.06 | 0.04 | -0.02 | 0.00 | 0.00 | 0.12 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.12 | 0.08 | -0.04 | 0.06 | 0.08 | +0.02 | 0.00 | 0.00 | 0.00 | -0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.12 | 0.10 | -0.02 | 0.06 | 0.10 | +0.04 | 0.00 | 0.00 | 0.00 | -0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | temperature | baseline | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.08 | 0.08 | +0.00 | 0.00 | 0.02 | 0.02 | +0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.10 | 0.12 | +0.02 | 0.08 | 0.04 | -0.04 | 0.00 | 0.02 | 0.08 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | temperature | residual_full | 16-18-20 | 0.10 | 0.08 | -0.02 | 0.08 | 0.08 | +0.00 | 0.00 | 0.00 | 0.00 | -0.02 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.10 | 0.08 | -0.02 | 0.08 | 0.06 | -0.02 | 0.00 | 0.02 | 0.02 | -0.00 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.08 | 0.15 | +0.06 | 0.00 | 0.02 | -0.04 | -0.06 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | top_p | baseline | 16-18-20 | 0.48 | 0.48 | +0.00 | 0.10 | 0.10 | +0.00 | 0.00 | 0.15 | 0.38 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.48 | 0.48 | +0.00 | 0.10 | 0.10 | +0.00 | 0.00 | 0.10 | 0.38 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.48 | 0.48 | +0.00 | 0.10 | 0.15 | +0.04 | 0.00 | 0.04 | 0.33 | -0.04 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.48 | 0.48 | +0.00 | 0.10 | 0.15 | +0.04 | 0.00 | 0.08 | 0.33 | -0.04 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.48 | 0.48 | +0.00 | 0.10 | 0.10 | +0.00 | 0.00 | 0.10 | 0.38 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | baseline | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.12 | 0.12 | +0.00 | 0.00 | 0.02 | 0.23 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.12 | 0.19 | +0.06 | 0.02 | 0.06 | 0.15 | -0.08 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.35 | 0.40 | +0.04 | 0.12 | 0.06 | -0.06 | 0.02 | 0.08 | 0.31 | +0.08 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.35 | 0.42 | +0.06 | 0.12 | 0.10 | -0.02 | 0.02 | 0.12 | 0.29 | +0.06 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.35 | 0.35 | +0.00 | 0.12 | 0.19 | +0.06 | 0.02 | 0.06 | 0.15 | -0.08 | flat |
| deepseek7b | animal_tool | forbidden_definition | top_p | baseline | 16-18-20 | 0.27 | 0.27 | +0.00 | 0.10 | 0.10 | +0.00 | 0.00 | 0.02 | 0.17 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.27 | 0.23 | -0.04 | 0.10 | 0.08 | -0.02 | 0.00 | 0.02 | 0.15 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_definition | top_p | residual_full | 16-18-20 | 0.27 | 0.27 | +0.00 | 0.10 | 0.10 | +0.00 | 0.02 | 0.02 | 0.15 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.27 | 0.23 | -0.04 | 0.10 | 0.08 | -0.02 | 0.02 | 0.04 | 0.12 | -0.04 | flat |
| deepseek7b | animal_tool | forbidden_definition | top_p | readout | 16-18-20 | 0.27 | 0.23 | -0.04 | 0.10 | 0.08 | -0.02 | 0.00 | 0.02 | 0.15 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_definition | temperature | baseline | 16-18-20 | 0.25 | 0.25 | +0.00 | 0.10 | 0.10 | +0.00 | 0.04 | 0.02 | 0.10 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.25 | 0.12 | -0.12 | 0.10 | 0.19 | +0.08 | 0.00 | 0.02 | -0.06 | -0.17 | negative |
| deepseek7b | animal_tool | forbidden_definition | temperature | residual_full | 16-18-20 | 0.25 | 0.29 | +0.04 | 0.10 | 0.10 | +0.00 | 0.00 | 0.02 | 0.19 | +0.08 | flat |
| deepseek7b | animal_tool | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.25 | 0.23 | -0.02 | 0.10 | 0.10 | +0.00 | 0.00 | 0.06 | 0.12 | +0.02 | flat |
| deepseek7b | animal_tool | forbidden_definition | temperature | readout | 16-18-20 | 0.25 | 0.12 | -0.12 | 0.10 | 0.19 | +0.08 | 0.00 | 0.02 | -0.06 | -0.17 | negative |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | baseline | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.04 | 0.04 | +0.00 | 0.02 | 0.06 | 0.00 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.06 | 0.00 | -0.06 | 0.04 | 0.02 | -0.02 | 0.00 | 0.02 | -0.02 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.04 | 0.00 | -0.04 | 0.00 | 0.00 | 0.10 | +0.10 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.04 | 0.00 | -0.04 | 0.00 | 0.00 | 0.10 | +0.10 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.06 | 0.00 | -0.06 | 0.04 | 0.02 | -0.02 | 0.00 | 0.02 | -0.02 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | temperature | baseline | 16-18-20 | 0.04 | 0.04 | +0.00 | 0.02 | 0.02 | +0.00 | 0.00 | 0.00 | 0.02 | +0.00 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.04 | 0.04 | +0.00 | 0.02 | 0.02 | +0.00 | 0.02 | 0.02 | 0.00 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | temperature | residual_full | 16-18-20 | 0.04 | 0.04 | +0.00 | 0.02 | 0.04 | +0.02 | 0.00 | 0.00 | 0.00 | -0.02 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.04 | 0.04 | +0.00 | 0.02 | 0.02 | +0.00 | 0.04 | 0.00 | -0.02 | -0.04 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.04 | 0.04 | +0.00 | 0.02 | 0.02 | +0.00 | 0.02 | 0.02 | 0.00 | -0.02 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | baseline | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.15 | 0.15 | +0.00 | 0.02 | 0.19 | -0.06 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.15 | 0.23 | +0.08 | 0.02 | 0.08 | -0.15 | -0.08 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.10 | 0.17 | +0.06 | 0.15 | 0.19 | +0.04 | 0.00 | 0.10 | -0.02 | +0.04 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.15 | 0.23 | +0.08 | 0.00 | 0.12 | -0.08 | -0.02 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.10 | 0.10 | +0.00 | 0.15 | 0.23 | +0.08 | 0.02 | 0.08 | -0.15 | -0.08 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.17 | 0.17 | +0.00 | 0.00 | 0.31 | -0.02 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.15 | 0.10 | -0.04 | 0.17 | 0.29 | +0.13 | 0.04 | 0.19 | -0.23 | -0.21 | negative |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.15 | 0.19 | +0.04 | 0.17 | 0.21 | +0.04 | 0.00 | 0.21 | -0.02 | -0.00 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.17 | 0.19 | +0.02 | 0.02 | 0.21 | -0.06 | -0.04 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | readout | 16-18-20 | 0.15 | 0.10 | -0.04 | 0.17 | 0.29 | +0.13 | 0.04 | 0.19 | -0.23 | -0.21 | negative |
| deepseek7b | fruit_tool | forbidden_definition | top_p | baseline | 16-18-20 | 0.06 | 0.06 | +0.00 | 0.25 | 0.25 | +0.00 | 0.02 | 0.00 | -0.21 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.25 | 0.38 | +0.12 | 0.02 | 0.02 | -0.35 | -0.15 | negative |
| deepseek7b | fruit_tool | forbidden_definition | top_p | residual_full | 16-18-20 | 0.06 | 0.02 | -0.04 | 0.25 | 0.27 | +0.02 | 0.02 | 0.04 | -0.27 | -0.06 | flat |
| deepseek7b | fruit_tool | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.06 | 0.00 | -0.06 | 0.25 | 0.27 | +0.02 | 0.02 | 0.06 | -0.29 | -0.08 | flat |
| deepseek7b | fruit_tool | forbidden_definition | top_p | readout | 16-18-20 | 0.06 | 0.04 | -0.02 | 0.25 | 0.38 | +0.12 | 0.02 | 0.02 | -0.35 | -0.15 | negative |
| deepseek7b | fruit_tool | forbidden_definition | temperature | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.17 | 0.17 | +0.00 | 0.00 | 0.10 | -0.02 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_definition | temperature | residual_parallel | 16-18-20 | 0.15 | 0.10 | -0.04 | 0.17 | 0.27 | +0.10 | 0.02 | 0.02 | -0.19 | -0.17 | negative |
| deepseek7b | fruit_tool | forbidden_definition | temperature | residual_full | 16-18-20 | 0.15 | 0.10 | -0.04 | 0.17 | 0.12 | -0.04 | 0.00 | 0.06 | -0.02 | -0.00 | flat |
| deepseek7b | fruit_tool | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.15 | 0.08 | -0.06 | 0.17 | 0.25 | +0.08 | 0.00 | 0.06 | -0.17 | -0.15 | negative |
| deepseek7b | fruit_tool | forbidden_definition | temperature | readout | 16-18-20 | 0.15 | 0.10 | -0.04 | 0.17 | 0.27 | +0.10 | 0.02 | 0.02 | -0.19 | -0.17 | negative |
| deepseek7b | fruit_tool | forbidden_sentence_completion | top_p | baseline | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.04 | 0.04 | +0.00 | 0.02 | 0.00 | 0.08 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | top_p | residual_parallel | 16-18-20 | 0.15 | 0.08 | -0.06 | 0.04 | 0.08 | +0.04 | 0.00 | 0.00 | 0.00 | -0.08 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.15 | 0.17 | +0.02 | 0.04 | 0.06 | +0.02 | 0.00 | 0.00 | 0.10 | +0.02 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.15 | 0.15 | +0.00 | 0.04 | 0.08 | +0.04 | 0.00 | 0.00 | 0.06 | -0.02 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | top_p | readout | 16-18-20 | 0.15 | 0.08 | -0.06 | 0.04 | 0.08 | +0.04 | 0.00 | 0.00 | 0.00 | -0.08 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | baseline | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.08 | 0.08 | +0.00 | 0.00 | 0.00 | 0.04 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.08 | 0.12 | +0.04 | 0.00 | 0.00 | 0.04 | -0.00 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | residual_full | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.08 | 0.10 | +0.02 | 0.00 | 0.00 | 0.02 | -0.02 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | residual_perp | 16-18-20 | 0.12 | 0.12 | +0.00 | 0.08 | 0.12 | +0.04 | 0.00 | 0.02 | 0.00 | -0.04 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.08 | 0.12 | +0.04 | 0.00 | 0.00 | 0.04 | -0.00 | flat |

## Best Clean Paraphrase Gains

| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.31 | 0.65 | +0.33 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.65 | +0.35 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.29 | 0.56 | +0.27 | 0.02 | 0.02 | +0.00 | 0.00 | 0.04 | 0.54 | +0.33 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_full | 24-26-28 | 0.31 | 0.56 | +0.25 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.56 | +0.27 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_full | 24-26-28 | 0.29 | 0.54 | +0.25 | 0.02 | 0.02 | +0.00 | 0.02 | 0.02 | 0.50 | +0.29 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.31 | 0.52 | +0.21 | 0.02 | 0.06 | +0.04 | 0.08 | 0.12 | 0.38 | +0.31 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | residual_perp | 24-26-28 | 0.33 | 0.54 | +0.21 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.54 | +0.21 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | top_p | residual_full | 24-26-28 | 0.33 | 0.52 | +0.19 | 0.00 | 0.00 | +0.00 | 0.00 | 0.02 | 0.52 | +0.19 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.25 | 0.44 | +0.19 | 0.00 | 0.00 | +0.00 | 0.00 | 0.00 | 0.44 | +0.33 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_full | 24-26-28 | 0.25 | 0.44 | +0.19 | 0.00 | 0.02 | +0.02 | 0.02 | 0.00 | 0.40 | +0.29 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_full | 10-12-14 | 0.04 | 0.23 | +0.19 | 0.04 | 0.04 | +0.00 | 0.04 | 0.25 | 0.15 | +0.25 | clean_paraphrase_positive |
| glm4 | fruit_vegetable | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.08 | 0.27 | +0.19 | 0.48 | 0.48 | +0.00 | 0.00 | 0.08 | -0.21 | +0.21 | clean_paraphrase_positive |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.08 | 0.27 | +0.19 | 0.50 | 0.54 | +0.04 | 0.00 | 0.06 | -0.27 | +0.15 | clean_paraphrase_positive |
| glm4 | fruit_vegetable | forbidden_definition | temperature | residual_perp | 24-26-28 | 0.27 | 0.44 | +0.17 | 0.19 | 0.10 | -0.08 | 0.00 | 0.02 | 0.33 | +0.25 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.00 | -0.04 | 0.08 | 0.29 | 0.12 | +0.23 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.04 | +0.00 | 0.08 | 0.23 | 0.08 | +0.19 | clean_paraphrase_positive |
| qwen3 | vehicle_tool | forbidden_definition | top_p | readout | 10-12-14 | 0.04 | 0.21 | +0.17 | 0.04 | 0.04 | +0.00 | 0.08 | 0.23 | 0.08 | +0.19 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.38 | 0.54 | +0.17 | 0.00 | 0.04 | +0.04 | 0.06 | 0.12 | 0.44 | +0.25 | clean_paraphrase_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.31 | 0.50 | +0.19 | 0.02 | 0.08 | +0.06 | 0.04 | 0.10 | 0.38 | +0.31 | synonym_with_label_leak |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | residual_perp | 10-12-14 | 0.54 | 0.69 | +0.15 | 0.02 | 0.06 | +0.04 | 0.02 | 0.00 | 0.60 | +0.08 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | residual_perp | 10-12-14 | 0.52 | 0.67 | +0.15 | 0.00 | 0.02 | +0.02 | 0.00 | 0.02 | 0.65 | +0.12 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | top_p | residual_full | 10-12-14 | 0.52 | 0.67 | +0.15 | 0.00 | 0.04 | +0.04 | 0.00 | 0.00 | 0.62 | +0.10 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_perp | 10-12-14 | 0.38 | 0.50 | +0.12 | 0.04 | 0.00 | -0.04 | 0.00 | 0.10 | 0.50 | +0.19 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 24-26-28 | 0.38 | 0.50 | +0.12 | 0.00 | 0.02 | +0.02 | 0.10 | 0.15 | 0.38 | +0.19 | weak_clean_positive |
| glm4 | fruit_vegetable | forbidden_definition | top_p | residual_perp | 24-26-28 | 0.29 | 0.42 | +0.12 | 0.12 | 0.10 | -0.02 | 0.00 | 0.02 | 0.31 | +0.15 | weak_clean_positive |
| glm4 | fruit_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.08 | 0.19 | +0.10 | 0.50 | 0.56 | +0.06 | 0.00 | 0.10 | -0.38 | +0.04 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 24-26-28 | 0.29 | 0.40 | +0.10 | 0.02 | 0.23 | +0.21 | 0.02 | 0.00 | 0.15 | -0.06 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_definition | top_p | readout | 24-26-28 | 0.29 | 0.40 | +0.10 | 0.02 | 0.23 | +0.21 | 0.02 | 0.00 | 0.15 | -0.06 | weak_clean_positive |
| glm4 | fruit_vegetable | forbidden_definition | temperature | residual_full | 24-26-28 | 0.27 | 0.35 | +0.08 | 0.19 | 0.08 | -0.10 | 0.00 | 0.02 | 0.27 | +0.19 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_sentence_completion | temperature | residual_full | 10-12-14 | 0.54 | 0.62 | +0.08 | 0.02 | 0.08 | +0.06 | 0.02 | 0.04 | 0.52 | +0.00 | weak_clean_positive |
| glm4 | animal_tool | forbidden_natural_qa | top_p | residual_full | 24-26-28 | 0.35 | 0.44 | +0.08 | 0.12 | 0.02 | -0.10 | 0.00 | 0.02 | 0.42 | +0.21 | weak_clean_positive |
| glm4 | fruit_tool | forbidden_definition | temperature | residual_full | 24-26-28 | 0.29 | 0.38 | +0.08 | 0.06 | 0.02 | -0.04 | 0.00 | 0.04 | 0.35 | +0.12 | weak_clean_positive |
| qwen3 | vehicle_tool | forbidden_definition | temperature | residual_full | 10-12-14 | 0.19 | 0.27 | +0.08 | 0.02 | 0.04 | +0.02 | 0.04 | 0.21 | 0.19 | +0.10 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | temperature | residual_full | 10-12-14 | 0.35 | 0.44 | +0.08 | 0.00 | 0.00 | +0.00 | 0.00 | 0.12 | 0.44 | +0.08 | weak_clean_positive |
| qwen3 | fruit_tool | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.35 | 0.44 | +0.08 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.44 | +0.08 | weak_clean_positive |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | residual_parallel | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.00 | 0.17 | +0.17 | 0.00 | 0.00 | 0.23 | -0.06 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_sentence_completion | temperature | readout | 24-26-28 | 0.31 | 0.40 | +0.08 | 0.00 | 0.17 | +0.17 | 0.00 | 0.00 | 0.23 | -0.06 | label_gate_leak |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.08 | 0.15 | +0.06 | 0.15 | 0.17 | +0.02 | 0.00 | 0.29 | -0.02 | +0.04 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_full | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.00 | 0.17 | 0.42 | +0.10 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.60 | 0.67 | +0.06 | 0.08 | 0.06 | -0.02 | 0.00 | 0.10 | 0.60 | +0.08 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.60 | 0.67 | +0.06 | 0.08 | 0.06 | -0.02 | 0.00 | 0.10 | 0.60 | +0.08 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | residual_parallel | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.02 | 0.23 | 0.40 | +0.08 | flat |
| qwen3 | fruit_tool | forbidden_definition | top_p | readout | 10-12-14 | 0.38 | 0.44 | +0.06 | 0.04 | 0.02 | -0.02 | 0.02 | 0.23 | 0.40 | +0.08 | flat |
| glm4 | fruit_vegetable | forbidden_definition | top_p | residual_full | 24-26-28 | 0.29 | 0.35 | +0.06 | 0.12 | 0.10 | -0.02 | 0.00 | 0.04 | 0.25 | +0.08 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_full | 16-18-20 | 0.12 | 0.19 | +0.06 | 0.17 | 0.15 | -0.02 | 0.00 | 0.04 | 0.04 | +0.08 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.12 | 0.19 | +0.06 | 0.23 | 0.23 | +0.00 | 0.00 | 0.02 | -0.04 | +0.06 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | residual_perp | 16-18-20 | 0.35 | 0.42 | +0.06 | 0.12 | 0.10 | -0.02 | 0.02 | 0.12 | 0.29 | +0.06 | flat |
| glm4 | vehicle_tool | forbidden_definition | temperature | residual_parallel | 24-26-28 | 0.25 | 0.31 | +0.06 | 0.00 | 0.17 | +0.17 | 0.02 | 0.00 | 0.13 | +0.02 | label_gate_leak |
| glm4 | vehicle_tool | forbidden_definition | temperature | readout | 24-26-28 | 0.25 | 0.31 | +0.06 | 0.00 | 0.17 | +0.17 | 0.02 | 0.00 | 0.13 | +0.02 | label_gate_leak |
| glm4 | fruit_vegetable | forbidden_natural_qa | temperature | residual_perp | 24-26-28 | 0.19 | 0.25 | +0.06 | 0.38 | 0.46 | +0.08 | 0.00 | 0.06 | -0.21 | +0.00 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | temperature | residual_perp | 16-18-20 | 0.15 | 0.21 | +0.06 | 0.19 | 0.25 | +0.06 | 0.00 | 0.06 | -0.04 | +0.00 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | residual_full | 16-18-20 | 0.10 | 0.17 | +0.06 | 0.15 | 0.19 | +0.04 | 0.00 | 0.10 | -0.02 | +0.04 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 10-12-14 | 0.31 | 0.35 | +0.04 | 0.12 | 0.23 | +0.10 | 0.08 | 0.10 | 0.04 | +0.08 | flat |
| deepseek7b | animal_tool | forbidden_definition | temperature | residual_full | 16-18-20 | 0.25 | 0.29 | +0.04 | 0.10 | 0.10 | +0.00 | 0.00 | 0.02 | 0.19 | +0.08 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.04 | 0.00 | -0.04 | 0.00 | 0.00 | 0.10 | +0.10 | flat |
| deepseek7b | animal_tool | forbidden_sentence_completion | top_p | residual_perp | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.04 | 0.00 | -0.04 | 0.00 | 0.00 | 0.10 | +0.10 | flat |
| deepseek7b | fruit_vegetable | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.40 | 0.38 | -0.02 | 0.00 | 0.04 | -0.27 | +0.06 | flat |
| deepseek7b | fruit_vegetable | forbidden_natural_qa | temperature | residual_parallel | 16-18-20 | 0.06 | 0.10 | +0.04 | 0.17 | 0.15 | -0.02 | 0.00 | 0.31 | -0.04 | +0.06 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.27 | 0.31 | +0.04 | 0.08 | 0.08 | -0.25 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_natural_qa | top_p | readout | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.27 | 0.31 | +0.04 | 0.08 | 0.08 | -0.25 | +0.02 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 16-18-20 | 0.10 | 0.15 | +0.04 | 0.15 | 0.23 | +0.08 | 0.00 | 0.12 | -0.08 | -0.02 | flat |
| qwen3 | vehicle_tool | forbidden_definition | temperature | residual_perp | 10-12-14 | 0.19 | 0.23 | +0.04 | 0.02 | 0.00 | -0.02 | 0.04 | 0.25 | 0.19 | +0.10 | flat |
| deepseek7b | fruit_vegetable | forbidden_sentence_completion | top_p | residual_full | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.06 | 0.04 | -0.02 | 0.00 | 0.00 | 0.12 | +0.06 | flat |
| glm4 | fruit_tool | forbidden_sentence_completion | temperature | residual_perp | 24-26-28 | 0.12 | 0.17 | +0.04 | 0.02 | 0.00 | -0.02 | 0.04 | 0.00 | 0.12 | +0.06 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_perp | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.15 | -0.02 | 0.00 | 0.04 | 0.02 | +0.06 | flat |
| glm4 | fruit_tool | forbidden_natural_qa | temperature | residual_full | 24-26-28 | 0.15 | 0.19 | +0.04 | 0.46 | 0.46 | +0.00 | 0.00 | 0.04 | -0.27 | +0.04 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | residual_parallel | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.19 | +0.02 | 0.00 | 0.06 | -0.02 | +0.02 | flat |
| deepseek7b | vehicle_tool | forbidden_definition | top_p | readout | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.17 | 0.19 | +0.02 | 0.00 | 0.06 | -0.02 | +0.02 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | residual_parallel | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.08 | 0.12 | +0.04 | 0.00 | 0.00 | 0.04 | -0.00 | flat |
| deepseek7b | fruit_tool | forbidden_sentence_completion | temperature | readout | 16-18-20 | 0.12 | 0.17 | +0.04 | 0.08 | 0.12 | +0.04 | 0.00 | 0.00 | 0.04 | -0.00 | flat |
| deepseek7b | fruit_tool | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.15 | 0.19 | +0.04 | 0.17 | 0.21 | +0.04 | 0.00 | 0.21 | -0.02 | -0.00 | flat |
| deepseek7b | animal_tool | forbidden_natural_qa | temperature | residual_full | 16-18-20 | 0.35 | 0.40 | +0.04 | 0.12 | 0.06 | -0.06 | 0.02 | 0.08 | 0.31 | +0.08 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_perp | 10-12-14 | 0.88 | 0.92 | +0.04 | 0.00 | 0.00 | +0.00 | 0.00 | 0.06 | 0.92 | +0.04 | flat |
| qwen3 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.40 | 0.42 | +0.02 | 0.04 | 0.12 | +0.08 | 0.06 | 0.08 | 0.23 | +0.06 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.73 | 0.75 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.75 | +0.02 | flat |
| qwen3 | fruit_vegetable | forbidden_natural_qa | top_p | readout | 10-12-14 | 0.73 | 0.75 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.75 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_parallel | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.90 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | residual_full | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.08 | 0.90 | +0.02 | flat |
| qwen3 | animal_tool | forbidden_natural_qa | temperature | readout | 10-12-14 | 0.88 | 0.90 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.04 | 0.90 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | residual_parallel | 10-12-14 | 0.75 | 0.77 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.77 | +0.02 | flat |
| qwen3 | fruit_tool | forbidden_natural_qa | top_p | residual_perp | 10-12-14 | 0.75 | 0.77 | +0.02 | 0.00 | 0.00 | +0.00 | 0.00 | 0.15 | 0.77 | +0.02 | flat |

## Pair Max Clean Gain

| model | pair | max clean gain | label gain | score gain | row | class |
|---|---|---:|---:|---:|---|---|
| deepseek7b | animal_tool | +0.06 | -0.02 | +0.06 | forbidden_natural_qa temperature residual_perp 16-18-20 | flat |
| deepseek7b | fruit_tool | +0.06 | +0.04 | +0.04 | forbidden_natural_qa top_p residual_full 16-18-20 | flat |
| deepseek7b | fruit_vegetable | +0.06 | +0.02 | +0.04 | forbidden_natural_qa top_p residual_parallel 16-18-20 | flat |
| deepseek7b | vehicle_tool | +0.06 | -0.02 | +0.08 | forbidden_definition top_p residual_full 16-18-20 | flat |
| glm4 | animal_tool | +0.08 | -0.10 | +0.21 | forbidden_natural_qa top_p residual_full 24-26-28 | weak_clean_positive |
| glm4 | fruit_tool | +0.19 | +0.04 | +0.15 | forbidden_natural_qa top_p residual_perp 24-26-28 | clean_paraphrase_positive |
| glm4 | fruit_vegetable | +0.19 | +0.00 | +0.21 | forbidden_natural_qa top_p residual_perp 24-26-28 | clean_paraphrase_positive |
| glm4 | vehicle_tool | +0.33 | +0.00 | +0.35 | forbidden_sentence_completion temperature residual_perp 24-26-28 | clean_paraphrase_positive |
| qwen3 | animal_tool | +0.04 | +0.00 | +0.04 | forbidden_natural_qa temperature residual_perp 10-12-14 | flat |
| qwen3 | fruit_tool | +0.12 | -0.04 | +0.19 | forbidden_definition top_p residual_perp 10-12-14 | weak_clean_positive |
| qwen3 | fruit_vegetable | +0.06 | -0.02 | +0.08 | forbidden_natural_qa temperature residual_parallel 10-12-14 | flat |
| qwen3 | vehicle_tool | +0.19 | +0.00 | +0.25 | forbidden_definition top_p residual_full 10-12-14 | clean_paraphrase_positive |

## Representative Samples

| model | window | pair | scaffold | mode | condition | seed | quality | prompt | suffix |
|---|---|---|---|---|---|---:|---|---|---|
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | generic_only | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a type of propulsion system that generates thrust by expelling mass |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made from a single piece of wood |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling. It's |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | top_p | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling. It's |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a structure that can travel through space by using a propulsion system |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made from a single piece of wood |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made from a single piece of wood |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a craft that travels through space, propelled by powerful engines that |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a craft that travels through space, propelled by powerful engines that |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made for paddling on water. |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_natural_qa | temperature | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a long, narrow boat made from a single piece of wood |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | baseline | 101 | other | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a... Canoe is best described as a... Canoe |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 101 | generic_only | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as | : (a) an object that travels in space, ( |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 101 | other | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a... Canoe is best described as a long, narrow |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_full | 101 | other | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as | :   A. a structure that transports people and goods from |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_full | 101 | other | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a... Canoe is best described as a... Canoe |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_perp | 101 | other | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as | :   A. a structure that generates and propels itself |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | residual_perp | 101 | other | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a... Canoe is best described as a... Canoe |
| qwen3 | 10-12-14 | vehicle_tool | forbidden_definition | top_p | readout | 101 | generic_only | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as | : (a) an object that travels in space, ( |

| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | generic_only | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  an object that travels through the air and space. It is |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | generic_only | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of.water watercraftcraft, that usually is |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 101 | synonym_with_label_violation | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a form of transport. It carries humans, vehicles, and |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of craft that is small and can be moved by |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a high-powered transport system that carries objects from one place to |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a lightweight, usually open, watercraft, which is traditionally |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 101 | wrong_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a high-powered device used for traveling long distances in the sky |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | readout | 101 | synonym_with_label_violation | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a form of transport. It carries humans, vehicles, and |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | top_p | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of craft that is small and can be moved by |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  an invention that travels in the sky through force. It is |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of.  boat. A canoe is often used for |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a spacecraft that propels itself through space by expelling fuel |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of boat. It is often made from wood or |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a set of transport powered by fuel, and used for traveling |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a lightweight, simple craft that can be easily moved from one |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  an engine that propels a craft or satellite into space. |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a lightweight water craft usually made from wood.It, is |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | readout | 101 | clean_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  a spacecraft that propels itself through space by expelling fuel |
| glm4 | 24-26-28 | vehicle_tool | forbidden_natural_qa | temperature | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of boat. It is often made from wood or |
| glm4 | 24-26-28 | vehicle_tool | forbidden_definition | top_p | baseline | 101 | other | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as |  as a an means   ofA a means means of of |
| glm4 | 24-26-28 | vehicle_tool | forbidden_definition | top_p | baseline | 101 | clean_synonym | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a portable watercraft that is propelled by the paddler's |
| glm4 | 24-26-28 | vehicle_tool | forbidden_definition | top_p | residual_parallel | 101 | synonym_with_label_violation | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as |  as:   aA powerful means vehicle of of transportation transporting that |

| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | wrong_synonym | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  ___. A rocket is a device designed to propel itself into |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | baseline | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of...  A canoe is a type of watercraft |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  [ ].  Wait, I need to fill in the blank |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of vessel made from wood or other materials, typically |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  | ...  Think about other ways to describe a rocket without using the |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of vessel designed for transportation or recreation on water. |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  [a/an/...].  Wait, can I think of |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | residual_perp | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of vessel that is primarily used for transportation and recreation |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | readout | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  [ ].  Wait, I need to fill in the blank |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | top_p | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of vessel made from wood or other materials, typically |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  __________.  Okay, so I'm trying to figure out |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | baseline | 101 | generic_only | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of **raft**. It can also be considered |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 101 | generic_only | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  similar in function to another object, thus categorizing it as |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_parallel | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of floating object designed for transportation or recreation. It |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  __________.  Okay, so I'm trying to figure out |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_full | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of floatingcraft designed for transportation or recreation. It |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | other | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  __________.  Okay, so I'm trying to figure out |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | residual_perp | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of floatingcraft designed for transportation or recreation. It |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | readout | 101 | generic_only | What kind of thing is a rocket? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools.  |  similar in function to another object, thus categorizing it as |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_natural_qa | temperature | readout | 101 | clean_synonym | What kind of thing is a canoe? Answer naturally, but do not use these words: vehicle, vehicles, tool, tools. A |  a type of floating object designed for transportation or recreation. It |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_definition | top_p | baseline | 101 | other | Define rocket without using these words: vehicle, vehicles, tool, tools. Rocket is best described as | : adjective, noun, noun, adjective? Wait, no |
| deepseek7b | 16-18-20 | vehicle_tool | forbidden_definition | top_p | baseline | 101 | clean_synonym | Define canoe without using these words: vehicle, vehicles, tool, tools. Canoe is best described as |  a watercraft made from lightweight materials, typically used for transportation |

