# Phase552 Paraphrase Necessity and Writer Decomposition Summary

## qwen3

pair=vehicle_tool, window=[10, 12, 14], combos=['L10', 'L14', 'L10+L14', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_full', 'resid_remove_random_perp', 'resid_remove_perp_add_perp', 'attn_remove_perp', 'mlp_remove_perp', 'attn_remove_full', 'mlp_remove_full'], scaffold_modes=['forbidden_sentence_completion:temperature', 'forbidden_natural_qa:top_p', 'forbidden_definition:top_p'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], remove_scale=1.0, add_alpha=6.0

| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.47 | -0.07 | 0.01 | 0.01 | +0.00 | 0.61 | 0.00 | -0.06 | -0.07 | +0.00 | weak_drop |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.48 | -0.06 | 0.01 | 0.01 | +0.00 | 0.61 | 0.01 | -0.06 | -0.03 | -0.03 | weak_drop |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.49 | -0.05 | 0.01 | 0.02 | +0.01 | 0.62 | 0.01 | -0.06 | -0.07 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.54 | 0.49 | -0.05 | 0.01 | 0.01 | +0.00 | 0.62 | 0.01 | -0.05 | -0.03 | -0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.51 | -0.03 | 0.01 | 0.01 | +0.00 | 0.56 | 0.01 | -0.03 | -0.03 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.48 | 0.46 | -0.02 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | -0.03 | +0.01 | -0.03 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | resid_remove_random_perp | 0.22 | 0.20 | -0.02 | 0.03 | 0.01 | -0.02 | 0.36 | 0.03 | -0.05 | -0.02 | +0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.02 | +0.01 | 0.54 | 0.01 | -0.03 | +0.02 | -0.04 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.54 | 0.52 | -0.02 | 0.01 | 0.02 | +0.01 | 0.57 | 0.00 | -0.02 | -0.07 | +0.05 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.01 | +0.00 | 0.62 | 0.01 | -0.02 | +0.02 | -0.04 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.01 | +0.00 | 0.58 | 0.00 | -0.01 | -0.03 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | attn_remove_perp | 0.48 | 0.47 | -0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | -0.02 | +0.00 | -0.01 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | mlp_remove_full | 0.48 | 0.47 | -0.01 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | -0.01 | +0.01 | -0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | attn_remove_perp | 0.48 | 0.47 | -0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.00 | +0.01 | -0.02 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | attn_remove_full | 0.48 | 0.47 | -0.01 | 0.07 | 0.06 | -0.01 | 0.00 | 0.00 | +0.01 | +0.01 | -0.02 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | mlp_remove_perp | 0.22 | 0.21 | -0.01 | 0.03 | 0.02 | -0.01 | 0.31 | 0.01 | -0.01 | +0.03 | -0.04 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | mlp_remove_full | 0.22 | 0.21 | -0.01 | 0.03 | 0.02 | -0.01 | 0.35 | 0.00 | +0.03 | -0.02 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.53 | -0.01 | 0.01 | 0.03 | +0.02 | 0.61 | 0.01 | -0.03 | -0.07 | +0.06 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.54 | 0.53 | -0.01 | 0.01 | 0.01 | +0.00 | 0.61 | 0.00 | +0.00 | -0.07 | +0.06 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | attn_remove_full | 0.22 | 0.22 | +0.00 | 0.03 | 0.03 | +0.00 | 0.30 | 0.03 | -0.01 | +0.01 | -0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.48 | 0.48 | +0.00 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | -0.01 | +0.00 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.54 | 0.54 | +0.00 | 0.01 | 0.02 | +0.01 | 0.61 | 0.00 | -0.00 | +0.01 | -0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | attn_remove_perp | 0.22 | 0.22 | +0.00 | 0.03 | 0.01 | -0.02 | 0.34 | 0.03 | +0.00 | -0.02 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | attn_remove_perp | 0.22 | 0.22 | +0.00 | 0.03 | 0.01 | -0.02 | 0.36 | 0.01 | +0.02 | +0.01 | -0.01 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.00 | +0.01 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.00 | +0.01 | +0.00 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.00 | +0.01 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | attn_remove_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.01 | +0.04 | -0.03 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | attn_remove_full | 0.48 | 0.49 | +0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.01 | +0.01 | +0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | attn_remove_full | 0.48 | 0.49 | +0.01 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.01 | +0.00 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | mlp_remove_full | 0.48 | 0.49 | +0.01 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.01 | +0.00 | +0.01 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.02 | +0.01 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | attn_remove_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.06 | -0.01 | 0.00 | 0.00 | +0.04 | +0.01 | +0.00 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | resid_remove_full | 0.48 | 0.49 | +0.01 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.05 | +0.01 | +0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | resid_remove_full | 0.48 | 0.49 | +0.01 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.06 | +0.00 | +0.01 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | resid_remove_full | 0.22 | 0.23 | +0.01 | 0.03 | 0.04 | +0.01 | 0.29 | 0.01 | -0.03 | +0.01 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.22 | 0.23 | +0.01 | 0.03 | 0.06 | +0.03 | 0.40 | 0.03 | -0.01 | +0.01 | +0.00 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | resid_remove_random_perp | 0.22 | 0.23 | +0.01 | 0.03 | 0.01 | -0.02 | 0.39 | 0.03 | +0.00 | +0.01 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | resid_remove_random_perp | 0.22 | 0.23 | +0.01 | 0.03 | 0.01 | -0.02 | 0.39 | 0.02 | +0.02 | +0.01 | +0.00 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | resid_remove_full | 0.54 | 0.55 | +0.01 | 0.01 | 0.02 | +0.01 | 0.60 | 0.01 | +0.00 | -0.03 | +0.04 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.55 | +0.01 | 0.01 | 0.01 | +0.00 | 0.56 | 0.00 | +0.02 | +0.01 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | resid_remove_full | 0.48 | 0.50 | +0.02 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.03 | +0.01 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.48 | 0.50 | +0.02 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.03 | +0.04 | -0.02 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | attn_remove_full | 0.48 | 0.50 | +0.02 | 0.07 | 0.06 | -0.01 | 0.00 | 0.00 | +0.03 | +0.04 | -0.02 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | mlp_remove_full | 0.48 | 0.50 | +0.02 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.03 | +0.04 | -0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | mlp_remove_full | 0.48 | 0.50 | +0.02 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.03 | +0.01 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.48 | 0.50 | +0.02 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.04 | +0.00 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | resid_remove_perp | 0.48 | 0.50 | +0.02 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.05 | +0.00 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.48 | 0.50 | +0.02 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.07 | +0.00 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | mlp_remove_full | 0.22 | 0.24 | +0.02 | 0.03 | 0.02 | -0.01 | 0.36 | 0.01 | +0.01 | +0.01 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | resid_remove_perp | 0.22 | 0.24 | +0.02 | 0.03 | 0.04 | +0.01 | 0.35 | 0.02 | +0.01 | +0.01 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | attn_remove_perp | 0.22 | 0.24 | +0.02 | 0.03 | 0.01 | -0.02 | 0.30 | 0.01 | +0.04 | +0.01 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | resid_remove_perp | 0.22 | 0.24 | +0.02 | 0.03 | 0.03 | +0.00 | 0.34 | 0.01 | +0.04 | -0.02 | +0.04 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | mlp_remove_full | 0.22 | 0.24 | +0.02 | 0.03 | 0.01 | -0.02 | 0.30 | 0.01 | +0.05 | +0.01 | +0.01 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | attn_remove_perp | 0.22 | 0.24 | +0.02 | 0.03 | 0.02 | -0.01 | 0.31 | 0.01 | +0.06 | +0.03 | -0.01 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.54 | 0.56 | +0.02 | 0.01 | 0.03 | +0.02 | 0.68 | 0.01 | +0.00 | +0.01 | +0.01 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.56 | +0.02 | 0.01 | 0.03 | +0.02 | 0.57 | 0.01 | +0.00 | +0.01 | +0.01 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.56 | +0.02 | 0.01 | 0.03 | +0.02 | 0.61 | 0.00 | +0.00 | +0.01 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.56 | +0.02 | 0.01 | 0.03 | +0.02 | 0.54 | 0.00 | +0.00 | +0.02 | +0.00 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.54 | 0.56 | +0.02 | 0.01 | 0.01 | +0.00 | 0.61 | 0.01 | +0.02 | +0.02 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.51 | +0.03 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.05 | +0.04 | -0.01 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | resid_remove_perp | 0.48 | 0.51 | +0.03 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.06 | +0.04 | -0.01 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | resid_remove_perp | 0.48 | 0.51 | +0.03 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.07 | +0.01 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.48 | 0.51 | +0.03 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.07 | +0.01 | +0.02 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | resid_remove_random_perp | 0.22 | 0.25 | +0.03 | 0.03 | 0.03 | +0.00 | 0.34 | 0.02 | +0.01 | +0.03 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.54 | 0.57 | +0.03 | 0.01 | 0.03 | +0.02 | 0.56 | 0.00 | +0.02 | +0.01 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.54 | 0.57 | +0.03 | 0.01 | 0.02 | +0.01 | 0.64 | 0.01 | +0.02 | -0.03 | +0.06 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.54 | 0.57 | +0.03 | 0.01 | 0.02 | +0.01 | 0.54 | 0.01 | +0.02 | +0.02 | +0.01 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | mlp_remove_perp | 0.22 | 0.25 | +0.03 | 0.03 | 0.02 | -0.01 | 0.29 | 0.00 | +0.02 | +0.01 | +0.02 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | attn_remove_full | 0.22 | 0.25 | +0.03 | 0.03 | 0.03 | +0.00 | 0.33 | 0.03 | +0.02 | +0.03 | +0.00 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | mlp_remove_perp | 0.22 | 0.25 | +0.03 | 0.03 | 0.02 | -0.01 | 0.31 | 0.01 | +0.02 | +0.01 | +0.02 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | attn_remove_full | 0.22 | 0.25 | +0.03 | 0.03 | 0.01 | -0.02 | 0.31 | 0.02 | +0.03 | -0.02 | +0.05 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.54 | 0.57 | +0.03 | 0.01 | 0.01 | +0.00 | 0.60 | 0.00 | +0.04 | -0.03 | +0.06 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | attn_remove_full | 0.22 | 0.25 | +0.03 | 0.03 | 0.01 | -0.02 | 0.33 | 0.02 | +0.04 | +0.01 | +0.02 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | resid_remove_full | 0.22 | 0.25 | +0.03 | 0.03 | 0.01 | -0.02 | 0.26 | 0.01 | +0.07 | +0.03 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | resid_remove_perp | 0.22 | 0.26 | +0.04 | 0.03 | 0.01 | -0.02 | 0.32 | 0.02 | +0.05 | +0.03 | +0.01 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | mlp_remove_full | 0.22 | 0.26 | +0.04 | 0.03 | 0.01 | -0.02 | 0.31 | 0.01 | +0.05 | +0.03 | +0.01 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.48 | 0.52 | +0.04 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.05 | +0.04 | +0.00 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.48 | 0.52 | +0.04 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.06 | +0.01 | +0.03 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | add_perp | 0.22 | 0.26 | +0.04 | 0.03 | 0.03 | +0.00 | 0.35 | 0.01 | +0.06 | +0.01 | +0.03 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | add_perp | 0.22 | 0.26 | +0.04 | 0.03 | 0.01 | -0.02 | 0.33 | 0.01 | +0.09 | +0.01 | +0.03 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.54 | 0.58 | +0.04 | 0.01 | 0.02 | +0.01 | 0.64 | 0.00 | +0.04 | -0.07 | +0.11 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | resid_remove_full | 0.54 | 0.58 | +0.04 | 0.01 | 0.01 | +0.00 | 0.58 | 0.01 | +0.04 | -0.07 | +0.11 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | resid_remove_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.07 | +0.00 | 0.00 | 0.00 | +0.07 | +0.01 | +0.04 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | resid_remove_full | 0.22 | 0.27 | +0.05 | 0.03 | 0.03 | +0.00 | 0.34 | 0.01 | +0.07 | -0.02 | +0.07 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.10 | +0.03 | 0.00 | 0.00 | +0.09 | +0.00 | +0.05 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | resid_remove_full | 0.48 | 0.53 | +0.05 | 0.07 | 0.06 | -0.01 | 0.00 | 0.00 | +0.10 | +0.04 | +0.01 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.11 | +0.01 | +0.04 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.11 | +0.04 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | resid_remove_full | 0.54 | 0.59 | +0.05 | 0.01 | 0.03 | +0.02 | 0.67 | 0.00 | +0.04 | +0.02 | +0.03 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.05 | +0.04 | 0.60 | 0.00 | +0.02 | +0.01 | +0.05 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.03 | +0.02 | 0.57 | 0.01 | +0.04 | -0.03 | +0.09 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | resid_remove_full | 0.22 | 0.28 | +0.06 | 0.03 | 0.01 | -0.02 | 0.38 | 0.04 | +0.06 | +0.01 | +0.05 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | mlp_remove_perp | 0.22 | 0.28 | +0.06 | 0.03 | 0.02 | -0.01 | 0.28 | 0.02 | +0.06 | -0.02 | +0.08 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.00 | -0.01 | 0.58 | 0.01 | +0.07 | +0.01 | +0.05 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.00 | -0.01 | 0.61 | 0.01 | +0.07 | -0.07 | +0.14 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.00 | -0.01 | 0.67 | 0.00 | +0.08 | +0.02 | +0.04 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | resid_remove_perp | 0.22 | 0.29 | +0.07 | 0.03 | 0.03 | +0.00 | 0.31 | 0.01 | +0.08 | +0.01 | +0.06 | flat |
| qwen3 | L10 | 10 | forbidden_definition | top_p | add_perp | 0.22 | 0.29 | +0.07 | 0.03 | 0.00 | -0.03 | 0.27 | 0.01 | +0.11 | +0.03 | +0.04 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.54 | 0.61 | +0.07 | 0.01 | 0.01 | +0.00 | 0.55 | 0.00 | +0.08 | -0.03 | +0.10 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | add_perp | 0.22 | 0.30 | +0.08 | 0.03 | 0.06 | +0.03 | 0.29 | 0.00 | +0.10 | -0.02 | +0.10 | positive_add_or_release |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | resid_remove_full | 0.54 | 0.62 | +0.08 | 0.01 | 0.01 | +0.00 | 0.66 | 0.01 | +0.08 | +0.01 | +0.07 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.64 | +0.09 | 0.01 | 0.03 | +0.02 | 0.58 | 0.00 | +0.08 | +0.02 | +0.07 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.64 | +0.09 | 0.01 | 0.01 | +0.00 | 0.64 | 0.00 | +0.10 | -0.07 | +0.17 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.54 | 0.64 | +0.09 | 0.01 | 0.01 | +0.00 | 0.56 | 0.00 | +0.10 | +0.02 | +0.07 | positive_add_or_release |
| qwen3 | L10 | 10 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.22 | 0.31 | +0.09 | 0.03 | 0.02 | -0.01 | 0.33 | 0.00 | +0.11 | +0.03 | +0.06 | positive_add_or_release |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.22 | 0.31 | +0.09 | 0.03 | 0.03 | +0.00 | 0.34 | 0.00 | +0.11 | +0.01 | +0.08 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.22 | 0.33 | +0.11 | 0.03 | 0.06 | +0.03 | 0.28 | 0.00 | +0.10 | -0.02 | +0.14 | positive_add_or_release |

### Best drops by condition

| model | condition | best row | clean delta | score delta | random delta | class |
|---|---|---|---:|---:|---:|---|
| qwen3 | add_perp | L10 forbidden_natural_qa top_p | +0.01 | +0.02 | +0.01 | flat |
| qwen3 | resid_remove_perp | all forbidden_natural_qa top_p | +0.02 | +0.05 | +0.00 | flat |
| qwen3 | resid_remove_full | L10+L14 forbidden_natural_qa top_p | +0.01 | +0.05 | +0.01 | flat |
| qwen3 | resid_remove_random_perp | L14 forbidden_sentence_completion temperature | -0.07 | -0.06 | -0.07 | weak_drop |
| qwen3 | resid_remove_perp_add_perp | L10 forbidden_natural_qa top_p | +0.01 | +0.00 | +0.01 | flat |
| qwen3 | attn_remove_perp | L10+L14 forbidden_sentence_completion temperature | -0.05 | -0.05 | -0.03 | flat |
| qwen3 | mlp_remove_perp | L10 forbidden_natural_qa top_p | -0.02 | -0.03 | +0.01 | flat |
| qwen3 | attn_remove_full | all forbidden_sentence_completion temperature | -0.02 | -0.02 | +0.02 | flat |
| qwen3 | mlp_remove_full | L10+L14 forbidden_sentence_completion temperature | -0.06 | -0.06 | -0.03 | weak_drop |

## glm4

pair=vehicle_tool, window=[24, 26, 28], combos=['L24', 'L28', 'L24+L28', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_full', 'resid_remove_random_perp', 'resid_remove_perp_add_perp', 'attn_remove_perp', 'mlp_remove_perp', 'attn_remove_full', 'mlp_remove_full'], scaffold_modes=['forbidden_sentence_completion:temperature', 'forbidden_natural_qa:top_p', 'forbidden_definition:top_p'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], remove_scale=1.0, add_alpha=6.0

| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.09 | -0.20 | 0.00 | 0.00 | +0.00 | 0.67 | 0.00 | -0.20 | +0.00 | -0.20 | necessity_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.14 | -0.16 | 0.00 | 0.02 | +0.02 | 0.69 | 0.00 | -0.18 | +0.00 | -0.16 | necessity_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.15 | -0.15 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.17 | +0.01 | -0.16 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.24 | -0.15 | 0.01 | 0.00 | -0.01 | 0.04 | 0.00 | -0.16 | -0.03 | -0.11 | necessity_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.16 | -0.14 | 0.00 | 0.01 | +0.01 | 0.66 | 0.00 | -0.19 | +0.01 | -0.15 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.16 | -0.14 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.13 | -0.02 | -0.11 | necessity_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.26 | -0.12 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | -0.14 | -0.01 | -0.11 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.18 | -0.11 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.14 | -0.02 | -0.09 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.19 | -0.10 | 0.00 | 0.00 | +0.00 | 0.61 | 0.00 | -0.13 | +0.01 | -0.11 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.19 | -0.10 | 0.00 | 0.00 | +0.00 | 0.61 | 0.00 | -0.10 | +0.01 | -0.11 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.29 | -0.09 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.15 | +0.03 | -0.12 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.12 | -0.03 | -0.06 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.12 | +0.03 | -0.12 | weak_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.00 | -0.01 | 0.04 | 0.00 | -0.11 | -0.01 | -0.08 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.29 | -0.09 | 0.01 | 0.01 | +0.00 | 0.07 | 0.00 | -0.10 | +0.00 | -0.09 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.30 | -0.08 | 0.01 | 0.00 | -0.01 | 0.03 | 0.00 | -0.10 | +0.03 | -0.11 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.30 | -0.08 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.09 | -0.03 | -0.05 | weak_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.21 | -0.08 | 0.00 | 0.00 | +0.00 | 0.73 | 0.00 | -0.06 | +0.01 | -0.09 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.31 | -0.07 | 0.01 | 0.01 | +0.00 | 0.07 | 0.00 | -0.08 | +0.00 | -0.07 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.31 | -0.07 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | -0.08 | +0.00 | -0.07 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.22 | -0.07 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.07 | -0.02 | -0.05 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.22 | -0.07 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.05 | +0.01 | -0.08 | weak_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.65 | 0.00 | -0.06 | +0.01 | -0.07 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.06 | +0.01 | -0.07 | weak_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.05 | +0.00 | -0.06 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.05 | -0.02 | -0.04 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.05 | +0.01 | -0.07 | weak_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.04 | +0.00 | -0.06 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.09 | -0.03 | -0.03 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.07 | +0.03 | -0.09 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.02 | +0.01 | 0.10 | 0.00 | -0.06 | +0.00 | -0.06 | weak_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.05 | 0.00 | -0.07 | -0.01 | -0.04 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.07 | +0.00 | -0.05 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.01 | +0.00 | -0.05 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.24 | -0.05 | 0.00 | 0.00 | +0.00 | 0.71 | 0.01 | -0.05 | +0.01 | -0.06 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.26 | -0.05 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.12 | +0.07 | -0.12 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.34 | -0.04 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.11 | -0.03 | -0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.34 | -0.04 | 0.01 | 0.01 | +0.00 | 0.05 | 0.00 | -0.07 | -0.03 | -0.01 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.34 | -0.04 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.05 | -0.01 | -0.03 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.25 | -0.04 | 0.00 | 0.00 | +0.00 | 0.66 | 0.00 | -0.04 | +0.00 | -0.04 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.25 | -0.04 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.04 | +0.00 | -0.04 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.28 | -0.03 | 0.01 | 0.01 | +0.00 | 0.02 | 0.00 | -0.14 | +0.07 | -0.10 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_random_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.03 | +0.02 | 0.07 | 0.00 | -0.10 | -0.03 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | -0.06 | +0.00 | -0.03 | flat |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.06 | +0.03 | -0.06 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.06 | -0.01 | -0.02 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | -0.04 | -0.01 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.26 | -0.03 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.02 | -0.02 | -0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.26 | -0.03 | 0.00 | 0.00 | +0.00 | 0.64 | 0.00 | -0.01 | -0.02 | -0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.29 | 0.27 | -0.02 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.00 | -0.02 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.29 | 0.27 | -0.02 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | +0.01 | +0.01 | -0.03 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | resid_remove_perp | 0.31 | 0.29 | -0.02 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.10 | +0.05 | -0.07 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | resid_remove_perp | 0.31 | 0.30 | -0.01 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.10 | +0.07 | -0.08 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | resid_remove_perp | 0.31 | 0.30 | -0.01 | 0.01 | 0.00 | -0.01 | 0.03 | 0.00 | -0.07 | +0.07 | -0.08 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_random_perp | 0.39 | 0.38 | -0.01 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.05 | -0.01 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.28 | -0.01 | 0.00 | 0.00 | +0.00 | 0.65 | 0.00 | -0.01 | +0.01 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.39 | 0.38 | -0.01 | 0.01 | 0.02 | +0.01 | 0.06 | 0.00 | +0.03 | -0.03 | +0.02 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_random_perp | 0.39 | 0.39 | +0.00 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.03 | +0.00 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.39 | +0.00 | 0.01 | 0.01 | +0.00 | 0.05 | 0.00 | +0.00 | +0.03 | -0.03 | flat |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.39 | 0.39 | +0.00 | 0.01 | 0.04 | +0.03 | 0.06 | 0.00 | +0.01 | +0.03 | -0.03 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.29 | 0.29 | +0.00 | 0.00 | 0.00 | +0.00 | 0.64 | 0.00 | +0.02 | +0.00 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.29 | +0.00 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | +0.02 | +0.01 | -0.01 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.29 | 0.30 | +0.01 | 0.00 | 0.00 | +0.00 | 0.74 | 0.00 | +0.02 | +0.01 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.29 | 0.30 | +0.01 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | +0.03 | +0.01 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.29 | 0.30 | +0.01 | 0.00 | 0.00 | +0.00 | 0.64 | 0.00 | +0.03 | +0.01 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | add_perp | 0.39 | 0.40 | +0.01 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | +0.05 | -0.01 | +0.02 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.39 | 0.40 | +0.01 | 0.01 | 0.00 | -0.01 | 0.08 | 0.00 | +0.09 | -0.01 | +0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.32 | +0.01 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.05 | +0.05 | -0.04 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.32 | +0.01 | 0.01 | 0.02 | +0.01 | 0.01 | 0.00 | -0.05 | +0.06 | -0.05 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | attn_remove_perp | 0.31 | 0.33 | +0.02 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.00 | +0.07 | -0.05 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.29 | 0.31 | +0.02 | 0.00 | 0.01 | +0.01 | 0.76 | 0.01 | +0.01 | +0.00 | +0.02 | flat |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_random_perp | 0.39 | 0.42 | +0.03 | 0.01 | 0.02 | +0.01 | 0.08 | 0.00 | -0.02 | +0.03 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | mlp_remove_full | 0.31 | 0.34 | +0.03 | 0.01 | 0.01 | +0.00 | 0.02 | 0.00 | +0.00 | +0.06 | -0.03 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.31 | 0.34 | +0.03 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.03 | +0.07 | -0.04 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | attn_remove_full | 0.31 | 0.34 | +0.03 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | +0.03 | +0.07 | -0.04 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | attn_remove_full | 0.31 | 0.34 | +0.03 | 0.01 | 0.02 | +0.01 | 0.01 | 0.00 | +0.03 | +0.07 | -0.04 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.33 | +0.04 | 0.00 | 0.00 | +0.00 | 0.65 | 0.00 | +0.04 | +0.01 | +0.03 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | attn_remove_perp | 0.31 | 0.35 | +0.04 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | +0.01 | +0.07 | -0.03 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.31 | 0.35 | +0.04 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.04 | +0.06 | -0.02 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | attn_remove_full | 0.31 | 0.35 | +0.04 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | +0.05 | +0.06 | -0.02 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.31 | 0.35 | +0.04 | 0.01 | 0.02 | +0.01 | 0.01 | 0.00 | +0.05 | +0.06 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.31 | 0.35 | +0.04 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.06 | +0.05 | -0.01 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.31 | 0.35 | +0.04 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | +0.08 | +0.07 | -0.03 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | mlp_remove_full | 0.31 | 0.36 | +0.05 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.03 | +0.07 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | attn_remove_perp | 0.31 | 0.36 | +0.05 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.03 | +0.05 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | attn_remove_full | 0.31 | 0.36 | +0.05 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | +0.04 | +0.05 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.31 | 0.36 | +0.05 | 0.01 | 0.02 | +0.01 | 0.00 | 0.00 | +0.06 | +0.07 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.29 | 0.34 | +0.05 | 0.00 | 0.01 | +0.01 | 0.70 | 0.00 | +0.07 | -0.02 | +0.07 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | attn_remove_perp | 0.31 | 0.36 | +0.05 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.07 | +0.06 | -0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.31 | 0.36 | +0.05 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.08 | +0.05 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | resid_remove_perp | 0.31 | 0.38 | +0.06 | 0.01 | 0.03 | +0.02 | 0.01 | 0.00 | -0.02 | +0.06 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.31 | 0.38 | +0.06 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | +0.04 | +0.05 | +0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | mlp_remove_full | 0.31 | 0.38 | +0.06 | 0.01 | 0.00 | -0.01 | 0.03 | 0.00 | +0.06 | +0.05 | +0.01 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.31 | 0.38 | +0.06 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | +0.07 | +0.06 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.31 | 0.38 | +0.06 | 0.01 | 0.02 | +0.01 | 0.01 | 0.00 | +0.08 | +0.07 | -0.01 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | add_perp | 0.39 | 0.45 | +0.06 | 0.01 | 0.00 | -0.01 | 0.08 | 0.00 | +0.11 | +0.00 | +0.06 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.36 | +0.07 | 0.00 | 0.01 | +0.01 | 0.68 | 0.00 | +0.09 | +0.00 | +0.07 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | mlp_remove_full | 0.31 | 0.39 | +0.07 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | +0.09 | +0.07 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.31 | 0.39 | +0.07 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | +0.09 | +0.07 | +0.00 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.31 | 0.39 | +0.07 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 | +0.09 | +0.07 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.44 | +0.12 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.17 | +0.06 | +0.06 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.46 | +0.15 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | +0.26 | +0.07 | +0.07 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | add_perp | 0.39 | 0.53 | +0.15 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | +0.28 | +0.03 | +0.11 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | add_perp | 0.39 | 0.55 | +0.17 | 0.01 | 0.00 | -0.01 | 0.05 | 0.00 | +0.27 | -0.03 | +0.20 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.49 | +0.18 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.31 | +0.05 | +0.12 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.49 | +0.20 | 0.00 | 0.00 | +0.00 | 0.60 | 0.00 | +0.21 | -0.02 | +0.22 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.54 | +0.23 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | +0.39 | +0.07 | +0.16 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.56 | +0.27 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | +0.29 | +0.01 | +0.26 | positive_add_or_release |

### Best drops by condition

| model | condition | best row | clean delta | score delta | random delta | class |
|---|---|---|---:|---:|---:|---|
| glm4 | add_perp | L24 forbidden_definition top_p | +0.01 | +0.05 | -0.01 | flat |
| glm4 | resid_remove_perp | L24 forbidden_sentence_completion temperature | -0.20 | -0.20 | +0.00 | necessity_drop |
| glm4 | resid_remove_full | L24 forbidden_sentence_completion temperature | -0.16 | -0.18 | +0.00 | necessity_drop |
| glm4 | resid_remove_random_perp | L24+L28 forbidden_definition top_p | -0.03 | -0.10 | -0.03 | flat |
| glm4 | resid_remove_perp_add_perp | L28 forbidden_definition top_p | -0.05 | -0.01 | +0.00 | flat |
| glm4 | attn_remove_perp | all forbidden_sentence_completion temperature | -0.06 | -0.06 | +0.01 | weak_drop |
| glm4 | mlp_remove_perp | all forbidden_sentence_completion temperature | -0.07 | -0.05 | +0.01 | weak_drop |
| glm4 | attn_remove_full | L28 forbidden_sentence_completion temperature | -0.08 | -0.06 | +0.01 | weak_drop |
| glm4 | mlp_remove_full | all forbidden_definition top_p | -0.09 | -0.15 | +0.03 | weak_drop |

## deepseek7b

pair=vehicle_tool, window=[16, 18, 20], combos=['L16', 'L20', 'L16+L20', 'all'], conditions=['baseline', 'add_perp', 'resid_remove_perp', 'resid_remove_full', 'resid_remove_random_perp', 'resid_remove_perp_add_perp', 'attn_remove_perp', 'mlp_remove_perp', 'attn_remove_full', 'mlp_remove_full'], scaffold_modes=['forbidden_sentence_completion:temperature', 'forbidden_natural_qa:top_p', 'forbidden_definition:top_p'], train_n=12, test_n=12, sample_seeds=[101, 103, 107, 109, 113, 127, 131, 137], remove_scale=1.0, add_alpha=6.0

| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.14 | -0.05 | 0.19 | 0.22 | +0.03 | 0.21 | 0.00 | -0.09 | -0.01 | -0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.14 | -0.05 | 0.19 | 0.22 | +0.03 | 0.18 | 0.00 | -0.08 | -0.01 | -0.04 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.14 | -0.05 | 0.19 | 0.17 | -0.02 | 0.20 | 0.00 | -0.03 | -0.05 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.14 | -0.04 | 0.02 | 0.00 | -0.02 | 0.53 | 0.01 | -0.03 | -0.04 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | resid_remove_random_perp | 0.16 | 0.11 | -0.04 | 0.10 | 0.16 | +0.05 | 0.18 | 0.11 | -0.14 | -0.04 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.20 | +0.01 | 0.20 | 0.00 | -0.07 | -0.03 | -0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.19 | +0.00 | 0.21 | 0.01 | -0.06 | -0.03 | -0.01 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.15 | -0.04 | 0.19 | 0.21 | +0.02 | 0.22 | 0.00 | -0.05 | -0.01 | -0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.19 | +0.00 | 0.23 | 0.00 | -0.02 | -0.05 | +0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.21 | +0.02 | 0.20 | 0.00 | -0.08 | -0.03 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.12 | -0.03 | 0.10 | 0.16 | +0.05 | 0.20 | 0.08 | -0.08 | -0.04 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.24 | +0.05 | 0.18 | 0.00 | -0.07 | -0.03 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.23 | 0.00 | -0.06 | -0.05 | +0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.19 | +0.00 | 0.20 | 0.01 | -0.05 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.18 | 0.00 | -0.05 | -0.01 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.22 | 0.00 | -0.05 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.03 | -0.01 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.15 | -0.03 | 0.02 | 0.02 | +0.00 | 0.50 | 0.00 | -0.03 | -0.03 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.21 | 0.00 | -0.03 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.18 | 0.15 | -0.03 | 0.02 | 0.01 | -0.01 | 0.54 | 0.00 | -0.02 | -0.04 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.15 | -0.03 | 0.02 | 0.01 | -0.01 | 0.59 | 0.00 | -0.02 | +0.01 | -0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.17 | -0.02 | 0.20 | 0.00 | +0.00 | -0.01 | -0.02 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.17 | -0.02 | 0.25 | 0.00 | +0.02 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.14 | -0.02 | 0.10 | 0.17 | +0.06 | 0.22 | 0.11 | -0.14 | +0.01 | -0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | attn_remove_perp | 0.16 | 0.14 | -0.02 | 0.10 | 0.17 | +0.06 | 0.23 | 0.09 | -0.11 | +0.02 | -0.04 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.14 | -0.02 | 0.10 | 0.16 | +0.05 | 0.21 | 0.09 | -0.09 | +0.02 | -0.04 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.23 | 0.00 | -0.05 | -0.03 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.04 | -0.03 | +0.01 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.24 | 0.00 | -0.04 | -0.05 | +0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.22 | +0.03 | 0.21 | 0.00 | -0.04 | -0.05 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.22 | 0.00 | -0.04 | -0.01 | -0.01 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.18 | -0.01 | 0.23 | 0.00 | -0.03 | -0.05 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.48 | 0.01 | -0.03 | -0.04 | +0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.45 | 0.01 | -0.03 | -0.03 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.17 | -0.02 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.03 | -0.03 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.58 | 0.00 | -0.02 | +0.01 | -0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.18 | 0.16 | -0.02 | 0.02 | 0.01 | -0.01 | 0.55 | 0.01 | -0.02 | -0.04 | +0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.18 | 0.17 | -0.01 | 0.02 | 0.02 | +0.00 | 0.67 | 0.01 | -0.03 | -0.03 | +0.02 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | mlp_remove_perp | 0.16 | 0.15 | -0.01 | 0.10 | 0.17 | +0.06 | 0.21 | 0.10 | -0.10 | +0.02 | -0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | attn_remove_perp | 0.16 | 0.15 | -0.01 | 0.10 | 0.18 | +0.07 | 0.26 | 0.09 | -0.09 | +0.01 | -0.02 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.18 | -0.01 | 0.19 | 0.20 | +0.01 | 0.21 | 0.00 | -0.05 | -0.03 | +0.02 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.15 | -0.01 | 0.10 | 0.12 | +0.02 | 0.22 | 0.08 | -0.04 | +0.03 | -0.04 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | attn_remove_perp | 0.16 | 0.15 | -0.01 | 0.10 | 0.14 | +0.03 | 0.21 | 0.07 | -0.03 | +0.03 | -0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.19 | 0.18 | -0.01 | 0.19 | 0.22 | +0.03 | 0.18 | 0.00 | -0.02 | -0.01 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.18 | -0.01 | 0.19 | 0.17 | -0.02 | 0.23 | 0.00 | +0.00 | -0.01 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.18 | -0.01 | 0.19 | 0.19 | +0.00 | 0.24 | 0.00 | +0.01 | -0.05 | +0.04 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | add_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.17 | +0.06 | 0.25 | 0.14 | -0.14 | +0.03 | -0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | mlp_remove_full | 0.16 | 0.16 | +0.00 | 0.10 | 0.18 | +0.07 | 0.22 | 0.08 | -0.09 | +0.02 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.15 | +0.04 | 0.25 | 0.09 | -0.06 | -0.04 | +0.04 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.04 | -0.03 | +0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.04 | -0.03 | +0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.20 | 0.00 | -0.04 | -0.03 | +0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | resid_remove_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.14 | +0.03 | 0.26 | 0.07 | -0.04 | +0.03 | -0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.03 | -0.05 | +0.05 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.18 | 0.18 | +0.00 | 0.02 | 0.05 | +0.03 | 0.53 | 0.00 | -0.03 | +0.09 | -0.09 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | add_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.14 | +0.03 | 0.24 | 0.07 | -0.03 | +0.01 | -0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | attn_remove_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.14 | +0.03 | 0.21 | 0.07 | -0.02 | -0.04 | +0.04 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.18 | +0.00 | 0.02 | 0.03 | +0.01 | 0.44 | 0.01 | -0.02 | +0.09 | -0.09 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.19 | +0.00 | 0.19 | 0.21 | +0.02 | 0.19 | 0.00 | -0.01 | -0.05 | +0.05 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.19 | +0.00 | 0.22 | 0.00 | -0.01 | -0.03 | +0.03 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.18 | 0.18 | +0.00 | 0.02 | 0.03 | +0.01 | 0.51 | 0.00 | -0.01 | -0.03 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.18 | +0.00 | 0.02 | 0.01 | -0.01 | 0.52 | 0.00 | +0.01 | -0.04 | +0.04 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.18 | 0.18 | +0.00 | 0.02 | 0.01 | -0.01 | 0.52 | 0.00 | +0.01 | -0.03 | +0.03 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | mlp_remove_perp | 0.16 | 0.17 | +0.01 | 0.10 | 0.18 | +0.07 | 0.23 | 0.11 | -0.11 | -0.04 | +0.05 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | resid_remove_random_perp | 0.16 | 0.17 | +0.01 | 0.10 | 0.15 | +0.04 | 0.22 | 0.10 | -0.06 | +0.01 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | mlp_remove_full | 0.16 | 0.17 | +0.01 | 0.10 | 0.16 | +0.05 | 0.21 | 0.09 | -0.05 | +0.03 | -0.02 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | mlp_remove_full | 0.16 | 0.17 | +0.01 | 0.10 | 0.14 | +0.03 | 0.27 | 0.09 | -0.03 | +0.01 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | resid_remove_full | 0.18 | 0.19 | +0.01 | 0.02 | 0.03 | +0.01 | 0.53 | 0.00 | +0.00 | +0.09 | -0.08 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.19 | +0.01 | 0.02 | 0.02 | +0.00 | 0.46 | 0.00 | +0.01 | +0.01 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.20 | +0.01 | 0.19 | 0.20 | +0.01 | 0.20 | 0.00 | +0.01 | -0.03 | +0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.18 | 0.19 | +0.01 | 0.02 | 0.00 | -0.02 | 0.62 | 0.00 | +0.03 | -0.04 | +0.05 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.18 | 0.20 | +0.02 | 0.02 | 0.02 | +0.00 | 0.52 | 0.01 | +0.01 | +0.09 | -0.07 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.18 | 0.20 | +0.02 | 0.02 | 0.02 | +0.00 | 0.49 | 0.00 | +0.02 | +0.09 | -0.07 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.18 | 0.20 | +0.02 | 0.02 | 0.01 | -0.01 | 0.50 | 0.00 | +0.03 | -0.04 | +0.06 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.18 | 0.20 | +0.02 | 0.02 | 0.01 | -0.01 | 0.51 | 0.00 | +0.03 | -0.03 | +0.05 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | resid_remove_full | 0.16 | 0.18 | +0.02 | 0.10 | 0.19 | +0.08 | 0.22 | 0.14 | -0.12 | +0.01 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | add_perp | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.25 | 0.10 | -0.11 | +0.02 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | add_perp | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.22 | 0.09 | -0.09 | -0.04 | +0.06 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | resid_remove_random_perp | 0.16 | 0.18 | +0.02 | 0.10 | 0.10 | +0.00 | 0.19 | 0.08 | +0.01 | +0.02 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.19 | 0.21 | +0.02 | 0.19 | 0.18 | -0.01 | 0.21 | 0.00 | +0.03 | -0.03 | +0.05 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | resid_remove_perp | 0.16 | 0.19 | +0.03 | 0.10 | 0.22 | +0.11 | 0.24 | 0.09 | -0.14 | -0.04 | +0.07 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | mlp_remove_full | 0.16 | 0.19 | +0.03 | 0.10 | 0.16 | +0.05 | 0.19 | 0.09 | -0.03 | -0.04 | +0.07 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | mlp_remove_perp | 0.16 | 0.19 | +0.03 | 0.10 | 0.14 | +0.03 | 0.28 | 0.07 | -0.02 | +0.03 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | resid_remove_random_perp | 0.16 | 0.19 | +0.03 | 0.10 | 0.12 | +0.02 | 0.22 | 0.08 | +0.00 | +0.03 | +0.00 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.18 | 0.21 | +0.03 | 0.02 | 0.04 | +0.02 | 0.57 | 0.01 | +0.00 | +0.09 | -0.06 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.18 | 0.21 | +0.03 | 0.02 | 0.03 | +0.01 | 0.62 | 0.01 | +0.01 | +0.01 | +0.02 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.21 | +0.03 | 0.02 | 0.03 | +0.01 | 0.47 | 0.00 | +0.02 | +0.09 | -0.06 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | resid_remove_full | 0.18 | 0.21 | +0.03 | 0.02 | 0.02 | +0.00 | 0.56 | 0.01 | +0.02 | -0.04 | +0.07 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.16 | 0.20 | +0.04 | 0.10 | 0.21 | +0.10 | 0.25 | 0.14 | -0.14 | +0.02 | +0.02 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.16 | 0.20 | +0.04 | 0.10 | 0.18 | +0.07 | 0.25 | 0.15 | -0.10 | +0.03 | +0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | resid_remove_full | 0.16 | 0.20 | +0.04 | 0.10 | 0.16 | +0.05 | 0.24 | 0.09 | -0.03 | -0.04 | +0.08 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.16 | 0.20 | +0.04 | 0.10 | 0.15 | +0.04 | 0.25 | 0.07 | +0.01 | +0.01 | +0.03 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.22 | +0.04 | 0.02 | 0.02 | +0.00 | 0.52 | 0.01 | +0.03 | -0.03 | +0.07 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.18 | 0.22 | +0.04 | 0.02 | 0.01 | -0.01 | 0.50 | 0.02 | +0.03 | +0.01 | +0.03 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.18 | 0.22 | +0.04 | 0.02 | 0.00 | -0.02 | 0.61 | 0.00 | +0.05 | -0.03 | +0.07 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | resid_remove_full | 0.18 | 0.23 | +0.05 | 0.02 | 0.03 | +0.01 | 0.65 | 0.01 | +0.03 | -0.03 | +0.08 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | resid_remove_full | 0.18 | 0.23 | +0.05 | 0.02 | 0.02 | +0.00 | 0.69 | 0.01 | +0.04 | +0.01 | +0.04 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.18 | 0.23 | +0.05 | 0.02 | 0.00 | -0.02 | 0.55 | 0.00 | +0.07 | +0.01 | +0.04 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | resid_remove_perp | 0.16 | 0.21 | +0.05 | 0.10 | 0.22 | +0.11 | 0.25 | 0.19 | -0.18 | +0.02 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | mlp_remove_perp | 0.16 | 0.21 | +0.05 | 0.10 | 0.12 | +0.02 | 0.20 | 0.08 | +0.02 | +0.01 | +0.04 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | resid_remove_full | 0.16 | 0.22 | +0.06 | 0.10 | 0.21 | +0.10 | 0.22 | 0.10 | -0.08 | +0.03 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | resid_remove_perp | 0.16 | 0.22 | +0.06 | 0.10 | 0.19 | +0.08 | 0.25 | 0.10 | -0.04 | +0.01 | +0.05 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.18 | 0.24 | +0.06 | 0.02 | 0.02 | +0.00 | 0.61 | 0.01 | +0.05 | -0.04 | +0.10 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.18 | 0.24 | +0.06 | 0.02 | 0.01 | -0.01 | 0.68 | 0.01 | +0.06 | +0.01 | +0.05 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.18 | 0.25 | +0.07 | 0.02 | 0.01 | -0.01 | 0.55 | 0.01 | +0.07 | +0.09 | -0.02 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.27 | +0.09 | 0.02 | 0.04 | +0.02 | 0.49 | 0.01 | +0.06 | +0.01 | +0.08 | positive_add_or_release |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.27 | +0.09 | 0.02 | 0.03 | +0.01 | 0.50 | 0.01 | +0.07 | +0.09 | +0.00 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | resid_remove_full | 0.16 | 0.25 | +0.09 | 0.10 | 0.18 | +0.07 | 0.23 | 0.15 | -0.05 | +0.02 | +0.07 | positive_add_or_release |

### Best drops by condition

| model | condition | best row | clean delta | score delta | random delta | class |
|---|---|---|---:|---:|---:|---|
| deepseek7b | add_perp | L16 forbidden_natural_qa top_p | -0.03 | -0.06 | -0.05 | flat |
| deepseek7b | resid_remove_perp | L16 forbidden_natural_qa top_p | -0.04 | -0.02 | -0.05 | flat |
| deepseek7b | resid_remove_full | L20 forbidden_natural_qa top_p | -0.05 | -0.08 | -0.01 | flat |
| deepseek7b | resid_remove_random_perp | L16 forbidden_natural_qa top_p | -0.05 | -0.03 | -0.05 | flat |
| deepseek7b | resid_remove_perp_add_perp | all forbidden_natural_qa top_p | -0.02 | -0.04 | -0.03 | flat |
| deepseek7b | attn_remove_perp | all forbidden_natural_qa top_p | -0.04 | -0.07 | -0.03 | flat |
| deepseek7b | mlp_remove_perp | L20 forbidden_natural_qa top_p | -0.05 | -0.09 | -0.01 | flat |
| deepseek7b | attn_remove_full | L16+L20 forbidden_definition top_p | -0.03 | -0.08 | -0.04 | flat |
| deepseek7b | mlp_remove_full | L20 forbidden_natural_qa top_p | -0.04 | -0.05 | -0.01 | flat |

## Cross-Model Strongest Drops

| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.09 | -0.20 | 0.00 | 0.00 | +0.00 | 0.67 | 0.00 | -0.20 | +0.00 | -0.20 | necessity_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.14 | -0.16 | 0.00 | 0.02 | +0.02 | 0.69 | 0.00 | -0.18 | +0.00 | -0.16 | necessity_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.15 | -0.15 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.17 | +0.01 | -0.16 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.24 | -0.15 | 0.01 | 0.00 | -0.01 | 0.04 | 0.00 | -0.16 | -0.03 | -0.11 | necessity_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.16 | -0.14 | 0.00 | 0.01 | +0.01 | 0.66 | 0.00 | -0.19 | +0.01 | -0.15 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.16 | -0.14 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.13 | -0.02 | -0.11 | necessity_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.26 | -0.12 | 0.01 | 0.02 | +0.01 | 0.03 | 0.00 | -0.14 | -0.01 | -0.11 | necessity_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.18 | -0.11 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.14 | -0.02 | -0.09 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_perp | 0.29 | 0.19 | -0.10 | 0.00 | 0.00 | +0.00 | 0.61 | 0.00 | -0.13 | +0.01 | -0.11 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | resid_remove_full | 0.29 | 0.19 | -0.10 | 0.00 | 0.00 | +0.00 | 0.61 | 0.00 | -0.10 | +0.01 | -0.11 | necessity_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.29 | -0.09 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.15 | +0.03 | -0.12 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.12 | -0.03 | -0.06 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.12 | +0.03 | -0.12 | weak_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.29 | -0.09 | 0.01 | 0.00 | -0.01 | 0.04 | 0.00 | -0.11 | -0.01 | -0.08 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.29 | -0.09 | 0.01 | 0.01 | +0.00 | 0.07 | 0.00 | -0.10 | +0.00 | -0.09 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.30 | -0.08 | 0.01 | 0.00 | -0.01 | 0.03 | 0.00 | -0.10 | +0.03 | -0.11 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.30 | -0.08 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.09 | -0.03 | -0.05 | weak_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.21 | -0.08 | 0.00 | 0.00 | +0.00 | 0.73 | 0.00 | -0.06 | +0.01 | -0.09 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_perp | 0.39 | 0.31 | -0.07 | 0.01 | 0.01 | +0.00 | 0.07 | 0.00 | -0.08 | +0.00 | -0.07 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_full | 0.39 | 0.31 | -0.07 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | -0.08 | +0.00 | -0.07 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.22 | -0.07 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.07 | -0.02 | -0.05 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.22 | -0.07 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.05 | +0.01 | -0.08 | weak_drop |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.47 | -0.07 | 0.01 | 0.01 | +0.00 | 0.61 | 0.00 | -0.06 | -0.07 | +0.00 | weak_drop |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.65 | 0.00 | -0.06 | +0.01 | -0.07 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.69 | 0.00 | -0.06 | +0.01 | -0.07 | weak_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.05 | +0.00 | -0.06 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | -0.05 | -0.02 | -0.04 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.05 | +0.01 | -0.07 | weak_drop |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.23 | -0.06 | 0.00 | 0.00 | +0.00 | 0.71 | 0.00 | -0.04 | +0.00 | -0.06 | weak_drop |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.09 | -0.03 | -0.03 | weak_drop |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.07 | +0.03 | -0.09 | weak_drop |
| glm4 | L28 | 28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.32 | -0.06 | 0.01 | 0.02 | +0.01 | 0.10 | 0.00 | -0.06 | +0.00 | -0.06 | weak_drop |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.48 | -0.06 | 0.01 | 0.01 | +0.00 | 0.61 | 0.01 | -0.06 | -0.03 | -0.03 | weak_drop |
| glm4 | L24 | 24 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.05 | 0.00 | -0.07 | -0.01 | -0.04 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.07 | +0.00 | -0.05 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | resid_remove_perp_add_perp | 0.39 | 0.33 | -0.05 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.01 | +0.00 | -0.05 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.14 | -0.05 | 0.19 | 0.22 | +0.03 | 0.21 | 0.00 | -0.09 | -0.01 | -0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.14 | -0.05 | 0.19 | 0.22 | +0.03 | 0.18 | 0.00 | -0.08 | -0.01 | -0.04 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.24 | -0.05 | 0.00 | 0.00 | +0.00 | 0.71 | 0.01 | -0.05 | +0.01 | -0.06 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.14 | -0.05 | 0.19 | 0.17 | -0.02 | 0.20 | 0.00 | -0.03 | -0.05 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.26 | -0.05 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.12 | +0.07 | -0.12 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.49 | -0.05 | 0.01 | 0.02 | +0.01 | 0.62 | 0.01 | -0.06 | -0.07 | +0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.54 | 0.49 | -0.05 | 0.01 | 0.01 | +0.00 | 0.62 | 0.01 | -0.05 | -0.03 | -0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.34 | -0.04 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.11 | -0.03 | -0.01 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.34 | -0.04 | 0.01 | 0.01 | +0.00 | 0.05 | 0.00 | -0.07 | -0.03 | -0.01 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | mlp_remove_full | 0.39 | 0.34 | -0.04 | 0.01 | 0.01 | +0.00 | 0.06 | 0.00 | -0.05 | -0.01 | -0.03 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | attn_remove_full | 0.29 | 0.25 | -0.04 | 0.00 | 0.00 | +0.00 | 0.66 | 0.00 | -0.04 | +0.00 | -0.04 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.29 | 0.25 | -0.04 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.04 | +0.00 | -0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.14 | -0.04 | 0.02 | 0.00 | -0.02 | 0.53 | 0.01 | -0.03 | -0.04 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | resid_remove_random_perp | 0.16 | 0.11 | -0.04 | 0.10 | 0.16 | +0.05 | 0.18 | 0.11 | -0.14 | -0.04 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.20 | +0.01 | 0.20 | 0.00 | -0.07 | -0.03 | -0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.19 | +0.00 | 0.21 | 0.01 | -0.06 | -0.03 | -0.01 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.15 | -0.04 | 0.19 | 0.21 | +0.02 | 0.22 | 0.00 | -0.05 | -0.01 | -0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.15 | -0.04 | 0.19 | 0.19 | +0.00 | 0.23 | 0.00 | -0.02 | -0.05 | +0.01 | flat |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | resid_remove_full | 0.31 | 0.28 | -0.03 | 0.01 | 0.01 | +0.00 | 0.02 | 0.00 | -0.14 | +0.07 | -0.10 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | resid_remove_random_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.03 | +0.02 | 0.07 | 0.00 | -0.10 | -0.03 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | mlp_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.21 | +0.02 | 0.20 | 0.00 | -0.08 | -0.03 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.12 | -0.03 | 0.10 | 0.16 | +0.05 | 0.20 | 0.08 | -0.08 | -0.04 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.24 | +0.05 | 0.18 | 0.00 | -0.07 | -0.03 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | mlp_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | -0.06 | +0.00 | -0.03 | flat |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.04 | 0.00 | -0.06 | +0.03 | -0.06 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.23 | 0.00 | -0.06 | -0.05 | +0.02 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | attn_remove_perp | 0.39 | 0.35 | -0.03 | 0.01 | 0.02 | +0.01 | 0.05 | 0.00 | -0.06 | -0.01 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.19 | +0.00 | 0.20 | 0.01 | -0.05 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | resid_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.18 | 0.00 | -0.05 | -0.01 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.22 | 0.00 | -0.05 | -0.03 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | attn_remove_full | 0.39 | 0.35 | -0.03 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | -0.04 | -0.01 | -0.02 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.54 | 0.51 | -0.03 | 0.01 | 0.01 | +0.00 | 0.56 | 0.01 | -0.03 | -0.03 | +0.00 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.03 | -0.01 | -0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.18 | 0.15 | -0.03 | 0.02 | 0.02 | +0.00 | 0.50 | 0.00 | -0.03 | -0.03 | +0.00 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.21 | 0.00 | -0.03 | -0.03 | +0.00 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.29 | 0.26 | -0.03 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.02 | -0.02 | -0.01 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | attn_remove_perp | 0.18 | 0.15 | -0.03 | 0.02 | 0.01 | -0.01 | 0.54 | 0.00 | -0.02 | -0.04 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.15 | -0.03 | 0.02 | 0.01 | -0.01 | 0.59 | 0.00 | -0.02 | +0.01 | -0.04 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.29 | 0.26 | -0.03 | 0.00 | 0.00 | +0.00 | 0.64 | 0.00 | -0.01 | -0.02 | -0.01 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.17 | -0.02 | 0.20 | 0.00 | +0.00 | -0.01 | -0.02 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_random_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.17 | -0.02 | 0.25 | 0.00 | +0.02 | -0.03 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.48 | 0.46 | -0.02 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | -0.03 | +0.01 | -0.03 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | resid_remove_random_perp | 0.29 | 0.27 | -0.02 | 0.00 | 0.00 | +0.00 | 0.68 | 0.00 | -0.00 | -0.02 | +0.00 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | resid_remove_perp_add_perp | 0.29 | 0.27 | -0.02 | 0.00 | 0.00 | +0.00 | 0.70 | 0.00 | +0.01 | +0.01 | -0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.14 | -0.02 | 0.10 | 0.17 | +0.06 | 0.22 | 0.11 | -0.14 | +0.01 | -0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | attn_remove_perp | 0.16 | 0.14 | -0.02 | 0.10 | 0.17 | +0.06 | 0.23 | 0.09 | -0.11 | +0.02 | -0.04 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | attn_remove_full | 0.16 | 0.14 | -0.02 | 0.10 | 0.16 | +0.05 | 0.21 | 0.09 | -0.09 | +0.02 | -0.04 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.23 | 0.00 | -0.05 | -0.03 | +0.01 | flat |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | resid_remove_random_perp | 0.22 | 0.20 | -0.02 | 0.03 | 0.01 | -0.02 | 0.36 | 0.03 | -0.05 | -0.02 | +0.00 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | resid_remove_perp_add_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.04 | -0.03 | +0.01 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | resid_remove_full | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.24 | 0.00 | -0.04 | -0.05 | +0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | attn_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.22 | +0.03 | 0.21 | 0.00 | -0.04 | -0.05 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.22 | 0.00 | -0.04 | -0.01 | -0.01 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | mlp_remove_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.18 | -0.01 | 0.23 | 0.00 | -0.03 | -0.05 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.48 | 0.01 | -0.03 | -0.04 | +0.02 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | attn_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.45 | 0.01 | -0.03 | -0.03 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | attn_remove_full | 0.19 | 0.17 | -0.02 | 0.19 | 0.19 | +0.00 | 0.20 | 0.00 | -0.03 | -0.03 | +0.01 | flat |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.18 | 0.16 | -0.02 | 0.02 | 0.02 | +0.00 | 0.58 | 0.00 | -0.02 | +0.01 | -0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.18 | 0.16 | -0.02 | 0.02 | 0.01 | -0.01 | 0.55 | 0.01 | -0.02 | -0.04 | +0.02 | flat |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | resid_remove_perp | 0.31 | 0.29 | -0.02 | 0.01 | 0.00 | -0.01 | 0.02 | 0.00 | -0.10 | +0.05 | -0.07 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | mlp_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.02 | +0.01 | 0.54 | 0.01 | -0.03 | +0.02 | -0.04 | flat |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | mlp_remove_perp | 0.54 | 0.52 | -0.02 | 0.01 | 0.02 | +0.01 | 0.57 | 0.00 | -0.02 | -0.07 | +0.05 | flat |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.01 | +0.00 | 0.62 | 0.01 | -0.02 | +0.02 | -0.04 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | attn_remove_full | 0.54 | 0.52 | -0.02 | 0.01 | 0.01 | +0.00 | 0.58 | 0.00 | -0.01 | -0.03 | +0.01 | flat |

## Cross-Model Add Checks

| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | class |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm4 | all | 24,26,28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.56 | +0.27 | 0.00 | 0.00 | +0.00 | 0.62 | 0.00 | +0.29 | +0.01 | +0.26 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.54 | +0.23 | 0.01 | 0.03 | +0.02 | 0.00 | 0.00 | +0.39 | +0.07 | +0.16 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.49 | +0.20 | 0.00 | 0.00 | +0.00 | 0.60 | 0.00 | +0.21 | -0.02 | +0.22 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.49 | +0.18 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.31 | +0.05 | +0.12 | positive_add_or_release |
| glm4 | L24+L28 | 24,28 | forbidden_definition | top_p | add_perp | 0.39 | 0.55 | +0.17 | 0.01 | 0.00 | -0.01 | 0.05 | 0.00 | +0.27 | -0.03 | +0.20 | positive_add_or_release |
| glm4 | L24 | 24 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.46 | +0.15 | 0.01 | 0.01 | +0.00 | 0.01 | 0.00 | +0.26 | +0.07 | +0.07 | positive_add_or_release |
| glm4 | all | 24,26,28 | forbidden_definition | top_p | add_perp | 0.39 | 0.53 | +0.15 | 0.01 | 0.00 | -0.01 | 0.06 | 0.00 | +0.28 | +0.03 | +0.11 | positive_add_or_release |
| glm4 | L28 | 28 | forbidden_natural_qa | top_p | add_perp | 0.31 | 0.44 | +0.12 | 0.01 | 0.00 | -0.01 | 0.01 | 0.00 | +0.17 | +0.06 | +0.06 | positive_add_or_release |
| qwen3 | L14 | 14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.64 | +0.09 | 0.01 | 0.01 | +0.00 | 0.64 | 0.00 | +0.10 | -0.07 | +0.17 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.64 | +0.09 | 0.01 | 0.03 | +0.02 | 0.58 | 0.00 | +0.08 | +0.02 | +0.07 | positive_add_or_release |
| deepseek7b | all | 16,18,20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.27 | +0.09 | 0.02 | 0.04 | +0.02 | 0.49 | 0.01 | +0.06 | +0.01 | +0.08 | positive_add_or_release |
| qwen3 | all | 10,12,14 | forbidden_definition | top_p | add_perp | 0.22 | 0.30 | +0.08 | 0.03 | 0.06 | +0.03 | 0.29 | 0.00 | +0.10 | -0.02 | +0.10 | positive_add_or_release |
| qwen3 | L10 | 10 | forbidden_definition | top_p | add_perp | 0.22 | 0.29 | +0.07 | 0.03 | 0.00 | -0.03 | 0.27 | 0.01 | +0.11 | +0.03 | +0.04 | flat |
| glm4 | L24 | 24 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.36 | +0.07 | 0.00 | 0.01 | +0.01 | 0.68 | 0.00 | +0.09 | +0.00 | +0.07 | flat |
| qwen3 | L10 | 10 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.00 | -0.01 | 0.58 | 0.01 | +0.07 | +0.01 | +0.05 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_sentence_completion | temperature | add_perp | 0.54 | 0.60 | +0.06 | 0.01 | 0.03 | +0.02 | 0.57 | 0.01 | +0.04 | -0.03 | +0.09 | flat |
| glm4 | L28 | 28 | forbidden_definition | top_p | add_perp | 0.39 | 0.45 | +0.06 | 0.01 | 0.00 | -0.01 | 0.08 | 0.00 | +0.11 | +0.00 | +0.06 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.11 | +0.01 | +0.04 | flat |
| qwen3 | all | 10,12,14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.53 | +0.05 | 0.07 | 0.10 | +0.03 | 0.00 | 0.00 | +0.09 | +0.00 | +0.05 | flat |
| qwen3 | L14 | 14 | forbidden_definition | top_p | add_perp | 0.22 | 0.26 | +0.04 | 0.03 | 0.01 | -0.02 | 0.33 | 0.01 | +0.09 | +0.01 | +0.03 | flat |
| qwen3 | L10+L14 | 10,14 | forbidden_definition | top_p | add_perp | 0.22 | 0.26 | +0.04 | 0.03 | 0.03 | +0.00 | 0.35 | 0.01 | +0.06 | +0.01 | +0.03 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.22 | +0.04 | 0.02 | 0.02 | +0.00 | 0.52 | 0.01 | +0.03 | -0.03 | +0.07 | flat |
| glm4 | L28 | 28 | forbidden_sentence_completion | temperature | add_perp | 0.29 | 0.33 | +0.04 | 0.00 | 0.00 | +0.00 | 0.65 | 0.00 | +0.04 | +0.01 | +0.03 | flat |
| deepseek7b | L16 | 16 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.21 | +0.03 | 0.02 | 0.03 | +0.01 | 0.47 | 0.00 | +0.02 | +0.09 | -0.06 | flat |
| qwen3 | L14 | 14 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.51 | +0.03 | 0.07 | 0.08 | +0.01 | 0.00 | 0.00 | +0.05 | +0.04 | -0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_definition | top_p | add_perp | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.22 | 0.09 | -0.09 | -0.04 | +0.06 | flat |
| deepseek7b | all | 16,18,20 | forbidden_definition | top_p | add_perp | 0.16 | 0.18 | +0.02 | 0.10 | 0.20 | +0.09 | 0.25 | 0.10 | -0.11 | +0.02 | +0.00 | flat |
| qwen3 | L10 | 10 | forbidden_natural_qa | top_p | add_perp | 0.48 | 0.49 | +0.01 | 0.07 | 0.09 | +0.02 | 0.00 | 0.00 | +0.02 | +0.01 | +0.00 | flat |
| glm4 | L24 | 24 | forbidden_definition | top_p | add_perp | 0.39 | 0.40 | +0.01 | 0.01 | 0.01 | +0.00 | 0.08 | 0.00 | +0.05 | -0.01 | +0.02 | flat |
| deepseek7b | L16 | 16 | forbidden_definition | top_p | add_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.17 | +0.06 | 0.25 | 0.14 | -0.14 | +0.03 | -0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_sentence_completion | temperature | add_perp | 0.18 | 0.18 | +0.00 | 0.02 | 0.01 | -0.01 | 0.52 | 0.00 | +0.01 | -0.04 | +0.04 | flat |
| deepseek7b | L20 | 20 | forbidden_definition | top_p | add_perp | 0.16 | 0.16 | +0.00 | 0.10 | 0.14 | +0.03 | 0.24 | 0.07 | -0.03 | +0.01 | -0.01 | flat |
| deepseek7b | L16+L20 | 16,20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.04 | -0.03 | +0.03 | flat |
| deepseek7b | all | 16,18,20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.19 | +0.00 | 0.19 | 0.22 | +0.03 | 0.19 | 0.00 | -0.04 | -0.03 | +0.03 | flat |
| deepseek7b | L20 | 20 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.17 | -0.02 | 0.19 | 0.20 | +0.01 | 0.22 | 0.00 | -0.04 | -0.01 | -0.01 | flat |
| deepseek7b | L16 | 16 | forbidden_natural_qa | top_p | add_perp | 0.19 | 0.16 | -0.03 | 0.19 | 0.20 | +0.01 | 0.23 | 0.00 | -0.06 | -0.05 | +0.02 | flat |

