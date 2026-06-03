# Phase342 Object-Property Binding Validation Summary

## qwen3

core_validation=True

| baseline | MLP | Attention | Full | n_valid | MLP>Attn |
|---|---:|---:|---:|---:|---:|
| The entity | 93.2 | 9.4 | 97.2 | 19 | True |
| The item | 70.1 | 23.8 | 80.5 | 22 | True |
| The object | 90.9 | 10.9 | 93.8 | 19 | True |
| The thing | 94.1 | 6.6 | 91.5 | 18 | True |

mean_mlp=87.08
mean_attn=12.68
mean_full=90.75
identity_block=identity_L0-2_full
identity_recovery=99.60
identity_plus_compute=100.30
mean_category_accuracy=0.4653
mean_category_chance=0.1429

### Validity

- all_baselines_mlp_gt_attn: True
- identity_recovery_ge_95: True
- min_valid_pairs_ge_12: True
- category_probe_above_chance: True
- passes_core_validation: True

## glm4

core_validation=True

| baseline | MLP | Attention | Full | n_valid | MLP>Attn |
|---|---:|---:|---:|---:|---:|
| The entity | 63.7 | 14.2 | 72.0 | 22 | True |
| The item | 45.5 | 10.9 | 56.1 | 22 | True |
| The object | 65.3 | 16.7 | 74.0 | 19 | True |
| The thing | 56.9 | 2.6 | 62.1 | 17 | True |

mean_mlp=57.85
mean_attn=11.10
mean_full=66.05
identity_block=identity_L0-4_full
identity_recovery=100.00
identity_plus_compute=100.20
mean_category_accuracy=0.4653
mean_category_chance=0.1429

### Validity

- all_baselines_mlp_gt_attn: True
- identity_recovery_ge_95: True
- min_valid_pairs_ge_12: True
- category_probe_above_chance: True
- passes_core_validation: True

## deepseek7b

core_validation=True

| baseline | MLP | Attention | Full | n_valid | MLP>Attn |
|---|---:|---:|---:|---:|---:|
| The entity | 81.6 | -0.6 | 96.0 | 18 | True |
| The item | 56.6 | 16.2 | 83.7 | 15 | True |
| The object | 64.9 | -5.2 | 73.9 | 18 | True |
| The thing | 86.0 | -1.6 | 91.1 | 21 | True |

mean_mlp=72.28
mean_attn=2.20
mean_full=86.17
identity_block=identity_L0-2_full
identity_recovery=116.10
identity_plus_compute=115.00
mean_category_accuracy=0.5000
mean_category_chance=0.1429

### Validity

- all_baselines_mlp_gt_attn: True
- identity_recovery_ge_95: True
- min_valid_pairs_ge_12: True
- category_probe_above_chance: True
- passes_core_validation: True
