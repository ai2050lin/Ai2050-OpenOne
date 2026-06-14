# Phase 116 Cross-model Subspace Basis Component Audit

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, clothing, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- ranks: [8, 16]; scale: 1.5; cumulative sizes: [1, 2, 4, 8, 16]
- component labels: support, release, mixed, weak

## Cross-model Table
| model | category | rank | best single | best random single | best cumulative | support set | release set | mixed set | class |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | number | 8 | mixed0 T-2.90 R+1.59 | 0 T-0.02 R+0.03 | target_sorted4 T-3.12 R+1.42 | NA | release3 T+0.70 R+1.77 | mixed2 T-3.10 R+1.05 | release_components |
| qwen3 | number | 16 | mixed0 T-2.90 R+1.59 | 4 T-0.28 R+0.06 | target_sorted4 T-3.67 R+0.86 | NA | release3 T+0.70 R+1.77 | mixed2 T-3.10 R+1.05 | release_components |
| qwen3 | container | 8 | support1 T-0.66 R+0.00 | 3 T-0.03 R+0.02 | target_sorted8 T-2.06 R+1.54 | support2 T-1.10 R+0.00 | release2 T+0.15 R+1.26 | mixed1 T-0.46 R+2.05 | release_components |
| qwen3 | container | 16 | support1 T-0.66 R+0.00 | 5 T-0.04 R+0.03 | target_sorted8 T-2.66 R+0.40 | support3 T-1.65 R+0.00 | release3 T+0.24 R+1.88 | mixed1 T-0.46 R+2.05 | release_components |
| qwen3 | clothing | 8 | support1 T-0.72 R+0.00 | 7 T-0.06 R+0.05 | target_sorted4 T-1.54 R+0.00 | support2 T-1.03 R+0.00 | release1 T+0.45 R+1.38 | mixed1 T-0.38 R+0.41 | release_components |
| qwen3 | clothing | 16 | support1 T-0.72 R+0.00 | 14 T-0.21 R+0.13 | target_sorted8 T-1.62 R+0.00 | support2 T-1.03 R+0.00 | release3 T+2.00 R+2.64 | mixed1 T-0.38 R+0.41 | release_components |
| qwen3 | plant | 8 | support1 T-1.02 R+0.00 | 4 T-0.09 R+0.03 | target_sorted4 T-1.46 R+0.00 | support1 T-1.02 R+0.00 | release2 T+0.84 R+1.31 | NA | release_components |
| qwen3 | plant | 16 | support1 T-1.02 R+0.00 | 5 T-0.06 R+0.11 | target_sorted8 T-2.85 R+0.00 | support2 T-1.39 R+0.00 | release3 T+1.09 R+1.62 | mixed1 T-0.47 R+0.26 | release_components |
| glm4 | number | 8 | support0 T-0.37 R+0.13 | 7 T-0.03 R+0.01 | target_sorted8 T-0.83 R+0.40 | support1 T-0.37 R+0.13 | NA | NA | weak_or_mixed |
| glm4 | number | 16 | support0 T-0.37 R+0.13 | 9 T-0.02 R+0.00 | target_sorted16 T-0.90 R+0.68 | support2 T-0.60 R+0.25 | NA | NA | weak_or_mixed |
| glm4 | container | 8 | weak0 T-0.17 R+0.33 | 0 T-0.01 R+0.01 | target_sorted4 T-0.45 R+0.16 | NA | NA | NA | weak_or_mixed |
| glm4 | container | 16 | weak0 T-0.17 R+0.33 | 14 T-0.01 R+0.01 | target_sorted8 T-0.46 R+0.15 | NA | NA | NA | weak_or_mixed |
| glm4 | clothing | 8 | weak5 T-0.11 R+0.00 | 4 T-0.01 R+0.01 | target_sorted8 T-0.28 R+0.20 | NA | NA | NA | weak_or_mixed |
| glm4 | clothing | 16 | weak5 T-0.11 R+0.00 | 6 T-0.01 R+0.02 | target_sorted8 T-0.44 R+0.03 | NA | NA | NA | weak_or_mixed |
| glm4 | plant | 8 | weak6 T-0.07 R+0.06 | 7 T-0.01 R+0.02 | target_sorted4 T-0.10 R+0.07 | NA | NA | NA | weak_or_mixed |
| glm4 | plant | 16 | weak6 T-0.07 R+0.06 | 15 T-0.01 R+0.01 | target_sorted8 T-0.21 R+0.04 | NA | NA | NA | weak_or_mixed |
| deepseek7b | number | 8 | support1 T-5.55 R+0.00 | 5 T-0.20 R+0.00 | target_sorted8 T-10.93 R+0.00 | support5 T-10.67 R+0.00 | release1 T+0.09 R+0.97 | NA | distributed_support |
| deepseek7b | number | 16 | support1 T-5.55 R+0.00 | 2 T-0.13 R+0.00 | target_sorted16 T-12.58 R+0.00 | support8 T-12.49 R+0.00 | release2 T+0.62 R+1.83 | NA | distributed_support |
| deepseek7b | container | 8 | support6 T-2.92 R+0.00 | 1 T-0.16 R+0.00 | target_sorted8 T-10.94 R+0.00 | support5 T-10.85 R+0.00 | NA | NA | distributed_support |
| deepseek7b | container | 16 | support6 T-2.92 R+0.00 | 11 T-0.17 R+0.00 | target_sorted8 T-13.55 R+0.00 | support8 T-13.55 R+0.00 | release1 T-0.16 R+1.22 | NA | distributed_support |
| deepseek7b | clothing | 8 | support0 T-3.44 R+0.00 | 5 T-0.11 R+0.02 | target_sorted4 T-5.31 R+0.00 | support4 T-5.31 R+0.00 | release1 T+0.33 R+0.69 | NA | support_set_clean |
| deepseek7b | clothing | 16 | support0 T-3.44 R+0.00 | 4 T-0.10 R+0.01 | target_sorted4 T-5.31 R+0.00 | support5 T-5.58 R+0.00 | release3 T+2.07 R+1.67 | mixed1 T-0.60 R+0.70 | support_set_clean |
| deepseek7b | plant | 8 | support0 T-4.93 R+0.00 | 2 T-0.16 R+0.00 | target_sorted8 T-9.40 R+0.00 | support5 T-9.33 R+0.00 | NA | NA | distributed_support |
| deepseek7b | plant | 16 | support0 T-4.93 R+0.00 | 2 T-0.14 R+0.00 | target_sorted8 T-9.71 R+0.00 | support7 T-9.66 R+0.00 | release1 T+0.17 R+0.53 | NA | distributed_support |

## Objective Reading Rules
- distributed_support means cumulative top basis components are much stronger than any single component.
- compact_support means one basis component alone has a strong target-down effect.
- release_components means basis-level release is directly visible.

## Hard Limits
- SVD basis ordering is geometric, not guaranteed to be causal ordering.
- Component labels are heuristic and should be treated as audit tags, not final theory.
- This phase still uses DCF logits, not open generation.
