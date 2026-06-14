# Phase 110 Cross-model Orthogonal Subspace Split

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, time, container, clothing, furniture, plant; train/test objects per category: 12/12; templates: 4; prompts per category: 48
- components: orthogonal_full, neighbor_aligned, transport_aligned, residual, random_same_norm
- positions: answer_last, both; scales: 1.0, 1.5

## Cross-model Table
| model | category | frac N/T/R | best neighbor | best transport | best residual | best orth | best random | residual release | class |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | number | 0.54/0.27/0.80 | both s1.5 Δ-1.91 | answer_last s1.5 Δ-3.43 | both s1.5 Δ-0.37 | both s1.5 Δ-3.05 | answer_last s1.5 Δ-0.12 | fruit +0.15 | transport_support |
| qwen3 | time | 0.57/0.15/0.81 | both s1.5 Δ-1.95 | answer_last s1.5 Δ-1.84 | answer_last s1.5 Δ+0.03 | both s1.5 Δ-0.64 | answer_last s1.5 Δ-0.00 | plant +0.17 | neighbor_competition |
| qwen3 | container | 0.46/0.28/0.84 | answer_last s1.0 Δ+0.24 | answer_last s1.5 Δ-1.75 | both s1.5 Δ+0.00 | both s1.0 Δ+0.25 | both s1.5 Δ-0.04 | none +0.00 | transport_support |
| qwen3 | clothing | 0.34/0.39/0.85 | answer_last s1.0 Δ+0.71 | answer_last s1.5 Δ-1.43 | both s1.5 Δ+0.01 | answer_last s1.0 Δ+0.72 | both s1.5 Δ-0.07 | weather +0.29 | transport_support |
| qwen3 | furniture | 0.54/0.33/0.78 | answer_last s1.0 Δ+0.55 | both s1.5 Δ-0.56 | answer_last s1.5 Δ-0.46 | both s1.0 Δ+1.93 | answer_last s1.5 Δ-0.04 | none +0.00 | mixed |
| qwen3 | plant | 0.49/0.28/0.83 | both s1.0 Δ+0.10 | answer_last s1.5 Δ-5.97 | both s1.5 Δ-0.29 | both s1.5 Δ-0.02 | both s1.0 Δ+0.15 | sound +0.60 | transport_support |
| glm4 | number | 0.64/0.04/0.77 | both s1.5 Δ-0.14 | answer_last s1.5 Δ-0.09 | both s1.0 Δ+0.07 | both s1.0 Δ+0.08 | both s1.5 Δ-0.01 | container +0.31 | weak |
| glm4 | time | 0.74/0.07/0.67 | answer_last s1.5 Δ-0.47 | answer_last s1.5 Δ-0.05 | both s1.5 Δ-0.01 | answer_last s1.5 Δ-0.18 | answer_last s1.0 Δ-0.01 | action +0.06 | weak |
| glm4 | container | 0.84/0.06/0.53 | answer_last s1.0 Δ+0.05 | answer_last s1.5 Δ-0.07 | both s1.5 Δ-0.01 | both s1.0 Δ+0.00 | both s1.0 Δ-0.01 | place +0.04 | weak |
| glm4 | clothing | 0.89/0.01/0.45 | answer_last s1.5 Δ-0.07 | answer_last s1.5 Δ-0.07 | both s1.0 Δ-0.04 | both s1.5 Δ-0.08 | both s1.0 Δ-0.00 | event +0.05 | weak |
| glm4 | furniture | 0.93/0.00/0.37 | answer_last s1.0 Δ+0.07 | answer_last s1.0 Δ-0.03 | answer_last s1.0 Δ-0.01 | answer_last s1.0 Δ+0.06 | answer_last s1.5 Δ+0.01 | fruit +0.03 | weak |
| glm4 | plant | 0.90/0.04/0.44 | answer_last s1.5 Δ-0.02 | both s1.0 Δ+0.01 | both s1.5 Δ-0.05 | both s1.5 Δ-0.06 | answer_last s1.5 Δ-0.00 | fruit +0.04 | weak |
| deepseek7b | number | 0.41/0.22/0.89 | both s1.5 Δ-0.94 | answer_last s1.5 Δ+1.06 | both s1.5 Δ-2.76 | both s1.5 Δ-4.95 | both s1.5 Δ+0.07 | none +0.00 | residual_support |
| deepseek7b | time | 0.46/0.18/0.87 | both s1.5 Δ-0.82 | both s1.5 Δ-0.61 | both s1.5 Δ-0.93 | both s1.5 Δ+0.06 | both s1.5 Δ-0.19 | none +0.00 | mixed |
| deepseek7b | container | 0.30/0.31/0.90 | both s1.5 Δ-0.24 | both s1.5 Δ-5.68 | both s1.5 Δ-1.44 | both s1.5 Δ-3.15 | both s1.0 Δ-0.10 | none +0.00 | transport_support |
| deepseek7b | clothing | 0.28/0.44/0.85 | answer_last s1.0 Δ-0.18 | both s1.5 Δ-5.17 | answer_last s1.5 Δ-0.91 | both s1.0 Δ+1.22 | answer_last s1.0 Δ-0.12 | event +0.01 | transport_support |
| deepseek7b | furniture | 0.44/0.35/0.83 | both s1.5 Δ+0.07 | both s1.5 Δ-3.85 | answer_last s1.5 Δ-0.03 | answer_last s1.0 Δ+0.31 | both s1.5 Δ-0.10 | light +0.23 | transport_support |
| deepseek7b | plant | 0.42/0.34/0.84 | answer_last s1.0 Δ+0.66 | both s1.5 Δ-3.28 | both s1.0 Δ-0.12 | answer_last s1.0 Δ+1.05 | answer_last s1.0 Δ-0.02 | number +0.09 | transport_support |

## Objective Facts
- Qwen3 number/time show real target-down effects in neighbor and transport components; number is strongest in transport.
- Qwen3 container/clothing/plant expose component cancellation: transport removal can reduce target strongly even when full orthogonal removal is weak or target-up.
- DS7B container/clothing/furniture/plant are transport-dominant: removing the object-to-answer transport-aligned component gives large target-down effects.
- DS7B number differs from the above pattern: the residual component gives the strongest subcomponent target-down, while full orthogonal remains strongest overall.
- GLM4 effects remain weak, so it is still unsuitable for strong mechanism conclusions in this probe family.

## Hard Limits
- The transport direction is a mean object_last to answer_last vector, not a direct causal path proof.
- Neighbor basis is hand-defined by category adjacency and can miss hidden competitors.
- Single-layer intervention can create cancellation artifacts; multi-layer cumulative tests are still needed.
- Some component removals are stronger than full orthogonal removal, indicating non-additive interaction inside the residual stream.

