# Phase 114 Cross-model Answer-site Causal Subspace

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, clothing, plant; train/test objects per category: 12/12; templates: 4; prompts per category: 48
- ranks: [1, 2, 4, 8, 16]; scales: [1.0, 1.5]; layer: model-specific causal peak
- subspace: SVD of answer-site target-vs-other category contrast rows

## Cross-model Table
| model | category | best T_c | best answer contrast subspace | best random subspace | class |
|---|---|---|---|---|---|
| qwen3 | number | r1 s1.5 T-3.43 R+0.87 | r2 s1.5 T-3.12 R+0.78 | r16 s1.5 T-0.50 R+0.01 | similar |
| qwen3 | container | r1 s1.5 T-1.74 R+0.12 | r16 s1.5 T-2.59 R+2.03 | r4 s1.5 T-0.12 R+0.14 | subspace_slightly_stronger |
| qwen3 | clothing | r1 s1.5 T-1.42 R+1.12 | r8 s1.5 T-0.47 R+0.69 | r4 s1.5 T-0.04 R+0.17 | single_direction_stronger |
| qwen3 | plant | r1 s1.5 T-5.98 R+0.73 | r2 s1.5 T-1.26 R+0.00 | r16 s1.5 T-0.21 R+0.84 | single_direction_stronger |
| glm4 | number | r1 s1.5 T-0.10 R+0.06 | r16 s1.5 T-0.86 R+1.22 | r8 s1.5 T-0.02 R+0.04 | subspace_slightly_stronger |
| glm4 | container | r1 s1.5 T-0.08 R+0.10 | r16 s1.5 T-0.53 R+0.00 | r16 s1.0 T-0.03 R+0.00 | subspace_slightly_stronger |
| glm4 | clothing | r1 s1.5 T-0.07 R+0.09 | r8 s1.5 T-0.34 R+0.09 | r1 s1.5 T-0.01 R+0.01 | weak |
| glm4 | plant | r1 s1.0 T+0.01 R+0.06 | r16 s1.5 T-0.13 R+0.00 | r1 s1.5 T-0.00 R+0.01 | weak |
| deepseek7b | number | r1 s1.5 T+1.11 R+1.22 | r16 s1.5 T-11.75 R+0.00 | r16 s1.5 T-0.30 R+0.00 | subspace_stronger |
| deepseek7b | container | r1 s1.5 T-5.60 R+0.00 | r16 s1.5 T-12.42 R+0.00 | r2 s1.5 T-0.30 R+0.05 | subspace_stronger |
| deepseek7b | clothing | r1 s1.5 T-5.22 R+0.18 | r8 s1.5 T-4.99 R+0.00 | r1 s1.5 T-0.23 R+0.00 | similar |
| deepseek7b | plant | r1 s1.5 T-3.19 R+0.00 | r8 s1.5 T-7.93 R+0.00 | r4 s1.5 T-0.43 R+0.00 | subspace_stronger |

## Objective Reading Rules
- subspace_stronger means the answer contrast subspace reduces target logits at least 1.0 more than T_c.
- control_sensitive means random same-rank subspace is too close or stronger.
- R is max positive non-target release delta.

## Hard Limits
- The contrast subspace is category geometry, not yet an automatically discovered causal subspace.
- Random controls are same rank but not matched to norm spectrum.
- This phase still uses DCF logits, not open generation.
