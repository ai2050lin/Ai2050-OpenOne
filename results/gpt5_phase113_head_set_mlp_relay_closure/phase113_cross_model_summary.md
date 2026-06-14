# Phase 113 Cross-model Head Set and MLP Relay Closure

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, clothing, plant; train/test objects per category: 12/12; templates: 4; prompts per category: 48
- layers: peak-3 ... peak; candidate heads: 16; set sizes: [1, 2, 4, 8, 16]
- conditions: source/projection/target/mixed/random head sets; heads only, MLP only, heads+MLP

## Cross-model Table
| model | category | T_c reference | best heads only | best heads+MLP | best MLP only | best random | class |
|---|---|---|---|---|---|---|---|
| qwen3 | number | tc  k T-3.43 R+1.00 A+0.00 | target heads_only k4 T-0.33 R+0.10 A-2.19 | target heads_plus_mlp k8 T+4.00 R-1.17 A-179.39 | source mlp_only k1 T+4.18 R-1.22 A-175.66 | random heads_only k8 T-0.02 R+0.01 A-2.32 | not_closed |
| qwen3 | container | tc  k T-1.75 R+1.00 A+0.00 | target heads_only k8 T-0.15 R+0.09 A-2.87 | target heads_plus_mlp k8 T+2.39 R-1.37 A-203.90 | source mlp_only k1 T+2.63 R-1.51 A-199.42 | random heads_only k2 T-0.01 R+0.01 A-0.22 | not_closed |
| qwen3 | clothing | tc  k T-1.43 R+1.00 A+0.00 | target heads_only k8 T-0.72 R+0.50 A-9.53 | target heads_plus_mlp k8 T+1.58 R-1.11 A-185.79 | source mlp_only k1 T+2.49 R-1.75 A-172.45 | random heads_only k16 T-0.35 R+0.25 A-15.78 | partial_closure |
| qwen3 | plant | tc  k T-5.97 R+1.00 A+0.00 | target heads_only k4 T-0.59 R+0.10 A-3.32 | target heads_plus_mlp k2 T+3.07 R-0.51 A-159.42 | source mlp_only k1 T+3.48 R-0.58 A-154.01 | random heads_only k16 T-0.02 R+0.00 A-7.68 | not_closed |
| glm4 | number | tc  k T-0.09 R+1.00 A+0.00 | target heads_only k8 T-0.08 R+0.90 A+0.08 | source heads_plus_mlp k8 T-0.21 R+2.25 A+0.36 | source mlp_only k1 T-0.07 R+0.79 A+0.36 | random heads_plus_mlp k16 T-0.22 R+2.34 A+0.34 | weak_reference |
| glm4 | container | tc  k T-0.07 R+1.00 A+0.00 | target heads_only k8 T-0.03 R+0.44 A+0.05 | source heads_plus_mlp k16 T-0.09 R+1.30 A+0.99 | source mlp_only k1 T-0.03 R+0.48 A+0.97 | random heads_plus_mlp k4 T-0.12 R+1.71 A+0.95 | weak_reference |
| glm4 | clothing | tc  k T-0.07 R+1.00 A+0.00 | target heads_only k8 T-0.02 R+0.34 A+0.07 | source heads_plus_mlp k16 T+0.19 R-2.65 A+0.89 | source mlp_only k1 T+0.21 R-2.87 A+0.91 | random heads_only k4 T-0.04 R+0.53 A-0.02 | weak_reference |
| glm4 | plant | tc  k T+0.02 R+1.00 A+0.00 | target heads_only k2 T-0.02 R-0.75 A-0.02 | source heads_plus_mlp k1 T-0.10 R-4.53 A-0.86 | source mlp_only k1 T-0.09 R-4.08 A-0.85 | random heads_plus_mlp k2 T-0.11 R-4.99 A-0.86 | weak_reference |
| deepseek7b | number | tc  k T+1.06 R+1.00 A+0.00 | target heads_only k8 T-0.49 R-0.46 A-16.30 | source heads_plus_mlp k1 T-1.24 R-1.18 A-187.11 | source mlp_only k1 T-1.35 R-1.28 A-187.94 | random heads_plus_mlp k16 T-1.63 R-1.55 A-192.61 | weak_reference |
| deepseek7b | container | tc  k T-5.50 R+1.00 A+0.00 | target heads_only k1 T-0.28 R+0.05 A-4.07 | target heads_plus_mlp k2 T+0.34 R-0.06 A-307.50 | source mlp_only k1 T+0.14 R-0.03 A-303.14 | random heads_only k8 T-0.15 R+0.03 A-8.32 | not_closed |
| deepseek7b | clothing | tc  k T-5.04 R+1.00 A+0.00 | target heads_only k8 T-0.78 R+0.16 A-20.82 | target heads_plus_mlp k1 T+1.39 R-0.28 A-328.14 | source mlp_only k1 T+1.44 R-0.29 A-328.47 | random heads_only k4 T-0.45 R+0.09 A-8.60 | not_closed |
| deepseek7b | plant | tc  k T-3.20 R+1.00 A+0.00 | target heads_only k8 T-0.32 R+0.10 A-4.44 | target heads_plus_mlp k4 T+0.55 R-0.17 A-314.72 | source mlp_only k1 T+0.92 R-0.29 A-311.61 | random heads_only k16 T-0.28 R+0.09 A-28.47 | not_closed |

## Objective Reading Rules
- R is target_delta divided by the answer-site T_c removal target_delta; useful closure should be positive and close to 1.
- not_closed means head sets and MLP relay did not approach the T_c reference.
- control_sensitive means random head sets were comparable or stronger than selected sets.

## Hard Limits
- MLP ablation zeroes MLP output at answer_last across scanned layers; it is a coarse intervention.
- Candidate projection heads are chosen from the expanded source-head pool, not from all heads.
- Generation audit and Q/K/V split are still not included.
