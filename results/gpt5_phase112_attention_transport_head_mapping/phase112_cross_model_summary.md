# Phase 112 Cross-model Attention Transport Head Mapping

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, time, container, clothing, furniture, plant; train/test objects per category: 12/12; templates: 4; prompts per category: 48
- layers: peak-3 ... peak; selected heads per category: 8
- source score: answer_last attention mass to object_span + object_last
- intervention: zero selected head slice at answer_last before o_proj

## Cross-model Table
| model | category | top source head | strongest target-down head | strongest projection-down head | class |
|---|---|---|---|---|---|
| qwen3 | number | L35 H21 obj0.057 T+0.00 A+0.00 | L33 H24 obj0.028 T-0.30 A-1.86 | L35 H8 obj0.033 T+0.01 A-4.83 | weak |
| qwen3 | time | L33 H9 obj0.078 T+0.00 A+0.00 | L35 H21 obj0.062 T-0.02 A-2.15 | L35 H8 obj0.048 T+0.00 A-5.78 | projection_only |
| qwen3 | container | L35 H27 obj0.093 T+0.00 A+0.00 | L34 H21 obj0.077 T-0.03 A-0.86 | L35 H8 obj0.037 T+0.03 A-4.73 | projection_only |
| qwen3 | clothing | L33 H9 obj0.111 T+0.00 A+0.00 | L33 H9 obj0.111 T-0.07 A+0.31 | L35 H8 obj0.039 T-0.02 A-5.42 | projection_only |
| qwen3 | furniture | L35 H21 obj0.101 T+0.00 A+0.00 | L35 H28 obj0.080 T-0.05 A+0.06 | L35 H8 obj0.038 T-0.03 A-4.85 | projection_only |
| qwen3 | plant | L34 H21 obj0.117 T+0.00 A+0.00 | L35 H27 obj0.110 T-0.02 A-0.12 | L35 H21 obj0.077 T+0.03 A-2.18 | projection_only |
| glm4 | number | L16 H11 obj0.146 T+0.00 A+0.00 | L16 H11 obj0.146 T-0.03 A+0.03 | L17 H28 obj0.121 T-0.01 A-0.00 | weak |
| glm4 | time | L16 H11 obj0.156 T+0.00 A+0.00 | L17 H17 obj0.123 T-0.01 A+0.01 | L18 H1 obj0.123 T+0.02 A-0.01 | weak |
| glm4 | container | L15 H1 obj0.140 T+0.00 A+0.00 | L15 H1 obj0.140 T-0.02 A+0.01 | L17 H20 obj0.127 T+0.00 A+0.00 | weak |
| glm4 | clothing | L15 H1 obj0.149 T+0.00 A+0.00 | L18 H18 obj0.126 T-0.02 A+0.02 | L17 H20 obj0.130 T-0.01 A-0.00 | weak |
| glm4 | furniture | L15 H1 obj0.135 T+0.00 A+0.00 | L16 H8 obj0.122 T-0.00 A+0.01 | L17 H20 obj0.123 T+0.01 A+0.00 | weak |
| glm4 | plant | L15 H1 obj0.153 T+0.00 A+0.00 | L15 H1 obj0.153 T-0.01 A-0.00 | L16 H11 obj0.123 T+0.00 A-0.04 | weak |
| deepseek7b | number | L24 H17 obj0.174 T+0.00 A+0.00 | L24 H22 obj0.103 T-0.08 A-3.64 | L24 H22 obj0.103 T-0.08 A-3.64 | projection_only |
| deepseek7b | time | L25 H19 obj0.202 T+0.00 A+0.00 | L24 H22 obj0.093 T-0.06 A-6.62 | L24 H22 obj0.093 T-0.06 A-6.62 | projection_only |
| deepseek7b | container | L25 H19 obj0.228 T+0.00 A+0.00 | L25 H15 obj0.120 T-0.27 A-4.97 | L24 H22 obj0.115 T-0.14 A-5.84 | weak |
| deepseek7b | clothing | L25 H19 obj0.229 T+0.00 A+0.00 | L24 H17 obj0.146 T-0.40 A+0.62 | L25 H15 obj0.151 T-0.02 A-7.61 | weak |
| deepseek7b | furniture | L25 H19 obj0.273 T+0.00 A+0.00 | L24 H2 obj0.153 T-0.08 A+0.82 | L25 H15 obj0.184 T+0.02 A-6.33 | projection_only |
| deepseek7b | plant | L24 H6 obj0.311 T+0.00 A+0.00 | L25 H24 obj0.173 T-0.16 A-0.61 | L25 H15 obj0.130 T+0.03 A-6.65 | projection_only |

## Objective Reading Rules
- candidate_transport_head means a high object-source attention head also causes target logits to drop when ablated.
- projection_only means the monitored T_c projection moves but logits do not.
- low_source_attention means answer_last barely attends to object sources in the scanned layers.

## Hard Limits
- Head ablation zeroes one head slice at answer_last before o_proj; it does not separate Q/K/V causes.
- Attention mass is only a candidate selector, not a causal metric.
- This phase does not yet perform value transplant or generation audit.
