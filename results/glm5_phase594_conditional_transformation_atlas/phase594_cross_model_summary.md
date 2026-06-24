# Phase594 Cross-Model Summary

Conditional transformation atlas: incoming/outgoing state, residual update, attention update, and MLP update.

## qwen3

- target_rows=5, nodes=[('prompt_last', 34), ('prompt_last', 33), ('prompt_last', 32), ('query_category', 32), ('query_category', 34), ('prompt_last', 30)]

| rank | key | value | correct_specific | positive_rate |
|---:|---|---:|---:|---:|
| 1 | prompt_last|L34|outgoing | +3.740 | +1.292 | 0.80 |
| 2 | prompt_last|L34|incoming | +3.169 | +1.201 | 0.80 |
| 3 | prompt_last|L33|outgoing | +3.169 | +1.201 | 0.80 |
| 4 | prompt_last|L33|incoming | +3.001 | +1.175 | 0.80 |
| 5 | prompt_last|L32|outgoing | +3.001 | +1.175 | 0.80 |
| 6 | query_category|L32|outgoing | +1.956 | +0.767 | 0.80 |
| 7 | query_category|L34|outgoing | +1.864 | +0.668 | 0.80 |
| 8 | prompt_last|L30|outgoing | +1.810 | +0.789 | 0.80 |
| 9 | prompt_last|L32|incoming | +1.762 | +0.716 | 0.80 |
| 10 | query_category|L34|incoming | +1.705 | +0.587 | 0.80 |
| 11 | prompt_last|L30|incoming | +1.512 | +0.611 | 0.80 |
| 12 | query_category|L32|incoming | +1.445 | +0.441 | 0.80 |
| 13 | prompt_last|L32|transition_gain | +1.239 | +0.460 | 0.80 |
| 14 | prompt_last|L32|residual_update | +1.239 | +0.460 | 0.80 |
| 15 | prompt_last|L32|attn_update | +1.092 | +0.369 | 0.80 |

## glm4

- target_rows=4, nodes=[('prompt_last', 38), ('prompt_last', 39), ('prompt_last', 37), ('prompt_last', 36), ('prompt_last', 35), ('prompt_last', 34)]

| rank | key | value | correct_specific | positive_rate |
|---:|---|---:|---:|---:|
| 1 | prompt_last|L38|outgoing | +0.821 | +0.383 | 0.50 |
| 2 | prompt_last|L39|incoming | +0.821 | +0.383 | 0.50 |
| 3 | prompt_last|L39|outgoing | +0.704 | +0.364 | 0.75 |
| 4 | prompt_last|L38|incoming | +0.468 | +0.253 | 0.50 |
| 5 | prompt_last|L37|outgoing | +0.468 | +0.253 | 0.50 |
| 6 | prompt_last|L37|incoming | +0.369 | +0.181 | 0.50 |
| 7 | prompt_last|L36|outgoing | +0.369 | +0.181 | 0.50 |
| 8 | prompt_last|L36|incoming | +0.361 | +0.163 | 0.50 |
| 9 | prompt_last|L35|outgoing | +0.361 | +0.163 | 0.50 |
| 10 | prompt_last|L38|transition_gain | +0.353 | +0.130 | 1.00 |
| 11 | prompt_last|L38|residual_update | +0.353 | +0.130 | 1.00 |
| 12 | prompt_last|L38|mlp_update | +0.330 | +0.122 | 0.75 |
| 13 | prompt_last|L35|incoming | +0.314 | +0.138 | 0.50 |
| 14 | prompt_last|L34|outgoing | +0.314 | +0.138 | 0.50 |
| 15 | prompt_last|L34|incoming | +0.289 | +0.125 | 0.50 |

## deepseek7b

- target_rows=21, nodes=[('rule_value', 26), ('prompt_last', 26), ('rule_relation', 18), ('rule_relation', 20), ('query_relation', 16), ('query_relation', 19)]

| rank | key | value | correct_specific | positive_rate |
|---:|---|---:|---:|---:|
| 1 | rule_value|L26|outgoing | +1.210 | +0.718 | 0.43 |
| 2 | rule_value|L26|mlp_update | +1.206 | +0.309 | 0.76 |
| 3 | prompt_last|L26|outgoing | +1.054 | +2.296 | 0.38 |
| 4 | rule_relation|L18|outgoing | +0.774 | +0.089 | 0.76 |
| 5 | prompt_last|L26|incoming | +0.760 | +1.691 | 0.48 |
| 6 | rule_relation|L20|outgoing | +0.714 | +0.117 | 0.57 |
| 7 | rule_value|L26|incoming | +0.616 | +0.469 | 0.62 |
| 8 | rule_value|L26|residual_update | +0.594 | +0.249 | 0.48 |
| 9 | rule_value|L26|transition_gain | +0.594 | +0.249 | 0.48 |
| 10 | rule_relation|L20|incoming | +0.544 | +0.113 | 0.76 |
| 11 | query_relation|L19|mlp_update | +0.519 | +0.195 | 0.86 |
| 12 | query_relation|L16|outgoing | +0.509 | +0.281 | 0.81 |
| 13 | query_relation|L19|outgoing | +0.498 | +0.314 | 0.67 |
| 14 | query_relation|L19|residual_update | +0.410 | +0.190 | 0.76 |
| 15 | query_relation|L19|transition_gain | +0.410 | +0.190 | 0.76 |

## Objective facts

- Qwen3 top signals are mostly incoming/outgoing state at late prompt_last; component update evidence is weaker.
- GLM4 has a visible prompt_last L38 transition/residual/MLP update signal, but target rows remain only 4.
- DS7B rule_value L26 is the strongest transition point: outgoing +1.210, incoming +0.616, transition_gain +0.594.
- DS7B rule_value L26 MLP update is +1.206 specific_margin, close to outgoing +1.210, suggesting the candidate-specific ranking is largely generated inside that layer update.
- DS7B query_relation L19 MLP update +0.519 also appears as a non-final relation-path update signal.
- These remain projection-level transition edges, not causal patch repair, but they are more mechanistically localized than Phase592 residual projection peaks.
