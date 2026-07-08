# Phase240 Gate/Up/Product Protocol Trace

behavior_rows: 18
gate_product_trace_rows: 90
residual_trace_rows: 54
mean_behavior_score: 0.65
protocol_match_rate: 0.0

## Model Decisions

| model | decision | strict delta | margin delta | protocol match | over generation |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | protocol_state_written_but_readout_competition_failed | 0.148364 | -1.5208 | 0.0 | 1.0 |
| glm4 | protocol_state_written_but_readout_competition_failed | 0.498715 | -0.7917 | 0.0 | 1.0 |
| deepseek7b | protocol_state_written_but_readout_competition_failed | 0.598917 | -1.8958 | 0.0 | 0.6667 |

## Top Component Deltas

| model | variant | level | layer | component | relative delta | cosine | rows |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| qwen3 | target_seeded | gate_up_product | 29 | recomputed_product | 1.070104 | 0.09003 | 1 |
| qwen3 | target_seeded | gate_up_product | 29 | product | 1.069946 | 0.089987 | 1 |
| deepseek7b | target_seeded | gate_up_product | 24 | product | 1.0574 | 0.395222 | 1 |
| deepseek7b | target_seeded | gate_up_product | 24 | recomputed_product | 1.057355 | 0.3954 | 1 |
| qwen3 | target_seeded | gate_up_product | 29 | down_out | 1.03768 | 0.19734 | 1 |
| glm4 | target_seeded | gate_up_product | 30 | product | 0.978083 | 0.236137 | 1 |
| glm4 | target_seeded | gate_up_product | 30 | recomputed_product | 0.978049 | 0.236383 | 1 |
| glm4 | target_seeded | gate_up_product | 30 | down_out | 0.971772 | 0.244443 | 1 |
| qwen3 | target_seeded | gate_up_product | 29 | up | 0.963972 | 0.462463 | 1 |
| deepseek7b | target_seeded | residual_state | 27 | residual_state | 0.945325 | 0.62961 | 1 |
| deepseek7b | target_seeded | gate_up_product | 24 | down_out | 0.926155 | 0.594789 | 1 |
| glm4 | target_seeded | gate_up_product | 30 | up | 0.896696 | 0.524248 | 1 |
| qwen3 | target_seeded | residual_state | 29 | residual_state | 0.883942 | 0.569212 | 1 |
| deepseek7b | target_seeded | gate_up_product | 24 | up | 0.875434 | 0.58768 | 1 |
| qwen3 | target_seeded | residual_state | 31 | residual_state | 0.875292 | 0.537208 | 1 |
| glm4 | target_seeded | residual_state | 28 | residual_state | 0.871095 | 0.600188 | 1 |
| glm4 | target_seeded | gate_up_product | 30 | gate | 0.850289 | 0.565508 | 1 |
| qwen3 | target_seeded | residual_state | 33 | residual_state | 0.837955 | 0.609271 | 1 |
| deepseek7b | target_seeded | residual_state | 24 | residual_state | 0.828475 | 0.697909 | 1 |
| glm4 | target_seeded | residual_state | 30 | residual_state | 0.810199 | 0.621478 | 1 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | recomputed_product | 0.77815 | 0.716725 | 1 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | product | 0.777836 | 0.716762 | 1 |
| glm4 | target_seeded | residual_state | 32 | residual_state | 0.754235 | 0.66619 | 1 |
| glm4 | short_answer_instruction | gate_up_product | 30 | product | 0.720566 | 0.726445 | 1 |
| glm4 | short_answer_instruction | gate_up_product | 30 | recomputed_product | 0.719564 | 0.727385 | 1 |
| glm4 | short_answer_instruction | gate_up_product | 30 | down_out | 0.719149 | 0.720963 | 1 |
| glm4 | explain_instruction | gate_up_product | 30 | product | 0.696215 | 0.732007 | 1 |
| glm4 | explain_instruction | gate_up_product | 30 | recomputed_product | 0.695959 | 0.732195 | 1 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | down_out | 0.695581 | 0.768959 | 1 |
| glm4 | explain_instruction | gate_up_product | 30 | down_out | 0.692454 | 0.731441 | 1 |

## Caution

This phase is a trace, not a causal closure. It marks whether protocol prompts change gate/up/product/residual states and whether the change reaches readout competition.
