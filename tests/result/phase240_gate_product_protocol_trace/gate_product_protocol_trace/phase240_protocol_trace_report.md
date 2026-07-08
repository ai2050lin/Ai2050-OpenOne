# Phase240 Gate/Up/Product Protocol Trace

behavior_rows: 108
gate_product_trace_rows: 540
residual_trace_rows: 324
mean_behavior_score: 0.6278
protocol_match_rate: 0.0

## Model Decisions

| model | decision | strict delta | margin delta | protocol match | over generation |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | protocol_state_written_but_readout_competition_failed | 0.172966 | -0.066 | 0.0 | 1.0 |
| glm4 | protocol_state_written_but_readout_competition_failed | 0.604032 | -1.0608 | 0.0 | 1.0 |
| deepseek7b | protocol_state_written_but_readout_competition_failed | 0.682197 | -2.6493 | 0.0 | 0.6667 |

## Top Component Deltas

| model | variant | level | layer | component | relative delta | cosine | rows |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| deepseek7b | target_seeded | gate_up_product | 24 | product | 1.101602 | 0.37122 | 6 |
| deepseek7b | target_seeded | gate_up_product | 24 | recomputed_product | 1.101519 | 0.371307 | 6 |
| qwen3 | target_seeded | gate_up_product | 29 | recomputed_product | 1.042922 | 0.137354 | 6 |
| qwen3 | target_seeded | gate_up_product | 29 | product | 1.042874 | 0.137357 | 6 |
| qwen3 | target_seeded | gate_up_product | 29 | down_out | 1.009176 | 0.238086 | 6 |
| glm4 | target_seeded | gate_up_product | 30 | recomputed_product | 0.987121 | 0.266164 | 6 |
| glm4 | target_seeded | gate_up_product | 30 | product | 0.987039 | 0.266227 | 6 |
| glm4 | target_seeded | gate_up_product | 30 | down_out | 0.973542 | 0.278747 | 6 |
| deepseek7b | target_seeded | gate_up_product | 24 | down_out | 0.972871 | 0.5781 | 6 |
| qwen3 | target_seeded | gate_up_product | 29 | up | 0.93609 | 0.484027 | 6 |
| deepseek7b | target_seeded | residual_state | 27 | residual_state | 0.900597 | 0.627463 | 6 |
| glm4 | target_seeded | gate_up_product | 30 | up | 0.891895 | 0.531987 | 6 |
| deepseek7b | target_seeded | gate_up_product | 24 | up | 0.88987 | 0.572828 | 6 |
| qwen3 | target_seeded | residual_state | 29 | residual_state | 0.873006 | 0.572094 | 6 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | recomputed_product | 0.870736 | 0.649198 | 6 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | product | 0.870664 | 0.649188 | 6 |
| glm4 | target_seeded | residual_state | 28 | residual_state | 0.869926 | 0.604383 | 6 |
| deepseek7b | target_seeded | residual_state | 24 | residual_state | 0.852115 | 0.686162 | 6 |
| glm4 | target_seeded | gate_up_product | 30 | gate | 0.848162 | 0.57284 | 6 |
| qwen3 | target_seeded | residual_state | 31 | residual_state | 0.845656 | 0.580876 | 6 |
| glm4 | short_answer_instruction | gate_up_product | 30 | product | 0.82097 | 0.568755 | 6 |
| glm4 | short_answer_instruction | gate_up_product | 30 | recomputed_product | 0.820695 | 0.569038 | 6 |
| qwen3 | target_seeded | residual_state | 33 | residual_state | 0.817154 | 0.629083 | 6 |
| glm4 | short_answer_instruction | gate_up_product | 30 | down_out | 0.806421 | 0.586289 | 6 |
| deepseek7b | explain_instruction | gate_up_product | 24 | recomputed_product | 0.798952 | 0.662727 | 6 |
| deepseek7b | explain_instruction | gate_up_product | 24 | product | 0.79895 | 0.662776 | 6 |
| glm4 | target_seeded | residual_state | 30 | residual_state | 0.792968 | 0.645574 | 6 |
| glm4 | one_word_strict | gate_up_product | 30 | product | 0.770894 | 0.634103 | 6 |
| glm4 | one_word_strict | gate_up_product | 30 | recomputed_product | 0.770844 | 0.634184 | 6 |
| deepseek7b | strong_answer_anchor | gate_up_product | 24 | down_out | 0.768958 | 0.719575 | 6 |

## Caution

This phase is a trace, not a causal closure. It marks whether protocol prompts change gate/up/product/residual states and whether the change reaches readout competition.
