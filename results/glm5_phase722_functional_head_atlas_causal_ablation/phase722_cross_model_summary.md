# Phase 722 Functional Head Atlas Causal Ablation Validation

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: local zero ablation at answer_last o_proj input.

## Most Harmful Candidate Heads

### qwen3

| family | head | mean_logprob_delta | mean_rank_delta | top1_drop | source_focus |
|---|---:|---:|---:|---:|---:|
| simple_grammar_protocol_route | L28H0 | -0.0130 | 0.00 | 0.042 | 1.1737 |
| simple_grammar_protocol_route | L23H4 | -0.0024 | 0.00 | 0.000 | 0.9848 |
| color_value_reuse_difference | L21H23 | -0.0023 | 0.00 | 0.000 | 0.7432 |
| translation_language_route | L24H29 | -0.0023 | 0.00 | 0.000 | 0.8462 |
| translation_language_route | L29H11 | -0.0000 | 0.00 | 0.000 | 0.7819 |
| fruit_identity_reuse_difference | L26H26 | -0.0000 | 0.00 | 0.000 | 0.9459 |
| translation_language_route | L28H0 | 0.0000 | 0.00 | 0.000 | 0.9079 |
| fruit_identity_reuse_difference | L24H29 | 0.0004 | 0.00 | 0.000 | 0.8284 |
| color_value_reuse_difference | L24H29 | 0.0024 | 0.00 | 0.000 | 0.7645 |
| fruit_identity_reuse_difference | L28H0 | 0.0025 | 0.00 | 0.000 | 0.8492 |
| color_value_reuse_difference | L28H0 | 0.0035 | 0.00 | 0.000 | 0.6661 |
| simple_grammar_protocol_route | L20H15 | 0.0041 | 0.00 | 0.000 | 1.0790 |

### glm4

| family | head | mean_logprob_delta | mean_rank_delta | top1_drop | source_focus |
|---|---:|---:|---:|---:|---:|
| fruit_identity_reuse_difference | L24H19 | -0.0106 | 0.00 | 0.000 | 0.9568 |
| translation_language_route | L29H28 | -0.0086 | 0.00 | 0.000 | 0.9515 |
| translation_language_route | L23H10 | -0.0052 | -0.04 | 0.000 | 0.9485 |
| fruit_identity_reuse_difference | L29H28 | 0.0015 | 0.00 | 0.000 | 0.9717 |
| color_value_reuse_difference | L29H28 | 0.0019 | 0.00 | 0.000 | 0.9344 |
| color_value_reuse_difference | L29H18 | 0.0035 | 0.00 | 0.000 | 0.8740 |
| translation_language_route | L23H26 | 0.0046 | 0.00 | 0.000 | 0.9250 |
| fruit_identity_reuse_difference | L29H18 | 0.0064 | 0.00 | 0.000 | 0.9598 |
| simple_grammar_protocol_route | L23H10 | 0.0099 | 0.00 | 0.000 | 1.2119 |
| simple_grammar_protocol_route | L29H18 | 0.0102 | 0.04 | 0.042 | 1.1837 |
| color_value_reuse_difference | L21H15 | 0.0129 | 0.00 | 0.000 | 0.9494 |
| simple_grammar_protocol_route | L29H26 | 0.0162 | 0.00 | 0.000 | 1.2467 |

### deepseek7b

| family | head | mean_logprob_delta | mean_rank_delta | top1_drop | source_focus |
|---|---:|---:|---:|---:|---:|
| fruit_identity_reuse_difference | L20H17 | -2.8909 | 28.46 | 0.250 | 0.4573 |
| fruit_identity_reuse_difference | L27H23 | -0.8819 | 5.62 | 0.125 | 0.4246 |
| translation_language_route | L24H21 | -0.7647 | 62.17 | 0.083 | 0.4545 |
| simple_grammar_protocol_route | L22H24 | -0.4171 | 1.46 | 0.083 | 0.8706 |
| fruit_identity_reuse_difference | L23H0 | -0.4158 | 2.33 | 0.042 | 0.4413 |
| simple_grammar_protocol_route | L22H1 | -0.3533 | 1.17 | 0.083 | 0.9293 |
| translation_language_route | L23H4 | -0.2360 | 12.38 | 0.042 | 0.4264 |
| color_value_reuse_difference | L23H6 | -0.1044 | 0.21 | 0.042 | 0.4596 |
| simple_grammar_protocol_route | L21H25 | -0.0870 | 0.17 | 0.000 | 0.8974 |
| translation_language_route | L23H0 | -0.0224 | 4.67 | 0.042 | 0.4568 |
| color_value_reuse_difference | L22H1 | 0.0026 | -0.29 | 0.083 | 0.3809 |
| color_value_reuse_difference | L23H0 | 0.0222 | -0.17 | 0.000 | 0.5027 |

## Strict Interpretation

- Negative logprob delta means zeroing the head hurt the target first token.
- This is a necessity hint, not a sufficiency proof.
- Full phrase likelihood and natural generation closure still need validation.
