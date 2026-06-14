# Phase 126 Cross-model Residual Gap Decomposition

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: qwen3: L32-L35 monitor L35; glm4: L15-L18 monitor L18; deepseek7b: L24-L27 monitor L27; rank: 16; components: layer_input, attention_output, mlp_output, layer_output

| model | category | layer input | attention output | MLP output | layer output | attention+MLP | class |
|---|---|---|---|---|---|---|---|
| qwen3 | number | L34 layer_input T-0.13 R+0.45 A+10.97 | L35 attention_output T-0.06 R+0.00 A+0.00 | L34 mlp_output T-0.00 R+0.09 A+0.27 | L33 layer_output T-0.13 R+0.45 A+10.97 | L35 attention_plus_mlp T-0.02 R+0.06 A+0.00 ratio+0.29 | weak_residual_output |
| qwen3 | container | L32 layer_input T-0.09 R+0.41 A-12.29 | L35 attention_output T-0.01 R+0.01 A+0.00 | L32 mlp_output T-0.07 R+0.06 A-0.25 | L32 layer_output T-0.04 R+0.26 A-7.09 | L32 attention_plus_mlp T-0.06 R+0.06 A-0.19 ratio+1.37 | weak_residual_output |
| qwen3 | plant | L33 layer_input T-0.22 R+0.29 A+12.42 | L33 attention_output T+0.00 R+0.03 A-0.33 | L34 mlp_output T+0.04 R+0.06 A+0.17 | L32 layer_output T-0.22 R+0.29 A+12.42 | L34 attention_plus_mlp T+0.04 R+0.06 A+0.53 ratio+0.15 | weak_residual_output |
| glm4 | number | L18 layer_input T-0.26 R+0.27 A+0.07 | L18 attention_output T-0.08 R+0.02 A+0.00 | L17 mlp_output T-0.09 R+0.01 A+0.02 | L17 layer_output T-0.26 R+0.27 A+0.07 | L18 attention_plus_mlp T-0.12 R+0.04 A+0.00 ratio+0.52 | weak_residual_output |
| glm4 | container | L18 layer_input T-0.01 R+0.70 A-0.01 | L17 attention_output T-0.02 R+0.01 A+0.00 | L15 mlp_output T-0.04 R+0.06 A+0.00 | L17 layer_output T-0.01 R+0.70 A-0.01 | L18 attention_plus_mlp T-0.03 R+0.03 A+0.00 ratio-17.99 | weak_residual_output |
| glm4 | plant | L18 layer_input T-0.02 R+0.41 A+0.11 | L15 attention_output T+0.01 R+0.06 A+0.00 | L16 mlp_output T-0.04 R+0.10 A+0.03 | L17 layer_output T-0.02 R+0.41 A+0.11 | L16 attention_plus_mlp T-0.03 R+0.06 A+0.02 ratio-6.79 | weak_residual_output |
| deepseek7b | number | L25 layer_input T-2.05 R+0.00 A-50.06 | L27 attention_output T-0.03 R+0.00 A+0.00 | L24 mlp_output T-0.28 R+0.00 A-2.99 | L27 layer_output T-2.51 R+0.54 A+0.00 | L24 attention_plus_mlp T-0.25 R+0.00 A-3.64 ratio+0.12 | upstream_carry_candidate |
| deepseek7b | container | L26 layer_input T-1.22 R+0.00 A+23.43 | L24 attention_output T-0.07 R+0.00 A+0.71 | L24 mlp_output T-0.32 R+0.00 A+1.97 | L27 layer_output T-2.66 R+0.88 A+0.00 | L24 attention_plus_mlp T-0.25 R+0.00 A+2.46 ratio+0.21 | residual_carry_or_norm_candidate |
| deepseek7b | plant | L25 layer_input T-2.28 R+0.00 A-33.06 | L24 attention_output T-0.09 R+0.10 A+0.45 | L25 mlp_output T-0.18 R+0.00 A-2.74 | L27 layer_output T-2.42 R+1.56 A+0.00 | L26 attention_plus_mlp T-0.13 R+0.00 A+2.92 ratio+0.11 | upstream_carry_candidate |

## Reading Rules
- layer_output is the residual stream after the full block and acts as the residual reference.
- attention+MLP ratio is measured against the layer_output condition at its own best row.
- A is answer projection delta at the peak answer site.
