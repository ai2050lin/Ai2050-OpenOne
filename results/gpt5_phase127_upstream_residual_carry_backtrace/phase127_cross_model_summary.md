# Phase 127 Cross-model Upstream Residual Carry Backtrace

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: qwen3: L20-L35 monitor L35; glm4: L8-L18 monitor L18; deepseek7b: L12-L27 monitor L27; rank: 16; onset threshold: -0.5

| model | category | input onset | output onset | best input | best output | final input | final output | class |
|---|---|---|---|---|---|---|---|---|
| qwen3 | number | L21 | L20 | L30 T-0.92 A+16.51 | L29 T-0.92 A+16.51 | L35 T-0.05 A+10.58 | L35 T-0.07 A+0.00 | late_output_emergence |
| qwen3 | container | L21 | L20 | L21 T-0.79 A-29.65 | L20 T-0.79 A-29.65 | L35 T-0.04 A-8.84 | L35 T+0.07 A+0.00 | late_output_emergence |
| qwen3 | plant | L20 | L20 | L26 T-0.88 A+1.73 | L25 T-0.88 A+1.73 | L35 T+0.27 A+10.15 | L35 T+0.24 A+0.00 | upstream_residual_carry |
| glm4 | number | LNone | LNone | L18 T-0.26 A+0.07 | L17 T-0.26 A+0.07 | L18 T-0.26 A+0.07 | L18 T-0.23 A+0.00 | weak_residual_path |
| glm4 | container | LNone | LNone | L13 T-0.08 A-0.02 | L12 T-0.08 A-0.02 | L18 T-0.01 A-0.01 | L18 T+0.00 A+0.00 | weak_residual_path |
| glm4 | plant | LNone | LNone | L13 T-0.08 A+0.33 | L12 T-0.08 A+0.33 | L18 T-0.02 A+0.11 | L18 T-0.01 A+0.00 | weak_residual_path |
| deepseek7b | number | L21 | L20 | L25 T-2.05 A-50.06 | L27 T-2.51 A+0.00 | L27 T-0.96 A-8.98 | L27 T-2.51 A+0.00 | late_output_emergence |
| deepseek7b | container | L20 | L19 | L26 T-1.22 A+23.43 | L27 T-2.66 A+0.00 | L27 T-0.98 A+8.08 | L27 T-2.66 A+0.00 | late_output_emergence |
| deepseek7b | plant | L23 | L22 | L25 T-2.28 A-33.06 | L27 T-2.42 A+0.00 | L27 T-1.21 A-2.30 | L27 T-2.42 A+0.00 | late_output_emergence |

## Reading Rules
- onset is the first scanned layer with target_delta <= -0.5.
- final output re-emergence means the final scanned layer output is much stronger than its input.
