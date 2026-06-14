# Phase 123 Cross-model Attention MLP Writer Localization

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: qwen3: L32-L35 monitor L35; glm4: L15-L18 monitor L18; deepseek7b: L24-L27 monitor L27; rank: 16; top-k heads/group: 4

## Cross-model Table
| model | category | best pre-head | best object-head | best self-head | best random-head | best pre-MLP | best answer-MLP | class |
|---|---|---|---|---|---|---|---|---|
| qwen3 | number | L33 H24 pre_answer_top T-0.30 R+0.00 Aproj-1.15 | L35 H28 object_top T-0.01 R+0.05 Aproj-0.01 | L35 H28 self_top T-0.01 R+0.05 Aproj-0.01 | L34 H14 random T-0.00 R+0.01 Aproj+0.03 | L32 pre_answer T-0.06 R+0.10 Aproj+1.26 | L34 answer_last T-0.31 R+0.46 Aproj+31.25 | weak_or_control_like |
| qwen3 | container | L33 H24 pre_answer_top T-0.08 R+0.00 Aproj+1.80 | L34 H21 object_top T-0.02 R+0.01 Aproj-0.66 | L35 H29 self_top T-0.01 R+0.07 Aproj-1.82 | L33 H24 random T-0.08 R+0.00 Aproj+1.80 | L32 pre_answer T-0.08 R+0.11 Aproj-1.12 | L32 answer_last T-0.12 R+0.29 Aproj-13.95 | weak_or_control_like |
| qwen3 | plant | L33 H24 pre_answer_top T-0.55 R+0.00 Aproj-1.77 | L35 H27 object_top T-0.02 R+0.02 Aproj-0.36 | L33 H31 self_top T-0.03 R+0.02 Aproj+0.47 | L35 H22 random T-0.01 R+0.03 Aproj+2.74 | L34 pre_answer T-0.00 R+0.11 Aproj+0.31 | L34 answer_last T-0.24 R+0.53 Aproj+23.74 | weak_or_control_like |
| glm4 | number | L17 H1 pre_answer_top T-0.03 R+0.00 Aproj+0.02 | L17 H31 object_top T-0.02 R+0.00 Aproj-0.00 | L15 H29 self_top T-0.03 R+0.01 Aproj+0.00 | L15 H23 random T-0.02 R+0.03 Aproj+0.02 | L18 pre_answer T-0.06 R+0.11 Aproj+0.00 | L17 answer_last T-0.07 R+0.07 Aproj+0.07 | weak_or_control_like |
| glm4 | container | L17 H13 pre_answer_top T-0.01 R+0.03 Aproj+0.01 | L18 H10 object_top T-0.01 R+0.00 Aproj-0.01 | L15 H5 self_top T-0.00 R+0.03 Aproj-0.00 | L18 H9 random T-0.01 R+0.02 Aproj+0.00 | L17 pre_answer T-0.03 R+0.08 Aproj+0.01 | L17 answer_last T-0.06 R+0.06 Aproj-0.08 | weak_or_control_like |
| glm4 | plant | L17 H24 pre_answer_top T-0.01 R+0.03 Aproj+0.01 | L15 H1 object_top T-0.01 R+0.03 Aproj+0.00 | L18 H11 self_top T+0.00 R+0.04 Aproj-0.04 | L15 H1 random T-0.01 R+0.03 Aproj+0.00 | L16 pre_answer T-0.06 R+0.21 Aproj+0.01 | L18 answer_last T-0.03 R+0.16 Aproj+0.13 | weak_or_control_like |
| deepseek7b | number | L26 H17 pre_answer_top T-0.05 R+0.06 Aproj-11.14 | L24 H22 object_top T-0.09 R+0.12 Aproj-14.81 | L24 H25 self_top T-0.06 R+0.06 Aproj-12.21 | L26 H11 random T-0.05 R+0.10 Aproj-0.59 | L26 pre_answer T-0.33 R+0.00 Aproj-0.61 | L27 answer_last T-1.16 R+0.30 Aproj-183.18 | answer_mlp_readout_candidate |
| deepseek7b | container | L25 H20 pre_answer_top T-0.09 R+0.06 Aproj-1.66 | L25 H24 object_top T-0.14 R+0.00 Aproj-0.03 | L27 H8 self_top T-0.23 R+0.00 Aproj+13.23 | L24 H22 random T-0.18 R+0.00 Aproj+15.34 | L24 pre_answer T-0.34 R+0.00 Aproj-0.10 | L24 answer_last T-0.11 R+0.26 Aproj+30.26 | weak_or_control_like |
| deepseek7b | plant | L26 H17 pre_answer_top T-0.16 R+0.00 Aproj-11.85 | L24 H6 object_top T-0.03 R+0.13 Aproj+2.94 | L27 H8 self_top T-0.12 R+0.15 Aproj-12.23 | L27 H1 random T-0.06 R+0.08 Aproj+0.14 | L26 pre_answer T-0.39 R+0.00 Aproj+0.31 | L24 answer_last T-0.15 R+0.36 Aproj-21.69 | weak_or_control_like |

## Reading Rules
- pre-head means answer-token attention heads selected by post_object/pre-answer attention mass.
- object-head and random-head are controls.
- Aproj is the peak answer_last projection delta on the selected answer-site monitor axis.
- writer_candidate requires target drop and answer projection drop, while beating object/random controls.
