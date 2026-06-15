# Phase 140 Cross-model Clean Restore Competition Summary

## qwen3

Peak layer: L35; true last layer: L36; rank: 8; train/test: 8/16; lambda: 0.5

| category | transfer | remove | best target | best clean | first tokens |
|---|---|---|---|---|---|
| number | R2=+0.28, cos=+0.99 | T-1.21 R+0.78 | input_answer s2.0 T+7.26 R+8.33 rec+7.02 clean+2.86 comp=communication:+8.33 | input_answer s2.0 T+7.26 R+8.33 rec+7.02 clean+2.86 comp=communication:+8.33 | clidean:0.35, STRUCTOR:0.25, theless:0.17 |
| container | R2=+0.30, cos=+0.99 | T-0.55 R+0.41 | input_answer s2.0 T+6.45 R+8.21 rec+12.77 clean+8.67 comp=communication:+8.21 | input_answer s2.0 T+6.45 R+8.21 rec+12.77 clean+8.67 comp=communication:+8.21 | theless:0.49, terior:0.25, asticsearch:0.14 |
| plant | R2=+0.60, cos=+0.99 | T+0.51 R+0.91 | input_answer s2.0 T+4.84 R+7.19 rec+8.52 clean+4.93 comp=building:+7.19 | input_answer s2.0 T+4.84 R+7.19 rec+8.52 clean+4.93 comp=building:+7.19 |  abstract:0.24, [](:0.21,  erection:0.11 |
| time | R2=+0.35, cos=+0.99 | T-0.24 R+0.91 | input_answer s2.0 T+5.48 R+7.93 rec+23.40 clean+19.43 comp=communication:+7.93 | input_answer s2.0 T+5.48 R+7.93 rec+23.40 clean+19.43 comp=communication:+7.93 | luetooth:0.29, STRACT:0.26,  whichever:0.17 |
| clothing | R2=+0.69, cos=+0.99 | T+0.65 R+0.71 | input_answer s2.0 T+2.96 R+7.07 rec+3.56 clean+0.03 comp=communication:+7.07 | input_answer s2.0 T+2.96 R+7.07 rec+3.56 clean+0.03 comp=communication:+7.07 |  gonna:0.26,  wearer:0.23, REDIENT:0.10 |
| furniture | R2=+0.61, cos=+0.99 | T-0.11 R+0.78 | input_answer s2.0 T+4.38 R+7.31 rec+40.49 clean+36.84 comp=communication:+7.31 | input_answer s2.0 T+4.38 R+7.31 rec+40.49 clean+36.84 comp=communication:+7.31 |  gonna:0.27, STRACT:0.19,  __:0.10 |

## glm4

Peak layer: L18; true last layer: L40; rank: 8; train/test: 8/16; lambda: 0.5

| category | transfer | remove | best target | best clean | first tokens |
|---|---|---|---|---|---|
| number | R2=+0.42, cos=+0.93 | T-0.03 R+0.50 | block_output s0.25 T-0.29 R+0.20 rec-7.43 clean-7.53 comp=fruit:+0.20 | block_output s0.25 T-0.29 R+0.20 rec-7.43 clean-7.53 comp=fruit:+0.20 |  ":0.36,  the:0.32,  more:0.09 |
| container | R2=+0.46, cos=+0.88 | T-0.14 R+0.34 | input_answer s2.0 T+0.32 R+1.21 rec+3.36 clean+2.76 comp=plant:+1.21 | input_answer s2.0 T+0.32 R+1.21 rec+3.36 clean+2.76 comp=plant:+1.21 |  more:0.16,  determined:0.16,  abstract:0.11 |
| plant | R2=+0.30, cos=+0.88 | T-0.14 R+0.32 | input_answer s2.0 T+0.40 R+1.18 rec+3.90 clean+3.31 comp=container:+1.18 | input_answer s2.0 T+0.40 R+1.18 rec+3.90 clean+3.31 comp=container:+1.18 |  the:0.23,  ":0.19,  more:0.17 |
| time | R2=+0.53, cos=+0.93 | T+0.12 R+0.45 | block_output s0.25 T-0.19 R+0.22 rec-2.61 clean-2.72 comp=fruit:+0.22 | block_output s0.25 T-0.19 R+0.22 rec-2.61 clean-2.72 comp=fruit:+0.22 |  ":0.35,  the:0.22,  more:0.12 |
| clothing | R2=+0.40, cos=+0.88 | T+0.08 R+0.22 | input_answer s0.5 T+0.12 R+0.38 rec+0.59 clean+0.40 comp=profession:+0.38 | input_answer s0.5 T+0.12 R+0.38 rec+0.59 clean+0.40 comp=profession:+0.38 |  ":0.34,  the:0.32,  more:0.16 |
| furniture | R2=+0.28, cos=+0.84 | T+0.16 R+0.39 | block_output s0.25 T-0.14 R+0.00 rec-1.87 clean-1.87 comp=substance:-0.04 | block_output s0.25 T-0.14 R+0.00 rec-1.87 clean-1.87 comp=substance:-0.04 |  ":0.51,  the:0.25,  more:0.17 |

## deepseek7b

Peak layer: L27; true last layer: L28; rank: 8; train/test: 16/32; lambda: 0.5

| category | transfer | remove | best target | best clean | first tokens |
|---|---|---|---|---|---|
| number | R2=+0.26, cos=+0.99 | T-1.27 R+0.00 | block_output s2.0 T+1.12 R+1.74 rec+1.88 clean+1.01 comp=furniture:+1.74 | block_output s2.0 T+1.12 R+1.74 rec+1.88 clean+1.01 comp=furniture:+1.74 |  often:0.54,  likely:0.19,  a:0.17 |
| container | R2=+0.41, cos=+0.98 | T-1.41 R+0.00 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 clean+1.47 comp=machine:-0.07 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 clean+1.47 comp=machine:-0.07 |  ":0.19,  a:0.19,  the:0.15 |
| plant | R2=+0.32, cos=+0.98 | T-1.59 R+0.00 | block_output s2.0 T-0.11 R+1.37 rec+0.93 clean+0.24 comp=furniture:+1.37 | input_answer s0.5 T-0.71 R+0.17 rec+0.56 clean+0.47 comp=tool:+0.17 |  the:0.44,  ":0.23,  either:0.17 |
| time | R2=+0.66, cos=+0.99 | T-1.44 R+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 clean-0.12 comp=furniture:+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 clean-0.12 comp=furniture:+0.17 |  the:0.50,  either:0.17,  ":0.17 |
| clothing | R2=+0.49, cos=+0.98 | T+0.42 R+0.43 | block_output s1.5 T+0.89 R+0.65 rec+1.10 clean+0.78 comp=furniture:+0.65 | block_output s1.5 T+0.89 R+0.65 rec+1.10 clean+0.78 comp=furniture:+0.65 |  a:0.31,  the:0.19, :

:0.15 |
| furniture | R2=+0.41, cos=+0.98 | T-0.24 R+0.00 | block_output s1.0 T+0.09 R+0.19 rec+1.36 clean+1.27 comp=clothing:+0.19 | block_output s1.0 T+0.09 R+0.19 rec+1.36 clean+1.27 comp=clothing:+0.19 |  the:0.29, :

:0.17,  either:0.17 |

