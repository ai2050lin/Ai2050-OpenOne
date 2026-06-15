# Phase 141 Cross-model Constrained Clean Restore Summary

## qwen3

Peak layer: L35; true last layer: L36; rank: 8; train/test: 10/20; threshold: 0.25

| category | transfer | remove | clean count | constrained | min release | best target |
|---|---|---|---|---|---|---|
| number | R2=+0.17, cos=+0.99 | T-1.56 R+0.70 | 0 | NONE | attention_output s0.25 T-3.77 R+0.00 rec-1.41 clean=False comp=weather:-1.81 | mlp_input s1.5 T+13.59 R+13.14 rec+9.69 clean=False comp=sound:+13.14 |
| container | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | 0 | NONE | attention_output s0.5 T-1.01 R+0.00 rec-1.33 clean=False comp=weather:-1.13 | mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 |
| plant | R2=+0.60, cos=+0.99 | T+0.29 R+0.51 | 0 | NONE | attention_output s0.25 T-2.11 R+0.00 rec-8.28 clean=False comp=container:-0.91 | mlp_input s1.5 T+9.39 R+13.43 rec+31.34 clean=False comp=sound:+13.43 |
| time | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | attention_output s0.25 T-2.25 R+0.00 rec-53.04 clean=False comp=container:-0.62 | mlp_input s1.5 T+10.98 R+13.18 rec+264.36 clean=False comp=sound:+13.18 |
| clothing | R2=+0.60, cos=+0.99 | T+0.72 R+0.76 | 0 | NONE | attention_output s0.25 T-1.39 R+0.00 rec-2.93 clean=False comp=container:-0.77 | mlp_input s1.0 T+11.10 R+13.65 rec+14.42 clean=False comp=number:+13.65 |
| furniture | R2=+0.52, cos=+0.99 | T+0.29 R+0.88 | 0 | NONE | attention_output s0.25 T-1.69 R+0.00 rec-6.83 clean=False comp=container:-0.49 | mlp_input s1.0 T+11.02 R+13.54 rec+36.97 clean=False comp=number:+13.54 |

## glm4

Peak layer: L18; true last layer: L40; rank: 8; train/test: 10/20; threshold: 0.25

| category | transfer | remove | clean count | constrained | min release | best target |
|---|---|---|---|---|---|---|
| number | R2=+0.45, cos=+0.92 | T-0.01 R+0.53 | 0 | NONE | mlp_output s1.0 T-0.91 R+0.03 rec-143.23 clean=False comp=communication:+0.03 | mlp_input s0.25 T-0.10 R+0.46 rec-15.07 clean=False comp=fruit:+0.46 |
| container | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | 0 | NONE | block_output s1.0 T-0.62 R+0.00 rec-2.35 clean=False comp=plant:+0.00 | mlp_input s2.0 T-0.28 R+0.54 rec-0.51 clean=False comp=event:+0.54 |
| plant | R2=+0.40, cos=+0.92 | T-0.11 R+0.32 | 8 | block_output s1.0 T+0.03 R+0.09 rec+1.24 clean=True comp=weather:+0.09 | block_output s0.5 T-0.02 R+0.00 rec+0.83 clean=True comp=furniture:-0.05 | input_answer s2.0 T+0.56 R+1.14 rec+6.13 clean=False comp=weather:+1.14 |
| time | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | block_output s2.0 T-1.78 R+0.04 rec-12.13 clean=False comp=machine:+0.04 | mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 |
| clothing | R2=+0.23, cos=+0.82 | T+0.13 R+0.16 | 0 | NONE | attention_output s0.25 T+0.09 R+0.04 rec-0.27 clean=False comp=furniture:+0.04 | input_answer s0.5 T+0.17 R+0.15 rec+0.28 clean=False comp=furniture:+0.15 |
| furniture | R2=-0.01, cos=+0.67 | T+0.16 R+0.33 | 0 | NONE | mlp_output s0.5 T-0.05 R+0.04 rec-1.30 clean=False comp=communication:+0.04 | mlp_input s0.25 T+0.13 R+0.30 rec-0.17 clean=False comp=fruit:+0.30 |

## deepseek7b

Peak layer: L27; true last layer: L28; rank: 8; train/test: 20/40; threshold: 0.25

| category | transfer | remove | clean count | constrained | min release | best target |
|---|---|---|---|---|---|---|
| number | R2=+0.49, cos=+0.99 | T-1.34 R+0.15 | 1 | attention_output s0.25 T-0.54 R+0.00 rec+0.60 clean=True comp=vehicle:-0.03 | attention_output s0.25 T-0.54 R+0.00 rec+0.60 clean=True comp=vehicle:-0.03 | attention_output s2.0 T+1.07 R+1.15 rec+1.79 clean=False comp=animal:+1.15 |
| container | R2=+0.60, cos=+0.99 | T-1.78 R+0.00 | 1 | input_answer s1.0 T-0.09 R+0.13 rec+0.95 clean=True comp=machine:+0.13 | input_answer s0.5 T-1.13 R+0.00 rec+0.36 clean=False comp=machine:-0.39 | mlp_input s2.0 T+3.81 R+2.64 rec+3.14 clean=False comp=communication:+2.64 |
| plant | R2=+0.52, cos=+0.98 | T-1.84 R+0.13 | 0 | NONE | attention_output s0.25 T-0.95 R+0.00 rec+0.48 clean=False comp=tool:-0.29 | attention_output s2.0 T+0.67 R+2.24 rec+1.36 clean=False comp=tool:+2.24 |
| time | R2=+0.60, cos=+0.99 | T-2.07 R+0.75 | 0 | NONE | mlp_input s0.25 T-1.30 R+0.00 rec+0.37 clean=False comp=building:-0.27 | mlp_input s2.0 T-0.76 R+0.83 rec+0.63 clean=False comp=clothing:+0.83 |
| clothing | R2=+0.48, cos=+0.98 | T+0.67 R+0.62 | 0 | NONE | attention_output s0.25 T-0.06 R+0.00 rec-1.09 clean=False comp=furniture:-0.08 | block_output s0.5 T+0.82 R+0.70 rec+0.23 clean=False comp=furniture:+0.70 |
| furniture | R2=+0.52, cos=+0.98 | T-0.25 R+0.00 | 3 | block_output s0.25 T+0.10 R+0.05 rec+1.40 clean=True comp=clothing:+0.05 | attention_output s0.25 T-0.77 R+0.00 rec-2.03 clean=False comp=machine:-0.49 | block_output s2.0 T+0.73 R+1.22 rec+3.90 clean=False comp=machine:+1.22 |

