# Phase 141 Constrained Clean Restore: glm4

Generated: 2026-06-15 07:37:44
Peak layer: L18; true last layer: L40; rank: 8; train/test: 10/20; threshold: 0.25

| category | transfer | remove | constrained | min release | best target |
|---|---|---|---|---|---|
| number | R2=+0.45, cos=+0.92 | T-0.01 R+0.53 | NONE | mlp_output s1.0 T-0.91 R+0.03 rec-143.23 clean=False comp=communication:+0.03 | mlp_input s0.25 T-0.10 R+0.46 rec-15.07 clean=False comp=fruit:+0.46 |
| container | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | NONE | block_output s1.0 T-0.62 R+0.00 rec-2.35 clean=False comp=plant:+0.00 | mlp_input s2.0 T-0.28 R+0.54 rec-0.51 clean=False comp=event:+0.54 |
| plant | R2=+0.40, cos=+0.92 | T-0.11 R+0.32 | block_output s1.0 T+0.03 R+0.09 rec+1.24 clean=True comp=weather:+0.09 | block_output s0.5 T-0.02 R+0.00 rec+0.83 clean=True comp=furniture:-0.05 | input_answer s2.0 T+0.56 R+1.14 rec+6.13 clean=False comp=weather:+1.14 |
| time | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | NONE | block_output s2.0 T-1.78 R+0.04 rec-12.13 clean=False comp=machine:+0.04 | mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 |
| clothing | R2=+0.23, cos=+0.82 | T+0.13 R+0.16 | NONE | attention_output s0.25 T+0.09 R+0.04 rec-0.27 clean=False comp=furniture:+0.04 | input_answer s0.5 T+0.17 R+0.15 rec+0.28 clean=False comp=furniture:+0.15 |
| furniture | R2=-0.01, cos=+0.67 | T+0.16 R+0.33 | NONE | mlp_output s0.5 T-0.05 R+0.04 rec-1.30 clean=False comp=communication:+0.04 | mlp_input s0.25 T+0.13 R+0.30 rec-0.17 clean=False comp=fruit:+0.30 |
