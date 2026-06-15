# Phase 140 Clean Restore Competition: glm4

Generated: 2026-06-15 06:45:40
Peak layer: L18; true last layer: L40; rank: 8; train/test: 8/16; lambda: 0.5

| category | transfer | remove | best target restore | best clean restore | clean first tokens |
|---|---|---|---|---|---|
| number | R2=+0.42, cos=+0.93 | T-0.03 R+0.50 | block_output s0.25 T-0.29 R+0.20 rec-7.43 clean-7.53 comp=fruit:+0.20 | block_output s0.25 T-0.29 R+0.20 rec-7.43 clean-7.53 comp=fruit:+0.20 |  ":0.36,  the:0.32,  more:0.09 |
| container | R2=+0.46, cos=+0.88 | T-0.14 R+0.34 | input_answer s2.0 T+0.32 R+1.21 rec+3.36 clean+2.76 comp=plant:+1.21 | input_answer s2.0 T+0.32 R+1.21 rec+3.36 clean+2.76 comp=plant:+1.21 |  more:0.16,  determined:0.16,  abstract:0.11 |
| plant | R2=+0.30, cos=+0.88 | T-0.14 R+0.32 | input_answer s2.0 T+0.40 R+1.18 rec+3.90 clean+3.31 comp=container:+1.18 | input_answer s2.0 T+0.40 R+1.18 rec+3.90 clean+3.31 comp=container:+1.18 |  the:0.23,  ":0.19,  more:0.17 |
| time | R2=+0.53, cos=+0.93 | T+0.12 R+0.45 | block_output s0.25 T-0.19 R+0.22 rec-2.61 clean-2.72 comp=fruit:+0.22 | block_output s0.25 T-0.19 R+0.22 rec-2.61 clean-2.72 comp=fruit:+0.22 |  ":0.35,  the:0.22,  more:0.12 |
| clothing | R2=+0.40, cos=+0.88 | T+0.08 R+0.22 | input_answer s0.5 T+0.12 R+0.38 rec+0.59 clean+0.40 comp=profession:+0.38 | input_answer s0.5 T+0.12 R+0.38 rec+0.59 clean+0.40 comp=profession:+0.38 |  ":0.34,  the:0.32,  more:0.16 |
| furniture | R2=+0.28, cos=+0.84 | T+0.16 R+0.39 | block_output s0.25 T-0.14 R+0.00 rec-1.87 clean-1.87 comp=substance:-0.04 | block_output s0.25 T-0.14 R+0.00 rec-1.87 clean-1.87 comp=substance:-0.04 |  ":0.51,  the:0.25,  more:0.17 |
