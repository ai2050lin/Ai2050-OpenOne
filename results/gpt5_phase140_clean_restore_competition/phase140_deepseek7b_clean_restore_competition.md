# Phase 140 Clean Restore Competition: deepseek7b

Generated: 2026-06-15 06:46:50
Peak layer: L27; true last layer: L28; rank: 8; train/test: 16/32; lambda: 0.5

| category | transfer | remove | best target restore | best clean restore | clean first tokens |
|---|---|---|---|---|---|
| number | R2=+0.26, cos=+0.99 | T-1.27 R+0.00 | block_output s2.0 T+1.12 R+1.74 rec+1.88 clean+1.01 comp=furniture:+1.74 | block_output s2.0 T+1.12 R+1.74 rec+1.88 clean+1.01 comp=furniture:+1.74 |  often:0.54,  likely:0.19,  a:0.17 |
| container | R2=+0.41, cos=+0.98 | T-1.41 R+0.00 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 clean+1.47 comp=machine:-0.07 | input_answer s2.0 T+0.66 R+0.00 rec+1.47 clean+1.47 comp=machine:-0.07 |  ":0.19,  a:0.19,  the:0.15 |
| plant | R2=+0.32, cos=+0.98 | T-1.59 R+0.00 | block_output s2.0 T-0.11 R+1.37 rec+0.93 clean+0.24 comp=furniture:+1.37 | input_answer s0.5 T-0.71 R+0.17 rec+0.56 clean+0.47 comp=tool:+0.17 |  the:0.44,  ":0.23,  either:0.17 |
| time | R2=+0.66, cos=+0.99 | T-1.44 R+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 clean-0.12 comp=furniture:+0.17 | block_output s0.25 T-1.49 R+0.17 rec-0.04 clean-0.12 comp=furniture:+0.17 |  the:0.50,  either:0.17,  ":0.17 |
| clothing | R2=+0.49, cos=+0.98 | T+0.42 R+0.43 | block_output s1.5 T+0.89 R+0.65 rec+1.10 clean+0.78 comp=furniture:+0.65 | block_output s1.5 T+0.89 R+0.65 rec+1.10 clean+0.78 comp=furniture:+0.65 |  a:0.31,  the:0.19, :

:0.15 |
| furniture | R2=+0.41, cos=+0.98 | T-0.24 R+0.00 | block_output s1.0 T+0.09 R+0.19 rec+1.36 clean+1.27 comp=clothing:+0.19 | block_output s1.0 T+0.09 R+0.19 rec+1.36 clean+1.27 comp=clothing:+0.19 |  the:0.29, :

:0.17,  either:0.17 |
