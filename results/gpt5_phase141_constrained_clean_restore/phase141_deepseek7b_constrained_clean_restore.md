# Phase 141 Constrained Clean Restore: deepseek7b

Generated: 2026-06-15 07:39:51
Peak layer: L27; true last layer: L28; rank: 8; train/test: 20/40; threshold: 0.25

| category | transfer | remove | constrained | min release | best target |
|---|---|---|---|---|---|
| number | R2=+0.49, cos=+0.99 | T-1.34 R+0.15 | attention_output s0.25 T-0.54 R+0.00 rec+0.60 clean=True comp=vehicle:-0.03 | attention_output s0.25 T-0.54 R+0.00 rec+0.60 clean=True comp=vehicle:-0.03 | attention_output s2.0 T+1.07 R+1.15 rec+1.79 clean=False comp=animal:+1.15 |
| container | R2=+0.60, cos=+0.99 | T-1.78 R+0.00 | input_answer s1.0 T-0.09 R+0.13 rec+0.95 clean=True comp=machine:+0.13 | input_answer s0.5 T-1.13 R+0.00 rec+0.36 clean=False comp=machine:-0.39 | mlp_input s2.0 T+3.81 R+2.64 rec+3.14 clean=False comp=communication:+2.64 |
| plant | R2=+0.52, cos=+0.98 | T-1.84 R+0.13 | NONE | attention_output s0.25 T-0.95 R+0.00 rec+0.48 clean=False comp=tool:-0.29 | attention_output s2.0 T+0.67 R+2.24 rec+1.36 clean=False comp=tool:+2.24 |
| time | R2=+0.60, cos=+0.99 | T-2.07 R+0.75 | NONE | mlp_input s0.25 T-1.30 R+0.00 rec+0.37 clean=False comp=building:-0.27 | mlp_input s2.0 T-0.76 R+0.83 rec+0.63 clean=False comp=clothing:+0.83 |
| clothing | R2=+0.48, cos=+0.98 | T+0.67 R+0.62 | NONE | attention_output s0.25 T-0.06 R+0.00 rec-1.09 clean=False comp=furniture:-0.08 | block_output s0.5 T+0.82 R+0.70 rec+0.23 clean=False comp=furniture:+0.70 |
| furniture | R2=+0.52, cos=+0.98 | T-0.25 R+0.00 | block_output s0.25 T+0.10 R+0.05 rec+1.40 clean=True comp=clothing:+0.05 | attention_output s0.25 T-0.77 R+0.00 rec-2.03 clean=False comp=machine:-0.49 | block_output s2.0 T+0.73 R+1.22 rec+3.90 clean=False comp=machine:+1.22 |
