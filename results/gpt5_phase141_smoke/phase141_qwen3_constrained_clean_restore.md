# Phase 141 Constrained Clean Restore: qwen3

Generated: 2026-06-15 07:36:00
Peak layer: L35; true last layer: L36; rank: 4; train/test: 2/2; threshold: 0.25

| category | transfer | remove | constrained | min release | best target |
|---|---|---|---|---|---|
| number | R2=+0.05, cos=+0.99 | T+0.30 R+1.01 | NONE | input_answer s0.5 T-2.44 R+0.00 rec-9.03 clean=False comp=furniture:-1.17 | mlp_input s1.0 T+9.72 R+10.38 rec+31.05 clean=False comp=sound:+10.38 |
| container | R2=+0.09, cos=+0.99 | T-0.66 R+0.90 | NONE | attention_output s1.0 T-0.70 R+0.00 rec-0.07 clean=False comp=weather:-0.28 | mlp_input s0.5 T+7.02 R+10.53 rec+11.67 clean=False comp=number:+10.53 |
