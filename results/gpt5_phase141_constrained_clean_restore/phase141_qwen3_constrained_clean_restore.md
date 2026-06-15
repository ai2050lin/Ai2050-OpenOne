# Phase 141 Constrained Clean Restore: qwen3

Generated: 2026-06-15 07:36:22
Peak layer: L35; true last layer: L36; rank: 8; train/test: 10/20; threshold: 0.25

| category | transfer | remove | constrained | min release | best target |
|---|---|---|---|---|---|
| number | R2=+0.17, cos=+0.99 | T-1.56 R+0.70 | NONE | attention_output s0.25 T-3.77 R+0.00 rec-1.41 clean=False comp=weather:-1.81 | mlp_input s1.5 T+13.59 R+13.14 rec+9.69 clean=False comp=sound:+13.14 |
| container | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | NONE | attention_output s0.5 T-1.01 R+0.00 rec-1.33 clean=False comp=weather:-1.13 | mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 |
| plant | R2=+0.60, cos=+0.99 | T+0.29 R+0.51 | NONE | attention_output s0.25 T-2.11 R+0.00 rec-8.28 clean=False comp=container:-0.91 | mlp_input s1.5 T+9.39 R+13.43 rec+31.34 clean=False comp=sound:+13.43 |
| time | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | NONE | attention_output s0.25 T-2.25 R+0.00 rec-53.04 clean=False comp=container:-0.62 | mlp_input s1.5 T+10.98 R+13.18 rec+264.36 clean=False comp=sound:+13.18 |
| clothing | R2=+0.60, cos=+0.99 | T+0.72 R+0.76 | NONE | attention_output s0.25 T-1.39 R+0.00 rec-2.93 clean=False comp=container:-0.77 | mlp_input s1.0 T+11.10 R+13.65 rec+14.42 clean=False comp=number:+13.65 |
| furniture | R2=+0.52, cos=+0.99 | T+0.29 R+0.88 | NONE | attention_output s0.25 T-1.69 R+0.00 rec-6.83 clean=False comp=container:-0.49 | mlp_input s1.0 T+11.02 R+13.54 rec+36.97 clean=False comp=number:+13.54 |
