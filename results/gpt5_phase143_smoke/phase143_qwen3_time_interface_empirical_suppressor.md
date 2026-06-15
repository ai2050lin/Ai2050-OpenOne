# Phase 143 Time Interface Empirical Suppressor: qwen3

Generated: 2026-06-15 09:22:02
True last layer: L36; train/test: 2/2

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| container@L36 | R2=+0.09, cos=+0.99 | T-0.66 R+0.90 | 0 | NONE | L36 support mlp_input s0.5 T+7.02 R+10.53 rec+11.67 clean=False comp=number:+10.53 | L36 naive_joint input_answer s0.25 T+5.94 R+11.55 rec+10.04 clean=False comp=shape:+11.55 | L36 empirical_joint input_answer s0.25 T+5.94 R+11.55 rec+10.04 clean=False comp=shape:+11.55 |
| time@L36 | R2=+0.33, cos=+0.99 | T-0.08 R+0.92 | 0 | NONE | L36 support mlp_input s0.5 T+9.90 R+11.96 rec+122.72 clean=False comp=sound:+11.96 | L36 empirical_joint input_answer s0.25 T+10.77 R+11.54 rec+133.46 clean=False comp=fruit:+11.54 | L36 empirical_joint input_answer s0.25 T+10.77 R+11.54 rec+133.46 clean=False comp=fruit:+11.54 |
