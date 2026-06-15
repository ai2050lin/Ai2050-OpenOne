# Phase 143 Time Interface Empirical Suppressor: qwen3

Generated: 2026-06-15 09:22:30
True last layer: L36; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| container@L36 | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | 0 | NONE | L36 support mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 | L36 empirical_joint input_answer s0.25 T+5.85 R+10.94 rec+14.53 clean=False comp=sound:+10.94 | L36 empirical_joint input_answer s0.25 T+5.85 R+10.94 rec+14.53 clean=False comp=sound:+10.94 |
| time@L36 | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | L36 support mlp_input s0.5 T+9.97 R+12.34 rec+240.22 clean=False comp=sound:+12.34 | L36 naive_joint input_answer s0.25 T+12.40 R+12.81 rec+298.55 clean=False comp=number:+12.81 | L36 empirical_joint input_answer s0.25 T+11.11 R+12.79 rec+267.53 clean=False comp=animal:+12.79 |
| container@L35 | R2=+0.37, cos=+0.99 | T-1.43 R+0.30 | 0 | NONE | L35 support input_answer s0.5 T-1.03 R+0.11 rec+0.28 clean=False comp=clothing:+0.11 | L35 empirical_joint input_answer s0.5 T-0.77 R+0.14 rec+0.46 clean=False comp=food:+0.14 | L35 empirical_joint input_answer s0.5 T-0.77 R+0.14 rec+0.46 clean=False comp=food:+0.14 |
| time@L35 | R2=+0.39, cos=+0.98 | T-3.05 R+0.00 | 0 | NONE | L35 support mlp_input s0.5 T-2.24 R+1.25 rec+0.27 clean=False comp=clothing:+1.25 | L35 naive_joint mlp_input s0.5 T-1.60 R+1.79 rec+0.47 clean=False comp=furniture:+1.79 | L35 empirical_joint mlp_input s0.5 T-1.60 R+1.79 rec+0.47 clean=False comp=furniture:+1.79 |
