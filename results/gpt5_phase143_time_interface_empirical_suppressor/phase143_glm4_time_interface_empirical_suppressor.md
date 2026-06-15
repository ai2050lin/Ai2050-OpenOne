# Phase 143 Time Interface Empirical Suppressor: glm4

Generated: 2026-06-15 09:23:22
True last layer: L40; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| container@L40 | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | 0 | NONE | L40 support mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 | L40 empirical_joint input_answer s0.5 T+2.16 R+3.13 rec+12.70 clean=False comp=action:+3.13 | L40 empirical_joint input_answer s0.5 T+2.16 R+3.13 rec+12.70 clean=False comp=action:+3.13 |
| time@L40 | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | L40 support mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 | L40 empirical_joint input_answer s0.35 T+2.93 R+2.94 rec+17.33 clean=False comp=event:+2.94 | L40 empirical_joint input_answer s0.35 T+2.93 R+2.94 rec+17.33 clean=False comp=event:+2.94 |
| container@L39 | R2=+0.61, cos=+0.99 | T+0.80 R+1.06 | 0 | NONE | L39 support mlp_input s0.5 T+1.18 R+1.11 rec+0.47 clean=False comp=fruit:+1.11 | L39 empirical_joint input_answer s0.5 T+2.46 R+3.92 rec+2.07 clean=False comp=time:+3.92 | L39 empirical_joint input_answer s0.5 T+2.46 R+3.92 rec+2.07 clean=False comp=time:+3.92 |
| time@L39 | R2=+0.35, cos=+0.98 | T+0.68 R+0.88 | 0 | NONE | L39 support mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 | L39 naive_joint input_answer s0.5 T+4.23 R+4.15 rec+5.24 clean=False comp=action:+4.15 | L39 empirical_joint input_answer s0.35 T+2.98 R+3.31 rec+3.40 clean=False comp=action:+3.31 |
