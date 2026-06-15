# Phase 144 Dirty-Clean Contrast Container: qwen3

Generated: 2026-06-15 09:51:05
True last layer: L36; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| container@L36 | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | 0 | NONE | L36 support mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 | L36 contrast_joint mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 |
| time@L36 | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | L36 support mlp_input s1.0 T+10.72 R+12.95 rec+258.20 clean=False comp=sound:+12.95 | L36 contrast_joint mlp_input s1.0 T+10.72 R+12.95 rec+258.20 clean=False comp=sound:+12.95 |
| container@L35 | R2=+0.37, cos=+0.99 | T-1.43 R+0.30 | 0 | NONE | L35 support input_answer s0.5 T-1.03 R+0.11 rec+0.28 clean=False comp=clothing:+0.11 | L35 contrast_joint mlp_input s1.0 T-0.72 R+6.28 rec+0.50 clean=False comp=furniture:+6.28 |
| time@L35 | R2=+0.39, cos=+0.98 | T-3.05 R+0.00 | 0 | NONE | L35 support mlp_input s1.0 T-0.25 R+1.03 rec+0.92 clean=False comp=furniture:+1.03 | L35 contrast_joint mlp_input s1.0 T-0.25 R+1.03 rec+0.92 clean=False comp=furniture:+1.03 |
