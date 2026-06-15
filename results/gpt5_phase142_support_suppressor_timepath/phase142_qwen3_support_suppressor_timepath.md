# Phase 142 Support/Suppressor Timepath: qwen3

Generated: 2026-06-15 08:49:53
True last layer: L36; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best joint |
|---|---|---|---|---|---|---|
| plant@L36 | R2=+0.60, cos=+0.99 | T+0.29 R+0.51 | 0 | NONE | L36 support mlp_input s0.5 T+8.27 R+12.40 rec+27.47 clean=False comp=sound:+12.40 | L36 joint attention_output s0.25 T+1.05 R+1.40 rec+2.62 clean=False comp=vehicle:+1.40 |
| time@L36 | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | L36 support mlp_input s0.5 T+9.97 R+12.34 rec+240.22 clean=False comp=sound:+12.34 | L36 joint mlp_input s0.4 T+1.79 R+1.91 rec+44.04 clean=False comp=event:+1.91 |
| plant@L35 | R2=+0.45, cos=+0.97 | T-1.63 R+0.00 | 0 | NONE | L35 support attention_output s0.25 T-1.66 R+0.00 rec-0.02 clean=False comp=food:-0.40 | L35 joint mlp_input s0.4 T-1.15 R+3.21 rec+0.29 clean=False comp=clothing:+3.21 |
| time@L35 | R2=+0.39, cos=+0.98 | T-3.05 R+0.00 | 0 | NONE | L35 support mlp_input s0.5 T-2.24 R+1.25 rec+0.27 clean=False comp=clothing:+1.25 | L35 joint mlp_input s0.5 T-1.60 R+1.79 rec+0.47 clean=False comp=furniture:+1.79 |
