# Phase 142 Support/Suppressor Timepath: deepseek7b

Generated: 2026-06-15 08:52:08
True last layer: L28; train/test: 20/40

| category@layer | transfer | remove | clean count | best clean | best support | best joint |
|---|---|---|---|---|---|---|
| number@L28 | R2=+0.49, cos=+0.99 | T-1.34 R+0.15 | 2 | L28 support attention_output s0.3 T-0.43 R+0.12 rec+0.68 clean=True comp=animal:+0.12 | L28 support attention_output s0.5 T-0.05 R+0.61 rec+0.97 clean=False comp=animal:+0.61 | L28 joint attention_output s0.25 T-1.64 R+0.00 rec-0.22 clean=False comp=clothing:-0.29 |
| container@L28 | R2=+0.60, cos=+0.99 | T-1.78 R+0.00 | 0 | NONE | L28 support mlp_input s0.5 T+3.19 R+1.90 rec+2.79 clean=False comp=communication:+1.90 | L28 joint attention_output s0.25 T-2.18 R+0.41 rec-0.23 clean=False comp=clothing:+0.41 |
| plant@L28 | R2=+0.52, cos=+0.98 | T-1.84 R+0.13 | 2 | L28 support attention_output s0.35 T-0.69 R+0.11 rec+0.62 clean=True comp=tool:+0.11 | L28 support attention_output s0.5 T-0.37 R+0.62 rec+0.80 clean=False comp=tool:+0.62 | L28 joint attention_output s0.25 T-2.60 R+0.00 rec-0.42 clean=False comp=clothing:-0.04 |
| time@L28 | R2=+0.60, cos=+0.99 | T-2.07 R+0.75 | 0 | NONE | L28 support mlp_input s0.25 T-1.30 R+0.00 rec+0.37 clean=False comp=building:-0.27 | L28 joint attention_output s0.25 T-2.59 R+1.10 rec-0.25 clean=False comp=clothing:+1.10 |
| number@L27 | R2=+0.45, cos=+1.00 | T-1.39 R+1.40 | 0 | NONE | L27 support attention_output s0.3 T-1.46 R+0.00 rec-0.05 clean=False comp=furniture:-0.12 | L27 joint attention_output s0.25 T-1.26 R+1.32 rec+0.09 clean=False comp=furniture:+1.32 |
| container@L27 | R2=+0.64, cos=+1.00 | T-1.93 R+0.00 | 0 | NONE | L27 support attention_output s0.3 T-1.88 R+0.00 rec+0.03 clean=False comp=clothing:-1.05 | L27 joint mlp_input s0.5 T-0.98 R+0.00 rec+0.49 clean=False comp=weather:-2.37 |
| plant@L27 | R2=+0.67, cos=+0.99 | T-1.61 R+0.70 | 0 | NONE | L27 support attention_output s0.25 T-1.74 R+0.09 rec-0.08 clean=False comp=clothing:+0.09 | L27 joint attention_output s0.25 T-1.57 R+0.79 rec+0.02 clean=False comp=clothing:+0.79 |
| time@L27 | R2=+0.52, cos=+0.99 | T-2.36 R+0.71 | 3 | L27 support mlp_input s0.5 T-0.18 R+0.19 rec+0.92 clean=True comp=event:+0.19 | L27 support mlp_input s0.5 T-0.18 R+0.19 rec+0.92 clean=True comp=event:+0.19 | L27 joint attention_output s0.35 T-1.94 R+1.32 rec+0.18 clean=False comp=furniture:+1.32 |
| number@L26 | R2=+0.57, cos=+0.99 | T-1.16 R+0.88 | 0 | NONE | L26 support attention_output s0.4 T-0.62 R+0.80 rec+0.47 clean=False comp=furniture:+0.80 | L26 joint attention_output s0.35 T-0.89 R+0.95 rec+0.24 clean=False comp=furniture:+0.95 |
| container@L26 | R2=+0.65, cos=+0.99 | T-2.11 R+0.00 | 0 | NONE | L26 support attention_output s0.35 T-1.87 R+0.00 rec+0.11 clean=False comp=clothing:-0.03 | L26 joint attention_output s0.3 T-2.21 R+0.00 rec-0.05 clean=False comp=clothing:-0.31 |
| plant@L26 | R2=+0.73, cos=+0.99 | T-1.45 R+0.12 | 0 | NONE | L26 support attention_output s0.35 T-0.94 R+1.14 rec+0.36 clean=False comp=furniture:+1.14 | L26 joint attention_output s0.25 T-1.19 R+0.95 rec+0.18 clean=False comp=furniture:+0.95 |
| time@L26 | R2=+0.61, cos=+0.99 | T-2.36 R+0.32 | 0 | NONE | L26 support attention_output s0.5 T-1.20 R+1.82 rec+0.49 clean=False comp=furniture:+1.82 | L26 joint attention_output s0.5 T-1.91 R+1.15 rec+0.19 clean=False comp=furniture:+1.15 |
