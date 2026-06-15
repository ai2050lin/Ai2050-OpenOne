# Phase 142 Support/Suppressor Timepath: glm4

Generated: 2026-06-15 08:50:43
True last layer: L40; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best joint |
|---|---|---|---|---|---|---|
| plant@L40 | R2=+0.40, cos=+0.92 | T-0.11 R+0.32 | 8 | L40 joint attention_output s0.5 T+0.02 R+0.06 rec+1.21 clean=True comp=food:+0.06 | L40 support attention_output s0.5 T-0.04 R+0.00 rec+0.62 clean=True comp=tool:-0.04 | L40 joint attention_output s0.5 T+0.02 R+0.06 rec+1.21 clean=True comp=food:+0.06 |
| time@L40 | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | L40 support mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 | L40 joint mlp_input s0.25 T+0.39 R+0.51 rec+1.42 clean=False comp=fruit:+0.51 |
| plant@L39 | R2=+0.58, cos=+0.99 | T-0.90 R+1.03 | 0 | NONE | L39 support mlp_input s0.5 T-0.58 R+1.26 rec+0.36 clean=False comp=container:+1.26 | L39 joint mlp_input s0.5 T-0.84 R+1.03 rec+0.07 clean=False comp=number:+1.03 |
| time@L39 | R2=+0.35, cos=+0.98 | T+0.68 R+0.88 | 0 | NONE | L39 support mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 | L39 joint attention_output s0.5 T+1.37 R+1.65 rec+1.03 clean=False comp=number:+1.65 |
