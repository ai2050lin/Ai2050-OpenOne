# Phase 144 Dirty-Clean Contrast Container: glm4

Generated: 2026-06-15 09:51:56
True last layer: L40; train/test: 10/20

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| container@L40 | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | 0 | NONE | L40 support mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 | L40 contrast_joint mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 |
| time@L40 | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | L40 support mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 | L40 contrast_joint mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 |
| container@L39 | R2=+0.61, cos=+0.99 | T+0.80 R+1.06 | 0 | NONE | L39 support mlp_input s1.0 T+1.73 R+1.41 rec+1.16 clean=False comp=relation:+1.41 | L39 contrast_joint mlp_input s1.0 T+1.73 R+1.41 rec+1.16 clean=False comp=relation:+1.41 |
| time@L39 | R2=+0.35, cos=+0.98 | T+0.68 R+0.88 | 0 | NONE | L39 support mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 | L39 contrast_joint mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 |
