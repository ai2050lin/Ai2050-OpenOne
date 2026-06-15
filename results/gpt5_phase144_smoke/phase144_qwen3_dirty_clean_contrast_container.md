# Phase 144 Dirty-Clean Contrast Container: qwen3

Generated: 2026-06-15 09:50:40
True last layer: L36; train/test: 2/2

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| container@L36 | R2=+0.09, cos=+0.99 | T-0.66 R+0.90 | 0 | NONE | L36 support mlp_input s0.5 T+7.02 R+10.53 rec+11.67 clean=False comp=number:+10.53 | L36 contrast_joint mlp_input s0.5 T+6.98 R+10.42 rec+11.62 clean=False comp=number:+10.42 |
| time@L36 | R2=+0.33, cos=+0.99 | T-0.08 R+0.92 | 0 | NONE | L36 support mlp_input s0.5 T+9.90 R+11.96 rec+122.72 clean=False comp=sound:+11.96 | L36 contrast_joint mlp_input s0.5 T+9.90 R+11.96 rec+122.72 clean=False comp=sound:+11.96 |
