# Phase 144 Cross-model Dirty-Clean Contrast Container Summary

## qwen3

True last layer: L36; train/test: 10/20; offsets: [0, -1]; sites: ['input_answer', 'mlp_input']; modes: ['support', 'contrast_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| container@L36 | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | 0 | NONE | L36 support mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 | L36 contrast_joint mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 |
| time@L36 | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | L36 support mlp_input s1.0 T+10.72 R+12.95 rec+258.20 clean=False comp=sound:+12.95 | L36 contrast_joint mlp_input s1.0 T+10.72 R+12.95 rec+258.20 clean=False comp=sound:+12.95 |
| container@L35 | R2=+0.37, cos=+0.99 | T-1.43 R+0.30 | 0 | NONE | L35 support input_answer s0.5 T-1.03 R+0.11 rec+0.28 clean=False comp=clothing:+0.11 | L35 contrast_joint mlp_input s1.0 T-0.72 R+6.28 rec+0.50 clean=False comp=furniture:+6.28 |
| time@L35 | R2=+0.39, cos=+0.98 | T-3.05 R+0.00 | 0 | NONE | L35 support mlp_input s1.0 T-0.25 R+1.03 rec+0.92 clean=False comp=furniture:+1.03 | L35 contrast_joint mlp_input s1.0 T-0.25 R+1.03 rec+0.92 clean=False comp=furniture:+1.03 |

## glm4

True last layer: L40; train/test: 10/20; offsets: [0, -1]; sites: ['input_answer', 'mlp_input']; modes: ['support', 'contrast_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| container@L40 | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | 0 | NONE | L40 support mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 | L40 contrast_joint mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 |
| time@L40 | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | L40 support mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 | L40 contrast_joint mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 |
| container@L39 | R2=+0.61, cos=+0.99 | T+0.80 R+1.06 | 0 | NONE | L39 support mlp_input s1.0 T+1.73 R+1.41 rec+1.16 clean=False comp=relation:+1.41 | L39 contrast_joint mlp_input s1.0 T+1.73 R+1.41 rec+1.16 clean=False comp=relation:+1.41 |
| time@L39 | R2=+0.35, cos=+0.98 | T+0.68 R+0.88 | 0 | NONE | L39 support mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 | L39 contrast_joint mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 |

## deepseek7b

True last layer: L28; train/test: 20/40; offsets: [0, -1, -2]; sites: ['input_answer', 'attention_output', 'mlp_input']; modes: ['support', 'contrast_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best contrast |
|---|---|---|---|---|---|---|
| number@L28 | R2=+0.49, cos=+0.99 | T-1.34 R+0.15 | 1 | L28 support attention_output s0.25 T-0.54 R+0.00 rec+0.60 clean=True comp=vehicle:-0.03 | L28 support attention_output s1.5 T+0.88 R+1.16 rec+1.65 clean=False comp=animal:+1.16 | L28 contrast_joint attention_output s1.5 T+0.26 R+0.68 rec+1.19 clean=False comp=animal:+0.68 |
| plant@L28 | R2=+0.52, cos=+0.98 | T-1.84 R+0.13 | 2 | L28 contrast_joint attention_output s1.5 T-0.85 R+0.00 rec+0.54 clean=True comp=tool:-0.20 | L28 support attention_output s1.5 T+0.53 R+2.06 rec+1.29 clean=False comp=tool:+2.06 | L28 contrast_joint attention_output s1.5 T+0.26 R+1.60 rec+1.14 clean=False comp=tool:+1.60 |
| time@L28 | R2=+0.60, cos=+0.99 | T-2.07 R+0.75 | 0 | NONE | L28 support mlp_input s1.5 T-0.90 R+0.71 rec+0.56 clean=False comp=number:+0.71 | L28 contrast_joint mlp_input s1.5 T-1.26 R+0.35 rec+0.39 clean=False comp=number:+0.35 |
| container@L28 | R2=+0.60, cos=+0.99 | T-1.78 R+0.00 | 3 | L28 support input_answer s1.0 T-0.09 R+0.13 rec+0.95 clean=True comp=machine:+0.13 | L28 support mlp_input s1.5 T+3.78 R+2.59 rec+3.12 clean=False comp=communication:+2.59 | L28 contrast_joint mlp_input s1.5 T+2.14 R+2.22 rec+2.20 clean=False comp=number:+2.22 |
| number@L27 | R2=+0.45, cos=+1.00 | T-1.39 R+1.40 | 0 | NONE | L27 support input_answer s0.25 T-1.41 R+0.27 rec-0.01 clean=False comp=furniture:+0.27 | L27 contrast_joint input_answer s0.25 T-1.41 R+0.27 rec-0.01 clean=False comp=furniture:+0.27 |
| plant@L27 | R2=+0.67, cos=+0.99 | T-1.61 R+0.70 | 0 | NONE | L27 support attention_output s0.25 T-1.74 R+0.09 rec-0.08 clean=False comp=clothing:+0.09 | L27 contrast_joint mlp_input s1.5 T-4.21 R+0.97 rec-1.62 clean=False comp=shape:+0.97 |
| time@L27 | R2=+0.52, cos=+0.99 | T-2.36 R+0.71 | 1 | L27 support mlp_input s0.5 T-0.18 R+0.19 rec+0.92 clean=True comp=event:+0.19 | L27 support mlp_input s1.5 T+1.63 R+2.06 rec+1.69 clean=False comp=event:+2.06 | L27 contrast_joint mlp_input s1.5 T+2.05 R+2.51 rec+1.87 clean=False comp=event:+2.51 |
| container@L27 | R2=+0.64, cos=+1.00 | T-1.93 R+0.00 | 0 | NONE | L27 support input_answer s0.5 T-1.71 R+0.00 rec+0.12 clean=False comp=clothing:-1.21 | L27 contrast_joint mlp_input s0.5 T-3.58 R+1.60 rec-0.85 clean=False comp=clothing:+1.60 |
| number@L26 | R2=+0.57, cos=+0.99 | T-1.16 R+0.88 | 0 | NONE | L26 support attention_output s0.5 T-0.70 R+0.79 rec+0.40 clean=False comp=furniture:+0.79 | L26 contrast_joint attention_output s0.5 T-0.78 R+0.59 rec+0.33 clean=False comp=furniture:+0.59 |
| plant@L26 | R2=+0.73, cos=+0.99 | T-1.45 R+0.12 | 0 | NONE | L26 support input_answer s0.75 T-0.82 R+1.13 rec+0.43 clean=False comp=clothing:+1.13 | L26 contrast_joint attention_output s0.75 T-1.01 R+0.93 rec+0.31 clean=False comp=clothing:+0.93 |
| time@L26 | R2=+0.61, cos=+0.99 | T-2.36 R+0.32 | 0 | NONE | L26 support attention_output s1.25 T-0.88 R+2.47 rec+0.63 clean=False comp=furniture:+2.47 | L26 contrast_joint attention_output s1.25 T-0.91 R+2.59 rec+0.61 clean=False comp=furniture:+2.59 |
| container@L26 | R2=+0.65, cos=+0.99 | T-2.11 R+0.00 | 0 | NONE | L26 support attention_output s0.5 T-1.93 R+0.00 rec+0.09 clean=False comp=clothing:-0.04 | L26 contrast_joint attention_output s0.75 T-1.93 R+0.07 rec+0.09 clean=False comp=clothing:+0.07 |

