# Phase 143 Cross-model Time Interface Empirical Suppressor Summary

## qwen3

True last layer: L36; train/test: 10/20; offsets: [0, -1]; sites: ['input_answer', 'mlp_input']; modes: ['support', 'naive_joint', 'empirical_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| container@L36 | R2=+0.28, cos=+0.99 | T-0.43 R+0.57 | 0 | NONE | L36 support mlp_input s0.5 T+7.22 R+11.91 rec+17.68 clean=False comp=number:+11.91 | L36 empirical_joint input_answer s0.25 T+5.85 R+10.94 rec+14.53 clean=False comp=sound:+10.94 | L36 empirical_joint input_answer s0.25 T+5.85 R+10.94 rec+14.53 clean=False comp=sound:+10.94 |
| time@L36 | R2=+0.43, cos=+0.99 | T-0.04 R+0.92 | 0 | NONE | L36 support mlp_input s0.5 T+9.97 R+12.34 rec+240.22 clean=False comp=sound:+12.34 | L36 naive_joint input_answer s0.25 T+12.40 R+12.81 rec+298.55 clean=False comp=number:+12.81 | L36 empirical_joint input_answer s0.25 T+11.11 R+12.79 rec+267.53 clean=False comp=animal:+12.79 |
| container@L35 | R2=+0.37, cos=+0.99 | T-1.43 R+0.30 | 0 | NONE | L35 support input_answer s0.5 T-1.03 R+0.11 rec+0.28 clean=False comp=clothing:+0.11 | L35 empirical_joint input_answer s0.5 T-0.77 R+0.14 rec+0.46 clean=False comp=food:+0.14 | L35 empirical_joint input_answer s0.5 T-0.77 R+0.14 rec+0.46 clean=False comp=food:+0.14 |
| time@L35 | R2=+0.39, cos=+0.98 | T-3.05 R+0.00 | 0 | NONE | L35 support mlp_input s0.5 T-2.24 R+1.25 rec+0.27 clean=False comp=clothing:+1.25 | L35 naive_joint mlp_input s0.5 T-1.60 R+1.79 rec+0.47 clean=False comp=furniture:+1.79 | L35 empirical_joint mlp_input s0.5 T-1.60 R+1.79 rec+0.47 clean=False comp=furniture:+1.79 |

## glm4

True last layer: L40; train/test: 10/20; offsets: [0, -1]; sites: ['input_answer', 'mlp_input']; modes: ['support', 'naive_joint', 'empirical_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| container@L40 | R2=+0.55, cos=+0.86 | T-0.18 R+0.29 | 0 | NONE | L40 support mlp_input s0.25 T-0.30 R+0.26 rec-0.65 clean=False comp=building:+0.26 | L40 empirical_joint input_answer s0.5 T+2.16 R+3.13 rec+12.70 clean=False comp=action:+3.13 | L40 empirical_joint input_answer s0.5 T+2.16 R+3.13 rec+12.70 clean=False comp=action:+3.13 |
| time@L40 | R2=+0.52, cos=+0.93 | T+0.16 R+0.44 | 0 | NONE | L40 support mlp_input s0.25 T+0.02 R+0.48 rec-0.85 clean=False comp=fruit:+0.48 | L40 empirical_joint input_answer s0.35 T+2.93 R+2.94 rec+17.33 clean=False comp=event:+2.94 | L40 empirical_joint input_answer s0.35 T+2.93 R+2.94 rec+17.33 clean=False comp=event:+2.94 |
| container@L39 | R2=+0.61, cos=+0.99 | T+0.80 R+1.06 | 0 | NONE | L39 support mlp_input s0.5 T+1.18 R+1.11 rec+0.47 clean=False comp=fruit:+1.11 | L39 empirical_joint input_answer s0.5 T+2.46 R+3.92 rec+2.07 clean=False comp=time:+3.92 | L39 empirical_joint input_answer s0.5 T+2.46 R+3.92 rec+2.07 clean=False comp=time:+3.92 |
| time@L39 | R2=+0.35, cos=+0.98 | T+0.68 R+0.88 | 0 | NONE | L39 support mlp_input s0.25 T+0.26 R+1.05 rec-0.61 clean=False comp=property:+1.05 | L39 naive_joint input_answer s0.5 T+4.23 R+4.15 rec+5.24 clean=False comp=action:+4.15 | L39 empirical_joint input_answer s0.35 T+2.98 R+3.31 rec+3.40 clean=False comp=action:+3.31 |

## deepseek7b

True last layer: L28; train/test: 20/40; offsets: [0, -1, -2]; sites: ['input_answer', 'attention_output', 'mlp_input']; modes: ['support', 'naive_joint', 'empirical_joint']

| category@layer | transfer | remove | clean count | best clean | best support | best joint | best empirical |
|---|---|---|---|---|---|---|---|
| number@L28 | R2=+0.49, cos=+0.99 | T-1.34 R+0.15 | 4 | L28 support attention_output s0.3 T-0.43 R+0.12 rec+0.68 clean=True comp=animal:+0.12 | L28 support attention_output s0.5 T-0.05 R+0.61 rec+0.97 clean=False comp=animal:+0.61 | L28 naive_joint attention_output s0.2 T-1.63 R+0.00 rec-0.21 clean=False comp=clothing:-0.13 | L28 empirical_joint attention_output s0.3 T-1.65 R+0.00 rec-0.23 clean=False comp=clothing:-0.43 |
| container@L28 | R2=+0.60, cos=+0.99 | T-1.78 R+0.00 | 0 | NONE | L28 support mlp_input s0.5 T+3.19 R+1.90 rec+2.79 clean=False comp=communication:+1.90 | L28 empirical_joint mlp_input s0.45 T-1.32 R+0.00 rec+0.26 clean=False comp=sound:-0.99 | L28 empirical_joint mlp_input s0.45 T-1.32 R+0.00 rec+0.26 clean=False comp=sound:-0.99 |
| plant@L28 | R2=+0.52, cos=+0.98 | T-1.84 R+0.13 | 2 | L28 support attention_output s0.35 T-0.69 R+0.11 rec+0.62 clean=True comp=tool:+0.11 | L28 support attention_output s0.5 T-0.37 R+0.62 rec+0.80 clean=False comp=tool:+0.62 | L28 naive_joint attention_output s0.2 T-2.54 R+0.09 rec-0.38 clean=False comp=clothing:+0.09 | L28 empirical_joint attention_output s0.2 T-2.54 R+0.09 rec-0.38 clean=False comp=clothing:+0.09 |
| time@L28 | R2=+0.60, cos=+0.99 | T-2.07 R+0.75 | 1 | L28 support mlp_input s0.2 T-0.98 R+0.06 rec+0.53 clean=True comp=building:+0.06 | L28 support mlp_input s0.2 T-0.98 R+0.06 rec+0.53 clean=True comp=building:+0.06 | L28 empirical_joint input_answer s0.3 T-2.04 R+0.00 rec+0.01 clean=False comp=clothing:-0.01 | L28 empirical_joint input_answer s0.3 T-2.04 R+0.00 rec+0.01 clean=False comp=clothing:-0.01 |
| number@L27 | R2=+0.45, cos=+1.00 | T-1.39 R+1.40 | 0 | NONE | L27 support attention_output s0.2 T-1.24 R+0.63 rec+0.11 clean=False comp=furniture:+0.63 | L27 naive_joint attention_output s0.2 T-1.17 R+1.59 rec+0.16 clean=False comp=furniture:+1.59 | L27 empirical_joint attention_output s0.2 T-1.27 R+1.53 rec+0.08 clean=False comp=furniture:+1.53 |
| container@L27 | R2=+0.64, cos=+1.00 | T-1.93 R+0.00 | 0 | NONE | L27 support input_answer s0.4 T-1.65 R+0.00 rec+0.15 clean=False comp=clothing:-0.86 | L27 naive_joint mlp_input s0.5 T-0.98 R+0.00 rec+0.49 clean=False comp=weather:-2.37 | L27 empirical_joint attention_output s0.25 T-1.80 R+0.27 rec+0.07 clean=False comp=clothing:+0.27 |
| plant@L27 | R2=+0.67, cos=+0.99 | T-1.61 R+0.70 | 0 | NONE | L27 support input_answer s0.2 T-1.64 R+0.28 rec-0.02 clean=False comp=clothing:+0.28 | L27 empirical_joint attention_output s0.45 T-1.14 R+1.31 rec+0.29 clean=False comp=clothing:+1.31 | L27 empirical_joint attention_output s0.45 T-1.14 R+1.31 rec+0.29 clean=False comp=clothing:+1.31 |
| time@L27 | R2=+0.52, cos=+0.99 | T-2.36 R+0.71 | 4 | L27 support mlp_input s0.5 T-0.18 R+0.19 rec+0.92 clean=True comp=event:+0.19 | L27 support mlp_input s0.5 T-0.18 R+0.19 rec+0.92 clean=True comp=event:+0.19 | L27 empirical_joint attention_output s0.5 T-1.94 R+1.41 rec+0.18 clean=False comp=furniture:+1.41 | L27 empirical_joint attention_output s0.5 T-1.94 R+1.41 rec+0.18 clean=False comp=furniture:+1.41 |
| number@L26 | R2=+0.57, cos=+0.99 | T-1.16 R+0.88 | 0 | NONE | L26 support attention_output s0.4 T-0.62 R+0.80 rec+0.47 clean=False comp=furniture:+0.80 | L26 naive_joint attention_output s0.35 T-0.89 R+0.95 rec+0.24 clean=False comp=furniture:+0.95 | L26 empirical_joint attention_output s0.35 T-1.05 R+0.79 rec+0.10 clean=False comp=furniture:+0.79 |
| container@L26 | R2=+0.65, cos=+0.99 | T-2.11 R+0.00 | 0 | NONE | L26 support attention_output s0.35 T-1.87 R+0.00 rec+0.11 clean=False comp=clothing:-0.03 | L26 empirical_joint mlp_input s0.25 T-1.58 R+0.46 rec+0.25 clean=False comp=furniture:+0.46 | L26 empirical_joint mlp_input s0.25 T-1.58 R+0.46 rec+0.25 clean=False comp=furniture:+0.46 |
| plant@L26 | R2=+0.73, cos=+0.99 | T-1.45 R+0.12 | 0 | NONE | L26 support input_answer s0.5 T-0.85 R+1.00 rec+0.42 clean=False comp=clothing:+1.00 | L26 empirical_joint attention_output s0.5 T-0.81 R+1.11 rec+0.44 clean=False comp=clothing:+1.11 | L26 empirical_joint attention_output s0.5 T-0.81 R+1.11 rec+0.44 clean=False comp=clothing:+1.11 |
| time@L26 | R2=+0.61, cos=+0.99 | T-2.36 R+0.32 | 0 | NONE | L26 support attention_output s0.45 T-1.19 R+1.65 rec+0.50 clean=False comp=furniture:+1.65 | L26 empirical_joint attention_output s0.5 T-1.83 R+1.15 rec+0.23 clean=False comp=furniture:+1.15 | L26 empirical_joint attention_output s0.5 T-1.83 R+1.15 rec+0.23 clean=False comp=furniture:+1.15 |

