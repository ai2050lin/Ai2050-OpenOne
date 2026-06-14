# Phase 126 Residual Gap Decomposition: qwen3

Generated: 2026-06-14 19:33:21
Monitor layer: L35; patch layers: [32, 33, 34, 35]

| category | layer input | attention output | MLP output | layer output | attn+MLP |
|---|---|---|---|---|---|
| number | L34 layer_input T-0.13 R+0.45 A+10.97 | L35 attention_output T-0.06 R+0.00 A+0.00 | L34 mlp_output T-0.00 R+0.09 A+0.27 | L33 layer_output T-0.13 R+0.45 A+10.97 | L35 attention_plus_mlp T-0.02 R+0.06 A+0.00 |
| container | L32 layer_input T-0.09 R+0.41 A-12.29 | L35 attention_output T-0.01 R+0.01 A+0.00 | L32 mlp_output T-0.07 R+0.06 A-0.25 | L32 layer_output T-0.04 R+0.26 A-7.09 | L32 attention_plus_mlp T-0.06 R+0.06 A-0.19 |
| plant | L33 layer_input T-0.22 R+0.29 A+12.42 | L33 attention_output T+0.00 R+0.03 A-0.33 | L34 mlp_output T+0.04 R+0.06 A+0.17 | L32 layer_output T-0.22 R+0.29 A+12.42 | L34 attention_plus_mlp T+0.04 R+0.06 A+0.53 |
