# Phase59 Temporal Order Token Path Summary

## qwen3

accuracy=0.8021, mean_abs_margin=0.9961, n_cases=96

| rank | path | mean_signed_projection | std | n |
|---:|---|---:|---:|---:|
| 1 | L29:resid_out:last | 3.1995 | 2.8349 | 96 |
| 2 | L29:attn_out:last | 2.9250 | 4.3599 | 96 |
| 3 | L27:resid_out:last | 0.4705 | 1.1391 | 96 |
| 4 | L27:resid_in:last | 0.4697 | 1.6542 | 96 |
| 5 | L29:resid_in:last | 0.4233 | 3.3218 | 96 |
| 6 | L25:resid_in:last | 0.4061 | 0.6682 | 96 |
| 7 | L25:resid_out:last | 0.3872 | 2.0695 | 96 |
| 8 | L23:resid_out:last | 0.1016 | 0.3945 | 96 |
| 9 | L25:mlp_out:last | 0.1004 | 0.8739 | 96 |
| 10 | L23:attn_out:last | 0.0892 | 0.2244 | 96 |
| 11 | L23:resid_in:last | 0.0393 | 1.2320 | 96 |
| 12 | L21:attn_out:last | 0.0362 | 0.1372 | 96 |

## glm4

accuracy=0.8021, mean_abs_margin=0.7319, n_cases=96

| rank | path | mean_signed_projection | std | n |
|---:|---|---:|---:|---:|
| 1 | L38:resid_out:last | 0.9618 | 2.9675 | 96 |
| 2 | L38:resid_in:last | 0.5610 | 2.6258 | 96 |
| 3 | L36:resid_out:last | 0.5604 | 1.1168 | 96 |
| 4 | L36:resid_in:last | 0.5166 | 1.2627 | 96 |
| 5 | L33:resid_out:last | 0.4121 | 0.8782 | 96 |
| 6 | L38:mlp_out:last | 0.3899 | 1.3335 | 96 |
| 7 | L33:resid_in:last | 0.3628 | 0.8542 | 96 |
| 8 | L33:attn_out:last | 0.0603 | 0.1502 | 96 |
| 9 | L36:mlp_out:last | 0.0394 | 0.4410 | 96 |
| 10 | L30:resid_in:last | 0.0201 | 0.9761 | 96 |
| 11 | L30:resid_out:last | 0.0180 | 1.2575 | 96 |
| 12 | L38:attn_out:last | 0.0109 | 0.9103 | 96 |

## deepseek7b

accuracy=0.6562, mean_abs_margin=0.9391, n_cases=96

| rank | path | mean_signed_projection | std | n |
|---:|---|---:|---:|---:|
| 1 | L24:resid_out:last | 0.4095 | 8.2238 | 96 |
| 2 | L24:attn_out:last | 0.2907 | 1.5209 | 96 |
| 3 | L21:resid_in:last | 0.1162 | 1.9319 | 96 |
| 4 | L24:mlp_out:last | 0.1058 | 4.5707 | 96 |
| 5 | L21:resid_out:last | 0.1027 | 1.6293 | 96 |
| 6 | L19:attn_out:last | 0.0617 | 0.8796 | 96 |
| 7 | L19:resid_out:last | 0.0519 | 0.8557 | 96 |
| 8 | L21:attn_out:last | 0.0371 | 1.8245 | 96 |
| 9 | L23:resid_out:last | 0.0164 | 4.2785 | 96 |
| 10 | L24:resid_in:last | 0.0164 | 4.2785 | 96 |
| 11 | L23:mlp_out:last | 0.0110 | 5.1419 | 96 |
| 12 | L19:resid_in:last | 0.0066 | 1.1371 | 96 |
