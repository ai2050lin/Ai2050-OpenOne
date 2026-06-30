# Phase 786 Head Projection and MLP Activation-Channel Source Audit (smoke)

- Status: `complete`
- Attention evidence: o_proj input is split by head and projected to selected D+/D- output dimensions.
- MLP evidence: down_proj input activation channels are projected to selected D+/D- output dimensions.
- Strict interpretation: source attribution, not causal ablation yet.

## Concentration Summary

| model | source | subspace | budget | n | cases | head top1 | head top3 | head top8 | mlp top1 | mlp top8 | mlp top32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attention_head_o_proj` | `negative` | `negative_1024` | 32 | 1 | 0.129 | 0.349 | 0.658 | null | null | null |
| qwen3 | `attention_head_o_proj` | `positive` | `positive_1024` | 32 | 1 | 0.123 | 0.346 | 0.657 | null | null | null |
| qwen3 | `mlp_down_input_channel` | `negative` | `negative_1024` | 40 | 1 | null | null | null | 0.025 | 0.103 | 0.202 |
| qwen3 | `mlp_down_input_channel` | `positive` | `positive_1024` | 40 | 1 | null | null | null | 0.026 | 0.112 | 0.214 |
| glm4 | `mlp_down_input_channel` | `negative` | `negative_1024` | 48 | 1 | null | null | null | 0.017 | 0.053 | 0.111 |
| glm4 | `mlp_down_input_channel` | `positive` | `positive_1024` | 40 | 1 | null | null | null | 0.011 | 0.055 | 0.119 |
| deepseek7b | `attention_head_o_proj` | `negative` | `negative_1024` | 28 | 1 | 0.196 | 0.462 | 0.689 | null | null | null |
| deepseek7b | `attention_head_o_proj` | `positive` | `positive_1024` | 28 | 1 | 0.356 | 0.564 | 0.756 | null | null | null |
| deepseek7b | `mlp_down_input_channel` | `negative` | `negative_1024` | 40 | 1 | null | null | null | 0.015 | 0.062 | 0.135 |
| deepseek7b | `mlp_down_input_channel` | `positive` | `positive_1024` | 40 | 1 | null | null | null | 0.016 | 0.061 | 0.128 |

## Top Attention Heads

| model | route | component | subspace | budget | head | cases | signed | abs | positive rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 25 | 1 | 2.409 | 3.214 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 0 | 1 | 2.396 | 3.104 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 25 | 1 | -1.832 | 2.758 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 0 | 1 | -2.350 | 2.749 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 26 | 1 | 1.922 | 2.730 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 26 | 1 | -0.511 | 1.982 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 24 | 1 | 0.752 | 1.898 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 27 | 1 | 0.339 | 1.857 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 15 | 1 | 0.675 | 1.671 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 27 | 1 | -0.360 | 1.541 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 5 | 1 | 0.532 | 1.442 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 24 | 1 | -0.374 | 1.437 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 2 | 1 | 0.623 | 1.277 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 5 | 1 | -0.401 | 1.245 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 15 | 1 | -0.832 | 1.225 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 19 | 1 | 0.230 | 1.205 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 2 | 1 | -0.500 | 1.165 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `negative` | `negative_1024` | 19 | 1 | -0.182 | 0.933 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 23 | 1 | 0.199 | 0.858 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `positive` | `positive_1024` | 1 | 1 | 0.338 | 0.789 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 27 | 1 | 0.319 | 0.319 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 0 | 1 | -0.127 | 0.150 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 27 | 1 | -0.129 | 0.129 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 1 | 1 | 0.128 | 0.128 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 1 | 1 | -0.061 | 0.075 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 0 | 1 | 0.035 | 0.058 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 4 | 1 | 0.033 | 0.056 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 4 | 1 | -0.018 | 0.050 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 20 | 1 | -0.021 | 0.039 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 3 | 1 | 0.030 | 0.036 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 3 | 1 | -0.029 | 0.031 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 26 | 1 | 0.026 | 0.031 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 22 | 1 | -0.011 | 0.028 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 6 | 1 | 0.013 | 0.028 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 26 | 1 | -0.025 | 0.025 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 6 | 1 | -0.003 | 0.023 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 11 | 1 | -0.016 | 0.022 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 10 | 1 | 0.009 | 0.022 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `negative` | `negative_1024` | 16 | 1 | 0.012 | 0.022 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `positive` | `positive_1024` | 13 | 1 | 0.015 | 0.022 | 1.000 |

## Top MLP Activation Channels

| model | route | component | subspace | budget | channel | cases | signed | abs | positive rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 1439 | 1 | -1.418 | 1.418 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 1439 | 1 | 1.414 | 1.414 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 1147 | 1 | 1.205 | 1.205 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 935 | 1 | 1.026 | 1.026 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 2572 | 1 | 0.926 | 0.926 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 2572 | 1 | -0.842 | 0.842 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 935 | 1 | -0.751 | 0.751 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 298 | 1 | 0.730 | 0.730 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 991 | 1 | 0.544 | 0.544 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `positive` | `positive_1024` | 1730 | 1 | 0.519 | 0.519 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 516 | 1 | 0.495 | 0.495 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L33` | `positive` | `positive_1024` | 1275 | 1 | 0.481 | 0.481 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 548 | 1 | 0.472 | 0.472 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 516 | 1 | -0.453 | 0.453 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L33` | `positive` | `positive_1024` | 1166 | 1 | 0.431 | 0.431 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 123 | 1 | -0.415 | 0.415 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 186 | 1 | -0.403 | 0.403 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `negative` | `negative_1024` | 1730 | 1 | -0.396 | 0.396 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L33` | `negative` | `negative_1024` | 219 | 1 | -0.355 | 0.355 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `positive` | `positive_1024` | 3372 | 1 | 0.333 | 0.333 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 2801 | 1 | -0.347 | 0.347 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 2801 | 1 | 0.334 | 0.334 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 8235 | 1 | -0.292 | 0.292 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 8235 | 1 | 0.270 | 0.270 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 10916 | 1 | 0.210 | 0.210 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 9787 | 1 | 0.189 | 0.189 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `positive` | `positive_1024` | 462 | 1 | 0.142 | 0.142 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 10916 | 1 | -0.140 | 0.140 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 1237 | 1 | -0.136 | 0.136 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `negative` | `negative_1024` | 462 | 1 | -0.128 | 0.128 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 7525 | 1 | 0.128 | 0.128 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 10692 | 1 | -0.126 | 0.126 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 6393 | 1 | 0.123 | 0.123 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 7525 | 1 | -0.119 | 0.119 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 10692 | 1 | 0.118 | 0.118 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 1755 | 1 | 0.118 | 0.118 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `positive` | `positive_1024` | 4526 | 1 | 0.114 | 0.114 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `negative` | `negative_1024` | 10751 | 1 | -0.111 | 0.111 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 9787 | 1 | -0.111 | 0.111 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `positive` | `positive_1024` | 10751 | 1 | 0.107 | 0.107 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 2295 | 1 | -2.507 | 2.507 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 12212 | 1 | -1.879 | 1.879 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `positive` | `positive_1024` | 9289 | 1 | 1.693 | 1.693 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `negative` | `negative_1024` | 9289 | 1 | -1.667 | 1.667 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 1629 | 1 | -1.391 | 1.391 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 13660 | 1 | 1.391 | 1.391 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 1489 | 1 | -1.338 | 1.338 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 12212 | 1 | 1.331 | 1.331 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 13660 | 1 | -1.323 | 1.323 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 547 | 1 | -1.290 | 1.290 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 16230 | 1 | 1.198 | 1.198 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 17463 | 1 | 1.129 | 1.129 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 12909 | 1 | -1.084 | 1.084 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 1489 | 1 | 1.059 | 1.059 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 2644 | 1 | -0.976 | 0.976 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 16339 | 1 | 0.966 | 0.966 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `positive` | `positive_1024` | 9394 | 1 | 0.949 | 0.949 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 5848 | 1 | 0.944 | 0.944 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 6926 | 1 | 0.926 | 0.926 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `negative` | `negative_1024` | 13570 | 1 | -0.710 | 0.710 | 0.000 |

## Interpretation Boundary

- High concentration supports a route from signed residual subspace to architectural source units.
- It remains attribution evidence until head/channel causal patch or ablation confirms necessity.
