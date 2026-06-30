# Phase 786 Head Projection and MLP Activation-Channel Source Audit (main)

- Status: `complete`
- Attention evidence: o_proj input is split by head and projected to selected D+/D- output dimensions.
- MLP evidence: down_proj input activation channels are projected to selected D+/D- output dimensions.
- Strict interpretation: source attribution, not causal ablation yet.

## Concentration Summary

| model | source | subspace | budget | n | cases | head top1 | head top3 | head top8 | mlp top1 | mlp top8 | mlp top32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `attention_head_o_proj` | `all_negative` | `all_negative` | 960 | 6 | 0.115 | 0.283 | 0.580 | null | null | null |
| qwen3 | `attention_head_o_proj` | `all_positive` | `all_positive` | 960 | 6 | 0.128 | 0.294 | 0.587 | null | null | null |
| qwen3 | `attention_head_o_proj` | `negative` | `negative_1024` | 960 | 6 | 0.128 | 0.306 | 0.595 | null | null | null |
| qwen3 | `attention_head_o_proj` | `positive` | `positive_1024` | 960 | 6 | 0.152 | 0.324 | 0.609 | null | null | null |
| qwen3 | `mlp_down_input_channel` | `all_negative` | `all_negative` | 672 | 6 | null | null | null | 0.029 | 0.116 | 0.228 |
| qwen3 | `mlp_down_input_channel` | `all_positive` | `all_positive` | 672 | 6 | null | null | null | 0.032 | 0.128 | 0.243 |
| qwen3 | `mlp_down_input_channel` | `negative` | `negative_1024` | 672 | 6 | null | null | null | 0.024 | 0.098 | 0.199 |
| qwen3 | `mlp_down_input_channel` | `positive` | `positive_1024` | 672 | 6 | null | null | null | 0.025 | 0.107 | 0.210 |
| glm4 | `attention_head_o_proj` | `all_negative` | `all_negative` | 768 | 6 | 0.125 | 0.304 | 0.584 | null | null | null |
| glm4 | `attention_head_o_proj` | `all_positive` | `all_positive` | 768 | 6 | 0.132 | 0.310 | 0.587 | null | null | null |
| glm4 | `attention_head_o_proj` | `negative` | `negative_1024` | 704 | 6 | 0.177 | 0.377 | 0.650 | null | null | null |
| glm4 | `attention_head_o_proj` | `positive` | `positive_1024` | 704 | 6 | 0.197 | 0.396 | 0.656 | null | null | null |
| glm4 | `mlp_down_input_channel` | `all_negative` | `all_negative` | 768 | 6 | null | null | null | 0.040 | 0.104 | 0.179 |
| glm4 | `mlp_down_input_channel` | `all_positive` | `all_positive` | 768 | 6 | null | null | null | 0.043 | 0.107 | 0.184 |
| glm4 | `mlp_down_input_channel` | `negative` | `negative_1024` | 704 | 6 | null | null | null | 0.022 | 0.062 | 0.122 |
| glm4 | `mlp_down_input_channel` | `positive` | `positive_1024` | 736 | 6 | null | null | null | 0.023 | 0.066 | 0.127 |
| deepseek7b | `attention_head_o_proj` | `all_negative` | `all_negative` | 1008 | 6 | 0.152 | 0.341 | 0.584 | null | null | null |
| deepseek7b | `attention_head_o_proj` | `all_positive` | `all_positive` | 1008 | 6 | 0.155 | 0.342 | 0.585 | null | null | null |
| deepseek7b | `attention_head_o_proj` | `negative` | `negative_1024` | 1008 | 6 | 0.188 | 0.397 | 0.631 | null | null | null |
| deepseek7b | `attention_head_o_proj` | `positive` | `positive_1024` | 1008 | 6 | 0.209 | 0.411 | 0.639 | null | null | null |
| deepseek7b | `mlp_down_input_channel` | `all_negative` | `all_negative` | 576 | 6 | null | null | null | 0.020 | 0.077 | 0.163 |
| deepseek7b | `mlp_down_input_channel` | `all_positive` | `all_positive` | 576 | 6 | null | null | null | 0.019 | 0.078 | 0.163 |
| deepseek7b | `mlp_down_input_channel` | `negative` | `negative_1024` | 576 | 6 | null | null | null | 0.017 | 0.065 | 0.141 |
| deepseek7b | `mlp_down_input_channel` | `positive` | `positive_1024` | 576 | 6 | null | null | null | 0.015 | 0.065 | 0.140 |

## Top Attention Heads

| model | route | component | subspace | budget | head | cases | signed | abs | positive rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k6` | `attn:L31` | `all_positive` | `all_positive` | 28 | 6 | 10.392 | 12.891 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 5 | 6 | 8.263 | 12.797 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 26 | 6 | 5.843 | 10.069 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_negative` | `all_negative` | 5 | 6 | -0.205 | 9.278 | 0.667 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 25 | 6 | 4.103 | 8.701 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 0 | 6 | 3.224 | 8.141 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 25 | 6 | 2.836 | 8.094 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_negative` | `all_negative` | 1 | 6 | -3.487 | 8.091 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 1 | 6 | 1.516 | 7.863 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 27 | 6 | 1.204 | 7.707 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 19 | 6 | 2.348 | 7.701 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 28 | 6 | 0.616 | 7.567 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 29 | 6 | 0.296 | 7.524 | 0.667 |
| qwen3 | `lowercase_short_value:route_k6` | `attn:L35` | `all_negative` | `all_negative` | 26 | 6 | -1.130 | 7.469 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 24 | 6 | 2.330 | 7.437 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L35` | `all_positive` | `all_positive` | 0 | 6 | 3.005 | 7.363 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_positive` | `all_positive` | 15 | 6 | 2.205 | 7.016 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L31` | `all_negative` | `all_negative` | 28 | 6 | -2.441 | 7.013 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L32` | `all_positive` | `all_positive` | 31 | 6 | 3.224 | 6.969 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `attn:L34` | `all_negative` | `all_negative` | 19 | 6 | -2.033 | 6.969 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_positive` | `all_positive` | 9 | 6 | 1.705 | 2.393 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_negative` | `all_negative` | 9 | 6 | -0.500 | 1.570 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_negative` | `all_negative` | 8 | 6 | -0.708 | 1.173 | 0.667 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_positive` | `all_positive` | 24 | 6 | 0.283 | 1.088 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_negative` | `all_negative` | 22 | 6 | -0.399 | 1.002 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_negative` | `all_negative` | 28 | 6 | -0.512 | 0.975 | 0.167 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_negative` | `all_negative` | 24 | 6 | -0.301 | 0.962 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_positive` | `all_positive` | 8 | 6 | 0.406 | 0.952 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_positive` | `all_positive` | 22 | 6 | 0.395 | 0.951 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_positive` | `all_positive` | 7 | 6 | 0.433 | 0.918 | 0.833 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_positive` | `all_positive` | 4 | 6 | 0.368 | 0.868 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `all_negative` | `all_negative` | 26 | 6 | -0.322 | 0.859 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `all_negative` | `all_negative` | 18 | 6 | -0.423 | 0.855 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_positive` | `all_positive` | 28 | 6 | 0.325 | 0.843 | 0.833 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_negative` | `all_negative` | 4 | 6 | -0.318 | 0.839 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `all_positive` | `all_positive` | 18 | 6 | 0.365 | 0.823 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L29` | `all_positive` | `all_positive` | 26 | 6 | 0.292 | 0.818 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_negative` | `all_negative` | 26 | 6 | -0.218 | 0.818 | 0.333 |
| glm4 | `with_candidate_list:route_k6` | `attn:L32` | `all_positive` | `all_positive` | 26 | 6 | 0.222 | 0.749 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `attn:L33` | `all_positive` | `all_positive` | 31 | 6 | -0.039 | 0.693 | 0.167 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `all_positive` | `all_positive` | 14 | 6 | 18.367 | 27.720 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `all_negative` | `all_negative` | 14 | 6 | -8.985 | 21.127 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `all_positive` | `all_positive` | 23 | 6 | 8.538 | 17.250 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `all_negative` | `all_negative` | 23 | 6 | -8.518 | 16.616 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `all_negative` | `all_negative` | 19 | 6 | -10.284 | 16.541 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `all_positive` | `all_positive` | 5 | 6 | 7.348 | 16.346 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L25` | `all_positive` | `all_positive` | 25 | 6 | 8.981 | 16.138 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `all_positive` | `all_positive` | 19 | 6 | 9.218 | 15.608 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `all_negative` | `all_negative` | 5 | 6 | -7.781 | 15.581 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L25` | `all_negative` | `all_negative` | 25 | 6 | -8.455 | 15.286 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `all_negative` | `all_negative` | 11 | 6 | -8.155 | 14.585 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L23` | `all_positive` | `all_positive` | 11 | 6 | 8.467 | 14.149 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L25` | `all_positive` | `all_positive` | 9 | 6 | 6.752 | 13.758 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L25` | `all_negative` | `all_negative` | 9 | 6 | -6.586 | 13.181 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `all_positive` | `all_positive` | 16 | 6 | 1.383 | 12.745 | 0.333 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `all_negative` | `all_negative` | 16 | 6 | -5.372 | 11.931 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L26` | `all_positive` | `all_positive` | 15 | 6 | 3.250 | 11.620 | 0.833 |
| deepseek7b | `with_candidate_list:route_k6` | `attn:L27` | `all_positive` | `all_positive` | 19 | 6 | 5.198 | 11.125 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `all_positive` | `all_positive` | 27 | 6 | 7.651 | 10.602 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `attn:L19` | `all_negative` | `all_negative` | 27 | 6 | -7.567 | 10.282 | 0.000 |

## Top MLP Activation Channels

| model | route | component | subspace | budget | channel | cases | signed | abs | positive rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `all_negative` | `all_negative` | 8061 | 1 | -4.395 | 4.395 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `all_negative` | `all_negative` | 7125 | 1 | -4.144 | 4.144 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `all_positive` | `all_positive` | 352 | 2 | 3.653 | 3.653 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_positive` | `all_positive` | 352 | 2 | 3.445 | 3.445 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `all_positive` | `all_positive` | 2369 | 1 | 3.420 | 3.420 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_positive` | `all_positive` | 2572 | 4 | 3.411 | 3.411 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_negative` | `all_negative` | 4187 | 2 | -2.808 | 2.808 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `negative` | `negative_1024` | 8061 | 1 | -2.727 | 2.727 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L33` | `all_positive` | `all_positive` | 8825 | 1 | 2.601 | 2.601 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 2572 | 4 | 2.396 | 2.396 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_positive` | `all_positive` | 4187 | 2 | 2.313 | 2.313 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `all_positive` | `all_positive` | 2185 | 2 | 2.282 | 2.282 | 1.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `all_negative` | `all_negative` | 2369 | 1 | -2.236 | 2.236 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L34` | `negative` | `negative_1024` | 7125 | 1 | -2.090 | 2.090 | 0.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 352 | 2 | 1.946 | 1.946 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `negative` | `negative_1024` | 4187 | 2 | -1.896 | 1.896 | 0.000 |
| qwen3 | `with_candidate_list:route_k6` | `mlp:L35` | `positive` | `positive_1024` | 352 | 2 | 1.715 | 1.715 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L34` | `all_positive` | `all_positive` | 5541 | 1 | 1.699 | 1.699 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_positive` | `all_positive` | 1439 | 6 | 1.696 | 1.696 | 1.000 |
| qwen3 | `lowercase_short_value:route_k6` | `mlp:L35` | `all_negative` | `all_negative` | 2572 | 5 | -1.693 | 1.693 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_positive` | `all_positive` | 11316 | 2 | 1.226 | 1.226 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 11316 | 2 | 1.065 | 1.065 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_positive` | `all_positive` | 3652 | 1 | 1.061 | 1.061 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 3652 | 1 | 1.002 | 1.002 | 1.000 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `all_negative` | `all_negative` | 7605 | 1 | -0.919 | 0.919 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L34` | `all_positive` | `all_positive` | 7605 | 1 | 0.865 | 0.865 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_negative` | `all_negative` | 11316 | 2 | -0.847 | 0.847 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_positive` | `all_positive` | 8235 | 4 | 0.684 | 0.684 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_negative` | `all_negative` | 8235 | 4 | -0.626 | 0.626 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `positive` | `positive_1024` | 8235 | 4 | 0.475 | 0.475 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_negative` | `all_negative` | 2801 | 2 | -0.474 | 0.474 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 11316 | 3 | -0.460 | 0.460 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L34` | `all_negative` | `all_negative` | 10767 | 1 | -0.449 | 0.449 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_positive` | `all_positive` | 2801 | 2 | 0.448 | 0.448 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `negative` | `negative_1024` | 8235 | 4 | -0.436 | 0.436 | 0.000 |
| glm4 | `with_candidate_list:route_k6` | `mlp:L38` | `all_positive` | `all_positive` | 4191 | 2 | 0.434 | 0.434 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `all_negative` | `all_negative` | 1316 | 2 | -0.419 | 0.419 | 0.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L38` | `all_positive` | `all_positive` | 1316 | 2 | 0.409 | 0.409 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L34` | `all_positive` | `all_positive` | 10767 | 1 | 0.398 | 0.398 | 1.000 |
| glm4 | `lowercase_short_value:route_k6` | `mlp:L39` | `all_positive` | `all_positive` | 1 | 1 | 0.330 | 0.330 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_negative` | `all_negative` | 2295 | 3 | -4.456 | 4.456 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 13660 | 1 | 3.508 | 3.508 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_negative` | `all_negative` | 13660 | 1 | -3.397 | 3.397 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `all_negative` | `all_negative` | 17304 | 1 | -2.952 | 2.952 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `all_negative` | `all_negative` | 3132 | 1 | -2.920 | 2.920 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `all_negative` | `all_negative` | 2295 | 3 | -2.875 | 2.875 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 2295 | 3 | -2.863 | 2.863 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 3776 | 1 | 2.763 | 2.763 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 1109 | 1 | 2.755 | 2.755 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_negative` | `all_negative` | 12031 | 2 | -2.740 | 2.740 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L24` | `all_negative` | `all_negative` | 6121 | 1 | -2.654 | 2.654 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_negative` | `all_negative` | 12204 | 1 | -2.562 | 2.562 | 0.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 11633 | 1 | 2.547 | 2.547 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 16230 | 5 | 2.540 | 2.540 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `all_negative` | `all_negative` | 5145 | 2 | -2.537 | 2.537 | 0.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 3776 | 1 | 2.532 | 2.532 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `all_positive` | `all_positive` | 12031 | 2 | 2.519 | 2.519 | 1.000 |
| deepseek7b | `lowercase_short_value:route_k6` | `mlp:L26` | `all_positive` | `all_positive` | 5145 | 2 | 2.507 | 2.507 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `positive` | `positive_1024` | 13660 | 1 | 2.444 | 2.444 | 1.000 |
| deepseek7b | `with_candidate_list:route_k6` | `mlp:L27` | `negative` | `negative_1024` | 13660 | 1 | -2.428 | 2.428 | 0.000 |

## Interpretation Boundary

- High concentration supports a route from signed residual subspace to architectural source units.
- It remains attribution evidence until head/channel causal patch or ablation confirms necessity.
