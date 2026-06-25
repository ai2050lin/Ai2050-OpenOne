# Phase 632 Cross-Model Summary

目标：从 Phase631 的人工 readout direction 回溯自然写入器，审计每层/组件对 prefix margin 的自然差分贡献，并对 top writer 做 causal restore/remove/control。

## deepseek7b

- rows: 82 / raw_cases: 256 / target_seen: 82
- scan_layers: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
- downstream_layers: [22, 23, 24, 25, 26, 27]
- top_nodes: ['L26_layer_out', 'L27_layer_input', 'L27_layer_out', 'L25_layer_out', 'L26_layer_input', 'L24_layer_out']

### Natural Margin Writer Scan

| rank | node | mean_margin_delta | positive_rate | mean_cos | mean_delta_norm | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | L26_layer_out | 33.271 | 1.000 | 0.090 | 278.671 | 33.271 |
| 2 | L27_layer_input | 33.271 | 1.000 | 0.090 | 278.671 | 33.271 |
| 3 | L27_layer_out | 32.667 | 0.988 | 0.067 | 371.306 | 32.269 |
| 4 | L25_layer_out | 17.553 | 0.988 | 0.058 | 228.231 | 17.339 |
| 5 | L26_layer_input | 17.553 | 0.988 | 0.058 | 228.231 | 17.339 |
| 6 | L24_layer_out | 12.178 | 0.988 | 0.047 | 196.366 | 12.029 |
| 7 | L25_layer_input | 12.178 | 0.988 | 0.047 | 196.366 | 12.029 |
| 8 | L26_attn_out | 9.669 | 1.000 | 0.144 | 51.453 | 9.669 |
| 9 | L23_layer_out | 6.456 | 0.988 | 0.029 | 169.541 | 6.377 |
| 10 | L24_layer_input | 6.456 | 0.988 | 0.029 | 169.541 | 6.377 |
| 11 | L26_mlp_out | 6.106 | 0.963 | 0.040 | 115.220 | 5.882 |
| 12 | L25_attn_out | 3.830 | 1.000 | 0.064 | 46.253 | 3.830 |

### Causal Patch Audit

| node | mode | tok0 | exact | wrong_exact | mean_prefix_margin |
|---|---|---:|---:|---:|---:|
| __baseline__ | repair_prompt | 20/82 | 20/82 | 0/82 | -1.699 |
| __baseline__ | base | 0/82 | 0/82 | 0/82 | -6.356 |
| __baseline__ | semantic_cumulative | 0/82 | 0/82 | 0/82 | -6.356 |
| L26_layer_out | restore_semantic | 21/82 | 21/82 | 0/82 | -1.662 |
| L27_layer_input | restore_semantic | 21/82 | 21/82 | 0/82 | -1.662 |
| L25_layer_out | restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |
| L26_layer_input | restore_semantic | 21/82 | 21/82 | 0/82 | -1.755 |
| L27_layer_out | restore_semantic | 20/82 | 20/82 | 0/82 | -1.699 |
| L24_layer_out | restore_semantic | 19/82 | 19/82 | 0/82 | -1.936 |
| L26_layer_out | restore | 21/82 | 3/82 | 18/82 | -1.662 |
| L27_layer_input | restore | 21/82 | 3/82 | 18/82 | -1.662 |
| L25_layer_out | restore | 21/82 | 3/82 | 17/82 | -1.755 |
| L26_layer_input | restore | 21/82 | 3/82 | 17/82 | -1.755 |
| L27_layer_out | restore | 20/82 | 2/82 | 18/82 | -1.699 |
| L24_layer_out | restore | 19/82 | 2/82 | 16/82 | -1.936 |
| L24_layer_out | random_semantic | 1/82 | 1/82 | 0/82 | -6.060 |
| L24_layer_out | remove_from_repair | 1/82 | 1/82 | 0/82 | -6.101 |
| L26_layer_out | random_semantic | 0/82 | 0/82 | 0/82 | -6.182 |
| L26_layer_input | random_semantic | 0/82 | 0/82 | 0/82 | -6.208 |
| L27_layer_out | random_semantic | 0/82 | 0/82 | 0/82 | -6.234 |
| L27_layer_input | random_semantic | 0/82 | 0/82 | 0/82 | -6.262 |
| L25_layer_out | remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |
| L26_layer_input | remove_from_repair | 0/82 | 0/82 | 0/82 | -6.285 |
| L25_layer_out | random_semantic | 0/82 | 0/82 | 0/82 | -6.354 |
| L27_layer_out | remove_from_repair | 0/82 | 0/82 | 0/82 | -6.356 |
| L26_layer_out | remove_from_repair | 0/82 | 0/82 | 0/82 | -6.409 |
| L27_layer_input | remove_from_repair | 0/82 | 0/82 | 0/82 | -6.409 |

### Examples

- sample=0 node=__baseline__ mode=base tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\nTo solve'
- sample=0 node=__baseline__ mode=semantic_cumulative tok0=' ?\n\n' exact=False wrong=False margin=-5.812 text=' ?\n\n2\n'
- sample=0 node=L26_layer_out mode=restore_semantic tok0=' ?\n\n' exact=False wrong=False margin=-2.500 text=' ?\n\n2\n'
- sample=0 node=L26_layer_out mode=random_semantic tok0=' ?\n\n' exact=False wrong=False margin=-6.312 text=' ?\n\n2\n'
- sample=0 node=L26_layer_out mode=reverse_semantic tok0=' ?\n\n' exact=False wrong=False margin=-10.000 text=' ?\n\n2\n'

## glm4

- rows: 31 / raw_cases: 256 / target_seen: 31
- scan_layers: [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
- downstream_layers: [34, 35, 36, 37, 38, 39]
- top_nodes: ['L38_layer_out', 'L39_layer_input', 'L37_layer_out', 'L38_layer_input', 'L36_layer_out', 'L37_layer_input']

### Natural Margin Writer Scan

| rank | node | mean_margin_delta | positive_rate | mean_cos | mean_delta_norm | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | L38_layer_out | 3.842 | 1.000 | 0.085 | 49.438 | 3.842 |
| 2 | L39_layer_input | 3.842 | 1.000 | 0.085 | 49.438 | 3.842 |
| 3 | L37_layer_out | 3.068 | 1.000 | 0.117 | 28.465 | 3.068 |
| 4 | L38_layer_input | 3.068 | 1.000 | 0.117 | 28.465 | 3.068 |
| 5 | L36_layer_out | 2.809 | 1.000 | 0.118 | 25.959 | 2.809 |
| 6 | L37_layer_input | 2.809 | 1.000 | 0.118 | 25.959 | 2.809 |
| 7 | L39_layer_out | 2.662 | 0.968 | 0.046 | 64.040 | 2.576 |
| 8 | L35_layer_out | 2.507 | 0.968 | 0.112 | 24.328 | 2.426 |
| 9 | L36_layer_input | 2.507 | 0.968 | 0.112 | 24.328 | 2.426 |
| 10 | L34_layer_out | 2.343 | 0.968 | 0.114 | 22.393 | 2.267 |
| 11 | L35_layer_input | 2.343 | 0.968 | 0.114 | 22.393 | 2.267 |
| 12 | L33_layer_out | 2.078 | 0.968 | 0.107 | 21.197 | 2.011 |

### Causal Patch Audit

| node | mode | tok0 | exact | wrong_exact | mean_prefix_margin |
|---|---|---:|---:|---:|---:|
| __baseline__ | repair_prompt | 29/31 | 28/31 | 1/31 | 1.710 |
| __baseline__ | semantic_cumulative | 11/31 | 11/31 | 0/31 | -0.226 |
| __baseline__ | base | 11/31 | 2/31 | 9/31 | -0.226 |
| L38_layer_out | restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |
| L39_layer_input | restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |
| L37_layer_out | restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |
| L38_layer_input | restore_semantic | 29/31 | 29/31 | 0/31 | 1.712 |
| L36_layer_out | restore_semantic | 29/31 | 29/31 | 0/31 | 1.659 |
| L37_layer_input | restore_semantic | 29/31 | 29/31 | 0/31 | 1.659 |
| L39_layer_input | random_semantic | 14/31 | 14/31 | 0/31 | -0.282 |
| L36_layer_out | random_semantic | 12/31 | 12/31 | 0/31 | -0.258 |
| L37_layer_input | random_semantic | 12/31 | 12/31 | 0/31 | -0.260 |
| L38_layer_out | random_semantic | 11/31 | 11/31 | 0/31 | -0.252 |
| L38_layer_input | random_semantic | 11/31 | 11/31 | 0/31 | -0.288 |
| L36_layer_out | remove_from_repair | 10/31 | 10/31 | 0/31 | -0.202 |
| L37_layer_input | remove_from_repair | 10/31 | 10/31 | 0/31 | -0.202 |
| L38_layer_out | remove_from_repair | 10/31 | 10/31 | 0/31 | -0.238 |
| L39_layer_input | remove_from_repair | 10/31 | 10/31 | 0/31 | -0.238 |
| L37_layer_out | remove_from_repair | 9/31 | 9/31 | 0/31 | -0.228 |
| L38_layer_input | remove_from_repair | 9/31 | 9/31 | 0/31 | -0.228 |
| L37_layer_out | random_semantic | 8/31 | 8/31 | 0/31 | -0.274 |
| L38_layer_out | restore | 29/31 | 5/31 | 24/31 | 1.712 |
| L39_layer_input | restore | 29/31 | 5/31 | 24/31 | 1.712 |
| L37_layer_out | restore | 29/31 | 5/31 | 24/31 | 1.712 |
| L38_layer_input | restore | 29/31 | 5/31 | 24/31 | 1.712 |
| L36_layer_out | restore | 29/31 | 5/31 | 24/31 | 1.659 |
| L37_layer_input | restore | 29/31 | 5/31 | 24/31 | 1.659 |

### Examples

- sample=20 node=__baseline__ mode=base tok0=' v' exact=False wrong=True margin=0.500 text=' v22'
- sample=20 node=__baseline__ mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=0.500 text=' v05'
- sample=20 node=L38_layer_out mode=restore_semantic tok0=' v' exact=True wrong=False margin=2.875 text=' v05'
- sample=20 node=L38_layer_out mode=random_semantic tok0=' v' exact=True wrong=False margin=0.750 text=' v05'
- sample=20 node=L38_layer_out mode=reverse_semantic tok0=' o' exact=False wrong=False margin=-0.562 text=' o05'

## qwen3

- rows: 17 / raw_cases: 256 / target_seen: 17
- scan_layers: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- downstream_layers: [29, 30, 31, 32, 33, 34, 35]
- top_nodes: ['L34_layer_out', 'L35_layer_input', 'L33_layer_out', 'L34_layer_input', 'L32_layer_out', 'L33_layer_input']

### Natural Margin Writer Scan

| rank | node | mean_margin_delta | positive_rate | mean_cos | mean_delta_norm | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | L34_layer_out | 4.610 | 0.941 | 0.045 | 65.922 | 4.339 |
| 2 | L35_layer_input | 4.610 | 0.941 | 0.045 | 65.922 | 4.339 |
| 3 | L33_layer_out | 3.779 | 0.765 | 0.041 | 56.564 | 2.889 |
| 4 | L34_layer_input | 3.779 | 0.765 | 0.041 | 56.564 | 2.889 |
| 5 | L32_layer_out | 3.489 | 0.765 | 0.043 | 50.002 | 2.668 |
| 6 | L33_layer_input | 3.489 | 0.765 | 0.043 | 50.002 | 2.668 |
| 7 | L35_layer_out | 3.477 | 0.706 | 0.025 | 87.397 | 2.454 |
| 8 | L34_attn_out | 2.756 | 0.647 | 0.086 | 25.023 | 1.783 |
| 9 | L32_attn_out | 1.585 | 0.941 | 0.089 | 11.902 | 1.492 |
| 10 | L30_layer_out | 1.517 | 0.765 | 0.024 | 39.278 | 1.160 |
| 11 | L31_layer_input | 1.517 | 0.765 | 0.024 | 39.278 | 1.160 |
| 12 | L31_layer_out | 1.982 | 0.471 | 0.026 | 44.604 | 0.933 |

### Causal Patch Audit

| node | mode | tok0 | exact | wrong_exact | mean_prefix_margin |
|---|---|---:|---:|---:|---:|
| __baseline__ | repair_prompt | 14/17 | 11/17 | 3/17 | 1.110 |
| __baseline__ | semantic_cumulative | 10/17 | 10/17 | 0/17 | 0.213 |
| __baseline__ | base | 10/17 | 1/17 | 9/17 | 0.213 |
| L34_layer_out | restore_semantic | 14/17 | 14/17 | 0/17 | 1.044 |
| L35_layer_input | restore_semantic | 14/17 | 14/17 | 0/17 | 1.044 |
| L33_layer_out | restore_semantic | 12/17 | 12/17 | 0/17 | 0.765 |
| L34_layer_input | restore_semantic | 12/17 | 12/17 | 0/17 | 0.765 |
| L32_layer_out | restore_semantic | 11/17 | 11/17 | 0/17 | 0.647 |
| L33_layer_input | restore_semantic | 11/17 | 11/17 | 0/17 | 0.647 |
| L32_layer_out | random_semantic | 11/17 | 11/17 | 0/17 | 0.353 |
| L33_layer_input | random_semantic | 11/17 | 11/17 | 0/17 | 0.199 |
| L34_layer_out | random_semantic | 10/17 | 10/17 | 0/17 | 0.206 |
| L34_layer_input | random_semantic | 10/17 | 10/17 | 0/17 | 0.199 |
| L33_layer_out | random_semantic | 9/17 | 9/17 | 0/17 | 0.235 |
| L32_layer_out | remove_from_repair | 11/17 | 8/17 | 3/17 | 0.699 |
| L33_layer_input | remove_from_repair | 11/17 | 8/17 | 3/17 | 0.699 |
| L33_layer_out | remove_from_repair | 11/17 | 8/17 | 3/17 | 0.551 |
| L34_layer_input | remove_from_repair | 11/17 | 8/17 | 3/17 | 0.551 |
| L35_layer_input | random_semantic | 8/17 | 8/17 | 0/17 | 0.132 |
| L34_layer_out | remove_from_repair | 10/17 | 7/17 | 3/17 | 0.272 |
| L35_layer_input | remove_from_repair | 10/17 | 7/17 | 3/17 | 0.272 |
| L33_layer_out | reverse_semantic | 7/17 | 7/17 | 0/17 | -0.228 |
| L34_layer_input | reverse_semantic | 7/17 | 7/17 | 0/17 | -0.228 |
| L32_layer_out | reverse_semantic | 7/17 | 7/17 | 0/17 | -0.243 |
| L33_layer_input | reverse_semantic | 7/17 | 7/17 | 0/17 | -0.243 |
| L34_layer_out | reverse_semantic | 6/17 | 6/17 | 0/17 | -0.596 |
| L35_layer_input | reverse_semantic | 6/17 | 6/17 | 0/17 | -0.596 |

### Examples

- sample=22 node=__baseline__ mode=base tok0=' v' exact=False wrong=True margin=2.000 text=' v22'
- sample=22 node=__baseline__ mode=semantic_cumulative tok0=' v' exact=True wrong=False margin=2.000 text=' v05'
- sample=22 node=L34_layer_out mode=restore_semantic tok0=' v' exact=True wrong=False margin=1.500 text=' v05'
- sample=22 node=L34_layer_out mode=random_semantic tok0=' v' exact=True wrong=False margin=2.250 text=' v05'
- sample=22 node=L34_layer_out mode=reverse_semantic tok0=' v' exact=True wrong=False margin=2.500 text=' v05'
