# Phase 640 Cross-Model Summary

目标：扫描 separator boundary 的 layer/component writer，定位 inline protocol state 的写入候选。

## qwen3

- raw_cases: 256 / target_seen: 17 / cases_written: 17 / mode_rows: 2414
- target_only: True / top_k: 20
- scan_layers: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`
- control_layers: `[25, 26, 27, 28, 29]`
- filtered: `{'not_target': 239, 'separator_len_mismatch': 0, 'empty_patch': 0}`

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 17 | 14/17 | 0/17 | 1.2 | 1.272 | correct_prefix:14, space:3 |  v:14,  :3 |
| inline | 17 | 1/17 | 9/17 | 4.8 | -1.471 | newline:9, space:7, correct_prefix:1 |  ?\n\n:9,  :7,  v:1 |

### Best Restore Candidates

| layer | component | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---:|---:|---:|---:|---:|
| 26 | mlp_out | 17 | 16/17 | 0/17 | 1.1 | 1.199 |
| 4 | attn_out | 17 | 15/17 | 0/17 | 1.1 | 1.926 |
| 16 | attn_out | 17 | 15/17 | 0/17 | 1.1 | 2.419 |
| 20 | attn_out | 17 | 15/17 | 0/17 | 1.1 | 1.662 |
| 33 | attn_out | 17 | 15/17 | 0/17 | 1.1 | 1.654 |
| 35 | attn_out | 17 | 15/17 | 0/17 | 1.1 | 1.463 |
| 0 | layer_input | 17 | 15/17 | 0/17 | 1.2 | 1.074 |
| 10 | mlp_out | 17 | 14/17 | 0/17 | 1.2 | 2.581 |
| 8 | attn_out | 17 | 14/17 | 0/17 | 1.2 | 1.368 |
| 10 | attn_out | 17 | 14/17 | 0/17 | 1.2 | 1.640 |
| 18 | attn_out | 17 | 13/17 | 0/17 | 1.2 | 1.426 |
| 16 | mlp_out | 17 | 9/17 | 0/17 | 1.6 | 1.074 |
| 14 | mlp_out | 17 | 8/17 | 0/17 | 1.6 | 1.338 |
| 0 | attn_out | 17 | 14/17 | 1/17 | 1.2 | 0.816 |
| 0 | layer_out | 17 | 14/17 | 1/17 | 1.3 | 0.676 |
| 22 | mlp_out | 17 | 14/17 | 1/17 | 1.4 | 0.603 |
| 30 | attn_out | 17 | 13/17 | 1/17 | 1.2 | 1.169 |
| 34 | attn_out | 17 | 13/17 | 1/17 | 1.2 | 1.206 |
| 0 | mlp_out | 17 | 13/17 | 1/17 | 1.3 | 1.007 |
| 27 | attn_out | 17 | 13/17 | 1/17 | 1.3 | 1.338 |
| 29 | attn_out | 17 | 13/17 | 1/17 | 1.4 | 1.140 |
| 4 | mlp_out | 17 | 13/17 | 1/17 | 1.4 | 0.941 |
| 2 | mlp_out | 17 | 13/17 | 1/17 | 1.5 | 0.831 |
| 32 | mlp_out | 17 | 13/17 | 1/17 | 1.5 | 0.838 |
| 8 | mlp_out | 17 | 13/17 | 1/17 | 1.5 | 0.728 |
| 14 | attn_out | 17 | 13/17 | 1/17 | 1.6 | 0.816 |
| 30 | mlp_out | 17 | 13/17 | 1/17 | 1.6 | 0.544 |
| 31 | mlp_out | 17 | 13/17 | 1/17 | 1.6 | 0.544 |
| 26 | attn_out | 17 | 12/17 | 1/17 | 1.5 | 0.809 |
| 2 | attn_out | 17 | 12/17 | 1/17 | 1.6 | 0.706 |
| 12 | mlp_out | 17 | 12/17 | 1/17 | 1.7 | 0.588 |
| 31 | attn_out | 17 | 11/17 | 1/17 | 1.4 | 1.176 |

### Control Candidates

| layer | component | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---|---:|---:|---:|---:|---:|
| 26 | attn_out | random | 17 | 14/17 | 0/17 | 1.3 | 1.184 |
| 27 | mlp_out | random | 17 | 13/17 | 0/17 | 1.2 | 1.191 |
| 29 | mlp_out | random | 17 | 13/17 | 0/17 | 1.4 | 1.169 |
| 25 | mlp_out | random | 17 | 14/17 | 1/17 | 1.4 | 1.301 |
| 25 | attn_out | random | 17 | 13/17 | 1/17 | 1.2 | 1.301 |
| 27 | attn_out | random | 17 | 13/17 | 1/17 | 1.2 | 1.074 |
| 29 | attn_out | random | 17 | 13/17 | 1/17 | 1.2 | 1.338 |
| 28 | attn_out | random | 17 | 13/17 | 1/17 | 1.3 | 1.191 |
| 26 | mlp_out | random | 17 | 12/17 | 1/17 | 1.5 | 1.000 |
| 26 | layer_input | random | 17 | 12/17 | 1/17 | 1.9 | 0.566 |
| 29 | layer_out | random | 17 | 11/17 | 1/17 | 1.5 | 0.603 |
| 26 | layer_out | random | 17 | 11/17 | 1/17 | 1.7 | 0.713 |
| 28 | mlp_out | random | 17 | 13/17 | 2/17 | 1.4 | 1.140 |
| 27 | layer_out | random | 17 | 13/17 | 3/17 | 1.3 | 1.074 |
| 27 | layer_input | random | 17 | 12/17 | 3/17 | 1.8 | 0.743 |
| 28 | layer_out | random | 17 | 9/17 | 3/17 | 1.9 | 0.485 |
| 25 | layer_input | random | 17 | 9/17 | 5/17 | 2.0 | 0.478 |
| 29 | layer_input | random | 17 | 10/17 | 6/17 | 1.6 | 0.456 |
| 25 | layer_out | random | 17 | 8/17 | 6/17 | 2.0 | 0.213 |
| 28 | layer_input | random | 17 | 7/17 | 6/17 | 1.9 | 0.301 |
| 29 | mlp_out | reverse | 17 | 16/17 | 0/17 | 1.1 | 2.088 |
| 25 | mlp_out | reverse | 17 | 15/17 | 0/17 | 1.1 | 2.449 |
| 27 | mlp_out | reverse | 17 | 15/17 | 0/17 | 1.1 | 1.846 |
| 26 | attn_out | reverse | 17 | 14/17 | 0/17 | 1.2 | 1.397 |
| 28 | attn_out | reverse | 17 | 14/17 | 0/17 | 1.2 | 1.787 |
| 29 | attn_out | reverse | 17 | 13/17 | 0/17 | 1.2 | 1.360 |
| 28 | mlp_out | reverse | 17 | 12/17 | 0/17 | 1.3 | 1.522 |
| 27 | attn_out | reverse | 17 | 11/17 | 0/17 | 1.4 | 0.904 |
| 26 | mlp_out | reverse | 17 | 9/17 | 0/17 | 1.8 | 1.081 |
| 29 | layer_out | reverse | 17 | 8/17 | 0/17 | 1.5 | 3.294 |
| 27 | layer_out | reverse | 17 | 7/17 | 0/17 | 1.8 | 3.184 |
| 28 | layer_input | reverse | 17 | 7/17 | 0/17 | 1.8 | 3.184 |

### Component Timeline Restore

- layer_input: L0 tok0=15/17 nl=0/17 pmn=1.07; L2 tok0=13/17 nl=3/17 pmn=0.60; L4 tok0=8/17 nl=8/17 pmn=-0.08; L6 tok0=8/17 nl=9/17 pmn=-0.10; L8 tok0=2/17 nl=15/17 pmn=-0.83; L10 tok0=3/17 nl=14/17 pmn=-0.83; L12 tok0=6/17 nl=10/17 pmn=-0.18; L14 tok0=0/17 nl=17/17 pmn=-1.88; L16 tok0=0/17 nl=17/17 pmn=-2.23; L18 tok0=2/17 nl=13/17 pmn=-1.10; L20 tok0=1/17 nl=14/17 pmn=-1.23; L22 tok0=0/17 nl=16/17 pmn=-1.76; L23 tok0=0/17 nl=17/17 pmn=-2.27; L24 tok0=0/17 nl=17/17 pmn=-2.32; L25 tok0=0/17 nl=17/17 pmn=-2.30; L26 tok0=0/17 nl=17/17 pmn=-2.18; L27 tok0=0/17 nl=17/17 pmn=-2.29; L28 tok0=0/17 nl=16/17 pmn=-1.90; L29 tok0=0/17 nl=14/17 pmn=-1.76; L30 tok0=0/17 nl=15/17 pmn=-2.06; L31 tok0=0/17 nl=16/17 pmn=-2.01; L32 tok0=0/17 nl=16/17 pmn=-2.04; L33 tok0=1/17 nl=13/17 pmn=-1.54; L34 tok0=0/17 nl=14/17 pmn=-1.61; L35 tok0=1/17 nl=11/17 pmn=-1.47
- attn_out: L0 tok0=14/17 nl=1/17 pmn=0.82; L2 tok0=12/17 nl=1/17 pmn=0.71; L4 tok0=15/17 nl=0/17 pmn=1.93; L6 tok0=11/17 nl=1/17 pmn=0.93; L8 tok0=14/17 nl=0/17 pmn=1.37; L10 tok0=14/17 nl=0/17 pmn=1.64; L12 tok0=9/17 nl=1/17 pmn=0.76; L14 tok0=13/17 nl=1/17 pmn=0.82; L16 tok0=15/17 nl=0/17 pmn=2.42; L18 tok0=13/17 nl=0/17 pmn=1.43; L20 tok0=15/17 nl=0/17 pmn=1.66; L22 tok0=8/17 nl=3/17 pmn=0.14; L23 tok0=11/17 nl=3/17 pmn=0.34; L24 tok0=11/17 nl=5/17 pmn=0.46; L25 tok0=9/17 nl=1/17 pmn=1.01; L26 tok0=12/17 nl=1/17 pmn=0.81; L27 tok0=13/17 nl=1/17 pmn=1.34; L28 tok0=8/17 nl=2/17 pmn=0.30; L29 tok0=13/17 nl=1/17 pmn=1.14; L30 tok0=13/17 nl=1/17 pmn=1.17; L31 tok0=11/17 nl=1/17 pmn=1.18; L32 tok0=8/17 nl=9/17 pmn=0.02; L33 tok0=15/17 nl=0/17 pmn=1.65; L34 tok0=13/17 nl=1/17 pmn=1.21; L35 tok0=15/17 nl=0/17 pmn=1.46
- mlp_out: L0 tok0=13/17 nl=1/17 pmn=1.01; L2 tok0=13/17 nl=1/17 pmn=0.83; L4 tok0=13/17 nl=1/17 pmn=0.94; L6 tok0=10/17 nl=6/17 pmn=0.24; L8 tok0=13/17 nl=1/17 pmn=0.73; L10 tok0=14/17 nl=0/17 pmn=2.58; L12 tok0=12/17 nl=1/17 pmn=0.59; L14 tok0=8/17 nl=0/17 pmn=1.34; L16 tok0=9/17 nl=0/17 pmn=1.07; L18 tok0=5/17 nl=8/17 pmn=-0.19; L20 tok0=10/17 nl=2/17 pmn=0.68; L22 tok0=14/17 nl=1/17 pmn=0.60; L23 tok0=13/17 nl=2/17 pmn=0.87; L24 tok0=7/17 nl=10/17 pmn=-0.08; L25 tok0=8/17 nl=8/17 pmn=0.01; L26 tok0=16/17 nl=0/17 pmn=1.20; L27 tok0=8/17 nl=7/17 pmn=0.08; L28 tok0=12/17 nl=3/17 pmn=0.40; L29 tok0=12/17 nl=4/17 pmn=0.27; L30 tok0=13/17 nl=1/17 pmn=0.54; L31 tok0=13/17 nl=1/17 pmn=0.54; L32 tok0=13/17 nl=1/17 pmn=0.84; L33 tok0=13/17 nl=2/17 pmn=0.62; L34 tok0=10/17 nl=2/17 pmn=0.65; L35 tok0=9/17 nl=5/17 pmn=0.39
- layer_out: L0 tok0=14/17 nl=1/17 pmn=0.68; L2 tok0=11/17 nl=5/17 pmn=0.25; L4 tok0=10/17 nl=7/17 pmn=0.09; L6 tok0=9/17 nl=8/17 pmn=-0.05; L8 tok0=3/17 nl=14/17 pmn=-0.85; L10 tok0=4/17 nl=13/17 pmn=-0.46; L12 tok0=1/17 nl=16/17 pmn=-1.49; L14 tok0=0/17 nl=17/17 pmn=-2.17; L16 tok0=1/17 nl=16/17 pmn=-1.37; L18 tok0=2/17 nl=12/17 pmn=-1.07; L20 tok0=0/17 nl=16/17 pmn=-1.38; L22 tok0=0/17 nl=17/17 pmn=-2.27; L23 tok0=0/17 nl=17/17 pmn=-2.32; L24 tok0=0/17 nl=17/17 pmn=-2.30; L25 tok0=0/17 nl=17/17 pmn=-2.18; L26 tok0=0/17 nl=17/17 pmn=-2.29; L27 tok0=0/17 nl=16/17 pmn=-1.90; L28 tok0=0/17 nl=14/17 pmn=-1.76; L29 tok0=0/17 nl=15/17 pmn=-2.06; L30 tok0=0/17 nl=16/17 pmn=-2.01; L31 tok0=0/17 nl=16/17 pmn=-2.04; L32 tok0=1/17 nl=13/17 pmn=-1.54; L33 tok0=0/17 nl=14/17 pmn=-1.61; L34 tok0=1/17 nl=11/17 pmn=-1.47; L35 tok0=1/17 nl=9/17 pmn=-1.47

## glm4

- raw_cases: 256 / target_seen: 31 / cases_written: 31 / mode_rows: 4650
- target_only: True / top_k: 20
- scan_layers: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]`
- control_layers: `[30, 31, 32, 33, 34]`
- filtered: `{'not_target': 225, 'separator_len_mismatch': 0, 'empty_patch': 0}`

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| original | 31 | 29/31 | 0/31 | 1.1 | 80.722 | correct_prefix:29, word:2 |  v:29,  c:2 |
| inline | 31 | 27/31 | 0/31 | 1.2 | 71.648 | correct_prefix:27, explanation:3, word:1 |  v:27,  Yes:3,  c:1 |

### Best Restore Candidates

| layer | component | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---:|---:|---:|---:|---:|
| 6 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 80.746 |
| 39 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 80.758 |
| 0 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 86.840 |
| 0 | layer_out | 31 | 30/31 | 0/31 | 1.1 | 77.688 |
| 0 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 80.730 |
| 2 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 80.723 |
| 2 | layer_input | 31 | 30/31 | 0/31 | 1.1 | 74.673 |
| 2 | layer_out | 31 | 30/31 | 0/31 | 1.1 | 74.666 |
| 2 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 77.673 |
| 4 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 77.696 |
| 4 | layer_input | 31 | 30/31 | 0/31 | 1.1 | 77.704 |
| 4 | layer_out | 31 | 30/31 | 0/31 | 1.1 | 65.542 |
| 6 | layer_out | 31 | 30/31 | 0/31 | 1.1 | 62.466 |
| 8 | layer_input | 31 | 30/31 | 0/31 | 1.1 | 71.613 |
| 10 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 74.669 |
| 12 | layer_input | 31 | 30/31 | 0/31 | 1.1 | 77.655 |
| 26 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 62.498 |
| 27 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 83.780 |
| 28 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 77.731 |
| 29 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 99.000 |
| 29 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 80.720 |
| 30 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 83.810 |
| 33 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 80.740 |
| 33 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 92.948 |
| 35 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 74.676 |
| 35 | mlp_out | 31 | 30/31 | 0/31 | 1.1 | 86.840 |
| 37 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 80.706 |
| 38 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 71.578 |
| 39 | attn_out | 31 | 30/31 | 0/31 | 1.1 | 65.508 |
| 6 | layer_input | 31 | 30/31 | 0/31 | 1.1 | 59.443 |
| 12 | layer_out | 31 | 30/31 | 0/31 | 1.1 | 77.635 |
| 4 | mlp_out | 31 | 29/31 | 0/31 | 1.1 | 80.702 |

### Control Candidates

| layer | component | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---|---:|---:|---:|---:|---:|
| 32 | layer_input | random | 31 | 30/31 | 0/31 | 1.1 | 83.798 |
| 33 | layer_out | random | 31 | 30/31 | 0/31 | 1.1 | 83.793 |
| 31 | layer_out | random | 31 | 30/31 | 0/31 | 1.1 | 80.721 |
| 31 | mlp_out | random | 31 | 30/31 | 0/31 | 1.1 | 83.776 |
| 32 | mlp_out | random | 31 | 30/31 | 0/31 | 1.1 | 83.766 |
| 33 | mlp_out | random | 31 | 30/31 | 0/31 | 1.1 | 80.714 |
| 32 | layer_out | random | 31 | 30/31 | 0/31 | 1.1 | 86.798 |
| 31 | layer_input | random | 31 | 29/31 | 0/31 | 1.1 | 80.710 |
| 32 | attn_out | random | 31 | 29/31 | 0/31 | 1.1 | 77.671 |
| 33 | attn_out | random | 31 | 29/31 | 0/31 | 1.1 | 77.710 |
| 34 | attn_out | random | 31 | 29/31 | 0/31 | 1.1 | 80.732 |
| 34 | mlp_out | random | 31 | 29/31 | 0/31 | 1.1 | 74.668 |
| 30 | attn_out | random | 31 | 29/31 | 0/31 | 1.1 | 80.718 |
| 30 | layer_input | random | 31 | 29/31 | 0/31 | 1.1 | 77.744 |
| 30 | mlp_out | random | 31 | 29/31 | 0/31 | 1.1 | 80.726 |
| 31 | attn_out | random | 31 | 29/31 | 0/31 | 1.1 | 80.722 |
| 33 | layer_input | random | 31 | 29/31 | 0/31 | 1.1 | 74.675 |
| 34 | layer_input | random | 31 | 29/31 | 0/31 | 1.1 | 77.720 |
| 34 | layer_out | random | 31 | 29/31 | 0/31 | 1.1 | 89.855 |
| 30 | layer_out | random | 31 | 28/31 | 0/31 | 1.1 | 83.748 |
| 30 | attn_out | reverse | 31 | 30/31 | 0/31 | 1.1 | 77.664 |
| 31 | attn_out | reverse | 31 | 30/31 | 0/31 | 1.1 | 86.859 |
| 31 | mlp_out | reverse | 31 | 30/31 | 0/31 | 1.1 | 77.681 |
| 32 | attn_out | reverse | 31 | 30/31 | 0/31 | 1.1 | 83.782 |
| 32 | mlp_out | reverse | 31 | 30/31 | 0/31 | 1.1 | 83.768 |
| 33 | mlp_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 62.393 |
| 30 | mlp_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 71.613 |
| 32 | layer_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 40.883 |
| 33 | attn_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 80.703 |
| 33 | layer_input | reverse | 31 | 29/31 | 0/31 | 1.1 | 40.883 |
| 34 | attn_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 95.985 |
| 34 | mlp_out | reverse | 31 | 29/31 | 0/31 | 1.1 | 80.736 |

### Component Timeline Restore

- layer_input: L0 tok0=16/31 nl=0/31 pmn=99.00; L2 tok0=30/31 nl=0/31 pmn=74.67; L4 tok0=30/31 nl=0/31 pmn=77.70; L6 tok0=30/31 nl=0/31 pmn=59.44; L8 tok0=30/31 nl=0/31 pmn=71.61; L10 tok0=28/31 nl=0/31 pmn=80.69; L12 tok0=30/31 nl=0/31 pmn=77.66; L14 tok0=28/31 nl=0/31 pmn=77.65; L16 tok0=27/31 nl=0/31 pmn=83.80; L18 tok0=29/31 nl=0/31 pmn=77.62; L20 tok0=29/31 nl=0/31 pmn=77.69; L22 tok0=29/31 nl=0/31 pmn=86.83; L24 tok0=27/31 nl=0/31 pmn=92.95; L26 tok0=27/31 nl=0/31 pmn=95.96; L27 tok0=27/31 nl=0/31 pmn=99.00; L28 tok0=27/31 nl=0/31 pmn=99.00; L29 tok0=27/31 nl=0/31 pmn=99.00; L30 tok0=27/31 nl=0/31 pmn=99.00; L31 tok0=27/31 nl=0/31 pmn=99.00; L32 tok0=27/31 nl=0/31 pmn=99.00; L33 tok0=27/31 nl=0/31 pmn=99.00; L34 tok0=27/31 nl=0/31 pmn=99.00; L35 tok0=27/31 nl=0/31 pmn=95.96; L36 tok0=27/31 nl=0/31 pmn=95.96; L37 tok0=27/31 nl=0/31 pmn=95.96; L38 tok0=27/31 nl=0/31 pmn=92.92; L39 tok0=27/31 nl=0/31 pmn=83.78
- attn_out: L0 tok0=30/31 nl=0/31 pmn=86.84; L2 tok0=30/31 nl=0/31 pmn=80.72; L4 tok0=30/31 nl=0/31 pmn=77.70; L6 tok0=29/31 nl=0/31 pmn=80.71; L8 tok0=28/31 nl=0/31 pmn=83.79; L10 tok0=30/31 nl=0/31 pmn=74.67; L12 tok0=29/31 nl=0/31 pmn=83.74; L14 tok0=29/31 nl=0/31 pmn=80.70; L16 tok0=29/31 nl=0/31 pmn=83.76; L18 tok0=29/31 nl=0/31 pmn=86.86; L20 tok0=29/31 nl=0/31 pmn=86.80; L22 tok0=29/31 nl=0/31 pmn=86.87; L24 tok0=29/31 nl=0/31 pmn=83.81; L26 tok0=29/31 nl=0/31 pmn=95.98; L27 tok0=30/31 nl=0/31 pmn=83.78; L28 tok0=29/31 nl=0/31 pmn=89.86; L29 tok0=30/31 nl=0/31 pmn=99.00; L30 tok0=29/31 nl=0/31 pmn=83.79; L31 tok0=29/31 nl=0/31 pmn=65.48; L32 tok0=29/31 nl=0/31 pmn=80.71; L33 tok0=30/31 nl=0/31 pmn=80.74; L34 tok0=29/31 nl=0/31 pmn=38.00; L35 tok0=30/31 nl=0/31 pmn=74.68; L36 tok0=29/31 nl=0/31 pmn=80.70; L37 tok0=30/31 nl=0/31 pmn=80.71; L38 tok0=30/31 nl=0/31 pmn=71.58; L39 tok0=30/31 nl=0/31 pmn=65.51
- mlp_out: L0 tok0=30/31 nl=0/31 pmn=80.73; L2 tok0=30/31 nl=0/31 pmn=77.67; L4 tok0=29/31 nl=0/31 pmn=80.70; L6 tok0=30/31 nl=0/31 pmn=80.75; L8 tok0=29/31 nl=0/31 pmn=83.79; L10 tok0=29/31 nl=0/31 pmn=77.64; L12 tok0=24/31 nl=0/31 pmn=86.80; L14 tok0=28/31 nl=0/31 pmn=83.79; L16 tok0=28/31 nl=0/31 pmn=86.83; L18 tok0=28/31 nl=0/31 pmn=80.74; L20 tok0=28/31 nl=0/31 pmn=86.86; L22 tok0=25/31 nl=0/31 pmn=80.72; L24 tok0=27/31 nl=0/31 pmn=83.78; L26 tok0=30/31 nl=0/31 pmn=62.50; L27 tok0=29/31 nl=0/31 pmn=80.74; L28 tok0=30/31 nl=0/31 pmn=77.73; L29 tok0=30/31 nl=0/31 pmn=80.72; L30 tok0=30/31 nl=0/31 pmn=83.81; L31 tok0=29/31 nl=0/31 pmn=83.79; L32 tok0=29/31 nl=0/31 pmn=77.70; L33 tok0=30/31 nl=0/31 pmn=92.95; L34 tok0=29/31 nl=0/31 pmn=74.64; L35 tok0=30/31 nl=0/31 pmn=86.84; L36 tok0=29/31 nl=0/31 pmn=80.72; L37 tok0=29/31 nl=0/31 pmn=83.80; L38 tok0=29/31 nl=0/31 pmn=80.74; L39 tok0=30/31 nl=0/31 pmn=80.76
- layer_out: L0 tok0=30/31 nl=0/31 pmn=77.69; L2 tok0=30/31 nl=0/31 pmn=74.67; L4 tok0=30/31 nl=0/31 pmn=65.54; L6 tok0=30/31 nl=0/31 pmn=62.47; L8 tok0=29/31 nl=0/31 pmn=80.70; L10 tok0=28/31 nl=0/31 pmn=77.63; L12 tok0=30/31 nl=0/31 pmn=77.64; L14 tok0=27/31 nl=0/31 pmn=80.70; L16 tok0=29/31 nl=0/31 pmn=89.89; L18 tok0=29/31 nl=0/31 pmn=86.85; L20 tok0=29/31 nl=0/31 pmn=89.87; L22 tok0=27/31 nl=0/31 pmn=89.89; L24 tok0=27/31 nl=0/31 pmn=95.97; L26 tok0=27/31 nl=0/31 pmn=99.00; L27 tok0=27/31 nl=0/31 pmn=99.00; L28 tok0=27/31 nl=0/31 pmn=99.00; L29 tok0=27/31 nl=0/31 pmn=99.00; L30 tok0=27/31 nl=0/31 pmn=99.00; L31 tok0=27/31 nl=0/31 pmn=99.00; L32 tok0=27/31 nl=0/31 pmn=99.00; L33 tok0=27/31 nl=0/31 pmn=99.00; L34 tok0=27/31 nl=0/31 pmn=95.96; L35 tok0=27/31 nl=0/31 pmn=95.96; L36 tok0=27/31 nl=0/31 pmn=95.96; L37 tok0=27/31 nl=0/31 pmn=92.92; L38 tok0=27/31 nl=0/31 pmn=83.78; L39 tok0=27/31 nl=0/31 pmn=71.65

## deepseek7b

- raw_cases: 256 / target_seen: 82 / cases_written: 82 / mode_rows: 10004
- target_only: True / top_k: 20
- scan_layers: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- control_layers: `[19, 20, 21, 22, 23]`
- filtered: `{'not_target': 174, 'separator_len_mismatch': 0, 'empty_patch': 0}`

### Baselines

| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |
|---|---:|---:|---:|---:|---:|---|---|
| inline | 82 | 75/82 | 0/82 | 1.1 | 2.236 | correct_prefix:75, space:7 |  v:75,  :7 |
| original | 82 | 20/82 | 57/82 | 9.4 | -1.704 | newline:57, correct_prefix:20, word:3, space:1, explanation:1 |  ?\n\n:57,  v:20,  o:2,  c:1,  :1,  yes:1 |

### Best Restore Candidates

| layer | component | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---:|---:|---:|---:|---:|
| 20 | layer_out | 82 | 77/82 | 0/82 | 1.1 | 2.419 |
| 21 | layer_input | 82 | 77/82 | 0/82 | 1.1 | 2.419 |
| 27 | layer_out | 82 | 75/82 | 0/82 | 1.1 | 2.236 |
| 26 | layer_out | 82 | 75/82 | 0/82 | 1.1 | 2.458 |
| 27 | layer_input | 82 | 75/82 | 0/82 | 1.1 | 2.458 |
| 23 | layer_out | 82 | 72/82 | 0/82 | 1.1 | 2.534 |
| 24 | layer_input | 82 | 72/82 | 0/82 | 1.1 | 2.534 |
| 25 | layer_out | 82 | 71/82 | 0/82 | 1.1 | 2.421 |
| 26 | layer_input | 82 | 71/82 | 0/82 | 1.1 | 2.421 |
| 24 | layer_out | 82 | 70/82 | 0/82 | 1.1 | 2.520 |
| 25 | layer_input | 82 | 70/82 | 0/82 | 1.1 | 2.520 |
| 0 | layer_input | 82 | 35/82 | 0/82 | 3.9 | 0.563 |
| 18 | layer_out | 82 | 76/82 | 1/82 | 1.1 | 2.025 |
| 19 | layer_input | 82 | 76/82 | 1/82 | 1.1 | 2.025 |
| 19 | layer_out | 82 | 76/82 | 1/82 | 1.1 | 2.281 |
| 20 | layer_input | 82 | 76/82 | 1/82 | 1.1 | 2.281 |
| 17 | layer_out | 82 | 76/82 | 1/82 | 1.2 | 1.986 |
| 18 | layer_input | 82 | 76/82 | 1/82 | 1.2 | 1.986 |
| 22 | layer_out | 82 | 73/82 | 1/82 | 1.1 | 2.355 |
| 23 | layer_input | 82 | 73/82 | 1/82 | 1.1 | 2.355 |
| 21 | layer_out | 82 | 72/82 | 1/82 | 1.1 | 2.280 |
| 22 | layer_input | 82 | 72/82 | 1/82 | 1.1 | 2.280 |
| 16 | layer_out | 82 | 76/82 | 2/82 | 1.2 | 1.664 |
| 17 | layer_input | 82 | 76/82 | 2/82 | 1.2 | 1.664 |
| 14 | layer_out | 82 | 76/82 | 2/82 | 1.2 | 1.540 |
| 16 | layer_input | 82 | 75/82 | 3/82 | 1.3 | 1.375 |
| 14 | layer_input | 82 | 73/82 | 5/82 | 1.3 | 1.457 |
| 10 | layer_out | 82 | 64/82 | 12/82 | 1.8 | 1.113 |
| 10 | layer_input | 82 | 61/82 | 12/82 | 2.1 | 1.055 |
| 8 | layer_out | 82 | 60/82 | 12/82 | 2.3 | 1.042 |
| 6 | layer_out | 82 | 58/82 | 13/82 | 2.5 | 0.934 |
| 12 | layer_input | 82 | 61/82 | 15/82 | 1.9 | 0.928 |

### Control Candidates

| layer | component | control | n | tok0 | newline_top0 | rank | prefix-newline |
|---:|---|---|---:|---:|---:|---:|---:|
| 21 | layer_input | random | 82 | 19/82 | 46/82 | 12.1 | -1.688 |
| 20 | layer_out | random | 82 | 17/82 | 46/82 | 10.5 | -1.710 |
| 22 | layer_out | random | 82 | 21/82 | 47/82 | 11.2 | -1.783 |
| 19 | layer_input | random | 82 | 19/82 | 47/82 | 9.9 | -1.508 |
| 23 | layer_input | random | 82 | 19/82 | 47/82 | 12.6 | -1.707 |
| 21 | layer_out | random | 82 | 19/82 | 48/82 | 12.3 | -1.767 |
| 23 | layer_out | random | 82 | 18/82 | 50/82 | 13.1 | -1.802 |
| 20 | layer_input | random | 82 | 13/82 | 50/82 | 10.0 | -1.505 |
| 22 | attn_out | random | 82 | 21/82 | 51/82 | 9.9 | -1.660 |
| 22 | layer_input | random | 82 | 18/82 | 51/82 | 10.0 | -1.671 |
| 19 | layer_out | random | 82 | 18/82 | 52/82 | 14.6 | -1.819 |
| 19 | mlp_out | random | 82 | 20/82 | 54/82 | 9.4 | -1.637 |
| 21 | attn_out | random | 82 | 21/82 | 55/82 | 8.7 | -1.701 |
| 23 | attn_out | random | 82 | 20/82 | 55/82 | 9.5 | -1.666 |
| 20 | mlp_out | random | 82 | 19/82 | 55/82 | 9.0 | -1.629 |
| 21 | mlp_out | random | 82 | 19/82 | 55/82 | 9.3 | -1.707 |
| 19 | attn_out | random | 82 | 18/82 | 55/82 | 9.4 | -1.694 |
| 20 | attn_out | random | 82 | 20/82 | 56/82 | 10.3 | -1.754 |
| 22 | mlp_out | random | 82 | 20/82 | 57/82 | 9.2 | -1.711 |
| 23 | mlp_out | random | 82 | 20/82 | 57/82 | 9.9 | -1.733 |
| 19 | layer_input | reverse | 82 | 0/82 | 53/82 | 138.0 | -5.973 |
| 21 | attn_out | reverse | 82 | 19/82 | 54/82 | 13.9 | -1.900 |
| 19 | layer_out | reverse | 82 | 0/82 | 55/82 | 350.5 | -7.196 |
| 20 | layer_input | reverse | 82 | 0/82 | 55/82 | 350.5 | -7.196 |
| 21 | layer_out | reverse | 82 | 1/82 | 56/82 | 422.9 | -7.430 |
| 22 | layer_input | reverse | 82 | 1/82 | 56/82 | 422.9 | -7.430 |
| 20 | attn_out | reverse | 82 | 17/82 | 57/82 | 19.4 | -2.217 |
| 21 | mlp_out | reverse | 82 | 12/82 | 57/82 | 26.1 | -2.759 |
| 22 | mlp_out | reverse | 82 | 15/82 | 59/82 | 21.6 | -2.625 |
| 22 | attn_out | reverse | 82 | 14/82 | 60/82 | 31.3 | -3.055 |
| 19 | attn_out | reverse | 82 | 6/82 | 61/82 | 44.2 | -3.913 |
| 20 | layer_out | reverse | 82 | 1/82 | 61/82 | 367.0 | -7.345 |

### Component Timeline Restore

- layer_input: L0 tok0=35/82 nl=0/82 pmn=0.56; L2 tok0=41/82 nl=36/82 pmn=-0.13; L4 tok0=50/82 nl=24/82 pmn=0.41; L6 tok0=52/82 nl=19/82 pmn=0.66; L8 tok0=57/82 nl=15/82 pmn=0.90; L10 tok0=61/82 nl=12/82 pmn=1.05; L12 tok0=61/82 nl=15/82 pmn=0.93; L14 tok0=73/82 nl=5/82 pmn=1.46; L16 tok0=75/82 nl=3/82 pmn=1.38; L17 tok0=76/82 nl=2/82 pmn=1.66; L18 tok0=76/82 nl=1/82 pmn=1.99; L19 tok0=76/82 nl=1/82 pmn=2.03; L20 tok0=76/82 nl=1/82 pmn=2.28; L21 tok0=77/82 nl=0/82 pmn=2.42; L22 tok0=72/82 nl=1/82 pmn=2.28; L23 tok0=73/82 nl=1/82 pmn=2.36; L24 tok0=72/82 nl=0/82 pmn=2.53; L25 tok0=70/82 nl=0/82 pmn=2.52; L26 tok0=71/82 nl=0/82 pmn=2.42; L27 tok0=75/82 nl=0/82 pmn=2.46
- attn_out: L0 tok0=32/82 nl=44/82 pmn=-0.65; L2 tok0=17/82 nl=59/82 pmn=-1.93; L4 tok0=23/82 nl=53/82 pmn=-1.51; L6 tok0=21/82 nl=56/82 pmn=-1.91; L8 tok0=21/82 nl=57/82 pmn=-1.46; L10 tok0=20/82 nl=56/82 pmn=-1.59; L12 tok0=25/82 nl=50/82 pmn=-0.99; L14 tok0=21/82 nl=58/82 pmn=-1.43; L16 tok0=31/82 nl=45/82 pmn=-0.65; L17 tok0=25/82 nl=45/82 pmn=-1.03; L18 tok0=19/82 nl=58/82 pmn=-1.76; L19 tok0=36/82 nl=40/82 pmn=-0.23; L20 tok0=16/82 nl=54/82 pmn=-1.55; L21 tok0=17/82 nl=61/82 pmn=-1.59; L22 tok0=24/82 nl=53/82 pmn=-0.77; L23 tok0=30/82 nl=40/82 pmn=-0.51; L24 tok0=23/82 nl=57/82 pmn=-0.97; L25 tok0=14/82 nl=54/82 pmn=-2.05; L26 tok0=22/82 nl=56/82 pmn=-0.96; L27 tok0=18/82 nl=62/82 pmn=-1.96
- mlp_out: L0 tok0=23/82 nl=52/82 pmn=-1.32; L2 tok0=19/82 nl=56/82 pmn=-1.74; L4 tok0=23/82 nl=54/82 pmn=-1.58; L6 tok0=21/82 nl=58/82 pmn=-1.66; L8 tok0=19/82 nl=60/82 pmn=-1.82; L10 tok0=28/82 nl=31/82 pmn=-0.86; L12 tok0=23/82 nl=56/82 pmn=-1.44; L14 tok0=28/82 nl=46/82 pmn=-0.68; L16 tok0=12/82 nl=67/82 pmn=-1.98; L17 tok0=31/82 nl=45/82 pmn=-0.86; L18 tok0=33/82 nl=37/82 pmn=-0.34; L19 tok0=30/82 nl=32/82 pmn=-0.36; L20 tok0=29/82 nl=40/82 pmn=-0.47; L21 tok0=25/82 nl=54/82 pmn=-0.82; L22 tok0=28/82 nl=48/82 pmn=-0.84; L23 tok0=23/82 nl=35/82 pmn=-1.39; L24 tok0=21/82 nl=57/82 pmn=-1.56; L25 tok0=25/82 nl=50/82 pmn=-0.86; L26 tok0=21/82 nl=52/82 pmn=-1.57; L27 tok0=12/82 nl=39/82 pmn=-2.21
- layer_out: L0 tok0=33/82 nl=43/82 pmn=-0.63; L2 tok0=41/82 nl=37/82 pmn=-0.07; L4 tok0=51/82 nl=23/82 pmn=0.48; L6 tok0=58/82 nl=13/82 pmn=0.93; L8 tok0=60/82 nl=12/82 pmn=1.04; L10 tok0=64/82 nl=12/82 pmn=1.11; L12 tok0=62/82 nl=17/82 pmn=0.91; L14 tok0=76/82 nl=2/82 pmn=1.54; L16 tok0=76/82 nl=2/82 pmn=1.66; L17 tok0=76/82 nl=1/82 pmn=1.99; L18 tok0=76/82 nl=1/82 pmn=2.03; L19 tok0=76/82 nl=1/82 pmn=2.28; L20 tok0=77/82 nl=0/82 pmn=2.42; L21 tok0=72/82 nl=1/82 pmn=2.28; L22 tok0=73/82 nl=1/82 pmn=2.36; L23 tok0=72/82 nl=0/82 pmn=2.53; L24 tok0=70/82 nl=0/82 pmn=2.52; L25 tok0=71/82 nl=0/82 pmn=2.42; L26 tok0=75/82 nl=0/82 pmn=2.46; L27 tok0=75/82 nl=0/82 pmn=2.24
