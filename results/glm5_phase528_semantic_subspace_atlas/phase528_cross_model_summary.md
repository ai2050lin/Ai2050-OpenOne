# Phase528 Semantic Subspace Atlas Summary

## qwen3

layer=L12, alpha=8.0, train_n=10, test_n=6, attn=sdpa

### Direction Cosine Matrix

| dir | category | color | object |
|---|---:|---:|---:|
| category | +1.0000 | +0.0316 | +0.0739 |
| color | +0.0316 | +1.0000 | +0.0021 |
| object | +0.0739 | +0.0021 | +1.0000 |

### Readout Alignment

| dir | norm | readout norm % | semantic norm % | cos to readout |
|---|---:|---:|---:|---:|
| category | 13.0487 | 6.45 | 99.79 | +0.06450 |
| color | 6.3515 | 4.10 | 99.92 | +0.04096 |
| object | 6.6395 | 0.40 | 100.00 | +0.00399 |

### Selectivity Matrix: Δmargin

| direction -> task | category | color | object |
|---|---:|---:|---:|
| category | +1.7240 | +0.1146 | -0.2812 |
| color | -0.3490 | +0.3021 | -0.1146 |
| object | +0.4115 | -0.4896 | -0.1250 |

### Positive Control

| dir | own Δmargin | own Δtop1 | max other abs Δmargin | selectivity ratio |
|---|---:|---:|---:|---:|
| category | +1.7240 | +0.0000 | 0.2812 | 6.1296 |
| color | +0.3021 | +0.1667 | 0.3490 | 0.8657 |
| object | -0.1250 | +0.0000 | 0.4896 | 0.2553 |

## glm4

layer=L26, alpha=8.0, train_n=10, test_n=6, attn=sdpa

### Direction Cosine Matrix

| dir | category | color | object |
|---|---:|---:|---:|
| category | +1.0000 | +0.0213 | +0.1733 |
| color | +0.0213 | +1.0000 | +0.0310 |
| object | +0.1733 | +0.0310 | +1.0000 |

### Readout Alignment

| dir | norm | readout norm % | semantic norm % | cos to readout |
|---|---:|---:|---:|---:|
| category | 11.9065 | 3.71 | 99.93 | +0.03706 |
| color | 9.6532 | 18.04 | 98.36 | +0.18038 |
| object | 11.8925 | 6.66 | 99.78 | +0.06665 |

### Selectivity Matrix: Δmargin

| direction -> task | category | color | object |
|---|---:|---:|---:|
| category | +0.1328 | +0.4375 | -0.2689 |
| color | +0.4010 | +1.7135 | +0.0029 |
| object | +0.3932 | +1.4167 | +0.0137 |

### Positive Control

| dir | own Δmargin | own Δtop1 | max other abs Δmargin | selectivity ratio |
|---|---:|---:|---:|---:|
| category | +0.1328 | +0.0000 | 0.4375 | 0.3036 |
| color | +1.7135 | +0.0000 | 0.4010 | 4.2727 |
| object | +0.0137 | +0.0000 | 1.4167 | 0.0097 |

## deepseek7b

layer=L18, alpha=8.0, train_n=10, test_n=6, attn=sdpa

### Direction Cosine Matrix

| dir | category | color | object |
|---|---:|---:|---:|
| category | +1.0000 | +0.0104 | +0.0467 |
| color | +0.0104 | +1.0000 | +0.0362 |
| object | +0.0467 | +0.0362 | +1.0000 |

### Readout Alignment

| dir | norm | readout norm % | semantic norm % | cos to readout |
|---|---:|---:|---:|---:|
| category | 70.1382 | 0.52 | 100.00 | -0.00516 |
| color | 46.4210 | 1.77 | 99.98 | +0.01770 |
| object | 65.0971 | 0.28 | 100.00 | -0.00276 |

### Selectivity Matrix: Δmargin

| direction -> task | category | color | object |
|---|---:|---:|---:|
| category | +0.0156 | +0.0208 | -0.1042 |
| color | +0.0052 | +0.0000 | -0.1667 |
| object | +0.0260 | -0.0833 | -0.0104 |

### Positive Control

| dir | own Δmargin | own Δtop1 | max other abs Δmargin | selectivity ratio |
|---|---:|---:|---:|---:|
| category | +0.0156 | +0.0000 | 0.1042 | 0.1500 |
| color | +0.0000 | +0.0000 | 0.1667 | 0.0000 |
| object | -0.0104 | +0.0000 | 0.0833 | 0.1250 |

## Cross-model Compact

| model | mean abs offdiag cos | mean own Δ | mean off abs Δ | selectivity ratio | color positive Δ | category readout % | color readout % |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.0359 | +0.6337 | 0.2934 | 2.4169 | +0.3021 | 6.45 | 4.10 |
| glm4 | 0.0752 | +0.6200 | 0.4867 | 1.5286 | +1.7135 | 3.71 | 18.04 |
| deepseek7b | 0.0311 | +0.0017 | 0.0677 | 0.0917 | +0.0000 | 0.52 | 1.77 |

