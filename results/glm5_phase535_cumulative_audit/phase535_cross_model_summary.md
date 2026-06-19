# Phase535 Cumulative Audit Summary

## qwen3

windows={'early': [8, 10, 12], 'center': [10, 12, 14], 'late': [12, 14, 16]}, train_n=12, test_n=8, bridge_n=12, max_new_tokens=4, alphas=[2.0, 4.0, 6.0, 8.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

### fruit_nonfruit

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [8, 10, 12] | +0.848/+1.660/0.66/n | +0.906/+1.858/0.63/n | +0.648/+1.212/0.60/n | -0.062 | 0 |
| center [10, 12, 14] | +0.598/+1.512/0.55/n | +0.469/+1.665/0.47/n | +0.332/+1.133/0.30/n | -0.059 | 0 |
| late [12, 14, 16] | +0.219/+1.375/0.21/n | +0.219/+1.456/0.26/n | +0.164/+0.969/0.22/n | -0.027 | 0 |

Generation best_window=early

| condition | trace |
|---|---|
| baseline | hit=0.00, rank=610.8, m1=+0.260 |
| common | hit=0.08, rank=196.8, m1=+2.385 |
| direct | hit=0.08, rank=183.0, m1=+3.320 |
| random | hit=0.00, rank=883.2, m1=+0.255 |

### animal_vehicle

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [8, 10, 12] | +0.031/+0.163/0.07/n | +0.133/+0.452/0.21/n | +0.148/+0.309/0.25/n | +0.078 | 0 |
| center [10, 12, 14] | +0.070/+0.118/0.19/n | +0.023/+0.138/0.11/n | +0.141/+0.453/0.14/n | +0.086 | 0 |
| late [12, 14, 16] | +0.008/+0.130/0.01/n | -0.078/+0.107/0.18/n | +0.039/+0.184/0.04/n | +0.023 | 0 |

Generation best_window=center

| condition | trace |
|---|---|
| baseline | hit=0.08, rank=13.8, m1=+2.490 |
| common | hit=0.08, rank=13.3, m1=+2.729 |
| direct | hit=0.08, rank=13.2, m1=+2.669 |
| random | hit=0.08, rank=14.8, m1=+2.555 |

## glm4

windows={'early': [22, 24, 26], 'center': [24, 26, 28], 'late': [26, 28, 30]}, train_n=12, test_n=8, bridge_n=12, max_new_tokens=4, alphas=[2.0, 4.0, 6.0, 8.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

### fruit_nonfruit

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [22, 24, 26] | +0.992/+1.827/0.70/n | -0.195/+0.054/0.13/n | +0.957/+1.482/0.42/n | -0.092 | 0 |
| center [24, 26, 28] | +0.678/+1.196/0.53/n | -0.225/-0.086/0.20/n | +0.393/+0.625/0.36/n | -0.040 | 0 |
| late [26, 28, 30] | +0.442/+0.619/0.40/n | -0.428/-0.067/2.76/n | +0.284/+0.318/0.31/n | -0.030 | 0 |

Generation best_window=early

| condition | trace |
|---|---|
| baseline | hit=0.08, rank=94.9, m1=+1.280 |
| common | hit=0.00, rank=106.2, m1=+1.261 |
| direct | hit=0.08, rank=74.2, m1=-0.263 |
| random | hit=0.08, rank=90.8, m1=+1.327 |

### animal_vehicle

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [22, 24, 26] | +0.237/+1.469/0.08/n | -0.111/+0.348/0.23/n | +0.046/+1.253/0.05/n | +0.420 | 0 |
| center [24, 26, 28] | -0.109/+1.061/0.05/n | -0.127/+0.202/0.18/n | -0.103/+0.453/0.20/n | +0.380 | 0 |
| late [26, 28, 30] | -0.178/+0.384/0.29/n | -0.196/+0.060/0.27/n | -0.099/+0.316/0.20/n | +0.057 | 0 |

Generation best_window=early

| condition | trace |
|---|---|
| baseline | hit=0.50, rank=25.8, m1=+2.503 |
| common | hit=0.00, rank=278.3, m1=+2.761 |
| direct | hit=0.17, rank=47.4, m1=+2.418 |
| random | hit=0.08, rank=151.7, m1=+2.172 |

## deepseek7b

windows={'early': [14, 16, 18], 'center': [16, 18, 20], 'late': [18, 20, 22]}, train_n=12, test_n=8, bridge_n=12, max_new_tokens=4, alphas=[2.0, 4.0, 6.0, 8.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

### fruit_nonfruit

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [14, 16, 18] | +0.039/+0.057/0.28/n | +0.047/+0.057/0.46/n | +0.062/+0.098/0.42/n | +0.102 | 0 |
| center [16, 18, 20] | +0.062/+0.069/0.36/n | +0.008/+0.135/0.03/n | +0.062/+0.131/0.29/n | +0.090 | 0 |
| late [18, 20, 22] | +0.078/+0.249/0.19/n | +0.023/+0.167/0.09/n | +0.062/+0.093/0.43/n | +0.047 | 0 |

Generation best_window=late

| condition | trace |
|---|---|
| baseline | hit=0.00, rank=499.8, m1=-0.891 |
| common | hit=0.00, rank=575.8, m1=-0.529 |
| direct | hit=0.00, rank=577.9, m1=-0.638 |
| random | hit=0.00, rank=527.3, m1=-0.880 |

### animal_vehicle

Transfer format: min / mean / ratio / pass.

| window | common | direct | shuffled_template | random max min | random pass count |
|---|---:|---:|---:|---:|---:|
| early [14, 16, 18] | +0.117/+0.284/0.43/n | +0.188/+0.263/1.20/n | +0.117/+0.238/0.64/n | +0.145 | 0 |
| center [16, 18, 20] | +0.164/+0.297/0.66/n | +0.133/+0.215/0.74/n | +0.133/+0.242/0.83/n | +0.066 | 0 |
| late [18, 20, 22] | +0.180/+0.273/0.72/n | +0.125/+0.170/0.89/n | +0.102/+0.219/0.47/n | +0.068 | 0 |

Generation best_window=late

| condition | trace |
|---|---|
| baseline | hit=0.00, rank=1388.2, m1=-1.818 |
| common | hit=0.00, rank=1403.4, m1=-1.503 |
| direct | hit=0.00, rank=1316.2, m1=-1.651 |
| random | hit=0.00, rank=1378.9, m1=-1.805 |

## Cross-model Compact

| model | verdict |
|---|---|
| qwen3 | no_clean_common |
| glm4 | no_clean_common |
| deepseek7b | no_clean_common |

