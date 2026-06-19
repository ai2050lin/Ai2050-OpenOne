# Phase537 Vehicle/Furniture Audit Summary

## qwen3

source=vehicle_furniture, offpairs=['fruit_tool', 'animal_tool', 'clothing_tool', 'fruit_vegetable'], train_n=12, test_n=8, bridge_n=12, alphas=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], seeds=8, attn=sdpa

Transfer format: source min / source mean / off-pair max abs / specificity.

| window | layers | common | direct | shuffled | random max | random pass-like | common pass |
|---|---|---:|---:|---:|---:|---:|---|
| early | [8, 10, 12] | +0.477/+0.919/0.746/0.64 | +0.211/+0.732/0.625/0.34 | +0.285/+0.512/0.707/0.40 | +0.156 | 0 | False |
| center | [10, 12, 14] | +1.172/+1.513/1.168/1.00 | +0.812/+1.376/1.551/0.52 | +0.211/+0.354/0.531/0.40 | +0.062 | 0 | True |
| late | [12, 14, 16] | +1.285/+1.792/1.922/0.67 | +1.227/+1.582/1.562/0.78 | +0.328/+0.831/1.039/0.32 | +0.031 | 0 | False |
| extended | [8, 10, 12, 14, 16] | +1.164/+1.517/1.387/0.84 | +0.633/+1.389/1.746/0.36 | +0.293/+0.520/0.707/0.41 | +0.094 | 0 | False |

Best common window: late, transfer=+1.285/+1.792/1.922/0.67

| off-pair | max abs delta at best common |
|---|---:|
| fruit_tool | 0.547 |
| animal_tool | 0.648 |
| clothing_tool | 1.922 |
| fruit_vegetable | 0.555 |

Generation bridge window: late

| condition | hit / best rank / first margin |
|---|---:|
| baseline | 0.33/15.8/+1.240 |
| common | 0.50/17.2/+2.331 (Δrank +1.4, Δmargin +1.091) |
| direct | 0.50/3.3/+2.531 (Δrank -12.5, Δmargin +1.292) |
| shuffled | 0.33/93.2/+1.370 (Δrank +77.3, Δmargin +0.130) |
| random | 0.25/14.5/+1.391 (Δrank -1.3, Δmargin +0.151) |

Verdict: single_window_candidate

## glm4

source=vehicle_furniture, offpairs=['fruit_tool', 'animal_tool', 'clothing_tool', 'fruit_vegetable'], train_n=12, test_n=8, bridge_n=12, alphas=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], seeds=8, attn=sdpa

Transfer format: source min / source mean / off-pair max abs / specificity.

| window | layers | common | direct | shuffled | random max | random pass-like | common pass |
|---|---|---:|---:|---:|---:|---:|---|
| early | [22, 24, 26] | +1.402/+2.154/2.954/0.47 | +1.383/+2.056/1.830/0.76 | +0.375/+0.836/2.929/0.13 | +0.043 | 0 | False |
| center | [24, 26, 28] | +1.492/+1.796/3.037/0.49 | +1.699/+2.556/2.096/0.81 | -0.247/+0.572/0.844/0.29 | +0.209 | 0 | False |
| late | [26, 28, 30] | +1.487/+2.028/1.977/0.75 | +1.943/+2.541/1.191/1.63 | -0.039/+0.716/1.326/0.03 | +0.309 | 0 | False |
| extended | [22, 24, 26, 28, 30] | +1.577/+2.097/2.689/0.59 | +1.758/+2.826/3.669/0.48 | +0.209/+0.565/2.640/0.08 | +0.113 | 0 | False |

Best common window: extended, transfer=+1.577/+2.097/2.689/0.59

| off-pair | max abs delta at best common |
|---|---:|
| fruit_tool | 0.666 |
| animal_tool | 2.689 |
| clothing_tool | 0.684 |
| fruit_vegetable | 0.857 |

Generation bridge window: extended

| condition | hit / best rank / first margin |
|---|---:|
| baseline | 0.00/82.4/-0.504 |
| common | 0.00/32.2/+0.853 (Δrank -50.3, Δmargin +1.356) |
| direct | 0.00/22.1/+2.375 (Δrank -60.3, Δmargin +2.879) |
| shuffled | 0.00/80.7/-0.538 (Δrank -1.8, Δmargin -0.034) |
| random | 0.00/81.5/+0.112 (Δrank -0.9, Δmargin +0.616) |

Verdict: strong_but_not_clean:extended

## deepseek7b

source=vehicle_furniture, offpairs=['fruit_tool', 'animal_tool', 'clothing_tool', 'fruit_vegetable'], train_n=12, test_n=8, bridge_n=12, alphas=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], seeds=8, attn=sdpa

Transfer format: source min / source mean / off-pair max abs / specificity.

| window | layers | common | direct | shuffled | random max | random pass-like | common pass |
|---|---|---:|---:|---:|---:|---:|---|
| early | [14, 16, 18] | -0.041/+0.043/0.066/0.61 | +0.063/+0.340/0.354/0.18 | -0.040/+0.003/0.078/0.51 | +0.047 | 0 | False |
| center | [16, 18, 20] | -0.018/+0.308/0.352/0.05 | +0.153/+0.405/0.256/0.60 | -0.042/-0.003/0.128/0.33 | +0.031 | 0 | False |
| late | [18, 20, 22] | +0.059/+0.341/0.305/0.19 | +0.224/+0.407/0.398/0.56 | -0.044/-0.002/0.100/0.44 | -0.004 | 0 | False |
| extended | [14, 16, 18, 20, 22] | -0.005/+0.414/0.453/0.01 | +0.287/+0.607/0.482/0.59 | -0.081/-0.014/0.156/0.52 | +0.023 | 0 | False |

Best common window: late, transfer=+0.059/+0.341/0.305/0.19

| off-pair | max abs delta at best common |
|---|---:|
| fruit_tool | 0.164 |
| animal_tool | 0.305 |
| clothing_tool | 0.218 |
| fruit_vegetable | 0.277 |

Generation bridge window: late

| condition | hit / best rank / first margin |
|---|---:|
| baseline | 0.00/2967.2/-5.619 |
| common | 0.00/1607.6/-5.590 (Δrank -1359.6, Δmargin +0.030) |
| direct | 0.00/2117.9/-5.219 (Δrank -849.2, Δmargin +0.401) |
| shuffled | 0.00/3862.2/-5.673 (Δrank +895.0, Δmargin -0.054) |
| random | 0.00/2955.1/-5.605 (Δrank -12.1, Δmargin +0.014) |

Verdict: not_clean

## Cross-model Compact

| model | best common window | source min | specificity | verdict |
|---|---|---:|---:|---|
| qwen3 | late | +1.285 | 0.67 | single_window_candidate |
| glm4 | extended | +1.577 | 0.59 | strong_but_not_clean:extended |
| deepseek7b | late | +0.059 | 0.19 | not_clean |

