# Phase560 Continuous Prototype Surgery Summary (continuous_static)

## Object Audit

| repeat | object | tokens | chars |
|---|---|---:|---:|
| repeat0 | tram | 1 | 4 |
| repeat1 | subway | 2 | 6 |
| repeat2 | helicopter | 2 | 10 |
| repeat3 | tractor | 1 | 7 |
| repeat4 | rocket | 1 | 6 |
| repeat5 | canoe | 2 | 5 |
| repeat6 | ferry | 2 | 5 |
| repeat7 | jeep | 2 | 4 |
| repeat8 | ambulance | 2 | 9 |
| repeat9 | cart | 1 | 4 |
| repeat10 | sled | 1 | 4 |
| repeat11 | wagon | 1 | 5 |

## Route: forbidden_sentence_completion:temperature<-forbidden_definition

### clean_non_object_rate

| model | base | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.48 | 0.61 | 0.00 | 0.00 | 0.01 | 0.00 | 0.02 | 0.00 | 0.00 |
| glm4 | 0.25 | 0.22 | 0.00 | 0.01 | 0.00 | 0.04 | 0.00 | 0.01 | 0.01 |
| deepseek7b | 0.22 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### steering gain versus baseline

| model | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | +0.14 | -0.48 | -0.48 | -0.47 | -0.48 | -0.46 | -0.48 | -0.48 |
| glm4 | -0.03 | -0.25 | -0.24 | -0.25 | -0.21 | -0.25 | -0.24 | -0.24 |
| deepseek7b | +0.03 | -0.22 | -0.22 | -0.22 | -0.22 | -0.22 | -0.22 | -0.22 |

### object_echo_rate

| model | base | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.64 | 0.51 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm4 | 0.76 | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| deepseek7b | 0.50 | 0.61 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Route: forbidden_definition:top_p<-forbidden_definition

### clean_non_object_rate

| model | base | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.26 | 0.28 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm4 | 0.41 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| deepseek7b | 0.17 | 0.21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |

### steering gain versus baseline

| model | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | +0.02 | -0.26 | -0.26 | -0.26 | -0.26 | -0.26 | -0.26 | -0.26 |
| glm4 | -0.11 | -0.41 | -0.41 | -0.41 | -0.41 | -0.41 | -0.41 | -0.41 |
| deepseek7b | +0.04 | -0.17 | -0.17 | -0.17 | -0.17 | -0.17 | -0.17 | -0.16 |

### object_echo_rate

| model | base | remove | repeat0(tram) | repeat2(heli) | repeat4(rocket) | repeat10(sled) | mean | pca1 | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.38 | 0.34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| glm4 | 0.08 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| deepseek7b | 0.22 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 |

## Timing

| model | minutes | seeds | max tokens |
|---|---:|---|---:|
| qwen3 | 18.8 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |
| glm4 | 33.74 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |
| deepseek7b | 25.62 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |
