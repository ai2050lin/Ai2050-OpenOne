# Phase 157 Cross-model Final Residual LM-head Competition Summary

## qwen3

cases=180, attention=L36, mlp=L36, heads=32

### All / difficult / multiple-choice

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | -0.294 | -0.105 | -0.086 | +0.012 | -1.262 | -1.552 | -1.017 | -1.271 | -1.155 | -1.399 | -1.312 | -1.310 | 42.042 | 45.622 |
| difficult_formats | 144 | -0.644 | +0.017 | +0.063 | +0.039 | -0.766 | -1.528 | -0.672 | -0.565 | -0.675 | -1.431 | -1.018 | -0.650 | 39.443 | 43.402 |
| multiple_choice_control | 36 | 1.109 | -0.590 | -0.683 | -0.096 | -3.247 | -1.652 | -2.394 | -4.093 | -3.077 | -1.273 | -2.489 | -3.953 | 52.438 | 54.503 |

### By format

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | -1.064 | +0.076 | +0.080 | -0.016 | -0.616 | -2.267 | -0.750 | -0.533 | -0.149 | -1.951 | -0.396 | -0.069 | 38.393 | 43.030 |
| label_colon | 36 | 0.166 | +0.241 | +0.387 | +0.004 | -0.608 | -1.521 | -1.034 | -0.881 | -0.889 | -1.823 | -1.679 | -1.351 | 38.251 | 44.672 |
| list_answer | 36 | -0.889 | -0.298 | -0.371 | +0.080 | -1.079 | -1.506 | -1.173 | -1.184 | -1.201 | -1.562 | -1.669 | -1.333 | 38.991 | 42.087 |
| multiple_choice | 36 | 1.109 | -0.590 | -0.683 | -0.096 | -3.247 | -1.652 | -2.394 | -4.093 | -3.077 | -1.273 | -2.489 | -3.953 | 52.438 | 54.503 |
| quoted_answer | 36 | -0.790 | +0.049 | +0.155 | +0.089 | -0.759 | -0.816 | +0.268 | +0.339 | -0.459 | -0.387 | -0.330 | +0.155 | 42.137 | 43.819 |

### By category

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | -1.589 | +0.129 | +0.262 | +0.081 | -1.107 | -1.565 | -1.114 | -1.333 | -0.915 | -1.583 | -1.366 | -1.443 | 41.782 | 45.127 |
| container | 30 | 0.372 | +0.285 | +0.349 | +0.005 | -0.804 | -1.526 | -0.966 | -1.102 | -0.730 | -1.418 | -1.361 | -1.181 | 41.159 | 44.625 |
| furniture | 30 | -1.717 | +0.180 | +0.322 | +0.169 | -0.971 | -1.735 | -0.999 | -1.179 | -0.823 | -1.591 | -1.245 | -1.359 | 41.631 | 44.979 |
| number | 30 | -0.327 | -0.532 | -0.637 | -0.007 | -1.516 | -1.249 | -0.840 | -1.333 | -1.442 | -0.969 | -1.046 | -1.294 | 43.385 | 47.375 |
| plant | 30 | 1.544 | -0.687 | -0.608 | -0.191 | -1.840 | -1.421 | -1.170 | -1.278 | -1.671 | -1.294 | -1.571 | -1.235 | 41.804 | 45.783 |
| time | 30 | -0.045 | -0.004 | -0.204 | +0.017 | -1.333 | -1.820 | -1.010 | -1.398 | -1.350 | -1.540 | -1.286 | -1.349 | 42.491 | 45.843 |

### By family

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | -0.334 | -0.189 | -0.207 | +0.028 | -1.135 | -1.145 | -1.086 | -1.306 | -0.664 | -0.572 | -0.946 | -1.029 | 36.820 | 40.796 |
| neutral | 60 | -1.182 | +0.105 | +0.155 | +0.076 | -0.902 | -1.890 | -0.993 | -1.056 | -1.085 | -2.091 | -1.532 | -1.342 | 43.181 | 47.217 |
| short | 60 | 0.635 | -0.230 | -0.207 | -0.068 | -1.749 | -1.622 | -0.971 | -1.450 | -1.716 | -1.534 | -1.460 | -1.560 | 46.125 | 48.854 |

### By split

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | -0.268 | -0.061 | -0.043 | +0.015 | -1.190 | -1.560 | -0.983 | -1.183 | -1.093 | -1.433 | -1.257 | -1.220 | 42.277 | 45.808 |
| front_back | 90 | -0.319 | -0.148 | -0.129 | +0.010 | -1.334 | -1.545 | -1.050 | -1.358 | -1.218 | -1.366 | -1.367 | -1.401 | 41.807 | 45.436 |

### Cases

| case | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_genericΔ |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | -1.63 | +0.59 | +0.69 | +0.00 | -0.50 | -0.84 | -1.08 |
| back_front:long:answer_one_word:container | -1.68 | +0.51 | +0.57 | -0.01 | -0.48 | -1.46 | -0.99 |
| back_front:long:answer_one_word:furniture | -1.63 | +0.49 | +0.58 | -0.07 | -0.56 | -2.45 | -1.06 |
| back_front:long:answer_one_word:number | -1.65 | +0.51 | +0.52 | -0.06 | -0.55 | -1.83 | -1.06 |
| back_front:long:answer_one_word:plant | -1.48 | +0.44 | +0.53 | +0.01 | -0.62 | -1.18 | -1.06 |
| back_front:long:answer_one_word:time | -1.59 | +0.51 | +0.68 | +0.07 | -0.52 | -2.30 | -1.03 |
| back_front:long:label_colon:clothing | 0.20 | +1.67 | +2.24 | -0.00 | +0.20 | -0.76 | -0.76 |
| back_front:long:label_colon:container | -0.01 | +0.53 | +0.57 | -0.13 | -0.59 | -0.95 | -0.66 |
| back_front:long:label_colon:furniture | 0.41 | +0.95 | +1.46 | +0.09 | -0.42 | -1.24 | -0.57 |
| back_front:long:label_colon:number | -1.94 | -0.74 | -0.82 | -0.05 | -1.66 | -1.44 | -1.13 |
| back_front:long:label_colon:plant | 2.03 | +1.11 | +1.05 | -0.11 | -0.63 | -1.04 | -1.42 |
| back_front:long:label_colon:time | -2.23 | -0.53 | -0.58 | +0.24 | -1.92 | -1.50 | -1.18 |
| back_front:long:list_answer:clothing | -0.54 | +0.15 | +0.32 | +0.22 | -1.13 | -1.03 | -1.31 |
| back_front:long:list_answer:container | -0.37 | -0.17 | +0.04 | -0.01 | -1.48 | -1.06 | -1.48 |
| back_front:long:list_answer:furniture | -0.71 | -0.07 | -0.19 | +0.05 | -1.26 | -1.20 | -1.33 |
| back_front:long:list_answer:number | -0.39 | -0.31 | +0.07 | +0.03 | -1.37 | -1.54 | -1.37 |
| back_front:long:list_answer:plant | -0.41 | +0.03 | +0.07 | +0.01 | -1.51 | -0.93 | -1.35 |
| back_front:long:list_answer:time | -0.48 | -0.03 | -0.09 | +0.19 | -1.20 | -1.27 | -1.30 |
| back_front:long:multiple_choice:clothing | -5.96 | +0.80 | +0.78 | +0.09 | -1.35 | -2.15 | -3.43 |
| back_front:long:multiple_choice:container | 4.17 | +0.19 | +0.12 | -0.33 | -2.37 | -1.16 | -3.87 |
| back_front:long:multiple_choice:furniture | -7.08 | +0.75 | +0.96 | -0.02 | -1.64 | -2.50 | -3.83 |
| back_front:long:multiple_choice:number | 3.27 | -3.56 | -4.74 | -0.03 | -6.49 | -0.19 | -4.92 |
| back_front:long:multiple_choice:plant | 6.16 | -0.31 | -0.97 | -0.09 | -1.93 | -2.07 | -3.77 |
| back_front:long:multiple_choice:time | 4.75 | -3.96 | -5.90 | -0.19 | -7.28 | -0.45 | -4.56 |
| back_front:long:quoted_answer:clothing | 0.31 | -0.27 | -0.38 | +0.08 | -0.47 | -0.31 | +0.10 |
| back_front:long:quoted_answer:container | -0.13 | +0.19 | +0.15 | +0.32 | +0.09 | -1.75 | +0.33 |
| back_front:long:quoted_answer:furniture | -0.99 | +0.61 | +0.37 | -0.51 | +0.20 | -1.63 | +0.23 |
| back_front:long:quoted_answer:number | -0.12 | +0.31 | +0.32 | +0.26 | -0.15 | -1.24 | +0.22 |
| back_front:long:quoted_answer:plant | -0.26 | +0.20 | +0.29 | -0.07 | +0.02 | -1.13 | +0.29 |
| back_front:long:quoted_answer:time | 0.02 | +0.28 | +0.37 | +0.08 | +0.15 | -1.13 | +0.35 |
| back_front:neutral:answer_one_word:clothing | -2.37 | +0.54 | +0.47 | +0.19 | -0.08 | -2.79 | -0.11 |
| back_front:neutral:answer_one_word:container | -1.65 | +0.66 | +0.58 | -0.01 | +0.59 | -2.84 | +0.66 |
| back_front:neutral:answer_one_word:furniture | -1.79 | +0.71 | +0.72 | +0.00 | +0.25 | -3.17 | -0.03 |
| back_front:neutral:answer_one_word:number | -2.24 | +0.65 | +0.74 | +0.04 | +1.57 | -2.37 | +1.34 |
| back_front:neutral:answer_one_word:plant | -1.52 | +0.54 | +0.44 | +0.17 | -0.18 | -2.89 | -0.22 |
| back_front:neutral:answer_one_word:time | -2.32 | +0.51 | +0.56 | +0.17 | +0.24 | -3.12 | -0.09 |
| back_front:neutral:label_colon:clothing | 1.08 | +0.73 | +1.09 | +0.13 | -0.54 | -2.20 | -0.89 |
| back_front:neutral:label_colon:container | -1.33 | +0.39 | +0.15 | +0.05 | -0.37 | -1.73 | -0.71 |
| back_front:neutral:label_colon:furniture | -1.79 | +0.69 | +0.81 | +0.07 | -0.34 | -1.75 | -1.15 |
| back_front:neutral:label_colon:number | -2.81 | -0.79 | -0.73 | +0.17 | -1.22 | -2.14 | -0.87 |
| back_front:neutral:label_colon:plant | 0.14 | +0.07 | +0.21 | -0.01 | -1.05 | -1.96 | -0.72 |
| back_front:neutral:label_colon:time | -1.70 | +0.14 | +0.32 | +0.05 | -0.80 | -2.35 | -1.18 |
| back_front:neutral:list_answer:clothing | -1.69 | -0.17 | -0.25 | -0.01 | -0.69 | -2.12 | -0.69 |
| back_front:neutral:list_answer:container | -1.78 | -0.27 | -0.32 | -0.03 | -0.52 | -2.14 | -0.52 |
| back_front:neutral:list_answer:furniture | -1.79 | -0.25 | -0.46 | -0.04 | -0.56 | -2.24 | -0.56 |
| back_front:neutral:list_answer:number | -1.47 | -0.15 | -0.32 | +0.29 | +0.38 | -1.82 | +0.38 |
| back_front:neutral:list_answer:plant | -1.74 | -0.32 | -0.55 | +0.02 | -0.82 | -2.01 | -0.80 |
| back_front:neutral:list_answer:time | -1.46 | -0.14 | -0.23 | +0.29 | -0.43 | -2.25 | -0.56 |
| back_front:neutral:multiple_choice:clothing | -5.77 | -0.26 | +0.32 | +0.11 | -1.49 | -1.24 | -2.61 |
| back_front:neutral:multiple_choice:container | 3.78 | +0.60 | +0.46 | -0.42 | -1.60 | -3.56 | -1.96 |
| back_front:neutral:multiple_choice:furniture | -6.08 | -0.31 | +0.42 | +0.88 | -1.32 | -1.01 | -2.89 |
| back_front:neutral:multiple_choice:number | -1.39 | -0.71 | -0.49 | +0.01 | -1.68 | -0.98 | -2.23 |
| back_front:neutral:multiple_choice:plant | 6.08 | -2.75 | -2.72 | -0.16 | -4.30 | -0.90 | -2.57 |
| back_front:neutral:multiple_choice:time | 3.02 | +0.62 | +0.50 | -0.16 | -1.33 | -2.40 | -2.81 |
| back_front:neutral:quoted_answer:clothing | -1.65 | -0.00 | +0.11 | +0.30 | -1.23 | -1.07 | -0.30 |
| back_front:neutral:quoted_answer:container | -0.22 | -0.28 | +0.18 | +0.01 | -0.50 | -0.86 | -0.25 |
| back_front:neutral:quoted_answer:furniture | -1.91 | +0.81 | +1.24 | +0.14 | -0.26 | -1.29 | -0.45 |
| back_front:neutral:quoted_answer:number | -1.03 | +0.41 | +0.28 | +0.10 | +0.85 | -0.28 | -0.29 |
| back_front:neutral:quoted_answer:plant | -1.70 | -0.02 | +0.29 | +0.04 | -1.41 | -0.97 | -0.12 |
| back_front:neutral:quoted_answer:time | -1.31 | +0.20 | +0.19 | -0.10 | -0.79 | -0.96 | -0.42 |
| back_front:short:answer_one_word:clothing | -0.09 | -0.30 | -0.10 | -0.71 | -1.70 | -1.56 | -1.38 |
| back_front:short:answer_one_word:container | 0.49 | -0.67 | -0.62 | -0.11 | -0.69 | -2.20 | +0.12 |
| back_front:short:answer_one_word:furniture | 0.70 | -0.23 | -0.07 | +0.14 | -1.32 | -3.31 | -0.72 |
| back_front:short:answer_one_word:number | -0.57 | -0.49 | -0.55 | -0.00 | +0.20 | -2.48 | +0.55 |
| back_front:short:answer_one_word:plant | 2.97 | -4.07 | -3.96 | -0.70 | -5.51 | -2.91 | -1.30 |
| back_front:short:answer_one_word:time | -0.40 | +0.15 | +0.11 | +0.04 | -0.32 | -2.91 | -0.45 |
| back_front:short:label_colon:clothing | 2.73 | +1.06 | +1.38 | +0.14 | -0.11 | -1.85 | -1.08 |
| back_front:short:label_colon:container | 1.69 | +1.99 | +2.32 | -0.12 | +1.15 | -1.42 | -0.64 |
| back_front:short:label_colon:furniture | 1.27 | -0.26 | -0.27 | -0.04 | -0.82 | -1.46 | -0.74 |
| back_front:short:label_colon:number | 0.41 | +0.18 | +0.11 | +0.21 | +0.08 | -1.31 | -0.10 |
| back_front:short:label_colon:plant | 3.30 | -0.43 | -1.13 | -0.32 | -0.60 | -1.56 | -0.95 |
| back_front:short:label_colon:time | 1.37 | +0.55 | +0.30 | +0.39 | +0.12 | -1.15 | -0.54 |
| back_front:short:list_answer:clothing | -0.76 | -0.49 | -0.64 | -0.03 | -1.49 | -1.46 | -1.98 |
| back_front:short:list_answer:container | -0.72 | -0.81 | -0.73 | +0.43 | -1.08 | -1.61 | -1.36 |
| back_front:short:list_answer:furniture | -0.47 | -0.65 | -0.78 | +0.47 | -1.55 | -1.31 | -1.83 |
| back_front:short:list_answer:number | -0.87 | -1.20 | -1.31 | +0.07 | -1.10 | -0.97 | -1.22 |
| back_front:short:list_answer:plant | -0.06 | -0.79 | -0.94 | -0.30 | -1.80 | -1.32 | -1.97 |
| back_front:short:list_answer:time | -0.55 | -0.81 | -0.85 | -0.07 | -1.21 | -1.87 | -1.38 |
| back_front:short:multiple_choice:clothing | -4.75 | +1.96 | +2.02 | +1.61 | -1.99 | -3.65 | -4.06 |
| back_front:short:multiple_choice:container | 6.97 | +0.63 | +0.76 | +0.01 | -2.62 | -2.48 | -4.42 |
| back_front:short:multiple_choice:furniture | -6.89 | +1.52 | +1.75 | +0.42 | -2.15 | -2.69 | -4.16 |
| back_front:short:multiple_choice:number | 5.24 | +0.56 | +0.81 | -0.97 | -5.76 | +0.12 | -6.39 |
| back_front:short:multiple_choice:plant | 8.15 | -4.99 | -3.49 | -1.23 | -7.38 | +0.25 | -4.16 |
| back_front:short:multiple_choice:time | 2.66 | +1.34 | +1.26 | -0.28 | -3.20 | -2.08 | -4.78 |
| back_front:short:quoted_answer:clothing | -1.28 | -0.54 | -0.44 | +0.27 | -1.60 | +0.10 | +0.67 |
| back_front:short:quoted_answer:container | 0.69 | -0.38 | -0.45 | -0.17 | -1.86 | +0.04 | +0.76 |
| back_front:short:quoted_answer:furniture | 1.16 | +0.64 | +0.61 | -0.07 | -1.02 | -0.30 | +1.94 |
| back_front:short:quoted_answer:number | -0.64 | +0.76 | +0.68 | -0.03 | -0.04 | -0.33 | -0.49 |
| back_front:short:quoted_answer:plant | 1.15 | -1.86 | -1.90 | -0.01 | -3.18 | +0.43 | +1.64 |
| back_front:short:quoted_answer:time | -0.70 | -0.62 | -0.87 | -0.10 | -1.45 | -0.03 | +0.57 |
| front_back:long:answer_one_word:clothing | -1.65 | +0.52 | +0.67 | +0.07 | -0.66 | -0.92 | -1.18 |
| front_back:long:answer_one_word:container | -1.81 | +0.51 | +0.50 | -0.02 | -0.56 | -0.97 | -1.10 |
| front_back:long:answer_one_word:furniture | -1.65 | +0.52 | +0.57 | -0.21 | -0.57 | -1.95 | -1.09 |
| front_back:long:answer_one_word:number | -1.75 | +0.44 | +0.40 | +0.06 | -0.71 | -1.40 | -1.16 |
| front_back:long:answer_one_word:plant | -1.49 | +0.46 | +0.55 | -0.02 | -0.67 | -1.15 | -1.12 |
| front_back:long:answer_one_word:time | -1.60 | +0.47 | +0.50 | +0.05 | -0.43 | -3.02 | -0.91 |
| front_back:long:label_colon:clothing | 0.71 | +0.22 | +0.76 | +0.02 | -1.31 | -2.14 | -1.10 |
| front_back:long:label_colon:container | -1.03 | -0.73 | -0.85 | +0.05 | -1.22 | -1.69 | -0.31 |
| front_back:long:label_colon:furniture | 1.39 | -0.34 | +0.21 | +0.59 | -1.13 | -2.41 | -0.31 |
| front_back:long:label_colon:number | -1.29 | -2.18 | -2.14 | +0.16 | -2.26 | -2.31 | -0.79 |
| front_back:long:label_colon:plant | 1.25 | -0.55 | +0.04 | +0.25 | -1.38 | -1.83 | -0.87 |
| front_back:long:label_colon:time | -1.67 | -0.65 | -0.70 | +0.09 | -1.50 | -2.06 | -0.90 |
| front_back:long:list_answer:clothing | -0.66 | +0.10 | +0.34 | +0.05 | -1.29 | -1.25 | -1.63 |
| front_back:long:list_answer:container | -0.62 | -0.02 | +0.03 | +0.14 | -1.53 | -0.76 | -1.53 |
| front_back:long:list_answer:furniture | -0.67 | -0.13 | -0.13 | +0.09 | -1.57 | -0.35 | -1.57 |
| front_back:long:list_answer:number | -0.62 | -0.12 | -0.02 | -0.09 | -1.79 | -0.76 | -1.79 |
| front_back:long:list_answer:plant | -0.61 | +0.16 | +0.12 | +0.13 | -1.26 | -0.70 | -1.41 |
| front_back:long:list_answer:time | -0.48 | +0.01 | +0.01 | +0.09 | -1.01 | -1.67 | -1.26 |
| front_back:long:multiple_choice:clothing | -5.86 | -1.62 | -1.68 | -0.12 | -0.74 | +0.88 | -2.72 |
| front_back:long:multiple_choice:container | 4.44 | +0.57 | +0.34 | -0.11 | +0.86 | +0.97 | -2.64 |
| front_back:long:multiple_choice:furniture | -8.12 | -1.52 | -1.50 | +0.17 | -0.83 | +0.69 | -2.52 |
| front_back:long:multiple_choice:number | 4.34 | -4.21 | -6.31 | +0.01 | -4.17 | +2.96 | -3.42 |
| front_back:long:multiple_choice:plant | 6.33 | +1.90 | +1.56 | -0.70 | +1.39 | +0.06 | -2.51 |
| front_back:long:multiple_choice:time | 4.12 | -2.46 | -4.86 | -0.03 | -2.53 | +2.05 | -3.21 |
| front_back:long:quoted_answer:clothing | 0.52 | -1.57 | -0.85 | +0.11 | -1.33 | -0.81 | -0.12 |
| front_back:long:quoted_answer:container | -0.56 | +0.04 | +0.48 | +0.21 | -0.40 | -1.49 | +0.58 |
| front_back:long:quoted_answer:furniture | -0.70 | -0.14 | +0.26 | +0.02 | -0.42 | -1.76 | +0.27 |
| front_back:long:quoted_answer:number | -0.00 | -0.72 | -0.10 | +0.12 | -0.80 | -2.32 | +0.29 |
| front_back:long:quoted_answer:plant | -0.35 | -0.42 | +0.07 | +0.26 | -0.40 | -1.53 | +0.24 |
| front_back:long:quoted_answer:time | -0.00 | -0.74 | +0.22 | +0.19 | -0.41 | -1.32 | +0.39 |
| front_back:neutral:answer_one_word:clothing | -1.91 | +0.69 | +0.58 | +0.10 | -0.02 | -2.65 | -0.06 |
| front_back:neutral:answer_one_word:container | -2.03 | +0.73 | +0.64 | +0.22 | +0.07 | -2.50 | -0.13 |
| front_back:neutral:answer_one_word:furniture | -2.10 | +0.65 | +0.33 | +0.17 | -0.12 | -2.62 | -0.21 |
| front_back:neutral:answer_one_word:number | -1.96 | +0.68 | +0.62 | +0.10 | +0.59 | -2.83 | +0.46 |
| front_back:neutral:answer_one_word:plant | -1.81 | +0.75 | +0.83 | +0.25 | -0.28 | -2.88 | -0.73 |
| front_back:neutral:answer_one_word:time | -2.55 | +0.60 | +0.51 | +0.19 | -0.20 | -2.79 | -0.40 |
| front_back:neutral:label_colon:clothing | 0.47 | +0.84 | +1.13 | -0.06 | -0.25 | -1.79 | -0.74 |
| front_back:neutral:label_colon:container | -2.48 | -0.06 | +0.00 | +0.11 | -0.99 | -0.73 | -1.15 |
| front_back:neutral:label_colon:furniture | -0.68 | +1.04 | +1.41 | +0.16 | -0.17 | -1.10 | -1.36 |
| front_back:neutral:label_colon:number | -2.01 | -0.66 | -0.51 | +0.02 | -1.34 | -1.23 | -1.25 |
| front_back:neutral:label_colon:plant | 0.11 | +0.41 | +0.89 | -0.11 | -0.69 | -1.35 | -0.39 |
| front_back:neutral:label_colon:time | -1.11 | +0.07 | +0.08 | +0.03 | -0.86 | -1.88 | -1.55 |
| front_back:neutral:list_answer:clothing | -1.67 | -0.31 | -0.50 | +0.01 | -1.59 | -2.41 | -1.59 |
| front_back:neutral:list_answer:container | -1.82 | -0.35 | -0.48 | +0.38 | -1.37 | -2.12 | -1.37 |
| front_back:neutral:list_answer:furniture | -1.66 | -0.57 | -0.81 | +0.14 | -1.41 | -1.63 | -1.41 |
| front_back:neutral:list_answer:number | -1.60 | -0.24 | -0.39 | +0.04 | -1.14 | -2.01 | -1.14 |
| front_back:neutral:list_answer:plant | -1.15 | +0.04 | -0.08 | -0.01 | -0.93 | -2.40 | -1.66 |
| front_back:neutral:list_answer:time | -1.87 | -0.25 | -0.38 | +0.00 | -1.58 | -2.07 | -1.59 |
| front_back:neutral:multiple_choice:clothing | -5.74 | -0.82 | -0.12 | +0.36 | -2.43 | -1.61 | -3.56 |
| front_back:neutral:multiple_choice:container | 4.59 | +1.50 | +1.45 | -0.52 | -1.59 | -3.63 | -3.34 |
| front_back:neutral:multiple_choice:furniture | -6.48 | -0.73 | -0.05 | +0.92 | -2.37 | -1.64 | -3.54 |
| front_back:neutral:multiple_choice:number | 2.37 | -2.42 | -2.43 | -0.06 | -4.82 | -1.24 | -4.42 |
| front_back:neutral:multiple_choice:plant | 6.75 | -2.46 | -2.52 | -0.24 | -4.67 | -0.97 | -3.50 |
| front_back:neutral:multiple_choice:time | 5.57 | +0.91 | +0.26 | -0.87 | -3.05 | -5.50 | -4.64 |
| front_back:neutral:quoted_answer:clothing | -1.88 | +0.10 | +0.12 | +0.19 | -0.93 | -0.69 | -0.10 |
| front_back:neutral:quoted_answer:container | -1.29 | +0.53 | +0.81 | +0.28 | -0.37 | -0.77 | -0.37 |
| front_back:neutral:quoted_answer:furniture | -2.35 | +0.88 | +1.35 | -0.07 | -0.33 | -1.46 | -0.08 |
| front_back:neutral:quoted_answer:number | -2.59 | +1.30 | +0.98 | +0.15 | +0.04 | +0.02 | -0.01 |
| front_back:neutral:quoted_answer:plant | -2.39 | +0.27 | +0.18 | +0.25 | -1.17 | -0.79 | +0.12 |
| front_back:neutral:quoted_answer:time | -3.23 | +1.34 | +1.38 | +0.17 | -0.51 | -0.75 | -0.03 |
| front_back:short:answer_one_word:clothing | -0.30 | -0.14 | -0.06 | +0.05 | -1.07 | -1.92 | -0.96 |
| front_back:short:answer_one_word:container | -0.37 | -0.11 | -0.13 | -0.18 | -0.23 | -1.85 | +0.00 |
| front_back:short:answer_one_word:furniture | 0.75 | -0.76 | -0.58 | -0.07 | -1.61 | -2.26 | -0.82 |
| front_back:short:answer_one_word:number | -0.38 | -0.97 | -0.95 | -0.01 | -0.66 | -1.44 | +0.09 |
| front_back:short:answer_one_word:plant | 3.49 | -3.18 | -3.49 | -0.56 | -4.12 | -2.29 | -1.12 |
| front_back:short:answer_one_word:time | -0.72 | -0.18 | -0.52 | +0.05 | -0.72 | -3.59 | -0.83 |
| front_back:short:label_colon:clothing | 1.78 | -0.14 | +0.23 | -0.60 | -0.80 | -1.53 | -1.19 |
| front_back:short:label_colon:container | -0.06 | +2.40 | +3.10 | +0.06 | +1.41 | -1.30 | -1.01 |
| front_back:short:label_colon:furniture | 2.75 | -0.08 | +0.20 | -0.92 | -0.83 | -1.60 | -0.54 |
| front_back:short:label_colon:number | 0.11 | +0.43 | +0.87 | +0.08 | -0.04 | -0.33 | -0.87 |
| front_back:short:label_colon:plant | 3.62 | +0.60 | +0.60 | -0.44 | +0.58 | -1.46 | -0.97 |
| front_back:short:label_colon:time | 1.24 | +0.73 | +0.15 | -0.09 | +0.40 | -0.21 | -1.13 |
| front_back:short:list_answer:clothing | -0.60 | -0.44 | -0.65 | -0.14 | -0.71 | -2.05 | -0.89 |
| front_back:short:list_answer:container | -0.43 | -0.38 | -0.34 | +0.40 | -0.33 | -0.69 | -0.45 |
| front_back:short:list_answer:furniture | -0.48 | -0.49 | -0.67 | +0.04 | -0.85 | -0.71 | -0.85 |
| front_back:short:list_answer:number | -0.60 | -0.60 | -0.99 | -0.07 | -0.39 | -0.62 | -0.39 |
| front_back:short:list_answer:plant | 0.03 | -0.35 | -0.57 | -0.17 | -0.85 | -2.05 | -0.93 |
| front_back:short:list_answer:time | -0.27 | -0.37 | -0.68 | +0.27 | -0.44 | -1.85 | -0.58 |
| front_back:short:multiple_choice:clothing | -4.28 | +1.05 | +0.96 | +0.02 | -4.78 | -4.85 | -6.29 |
| front_back:short:multiple_choice:container | 7.68 | +0.12 | +0.38 | -0.83 | -4.84 | -2.72 | -6.34 |
| front_back:short:multiple_choice:furniture | -7.39 | +1.69 | +1.63 | +2.33 | -3.06 | -4.58 | -6.11 |
| front_back:short:multiple_choice:number | 3.43 | -2.67 | -2.93 | -1.02 | -10.00 | -4.57 | -7.63 |
| front_back:short:multiple_choice:plant | 8.05 | -3.68 | -2.00 | -1.28 | -7.80 | -3.01 | -6.61 |
| front_back:short:multiple_choice:time | 3.77 | +1.50 | +1.38 | -0.74 | -5.57 | -3.62 | -6.96 |
| front_back:short:quoted_answer:clothing | -4.44 | -0.05 | -0.70 | -0.04 | -1.14 | -0.27 | +1.05 |
| front_back:short:quoted_answer:container | -2.98 | +0.68 | +0.75 | +0.48 | -0.70 | -0.35 | +1.06 |
| front_back:short:quoted_answer:furniture | 3.48 | -0.03 | +0.29 | +0.19 | -1.06 | -1.15 | +1.91 |
| front_back:short:quoted_answer:number | -1.05 | +0.54 | +0.20 | +0.25 | -1.05 | -0.58 | -0.40 |
| front_back:short:quoted_answer:plant | 1.65 | -1.40 | -1.64 | -0.58 | -2.04 | -0.09 | +1.61 |
| front_back:short:quoted_answer:time | -1.66 | +0.72 | +0.74 | +0.49 | -1.65 | -0.54 | +1.01 |

## glm4

cases=180, attention=L39, mlp=L40, heads=32

### All / difficult / multiple-choice

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | -0.359 | +0.005 | -0.022 | +0.018 | -0.113 | +1.206 | -0.109 | -0.033 | +0.054 | +1.260 | -0.573 | +0.225 | 73.927 | 72.747 |
| difficult_formats | 144 | -0.684 | +0.026 | +0.009 | +0.044 | +0.191 | +1.818 | -0.015 | -0.173 | +0.320 | +1.790 | -0.605 | +0.090 | 80.949 | 79.852 |
| multiple_choice_control | 36 | 0.940 | -0.076 | -0.149 | -0.082 | -1.332 | -1.243 | -0.485 | +0.530 | -1.009 | -0.862 | -0.446 | +0.765 | 45.842 | 44.327 |

### By format

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | -0.685 | +0.183 | +0.221 | +0.024 | +0.416 | +0.919 | -0.433 | -0.080 | +0.424 | +0.900 | -0.447 | +0.113 | 73.873 | 72.375 |
| label_colon | 36 | 0.821 | +0.048 | +0.080 | +0.068 | +0.392 | +2.927 | -0.299 | -0.197 | +0.625 | +2.903 | -0.244 | +0.035 | 83.809 | 83.355 |
| list_answer | 36 | -1.482 | -0.079 | -0.132 | +0.035 | -0.096 | +2.078 | -0.338 | -0.969 | -0.000 | +1.930 | -0.646 | -0.633 | 80.182 | 80.351 |
| multiple_choice | 36 | 0.940 | -0.076 | -0.149 | -0.082 | -1.332 | -1.243 | -0.485 | +0.530 | -1.009 | -0.862 | -0.446 | +0.765 | 45.842 | 44.327 |
| quoted_answer | 36 | -1.387 | -0.048 | -0.132 | +0.047 | +0.053 | +1.347 | +1.010 | +0.552 | +0.231 | +1.429 | -1.083 | +0.846 | 85.931 | 83.326 |

### By category

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | -1.316 | -0.454 | -0.435 | +0.065 | -0.751 | +1.061 | +0.004 | -0.048 | -0.579 | +1.111 | -0.344 | +0.246 | 74.994 | 73.677 |
| container | 30 | 0.214 | -0.072 | -0.111 | -0.031 | -0.056 | +1.303 | -0.146 | -0.101 | +0.136 | +1.321 | -0.666 | +0.143 | 74.857 | 73.895 |
| furniture | 30 | -1.585 | -0.314 | -0.332 | +0.013 | -0.852 | +0.812 | -0.058 | -0.091 | -0.650 | +0.883 | -0.463 | +0.212 | 74.406 | 72.862 |
| number | 30 | -0.336 | +0.392 | +0.370 | +0.014 | +0.502 | +1.443 | -0.265 | -0.105 | +0.695 | +1.530 | -0.666 | +0.195 | 74.403 | 73.742 |
| plant | 30 | 0.982 | +0.551 | +0.522 | +0.043 | +0.542 | +1.003 | -0.033 | +0.132 | +0.649 | +1.048 | -0.697 | +0.312 | 72.141 | 70.542 |
| time | 30 | -0.112 | -0.070 | -0.147 | +0.006 | -0.064 | +1.611 | -0.155 | +0.016 | +0.073 | +1.665 | -0.603 | +0.245 | 72.764 | 71.763 |

### By family

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | -0.961 | -0.286 | -0.354 | +0.023 | -0.595 | +0.239 | +0.039 | +0.269 | -0.300 | +0.428 | -0.500 | +0.583 | 68.263 | 66.012 |
| neutral | 60 | -0.670 | +0.187 | +0.155 | -0.001 | +0.429 | +1.449 | +0.138 | +0.049 | +0.355 | +1.242 | -0.386 | +0.201 | 78.581 | 78.379 |
| short | 60 | 0.555 | +0.115 | +0.133 | +0.033 | -0.173 | +1.929 | -0.503 | -0.416 | +0.107 | +2.110 | -0.834 | -0.107 | 74.938 | 73.849 |

### By split

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | -0.378 | +0.145 | +0.099 | +0.021 | -0.019 | +1.306 | -0.094 | -0.023 | +0.134 | +1.349 | -0.529 | +0.250 | 75.011 | 73.850 |
| front_back | 90 | -0.339 | -0.134 | -0.144 | +0.016 | -0.208 | +1.105 | -0.123 | -0.042 | -0.026 | +1.171 | -0.618 | +0.201 | 72.844 | 71.644 |

### Cases

| case | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_genericΔ |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | -0.53 | -0.86 | -0.55 | -0.00 | -0.68 | -0.21 | -0.02 |
| back_front:long:answer_one_word:container | -0.41 | -0.20 | -0.10 | -0.09 | -0.45 | +0.20 | +0.05 |
| back_front:long:answer_one_word:furniture | -0.87 | -0.51 | -0.12 | -0.16 | -0.42 | -0.21 | -0.10 |
| back_front:long:answer_one_word:number | -1.22 | -0.03 | +0.09 | +0.27 | -0.13 | -0.50 | +0.20 |
| back_front:long:answer_one_word:plant | -1.03 | -0.05 | +0.02 | +0.12 | -0.26 | -0.50 | +0.05 |
| back_front:long:answer_one_word:time | -0.62 | -0.16 | +0.20 | +0.17 | -0.16 | -0.28 | +0.14 |
| back_front:long:label_colon:clothing | -0.21 | -0.97 | -0.89 | +0.32 | -1.52 | +1.33 | +0.66 |
| back_front:long:label_colon:container | -1.06 | +0.74 | +0.78 | -0.04 | +0.94 | +1.76 | +0.88 |
| back_front:long:label_colon:furniture | -0.98 | -0.84 | -0.94 | +0.12 | -1.19 | +1.71 | +0.86 |
| back_front:long:label_colon:number | 1.08 | +0.48 | +0.80 | +0.11 | +0.96 | +3.86 | +0.97 |
| back_front:long:label_colon:plant | 0.45 | +2.62 | +2.47 | -0.00 | +2.57 | +0.90 | +0.71 |
| back_front:long:label_colon:time | 0.30 | -0.09 | -0.17 | +0.12 | -0.58 | +1.47 | +0.93 |
| back_front:long:list_answer:clothing | -3.63 | -1.21 | -1.20 | +0.14 | -1.32 | -0.17 | -1.46 |
| back_front:long:list_answer:container | -0.72 | -0.39 | -0.50 | +0.00 | -0.94 | -0.23 | -1.43 |
| back_front:long:list_answer:furniture | -3.82 | -0.92 | -1.19 | +0.04 | -1.42 | -0.27 | -1.51 |
| back_front:long:list_answer:number | -2.38 | +0.09 | -0.09 | +0.16 | -1.05 | +0.96 | -1.05 |
| back_front:long:list_answer:plant | -3.03 | -0.47 | -0.77 | +0.10 | -0.71 | -0.52 | -1.41 |
| back_front:long:list_answer:time | -3.57 | +0.46 | +0.21 | -0.03 | -1.20 | -0.49 | -1.50 |
| back_front:long:multiple_choice:clothing | -5.17 | +3.23 | +2.98 | -0.28 | -0.04 | -3.95 | +0.63 |
| back_front:long:multiple_choice:container | 4.08 | -3.62 | -3.38 | -0.14 | -4.34 | -0.46 | +0.81 |
| back_front:long:multiple_choice:furniture | -5.98 | +4.19 | +3.74 | -0.13 | -0.03 | -4.49 | +0.58 |
| back_front:long:multiple_choice:number | 3.41 | -2.41 | -2.56 | -0.25 | -2.48 | -0.07 | +0.23 |
| back_front:long:multiple_choice:plant | 5.30 | -1.52 | -1.23 | +0.24 | -2.29 | -0.32 | +1.13 |
| back_front:long:multiple_choice:time | 6.08 | -1.37 | -1.72 | -0.09 | -1.75 | -1.04 | +0.59 |
| back_front:long:quoted_answer:clothing | -2.43 | -1.11 | -1.46 | -0.06 | -0.79 | +1.78 | +1.23 |
| back_front:long:quoted_answer:container | -0.39 | -0.73 | -0.95 | -0.04 | -0.49 | +2.09 | +1.16 |
| back_front:long:quoted_answer:furniture | -3.07 | -0.79 | -1.35 | +0.07 | -0.75 | +2.14 | +1.09 |
| back_front:long:quoted_answer:number | -1.62 | -0.22 | -0.75 | +0.02 | +0.94 | +1.12 | +1.16 |
| back_front:long:quoted_answer:plant | -2.18 | +0.32 | +0.15 | +0.10 | +0.21 | +1.12 | +1.12 |
| back_front:long:quoted_answer:time | -2.65 | -1.16 | -1.52 | +0.25 | +0.00 | +1.70 | +1.19 |
| back_front:neutral:answer_one_word:clothing | -0.88 | -0.35 | -0.43 | +0.14 | +0.86 | +1.40 | +0.27 |
| back_front:neutral:answer_one_word:container | -0.95 | +0.69 | +0.44 | -0.00 | +1.17 | +2.27 | -0.23 |
| back_front:neutral:answer_one_word:furniture | -0.91 | -0.14 | -0.27 | +0.00 | +0.64 | +1.46 | +0.10 |
| back_front:neutral:answer_one_word:number | -1.12 | +1.26 | +0.79 | -0.04 | +1.51 | +1.79 | +0.11 |
| back_front:neutral:answer_one_word:plant | -0.71 | +0.43 | +0.20 | +0.03 | +1.00 | -0.34 | +0.63 |
| back_front:neutral:answer_one_word:time | -1.09 | +0.02 | -0.30 | -0.18 | +0.64 | +2.71 | -0.19 |
| back_front:neutral:label_colon:clothing | 0.44 | -2.90 | -2.64 | +0.29 | -2.70 | +3.55 | -0.82 |
| back_front:neutral:label_colon:container | -0.68 | -0.36 | -0.35 | -0.07 | +0.22 | +2.84 | -0.57 |
| back_front:neutral:label_colon:furniture | -1.27 | -1.36 | -1.09 | +0.03 | -1.27 | +3.25 | -0.53 |
| back_front:neutral:label_colon:number | -1.66 | +2.05 | +2.44 | -0.04 | +1.96 | +4.13 | +0.04 |
| back_front:neutral:label_colon:plant | 1.34 | +2.10 | +2.02 | -0.01 | +2.21 | +2.96 | -0.41 |
| back_front:neutral:label_colon:time | -0.38 | +2.07 | +2.05 | -0.02 | +2.33 | +3.63 | -0.18 |
| back_front:neutral:list_answer:clothing | -1.52 | -1.67 | -1.49 | +0.14 | -0.50 | +4.32 | +0.02 |
| back_front:neutral:list_answer:container | -0.59 | +0.45 | +0.61 | -0.09 | +1.82 | +3.70 | -0.38 |
| back_front:neutral:list_answer:furniture | -1.36 | -1.10 | -0.88 | +0.03 | -0.48 | +3.56 | -0.11 |
| back_front:neutral:list_answer:number | -0.64 | +1.70 | +1.67 | -0.05 | +2.32 | +3.56 | -0.09 |
| back_front:neutral:list_answer:plant | -1.81 | +2.05 | +1.37 | +0.01 | +3.37 | +3.79 | +0.35 |
| back_front:neutral:list_answer:time | -0.89 | -1.77 | -1.63 | +0.05 | -0.10 | +4.10 | +0.18 |
| back_front:neutral:multiple_choice:clothing | -4.30 | +2.66 | +2.67 | -0.09 | +1.27 | -1.39 | -0.30 |
| back_front:neutral:multiple_choice:container | 2.02 | -0.87 | -0.56 | -0.16 | -2.16 | -1.30 | +0.23 |
| back_front:neutral:multiple_choice:furniture | -4.50 | +2.92 | +2.84 | -0.17 | +0.98 | -1.95 | -0.04 |
| back_front:neutral:multiple_choice:number | -0.73 | +1.34 | +1.04 | -0.07 | -0.41 | -1.74 | -0.18 |
| back_front:neutral:multiple_choice:plant | 3.65 | -2.34 | -2.28 | +0.09 | -2.30 | +0.00 | +0.26 |
| back_front:neutral:multiple_choice:time | 0.30 | +0.38 | +0.37 | -0.16 | -0.45 | -0.83 | -0.36 |
| back_front:neutral:quoted_answer:clothing | -1.88 | +0.34 | +0.02 | +0.14 | +0.35 | -0.19 | +0.89 |
| back_front:neutral:quoted_answer:container | 1.01 | +0.20 | +0.41 | +0.07 | +0.63 | +0.15 | +0.87 |
| back_front:neutral:quoted_answer:furniture | 0.32 | +0.17 | +0.37 | +0.02 | +0.35 | +1.15 | +0.95 |
| back_front:neutral:quoted_answer:number | -2.03 | +0.59 | +0.37 | +0.07 | +0.95 | -0.33 | +0.96 |
| back_front:neutral:quoted_answer:plant | -0.41 | +0.70 | +0.50 | -0.02 | +0.87 | -0.68 | +0.91 |
| back_front:neutral:quoted_answer:time | -1.78 | +0.41 | +0.25 | +0.07 | +0.47 | -0.08 | +1.02 |
| back_front:short:answer_one_word:clothing | -0.48 | +0.14 | +0.38 | +0.06 | -0.45 | +1.01 | -0.83 |
| back_front:short:answer_one_word:container | -0.23 | +2.20 | +2.09 | -0.05 | +2.48 | +2.23 | -0.83 |
| back_front:short:answer_one_word:furniture | -0.98 | -0.05 | +0.04 | -0.11 | -0.70 | +1.77 | -0.62 |
| back_front:short:answer_one_word:number | -1.19 | +1.23 | +1.11 | -0.06 | +0.63 | +3.46 | -0.57 |
| back_front:short:answer_one_word:plant | 0.30 | +3.30 | +3.11 | +0.17 | +2.91 | +0.89 | -0.41 |
| back_front:short:answer_one_word:time | -0.81 | +0.44 | +0.41 | +0.02 | +0.20 | +3.11 | +0.02 |
| back_front:short:label_colon:clothing | 4.40 | -1.23 | -1.29 | +0.14 | -1.44 | +4.14 | -1.13 |
| back_front:short:label_colon:container | 2.76 | +2.45 | +2.20 | -0.30 | +3.04 | +3.40 | -1.05 |
| back_front:short:label_colon:furniture | 3.04 | -3.13 | -3.05 | +0.28 | -2.85 | +3.82 | -0.91 |
| back_front:short:label_colon:number | 0.56 | -1.32 | -1.45 | +0.05 | +2.69 | +6.85 | -0.46 |
| back_front:short:label_colon:plant | 2.84 | +3.44 | +3.04 | +0.06 | +3.69 | +2.83 | -1.32 |
| back_front:short:label_colon:time | 1.90 | -1.71 | -1.56 | -0.11 | +0.82 | +6.47 | -0.98 |
| back_front:short:list_answer:clothing | 0.23 | -0.97 | -0.86 | +0.02 | -1.42 | +3.35 | -1.03 |
| back_front:short:list_answer:container | -0.73 | +0.16 | -0.05 | +0.05 | -0.09 | +2.88 | -1.60 |
| back_front:short:list_answer:furniture | -0.38 | -1.46 | -1.31 | +0.06 | -1.93 | +2.85 | -1.46 |
| back_front:short:list_answer:number | -0.43 | +2.48 | +2.72 | -0.03 | +1.62 | +1.52 | -1.55 |
| back_front:short:list_answer:plant | 0.94 | +3.16 | +2.97 | -0.05 | +3.16 | +3.31 | -1.07 |
| back_front:short:list_answer:time | -0.83 | +0.19 | +0.05 | +0.09 | -0.29 | +2.90 | -0.94 |
| back_front:short:multiple_choice:clothing | -3.27 | +0.73 | +0.27 | -0.22 | -1.55 | -2.55 | +0.89 |
| back_front:short:multiple_choice:container | 5.79 | -2.66 | -2.67 | -0.06 | -3.47 | -0.81 | +0.85 |
| back_front:short:multiple_choice:furniture | -4.47 | +2.56 | +2.08 | -0.16 | -1.32 | -3.95 | +1.29 |
| back_front:short:multiple_choice:number | 3.95 | +0.62 | +0.98 | -0.06 | -0.73 | -1.44 | +0.30 |
| back_front:short:multiple_choice:plant | 5.02 | -2.24 | -1.83 | +0.34 | -3.51 | +0.21 | +1.32 |
| back_front:short:multiple_choice:time | 6.33 | -0.36 | -0.42 | -0.02 | -1.30 | -0.92 | +0.45 |
| back_front:short:quoted_answer:clothing | -1.92 | -0.45 | -0.26 | +0.10 | -1.12 | +3.01 | -0.10 |
| back_front:short:quoted_answer:container | -1.34 | +0.75 | +0.62 | +0.02 | +0.88 | +3.20 | -0.28 |
| back_front:short:quoted_answer:furniture | -2.24 | -1.01 | -0.98 | +0.24 | -0.99 | +3.48 | -0.28 |
| back_front:short:quoted_answer:number | -0.61 | +0.55 | +0.72 | +0.12 | +1.03 | +1.51 | -0.81 |
| back_front:short:quoted_answer:plant | -0.91 | +1.79 | +1.99 | +0.07 | +1.38 | +1.80 | -0.21 |
| back_front:short:quoted_answer:time | -1.41 | +1.28 | +1.41 | +0.12 | +0.80 | +3.32 | -0.07 |
| front_back:long:answer_one_word:clothing | 0.25 | -1.36 | -0.80 | +0.25 | -0.75 | -0.46 | +0.23 |
| front_back:long:answer_one_word:container | -0.45 | -0.83 | -0.64 | -0.09 | -0.02 | -0.11 | +0.36 |
| front_back:long:answer_one_word:furniture | -0.84 | -0.33 | -0.02 | -0.09 | -0.21 | -0.82 | +0.09 |
| front_back:long:answer_one_word:number | -0.79 | -0.29 | +0.23 | +0.44 | +0.32 | -0.38 | +0.10 |
| front_back:long:answer_one_word:plant | -0.57 | -0.46 | -0.15 | -0.00 | +0.10 | -0.57 | +0.22 |
| front_back:long:answer_one_word:time | -0.77 | -0.29 | +0.12 | -0.14 | +0.07 | -0.24 | +0.46 |
| front_back:long:label_colon:clothing | 0.15 | -0.50 | -0.45 | +0.27 | -1.10 | +1.54 | +0.46 |
| front_back:long:label_colon:container | -1.49 | +1.57 | +1.55 | +0.16 | +1.65 | +1.19 | +1.04 |
| front_back:long:label_colon:furniture | 0.14 | -1.63 | -1.80 | +0.23 | -1.86 | +1.01 | +0.50 |
| front_back:long:label_colon:number | 0.69 | +0.48 | +0.82 | +0.09 | +0.40 | +2.72 | +0.70 |
| front_back:long:label_colon:plant | 0.68 | +2.61 | +2.55 | -0.12 | +2.42 | +0.70 | +0.92 |
| front_back:long:label_colon:time | -0.88 | -0.23 | -0.22 | +0.01 | -0.30 | +2.53 | +1.19 |
| front_back:long:list_answer:clothing | -3.96 | -1.45 | -1.50 | +0.18 | -1.53 | -0.02 | -1.51 |
| front_back:long:list_answer:container | -3.15 | -0.74 | -1.04 | -0.03 | -1.12 | -0.15 | -1.62 |
| front_back:long:list_answer:furniture | -3.80 | -1.21 | -1.41 | +0.11 | -1.56 | -0.33 | -1.62 |
| front_back:long:list_answer:number | -3.36 | +0.28 | +0.18 | +0.09 | -0.58 | -0.06 | -1.29 |
| front_back:long:list_answer:plant | -2.34 | -0.67 | -0.85 | +0.05 | -0.68 | -0.27 | -1.45 |
| front_back:long:list_answer:time | -3.34 | +0.20 | -0.03 | +0.03 | -1.46 | -0.25 | -1.58 |
| front_back:long:multiple_choice:clothing | -5.05 | +2.61 | +2.24 | -0.13 | -0.01 | -3.67 | +0.52 |
| front_back:long:multiple_choice:container | 3.58 | -3.95 | -4.34 | +0.03 | -4.47 | +0.43 | +0.49 |
| front_back:long:multiple_choice:furniture | -5.93 | +3.92 | +3.39 | -0.30 | -0.08 | -4.48 | +0.52 |
| front_back:long:multiple_choice:number | 1.09 | -1.31 | -1.14 | -0.29 | -1.87 | -0.49 | +0.09 |
| front_back:long:multiple_choice:plant | 5.96 | -2.02 | -1.73 | +0.09 | -2.45 | +0.89 | +1.14 |
| front_back:long:multiple_choice:time | 4.28 | -0.28 | -1.06 | +0.02 | -0.60 | -1.50 | +0.43 |
| front_back:long:quoted_answer:clothing | -2.11 | -1.57 | -1.80 | +0.16 | -1.14 | +1.41 | +1.32 |
| front_back:long:quoted_answer:container | -1.11 | -0.06 | -0.23 | -0.22 | +0.12 | +1.77 | +1.43 |
| front_back:long:quoted_answer:furniture | -2.87 | -0.68 | -1.09 | -0.09 | -0.49 | +1.52 | +1.17 |
| front_back:long:quoted_answer:number | -1.68 | +0.27 | -0.11 | -0.20 | +1.47 | +0.41 | +1.20 |
| front_back:long:quoted_answer:plant | -0.46 | -0.81 | -0.70 | +0.01 | -0.21 | +1.85 | +1.47 |
| front_back:long:quoted_answer:time | -2.67 | -0.94 | -1.23 | -0.21 | -0.42 | +1.72 | +1.30 |
| front_back:neutral:answer_one_word:clothing | -0.90 | -0.16 | -0.15 | +0.13 | +1.04 | +0.61 | +0.42 |
| front_back:neutral:answer_one_word:container | -0.72 | -0.10 | -0.11 | +0.08 | +0.99 | +1.41 | +0.18 |
| front_back:neutral:answer_one_word:furniture | -0.50 | -0.48 | -0.38 | -0.10 | +0.48 | +0.51 | -0.11 |
| front_back:neutral:answer_one_word:number | -0.82 | +0.23 | -0.18 | -0.15 | +1.14 | +1.27 | +0.12 |
| front_back:neutral:answer_one_word:plant | -0.51 | -0.27 | -0.21 | -0.02 | +0.50 | -0.13 | +0.59 |
| front_back:neutral:answer_one_word:time | -1.36 | +0.11 | -0.18 | -0.16 | +1.09 | +1.57 | +0.13 |
| front_back:neutral:label_colon:clothing | 0.74 | -1.68 | -1.38 | +0.24 | -1.47 | +3.08 | -0.62 |
| front_back:neutral:label_colon:container | -1.53 | -0.44 | -0.40 | -0.01 | +0.18 | +2.73 | -0.54 |
| front_back:neutral:label_colon:furniture | 0.37 | -2.88 | -2.80 | +0.16 | -2.38 | +1.86 | -0.32 |
| front_back:neutral:label_colon:number | -0.52 | +0.46 | +0.42 | +0.04 | +0.63 | +3.03 | -0.51 |
| front_back:neutral:label_colon:plant | 2.09 | +1.45 | +1.50 | -0.13 | +1.64 | +2.52 | -0.51 |
| front_back:neutral:label_colon:time | 1.05 | +1.23 | +1.39 | +0.09 | +1.02 | +2.87 | -0.46 |
| front_back:neutral:list_answer:clothing | -1.47 | -1.43 | -1.09 | +0.01 | -0.68 | +4.23 | +0.36 |
| front_back:neutral:list_answer:container | -1.55 | +0.52 | +0.57 | +0.02 | +1.52 | +3.55 | -0.07 |
| front_back:neutral:list_answer:furniture | -0.74 | -1.68 | -1.11 | +0.16 | -0.64 | +3.70 | -0.20 |
| front_back:neutral:list_answer:number | -1.20 | +1.61 | +1.31 | +0.02 | +1.89 | +2.82 | -0.57 |
| front_back:neutral:list_answer:plant | -2.28 | +1.32 | +0.87 | +0.03 | +2.01 | +3.24 | +0.30 |
| front_back:neutral:list_answer:time | -3.09 | +0.38 | +0.12 | +0.07 | +0.72 | +3.58 | -0.14 |
| front_back:neutral:multiple_choice:clothing | -4.20 | +0.61 | +0.60 | -0.30 | +0.54 | -0.07 | -0.03 |
| front_back:neutral:multiple_choice:container | 1.71 | -0.80 | -0.94 | -0.08 | -1.73 | -0.92 | -0.21 |
| front_back:neutral:multiple_choice:furniture | -4.49 | +1.71 | +1.72 | -0.12 | +0.65 | -1.06 | -0.88 |
| front_back:neutral:multiple_choice:number | 0.87 | -0.19 | -0.73 | -0.19 | -0.25 | +0.15 | -0.46 |
| front_back:neutral:multiple_choice:plant | 5.78 | -1.70 | -1.31 | +0.00 | -1.64 | +0.27 | +0.59 |
| front_back:neutral:multiple_choice:time | 0.96 | +1.20 | +1.07 | -0.02 | +0.59 | -0.75 | -0.23 |
| front_back:neutral:quoted_answer:clothing | -1.71 | -0.13 | -0.18 | +0.11 | -0.20 | +0.37 | +0.65 |
| front_back:neutral:quoted_answer:container | 0.36 | +0.53 | +0.59 | -0.02 | +0.58 | +0.12 | +0.52 |
| front_back:neutral:quoted_answer:furniture | 0.67 | +0.41 | +0.68 | +0.05 | +0.64 | +1.11 | +0.34 |
| front_back:neutral:quoted_answer:number | -1.69 | +0.51 | +0.41 | +0.03 | +0.49 | -0.40 | +0.39 |
| front_back:neutral:quoted_answer:plant | -0.00 | +0.43 | +0.19 | +0.01 | +0.61 | -0.40 | +0.43 |
| front_back:neutral:quoted_answer:time | -2.54 | +0.78 | +0.52 | +0.00 | +0.19 | +0.60 | +0.33 |
| front_back:short:answer_one_word:clothing | -0.38 | -0.29 | -0.02 | +0.15 | -0.88 | +1.13 | -0.54 |
| front_back:short:answer_one_word:container | -1.42 | +2.09 | +1.89 | -0.04 | +1.89 | +0.88 | -0.97 |
| front_back:short:answer_one_word:furniture | 0.15 | -0.48 | -0.45 | +0.05 | -1.09 | +1.89 | -0.77 |
| front_back:short:answer_one_word:number | -1.28 | -0.34 | -0.42 | +0.03 | -0.57 | +3.02 | -0.62 |
| front_back:short:answer_one_word:plant | 0.47 | +2.59 | +2.44 | +0.25 | +2.20 | -0.02 | -0.05 |
| front_back:short:answer_one_word:time | -0.52 | -0.13 | -0.11 | -0.02 | -0.11 | +3.24 | -0.49 |
| front_back:short:label_colon:clothing | 4.09 | -1.04 | -1.05 | +0.30 | -1.00 | +3.95 | -1.03 |
| front_back:short:label_colon:container | 0.70 | +3.15 | +3.30 | +0.17 | +2.92 | +1.63 | -1.19 |
| front_back:short:label_colon:furniture | 3.73 | -2.61 | -2.58 | +0.10 | -2.47 | +2.88 | -0.79 |
| front_back:short:label_colon:number | 0.86 | -0.75 | -0.88 | -0.05 | +1.62 | +5.22 | -0.69 |
| front_back:short:label_colon:plant | 3.62 | +1.95 | +1.89 | -0.15 | +1.90 | +1.83 | -1.42 |
| front_back:short:label_colon:time | 2.21 | -1.45 | -1.34 | +0.11 | +0.44 | +5.17 | -0.53 |
| front_back:short:list_answer:clothing | 0.71 | -0.96 | -0.64 | -0.02 | -1.33 | +3.04 | -1.44 |
| front_back:short:list_answer:container | -0.45 | -0.36 | -0.57 | +0.09 | -0.90 | +2.69 | -1.78 |
| front_back:short:list_answer:furniture | 0.57 | -1.76 | -1.45 | -0.05 | -2.29 | +2.25 | -1.66 |
| front_back:short:list_answer:number | -0.49 | +1.56 | +1.60 | +0.07 | +1.05 | +1.89 | -1.79 |
| front_back:short:list_answer:plant | 2.00 | +1.67 | +1.84 | -0.16 | +1.86 | +2.81 | -1.50 |
| front_back:short:list_answer:time | -0.28 | -0.96 | -1.16 | -0.02 | -0.57 | +2.95 | -1.27 |
| front_back:short:multiple_choice:clothing | -3.34 | +0.21 | -0.36 | -0.33 | -1.05 | -1.26 | +1.32 |
| front_back:short:multiple_choice:container | 5.70 | -1.77 | -1.64 | -0.12 | -2.40 | -0.51 | +1.07 |
| front_back:short:multiple_choice:furniture | -4.87 | +0.77 | +0.43 | -0.09 | -1.18 | -1.95 | +1.91 |
| front_back:short:multiple_choice:number | 3.76 | +0.21 | +0.97 | +0.14 | -0.84 | -1.37 | +1.23 |
| front_back:short:multiple_choice:plant | 5.72 | -2.61 | -2.45 | +0.23 | -3.80 | -0.01 | +1.91 |
| front_back:short:multiple_choice:time | 4.78 | -0.62 | -0.40 | -0.09 | -1.46 | -1.43 | +0.97 |
| front_back:short:quoted_answer:clothing | -1.16 | -1.89 | -1.70 | +0.09 | -1.91 | +2.54 | -0.45 |
| front_back:short:quoted_answer:container | -2.30 | +0.20 | +0.07 | +0.05 | -0.13 | +2.46 | -0.22 |
| front_back:short:quoted_answer:furniture | -1.70 | -1.05 | -0.98 | +0.20 | -1.71 | +1.98 | -0.21 |
| front_back:short:quoted_answer:number | -0.87 | +0.63 | +0.76 | +0.12 | +0.34 | +0.77 | -0.30 |
| front_back:short:quoted_answer:plant | -0.46 | -0.22 | +0.04 | -0.03 | -0.49 | +1.92 | -0.36 |
| front_back:short:quoted_answer:time | -2.09 | +0.25 | +0.47 | +0.24 | -0.57 | +2.48 | +0.08 |

## deepseek7b

cases=180, attention=L28, mlp=L28, heads=28

### All / difficult / multiple-choice

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | -1.488 | -0.826 | -0.586 | +0.207 | -2.541 | -2.569 | -2.149 | -1.590 | -4.363 | -3.447 | -3.999 | -3.303 | 131.618 | 145.121 |
| difficult_formats | 144 | -1.611 | -0.721 | -0.393 | +0.251 | -2.182 | -2.509 | -1.831 | -1.114 | -3.911 | -3.042 | -3.757 | -3.004 | 134.750 | 147.393 |
| multiple_choice_control | 36 | -0.995 | -1.242 | -1.360 | +0.031 | -3.977 | -2.809 | -3.420 | -3.495 | -6.169 | -5.068 | -4.969 | -4.496 | 119.088 | 136.036 |

### By format

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | -2.096 | -0.966 | -0.625 | +0.201 | -4.422 | -5.052 | -4.745 | -3.494 | -7.034 | -6.039 | -8.031 | -6.012 | 162.571 | 182.244 |
| label_colon | 36 | -0.771 | -1.451 | -1.239 | +0.276 | -1.778 | -1.905 | -0.233 | -0.483 | -3.328 | -2.319 | -2.105 | -1.895 | 131.602 | 141.668 |
| list_answer | 36 | -1.743 | -0.717 | -0.419 | +0.136 | -3.655 | -3.643 | -4.272 | -3.707 | -5.693 | -3.790 | -6.150 | -5.932 | 134.857 | 155.653 |
| multiple_choice | 36 | -0.995 | -1.242 | -1.360 | +0.031 | -3.977 | -2.809 | -3.420 | -3.495 | -6.169 | -5.068 | -4.969 | -4.496 | 119.088 | 136.036 |
| quoted_answer | 36 | -1.835 | +0.247 | +0.713 | +0.392 | +1.128 | +0.563 | +1.925 | +3.228 | +0.410 | -0.021 | +1.258 | +1.821 | 109.970 | 110.005 |

### By category

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | -2.618 | -0.992 | -0.377 | +0.275 | -2.529 | -2.310 | -1.829 | -1.345 | -3.831 | -3.040 | -3.477 | -2.714 | 132.265 | 145.197 |
| container | 30 | -0.979 | -0.114 | -0.206 | +0.140 | -2.120 | -2.692 | -2.293 | -1.698 | -4.470 | -3.834 | -4.383 | -3.682 | 132.028 | 146.077 |
| furniture | 30 | -2.867 | -0.897 | -0.325 | +0.341 | -2.494 | -2.463 | -1.939 | -1.427 | -3.864 | -3.284 | -3.724 | -2.922 | 130.356 | 143.318 |
| number | 30 | -1.121 | -1.088 | -0.838 | +0.221 | -3.001 | -2.577 | -2.526 | -1.921 | -5.196 | -3.489 | -4.704 | -3.807 | 133.279 | 148.673 |
| plant | 30 | -0.096 | -0.671 | -0.962 | +0.077 | -2.182 | -2.698 | -2.055 | -1.485 | -4.163 | -3.279 | -3.558 | -3.094 | 128.969 | 140.807 |
| time | 30 | -1.246 | -1.192 | -0.808 | +0.189 | -2.920 | -2.674 | -2.251 | -1.666 | -4.652 | -3.758 | -4.152 | -3.598 | 132.808 | 146.656 |

### By family

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | -2.273 | -0.399 | -0.155 | +0.348 | -0.893 | -0.596 | -0.431 | +0.076 | -1.754 | -1.014 | -1.155 | -0.654 | 84.112 | 95.072 |
| neutral | 60 | -1.618 | -1.072 | -0.591 | +0.201 | -3.601 | -3.078 | -3.695 | -2.672 | -5.880 | -4.417 | -6.399 | -4.897 | 163.290 | 175.789 |
| short | 60 | -0.572 | -1.005 | -1.012 | +0.072 | -3.128 | -4.034 | -2.321 | -2.174 | -5.455 | -4.910 | -4.444 | -4.356 | 147.451 | 164.504 |

### By split

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | -1.437 | -0.651 | -0.450 | +0.188 | -2.384 | -2.397 | -2.367 | -1.432 | -4.276 | -3.283 | -4.294 | -3.102 | 133.494 | 147.695 |
| front_back | 90 | -1.539 | -1.000 | -0.722 | +0.226 | -2.698 | -2.741 | -1.931 | -1.749 | -4.450 | -3.612 | -3.705 | -3.503 | 129.741 | 142.548 |

### Cases

| case | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_genericΔ |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | -3.62 | +0.39 | +0.31 | -0.12 | +0.52 | -0.81 | +1.27 |
| back_front:long:answer_one_word:container | -1.55 | -0.12 | -0.09 | +0.13 | -0.62 | -1.34 | +0.26 |
| back_front:long:answer_one_word:furniture | -1.90 | -1.65 | -1.17 | +0.23 | -1.63 | +0.78 | +0.95 |
| back_front:long:answer_one_word:number | -3.08 | +0.30 | -0.41 | +0.48 | -0.46 | -0.46 | -0.12 |
| back_front:long:answer_one_word:plant | -1.41 | -2.23 | -2.28 | +0.42 | -3.02 | -0.67 | -0.07 |
| back_front:long:answer_one_word:time | -3.27 | +0.03 | -0.16 | +0.34 | -1.56 | -1.89 | -1.52 |
| back_front:long:label_colon:clothing | -2.59 | +0.22 | +1.59 | +0.29 | -2.03 | +0.21 | +0.08 |
| back_front:long:label_colon:container | -1.83 | +1.93 | +1.91 | +0.81 | +0.73 | -0.55 | -1.07 |
| back_front:long:label_colon:furniture | -2.45 | +0.48 | +1.70 | +0.88 | -1.73 | -0.25 | -0.88 |
| back_front:long:label_colon:number | -0.02 | +1.41 | +1.88 | +0.38 | -1.11 | -0.22 | -1.57 |
| back_front:long:label_colon:plant | -0.81 | +1.07 | +0.79 | +0.27 | -0.34 | -0.14 | -0.92 |
| back_front:long:label_colon:time | -3.77 | +2.09 | +2.62 | +0.70 | -0.12 | -0.41 | -1.35 |
| back_front:long:list_answer:clothing | -2.99 | -1.66 | -1.29 | +0.09 | -2.59 | -0.55 | -2.79 |
| back_front:long:list_answer:container | -1.20 | -0.77 | -0.19 | +0.21 | -1.70 | -1.57 | -3.00 |
| back_front:long:list_answer:furniture | -3.89 | -2.33 | -1.59 | +0.98 | -2.79 | -0.19 | -2.77 |
| back_front:long:list_answer:number | -2.58 | -0.03 | -0.51 | +0.52 | -3.27 | -0.14 | -2.04 |
| back_front:long:list_answer:plant | -3.00 | -1.27 | -1.16 | +0.63 | -2.69 | -1.32 | -3.07 |
| back_front:long:list_answer:time | -3.32 | -0.59 | +1.36 | +0.39 | -2.11 | -2.07 | -3.79 |
| back_front:long:multiple_choice:clothing | -5.80 | -0.59 | +0.80 | +0.24 | +1.08 | +1.67 | +0.67 |
| back_front:long:multiple_choice:container | 4.04 | +1.39 | -3.20 | -1.80 | +1.61 | -0.33 | +0.29 |
| back_front:long:multiple_choice:furniture | -6.07 | -1.84 | -0.57 | +0.25 | +0.09 | +1.92 | +0.21 |
| back_front:long:multiple_choice:number | 0.57 | -4.91 | -5.66 | +0.24 | -2.12 | +2.78 | +0.27 |
| back_front:long:multiple_choice:plant | 2.11 | -2.64 | -5.01 | -0.83 | -1.38 | +1.73 | +0.24 |
| back_front:long:multiple_choice:time | 1.60 | -4.86 | -4.82 | -0.04 | -2.95 | +1.95 | +0.89 |
| back_front:long:quoted_answer:clothing | -4.10 | +0.78 | +1.78 | +0.84 | +4.29 | -1.36 | +7.62 |
| back_front:long:quoted_answer:container | -1.24 | -1.49 | -1.79 | +0.31 | -0.16 | -1.77 | +5.62 |
| back_front:long:quoted_answer:furniture | -3.35 | +0.08 | +1.11 | +0.43 | +2.96 | -1.05 | +6.50 |
| back_front:long:quoted_answer:number | -1.04 | -1.74 | -0.97 | +0.20 | +2.04 | -1.35 | +6.31 |
| back_front:long:quoted_answer:plant | -2.11 | -0.87 | -0.70 | +0.45 | +0.82 | -2.23 | +6.25 |
| back_front:long:quoted_answer:time | -2.25 | -1.23 | -0.53 | +0.58 | +0.83 | -2.35 | +5.57 |
| back_front:neutral:answer_one_word:clothing | -2.78 | -2.44 | -1.59 | +0.05 | -5.04 | -4.25 | -3.64 |
| back_front:neutral:answer_one_word:container | -2.20 | -0.31 | +0.64 | +0.09 | -4.49 | -4.46 | -3.48 |
| back_front:neutral:answer_one_word:furniture | -2.27 | -3.00 | -3.10 | +0.19 | -5.17 | -3.85 | -3.39 |
| back_front:neutral:answer_one_word:number | -1.86 | +0.10 | +0.89 | +0.15 | -4.48 | -4.76 | -3.57 |
| back_front:neutral:answer_one_word:plant | -1.57 | -1.21 | -0.57 | +0.28 | -2.86 | -5.94 | -3.21 |
| back_front:neutral:answer_one_word:time | -2.11 | -1.13 | +1.11 | +0.09 | -5.15 | -5.22 | -4.24 |
| back_front:neutral:label_colon:clothing | -2.22 | +0.55 | +2.17 | +0.14 | -2.94 | -2.86 | -2.50 |
| back_front:neutral:label_colon:container | -3.55 | +1.35 | +2.16 | +0.09 | -3.79 | -4.09 | -3.48 |
| back_front:neutral:label_colon:furniture | -3.25 | +0.72 | +1.32 | +0.02 | -3.62 | -3.90 | -3.02 |
| back_front:neutral:label_colon:number | -1.91 | -1.89 | +0.39 | +0.57 | -5.48 | -4.70 | -2.72 |
| back_front:neutral:label_colon:plant | -1.45 | +1.01 | +0.40 | +1.27 | -3.76 | -1.88 | -3.12 |
| back_front:neutral:label_colon:time | -2.95 | +0.08 | +0.51 | +0.62 | -3.79 | -2.20 | -2.10 |
| back_front:neutral:list_answer:clothing | -1.16 | -1.06 | -0.82 | +0.00 | -3.66 | -2.20 | -2.50 |
| back_front:neutral:list_answer:container | -1.22 | +0.90 | +0.71 | -0.17 | -4.04 | -3.65 | -4.06 |
| back_front:neutral:list_answer:furniture | -1.57 | -0.89 | -0.10 | -0.19 | -3.50 | -2.54 | -1.99 |
| back_front:neutral:list_answer:number | -1.48 | -0.35 | +0.03 | +0.00 | -4.55 | -4.38 | -3.63 |
| back_front:neutral:list_answer:plant | -0.55 | +1.32 | +0.17 | -0.12 | -1.93 | -3.16 | -3.29 |
| back_front:neutral:list_answer:time | -1.71 | -0.39 | +0.10 | -0.05 | -4.50 | -4.71 | -3.56 |
| back_front:neutral:multiple_choice:clothing | -5.16 | -0.84 | -0.05 | -0.12 | -5.30 | -4.47 | -5.08 |
| back_front:neutral:multiple_choice:container | 0.55 | +2.48 | +3.75 | +0.76 | -2.08 | -4.55 | -5.03 |
| back_front:neutral:multiple_choice:furniture | -4.41 | -1.11 | +0.51 | +0.33 | -5.33 | -3.97 | -5.02 |
| back_front:neutral:multiple_choice:number | 0.25 | -2.95 | -3.59 | +0.00 | -6.15 | -2.55 | -5.96 |
| back_front:neutral:multiple_choice:plant | -0.06 | -1.21 | -4.13 | +1.12 | -3.28 | -3.65 | -5.21 |
| back_front:neutral:multiple_choice:time | -1.79 | -2.48 | -2.38 | +0.50 | -5.83 | -3.35 | -4.82 |
| back_front:neutral:quoted_answer:clothing | -2.28 | -0.86 | -0.18 | +0.51 | +0.98 | +1.04 | +2.90 |
| back_front:neutral:quoted_answer:container | -0.66 | -0.19 | +0.14 | +0.09 | +1.68 | +1.37 | +3.55 |
| back_front:neutral:quoted_answer:furniture | -1.43 | -1.02 | -0.79 | +0.30 | +0.23 | +1.30 | +3.20 |
| back_front:neutral:quoted_answer:number | 0.98 | -0.08 | -0.05 | -0.13 | +0.84 | +0.26 | +2.31 |
| back_front:neutral:quoted_answer:plant | 0.12 | +0.48 | +0.92 | -0.02 | +1.66 | +0.48 | +2.93 |
| back_front:neutral:quoted_answer:time | -0.13 | -1.46 | -1.36 | -0.04 | -0.20 | +0.59 | +2.62 |
| back_front:short:answer_one_word:clothing | -1.74 | -1.67 | -0.96 | +0.09 | -7.23 | -6.82 | -7.12 |
| back_front:short:answer_one_word:container | -1.13 | +0.59 | +0.62 | +0.16 | -6.27 | -8.13 | -6.02 |
| back_front:short:answer_one_word:furniture | -1.33 | -0.48 | -0.51 | +0.20 | -5.81 | -6.83 | -5.02 |
| back_front:short:answer_one_word:number | -2.06 | -0.46 | +0.46 | +0.16 | -5.66 | -6.82 | -6.44 |
| back_front:short:answer_one_word:plant | -1.44 | +0.19 | +0.39 | +0.06 | -5.66 | -6.74 | -5.66 |
| back_front:short:answer_one_word:time | -0.99 | -0.93 | -1.63 | +0.17 | -6.52 | -7.22 | -6.05 |
| back_front:short:label_colon:clothing | 2.72 | -5.73 | -6.25 | +0.21 | -4.05 | -1.74 | +0.50 |
| back_front:short:label_colon:container | 0.26 | -2.36 | -3.47 | -0.07 | +0.19 | -1.94 | +0.52 |
| back_front:short:label_colon:furniture | 0.55 | -3.85 | -4.66 | +0.80 | -1.98 | -2.64 | +1.03 |
| back_front:short:label_colon:number | 1.76 | +0.26 | -0.20 | -0.16 | -0.73 | -2.80 | -3.48 |
| back_front:short:label_colon:plant | 4.56 | -2.68 | -5.00 | -0.92 | -0.29 | -1.57 | +0.06 |
| back_front:short:label_colon:time | 1.88 | -4.06 | -4.06 | +0.00 | -3.27 | -4.63 | -0.50 |
| back_front:short:list_answer:clothing | -0.94 | -0.22 | -0.52 | +0.00 | -4.09 | -4.12 | -4.06 |
| back_front:short:list_answer:container | -1.03 | +0.00 | +0.50 | +0.03 | -3.56 | -6.06 | -4.41 |
| back_front:short:list_answer:furniture | -0.84 | -0.88 | -0.62 | +0.02 | -3.87 | -5.59 | -3.99 |
| back_front:short:list_answer:number | -1.84 | -1.87 | -0.04 | +0.15 | -3.44 | -5.52 | -3.75 |
| back_front:short:list_answer:plant | -0.88 | +0.09 | +0.87 | +0.02 | -2.01 | -5.55 | -2.28 |
| back_front:short:list_answer:time | -0.81 | -0.32 | -0.08 | -0.17 | -4.20 | -6.03 | -4.21 |
| back_front:short:multiple_choice:clothing | -5.81 | +0.62 | +1.76 | +0.49 | -5.38 | -6.00 | -4.86 |
| back_front:short:multiple_choice:container | 4.44 | -0.53 | -2.90 | -0.05 | -6.59 | -4.25 | -5.20 |
| back_front:short:multiple_choice:furniture | -7.09 | +0.75 | +2.31 | +0.73 | -5.99 | -6.74 | -5.27 |
| back_front:short:multiple_choice:number | 0.09 | -1.73 | -2.63 | -0.05 | -7.27 | -5.98 | -4.88 |
| back_front:short:multiple_choice:plant | 2.16 | +0.21 | +0.77 | -1.16 | -4.84 | -7.09 | -5.09 |
| back_front:short:multiple_choice:time | 2.98 | -3.54 | -5.02 | -0.30 | -8.62 | -3.70 | -4.96 |
| back_front:short:quoted_answer:clothing | -2.93 | +1.78 | +2.41 | +0.30 | +2.18 | +1.84 | +3.20 |
| back_front:short:quoted_answer:container | -0.10 | -0.02 | -0.04 | -0.11 | +0.73 | +0.88 | +2.75 |
| back_front:short:quoted_answer:furniture | -2.79 | +2.31 | +2.92 | +0.38 | +1.98 | +1.17 | +2.10 |
| back_front:short:quoted_answer:number | -3.81 | +0.86 | +2.50 | +0.26 | -0.38 | +1.07 | +1.94 |
| back_front:short:quoted_answer:plant | -1.07 | +0.35 | +0.40 | +0.28 | +1.49 | +0.88 | +2.87 |
| back_front:short:quoted_answer:time | -1.30 | +1.24 | +1.27 | +0.62 | +1.59 | +1.18 | +2.53 |
| front_back:long:answer_one_word:clothing | -3.96 | -0.65 | -0.49 | +0.20 | -3.37 | -3.27 | -2.38 |
| front_back:long:answer_one_word:container | -3.91 | +0.50 | -0.03 | +0.75 | -3.57 | -4.13 | -3.46 |
| front_back:long:answer_one_word:furniture | -2.04 | -1.45 | -0.96 | +0.21 | -3.89 | -2.72 | -2.30 |
| front_back:long:answer_one_word:number | -3.33 | -0.91 | -0.65 | +0.23 | -3.59 | -3.70 | -3.09 |
| front_back:long:answer_one_word:plant | -1.20 | -0.97 | -1.25 | -0.04 | -4.12 | -3.07 | -2.37 |
| front_back:long:answer_one_word:time | -3.20 | -0.26 | -0.33 | +0.25 | -3.25 | -2.15 | -2.46 |
| front_back:long:label_colon:clothing | -2.77 | -0.92 | +0.42 | +0.71 | -0.61 | +1.05 | +1.67 |
| front_back:long:label_colon:container | -3.50 | +1.52 | +1.49 | +1.55 | +1.67 | +1.18 | +1.12 |
| front_back:long:label_colon:furniture | -2.92 | -0.78 | +0.20 | +0.60 | -0.25 | +1.23 | +1.63 |
| front_back:long:label_colon:number | 1.38 | +0.02 | +0.40 | +0.16 | -0.01 | +0.62 | +0.62 |
| front_back:long:label_colon:plant | -0.46 | +0.87 | +1.41 | +0.11 | +1.90 | +1.53 | +2.02 |
| front_back:long:label_colon:time | -3.78 | +0.89 | +1.89 | +0.51 | +0.75 | +0.43 | +1.04 |
| front_back:long:list_answer:clothing | -3.28 | -0.77 | -0.60 | +0.02 | -2.56 | -1.83 | -2.98 |
| front_back:long:list_answer:container | -4.09 | +1.05 | +0.17 | +0.66 | -2.74 | -2.42 | -3.33 |
| front_back:long:list_answer:furniture | -3.24 | -0.57 | -0.76 | +0.51 | -1.98 | -1.79 | -2.44 |
| front_back:long:list_answer:number | -2.98 | -1.35 | -2.18 | +0.27 | -2.96 | -1.87 | -3.58 |
| front_back:long:list_answer:plant | -1.77 | +0.08 | -0.78 | +0.12 | -2.25 | -1.92 | -2.56 |
| front_back:long:list_answer:time | -2.90 | -0.50 | -0.81 | +0.15 | -2.55 | -1.72 | -3.48 |
| front_back:long:multiple_choice:clothing | -5.41 | +1.70 | +3.17 | +0.71 | +1.10 | -0.59 | -0.16 |
| front_back:long:multiple_choice:container | 3.41 | -2.14 | -2.22 | -0.61 | -1.45 | +0.96 | -0.70 |
| front_back:long:multiple_choice:furniture | -6.56 | +1.12 | +2.27 | +1.32 | +0.57 | -0.55 | -0.18 |
| front_back:long:multiple_choice:number | -1.76 | -1.86 | -0.48 | +0.30 | -2.66 | -0.80 | -0.54 |
| front_back:long:multiple_choice:plant | 3.31 | +0.02 | +0.32 | -0.23 | -0.22 | -1.81 | -0.36 |
| front_back:long:multiple_choice:time | 0.02 | -3.04 | -2.00 | -0.12 | -2.98 | +0.20 | -0.53 |
| front_back:long:quoted_answer:clothing | -2.84 | +0.07 | +0.91 | +0.64 | +0.48 | +1.06 | +3.20 |
| front_back:long:quoted_answer:container | -4.28 | +2.22 | +2.36 | +1.14 | +1.28 | +0.80 | +2.29 |
| front_back:long:quoted_answer:furniture | -3.93 | -0.10 | +1.16 | +0.70 | +0.89 | -0.31 | +2.12 |
| front_back:long:quoted_answer:number | -3.07 | +1.29 | +1.12 | +0.21 | +0.98 | +0.27 | +2.50 |
| front_back:long:quoted_answer:plant | -2.37 | +0.12 | +0.91 | +0.59 | +0.42 | +1.30 | +2.54 |
| front_back:long:quoted_answer:time | -4.05 | +1.50 | +2.26 | +0.75 | +0.79 | +0.28 | +2.62 |
| front_back:neutral:answer_one_word:clothing | -2.30 | -3.43 | -3.40 | +0.20 | -6.30 | -5.87 | -3.47 |
| front_back:neutral:answer_one_word:container | -3.02 | -1.57 | +0.30 | +0.20 | -5.54 | -6.05 | -3.54 |
| front_back:neutral:answer_one_word:furniture | -2.50 | -1.96 | -1.10 | +0.36 | -6.67 | -6.75 | -4.57 |
| front_back:neutral:answer_one_word:number | -2.48 | -2.98 | -1.71 | +0.23 | -4.90 | -6.83 | -2.48 |
| front_back:neutral:answer_one_word:plant | -1.14 | -0.37 | -1.06 | +0.13 | -4.50 | -6.25 | -3.52 |
| front_back:neutral:answer_one_word:time | -1.84 | -3.27 | -0.40 | +0.30 | -5.26 | -5.61 | -2.25 |
| front_back:neutral:label_colon:clothing | -1.03 | -1.58 | -1.29 | +0.22 | -4.54 | -3.71 | -3.09 |
| front_back:neutral:label_colon:container | -3.84 | -1.28 | +0.96 | +0.28 | -3.93 | -4.66 | -3.18 |
| front_back:neutral:label_colon:furniture | -3.56 | -1.65 | -1.23 | +0.47 | -4.70 | -4.35 | -3.17 |
| front_back:neutral:label_colon:number | -0.70 | -3.15 | -2.23 | +0.58 | -6.34 | -3.74 | -3.03 |
| front_back:neutral:label_colon:plant | -1.22 | -1.53 | -0.93 | +0.50 | -3.73 | -4.52 | -2.54 |
| front_back:neutral:label_colon:time | -2.86 | -2.35 | -0.54 | +0.36 | -4.78 | -3.96 | -2.89 |
| front_back:neutral:list_answer:clothing | -1.21 | -2.12 | -1.74 | +0.01 | -5.31 | -3.36 | -4.38 |
| front_back:neutral:list_answer:container | -1.81 | -0.96 | -0.54 | +0.20 | -4.72 | -3.80 | -4.00 |
| front_back:neutral:list_answer:furniture | -1.74 | -1.79 | -1.62 | -0.06 | -5.70 | -3.55 | -4.39 |
| front_back:neutral:list_answer:number | -1.42 | -2.03 | -2.16 | +0.23 | -5.68 | -3.63 | -4.96 |
| front_back:neutral:list_answer:plant | -0.92 | -0.88 | -0.19 | +0.05 | -4.35 | -4.44 | -5.29 |
| front_back:neutral:list_answer:time | -0.93 | -0.64 | +0.52 | +0.14 | -5.61 | -4.57 | -5.09 |
| front_back:neutral:multiple_choice:clothing | -4.96 | -2.27 | -0.69 | +0.14 | -6.37 | -4.10 | -5.62 |
| front_back:neutral:multiple_choice:container | -0.32 | -1.95 | -2.98 | +0.12 | -5.70 | -3.91 | -6.82 |
| front_back:neutral:multiple_choice:furniture | -4.34 | -2.95 | -1.59 | -0.54 | -6.44 | -4.23 | -5.27 |
| front_back:neutral:multiple_choice:number | 0.57 | -1.85 | -2.52 | -0.13 | -4.71 | -3.73 | -6.98 |
| front_back:neutral:multiple_choice:plant | 1.23 | -0.02 | -2.76 | +0.29 | -3.70 | -4.33 | -5.80 |
| front_back:neutral:multiple_choice:time | 0.26 | -2.75 | -2.80 | -0.22 | -5.52 | -3.76 | -6.75 |
| front_back:neutral:quoted_answer:clothing | -1.15 | -0.45 | -0.04 | +0.21 | +0.72 | +2.43 | +2.41 |
| front_back:neutral:quoted_answer:container | -0.92 | -0.23 | -0.52 | +0.11 | +1.05 | +2.30 | +2.28 |
| front_back:neutral:quoted_answer:furniture | -2.66 | -1.18 | -0.75 | +0.34 | +0.38 | +2.43 | +1.88 |
| front_back:neutral:quoted_answer:number | -0.45 | +0.23 | +0.34 | +0.07 | +1.01 | +0.32 | +3.17 |
| front_back:neutral:quoted_answer:plant | 0.73 | -1.43 | -0.60 | +0.46 | -0.01 | +2.11 | +2.10 |
| front_back:neutral:quoted_answer:time | -0.73 | -0.05 | +0.63 | +0.49 | +1.30 | +1.73 | +2.05 |
| front_back:short:answer_one_word:clothing | -1.88 | -2.07 | -0.86 | +0.29 | -6.48 | -8.85 | -5.82 |
| front_back:short:answer_one_word:container | -1.29 | -0.57 | -1.42 | +0.09 | -5.07 | -7.56 | -5.07 |
| front_back:short:answer_one_word:furniture | -1.18 | +0.48 | +0.39 | +0.02 | -5.81 | -9.15 | -5.91 |
| front_back:short:answer_one_word:number | -1.43 | -0.53 | -1.12 | +0.55 | -5.71 | -7.26 | -5.83 |
| front_back:short:answer_one_word:plant | -1.24 | -0.04 | +0.12 | +0.12 | -5.71 | -8.36 | -5.60 |
| front_back:short:answer_one_word:time | -1.18 | -0.66 | -0.47 | -0.03 | -4.33 | -8.82 | -4.62 |
| front_back:short:label_colon:clothing | 2.50 | -6.41 | -7.55 | +0.12 | -0.88 | -2.58 | +3.21 |
| front_back:short:label_colon:container | -0.70 | -2.84 | -2.91 | -0.04 | +1.54 | -2.87 | +2.35 |
| front_back:short:label_colon:furniture | -0.40 | -3.85 | -4.65 | +0.18 | +2.45 | -1.89 | +3.57 |
| front_back:short:label_colon:number | 2.83 | -4.08 | -5.33 | -0.01 | -1.23 | -1.31 | +0.70 |
| front_back:short:label_colon:plant | 6.25 | -10.29 | -11.85 | -0.70 | -3.12 | -2.30 | +3.92 |
| front_back:short:label_colon:time | 2.09 | -5.41 | -6.64 | -1.59 | -0.12 | -2.40 | +3.18 |
| front_back:short:list_answer:clothing | -0.84 | -1.32 | -1.03 | +0.14 | -5.18 | -5.79 | -5.06 |
| front_back:short:list_answer:container | -1.34 | -1.23 | -0.27 | -0.04 | -4.59 | -5.90 | -4.54 |
| front_back:short:list_answer:furniture | -1.19 | -0.28 | +0.32 | +0.05 | -4.34 | -6.03 | -4.42 |
| front_back:short:list_answer:number | -1.34 | -1.30 | -0.67 | +0.25 | -4.80 | -5.71 | -4.80 |
| front_back:short:list_answer:plant | -0.33 | -0.09 | +0.91 | -0.15 | -3.80 | -6.33 | -4.46 |
| front_back:short:list_answer:time | -0.38 | -0.77 | -0.45 | -0.01 | -3.96 | -7.13 | -4.48 |
| front_back:short:multiple_choice:clothing | -5.59 | -0.66 | +0.42 | +0.35 | -5.12 | -4.46 | -4.37 |
| front_back:short:multiple_choice:container | 4.37 | -0.33 | -1.02 | -1.08 | -5.14 | -4.58 | -4.43 |
| front_back:short:multiple_choice:furniture | -6.13 | -0.75 | +0.27 | +0.48 | -5.06 | -4.31 | -4.26 |
| front_back:short:multiple_choice:number | -0.81 | -1.52 | -1.70 | -0.10 | -7.20 | -5.74 | -4.64 |
| front_back:short:multiple_choice:plant | 2.23 | +1.00 | -0.09 | -0.44 | -4.60 | -5.19 | -5.19 |
| front_back:short:multiple_choice:time | 2.07 | -2.71 | -4.48 | +0.57 | -7.64 | -3.60 | -4.42 |
| front_back:short:quoted_answer:clothing | -2.42 | +1.85 | +2.32 | +1.28 | +1.84 | +0.97 | +2.84 |
| front_back:short:quoted_answer:container | -1.71 | +1.55 | +1.71 | +0.36 | +1.70 | +0.34 | +2.87 |
| front_back:short:quoted_answer:furniture | -1.55 | +1.50 | +1.56 | +0.05 | +1.93 | +0.48 | +2.26 |
| front_back:short:quoted_answer:number | -2.59 | +0.46 | +1.65 | +1.01 | -0.04 | +1.36 | +2.63 |
| front_back:short:quoted_answer:plant | -0.56 | +0.81 | +1.15 | -0.25 | +0.39 | -0.50 | +2.12 |
| front_back:short:quoted_answer:time | -2.02 | +1.83 | +2.43 | +0.73 | +1.95 | +0.94 | +3.58 |

### Cross-model difficult-format core

| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | mlp_hiddenΔ | k8+mlp_hiddenΔ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek7b | 144 | -1.611 | -0.721 | -0.393 | +0.251 | -2.182 | -2.509 | -1.831 | -1.114 | -3.911 | -3.042 | -3.757 | -3.004 | 134.750 | 147.393 |
| glm4 | 144 | -0.684 | +0.026 | +0.009 | +0.044 | +0.191 | +1.818 | -0.015 | -0.173 | +0.320 | +1.790 | -0.605 | +0.090 | 80.949 | 79.852 |
| qwen3 | 144 | -0.644 | +0.017 | +0.063 | +0.039 | -0.766 | -1.528 | -0.672 | -0.565 | -0.675 | -1.431 | -1.018 | -0.650 | 39.443 | 43.402 |

