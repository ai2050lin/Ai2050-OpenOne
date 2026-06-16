# Phase 156 Cross-model Set-Writer Surface Gate Closure Summary

## qwen3

cases=180, attention=L36, mlp=L36, heads=32, steps=3

### All cases

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.368 | 0.369 | 0.369 | 0.367 | 0.364 | 0.353 | 0.351 | 0.347 | +0.001 | +0.001 | -0.015 | -0.017 | -0.022 |
| difficult_formats | 144 | 0.218 | 0.217 | 0.219 | 0.217 | 0.213 | 0.199 | 0.195 | 0.190 | -0.001 | +0.001 | -0.019 | -0.023 | -0.028 |
| multiple_choice_control | 36 | 0.969 | 0.976 | 0.969 | 0.965 | 0.969 | 0.972 | 0.976 | 0.972 | +0.007 | +0.000 | +0.003 | +0.007 | +0.003 |

### By category

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.362 | 0.375 | 0.379 | 0.354 | 0.358 | 0.346 | 0.358 | 0.358 | +0.013 | +0.017 | -0.017 | -0.004 | -0.004 |
| container | 30 | 0.358 | 0.358 | 0.362 | 0.354 | 0.362 | 0.371 | 0.371 | 0.367 | +0.000 | +0.004 | +0.013 | +0.013 | +0.008 |
| furniture | 30 | 0.379 | 0.392 | 0.400 | 0.383 | 0.383 | 0.392 | 0.400 | 0.388 | +0.013 | +0.021 | +0.013 | +0.021 | +0.008 |
| number | 30 | 0.254 | 0.258 | 0.254 | 0.258 | 0.258 | 0.225 | 0.212 | 0.208 | +0.004 | +0.000 | -0.029 | -0.042 | -0.046 |
| plant | 30 | 0.521 | 0.508 | 0.492 | 0.517 | 0.504 | 0.463 | 0.446 | 0.425 | -0.013 | -0.029 | -0.058 | -0.075 | -0.096 |
| time | 30 | 0.333 | 0.321 | 0.325 | 0.333 | 0.317 | 0.325 | 0.321 | 0.333 | -0.012 | -0.008 | -0.008 | -0.012 | +0.000 |

### By format

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.149 | 0.146 | 0.163 | 0.156 | 0.149 | 0.101 | 0.111 | 0.118 | -0.003 | +0.014 | -0.049 | -0.038 | -0.031 |
| label_colon | 36 | 0.361 | 0.368 | 0.354 | 0.361 | 0.358 | 0.323 | 0.319 | 0.309 | +0.007 | -0.007 | -0.038 | -0.042 | -0.052 |
| list_answer | 36 | 0.208 | 0.201 | 0.201 | 0.198 | 0.191 | 0.215 | 0.201 | 0.191 | -0.007 | -0.007 | +0.007 | -0.007 | -0.017 |
| multiple_choice | 36 | 0.969 | 0.976 | 0.969 | 0.965 | 0.969 | 0.972 | 0.976 | 0.972 | +0.007 | +0.000 | +0.003 | +0.007 | +0.003 |
| quoted_answer | 36 | 0.153 | 0.153 | 0.156 | 0.153 | 0.153 | 0.156 | 0.149 | 0.142 | +0.000 | +0.003 | +0.003 | -0.003 | -0.010 |

### By family

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.248 | 0.256 | 0.252 | 0.244 | 0.248 | 0.250 | 0.250 | 0.240 | +0.008 | +0.004 | +0.002 | +0.002 | -0.008 |
| neutral | 60 | 0.292 | 0.292 | 0.287 | 0.287 | 0.283 | 0.279 | 0.285 | 0.283 | +0.000 | -0.004 | -0.013 | -0.006 | -0.008 |
| short | 60 | 0.565 | 0.558 | 0.567 | 0.569 | 0.560 | 0.531 | 0.519 | 0.517 | -0.006 | +0.002 | -0.033 | -0.046 | -0.048 |

### By split

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.392 | 0.388 | 0.388 | 0.388 | 0.381 | 0.358 | 0.360 | 0.362 | -0.004 | -0.004 | -0.033 | -0.032 | -0.029 |
| front_back | 90 | 0.344 | 0.350 | 0.350 | 0.346 | 0.347 | 0.349 | 0.343 | 0.331 | +0.006 | +0.006 | +0.004 | -0.001 | -0.014 |

### Cases

| case | clean | joint_k4 | joint_k8 | random_k4 | mlp_joint | k4+mlp | k8+mlp |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:clothing | 0.38 | 0.50 | 0.50 | 0.38 | 0.50 | 0.50 | 0.50 |
| back_front:long:label_colon:container | 0.12 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:furniture | 0.12 | 0.25 | 0.38 | 0.12 | 0.25 | 0.38 | 0.38 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:container | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.12 | 0.12 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:multiple_choice:clothing | 1.00 | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 | 0.88 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:furniture | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:clothing | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| back_front:neutral:label_colon:container | 0.25 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:plant | 0.62 | 0.62 | 0.50 | 0.75 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:label_colon:time | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:container | 0.50 | 0.62 | 0.62 | 0.25 | 0.38 | 0.38 | 0.50 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:plant | 0.25 | 0.25 | 0.12 | 0.25 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.25 | 0.25 | 0.25 |
| back_front:neutral:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:furniture | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| back_front:neutral:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:time | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:short:answer_one_word:clothing | 0.50 | 0.38 | 0.62 | 0.38 | 0.25 | 0.25 | 0.38 |
| back_front:short:answer_one_word:container | 0.50 | 0.50 | 0.75 | 0.62 | 0.50 | 0.62 | 0.50 |
| back_front:short:answer_one_word:furniture | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.62 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.12 | 0.12 | 0.12 |
| back_front:short:answer_one_word:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.12 | 0.25 | 0.38 |
| back_front:short:label_colon:clothing | 0.75 | 0.75 | 0.75 | 0.62 | 0.50 | 0.50 | 0.50 |
| back_front:short:label_colon:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.88 | 0.88 | 0.88 |
| back_front:short:label_colon:furniture | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| back_front:short:label_colon:number | 0.50 | 0.50 | 0.50 | 0.62 | 0.00 | 0.00 | 0.00 |
| back_front:short:label_colon:plant | 0.75 | 0.62 | 0.50 | 0.75 | 0.88 | 0.62 | 0.62 |
| back_front:short:label_colon:time | 0.62 | 0.62 | 0.62 | 0.62 | 0.75 | 0.62 | 0.75 |
| back_front:short:list_answer:clothing | 0.62 | 0.50 | 0.50 | 0.62 | 0.50 | 0.50 | 0.50 |
| back_front:short:list_answer:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.62 | 0.62 | 0.62 |
| back_front:short:list_answer:furniture | 0.12 | 0.12 | 0.12 | 0.12 | 0.38 | 0.38 | 0.25 |
| back_front:short:list_answer:number | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| back_front:short:list_answer:plant | 0.62 | 0.75 | 0.75 | 0.62 | 0.88 | 0.75 | 0.75 |
| back_front:short:list_answer:time | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| back_front:short:multiple_choice:clothing | 0.88 | 1.00 | 0.88 | 0.88 | 0.88 | 1.00 | 1.00 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:quoted_answer:clothing | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:short:quoted_answer:container | 0.75 | 0.75 | 0.75 | 0.75 | 0.62 | 0.62 | 0.62 |
| back_front:short:quoted_answer:furniture | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| back_front:short:quoted_answer:number | 0.38 | 0.38 | 0.38 | 0.38 | 0.50 | 0.50 | 0.50 |
| back_front:short:quoted_answer:plant | 0.88 | 0.62 | 0.62 | 0.88 | 0.25 | 0.25 | 0.25 |
| back_front:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:clothing | 0.38 | 0.62 | 0.62 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:furniture | 0.62 | 0.75 | 0.75 | 0.62 | 0.38 | 0.50 | 0.12 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:plant | 0.25 | 0.12 | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 | 0.25 |
| front_back:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:multiple_choice:clothing | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:number | 0.88 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.38 | 0.25 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| front_back:neutral:label_colon:clothing | 0.50 | 0.62 | 0.62 | 0.50 | 0.50 | 0.62 | 0.62 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:furniture | 0.38 | 0.38 | 0.25 | 0.38 | 0.38 | 0.25 | 0.25 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 |
| front_back:neutral:label_colon:time | 0.25 | 0.12 | 0.12 | 0.25 | 0.25 | 0.12 | 0.12 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:plant | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.25 |
| front_back:neutral:list_answer:time | 0.25 | 0.12 | 0.12 | 0.25 | 0.12 | 0.12 | 0.12 |
| front_back:neutral:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.50 | 0.50 | 0.38 | 0.50 | 0.50 | 0.38 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:clothing | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:short:answer_one_word:container | 0.12 | 0.12 | 0.25 | 0.25 | 0.12 | 0.12 | 0.12 |
| front_back:short:answer_one_word:furniture | 0.75 | 0.75 | 0.75 | 0.88 | 0.62 | 0.62 | 0.62 |
| front_back:short:answer_one_word:number | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:plant | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
| front_back:short:answer_one_word:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 |
| front_back:short:label_colon:clothing | 0.50 | 0.62 | 0.62 | 0.62 | 0.50 | 0.50 | 0.50 |
| front_back:short:label_colon:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.62 | 0.62 | 0.50 |
| front_back:short:label_colon:furniture | 0.88 | 0.88 | 0.88 | 0.88 | 0.62 | 0.62 | 0.62 |
| front_back:short:label_colon:number | 0.38 | 0.38 | 0.25 | 0.38 | 0.12 | 0.12 | 0.12 |
| front_back:short:label_colon:plant | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| front_back:short:label_colon:time | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| front_back:short:list_answer:clothing | 0.38 | 0.25 | 0.25 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:short:list_answer:container | 0.25 | 0.25 | 0.25 | 0.25 | 0.62 | 0.50 | 0.50 |
| front_back:short:list_answer:furniture | 0.00 | 0.00 | 0.12 | 0.00 | 0.25 | 0.25 | 0.25 |
| front_back:short:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:plant | 0.88 | 0.75 | 0.75 | 0.75 | 0.88 | 0.75 | 0.62 |
| front_back:short:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:short:multiple_choice:clothing | 0.88 | 1.00 | 1.00 | 0.88 | 0.88 | 1.00 | 1.00 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.75 | 0.75 | 0.75 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| front_back:short:quoted_answer:furniture | 0.75 | 0.75 | 0.88 | 0.75 | 1.00 | 1.00 | 1.00 |
| front_back:short:quoted_answer:number | 0.25 | 0.25 | 0.25 | 0.25 | 0.50 | 0.25 | 0.12 |
| front_back:short:quoted_answer:plant | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| front_back:short:quoted_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |

## glm4

cases=180, attention=L39, mlp=L40, heads=32, steps=3

### All cases

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.297 | 0.297 | 0.297 | 0.300 | 0.301 | 0.315 | 0.318 | 0.320 | +0.000 | +0.000 | +0.017 | +0.021 | +0.023 |
| difficult_formats | 144 | 0.131 | 0.129 | 0.130 | 0.132 | 0.132 | 0.204 | 0.195 | 0.200 | -0.002 | -0.001 | +0.073 | +0.064 | +0.069 |
| multiple_choice_control | 36 | 0.962 | 0.969 | 0.965 | 0.972 | 0.976 | 0.757 | 0.809 | 0.802 | +0.007 | +0.003 | -0.205 | -0.153 | -0.160 |

### By category

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.275 | 0.287 | 0.304 | 0.287 | 0.287 | 0.200 | 0.221 | 0.242 | +0.012 | +0.029 | -0.075 | -0.054 | -0.033 |
| container | 30 | 0.296 | 0.296 | 0.275 | 0.292 | 0.296 | 0.392 | 0.383 | 0.388 | +0.000 | -0.021 | +0.096 | +0.088 | +0.092 |
| furniture | 30 | 0.317 | 0.312 | 0.321 | 0.325 | 0.321 | 0.196 | 0.212 | 0.208 | -0.004 | +0.004 | -0.121 | -0.104 | -0.108 |
| number | 30 | 0.212 | 0.212 | 0.221 | 0.217 | 0.225 | 0.279 | 0.267 | 0.275 | +0.000 | +0.008 | +0.067 | +0.054 | +0.063 |
| plant | 30 | 0.379 | 0.375 | 0.371 | 0.375 | 0.383 | 0.537 | 0.537 | 0.525 | -0.004 | -0.008 | +0.158 | +0.158 | +0.146 |
| time | 30 | 0.304 | 0.300 | 0.292 | 0.304 | 0.292 | 0.283 | 0.287 | 0.283 | -0.004 | -0.012 | -0.021 | -0.017 | -0.021 |

### By format

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.056 | 0.049 | 0.049 | 0.056 | 0.056 | 0.122 | 0.125 | 0.125 | -0.007 | -0.007 | +0.066 | +0.069 | +0.069 |
| label_colon | 36 | 0.236 | 0.240 | 0.247 | 0.240 | 0.247 | 0.441 | 0.417 | 0.434 | +0.003 | +0.010 | +0.205 | +0.181 | +0.198 |
| list_answer | 36 | 0.146 | 0.142 | 0.142 | 0.139 | 0.139 | 0.156 | 0.135 | 0.139 | -0.003 | -0.003 | +0.010 | -0.010 | -0.007 |
| multiple_choice | 36 | 0.962 | 0.969 | 0.965 | 0.972 | 0.976 | 0.757 | 0.809 | 0.802 | +0.007 | +0.003 | -0.205 | -0.153 | -0.160 |
| quoted_answer | 36 | 0.087 | 0.087 | 0.083 | 0.094 | 0.087 | 0.097 | 0.104 | 0.101 | +0.000 | -0.003 | +0.010 | +0.017 | +0.014 |

### By family

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.212 | 0.215 | 0.217 | 0.210 | 0.215 | 0.248 | 0.248 | 0.252 | +0.002 | +0.004 | +0.035 | +0.035 | +0.040 |
| neutral | 60 | 0.267 | 0.271 | 0.269 | 0.267 | 0.275 | 0.258 | 0.250 | 0.240 | +0.004 | +0.002 | -0.008 | -0.017 | -0.027 |
| short | 60 | 0.412 | 0.406 | 0.406 | 0.423 | 0.412 | 0.438 | 0.456 | 0.469 | -0.006 | -0.006 | +0.025 | +0.044 | +0.056 |

### By split

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.306 | 0.307 | 0.304 | 0.304 | 0.311 | 0.318 | 0.321 | 0.322 | +0.001 | -0.001 | +0.012 | +0.015 | +0.017 |
| front_back | 90 | 0.289 | 0.287 | 0.290 | 0.296 | 0.290 | 0.311 | 0.315 | 0.318 | -0.001 | +0.001 | +0.022 | +0.026 | +0.029 |

### Cases

| case | clean | joint_k4 | joint_k8 | random_k4 | mlp_joint | k4+mlp | k8+mlp |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.38 | 0.38 |
| back_front:long:label_colon:furniture | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.75 | 0.75 |
| back_front:long:label_colon:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.88 | 0.88 | 0.88 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:container | 0.25 | 0.12 | 0.00 | 0.12 | 0.25 | 0.25 | 0.12 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.00 | 0.12 | 0.12 | 0.12 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 | 0.88 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 | 0.88 |
| back_front:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:clothing | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.50 | 0.38 | 0.50 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:number | 0.12 | 0.12 | 0.12 | 0.00 | 0.25 | 0.00 | 0.00 |
| back_front:neutral:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.88 | 0.88 | 0.88 |
| back_front:neutral:label_colon:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.38 | 0.38 | 0.38 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:container | 0.12 | 0.25 | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.00 | 0.00 |
| back_front:neutral:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.12 | 0.50 | 0.50 | 0.38 |
| back_front:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.12 |
| back_front:neutral:multiple_choice:clothing | 0.88 | 0.88 | 0.75 | 0.88 | 0.75 | 0.75 | 0.75 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 | 0.88 |
| back_front:neutral:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.38 |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 | 0.50 | 0.50 |
| back_front:neutral:multiple_choice:time | 0.88 | 0.88 | 0.88 | 1.00 | 0.50 | 0.50 | 0.50 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.25 | 0.25 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.12 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:clothing | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.25 | 0.38 |
| back_front:short:answer_one_word:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.88 | 0.88 | 0.88 |
| back_front:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.50 | 0.50 | 0.38 |
| back_front:short:answer_one_word:plant | 0.38 | 0.38 | 0.25 | 0.25 | 1.00 | 1.00 | 1.00 |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.25 | 0.25 |
| back_front:short:label_colon:clothing | 0.62 | 0.62 | 0.75 | 0.62 | 0.62 | 0.62 | 0.62 |
| back_front:short:label_colon:container | 0.50 | 0.50 | 0.38 | 0.50 | 1.00 | 1.00 | 1.00 |
| back_front:short:label_colon:furniture | 0.88 | 0.88 | 0.88 | 0.88 | 0.25 | 0.25 | 0.38 |
| back_front:short:label_colon:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:short:label_colon:plant | 0.38 | 0.50 | 0.50 | 0.62 | 1.00 | 1.00 | 1.00 |
| back_front:short:label_colon:time | 0.75 | 0.75 | 0.75 | 0.75 | 0.62 | 0.62 | 0.62 |
| back_front:short:list_answer:clothing | 0.25 | 0.25 | 0.38 | 0.25 | 0.12 | 0.12 | 0.25 |
| back_front:short:list_answer:container | 0.12 | 0.12 | 0.12 | 0.12 | 0.38 | 0.12 | 0.12 |
| back_front:short:list_answer:furniture | 0.25 | 0.25 | 0.12 | 0.25 | 0.00 | 0.00 | 0.00 |
| back_front:short:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| back_front:short:list_answer:plant | 0.62 | 0.62 | 0.62 | 0.62 | 0.88 | 0.88 | 0.88 |
| back_front:short:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.12 | 0.12 | 0.12 |
| back_front:short:multiple_choice:clothing | 0.88 | 0.88 | 0.88 | 0.88 | 0.38 | 0.50 | 0.62 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 | 0.50 | 0.50 |
| back_front:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.38 |
| back_front:short:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 | 0.50 | 0.38 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:quoted_answer:container | 0.25 | 0.25 | 0.12 | 0.25 | 0.62 | 0.62 | 0.62 |
| back_front:short:quoted_answer:furniture | 0.25 | 0.12 | 0.25 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:short:quoted_answer:number | 0.12 | 0.12 | 0.25 | 0.25 | 0.00 | 0.12 | 0.12 |
| back_front:short:quoted_answer:plant | 0.25 | 0.38 | 0.38 | 0.38 | 0.62 | 0.62 | 0.62 |
| back_front:short:quoted_answer:time | 0.12 | 0.12 | 0.00 | 0.12 | 0.25 | 0.25 | 0.25 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:clothing | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.12 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.12 | 0.25 |
| front_back:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.25 |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 0.88 | 0.88 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| front_back:long:list_answer:time | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.75 | 0.75 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 | 0.88 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:number | 0.88 | 1.00 | 1.00 | 1.00 | 0.62 | 0.75 | 0.75 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:clothing | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.25 | 0.25 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:furniture | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.38 | 0.38 |
| front_back:neutral:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.12 | 0.88 | 0.88 | 0.75 |
| front_back:neutral:label_colon:time | 0.38 | 0.38 | 0.38 | 0.38 | 0.75 | 0.75 | 0.75 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| front_back:neutral:list_answer:plant | 0.38 | 0.25 | 0.25 | 0.38 | 0.25 | 0.25 | 0.25 |
| front_back:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:multiple_choice:clothing | 0.75 | 0.88 | 0.88 | 0.75 | 0.38 | 0.38 | 0.25 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:furniture | 0.88 | 0.88 | 0.88 | 1.00 | 0.50 | 0.75 | 0.75 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 1.00 | 0.88 | 0.75 | 0.75 | 0.88 |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 | 0.88 |
| front_back:neutral:multiple_choice:time | 1.00 | 1.00 | 0.88 | 1.00 | 0.88 | 1.00 | 0.62 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.12 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.38 | 0.38 | 0.25 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.38 | 0.25 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:neutral:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:clothing | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| front_back:short:answer_one_word:container | 0.12 | 0.12 | 0.00 | 0.12 | 0.50 | 0.50 | 0.50 |
| front_back:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 |
| front_back:short:answer_one_word:plant | 0.38 | 0.25 | 0.25 | 0.38 | 0.88 | 0.88 | 0.88 |
| front_back:short:answer_one_word:time | 0.12 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.12 |
| front_back:short:label_colon:clothing | 0.62 | 0.62 | 0.75 | 0.88 | 0.75 | 0.88 | 1.00 |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 1.00 | 1.00 |
| front_back:short:label_colon:furniture | 0.88 | 0.88 | 0.88 | 0.88 | 0.62 | 0.75 | 0.62 |
| front_back:short:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.38 | 0.50 |
| front_back:short:label_colon:plant | 0.25 | 0.25 | 0.25 | 0.25 | 1.00 | 0.88 | 0.88 |
| front_back:short:label_colon:time | 0.38 | 0.38 | 0.38 | 0.25 | 0.50 | 0.50 | 0.50 |
| front_back:short:list_answer:clothing | 0.25 | 0.25 | 0.38 | 0.25 | 0.25 | 0.25 | 0.38 |
| front_back:short:list_answer:container | 0.25 | 0.25 | 0.12 | 0.25 | 0.25 | 0.25 | 0.25 |
| front_back:short:list_answer:furniture | 0.62 | 0.62 | 0.75 | 0.62 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:number | 0.12 | 0.00 | 0.00 | 0.12 | 0.25 | 0.25 | 0.25 |
| front_back:short:list_answer:plant | 0.75 | 0.62 | 0.62 | 0.75 | 1.00 | 0.88 | 0.88 |
| front_back:short:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:short:multiple_choice:clothing | 0.75 | 0.88 | 0.88 | 0.75 | 0.62 | 0.88 | 0.88 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.62 | 0.88 | 0.88 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 0.62 | 0.75 |
| front_back:short:multiple_choice:time | 1.00 | 0.88 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:quoted_answer:clothing | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:furniture | 0.38 | 0.38 | 0.38 | 0.50 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.12 |
| front_back:short:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.38 | 0.38 | 0.38 |
| front_back:short:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |

## deepseek7b

cases=180, attention=L28, mlp=L28, heads=28, steps=3

### All cases

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.235 | 0.240 | 0.231 | 0.233 | 0.226 | 0.251 | 0.236 | 0.216 | +0.005 | -0.004 | +0.017 | +0.001 | -0.019 |
| difficult_formats | 144 | 0.076 | 0.076 | 0.066 | 0.073 | 0.069 | 0.074 | 0.062 | 0.044 | +0.000 | -0.010 | -0.003 | -0.015 | -0.032 |
| multiple_choice_control | 36 | 0.868 | 0.892 | 0.889 | 0.875 | 0.851 | 0.962 | 0.934 | 0.903 | +0.024 | +0.021 | +0.094 | +0.066 | +0.035 |

### By category

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.208 | 0.208 | 0.233 | 0.212 | 0.221 | 0.208 | 0.200 | 0.196 | +0.000 | +0.025 | +0.000 | -0.008 | -0.013 |
| container | 30 | 0.263 | 0.283 | 0.246 | 0.267 | 0.258 | 0.279 | 0.271 | 0.250 | +0.021 | -0.017 | +0.017 | +0.008 | -0.013 |
| furniture | 30 | 0.183 | 0.204 | 0.204 | 0.188 | 0.188 | 0.200 | 0.196 | 0.196 | +0.021 | +0.021 | +0.017 | +0.013 | +0.013 |
| number | 30 | 0.208 | 0.204 | 0.196 | 0.208 | 0.212 | 0.254 | 0.217 | 0.196 | -0.004 | -0.013 | +0.046 | +0.008 | -0.013 |
| plant | 30 | 0.325 | 0.308 | 0.275 | 0.317 | 0.267 | 0.338 | 0.308 | 0.267 | -0.017 | -0.050 | +0.013 | -0.017 | -0.058 |
| time | 30 | 0.221 | 0.229 | 0.229 | 0.208 | 0.208 | 0.229 | 0.225 | 0.192 | +0.008 | +0.008 | +0.008 | +0.004 | -0.029 |

### By format

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.049 | 0.066 | 0.045 | 0.049 | 0.045 | 0.056 | 0.045 | 0.038 | +0.017 | -0.003 | +0.007 | -0.003 | -0.010 |
| label_colon | 36 | 0.069 | 0.062 | 0.076 | 0.069 | 0.062 | 0.059 | 0.031 | 0.014 | -0.007 | +0.007 | -0.010 | -0.038 | -0.056 |
| list_answer | 36 | 0.125 | 0.104 | 0.069 | 0.115 | 0.108 | 0.128 | 0.132 | 0.097 | -0.021 | -0.056 | +0.003 | +0.007 | -0.028 |
| multiple_choice | 36 | 0.868 | 0.892 | 0.889 | 0.875 | 0.851 | 0.962 | 0.934 | 0.903 | +0.024 | +0.021 | +0.094 | +0.066 | +0.035 |
| quoted_answer | 36 | 0.062 | 0.073 | 0.073 | 0.059 | 0.062 | 0.052 | 0.038 | 0.028 | +0.010 | +0.010 | -0.010 | -0.024 | -0.035 |

### By family

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.260 | 0.267 | 0.275 | 0.260 | 0.260 | 0.250 | 0.237 | 0.223 | +0.006 | +0.015 | -0.010 | -0.023 | -0.038 |
| neutral | 60 | 0.185 | 0.204 | 0.194 | 0.181 | 0.179 | 0.235 | 0.223 | 0.210 | +0.019 | +0.008 | +0.050 | +0.038 | +0.025 |
| short | 60 | 0.258 | 0.248 | 0.223 | 0.258 | 0.237 | 0.269 | 0.248 | 0.215 | -0.010 | -0.035 | +0.010 | -0.010 | -0.044 |

### By split

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.244 | 0.256 | 0.239 | 0.242 | 0.236 | 0.261 | 0.243 | 0.218 | +0.011 | -0.006 | +0.017 | -0.001 | -0.026 |
| front_back | 90 | 0.225 | 0.224 | 0.222 | 0.225 | 0.215 | 0.242 | 0.229 | 0.214 | -0.001 | -0.003 | +0.017 | +0.004 | -0.011 |

### Cases

| case | clean | joint_k4 | joint_k8 | random_k4 | mlp_joint | k4+mlp | k8+mlp |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:container | 0.25 | 0.38 | 0.12 | 0.25 | 0.38 | 0.25 | 0.25 |
| back_front:long:answer_one_word:furniture | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:long:answer_one_word:time | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 |
| back_front:long:label_colon:clothing | 0.00 | 0.12 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.50 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:list_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.12 | 0.00 | 0.00 |
| back_front:long:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:long:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 0.75 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:container | 0.62 | 0.62 | 0.62 | 0.62 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:plant | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| back_front:long:quoted_answer:time | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.38 | 0.12 | 0.00 | 0.38 | 0.25 | 0.25 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:label_colon:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| back_front:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:container | 0.38 | 0.38 | 0.12 | 0.38 | 0.38 | 0.50 | 0.38 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:list_answer:number | 0.12 | 0.12 | 0.00 | 0.12 | 0.50 | 0.50 | 0.38 |
| back_front:neutral:list_answer:plant | 0.38 | 0.38 | 0.12 | 0.38 | 0.25 | 0.25 | 0.25 |
| back_front:neutral:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:multiple_choice:clothing | 0.62 | 0.62 | 0.75 | 0.62 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:container | 0.75 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:furniture | 0.50 | 0.62 | 0.50 | 0.38 | 1.00 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:number | 0.88 | 0.88 | 0.75 | 0.88 | 0.88 | 1.00 | 1.00 |
| back_front:neutral:multiple_choice:plant | 0.62 | 0.50 | 0.50 | 0.75 | 0.88 | 0.88 | 0.75 |
| back_front:neutral:multiple_choice:time | 0.62 | 0.62 | 1.00 | 0.62 | 1.00 | 1.00 | 0.88 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 |
| back_front:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:container | 0.12 | 0.00 | 0.00 | 0.12 | 0.00 | 0.25 | 0.12 |
| back_front:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:answer_one_word:plant | 0.00 | 0.12 | 0.00 | 0.12 | 0.25 | 0.00 | 0.12 |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:label_colon:clothing | 0.38 | 0.38 | 0.38 | 0.38 | 0.00 | 0.00 | 0.00 |
| back_front:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| back_front:short:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:label_colon:number | 0.50 | 0.25 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 |
| back_front:short:label_colon:plant | 0.25 | 0.38 | 0.38 | 0.25 | 0.50 | 0.38 | 0.12 |
| back_front:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:list_answer:container | 0.25 | 0.12 | 0.00 | 0.25 | 0.50 | 0.38 | 0.25 |
| back_front:short:list_answer:furniture | 0.00 | 0.12 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:list_answer:number | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 |
| back_front:short:list_answer:plant | 0.25 | 0.12 | 0.12 | 0.12 | 0.25 | 0.50 | 0.12 |
| back_front:short:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 |
| back_front:short:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.75 |
| back_front:short:multiple_choice:container | 1.00 | 0.88 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| back_front:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.88 |
| back_front:short:multiple_choice:number | 0.88 | 1.00 | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 |
| back_front:short:multiple_choice:plant | 0.88 | 0.88 | 0.88 | 0.88 | 1.00 | 1.00 | 0.88 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.62 | 0.50 |
| back_front:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:quoted_answer:container | 0.25 | 0.50 | 0.38 | 0.25 | 0.25 | 0.25 | 0.12 |
| back_front:short:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_front:short:quoted_answer:number | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 |
| back_front:short:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 |
| back_front:short:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:furniture | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:answer_one_word:plant | 0.38 | 0.25 | 0.25 | 0.25 | 0.12 | 0.00 | 0.00 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:clothing | 0.00 | 0.00 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:label_colon:number | 0.00 | 0.12 | 0.38 | 0.00 | 0.00 | 0.12 | 0.12 |
| front_back:long:label_colon:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:list_answer:plant | 0.38 | 0.25 | 0.25 | 0.38 | 0.38 | 0.38 | 0.38 |
| front_back:long:list_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.12 | 0.12 |
| front_back:long:multiple_choice:clothing | 0.88 | 0.75 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:number | 0.50 | 0.62 | 0.62 | 0.62 | 1.00 | 1.00 | 0.88 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.50 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 |
| front_back:long:quoted_answer:plant | 0.25 | 0.38 | 0.38 | 0.25 | 0.38 | 0.38 | 0.38 |
| front_back:long:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:clothing | 0.38 | 0.38 | 0.25 | 0.38 | 0.25 | 0.25 | 0.12 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:label_colon:plant | 0.12 | 0.00 | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 |
| front_back:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:number | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:list_answer:plant | 0.25 | 0.12 | 0.12 | 0.12 | 0.25 | 0.12 | 0.00 |
| front_back:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| front_back:neutral:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:container | 0.62 | 0.62 | 0.62 | 0.88 | 0.75 | 0.62 | 0.62 |
| front_back:neutral:multiple_choice:furniture | 0.50 | 0.88 | 0.88 | 0.75 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.75 |
| front_back:neutral:multiple_choice:plant | 0.88 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| front_back:neutral:multiple_choice:time | 0.75 | 0.75 | 0.88 | 0.50 | 0.75 | 0.88 | 0.88 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.00 | 0.00 |
| front_back:neutral:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.38 | 0.25 |
| front_back:short:answer_one_word:time | 0.25 | 0.12 | 0.00 | 0.25 | 0.38 | 0.38 | 0.25 |
| front_back:short:label_colon:clothing | 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 | 0.00 |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:label_colon:number | 0.12 | 0.00 | 0.00 | 0.12 | 0.25 | 0.25 | 0.00 |
| front_back:short:label_colon:plant | 0.25 | 0.12 | 0.12 | 0.25 | 0.00 | 0.00 | 0.00 |
| front_back:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:list_answer:number | 0.25 | 0.12 | 0.00 | 0.25 | 0.12 | 0.12 | 0.00 |
| front_back:short:list_answer:plant | 0.62 | 0.50 | 0.00 | 0.62 | 0.38 | 0.62 | 0.62 |
| front_back:short:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.25 | 0.25 |
| front_back:short:multiple_choice:clothing | 0.75 | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:number | 0.75 | 0.88 | 0.88 | 0.75 | 1.00 | 0.62 | 0.75 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 0.75 | 1.00 | 0.75 | 0.75 | 0.75 |
| front_back:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:number | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |
| front_back:short:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| front_back:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 |

## Cross-model Difficult-format Core

### Difficult formats by model

| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | mlp_delta | k4+mlp_delta | k8+mlp_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek7b | 144 | 0.076 | 0.076 | 0.066 | 0.073 | 0.069 | 0.074 | 0.062 | 0.044 | +0.000 | -0.010 | -0.003 | -0.015 | -0.032 |
| glm4 | 144 | 0.131 | 0.129 | 0.130 | 0.132 | 0.132 | 0.204 | 0.195 | 0.200 | -0.002 | -0.001 | +0.073 | +0.064 | +0.069 |
| qwen3 | 144 | 0.218 | 0.217 | 0.219 | 0.217 | 0.213 | 0.199 | 0.195 | 0.190 | -0.001 | +0.001 | -0.019 | -0.023 | -0.028 |

