# Phase 158 Cross-model Step-wise Competition Trace Summary

## qwen3

cases=180, attention=L36, mlp=L36, heads=32, steps=3, top_k=20

### All / difficult / multiple-choice

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.368 | 0.353 | 0.347 | 0.360 | -0.015 | -0.022 | -0.008 | -0.294 | -0.398 | -6.554 | -5.438 | 0.231 | 0.077 | 0.000 | 0.096 | correct_surface | correct_surface | correct_surface |
| difficult_formats | 144 | 0.218 | 0.199 | 0.190 | 0.209 | -0.019 | -0.028 | -0.009 | -0.644 | -0.628 | -6.016 | -6.143 | 0.155 | 0.002 | 0.000 | 0.120 | fragment_trap | fragment_trap | fragment_trap |
| multiple_choice_control | 36 | 0.969 | 0.972 | 0.972 | 0.962 | +0.003 | +0.003 | -0.007 | 1.109 | 0.519 | -8.709 | -2.621 | 0.538 | 0.378 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |

### By format

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.149 | 0.101 | 0.118 | 0.146 | -0.049 | -0.031 | -0.003 | -1.064 | -0.988 | -4.317 | -7.100 | 0.108 | 0.000 | 0.000 | 0.413 | other | other | other |
| label_colon | 36 | 0.361 | 0.323 | 0.309 | 0.337 | -0.038 | -0.052 | -0.024 | 0.166 | 0.406 | -5.330 | -5.163 | 0.299 | 0.000 | 0.000 | 0.042 | correct_surface | correct_surface | fragment_trap |
| list_answer | 36 | 0.208 | 0.215 | 0.191 | 0.201 | +0.007 | -0.017 | -0.007 | -0.889 | -1.187 | -4.902 | -5.256 | 0.052 | 0.000 | 0.000 | 0.000 | object_copy_trap | object_copy_trap | other |
| multiple_choice | 36 | 0.969 | 0.972 | 0.972 | 0.962 | +0.003 | +0.003 | -0.007 | 1.109 | 0.519 | -8.709 | -2.621 | 0.538 | 0.378 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |
| quoted_answer | 36 | 0.153 | 0.156 | 0.142 | 0.153 | +0.003 | -0.010 | +0.000 | -0.790 | -0.742 | -9.514 | -7.051 | 0.160 | 0.007 | 0.000 | 0.024 | fragment_trap | fragment_trap | fragment_trap |

### By category

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.362 | 0.346 | 0.358 | 0.342 | -0.017 | -0.004 | -0.021 | -1.589 | -1.460 | -6.616 | -5.649 | 0.129 | 0.163 | 0.000 | 0.067 | correct_surface | correct_surface | correct_surface |
| container | 30 | 0.358 | 0.371 | 0.367 | 0.354 | +0.013 | +0.008 | -0.004 | 0.372 | 0.657 | -6.599 | -5.185 | 0.325 | 0.013 | 0.000 | 0.079 | correct_surface | correct_surface | correct_surface |
| furniture | 30 | 0.379 | 0.392 | 0.388 | 0.367 | +0.013 | +0.008 | -0.013 | -1.717 | -1.537 | -6.428 | -5.828 | 0.171 | 0.183 | 0.000 | 0.058 | correct_surface | correct_surface | correct_surface |
| number | 30 | 0.254 | 0.225 | 0.208 | 0.263 | -0.029 | -0.046 | +0.008 | -0.327 | -0.858 | -7.043 | -5.961 | 0.146 | 0.075 | 0.000 | 0.171 | other | other | other |
| plant | 30 | 0.521 | 0.463 | 0.425 | 0.508 | -0.058 | -0.096 | -0.013 | 1.544 | 0.857 | -5.749 | -5.365 | 0.371 | 0.008 | 0.000 | 0.087 | correct_surface | correct_surface | correct_surface |
| time | 30 | 0.333 | 0.325 | 0.333 | 0.325 | -0.008 | +0.000 | -0.008 | -0.045 | -0.049 | -6.891 | -4.642 | 0.246 | 0.021 | 0.000 | 0.113 | correct_surface | correct_surface | correct_surface |

### By family

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.248 | 0.250 | 0.240 | 0.242 | +0.002 | -0.008 | -0.006 | -0.334 | -0.523 | -6.975 | -5.503 | 0.133 | 0.079 | 0.000 | 0.002 | fragment_trap | fragment_trap | fragment_trap |
| neutral | 60 | 0.292 | 0.279 | 0.283 | 0.281 | -0.013 | -0.008 | -0.010 | -1.182 | -1.076 | -6.706 | -6.566 | 0.169 | 0.096 | 0.000 | 0.158 | other | other | correct_surface |
| short | 60 | 0.565 | 0.531 | 0.517 | 0.556 | -0.033 | -0.048 | -0.008 | 0.635 | 0.404 | -5.982 | -4.246 | 0.392 | 0.056 | 0.000 | 0.127 | correct_surface | correct_surface | correct_surface |

### By split

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.392 | 0.358 | 0.362 | 0.382 | -0.033 | -0.029 | -0.010 | -0.268 | -0.330 | -6.427 | -5.462 | 0.218 | 0.078 | 0.000 | 0.113 | correct_surface | correct_surface | correct_surface |
| front_back | 90 | 0.344 | 0.349 | 0.331 | 0.338 | +0.004 | -0.014 | -0.007 | -0.319 | -0.467 | -6.682 | -5.415 | 0.244 | 0.076 | 0.000 | 0.079 | correct_surface | correct_surface | correct_surface |

### Cases

| case | clean | mlp | k8+mlp | random | clean_traj | mlp_traj | k8+mlp_traj |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.50 | fragment_trap:0.88 |
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | fragment_trap:0.62 |
| back_front:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.75 | fragment_trap:0.62 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.62 | fragment_trap:0.62 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:label_colon:clothing | 0.38 | 0.50 | 0.50 | 0.25 | fragment_trap:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:long:label_colon:container | 0.12 | 0.00 | 0.00 | 0.12 | fragment_trap:0.88 | fragment_trap:0.75 | other:0.62 |
| back_front:long:label_colon:furniture | 0.12 | 0.25 | 0.38 | 0.12 | fragment_trap:0.50 | other:0.50 | other:0.50 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:label_colon:plant | 0.38 | 0.38 | 0.38 | 0.25 | fragment_trap:0.38 | other:0.38 | other:0.38 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.62 | other:0.62 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | fragment_trap:0.50 | other:0.62 |
| back_front:long:list_answer:container | 0.50 | 0.50 | 0.50 | 0.50 | format_then_answer:0.50 | format_then_answer:0.50 | format_then_answer:0.50 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.62 | other:0.50 |
| back_front:long:list_answer:plant | 0.25 | 0.25 | 0.12 | 0.25 | fragment_trap:0.50 | fragment_trap:0.62 | fragment_trap:0.88 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.50 | other:0.62 | other:1.00 |
| back_front:long:multiple_choice:clothing | 1.00 | 0.88 | 0.88 | 0.88 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:furniture | 0.88 | 1.00 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.88 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.88 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.75 | other:0.50 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.12 | 0.00 | other:1.00 | other:0.75 | fragment_trap:0.25 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:0.88 | other:0.62 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:1.00 | other:1.00 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.12 | 0.12 | other:0.88 | other:0.62 | object_copy_trap:0.62 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:0.88 | other:0.38 |
| back_front:neutral:label_colon:clothing | 0.75 | 0.75 | 0.75 | 0.75 | correct_surface:0.75 | correct_surface:0.75 | correct_surface:0.75 |
| back_front:neutral:label_colon:container | 0.25 | 0.12 | 0.12 | 0.12 | other:0.50 | other:0.38 | other:0.38 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.62 | object_copy_trap:0.50 | object_copy_trap:0.50 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:1.00 | other:1.00 |
| back_front:neutral:label_colon:plant | 0.62 | 0.12 | 0.12 | 0.75 | correct_surface:0.62 | other:0.38 | fragment_trap:0.50 |
| back_front:neutral:label_colon:time | 0.25 | 0.25 | 0.38 | 0.12 | other:0.75 | other:0.62 | other:0.50 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.75 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| back_front:neutral:list_answer:container | 0.50 | 0.38 | 0.50 | 0.50 | correct_surface:0.50 | correct_surface:0.38 | correct_surface:0.38 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | other:0.62 | other:0.38 | generic_continuation_trap:0.50 |
| back_front:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | generic_continuation_trap:1.00 | generic_continuation_trap:1.00 |
| back_front:neutral:list_answer:plant | 0.25 | 0.12 | 0.12 | 0.25 | object_copy_trap:0.50 | object_copy_trap:0.62 | other:0.38 |
| back_front:neutral:list_answer:time | 0.38 | 0.25 | 0.25 | 0.38 | correct_surface:0.38 | correct_surface:0.25 | generic_continuation_trap:0.38 |
| back_front:neutral:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:furniture | 0.88 | 0.88 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:neutral:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:time | 0.88 | 0.88 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.75 | other:0.38 | generic_continuation_trap:0.50 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:1.00 | generic_continuation_trap:0.50 | fragment_trap:0.62 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.75 | generic_continuation_trap:0.50 | fragment_trap:0.62 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:1.00 | quote_path_failure:1.00 | quote_path_failure:1.00 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.50 | object_copy_trap:0.38 | generic_continuation_trap:0.38 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | quote_path_failure:0.75 | quote_path_failure:0.75 | quote_path_failure:0.62 |
| back_front:short:answer_one_word:clothing | 0.50 | 0.25 | 0.38 | 0.38 | correct_surface:0.50 | other:0.62 | other:0.50 |
| back_front:short:answer_one_word:container | 0.50 | 0.50 | 0.50 | 0.62 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:short:answer_one_word:furniture | 0.50 | 0.50 | 0.62 | 0.38 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.62 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:1.00 | other:1.00 |
| back_front:short:answer_one_word:plant | 0.88 | 0.12 | 0.12 | 0.88 | correct_surface:0.88 | other:0.88 | other:0.88 |
| back_front:short:answer_one_word:time | 0.25 | 0.12 | 0.38 | 0.25 | other:0.75 | other:0.88 | other:0.62 |
| back_front:short:label_colon:clothing | 0.75 | 0.50 | 0.50 | 0.75 | correct_surface:0.75 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:short:label_colon:container | 0.75 | 0.88 | 0.88 | 0.75 | correct_surface:0.75 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:short:label_colon:furniture | 0.62 | 0.62 | 0.62 | 0.62 | correct_surface:0.62 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:label_colon:number | 0.50 | 0.00 | 0.00 | 0.62 | other:0.50 | generic_continuation_trap:1.00 | generic_continuation_trap:1.00 |
| back_front:short:label_colon:plant | 0.75 | 0.88 | 0.62 | 0.75 | correct_surface:0.75 | correct_surface:0.88 | correct_surface:0.62 |
| back_front:short:label_colon:time | 0.62 | 0.75 | 0.75 | 0.75 | correct_surface:0.62 | correct_surface:0.75 | correct_surface:0.75 |
| back_front:short:list_answer:clothing | 0.62 | 0.50 | 0.50 | 0.50 | correct_surface:0.62 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:short:list_answer:container | 0.75 | 0.62 | 0.62 | 0.62 | correct_surface:0.75 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:list_answer:furniture | 0.12 | 0.38 | 0.25 | 0.12 | object_copy_trap:0.38 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| back_front:short:list_answer:number | 0.25 | 0.00 | 0.00 | 0.00 | other:0.62 | other:1.00 | other:0.75 |
| back_front:short:list_answer:plant | 0.62 | 0.88 | 0.75 | 0.75 | correct_surface:0.62 | correct_surface:0.88 | correct_surface:0.75 |
| back_front:short:list_answer:time | 0.50 | 0.50 | 0.50 | 0.50 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:short:multiple_choice:clothing | 0.88 | 0.88 | 1.00 | 0.75 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:1.00 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:quoted_answer:clothing | 0.12 | 0.12 | 0.12 | 0.12 | fragment_trap:0.62 | fragment_trap:0.38 | fragment_trap:0.38 |
| back_front:short:quoted_answer:container | 0.75 | 0.62 | 0.62 | 0.75 | correct_surface:0.75 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:quoted_answer:furniture | 0.75 | 0.75 | 0.75 | 0.62 | correct_surface:0.75 | correct_surface:0.75 | correct_surface:0.75 |
| back_front:short:quoted_answer:number | 0.38 | 0.50 | 0.50 | 0.50 | object_copy_trap:0.62 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:short:quoted_answer:plant | 0.88 | 0.25 | 0.25 | 0.88 | correct_surface:0.88 | object_copy_trap:0.38 | object_copy_trap:0.50 |
| back_front:short:quoted_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.50 | object_copy_trap:0.38 | object_copy_trap:0.50 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.62 | fragment_trap:0.75 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.88 |
| front_back:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.75 | other:0.75 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.75 | other:0.50 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:label_colon:clothing | 0.38 | 0.38 | 0.38 | 0.38 | fragment_trap:0.50 | fragment_trap:0.50 | fragment_trap:0.50 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:1.00 | fragment_trap:0.75 |
| front_back:long:label_colon:furniture | 0.62 | 0.38 | 0.12 | 0.75 | correct_surface:0.62 | fragment_trap:0.62 | fragment_trap:0.62 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.88 |
| front_back:long:label_colon:plant | 0.25 | 0.12 | 0.00 | 0.12 | other:0.38 | fragment_trap:0.62 | other:0.50 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.62 | other:0.62 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.38 | object_copy_trap:0.50 | object_copy_trap:0.50 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | object_copy_trap:0.75 | object_copy_trap:0.38 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | fragment_trap:0.75 | other:0.75 |
| front_back:long:list_answer:plant | 0.25 | 0.38 | 0.25 | 0.25 | other:0.25 | fragment_trap:0.38 | fragment_trap:0.38 |
| front_back:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.75 | generic_continuation_trap:0.38 | other:0.75 |
| front_back:long:multiple_choice:clothing | 0.88 | 1.00 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:number | 0.88 | 1.00 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | object_copy_trap:0.75 | object_copy_trap:0.88 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.75 | other:0.75 | other:0.50 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | other:0.75 | other:0.62 | fragment_trap:0.38 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.88 | other:0.88 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.25 | 0.25 | 0.00 | other:0.88 | other:0.62 | other:0.38 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.12 | 0.00 | other:1.00 | other:0.50 | object_copy_trap:0.50 |
| front_back:neutral:label_colon:clothing | 0.50 | 0.50 | 0.62 | 0.50 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.62 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.38 | other:0.50 | other:0.50 |
| front_back:neutral:label_colon:furniture | 0.38 | 0.38 | 0.25 | 0.38 | other:0.38 | other:0.38 | other:0.38 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:0.50 | generic_continuation_trap:0.38 |
| front_back:neutral:label_colon:plant | 0.25 | 0.25 | 0.38 | 0.00 | other:0.62 | other:0.62 | other:0.38 |
| front_back:neutral:label_colon:time | 0.25 | 0.25 | 0.12 | 0.12 | other:0.75 | other:0.62 | fragment_trap:0.50 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.50 | object_copy_trap:0.62 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | other:0.62 | other:0.50 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.38 | object_copy_trap:0.50 |
| front_back:neutral:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | object_copy_trap:0.50 | generic_continuation_trap:0.38 |
| front_back:neutral:list_answer:plant | 0.38 | 0.38 | 0.25 | 0.38 | generic_continuation_trap:0.50 | generic_continuation_trap:0.38 | generic_continuation_trap:0.38 |
| front_back:neutral:list_answer:time | 0.25 | 0.12 | 0.12 | 0.12 | generic_continuation_trap:0.50 | generic_continuation_trap:0.50 | generic_continuation_trap:0.62 |
| front_back:neutral:multiple_choice:clothing | 1.00 | 1.00 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.88 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:neutral:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.62 | generic_continuation_trap:0.38 | generic_continuation_trap:0.50 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.75 | fragment_trap:0.62 | fragment_trap:0.62 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.12 | 0.12 | quote_path_failure:0.62 | generic_continuation_trap:0.62 | generic_continuation_trap:0.62 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.62 | quote_path_failure:0.50 | fragment_trap:0.38 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.50 | 0.38 | 0.25 | object_copy_trap:0.38 | correct_surface:0.50 | fragment_trap:0.38 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:0.38 | generic_continuation_trap:0.38 | fragment_trap:0.38 |
| front_back:short:answer_one_word:clothing | 0.38 | 0.38 | 0.38 | 0.38 | correct_surface:0.38 | other:0.50 | correct_surface:0.38 |
| front_back:short:answer_one_word:container | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.38 | other:0.38 | other:0.50 |
| front_back:short:answer_one_word:furniture | 0.75 | 0.62 | 0.62 | 0.75 | correct_surface:0.75 | correct_surface:0.62 | correct_surface:0.62 |
| front_back:short:answer_one_word:number | 0.25 | 0.00 | 0.00 | 0.25 | other:0.75 | other:1.00 | other:1.00 |
| front_back:short:answer_one_word:plant | 1.00 | 0.50 | 0.50 | 1.00 | correct_surface:1.00 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:answer_one_word:time | 0.12 | 0.12 | 0.00 | 0.12 | other:0.62 | other:0.75 | other:1.00 |
| front_back:short:label_colon:clothing | 0.50 | 0.50 | 0.50 | 0.50 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:label_colon:container | 0.25 | 0.62 | 0.50 | 0.12 | other:0.38 | correct_surface:0.62 | correct_surface:0.50 |
| front_back:short:label_colon:furniture | 0.88 | 0.62 | 0.62 | 0.50 | correct_surface:0.88 | correct_surface:0.62 | correct_surface:0.62 |
| front_back:short:label_colon:number | 0.38 | 0.12 | 0.12 | 0.50 | other:0.50 | other:0.75 | other:0.50 |
| front_back:short:label_colon:plant | 0.88 | 0.88 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:short:label_colon:time | 0.50 | 0.50 | 0.50 | 0.50 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:list_answer:clothing | 0.38 | 0.38 | 0.38 | 0.38 | correct_surface:0.38 | correct_surface:0.38 | correct_surface:0.38 |
| front_back:short:list_answer:container | 0.25 | 0.62 | 0.50 | 0.38 | other:0.38 | correct_surface:0.62 | correct_surface:0.50 |
| front_back:short:list_answer:furniture | 0.00 | 0.25 | 0.25 | 0.00 | other:0.38 | object_copy_trap:0.38 | other:0.38 |
| front_back:short:list_answer:number | 0.00 | 0.00 | 0.00 | 0.12 | other:0.50 | other:0.75 | other:0.75 |
| front_back:short:list_answer:plant | 0.88 | 0.88 | 0.62 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:0.62 |
| front_back:short:list_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | correct_surface:0.38 | other:0.50 | other:0.50 |
| front_back:short:multiple_choice:clothing | 0.88 | 0.88 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:1.00 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:number | 0.88 | 0.75 | 0.75 | 0.88 | correct_surface:0.88 | correct_surface:0.75 | correct_surface:0.75 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.88 | object_copy_trap:0.88 |
| front_back:short:quoted_answer:container | 0.00 | 0.12 | 0.00 | 0.00 | object_copy_trap:0.62 | fragment_trap:0.38 | object_copy_trap:0.38 |
| front_back:short:quoted_answer:furniture | 0.75 | 1.00 | 1.00 | 0.75 | correct_surface:0.75 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:quoted_answer:number | 0.25 | 0.50 | 0.12 | 0.25 | object_copy_trap:0.50 | correct_surface:0.50 | object_copy_trap:0.75 |
| front_back:short:quoted_answer:plant | 0.50 | 0.50 | 0.50 | 0.50 | correct_surface:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:quoted_answer:time | 0.38 | 0.38 | 0.38 | 0.38 | correct_surface:0.38 | correct_surface:0.38 | correct_surface:0.38 |

## glm4

cases=180, attention=L39, mlp=L40, heads=32, steps=3, top_k=20

### All / difficult / multiple-choice

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.297 | 0.315 | 0.320 | 0.301 | +0.017 | +0.023 | +0.004 | -0.359 | -0.353 | -4.745 | -6.261 | 0.222 | 0.060 | 0.001 | 0.000 | fragment_trap | correct_surface | correct_surface |
| difficult_formats | 144 | 0.131 | 0.204 | 0.200 | 0.134 | +0.073 | +0.069 | +0.003 | -0.684 | -0.658 | -4.922 | -6.779 | 0.184 | 0.014 | 0.001 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| multiple_choice_control | 36 | 0.962 | 0.757 | 0.802 | 0.972 | -0.205 | -0.160 | +0.010 | 0.940 | 0.863 | -4.038 | -4.189 | 0.375 | 0.247 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |

### By format

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.056 | 0.122 | 0.125 | 0.059 | +0.066 | +0.069 | +0.003 | -0.685 | -0.502 | -5.977 | -8.280 | 0.122 | 0.000 | 0.003 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| label_colon | 36 | 0.236 | 0.441 | 0.434 | 0.250 | +0.205 | +0.198 | +0.014 | 0.821 | 0.869 | -3.837 | -4.319 | 0.403 | 0.049 | 0.000 | 0.000 | fragment_trap | correct_surface | correct_surface |
| list_answer | 36 | 0.146 | 0.156 | 0.139 | 0.135 | +0.010 | -0.007 | -0.010 | -1.482 | -1.562 | -4.889 | -8.398 | 0.128 | 0.007 | 0.000 | 0.000 | other | other | other |
| multiple_choice | 36 | 0.962 | 0.757 | 0.802 | 0.972 | -0.205 | -0.160 | +0.010 | 0.940 | 0.863 | -4.038 | -4.189 | 0.375 | 0.247 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |
| quoted_answer | 36 | 0.087 | 0.097 | 0.101 | 0.090 | +0.010 | +0.014 | +0.003 | -1.387 | -1.435 | -4.984 | -6.119 | 0.083 | 0.000 | 0.000 | 0.000 | fragment_trap | fragment_trap | fragment_trap |

### By category

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.275 | 0.200 | 0.242 | 0.296 | -0.075 | -0.033 | +0.021 | -1.316 | -1.770 | -4.680 | -7.481 | 0.075 | 0.113 | 0.000 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| container | 30 | 0.296 | 0.392 | 0.388 | 0.287 | +0.096 | +0.092 | -0.008 | 0.214 | 0.142 | -4.275 | -5.669 | 0.317 | 0.021 | 0.000 | 0.000 | fragment_trap | correct_surface | correct_surface |
| furniture | 30 | 0.317 | 0.196 | 0.208 | 0.317 | -0.121 | -0.108 | +0.000 | -1.585 | -1.900 | -5.434 | -7.430 | 0.029 | 0.100 | 0.000 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| number | 30 | 0.212 | 0.279 | 0.275 | 0.212 | +0.067 | +0.063 | +0.000 | -0.336 | 0.057 | -4.633 | -6.191 | 0.188 | 0.062 | 0.000 | 0.000 | fragment_trap | correct_surface | fragment_trap |
| plant | 30 | 0.379 | 0.537 | 0.525 | 0.392 | +0.158 | +0.146 | +0.013 | 0.982 | 1.533 | -4.491 | -4.893 | 0.458 | 0.013 | 0.000 | 0.000 | fragment_trap | correct_surface | correct_surface |
| time | 30 | 0.304 | 0.283 | 0.283 | 0.304 | -0.021 | -0.021 | +0.000 | -0.112 | -0.183 | -4.957 | -5.904 | 0.267 | 0.054 | 0.004 | 0.000 | fragment_trap | correct_surface | correct_surface |

### By family

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.212 | 0.248 | 0.252 | 0.217 | +0.035 | +0.040 | +0.004 | -0.961 | -1.247 | -4.901 | -7.625 | 0.125 | 0.017 | 0.000 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| neutral | 60 | 0.267 | 0.258 | 0.240 | 0.273 | -0.008 | -0.027 | +0.006 | -0.670 | -0.483 | -4.979 | -5.710 | 0.185 | 0.098 | 0.000 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| short | 60 | 0.412 | 0.438 | 0.469 | 0.415 | +0.025 | +0.056 | +0.002 | 0.555 | 0.669 | -4.355 | -5.449 | 0.356 | 0.067 | 0.002 | 0.000 | correct_surface | correct_surface | correct_surface |

### By split

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.306 | 0.318 | 0.322 | 0.310 | +0.012 | +0.017 | +0.004 | -0.378 | -0.233 | -4.958 | -6.224 | 0.225 | 0.056 | 0.001 | 0.000 | fragment_trap | correct_surface | correct_surface |
| front_back | 90 | 0.289 | 0.311 | 0.318 | 0.293 | +0.022 | +0.029 | +0.004 | -0.339 | -0.474 | -4.532 | -6.298 | 0.219 | 0.065 | 0.000 | 0.000 | fragment_trap | correct_surface | correct_surface |

### Cases

| case | clean | mlp | k8+mlp | random | clean_traj | mlp_traj | k8+mlp_traj |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | fragment_trap:0.75 |
| back_front:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.62 | other:0.75 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:1.00 |
| back_front:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.62 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.50 |
| back_front:long:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.12 | fragment_trap:0.62 | generic_continuation_trap:0.50 | generic_continuation_trap:0.50 |
| back_front:long:label_colon:container | 0.00 | 0.50 | 0.38 | 0.00 | fragment_trap:0.62 | correct_surface:0.50 | correct_surface:0.38 |
| back_front:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | generic_continuation_trap:0.50 | generic_continuation_trap:0.50 |
| back_front:long:label_colon:number | 0.00 | 0.75 | 0.75 | 0.00 | fragment_trap:1.00 | correct_surface:0.75 | correct_surface:0.75 |
| back_front:long:label_colon:plant | 0.12 | 0.88 | 0.88 | 0.12 | fragment_trap:0.62 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | generic_continuation_trap:0.62 | generic_continuation_trap:0.62 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | object_copy_trap:0.50 | object_copy_trap:0.50 |
| back_front:long:list_answer:container | 0.25 | 0.25 | 0.12 | 0.12 | other:0.50 | other:0.75 | other:0.88 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | other:0.88 | other:0.88 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:1.00 | other:0.62 |
| back_front:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | fragment_trap:0.50 | other:0.75 | other:0.75 |
| back_front:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | other:0.75 | other:1.00 | other:1.00 |
| back_front:long:multiple_choice:clothing | 1.00 | 0.88 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:long:multiple_choice:container | 1.00 | 0.88 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.88 |
| back_front:long:quoted_answer:container | 0.00 | 0.00 | 0.12 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.88 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.88 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.88 |
| back_front:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | other:0.62 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.88 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.62 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | fragment_trap:0.50 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.62 |
| back_front:neutral:label_colon:clothing | 0.12 | 0.00 | 0.00 | 0.12 | object_copy_trap:0.38 | fragment_trap:0.50 | fragment_trap:0.50 |
| back_front:neutral:label_colon:container | 0.62 | 0.50 | 0.50 | 0.62 | correct_surface:0.62 | correct_surface:0.50 | fragment_trap:0.50 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.00 | 0.00 | 0.12 | object_copy_trap:0.88 | object_copy_trap:0.38 | object_copy_trap:0.62 |
| back_front:neutral:label_colon:number | 0.12 | 0.25 | 0.00 | 0.12 | fragment_trap:0.50 | fragment_trap:0.38 | fragment_trap:0.50 |
| back_front:neutral:label_colon:plant | 0.25 | 0.88 | 0.88 | 0.25 | fragment_trap:0.62 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:neutral:label_colon:time | 0.12 | 0.38 | 0.38 | 0.12 | object_copy_trap:0.75 | correct_surface:0.38 | correct_surface:0.38 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | list_path_failure:0.38 | other:0.62 | fragment_trap:0.50 |
| back_front:neutral:list_answer:container | 0.12 | 0.25 | 0.25 | 0.12 | list_path_failure:0.50 | fragment_trap:0.25 | list_path_failure:0.50 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | fragment_trap:0.50 | fragment_trap:0.75 |
| back_front:neutral:list_answer:number | 0.00 | 0.25 | 0.00 | 0.00 | list_path_failure:0.88 | other:0.75 | other:0.88 |
| back_front:neutral:list_answer:plant | 0.25 | 0.50 | 0.38 | 0.12 | object_copy_trap:0.38 | correct_surface:0.50 | correct_surface:0.38 |
| back_front:neutral:list_answer:time | 0.12 | 0.00 | 0.12 | 0.12 | object_copy_trap:0.62 | wrong_semantic:0.62 | other:0.62 |
| back_front:neutral:multiple_choice:clothing | 0.88 | 0.75 | 0.75 | 0.88 | correct_surface:0.88 | correct_surface:0.75 | correct_surface:0.75 |
| back_front:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:furniture | 1.00 | 0.88 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:neutral:multiple_choice:number | 1.00 | 0.25 | 0.38 | 1.00 | correct_surface:1.00 | fragment_trap:0.62 | fragment_trap:0.62 |
| back_front:neutral:multiple_choice:plant | 1.00 | 0.38 | 0.50 | 1.00 | correct_surface:1.00 | fragment_trap:0.38 | correct_surface:0.50 |
| back_front:neutral:multiple_choice:time | 0.88 | 0.50 | 0.50 | 1.00 | correct_surface:0.88 | correct_surface:0.50 | correct_surface:0.50 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | fragment_trap:0.88 | fragment_trap:0.88 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.25 | 0.25 | 0.00 | fragment_trap:0.50 | fragment_trap:0.62 | fragment_trap:0.50 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | fragment_trap:0.50 | fragment_trap:0.50 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | quote_path_failure:1.00 | quote_path_failure:0.62 | quote_path_failure:0.62 |
| back_front:neutral:quoted_answer:plant | 0.25 | 0.25 | 0.12 | 0.25 | fragment_trap:0.62 | fragment_trap:0.50 | fragment_trap:0.75 |
| back_front:neutral:quoted_answer:time | 0.12 | 0.00 | 0.00 | 0.12 | object_copy_trap:0.62 | quote_path_failure:0.75 | quote_path_failure:0.75 |
| back_front:short:answer_one_word:clothing | 0.12 | 0.12 | 0.38 | 0.12 | object_copy_trap:0.62 | object_copy_trap:0.50 | correct_surface:0.38 |
| back_front:short:answer_one_word:container | 0.62 | 0.88 | 0.88 | 0.62 | correct_surface:0.62 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.62 | object_copy_trap:0.50 |
| back_front:short:answer_one_word:number | 0.12 | 0.50 | 0.38 | 0.12 | object_copy_trap:0.75 | correct_surface:0.50 | object_copy_trap:0.50 |
| back_front:short:answer_one_word:plant | 0.38 | 1.00 | 1.00 | 0.38 | fragment_trap:0.62 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:answer_one_word:time | 0.00 | 0.25 | 0.25 | 0.12 | object_copy_trap:0.50 | object_copy_trap:0.62 | object_copy_trap:0.50 |
| back_front:short:label_colon:clothing | 0.62 | 0.62 | 0.62 | 0.62 | correct_surface:0.62 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:label_colon:container | 0.50 | 1.00 | 1.00 | 0.38 | correct_surface:0.50 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:label_colon:furniture | 0.88 | 0.25 | 0.38 | 0.88 | correct_surface:0.88 | fragment_trap:0.38 | correct_surface:0.38 |
| back_front:short:label_colon:number | 0.12 | 0.12 | 0.12 | 0.12 | fragment_trap:0.75 | wrong_semantic:0.88 | wrong_semantic:0.88 |
| back_front:short:label_colon:plant | 0.38 | 1.00 | 1.00 | 0.62 | correct_surface:0.38 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:label_colon:time | 0.75 | 0.62 | 0.62 | 0.75 | correct_surface:0.75 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:list_answer:clothing | 0.25 | 0.12 | 0.25 | 0.25 | other:0.50 | other:0.88 | other:0.75 |
| back_front:short:list_answer:container | 0.12 | 0.38 | 0.12 | 0.12 | other:0.88 | other:0.62 | other:0.88 |
| back_front:short:list_answer:furniture | 0.25 | 0.00 | 0.00 | 0.25 | other:0.50 | other:1.00 | other:0.88 |
| back_front:short:list_answer:number | 0.00 | 0.50 | 0.50 | 0.00 | object_copy_trap:0.50 | other:0.50 | other:0.50 |
| back_front:short:list_answer:plant | 0.62 | 0.88 | 0.88 | 0.62 | correct_surface:0.62 | correct_surface:0.88 | correct_surface:0.88 |
| back_front:short:list_answer:time | 0.38 | 0.12 | 0.12 | 0.38 | correct_surface:0.38 | other:0.88 | other:0.88 |
| back_front:short:multiple_choice:clothing | 0.88 | 0.38 | 0.62 | 0.88 | correct_surface:0.88 | object_copy_trap:0.50 | correct_surface:0.62 |
| back_front:short:multiple_choice:container | 1.00 | 0.38 | 0.50 | 1.00 | correct_surface:1.00 | fragment_trap:0.50 | correct_surface:0.50 |
| back_front:short:multiple_choice:furniture | 1.00 | 0.25 | 0.38 | 1.00 | correct_surface:1.00 | fragment_trap:0.38 | correct_surface:0.38 |
| back_front:short:multiple_choice:number | 1.00 | 0.88 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:1.00 |
| back_front:short:multiple_choice:plant | 1.00 | 0.38 | 0.38 | 1.00 | correct_surface:1.00 | fragment_trap:0.50 | fragment_trap:0.38 |
| back_front:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | object_copy_trap:1.00 | object_copy_trap:1.00 |
| back_front:short:quoted_answer:container | 0.25 | 0.62 | 0.62 | 0.25 | fragment_trap:0.62 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:quoted_answer:furniture | 0.25 | 0.00 | 0.00 | 0.12 | fragment_trap:0.38 | object_copy_trap:0.88 | object_copy_trap:0.75 |
| back_front:short:quoted_answer:number | 0.12 | 0.00 | 0.12 | 0.12 | fragment_trap:0.88 | object_copy_trap:0.88 | object_copy_trap:0.88 |
| back_front:short:quoted_answer:plant | 0.25 | 0.62 | 0.62 | 0.50 | correct_surface:0.25 | correct_surface:0.62 | correct_surface:0.62 |
| back_front:short:quoted_answer:time | 0.12 | 0.25 | 0.25 | 0.12 | generic_continuation_trap:0.38 | object_copy_trap:0.62 | object_copy_trap:0.62 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.75 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.88 | other:0.88 |
| front_back:long:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.88 | other:0.75 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.75 | other:0.50 |
| front_back:long:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.12 | fragment_trap:0.88 | other:0.75 | other:0.62 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.62 | fragment_trap:0.50 |
| front_back:long:label_colon:clothing | 0.25 | 0.00 | 0.12 | 0.25 | fragment_trap:0.62 | generic_continuation_trap:0.88 | generic_continuation_trap:0.75 |
| front_back:long:label_colon:container | 0.00 | 0.25 | 0.25 | 0.00 | fragment_trap:1.00 | generic_continuation_trap:0.62 | generic_continuation_trap:0.38 |
| front_back:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | generic_continuation_trap:0.62 | generic_continuation_trap:0.62 |
| front_back:long:label_colon:number | 0.00 | 0.12 | 0.25 | 0.00 | fragment_trap:1.00 | generic_continuation_trap:0.88 | generic_continuation_trap:0.75 |
| front_back:long:label_colon:plant | 0.00 | 0.88 | 0.88 | 0.00 | fragment_trap:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | generic_continuation_trap:1.00 | generic_continuation_trap:1.00 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | other:0.62 | other:0.50 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | other:0.88 | other:0.88 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | other:1.00 | other:0.88 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | other:0.88 | other:0.88 |
| front_back:long:list_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | fragment_trap:0.62 | other:0.75 | other:0.75 |
| front_back:long:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:1.00 | other:1.00 |
| front_back:long:multiple_choice:clothing | 1.00 | 0.75 | 0.75 | 1.00 | correct_surface:1.00 | correct_surface:0.75 | correct_surface:0.75 |
| front_back:long:multiple_choice:container | 1.00 | 0.88 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:number | 0.88 | 0.62 | 0.75 | 1.00 | correct_surface:0.88 | correct_surface:0.62 | correct_surface:0.75 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.75 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.88 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.88 |
| front_back:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:long:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.75 | other:1.00 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.62 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | other:0.62 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | other:0.75 |
| front_back:neutral:answer_one_word:plant | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.62 | other:0.75 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.75 | other:0.75 |
| front_back:neutral:label_colon:clothing | 0.50 | 0.25 | 0.25 | 0.50 | correct_surface:0.50 | fragment_trap:0.50 | fragment_trap:0.75 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.88 | wrong_semantic:0.62 | fragment_trap:0.62 |
| front_back:neutral:label_colon:furniture | 0.25 | 0.00 | 0.00 | 0.25 | object_copy_trap:0.38 | fragment_trap:0.62 | fragment_trap:0.62 |
| front_back:neutral:label_colon:number | 0.00 | 0.75 | 0.38 | 0.00 | object_copy_trap:0.75 | correct_surface:0.75 | correct_surface:0.38 |
| front_back:neutral:label_colon:plant | 0.25 | 0.88 | 0.75 | 0.25 | fragment_trap:0.62 | correct_surface:0.88 | correct_surface:0.75 |
| front_back:neutral:label_colon:time | 0.38 | 0.75 | 0.75 | 0.38 | fragment_trap:0.50 | correct_surface:0.75 | correct_surface:0.75 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.12 | object_copy_trap:0.50 | other:0.50 | object_copy_trap:0.38 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | other:0.62 | other:0.62 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.38 | fragment_trap:0.38 | fragment_trap:0.50 |
| front_back:neutral:list_answer:number | 0.00 | 0.12 | 0.00 | 0.00 | list_path_failure:0.75 | other:0.62 | fragment_trap:0.62 |
| front_back:neutral:list_answer:plant | 0.38 | 0.25 | 0.25 | 0.38 | object_copy_trap:0.50 | object_copy_trap:0.38 | object_copy_trap:0.50 |
| front_back:neutral:list_answer:time | 0.12 | 0.00 | 0.00 | 0.12 | object_copy_trap:0.62 | other:0.50 | object_copy_trap:0.62 |
| front_back:neutral:multiple_choice:clothing | 0.75 | 0.38 | 0.25 | 0.88 | correct_surface:0.75 | wrong_semantic:0.62 | wrong_semantic:0.75 |
| front_back:neutral:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:furniture | 0.88 | 0.50 | 0.75 | 1.00 | correct_surface:0.88 | correct_surface:0.50 | correct_surface:0.75 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.75 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.75 | correct_surface:0.88 |
| front_back:neutral:multiple_choice:plant | 1.00 | 0.88 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:neutral:multiple_choice:time | 1.00 | 0.88 | 0.62 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.62 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | fragment_trap:0.75 | fragment_trap:0.62 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.12 | 0.12 | 0.00 | object_copy_trap:0.62 | fragment_trap:0.62 | fragment_trap:0.50 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.38 | 0.25 | 0.00 | fragment_trap:0.50 | fragment_trap:0.62 | fragment_trap:0.75 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | fragment_trap:0.50 | fragment_trap:0.50 |
| front_back:neutral:quoted_answer:plant | 0.38 | 0.38 | 0.38 | 0.38 | fragment_trap:0.62 | fragment_trap:0.50 | fragment_trap:0.38 |
| front_back:neutral:quoted_answer:time | 0.25 | 0.00 | 0.00 | 0.25 | fragment_trap:0.50 | fragment_trap:0.62 | quote_path_failure:0.50 |
| front_back:short:answer_one_word:clothing | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.75 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| front_back:short:answer_one_word:container | 0.12 | 0.50 | 0.50 | 0.12 | object_copy_trap:0.50 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.50 | fragment_trap:0.50 | fragment_trap:0.50 |
| front_back:short:answer_one_word:number | 0.00 | 0.12 | 0.00 | 0.00 | object_copy_trap:1.00 | object_copy_trap:0.88 | object_copy_trap:1.00 |
| front_back:short:answer_one_word:plant | 0.38 | 0.88 | 0.88 | 0.38 | fragment_trap:0.62 | correct_surface:0.88 | correct_surface:0.88 |
| front_back:short:answer_one_word:time | 0.12 | 0.00 | 0.12 | 0.00 | object_copy_trap:0.50 | fragment_trap:0.50 | fragment_trap:0.50 |
| front_back:short:label_colon:clothing | 0.62 | 0.75 | 1.00 | 0.88 | correct_surface:0.62 | correct_surface:0.75 | correct_surface:1.00 |
| front_back:short:label_colon:container | 0.00 | 0.88 | 1.00 | 0.00 | fragment_trap:0.62 | correct_surface:0.88 | correct_surface:1.00 |
| front_back:short:label_colon:furniture | 0.88 | 0.62 | 0.62 | 0.88 | correct_surface:0.88 | correct_surface:0.62 | correct_surface:0.62 |
| front_back:short:label_colon:number | 0.00 | 0.50 | 0.50 | 0.00 | fragment_trap:0.75 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:label_colon:plant | 0.25 | 1.00 | 0.88 | 0.25 | fragment_trap:0.62 | correct_surface:1.00 | correct_surface:0.88 |
| front_back:short:label_colon:time | 0.38 | 0.50 | 0.50 | 0.38 | fragment_trap:0.62 | correct_surface:0.50 | correct_surface:0.50 |
| front_back:short:list_answer:clothing | 0.25 | 0.25 | 0.38 | 0.25 | correct_surface:0.25 | other:0.75 | other:0.62 |
| front_back:short:list_answer:container | 0.25 | 0.25 | 0.25 | 0.25 | other:0.75 | other:0.62 | other:0.75 |
| front_back:short:list_answer:furniture | 0.62 | 0.00 | 0.00 | 0.62 | correct_surface:0.62 | generic_continuation_trap:0.50 | other:0.75 |
| front_back:short:list_answer:number | 0.12 | 0.25 | 0.25 | 0.00 | other:0.50 | other:0.62 | other:0.62 |
| front_back:short:list_answer:plant | 0.75 | 1.00 | 0.88 | 0.62 | correct_surface:0.75 | correct_surface:1.00 | correct_surface:0.88 |
| front_back:short:list_answer:time | 0.12 | 0.00 | 0.00 | 0.12 | other:0.50 | other:0.62 | other:0.62 |
| front_back:short:multiple_choice:clothing | 0.75 | 0.62 | 0.88 | 0.75 | correct_surface:0.75 | correct_surface:0.62 | correct_surface:0.88 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:number | 0.88 | 0.62 | 0.88 | 0.88 | correct_surface:0.88 | correct_surface:0.62 | correct_surface:0.88 |
| front_back:short:multiple_choice:plant | 1.00 | 0.62 | 0.75 | 1.00 | correct_surface:1.00 | correct_surface:0.62 | correct_surface:0.75 |
| front_back:short:multiple_choice:time | 1.00 | 1.00 | 1.00 | 0.88 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:quoted_answer:clothing | 0.12 | 0.00 | 0.00 | 0.12 | fragment_trap:0.62 | object_copy_trap:1.00 | object_copy_trap:0.88 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | object_copy_trap:0.88 | object_copy_trap:0.75 |
| front_back:short:quoted_answer:furniture | 0.38 | 0.00 | 0.00 | 0.38 | correct_surface:0.38 | object_copy_trap:1.00 | object_copy_trap:0.75 |
| front_back:short:quoted_answer:number | 0.12 | 0.00 | 0.12 | 0.12 | fragment_trap:0.75 | object_copy_trap:1.00 | object_copy_trap:0.75 |
| front_back:short:quoted_answer:plant | 0.25 | 0.38 | 0.38 | 0.25 | fragment_trap:0.50 | object_copy_trap:0.62 | object_copy_trap:0.62 |
| front_back:short:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | object_copy_trap:0.38 | object_copy_trap:0.75 | object_copy_trap:0.75 |

## deepseek7b

cases=180, attention=L28, mlp=L28, heads=28, steps=3, top_k=20

### All / difficult / multiple-choice

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 180 | 0.235 | 0.251 | 0.216 | 0.233 | +0.017 | -0.019 | -0.002 | -1.488 | -2.314 | -7.342 | -4.849 | 0.108 | 0.106 | 0.002 | 0.094 | fragment_trap | fragment_trap | fragment_trap |
| difficult_formats | 144 | 0.076 | 0.074 | 0.044 | 0.070 | -0.003 | -0.032 | -0.006 | -1.611 | -2.333 | -6.972 | -5.636 | 0.062 | 0.000 | 0.003 | 0.118 | fragment_trap | fragment_trap | fragment_trap |
| multiple_choice_control | 36 | 0.868 | 0.962 | 0.903 | 0.882 | +0.094 | +0.035 | +0.014 | -0.995 | -2.237 | -8.819 | -1.701 | 0.295 | 0.531 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |

### By format

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answer_one_word | 36 | 0.049 | 0.056 | 0.038 | 0.049 | +0.007 | -0.010 | +0.000 | -2.096 | -3.061 | -7.857 | -4.784 | 0.031 | 0.000 | 0.010 | 0.010 | fragment_trap | other | fragment_trap |
| label_colon | 36 | 0.069 | 0.059 | 0.014 | 0.066 | -0.010 | -0.056 | -0.003 | -0.771 | -2.222 | -6.227 | -6.158 | 0.052 | 0.000 | 0.000 | 0.257 | fragment_trap | other | other |
| list_answer | 36 | 0.125 | 0.128 | 0.097 | 0.111 | +0.003 | -0.028 | -0.014 | -1.743 | -2.460 | -6.117 | -6.406 | 0.108 | 0.000 | 0.000 | 0.038 | object_copy_trap | object_copy_trap | object_copy_trap |
| multiple_choice | 36 | 0.868 | 0.962 | 0.903 | 0.882 | +0.094 | +0.035 | +0.014 | -0.995 | -2.237 | -8.819 | -1.701 | 0.295 | 0.531 | 0.000 | 0.000 | correct_surface | correct_surface | correct_surface |
| quoted_answer | 36 | 0.062 | 0.052 | 0.028 | 0.056 | -0.010 | -0.035 | -0.007 | -1.835 | -1.588 | -7.688 | -5.198 | 0.056 | 0.000 | 0.000 | 0.167 | fragment_trap | fragment_trap | fragment_trap |

### By category

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clothing | 30 | 0.208 | 0.208 | 0.196 | 0.212 | +0.000 | -0.013 | +0.004 | -2.618 | -3.610 | -7.831 | -5.597 | 0.013 | 0.179 | 0.000 | 0.100 | fragment_trap | fragment_trap | fragment_trap |
| container | 30 | 0.263 | 0.279 | 0.250 | 0.263 | +0.017 | -0.013 | +0.000 | -0.979 | -1.093 | -6.561 | -4.061 | 0.217 | 0.037 | 0.000 | 0.092 | fragment_trap | fragment_trap | fragment_trap |
| furniture | 30 | 0.183 | 0.200 | 0.196 | 0.183 | +0.017 | +0.013 | +0.000 | -2.867 | -3.764 | -7.615 | -6.099 | 0.013 | 0.154 | 0.000 | 0.087 | fragment_trap | fragment_trap | fragment_trap |
| number | 30 | 0.208 | 0.254 | 0.196 | 0.217 | +0.046 | -0.013 | +0.008 | -1.121 | -2.209 | -7.208 | -4.685 | 0.067 | 0.142 | 0.000 | 0.092 | fragment_trap | object_copy_trap | fragment_trap |
| plant | 30 | 0.325 | 0.338 | 0.267 | 0.296 | +0.013 | -0.058 | -0.029 | -0.096 | -0.766 | -7.413 | -3.998 | 0.246 | 0.021 | 0.004 | 0.087 | fragment_trap | correct_surface | fragment_trap |
| time | 30 | 0.221 | 0.229 | 0.192 | 0.225 | +0.008 | -0.029 | +0.004 | -1.246 | -2.438 | -7.421 | -4.655 | 0.096 | 0.104 | 0.008 | 0.108 | fragment_trap | other | fragment_trap |

### By family

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 60 | 0.260 | 0.250 | 0.223 | 0.265 | -0.010 | -0.038 | +0.004 | -2.273 | -2.673 | -6.753 | -3.972 | 0.100 | 0.140 | 0.000 | 0.117 | fragment_trap | fragment_trap | fragment_trap |
| neutral | 60 | 0.185 | 0.235 | 0.210 | 0.194 | +0.050 | +0.025 | +0.008 | -1.618 | -2.690 | -8.088 | -6.290 | 0.058 | 0.081 | 0.000 | 0.075 | fragment_trap | fragment_trap | other |
| short | 60 | 0.258 | 0.269 | 0.215 | 0.240 | +0.010 | -0.044 | -0.019 | -0.572 | -1.578 | -7.184 | -4.286 | 0.167 | 0.098 | 0.006 | 0.092 | fragment_trap | fragment_trap | fragment_trap |

### By split

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_front | 90 | 0.244 | 0.261 | 0.218 | 0.244 | +0.017 | -0.026 | +0.000 | -1.437 | -2.088 | -7.455 | -4.581 | 0.117 | 0.104 | 0.001 | 0.086 | fragment_trap | fragment_trap | fragment_trap |
| front_back | 90 | 0.225 | 0.242 | 0.214 | 0.221 | +0.017 | -0.011 | -0.004 | -1.539 | -2.539 | -7.229 | -5.118 | 0.100 | 0.108 | 0.003 | 0.103 | fragment_trap | fragment_trap | fragment_trap |

### Cases

| case | clean | mlp | k8+mlp | random | clean_traj | mlp_traj | k8+mlp_traj |
|---|---|---|---|---|---|---|---|
| back_front:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | other:0.62 | fragment_trap:0.62 |
| back_front:long:answer_one_word:container | 0.25 | 0.38 | 0.25 | 0.25 | other:0.62 | other:0.50 | fragment_trap:0.50 |
| back_front:long:answer_one_word:furniture | 0.25 | 0.00 | 0.00 | 0.25 | other:0.50 | other:0.75 | other:1.00 |
| back_front:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | other:1.00 | other:1.00 |
| back_front:long:answer_one_word:plant | 0.12 | 0.00 | 0.00 | 0.12 | other:0.38 | other:0.50 | object_copy_trap:0.50 |
| back_front:long:answer_one_word:time | 0.00 | 0.00 | 0.12 | 0.12 | other:0.50 | other:0.50 | fragment_trap:0.50 |
| back_front:long:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.12 | other:0.38 | fragment_trap:0.75 | fragment_trap:0.50 |
| back_front:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.12 | fragment_trap:0.75 | other:0.75 | fragment_trap:0.50 |
| back_front:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.88 | other:0.75 |
| back_front:long:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | other:0.62 |
| back_front:long:label_colon:plant | 0.00 | 0.12 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.62 | other:0.62 |
| back_front:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.50 | fragment_trap:0.50 |
| back_front:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.75 | other:0.38 |
| back_front:long:list_answer:container | 0.62 | 0.62 | 0.50 | 0.62 | correct_surface:0.62 | correct_surface:0.62 | correct_surface:0.50 |
| back_front:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.88 | object_copy_trap:0.88 | object_copy_trap:0.75 |
| back_front:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.75 | generic_continuation_trap:0.38 |
| back_front:long:list_answer:plant | 0.25 | 0.12 | 0.00 | 0.25 | object_copy_trap:0.62 | object_copy_trap:0.50 | other:0.38 |
| back_front:long:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | object_copy_trap:0.62 | object_copy_trap:0.50 | object_copy_trap:0.62 |
| back_front:long:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:number | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:multiple_choice:plant | 1.00 | 1.00 | 0.75 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.75 |
| back_front:long:multiple_choice:time | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:container | 0.62 | 0.00 | 0.00 | 0.62 | correct_surface:0.62 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.88 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:long:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:1.00 |
| back_front:long:quoted_answer:plant | 0.25 | 0.00 | 0.00 | 0.25 | object_copy_trap:0.50 | fragment_trap:1.00 | fragment_trap:0.88 |
| back_front:long:quoted_answer:time | 0.12 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | fragment_trap:1.00 | fragment_trap:0.88 |
| back_front:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.50 | other:0.50 |
| back_front:neutral:answer_one_word:container | 0.00 | 0.38 | 0.25 | 0.00 | fragment_trap:0.62 | other:0.38 | other:0.75 |
| back_front:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.50 | object_copy_trap:0.50 |
| back_front:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | other:0.62 |
| back_front:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.00 | 0.12 | fragment_trap:0.75 | object_copy_trap:0.38 | other:0.38 |
| back_front:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | other:0.38 | other:0.62 |
| back_front:neutral:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.12 | other:0.62 | other:0.50 | fragment_trap:0.50 |
| back_front:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.75 | other:0.88 | fragment_trap:0.75 |
| back_front:neutral:label_colon:furniture | 0.12 | 0.00 | 0.00 | 0.12 | other:0.62 | other:0.50 | fragment_trap:0.50 |
| back_front:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | other:1.00 | fragment_trap:0.38 | other:0.75 |
| back_front:neutral:label_colon:plant | 0.12 | 0.12 | 0.12 | 0.00 | other:0.62 | other:0.62 | other:0.50 |
| back_front:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.88 | other:1.00 |
| back_front:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.75 | other:0.50 | fragment_trap:0.50 |
| back_front:neutral:list_answer:container | 0.38 | 0.38 | 0.38 | 0.38 | other:0.38 | other:0.50 | correct_surface:0.38 |
| back_front:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.25 | object_copy_trap:0.62 | object_copy_trap:0.75 |
| back_front:neutral:list_answer:number | 0.12 | 0.50 | 0.38 | 0.25 | generic_continuation_trap:0.75 | correct_surface:0.50 | correct_surface:0.38 |
| back_front:neutral:list_answer:plant | 0.38 | 0.25 | 0.25 | 0.38 | correct_surface:0.25 | correct_surface:0.25 | correct_surface:0.25 |
| back_front:neutral:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.75 | other:0.50 | generic_continuation_trap:0.38 |
| back_front:neutral:multiple_choice:clothing | 0.62 | 1.00 | 1.00 | 0.62 | correct_surface:0.62 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:container | 0.75 | 1.00 | 1.00 | 1.00 | correct_surface:0.75 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:furniture | 0.50 | 1.00 | 1.00 | 0.50 | correct_surface:0.50 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:neutral:multiple_choice:number | 0.88 | 0.88 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:0.62 | correct_surface:0.88 |
| back_front:neutral:multiple_choice:plant | 0.62 | 0.88 | 0.75 | 0.75 | correct_surface:0.50 | correct_surface:0.88 | correct_surface:0.75 |
| back_front:neutral:multiple_choice:time | 0.62 | 1.00 | 0.88 | 0.75 | correct_surface:0.50 | correct_surface:1.00 | correct_surface:0.88 |
| back_front:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.50 | fragment_trap:0.75 |
| back_front:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:1.00 |
| back_front:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.75 | fragment_trap:0.50 |
| back_front:neutral:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | quote_path_failure:0.62 | fragment_trap:0.50 |
| back_front:neutral:quoted_answer:plant | 0.12 | 0.12 | 0.00 | 0.12 | fragment_trap:0.75 | fragment_trap:0.62 | fragment_trap:0.62 |
| back_front:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | fragment_trap:0.88 | fragment_trap:0.75 |
| back_front:short:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.62 | fragment_trap:0.38 | fragment_trap:0.50 |
| back_front:short:answer_one_word:container | 0.12 | 0.00 | 0.12 | 0.12 | fragment_trap:0.88 | fragment_trap:0.62 | fragment_trap:0.75 |
| back_front:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | fragment_trap:0.62 | fragment_trap:0.62 |
| back_front:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | fragment_trap:0.50 | fragment_trap:0.88 |
| back_front:short:answer_one_word:plant | 0.00 | 0.25 | 0.12 | 0.00 | other:0.62 | other:0.50 | other:0.75 |
| back_front:short:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | object_copy_trap:0.50 | object_copy_trap:0.62 |
| back_front:short:label_colon:clothing | 0.38 | 0.00 | 0.00 | 0.38 | fragment_trap:0.62 | fragment_trap:0.38 | punctuation_trap:1.00 |
| back_front:short:label_colon:container | 0.00 | 0.12 | 0.00 | 0.00 | fragment_trap:0.88 | punctuation_trap:0.25 | punctuation_trap:0.62 |
| back_front:short:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | punctuation_trap:0.50 | punctuation_trap:0.88 |
| back_front:short:label_colon:number | 0.50 | 0.50 | 0.00 | 0.38 | correct_surface:0.50 | correct_surface:0.50 | punctuation_trap:0.88 |
| back_front:short:label_colon:plant | 0.25 | 0.50 | 0.12 | 0.25 | fragment_trap:0.62 | correct_surface:0.50 | other:0.38 |
| back_front:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | fragment_trap:0.62 | punctuation_trap:0.75 |
| back_front:short:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | other:0.25 | other:0.50 |
| back_front:short:list_answer:container | 0.25 | 0.50 | 0.25 | 0.00 | other:0.38 | correct_surface:0.50 | other:0.50 |
| back_front:short:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | list_path_failure:0.50 | object_copy_trap:0.25 | list_path_failure:0.25 |
| back_front:short:list_answer:number | 0.12 | 0.12 | 0.00 | 0.12 | object_copy_trap:0.88 | object_copy_trap:0.88 | object_copy_trap:0.50 |
| back_front:short:list_answer:plant | 0.25 | 0.25 | 0.12 | 0.12 | list_path_failure:0.38 | object_copy_trap:0.38 | fragment_trap:0.50 |
| back_front:short:list_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.38 | object_copy_trap:0.50 | generic_continuation_trap:0.38 |
| back_front:short:multiple_choice:clothing | 1.00 | 1.00 | 0.75 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.75 |
| back_front:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 0.88 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| back_front:short:multiple_choice:furniture | 1.00 | 1.00 | 0.88 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.88 |
| back_front:short:multiple_choice:number | 0.88 | 0.88 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:0.88 | correct_surface:1.00 |
| back_front:short:multiple_choice:plant | 0.88 | 1.00 | 0.88 | 0.75 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:0.88 |
| back_front:short:multiple_choice:time | 1.00 | 0.88 | 0.50 | 1.00 | correct_surface:1.00 | correct_surface:0.88 | correct_surface:0.50 |
| back_front:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.75 | fragment_trap:0.88 |
| back_front:short:quoted_answer:container | 0.25 | 0.25 | 0.12 | 0.12 | fragment_trap:0.62 | fragment_trap:0.75 | fragment_trap:0.75 |
| back_front:short:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.75 | fragment_trap:1.00 |
| back_front:short:quoted_answer:number | 0.12 | 0.12 | 0.00 | 0.12 | object_copy_trap:0.88 | object_copy_trap:0.88 | fragment_trap:0.75 |
| back_front:short:quoted_answer:plant | 0.00 | 0.12 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.88 |
| back_front:short:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.88 | fragment_trap:1.00 |
| front_back:long:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.38 | punctuation_trap:0.62 | fragment_trap:1.00 |
| front_back:long:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.38 | punctuation_trap:0.75 | fragment_trap:0.50 |
| front_back:long:answer_one_word:furniture | 0.12 | 0.00 | 0.00 | 0.12 | fragment_trap:0.38 | punctuation_trap:0.50 | fragment_trap:0.88 |
| front_back:long:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.62 | fragment_trap:0.75 |
| front_back:long:answer_one_word:plant | 0.38 | 0.12 | 0.00 | 0.25 | other:0.38 | punctuation_trap:0.62 | fragment_trap:0.62 |
| front_back:long:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | other:0.62 | other:0.75 | fragment_trap:0.75 |
| front_back:long:label_colon:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | other:0.62 | other:0.62 |
| front_back:long:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.88 |
| front_back:long:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.75 |
| front_back:long:label_colon:number | 0.00 | 0.00 | 0.12 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.50 |
| front_back:long:label_colon:plant | 0.00 | 0.12 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.75 | fragment_trap:0.62 |
| front_back:long:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.88 | fragment_trap:0.50 |
| front_back:long:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.50 | list_path_failure:0.38 | other:0.38 |
| front_back:long:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.88 | object_copy_trap:0.62 | object_copy_trap:0.50 |
| front_back:long:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.62 | object_copy_trap:0.50 |
| front_back:long:list_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:1.00 | list_path_failure:0.50 | other:0.50 |
| front_back:long:list_answer:plant | 0.38 | 0.38 | 0.38 | 0.25 | generic_continuation_trap:0.38 | object_copy_trap:0.50 | object_copy_trap:0.50 |
| front_back:long:list_answer:time | 0.25 | 0.25 | 0.12 | 0.25 | object_copy_trap:0.50 | object_copy_trap:0.62 | other:0.75 |
| front_back:long:multiple_choice:clothing | 0.88 | 1.00 | 1.00 | 1.00 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:0.88 |
| front_back:long:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:number | 0.50 | 1.00 | 0.88 | 0.62 | correct_surface:0.50 | correct_surface:1.00 | correct_surface:0.75 |
| front_back:long:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:long:multiple_choice:time | 1.00 | 1.00 | 0.50 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:0.50 |
| front_back:long:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | object_copy_trap:0.62 | fragment_trap:0.75 |
| front_back:long:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.50 | object_copy_trap:0.50 | fragment_trap:0.62 |
| front_back:long:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.75 | object_copy_trap:0.62 | object_copy_trap:0.62 |
| front_back:long:quoted_answer:number | 0.00 | 0.12 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.62 | fragment_trap:0.75 |
| front_back:long:quoted_answer:plant | 0.25 | 0.38 | 0.38 | 0.25 | object_copy_trap:0.38 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| front_back:long:quoted_answer:time | 0.25 | 0.25 | 0.25 | 0.25 | object_copy_trap:0.62 | object_copy_trap:0.75 | object_copy_trap:0.50 |
| front_back:neutral:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | object_copy_trap:0.38 | other:0.50 |
| front_back:neutral:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | other:0.50 | other:0.50 |
| front_back:neutral:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | punctuation_trap:0.50 | other:0.50 |
| front_back:neutral:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | object_copy_trap:0.38 | object_copy_trap:0.50 |
| front_back:neutral:answer_one_word:plant | 0.12 | 0.12 | 0.00 | 0.12 | fragment_trap:0.75 | other:0.38 | other:0.50 |
| front_back:neutral:answer_one_word:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.75 | other:0.75 | other:0.88 |
| front_back:neutral:label_colon:clothing | 0.38 | 0.25 | 0.12 | 0.25 | other:0.50 | fragment_trap:0.50 | other:0.50 |
| front_back:neutral:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.50 | other:0.75 |
| front_back:neutral:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | other:0.62 | punctuation_trap:0.38 | fragment_trap:0.50 |
| front_back:neutral:label_colon:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.62 | other:1.00 |
| front_back:neutral:label_colon:plant | 0.12 | 0.12 | 0.00 | 0.00 | fragment_trap:0.62 | fragment_trap:0.38 | other:0.38 |
| front_back:neutral:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | other:0.88 | other:1.00 |
| front_back:neutral:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | object_copy_trap:0.75 | fragment_trap:0.62 |
| front_back:neutral:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | object_copy_trap:0.62 | object_copy_trap:0.50 |
| front_back:neutral:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | generic_continuation_trap:0.38 | object_copy_trap:0.50 | other:0.50 |
| front_back:neutral:list_answer:number | 0.12 | 0.00 | 0.00 | 0.12 | generic_continuation_trap:0.38 | object_copy_trap:0.75 | object_copy_trap:0.62 |
| front_back:neutral:list_answer:plant | 0.25 | 0.25 | 0.00 | 0.25 | fragment_trap:0.38 | object_copy_trap:0.25 | object_copy_trap:0.25 |
| front_back:neutral:list_answer:time | 0.12 | 0.12 | 0.12 | 0.12 | other:0.38 | other:0.62 | other:0.62 |
| front_back:neutral:multiple_choice:clothing | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:container | 0.62 | 0.75 | 0.62 | 0.75 | correct_surface:0.50 | correct_surface:0.75 | correct_surface:0.50 |
| front_back:neutral:multiple_choice:furniture | 0.50 | 1.00 | 1.00 | 0.50 | correct_surface:0.50 | correct_surface:1.00 | correct_surface:0.88 |
| front_back:neutral:multiple_choice:number | 0.88 | 0.88 | 0.75 | 0.88 | correct_surface:0.75 | correct_surface:0.88 | correct_surface:0.75 |
| front_back:neutral:multiple_choice:plant | 0.88 | 1.00 | 1.00 | 0.88 | correct_surface:0.88 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:neutral:multiple_choice:time | 0.75 | 0.75 | 0.88 | 0.75 | correct_surface:0.62 | correct_surface:0.75 | correct_surface:0.88 |
| front_back:neutral:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:1.00 | fragment_trap:0.75 |
| front_back:neutral:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:1.00 | fragment_trap:0.75 |
| front_back:neutral:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.62 | fragment_trap:0.75 |
| front_back:neutral:quoted_answer:number | 0.00 | 0.25 | 0.00 | 0.00 | fragment_trap:1.00 | fragment_trap:0.38 | fragment_trap:0.62 |
| front_back:neutral:quoted_answer:plant | 0.00 | 0.00 | 0.12 | 0.00 | fragment_trap:1.00 | fragment_trap:0.75 | fragment_trap:0.75 |
| front_back:neutral:quoted_answer:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.88 |
| front_back:short:answer_one_word:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.38 | object_copy_trap:0.50 | fragment_trap:0.50 |
| front_back:short:answer_one_word:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.50 | fragment_trap:0.38 | fragment_trap:0.50 |
| front_back:short:answer_one_word:furniture | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | fragment_trap:0.75 | fragment_trap:0.75 |
| front_back:short:answer_one_word:number | 0.00 | 0.00 | 0.00 | 0.00 | other:0.88 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| front_back:short:answer_one_word:plant | 0.00 | 0.25 | 0.25 | 0.00 | other:0.50 | object_copy_trap:0.50 | object_copy_trap:0.62 |
| front_back:short:answer_one_word:time | 0.25 | 0.38 | 0.25 | 0.25 | other:0.50 | correct_surface:0.38 | fragment_trap:0.38 |
| front_back:short:label_colon:clothing | 0.25 | 0.00 | 0.00 | 0.25 | fragment_trap:0.62 | other:0.88 | other:0.88 |
| front_back:short:label_colon:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.88 | other:0.62 |
| front_back:short:label_colon:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.62 | other:0.62 |
| front_back:short:label_colon:number | 0.12 | 0.25 | 0.00 | 0.38 | fragment_trap:0.62 | other:0.50 | other:0.62 |
| front_back:short:label_colon:plant | 0.25 | 0.00 | 0.00 | 0.00 | fragment_trap:0.62 | other:1.00 | other:1.00 |
| front_back:short:label_colon:time | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:1.00 | other:0.88 | other:0.75 |
| front_back:short:list_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | other:0.38 | list_path_failure:0.25 | list_path_failure:0.25 |
| front_back:short:list_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | other:0.50 | object_copy_trap:0.25 | generic_continuation_trap:0.38 |
| front_back:short:list_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.38 | fragment_trap:0.62 | object_copy_trap:0.38 |
| front_back:short:list_answer:number | 0.25 | 0.12 | 0.00 | 0.12 | other:0.50 | object_copy_trap:0.50 | object_copy_trap:0.38 |
| front_back:short:list_answer:plant | 0.62 | 0.38 | 0.62 | 0.62 | correct_surface:0.62 | correct_surface:0.38 | correct_surface:0.62 |
| front_back:short:list_answer:time | 0.00 | 0.25 | 0.25 | 0.00 | generic_continuation_trap:0.62 | object_copy_trap:0.38 | fragment_trap:0.38 |
| front_back:short:multiple_choice:clothing | 0.75 | 1.00 | 1.00 | 0.62 | correct_surface:0.75 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:container | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:furniture | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:number | 0.75 | 1.00 | 0.75 | 0.75 | correct_surface:0.75 | correct_surface:1.00 | correct_surface:0.75 |
| front_back:short:multiple_choice:plant | 1.00 | 1.00 | 1.00 | 1.00 | correct_surface:1.00 | correct_surface:1.00 | correct_surface:1.00 |
| front_back:short:multiple_choice:time | 1.00 | 0.75 | 0.75 | 1.00 | correct_surface:1.00 | correct_surface:0.75 | correct_surface:0.75 |
| front_back:short:quoted_answer:clothing | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.75 | fragment_trap:0.75 |
| front_back:short:quoted_answer:container | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:1.00 | fragment_trap:0.75 |
| front_back:short:quoted_answer:furniture | 0.00 | 0.00 | 0.00 | 0.00 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.62 |
| front_back:short:quoted_answer:number | 0.00 | 0.00 | 0.00 | 0.00 | object_copy_trap:0.62 | object_copy_trap:0.88 | object_copy_trap:0.88 |
| front_back:short:quoted_answer:plant | 0.12 | 0.12 | 0.12 | 0.12 | fragment_trap:0.88 | fragment_trap:0.88 | fragment_trap:0.88 |
| front_back:short:quoted_answer:time | 0.12 | 0.12 | 0.00 | 0.12 | fragment_trap:0.88 | fragment_trap:0.62 | fragment_trap:0.88 |

### Cross-model difficult-format core

| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | mlp_delta | k8+mlp_delta | random_delta | clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | top_clean_traj | top_mlp_traj | top_k8mlp_traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek7b | 144 | 0.076 | 0.074 | 0.044 | 0.070 | -0.003 | -0.032 | -0.006 | -1.611 | -2.333 | -6.972 | -5.636 | 0.062 | 0.000 | 0.003 | 0.118 | fragment_trap | fragment_trap | fragment_trap |
| glm4 | 144 | 0.131 | 0.204 | 0.200 | 0.134 | +0.073 | +0.069 | +0.003 | -0.684 | -0.658 | -4.922 | -6.779 | 0.184 | 0.014 | 0.001 | 0.000 | fragment_trap | fragment_trap | fragment_trap |
| qwen3 | 144 | 0.218 | 0.199 | 0.190 | 0.209 | -0.019 | -0.028 | -0.009 | -0.644 | -0.628 | -6.016 | -6.143 | 0.155 | 0.002 | 0.000 | 0.120 | fragment_trap | fragment_trap | fragment_trap |

